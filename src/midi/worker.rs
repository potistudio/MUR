//! Background MIDI input worker with hot-plug detection and auto-reconnect.
//!
//! The worker owns the `midir` connection on a dedicated thread and reports
//! everything the UI needs through a dedicated [`MidiEvent`] channel: decoded
//! [`MidiAction`]s, connection status transitions, the current port list, and
//! non-fatal warnings. Because the operating system MIDI backend can panic or
//! fail at any moment (device yanked out, driver reload, CoreMIDI restart), the
//! thread body runs inside [`std::panic::catch_unwind`] and is restarted with an
//! exponential backoff instead of taking the whole application down.
//!
//! Ports are re-enumerated once per second. When the connected port vanishes the
//! stale connection is closed immediately and the status switches to
//! [`MidiConnectionStatus::Reconnecting`]; as soon as the same device (manual
//! selection) or a matching Launch Control (automatic selection) reappears, the
//! worker reconnects without any user interaction (SPEC §6.1).

use std::panic::{self, AssertUnwindSafe};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use midir::{Ignore, MidiInput, MidiInputConnection};

use crate::model::MidiConnectionStatus;

use super::mapping::{MidiAction, parse_midi_message};

/// Interval between MIDI port enumeration passes while the worker runs.
pub const PORT_POLL_INTERVAL: Duration = Duration::from_millis(1000);
/// Delay between reconnection attempts for a missing or unopenable port.
pub const RECONNECT_INTERVAL: Duration = Duration::from_millis(1000);
/// Delay before the supervisor restarts a worker whose thread body panicked.
pub const RESTART_BACKOFF: Duration = Duration::from_millis(500);
/// Upper bound of the panic restart backoff.
pub const MAX_RESTART_BACKOFF: Duration = Duration::from_secs(8);
/// Uptime after which the restart backoff drops back to [`RESTART_BACKOFF`].
pub const RESTART_BACKOFF_RESET_AFTER: Duration = Duration::from_secs(30);
/// Client name MUR registers with the operating system MIDI backend.
pub const MIDI_CLIENT_NAME: &str = "MUR";
/// Port name assigned to the input connection by backends that support it.
pub const MIDI_CONNECTION_NAME: &str = "MUR MIDI In";

/// Substrings tried, in order, when no explicit port is selected.
///
/// The first matching port wins; virtual loopback ports are only accepted when
/// no physical controller matches.
const AUTO_DETECT_HINTS: [&str; 4] = ["Launch Control XL 3", "Launch Control", "LCXL", "Novation"];

/// Something the MIDI worker wants the control UI to know about.
#[derive(Debug, Clone, PartialEq)]
pub enum MidiEvent {
	/// A valid Channel 1 message was decoded into a control action.
	Action(MidiAction),
	/// The connection state changed since the previous event.
	StatusChanged(MidiConnectionStatus),
	/// The set of available input ports changed.
	PortsUpdated(Vec<String>),
	/// A non-fatal problem worth showing in the status bar.
	Warning(String),
}

/// Commands the UI can send to the MIDI worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MidiCommand {
	/// Select an explicit port, or `None` to return to automatic detection.
	SelectPort(Option<String>),
	/// Re-enumerate ports and retry the connection immediately.
	Rescan,
	/// Ask the worker to stop and release the MIDI port.
	Shutdown,
}

/// Chooses the port index the worker should connect to.
///
/// With `preferred: Some(name)` only that device is accepted: an exact match is
/// tried first, then a case-insensitive exact match, then a substring match. If
/// the device is absent the result is `None`, so an explicitly selected port
/// never silently falls back to a different controller while it is unplugged.
///
/// With `preferred: None` the worker auto-detects: the built-in Launch Control
/// hints are tried in order, a physical port is preferred over a virtual
/// loopback, and finally the first available port is used.
pub fn choose_preferred_port(ports: &[String], preferred: Option<&str>) -> Option<usize> {
	let preferred = preferred.map(str::trim).filter(|name| !name.is_empty());
	match preferred {
		Some(name) => match_explicit_port(ports, name),
		None => auto_detect_port(ports),
	}
}

/// Resolves an explicitly named port, never falling back to another device.
fn match_explicit_port(ports: &[String], preferred: &str) -> Option<usize> {
	if let Some(index) = ports.iter().position(|port| port == preferred) {
		return Some(index);
	}
	let wanted = preferred.to_lowercase();
	if let Some(index) = ports.iter().position(|port| port.to_lowercase() == wanted) {
		return Some(index);
	}
	ports.iter().position(|port| port.to_lowercase().contains(&wanted))
}

/// Resolves a port using the Launch Control hints, then any available port.
fn auto_detect_port(ports: &[String]) -> Option<usize> {
	if ports.is_empty() {
		return None;
	}
	for hint in AUTO_DETECT_HINTS {
		let hint = hint.to_lowercase();
		let mut virtual_match = None;
		for (index, port) in ports.iter().enumerate() {
			if !port.to_lowercase().contains(&hint) {
				continue;
			}
			if is_virtual_port(port) {
				virtual_match.get_or_insert(index);
			} else {
				return Some(index);
			}
		}
		if virtual_match.is_some() {
			return virtual_match;
		}
	}
	ports.iter().position(|port| !is_virtual_port(port)).or(Some(0))
}

/// Returns `true` for loopback ports that are not real controllers.
fn is_virtual_port(port: &str) -> bool {
	let lowered = port.to_lowercase();
	lowered.contains("virtual") || lowered.contains("iac driver") || lowered.contains("through")
}

/// Handle the UI keeps to receive MIDI events and steer the worker.
///
/// Dropping the handle sends [`MidiCommand::Shutdown`] and joins the worker, so
/// the port is always released before the application exits.
pub struct MidiWorkerHandle {
	events: Receiver<MidiEvent>,
	commands: Sender<MidiCommand>,
	thread: Option<std::thread::JoinHandle<()>>,
}

impl MidiWorkerHandle {
	/// Starts the background worker.
	///
	/// `initial_port` is the persisted manual selection; `None` enables automatic
	/// detection. A failure to spawn the thread is reported through a
	/// [`MidiEvent::Warning`] instead of panicking.
	pub fn spawn(initial_port: Option<String>) -> Self {
		let (event_tx, event_rx) = crossbeam_channel::unbounded();
		let (command_tx, command_rx) = crossbeam_channel::unbounded();
		let thread_events = event_tx.clone();

		let thread = std::thread::Builder::new()
			.name("mur-midi".to_string())
			.spawn(move || run_supervisor(thread_events, command_rx, initial_port))
			.ok();

		if thread.is_none() {
			let _ = event_tx.send(MidiEvent::Warning("failed to spawn the MIDI worker thread".to_string()));
			let _ = event_tx.send(MidiEvent::StatusChanged(MidiConnectionStatus::Disconnected));
		}

		Self {
			events: event_rx,
			commands: command_tx,
			thread,
		}
	}

	/// Returns the next event without blocking.
	///
	/// Returns `None` both when no event is pending and when the worker already
	/// stopped, which lets the UI poll every frame without special cases.
	pub fn try_recv(&self) -> Option<MidiEvent> {
		self.events.try_recv().ok()
	}

	/// Requests a different port, or `None` for automatic detection.
	///
	/// The request is ignored when the worker already stopped; the UI always
	/// receives the resulting [`MidiEvent::StatusChanged`] on the event channel.
	pub fn select_port(&self, port: Option<String>) {
		let _ = self.commands.send(MidiCommand::SelectPort(port));
	}

	/// Requests an immediate port re-enumeration and reconnect attempt.
	pub fn rescan(&self) {
		let _ = self.commands.send(MidiCommand::Rescan);
	}
}

impl Drop for MidiWorkerHandle {
	fn drop(&mut self) {
		let _ = self.commands.send(MidiCommand::Shutdown);
		if let Some(thread) = self.thread.take() {
			let _ = thread.join();
		}
	}
}

/// Thread entry point that restarts the worker after a backend panic.
fn run_supervisor(events: Sender<MidiEvent>, commands: Receiver<MidiCommand>, initial_port: Option<String>) {
	let last_port = initial_port.clone();
	supervise(
		|| run_worker(&events, &commands, initial_port.clone()),
		&events,
		&commands,
		last_port,
	);
}

/// Runs `worker` until it returns normally, restarting it after any panic.
///
/// Each restart is announced on the event channel together with a
/// [`MidiConnectionStatus::Reconnecting`] status, so the UI never keeps showing
/// a stale `Connected` state after the backend crashed. A shutdown request that
/// arrives while the worker is down wins over the restart.
fn supervise(
	mut worker: impl FnMut(),
	events: &Sender<MidiEvent>,
	commands: &Receiver<MidiCommand>,
	last_port: Option<String>,
) {
	let mut backoff = RESTART_BACKOFF;

	loop {
		let started = Instant::now();
		let outcome = panic::catch_unwind(AssertUnwindSafe(&mut worker));

		let Err(payload) = outcome else {
			return;
		};

		let detail = panic_payload_message(payload);
		let _ = events.send(MidiEvent::Warning(format!("MIDI worker panicked: {detail}")));
		let _ = events.send(MidiEvent::StatusChanged(MidiConnectionStatus::Reconnecting {
			last_port: last_port.clone(),
			error: Some(detail),
		}));

		loop {
			match commands.try_recv() {
				Ok(MidiCommand::Shutdown) | Err(TryRecvError::Disconnected) => return,
				Ok(_) => {}
				Err(TryRecvError::Empty) => break,
			}
		}

		// A short-lived worker that keeps crashing backs off further; a worker
		// that ran fine for a while restarts with the base delay.
		let wait = if started.elapsed() >= RESTART_BACKOFF_RESET_AFTER {
			RESTART_BACKOFF
		} else {
			backoff
		};
		let _ = events.send(MidiEvent::Warning(format!(
			"restarting the MIDI worker in {} ms",
			wait.as_millis()
		)));
		std::thread::sleep(wait);
		backoff = wait.saturating_mul(2).min(MAX_RESTART_BACKOFF);
	}
}

/// Extracts a human readable message from a caught panic payload.
fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
	if let Some(message) = payload.downcast_ref::<&'static str>() {
		return (*message).to_string();
	}
	if let Some(message) = payload.downcast_ref::<String>() {
		return message.clone();
	}
	"non-string panic payload".to_string()
}

/// One run of the worker loop. Returns only on [`MidiCommand::Shutdown`].
fn run_worker(events: &Sender<MidiEvent>, commands: &Receiver<MidiCommand>, initial_port: Option<String>) {
	let mut state = WorkerState::new(initial_port);
	let mut scanner: Option<MidiInput> = None;
	let mut keep_running = true;

	while keep_running {
		loop {
			match commands.try_recv() {
				Ok(command) => {
					if !state.handle_command(command, events) {
						keep_running = false;
						break;
					}
				}
				Err(TryRecvError::Empty) => break,
				Err(TryRecvError::Disconnected) => {
					keep_running = false;
					break;
				}
			}
		}

		if !keep_running {
			break;
		}

		state.tick(&mut scanner, events);

		match commands.recv_timeout(PORT_POLL_INTERVAL) {
			Ok(command) => {
				if !state.handle_command(command, events) {
					keep_running = false;
				}
			}
			Err(RecvTimeoutError::Timeout) => {}
			Err(RecvTimeoutError::Disconnected) => keep_running = false,
		}
	}

	state.shutdown(events);
}

/// A port currently held open by the worker.
struct ConnectedPort {
	/// Name reported by the backend, used for hot-unplug detection.
	name: String,
	/// Live connection; it is closed explicitly by
	/// [`WorkerState::release_connection`].
	connection: MidiInputConnection<()>,
}

/// Mutable state of one worker run.
struct WorkerState {
	/// Manual port selection, `None` for automatic detection.
	preferred: Option<String>,
	/// Ports reported by the last enumeration pass, sorted and deduplicated.
	ports: Vec<String>,
	/// Open connection, when a port is currently held.
	connected: Option<ConnectedPort>,
	/// Last status reported to the UI.
	status: MidiConnectionStatus,
	/// Earliest instant of the next reconnect attempt.
	next_connect_attempt: Instant,
	/// Earliest instant of the next backend initialization attempt.
	next_scanner_attempt: Instant,
	/// Last backend error, used to suppress duplicate warnings.
	last_error: Option<String>,
}

impl WorkerState {
	/// Creates the state for one worker run.
	fn new(preferred: Option<String>) -> Self {
		Self {
			preferred: normalize_port(preferred),
			ports: Vec::new(),
			connected: None,
			status: MidiConnectionStatus::Disconnected,
			next_connect_attempt: Instant::now(),
			next_scanner_attempt: Instant::now(),
			last_error: None,
		}
	}

	/// Applies one command; returns `false` when the worker should stop.
	fn handle_command(&mut self, command: MidiCommand, events: &Sender<MidiEvent>) -> bool {
		match command {
			MidiCommand::Shutdown => false,
			MidiCommand::Rescan => {
				self.next_connect_attempt = Instant::now();
				self.next_scanner_attempt = Instant::now();
				true
			}
			MidiCommand::SelectPort(port) => {
				self.select_port(port, events);
				true
			}
		}
	}

	/// Changes the manual selection and drops a connection to the old device.
	fn select_port(&mut self, port: Option<String>, events: &Sender<MidiEvent>) {
		let port = normalize_port(port);
		if self.preferred == port {
			return;
		}
		self.preferred = port;
		self.last_error = None;
		self.release_connection();
		self.next_connect_attempt = Instant::now();
		self.set_status(MidiConnectionStatus::Scanning, events);
	}

	/// One periodic step: initialize the backend, enumerate, then reconnect.
	fn tick(&mut self, scanner: &mut Option<MidiInput>, events: &Sender<MidiEvent>) {
		self.ensure_scanner(scanner, events);
		if let Some(input) = scanner.as_ref() {
			self.refresh_ports(input, events);
		}
		self.sync_connection(events);
	}

	/// Lazily creates the `midir` client used for port enumeration.
	fn ensure_scanner(&mut self, scanner: &mut Option<MidiInput>, events: &Sender<MidiEvent>) {
		if scanner.is_some() || Instant::now() < self.next_scanner_attempt {
			return;
		}

		match MidiInput::new(MIDI_CLIENT_NAME) {
			Ok(mut input) => {
				input.ignore(Ignore::All);
				*scanner = Some(input);
				self.last_error = None;
			}
			Err(error) => {
				let detail = error.to_string();
				self.next_scanner_attempt = Instant::now() + RECONNECT_INTERVAL;
				self.report_error(&detail, events);
				self.set_status(MidiConnectionStatus::Disconnected, events);
			}
		}
	}

	/// Re-enumerates input ports and publishes the list when it changed.
	fn refresh_ports(&mut self, input: &MidiInput, events: &Sender<MidiEvent>) {
		let mut names = Vec::with_capacity(input.port_count());
		for port in input.ports() {
			if let Ok(name) = input.port_name(&port) {
				names.push(name);
			}
		}
		names.sort_unstable();
		names.dedup();

		if names == self.ports {
			return;
		}

		self.ports = names;
		let _ = events.send(MidiEvent::PortsUpdated(self.ports.clone()));
	}

	/// Detects hot-unplug, switches devices when needed, and reconnects.
	fn sync_connection(&mut self, events: &Sender<MidiEvent>) {
		let best_name = choose_preferred_port(&self.ports, self.preferred.as_deref())
			.and_then(|index| self.ports.get(index).cloned());

		if let Some(connected) = self.connected.as_ref() {
			let present = self.ports.iter().any(|name| name == &connected.name);
			let keep = if self.preferred.is_some() {
				present
			} else {
				match best_name.as_deref() {
					Some(best) => best == connected.name,
					None => present,
				}
			};
			if keep {
				return;
			}

			let last_port = self.connected.as_ref().map(|connected| connected.name.clone());
			self.release_connection();
			if best_name.is_none() {
				self.set_status(MidiConnectionStatus::Reconnecting { last_port, error: None }, events);
			}
			self.next_connect_attempt = Instant::now();
		}

		if Instant::now() < self.next_connect_attempt {
			return;
		}

		let Some(name) = best_name else {
			let status = match self.preferred.as_deref() {
				Some(preferred) => MidiConnectionStatus::Reconnecting {
					last_port: Some(preferred.to_string()),
					error: self.last_error.clone(),
				},
				None => MidiConnectionStatus::Scanning,
			};
			self.set_status(status, events);
			self.next_connect_attempt = Instant::now() + RECONNECT_INTERVAL;
			return;
		};

		match open_port(&name, events) {
			Ok(connection) => {
				self.connected = Some(ConnectedPort {
					name: name.clone(),
					connection,
				});
				self.last_error = None;
				self.set_status(MidiConnectionStatus::Connected { port_name: name }, events);
			}
			Err(detail) => {
				self.report_error(&detail, events);
				self.set_status(
					MidiConnectionStatus::Reconnecting {
						last_port: Some(name),
						error: Some(detail),
					},
					events,
				);
				self.next_connect_attempt = Instant::now() + RECONNECT_INTERVAL;
			}
		}
	}

	/// Emits a warning for a backend failure, suppressing repeated messages.
	fn report_error(&mut self, detail: &str, events: &Sender<MidiEvent>) {
		if self.last_error.as_deref() == Some(detail) {
			return;
		}
		self.last_error = Some(detail.to_string());
		let _ = events.send(MidiEvent::Warning(detail.to_string()));
	}

	/// Publishes a new status, deduplicating unchanged values.
	fn set_status(&mut self, status: MidiConnectionStatus, events: &Sender<MidiEvent>) {
		if self.status == status {
			return;
		}
		self.status = status;
		let _ = events.send(MidiEvent::StatusChanged(self.status.clone()));
	}

	/// Releases the connection and reports the final state.
	fn shutdown(&mut self, events: &Sender<MidiEvent>) {
		self.release_connection();
		self.set_status(MidiConnectionStatus::Disconnected, events);
	}

	/// Closes the open port, if any, so the device is free for other software.
	fn release_connection(&mut self) {
		if let Some(connected) = self.connected.take() {
			// Closing hands the port back to the backend; the returned handle is
			// dropped immediately since the worker creates a fresh client.
			let (_input, _state) = connected.connection.close();
		}
	}
}

/// Normalizes a stored port name, treating blank names as automatic detection.
fn normalize_port(port: Option<String>) -> Option<String> {
	port.map(|name| name.trim().to_string()).filter(|name| !name.is_empty())
}

/// Opens `name` and forwards decoded messages to the event channel.
fn open_port(name: &str, events: &Sender<MidiEvent>) -> Result<MidiInputConnection<()>, String> {
	let mut input = MidiInput::new(MIDI_CLIENT_NAME).map_err(|error| error.to_string())?;
	input.ignore(Ignore::All);

	let port = input
		.ports()
		.into_iter()
		.find(|port| input.port_name(port).is_ok_and(|port_name| port_name == name))
		.ok_or_else(|| format!("port `{name}` is no longer available"))?;

	let sender = events.clone();
	input
		.connect(
			&port,
			MIDI_CONNECTION_NAME,
			move |_timestamp, bytes, _state| {
				if let Some(action) = parse_midi_message(bytes) {
					let _ = sender.send(MidiEvent::Action(action));
				}
			},
			(),
		)
		.map_err(|error| error.kind().to_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;

	fn ports(names: &[&str]) -> Vec<String> {
		names.iter().map(|name| (*name).to_string()).collect()
	}

	// ----------------------------------------------------- choose_preferred_port

	#[test]
	fn auto_detection_without_ports() {
		assert_eq!(choose_preferred_port(&[], None), None);
	}

	#[test]
	fn auto_detection_prefers_the_launch_control_xl_3() {
		let available = ports(&["IAC Driver Bus 1", "Launch Control XL 3: MIDI In", "Novation Launchpad"]);
		assert_eq!(choose_preferred_port(&available, None), Some(1));
	}

	#[test]
	fn auto_detection_hint_order_is_respected() {
		let available = ports(&["Novation Launchpad", "Launch Control XL 3", "LCXL MIDI"]);
		assert_eq!(choose_preferred_port(&available, None), Some(1));

		let available = ports(&["Novation Launchpad", "LCXL MIDI"]);
		assert_eq!(choose_preferred_port(&available, None), Some(1));

		let available = ports(&["Arturia KeyStep", "Novation Launchpad"]);
		assert_eq!(choose_preferred_port(&available, None), Some(1));
	}

	#[test]
	fn auto_detection_is_case_insensitive() {
		let available = ports(&["iac driver bus 1", "launch control xl 3 midi in"]);
		assert_eq!(choose_preferred_port(&available, None), Some(1));
	}

	#[test]
	fn auto_detection_prefers_physical_ports_over_virtual_ones() {
		let available = ports(&["Virtual Loopback", "IAC Driver Bus 1", "Arturia KeyStep"]);
		assert_eq!(choose_preferred_port(&available, None), Some(2));

		let available = ports(&["Virtual Launch Control", "Launch Control XL 3"]);
		assert_eq!(choose_preferred_port(&available, None), Some(1));
	}

	#[test]
	fn auto_detection_falls_back_to_the_first_port() {
		let available = ports(&["Virtual Loopback", "IAC Driver Bus 1"]);
		assert_eq!(choose_preferred_port(&available, None), Some(0));

		let available = ports(&["Arturia KeyStep", "Ableton Push"]);
		assert_eq!(choose_preferred_port(&available, None), Some(0));
	}

	#[test]
	fn explicit_selection_uses_an_exact_match_first() {
		let available = ports(&["Launch Control XL 3", "Launch Control XL 3 (copy)"]);
		assert_eq!(choose_preferred_port(&available, Some("Launch Control XL 3")), Some(0));
		assert_eq!(
			choose_preferred_port(&available, Some("Launch Control XL 3 (copy)")),
			Some(1)
		);
	}

	#[test]
	fn explicit_selection_falls_back_to_case_insensitive_and_substring_matches() {
		let available = ports(&["Launch Control XL 3: MIDI In"]);
		assert_eq!(
			choose_preferred_port(&available, Some("launch control xl 3: midi in")),
			Some(0)
		);
		assert_eq!(choose_preferred_port(&available, Some("MIDI In")), Some(0));
		assert_eq!(choose_preferred_port(&available, Some("MIDI")), Some(0));
	}

	#[test]
	fn explicit_selection_never_switches_to_another_device() {
		let available = ports(&["IAC Driver Bus 1", "Arturia KeyStep"]);
		assert_eq!(choose_preferred_port(&available, Some("Launch Control XL 3")), None);
		assert_eq!(choose_preferred_port(&[], Some("Launch Control XL 3")), None);
	}

	#[test]
	fn explicit_selection_ignores_blank_names() {
		let available = ports(&["Novation Launchpad"]);
		assert_eq!(choose_preferred_port(&available, Some("   ")), Some(0));
		assert_eq!(choose_preferred_port(&available, Some("")), Some(0));

		let available = ports(&["Launch Control XL 3", "Novation Launchpad"]);
		assert_eq!(
			choose_preferred_port(&available, Some("  Launch Control XL 3  ")),
			Some(0)
		);
	}

	#[test]
	fn explicit_selection_keeps_duplicate_ports_deterministic() {
		let available = ports(&["Launch Control XL 3", "Launch Control XL 3"]);
		assert_eq!(choose_preferred_port(&available, Some("Launch Control XL 3")), Some(0));
	}

	// ------------------------------------------------------------- worker state

	#[test]
	fn worker_state_normalizes_the_initial_selection() {
		assert_eq!(WorkerState::new(None).preferred, None);
		assert_eq!(WorkerState::new(Some("   ".to_string())).preferred, None);
		assert_eq!(
			WorkerState::new(Some("  Launch Control XL 3 ".to_string())).preferred,
			Some("Launch Control XL 3".to_string())
		);
	}

	#[test]
	fn shutdown_command_stops_the_worker() {
		let (tx, _rx) = crossbeam_channel::unbounded();
		let mut state = WorkerState::new(None);
		assert!(!state.handle_command(MidiCommand::Shutdown, &tx));
	}

	#[test]
	fn rescan_command_schedules_an_immediate_attempt() {
		let (tx, _rx) = crossbeam_channel::unbounded();
		let mut state = WorkerState::new(None);
		state.next_connect_attempt = Instant::now() + Duration::from_secs(60);
		assert!(state.handle_command(MidiCommand::Rescan, &tx));
		assert!(state.next_connect_attempt <= Instant::now());
	}

	#[test]
	fn selecting_a_port_emits_exactly_one_status_change() {
		let (tx, rx) = crossbeam_channel::unbounded();
		let mut state = WorkerState::new(None);

		state.select_port(Some("  Launch Control XL 3 ".to_string()), &tx);
		assert_eq!(state.preferred.as_deref(), Some("Launch Control XL 3"));
		assert!(matches!(
			rx.try_recv(),
			Ok(MidiEvent::StatusChanged(MidiConnectionStatus::Scanning))
		));
		assert!(rx.try_recv().is_err());

		// Selecting the same port again is a no-op.
		state.select_port(Some("Launch Control XL 3".to_string()), &tx);
		assert!(rx.try_recv().is_err());
	}

	#[test]
	fn selecting_a_blank_port_returns_to_auto_detection() {
		let (tx, rx) = crossbeam_channel::unbounded();
		let mut state = WorkerState::new(Some("Launch Control XL 3".to_string()));
		state.select_port(Some("".to_string()), &tx);
		assert_eq!(state.preferred, None);
		assert!(matches!(
			rx.try_recv(),
			Ok(MidiEvent::StatusChanged(MidiConnectionStatus::Scanning))
		));
	}

	#[test]
	fn repeated_backend_errors_are_reported_once() {
		let (tx, rx) = crossbeam_channel::unbounded();
		let mut state = WorkerState::new(None);

		state.report_error("backend exploded", &tx);
		state.report_error("backend exploded", &tx);
		state.report_error("backend exploded again", &tx);

		assert_eq!(rx.try_recv(), Ok(MidiEvent::Warning("backend exploded".to_string())));
		assert_eq!(
			rx.try_recv(),
			Ok(MidiEvent::Warning("backend exploded again".to_string()))
		);
		assert!(rx.try_recv().is_err());
	}

	#[test]
	fn status_changes_are_deduplicated() {
		let (tx, rx) = crossbeam_channel::unbounded();
		let mut state = WorkerState::new(None);

		state.set_status(MidiConnectionStatus::Scanning, &tx);
		state.set_status(MidiConnectionStatus::Scanning, &tx);
		state.set_status(
			MidiConnectionStatus::Connected {
				port_name: "XL3".to_string(),
			},
			&tx,
		);
		state.set_status(
			MidiConnectionStatus::Connected {
				port_name: "XL3".to_string(),
			},
			&tx,
		);

		assert_eq!(
			rx.try_recv(),
			Ok(MidiEvent::StatusChanged(MidiConnectionStatus::Scanning))
		);
		assert_eq!(
			rx.try_recv(),
			Ok(MidiEvent::StatusChanged(MidiConnectionStatus::Connected {
				port_name: "XL3".to_string()
			}))
		);
		assert!(rx.try_recv().is_err());
	}

	#[test]
	fn shutdown_sets_disconnected_once() {
		let (tx, rx) = crossbeam_channel::unbounded();
		let mut state = WorkerState::new(None);
		state.set_status(MidiConnectionStatus::Scanning, &tx);
		assert!(matches!(rx.try_recv(), Ok(MidiEvent::StatusChanged(_))));

		state.shutdown(&tx);
		assert_eq!(state.status, MidiConnectionStatus::Disconnected);
		state.shutdown(&tx);
		assert_eq!(
			rx.try_recv(),
			Ok(MidiEvent::StatusChanged(MidiConnectionStatus::Disconnected))
		);
		assert!(rx.try_recv().is_err());
	}

	#[test]
	fn panic_messages_are_extracted_from_every_payload_shape() {
		let message: Box<dyn std::any::Any + Send> = Box::new("backend panic");
		assert_eq!(panic_payload_message(message), "backend panic");

		let message: Box<dyn std::any::Any + Send> = Box::new(String::from("owned panic"));
		assert_eq!(panic_payload_message(message), "owned panic");

		let message: Box<dyn std::any::Any + Send> = Box::new(42_u32);
		assert_eq!(panic_payload_message(message), "non-string panic payload");
	}

	// ---------------------------------------------------------------- supervisor

	#[test]
	fn supervisor_restarts_a_panicking_worker() {
		use std::sync::atomic::{AtomicU32, Ordering};

		let (tx, rx) = crossbeam_channel::unbounded();
		let (_command_tx, command_rx) = crossbeam_channel::unbounded::<MidiCommand>();
		let runs = AtomicU32::new(0);

		supervise(
			|| {
				if runs.fetch_add(1, Ordering::SeqCst) == 0 {
					panic!("simulated midir backend panic");
				}
			},
			&tx,
			&command_rx,
			Some("Launch Control XL 3".to_string()),
		);

		assert_eq!(runs.load(Ordering::SeqCst), 2, "the worker is restarted once");
		let events: Vec<MidiEvent> = rx.try_iter().collect();
		assert!(events.iter().any(|event| matches!(
			event,
			MidiEvent::Warning(message) if message.contains("simulated midir backend panic")
		)));
		assert!(events.iter().any(|event| matches!(
			event,
			MidiEvent::StatusChanged(MidiConnectionStatus::Reconnecting {
				last_port: Some(port),
				error: Some(_),
			}) if port == "Launch Control XL 3"
		)));
		assert!(
			events
				.iter()
				.any(|event| matches!(event, MidiEvent::Warning(message) if message.contains("restarting")))
		);
	}

	#[test]
	fn supervisor_honours_a_pending_shutdown_after_a_panic() {
		use std::sync::atomic::{AtomicU32, Ordering};

		let (tx, rx) = crossbeam_channel::unbounded();
		let (command_tx, command_rx) = crossbeam_channel::unbounded();
		command_tx.send(MidiCommand::Shutdown).unwrap();
		let runs = AtomicU32::new(0);
		let started = Instant::now();

		supervise(
			|| {
				runs.fetch_add(1, Ordering::SeqCst);
				panic!("simulated midir backend panic");
			},
			&tx,
			&command_rx,
			None,
		);

		assert_eq!(runs.load(Ordering::SeqCst), 1, "the worker must not restart");
		assert!(started.elapsed() < RESTART_BACKOFF, "shutdown must skip the backoff");
		let events: Vec<MidiEvent> = rx.try_iter().collect();
		assert!(!events.iter().any(|event| matches!(
			event,
			MidiEvent::Warning(message) if message.contains("restarting")
		)));
	}

	// -------------------------------------------------------------- integration

	#[test]
	fn worker_handle_polls_without_a_device_and_shuts_down_cleanly() {
		let handle = MidiWorkerHandle::spawn(None);

		// None of these may block or panic even when no controller is attached.
		let _ = handle.try_recv();
		handle.rescan();
		handle.select_port(Some("MUR test port that does not exist".to_string()));
		handle.select_port(None);

		drop(handle);
	}

	#[test]
	fn worker_handle_reports_a_connection_status() {
		use std::time::Instant as StdInstant;

		let handle = MidiWorkerHandle::spawn(None);
		let deadline = StdInstant::now() + Duration::from_secs(5);
		let mut saw_status = false;

		while StdInstant::now() < deadline {
			match handle.try_recv() {
				Some(MidiEvent::StatusChanged(status)) => {
					assert!(matches!(
						status,
						MidiConnectionStatus::Disconnected
							| MidiConnectionStatus::Scanning
							| MidiConnectionStatus::Connected { .. }
							| MidiConnectionStatus::Reconnecting { .. }
					));
					saw_status = true;
					break;
				}
				Some(_) => {}
				None => std::thread::sleep(Duration::from_millis(20)),
			}
		}

		assert!(saw_status, "the worker must publish its initial connection status");
	}
}
