//! Application runtime: the dual-window winit event loop that ties every subsystem together.
//!
//! [`run`] owns the main thread for the whole show. It creates the control window (egui on
//! wgpu) and, when an external display is selected, the borderless output window, then
//! drives a fixed ~60 fps tick from `ControlFlow::WaitUntil`:
//!
//! 1. native quit requests (`Cmd+Q`, Dock "Quit") are turned into the exit guard;
//! 2. MIDI events, layer streamer events, and background clip loads are drained;
//! 3. layers advance, and every layer is synchronised with the [`LayerStreamer`];
//! 4. the master frame is composited once and presented to the output window first, then
//!    mirrored into the control UI;
//! 5. performance counters update and the debounced state snapshot is handed to a writer
//!    thread, so `fsync` never stalls a frame.
//!
//! Every failure path degrades instead of crashing (SPEC §9): corrupt clips only mark their
//! slot, a vanished display parks the output in `WaitingForMonitor`, a lost GPU device is
//! rebuilt, and closing either window only opens the exit confirmation guard.
//!
//! The session bookkeeping that does not need a window server — queued clip loads, layer
//! bindings waiting for those loads, project/state restoration, and exit-guard routing — is
//! implemented as plain functions and [`ClipLibrary`], so it is covered by unit tests.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::ModifiersState;
use winit::window::{UserAttentionType, Window, WindowAttributes, WindowId};

use crate::config::{AppPaths, CliArgs, PersistedState, ProjectConfig, StateSaver, load_state, save_state_atomic};
use crate::error::{ConfigError, LogEntry, MurError, RenderError, ToastLevel};
use crate::media::{self, ClipAsset, LayerStreamEvent, LayerStreamer};
use crate::midi::{MidiEvent, MidiWorkerHandle, apply_midi_action};
use crate::model::layer::{DEFAULT_CLIP_FPS, LAYER_COUNT, MAX_CLIP_FPS, MIN_CLIP_FPS};
use crate::model::{
	AppState, ClipId, ClipInfo, ClipKind, ClipStatus, DecodedFrame, LayerIndex, MidiConnectionStatus, MonitorSummary,
	OutputDisplaySelection, OutputRuntimeStatus, PlaybackState,
};
use crate::platform::{self, MonitorDescriptor, OutputPlacement, PlatformError};
use crate::render::{GpuContext, LayerRenderInput, MasterRenderer, select_surface_format};
use crate::ui::{UiApplyOutcome, UiCommand, UiState, apply_ui_command, map_key_event};

/// Target duration of one output frame (60 fps).
pub const TARGET_FRAME: Duration = Duration::from_micros(16_667);
/// How often connected monitors are re-enumerated for hot-plug detection.
pub const MONITOR_POLL_INTERVAL: Duration = Duration::from_millis(750);
/// Delay between attempts to rebuild a lost GPU device.
pub const GPU_RETRY_INTERVAL: Duration = Duration::from_secs(2);
/// Largest simulation step applied after a stall, so playheads never jump wildly.
pub const MAX_TICK_DT_SECS: f32 = 0.1;
/// Number of background clip loader threads.
pub const CLIP_LOADER_THREADS: usize = 2;
/// Extensions offered by the "+ Files" dialog: media plus TOML/JSON project files.
pub const DIALOG_EXTENSIONS: [&str; 6] = ["mov", "png", "jpg", "jpeg", "toml", "json"];
/// Title of the control window.
pub const CONTROL_WINDOW_TITLE: &str = "MUR — Live VJ Control";
/// Title of the control window while the GPU device is lost and being rebuilt.
pub const GPU_LOST_WINDOW_TITLE: &str = "MUR — Control [GPU LOST — retrying...]";
/// Title of the output window.
pub const OUTPUT_WINDOW_TITLE: &str = "MUR — Output";

/// Upper bound of MIDI events applied per tick, so a flood cannot starve rendering.
const MAX_MIDI_EVENTS_PER_TICK: usize = 512;
/// Smoothing factor of the FPS / frame-time moving averages.
const PERF_EMA_ALPHA: f32 = 0.1;
/// Shader time wraps after this many seconds to keep `f32` precision.
const SHADER_TIME_WRAP_SECS: f64 = 3600.0;
/// Minimum spacing between two ticks triggered by `RedrawRequested`.
const MIN_REDRAW_TICK_SPACING: Duration = Duration::from_millis(8);
/// The control window presents without vsync so it can never stall the output frame.
const CONTROL_PREFERS_VSYNC: bool = false;
/// The audience output is locked to the display refresh to avoid tearing.
const OUTPUT_PREFERS_VSYNC: bool = true;
/// Background color behind the egui panels.
const CONTROL_CLEAR_COLOR: wgpu::Color = wgpu::Color {
	r: 0.063,
	g: 0.071,
	b: 0.086,
	a: 1.0,
};

// ======================================================================= events

/// Result of one background clip load, as delivered to the event loop.
pub type ClipLoadResult = Result<(ClipInfo, ClipAsset), String>;

/// Events sent to the winit event loop from other threads.
#[derive(Debug)]
#[allow(
	clippy::large_enum_variant,
	reason = "one event per clip load; boxing would only add an allocation"
)]
pub enum AppEvent {
	/// A native quit request was intercepted (see [`platform::set_quit_waker`]).
	QuitRequested,
	/// A background clip load finished.
	ClipLoaded {
		/// Clip slot the load was queued for.
		clip_id: ClipId,
		/// Path that was loaded.
		path: PathBuf,
		/// Loaded metadata and asset, or a human-readable error.
		result: ClipLoadResult,
	},
	/// The "+ Files" dialog returned these paths.
	FilesPicked(Vec<PathBuf>),
	/// The "+ Folder" dialog returned this directory.
	FolderPicked(PathBuf),
}

// ============================================================ clip bookkeeping

/// A clip load queued on the background loader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadRequest {
	/// Slot the result belongs to.
	pub clip_id: ClipId,
	/// File to load.
	pub path: PathBuf,
}

/// A layer that should be bound to a clip as soon as the clip finishes loading.
///
/// Created when restoring the persisted state or applying a project file, because clips
/// load asynchronously and their [`ClipId`]s are only known once they are queued.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingBinding {
	/// Layer to bind.
	pub layer: LayerIndex,
	/// Clip path the layer should show.
	pub path: PathBuf,
	/// Playhead to restore, in clip frames.
	pub playhead_frame: f64,
	/// Keep the layer's current selection when binding instead of selecting the clip.
	///
	/// Set when restoring a persisted session, whose selection is restored separately
	/// and may differ from the active clip.
	pub keep_selection: bool,
}

/// What [`ClipLibrary::complete`] changed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClipLoadOutcome {
	/// The result belongs to a live slot; its asset should be registered with the streamer.
	pub accepted: bool,
	/// Layers already showing the clip whose `frame_count` / `clip_fps` were hydrated.
	pub hydrated_layers: [bool; LAYER_COUNT],
	/// Layers bound from a [`PendingBinding`].
	pub bound_layers: [bool; LAYER_COUNT],
	/// Layers that tried to use the clip and recorded the load error.
	pub failed_layers: [bool; LAYER_COUNT],
	/// The load error, when the clip failed.
	pub error: Option<String>,
}

impl ClipLoadOutcome {
	/// Returns `true` when `layer` must (re)start streaming the clip.
	pub fn layer_needs_restart(&self, layer: LayerIndex) -> bool {
		let index = layer.index();
		self.hydrated_layers.get(index).copied().unwrap_or(false)
			|| self.bound_layers.get(index).copied().unwrap_or(false)
	}
}

/// Tracks clips that are still loading and layer bindings waiting for them.
///
/// Queued clips get a placeholder [`ClipInfo`] in [`AppState::clips`] right away (status
/// `Ready`, no thumbnail, zero frames), which the clip grid renders as "Loading…". The
/// placeholder keeps the grid order stable and lets the operator trigger a clip before it
/// finished loading: the layer simply starts once [`ClipLibrary::complete`] hydrates it.
#[derive(Debug, Default)]
pub struct ClipLibrary {
	loading: HashMap<ClipId, PathBuf>,
	pending_bindings: Vec<PendingBinding>,
}

impl ClipLibrary {
	/// Creates an empty library.
	pub fn new() -> Self {
		Self::default()
	}

	/// Returns `true` while `clip_id` waits for the background loader.
	pub fn is_loading(&self, clip_id: ClipId) -> bool {
		self.loading.contains_key(&clip_id)
	}

	/// Number of clips currently loading.
	pub fn loading_count(&self) -> usize {
		self.loading.len()
	}

	/// Layer bindings still waiting for their clip.
	pub fn pending_bindings(&self) -> &[PendingBinding] {
		&self.pending_bindings
	}

	/// The pending binding of `layer`, if any.
	pub fn pending_binding(&self, layer: LayerIndex) -> Option<&PendingBinding> {
		self.pending_bindings.iter().find(|binding| binding.layer == layer)
	}

	/// Queues `path` for loading unless it is already loaded or queued.
	///
	/// A clip that previously failed is retried in its existing slot. Returns the request
	/// to hand to the background loader, or `None` when nothing needs loading.
	pub fn enqueue(&mut self, state: &mut AppState, path: &Path) -> Option<LoadRequest> {
		if let Some(existing) = state.clips.iter().find(|clip| clip.path == path) {
			let clip_id = existing.id;
			if existing.is_ready() || self.is_loading(clip_id) {
				return None;
			}
			state.upsert_clip(placeholder_clip(clip_id, path));
			self.loading.insert(clip_id, path.to_path_buf());
			return Some(LoadRequest {
				clip_id,
				path: path.to_path_buf(),
			});
		}

		let clip_id = state.allocate_clip_id();
		state.upsert_clip(placeholder_clip(clip_id, path));
		self.loading.insert(clip_id, path.to_path_buf());
		Some(LoadRequest {
			clip_id,
			path: path.to_path_buf(),
		})
	}

	/// Records that `binding.layer` should show `binding.path` once it loaded.
	///
	/// Replaces an older binding of the same layer.
	pub fn add_binding(&mut self, binding: PendingBinding) {
		self.pending_bindings.retain(|existing| existing.layer != binding.layer);
		self.pending_bindings.push(binding);
	}

	/// Drops the pending binding of `layer`, e.g. after the operator triggered something else.
	pub fn cancel_binding(&mut self, layer: LayerIndex) {
		self.pending_bindings.retain(|existing| existing.layer != layer);
	}

	/// Forgets a clip the operator removed; a late load result is then discarded.
	pub fn forget(&mut self, clip_id: ClipId) {
		if let Some(path) = self.loading.remove(&clip_id) {
			self.pending_bindings.retain(|binding| binding.path != path);
		}
	}

	/// Marks a queued clip as failed without a loader result (e.g. the loader is gone).
	pub fn fail(&mut self, state: &mut AppState, request: &LoadRequest, message: String) -> ClipLoadOutcome {
		self.complete(state, request.clip_id, &request.path, Err(message))
	}

	/// Applies a finished background load to the session.
	///
	/// On success the clip slot becomes `Ready`, every layer already showing the clip gets
	/// its `frame_count` and `clip_fps`, and pending bindings for `path` are bound (layers
	/// stay `Stopped`, as restored layers never start playing on their own). On failure
	/// the slot shows the error and every layer waiting for it records the error, while
	/// other clips and layers are untouched (SPEC §9.1).
	pub fn complete(
		&mut self,
		state: &mut AppState,
		clip_id: ClipId,
		path: &Path,
		result: Result<ClipInfo, String>,
	) -> ClipLoadOutcome {
		let mut outcome = ClipLoadOutcome::default();
		self.loading.remove(&clip_id);

		let slot_matches = state
			.clip_by_id(clip_id)
			.is_some_and(|clip| clip.path.as_path() == path);
		if !slot_matches {
			// The clip was removed while it loaded: drop the result only. `forget`
			// already dropped the bindings that waited for this slot, so any binding
			// for `path` that exists now belongs to a newer load of the same file
			// (e.g. a project re-queued it under a new id) and must survive.
			return outcome;
		}
		outcome.accepted = true;

		match result {
			Ok(mut info) => {
				info.id = clip_id;
				info.path = path.to_path_buf();
				info.status = ClipStatus::Ready;
				let frame_count = info.frame_count;
				let fps = sanitize_fps(info.fps);
				state.upsert_clip(info);

				for (index, layer) in state.layers.iter_mut().enumerate() {
					if layer.active_clip == Some(clip_id) {
						layer.frame_count = frame_count;
						layer.clip_fps = fps;
						layer.playhead_frame = clamp_playhead(layer.playhead_frame, frame_count);
						set_flag(&mut outcome.hydrated_layers, index);
					}
				}

				let (ready, waiting): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending_bindings)
					.into_iter()
					.partition(|binding| binding.path == path);
				self.pending_bindings = waiting;
				for binding in ready {
					let selection = state.layer(binding.layer).selected_clip;
					if bind_layer_to_clip(state, binding.layer, clip_id, binding.playhead_frame) {
						if binding.keep_selection
							&& selection.is_some()
							&& let Some(layer) = state.layer_mut(binding.layer)
						{
							layer.selected_clip = selection;
						}
						set_flag(&mut outcome.bound_layers, binding.layer.index());
					}
				}
			}
			Err(message) => {
				if let Some(clip) = state.clips.iter_mut().find(|clip| clip.id == clip_id) {
					clip.status = ClipStatus::Error(message.clone());
					clip.thumbnail = None;
				}
				state.dirty = true;

				for (index, layer) in state.layers.iter_mut().enumerate() {
					if layer.active_clip == Some(clip_id) {
						layer.active_clip = None;
						layer.playback_state = PlaybackState::Stopped;
						layer.frame_count = 0;
						layer.playhead_frame = 0.0;
						layer.last_error = Some(message.clone());
						set_flag(&mut outcome.failed_layers, index);
					}
				}

				let (failed, waiting): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending_bindings)
					.into_iter()
					.partition(|binding| binding.path == path);
				self.pending_bindings = waiting;
				for binding in failed {
					if let Some(layer) = state.layer_mut(binding.layer) {
						layer.last_error = Some(message.clone());
						set_flag(&mut outcome.failed_layers, binding.layer.index());
					}
				}
				outcome.error = Some(message);
			}
		}
		outcome
	}
}

/// Placeholder shown in the clip grid while `path` loads.
pub fn placeholder_clip(clip_id: ClipId, path: &Path) -> ClipInfo {
	let name = path
		.file_name()
		.map(|name| name.to_string_lossy().into_owned())
		.unwrap_or_else(|| path.display().to_string());
	let is_movie = path
		.extension()
		.and_then(|ext| ext.to_str())
		.is_some_and(|ext| ext.eq_ignore_ascii_case("mov"));
	ClipInfo {
		id: clip_id,
		path: path.to_path_buf(),
		name,
		kind: if is_movie { ClipKind::Hap } else { ClipKind::ImageRgba },
		width: 0,
		height: 0,
		fps: DEFAULT_CLIP_FPS,
		frame_count: 0,
		has_alpha: false,
		status: ClipStatus::Ready,
		thumbnail: None,
	}
}

/// Binds a loaded clip to an idle layer without starting playback.
///
/// Returns `false` (and leaves the layer untouched) when the layer already shows another
/// clip, i.e. the operator took over the layer while the clip was loading.
pub fn bind_layer_to_clip(state: &mut AppState, layer: LayerIndex, clip_id: ClipId, playhead_frame: f64) -> bool {
	let Some((frame_count, fps)) = state
		.clip_by_id(clip_id)
		.filter(|clip| clip.is_ready())
		.map(|clip| (clip.frame_count, sanitize_fps(clip.fps)))
	else {
		return false;
	};
	let Some(state_layer) = state.layer_mut(layer) else {
		return false;
	};
	if state_layer.active_clip.is_some_and(|active| active != clip_id) {
		return false;
	}
	state_layer.selected_clip = Some(clip_id);
	state_layer.active_clip = Some(clip_id);
	state_layer.frame_count = frame_count;
	state_layer.clip_fps = fps;
	state_layer.playhead_frame = clamp_playhead(playhead_frame, frame_count);
	state_layer.playback_state = PlaybackState::Stopped;
	state_layer.last_error = None;
	true
}

fn sanitize_fps(fps: f32) -> f32 {
	if (MIN_CLIP_FPS..=MAX_CLIP_FPS).contains(&fps) {
		fps
	} else {
		DEFAULT_CLIP_FPS
	}
}

fn clamp_playhead(playhead: f64, frame_count: u32) -> f64 {
	if !playhead.is_finite() || frame_count == 0 {
		return 0.0;
	}
	playhead.clamp(0.0, f64::from(frame_count.saturating_sub(1)))
}

fn set_flag(flags: &mut [bool; LAYER_COUNT], index: usize) {
	if let Some(flag) = flags.get_mut(index) {
		*flag = true;
	}
}

fn flag(flags: &[bool; LAYER_COUNT], index: usize) -> bool {
	flags.get(index).copied().unwrap_or(false)
}

/// Returns `true` when an autosave is due at `now` and restarts the debounce window.
///
/// The saver keeps its configured debounce interval (it is never replaced by a default
/// one); only the dirty flag and the last-save instant are reset.
fn take_due_save(saver: &mut StateSaver, now: Instant) -> bool {
	if !saver.is_due(now) {
		return false;
	}
	saver.mark_saved(now);
	true
}

// ================================================================ session setup

/// Side effects of applying a project file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProjectApplyOutcome {
	/// Clips to hand to the background loader.
	pub requests: Vec<LoadRequest>,
	/// The output selection changed.
	pub needs_output_sync: bool,
	/// New MIDI port preference, when the project sets one.
	pub midi_port_changed: Option<Option<String>>,
	/// Layers whose streamer must be released because the project rebinds them.
	pub cleared_layers: [bool; LAYER_COUNT],
	/// Layers bound immediately to an already-loaded clip.
	pub bound_layers: [bool; LAYER_COUNT],
}

/// Restores `persisted` into `state` and queues every clip it references.
///
/// Effects, output selection, MIDI port, focused layer, and per-layer settings apply
/// immediately; layer clips are recorded as [`PendingBinding`]s and bound (stopped) once
/// the background loader delivers them.
pub fn apply_persisted_state(
	state: &mut AppState,
	library: &mut ClipLibrary,
	persisted: &PersistedState,
) -> Vec<LoadRequest> {
	persisted.restore_into(state, |_| None);

	let mut requests = Vec::new();
	for path in &persisted.clip_paths {
		requests.extend(library.enqueue(state, path));
	}
	for (layer, layer_state) in LayerIndex::ALL.into_iter().zip(persisted.layers.iter()) {
		if let Some(path) = layer_state.clip_path.as_deref() {
			requests.extend(library.enqueue(state, path));
			library.add_binding(PendingBinding {
				layer,
				path: path.to_path_buf(),
				playhead_frame: layer_state.playhead_frame,
				keep_selection: true,
			});
		}
		// The selection does not need the clip to be loaded: its slot id is known as
		// soon as it is queued. A selection-only clip never becomes active.
		if let Some(path) = layer_state.selected_clip_path.as_deref() {
			requests.extend(library.enqueue(state, path));
			let selected = state.clips.iter().find(|clip| clip.path == path).map(|clip| clip.id);
			if let Some(state_layer) = state.layer_mut(layer) {
				state_layer.selected_clip = selected;
			}
		}
	}
	state.dirty = false;
	requests
}

/// Media paths a project contributes: its explicit clip list, or else its clip folder.
pub fn project_clip_paths(project: &ProjectConfig) -> Vec<PathBuf> {
	match project.explicit_clips() {
		Some(clips) => clips.to_vec(),
		None => project
			.clip_folder
			.as_deref()
			.map(media::scan_media_dir)
			.unwrap_or_default(),
	}
}

/// Applies a project file on top of the current session.
///
/// `clip_paths` usually comes from [`project_clip_paths`]. Layer entries set opacity,
/// speed, blend mode, and playback mode right away; their clips are bound immediately
/// when already loaded, otherwise once the loader delivers them.
pub fn apply_project(
	state: &mut AppState,
	library: &mut ClipLibrary,
	project: &ProjectConfig,
	clip_paths: &[PathBuf],
) -> ProjectApplyOutcome {
	let mut outcome = ProjectApplyOutcome::default();

	if let Some(effects) = project.effects.as_ref() {
		state.effects = effects.clone();
		state.effects.clamp_all();
		state.dirty = true;
	}
	if let Some(output) = project.output.as_ref() {
		state.output = output.clone();
		state.dirty = true;
		outcome.needs_output_sync = true;
	}
	if let Some(port) = project.preferred_midi_port.as_ref() {
		state.selected_midi_port = Some(port.clone());
		state.dirty = true;
		outcome.midi_port_changed = Some(Some(port.clone()));
	}

	for path in clip_paths {
		outcome.requests.extend(library.enqueue(state, path));
	}

	for (layer, settings) in LayerIndex::ALL.into_iter().zip(project.layers.iter()) {
		if let Some(state_layer) = state.layer_mut(layer) {
			state_layer.playback_mode = settings.playback_mode;
			state_layer.blend_mode = settings.blend_mode;
			state_layer.set_speed(settings.speed);
			state_layer.set_opacity(settings.opacity);
		}
		state.dirty = true;

		let Some(path) = settings.clip_path.as_deref() else {
			continue;
		};
		outcome.requests.extend(library.enqueue(state, path));
		let loaded = state
			.clips
			.iter()
			.find(|clip| clip.path == path && clip.is_ready())
			.map(|clip| clip.id)
			.filter(|clip_id| !library.is_loading(*clip_id));

		if let Some(state_layer) = state.layer_mut(layer)
			&& state_layer.active_clip.is_some()
		{
			state_layer.active_clip = None;
			state_layer.playback_state = PlaybackState::Stopped;
			state_layer.frame_count = 0;
			set_flag(&mut outcome.cleared_layers, layer.index());
		}

		match loaded {
			Some(clip_id) => {
				library.cancel_binding(layer);
				if bind_layer_to_clip(state, layer, clip_id, settings.playhead_frame) {
					set_flag(&mut outcome.bound_layers, layer.index());
				}
			}
			None => library.add_binding(PendingBinding {
				layer,
				path: path.to_path_buf(),
				playhead_frame: settings.playhead_frame,
				keep_selection: false,
			}),
		}
	}
	outcome
}

/// Captures the persisted snapshot, keeping layer clips that are still waiting to load.
///
/// Without this, saving while clips load would forget the restored layer assignments.
pub fn capture_persisted(state: &AppState, library: &ClipLibrary) -> PersistedState {
	let mut snapshot = PersistedState::capture(state);
	for (layer, persisted) in LayerIndex::ALL.into_iter().zip(snapshot.layers.iter_mut()) {
		if persisted.clip_path.is_none()
			&& let Some(binding) = library.pending_binding(layer)
		{
			persisted.clip_path = Some(binding.path.clone());
			persisted.playhead_frame = binding.playhead_frame;
		}
	}
	snapshot
}

/// Auto-selects the first external display when no output display is chosen yet.
///
/// When `restored` is `false` (first launch) the output is also switched on. With no
/// external display the output stays off, so a single-display setup never hides the
/// control window. Returns `true` when a display was selected.
pub fn auto_select_output(output: &mut OutputDisplaySelection, monitors: &[MonitorSummary], restored: bool) -> bool {
	if output.monitor_name.is_some() {
		return false;
	}
	match platform::pick_default_external_monitor(monitors) {
		Some(monitor) => {
			output.monitor_name = Some(monitor.id.clone());
			if !restored {
				output.enabled = true;
			}
			true
		}
		None => {
			output.enabled = false;
			false
		}
	}
}

/// Toast to show when the output runtime status changes, if any.
pub fn output_transition_notice(
	previous: &OutputRuntimeStatus,
	next: &OutputRuntimeStatus,
) -> Option<(ToastLevel, String, String)> {
	if previous == next {
		return None;
	}
	match (previous, next) {
		(OutputRuntimeStatus::Active { monitor_name, .. }, OutputRuntimeStatus::WaitingForMonitor { .. }) => Some((
			ToastLevel::Warning,
			"Output display lost".to_owned(),
			format!("`{monitor_name}` disconnected; output is waiting for the display to return"),
		)),
		(OutputRuntimeStatus::WaitingForMonitor { .. }, OutputRuntimeStatus::Active { monitor_name, .. }) => Some((
			ToastLevel::Info,
			"Output restored".to_owned(),
			format!("Output resumed on `{monitor_name}`"),
		)),
		(_, OutputRuntimeStatus::Error(message)) => {
			Some((ToastLevel::Error, "Output window error".to_owned(), message.clone()))
		}
		_ => None,
	}
}

/// What the event loop must do after a command outcome regarding the exit guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitAction {
	/// Nothing exit related happened.
	None,
	/// The guard was opened: bring the control window forward.
	ShowGuard,
	/// The operator confirmed: save, close the output, and quit.
	Quit,
}

impl ExitAction {
	/// Derives the exit action from a command outcome.
	pub fn from_outcome(outcome: &UiApplyOutcome) -> Self {
		if outcome.confirm_quit {
			Self::Quit
		} else if outcome.exit_requested {
			Self::ShowGuard
		} else {
			Self::None
		}
	}
}

/// Applies `cmd` to the session, keeping [`ClipLibrary`] consistent with clip removals.
pub fn apply_session_command(state: &mut AppState, library: &mut ClipLibrary, cmd: &UiCommand) -> UiApplyOutcome {
	let outcome = apply_ui_command(state, cmd);
	if let Some(clip_id) = outcome.remove_clip {
		library.forget(clip_id);
	}
	for layer in LayerIndex::ALL {
		if outcome.was_retriggered(layer) || outcome.was_cleared(layer) {
			library.cancel_binding(layer);
		}
	}
	outcome
}

/// Routes any exit gesture (close button, `Alt+F4`, `Cmd+Q`, Dock "Quit") to the guard.
///
/// Never quits by itself: it only opens the confirmation overlay (SPEC §9.2).
pub fn request_exit(state: &mut AppState) -> UiApplyOutcome {
	apply_ui_command(state, &UiCommand::RequestExit)
}

/// Routes one native quit request (`Cmd+Q`, the app menu "Quit", Dock "Quit").
///
/// The first request opens the exit guard; a request arriving while the guard is already
/// open confirms it, so pressing `Cmd+Q` twice quits like a native macOS app. Callers must
/// only invoke this after consuming the native flag with [`platform::poll_quit_requested`],
/// so a single press is never routed twice.
pub fn handle_native_quit_request(state: &mut AppState) -> UiApplyOutcome {
	let cmd = if state.exit_confirm_open {
		UiCommand::ConfirmExit
	} else {
		UiCommand::RequestExit
	};
	apply_ui_command(state, &cmd)
}

/// Returns `true` for commands that may repeat while a key is held.
///
/// Only continuous `Adjust*` commands repeat; triggers, toggles, clip stepping, layer focus
/// cycling, and the exit guard do not, so a held key can never machine-gun a retrigger,
/// race through the clip grid, or flip a toggle back and forth.
pub fn command_allows_key_repeat(cmd: &UiCommand) -> bool {
	matches!(
		cmd,
		UiCommand::AdjustLayerOpacity { .. }
			| UiCommand::AdjustLayerSpeed { .. }
			| UiCommand::AdjustRgbShiftIntensity(_)
			| UiCommand::AdjustGlitchIntensity(_)
			| UiCommand::AdjustStrobeIntensity(_)
			| UiCommand::AdjustStrobeSpeedHz(_)
			| UiCommand::AdjustBrightness(_)
			| UiCommand::AdjustContrast(_)
			| UiCommand::AdjustSaturation(_)
			| UiCommand::AdjustMasterDimmer(_)
	)
}

/// How a dropped or picked path is handled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestKind {
	/// Scan the directory for media.
	Folder,
	/// Load as a TOML/JSON project.
	Project,
	/// Queue as a clip.
	Media,
	/// Not something MUR can use.
	Unsupported,
}

/// Classifies a dropped or picked path.
pub fn classify_ingest_path(path: &Path) -> IngestKind {
	if path.is_dir() {
		return IngestKind::Folder;
	}
	let extension = path
		.extension()
		.and_then(|ext| ext.to_str())
		.map(str::to_ascii_lowercase);
	match extension.as_deref() {
		Some("toml" | "json") => IngestKind::Project,
		_ if media::is_supported_media_path(path) => IngestKind::Media,
		_ => IngestKind::Unsupported,
	}
}

/// Pushes a toast and a log-drawer entry, mirroring it to `tracing`.
pub fn notify(state: &mut AppState, level: ToastLevel, title: impl Into<String>, message: impl Into<String>) {
	let title = title.into();
	let message = message.into();
	match level {
		ToastLevel::Info => tracing::info!(target: "app", "{title}: {message}"),
		ToastLevel::Warning => tracing::warn!(target: "app", "{title}: {message}"),
		ToastLevel::Error => tracing::error!(target: "app", "{title}: {message}"),
	}
	state.toasts.push_log(LogEntry::new(
		wall_clock_timestamp(),
		level,
		title.clone(),
		message.clone(),
	));
	let now = state.elapsed_secs;
	state.push_toast(level, title, message, now);
}

/// `HH:MM:SS` UTC wall-clock time for log-drawer entries.
fn wall_clock_timestamp() -> String {
	let secs = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map(|elapsed| elapsed.as_secs())
		.unwrap_or_default();
	let of_day = secs % 86_400;
	format!("{:02}:{:02}:{:02}Z", of_day / 3600, (of_day / 60) % 60, of_day % 60)
}

// ============================================================ background workers

/// Callback the clip loader uses to deliver a result; returns `false` once nobody listens.
pub type ClipLoadSink = Arc<dyn Fn(ClipId, PathBuf, ClipLoadResult) -> bool + Send + Sync>;

/// Panic-isolated background clip loader (SPEC §9.2).
///
/// Loads run on [`CLIP_LOADER_THREADS`] worker threads sharing one queue. Each load is
/// wrapped in `catch_unwind`, and a dead pool is respawned on the next request.
pub struct ClipLoader {
	sender: Option<Sender<LoadRequest>>,
	sink: ClipLoadSink,
}

impl std::fmt::Debug for ClipLoader {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("ClipLoader")
			.field("running", &self.sender.is_some())
			.finish()
	}
}

impl ClipLoader {
	/// Starts the loader threads.
	pub fn spawn(sink: ClipLoadSink) -> Self {
		let sender = Self::spawn_pool(&sink);
		Self { sender, sink }
	}

	fn spawn_pool(sink: &ClipLoadSink) -> Option<Sender<LoadRequest>> {
		let (sender, receiver) = crossbeam_channel::unbounded::<LoadRequest>();
		let mut spawned = 0usize;
		for index in 0..CLIP_LOADER_THREADS {
			let receiver = receiver.clone();
			let sink = Arc::clone(sink);
			let result = std::thread::Builder::new()
				.name(format!("mur-clip-loader-{index}"))
				.spawn(move || clip_loader_main(&receiver, &sink));
			match result {
				Ok(_) => spawned += 1,
				Err(error) => tracing::error!(target: "app", "failed to spawn clip loader thread: {error}"),
			}
		}
		(spawned > 0).then_some(sender)
	}

	/// Queues a load. Returns the request back when no loader thread can take it.
	pub fn request(&mut self, request: LoadRequest) -> Result<(), LoadRequest> {
		let Err(request) = self.try_send(request) else {
			return Ok(());
		};
		tracing::warn!(target: "app", "clip loader pool stopped; respawning");
		self.sender = Self::spawn_pool(&self.sink);
		self.try_send(request)
	}

	fn try_send(&self, request: LoadRequest) -> Result<(), LoadRequest> {
		match self.sender.as_ref() {
			Some(sender) => sender.send(request).map_err(|error| error.into_inner()),
			None => Err(request),
		}
	}
}

fn clip_loader_main(receiver: &Receiver<LoadRequest>, sink: &ClipLoadSink) {
	for request in receiver.iter() {
		let LoadRequest { clip_id, path } = request;
		let result = load_clip_isolated(clip_id, &path);
		if !sink(clip_id, path, result) {
			break;
		}
	}
}

/// Loads a clip with panics turned into errors.
pub fn load_clip_isolated(clip_id: ClipId, path: &Path) -> ClipLoadResult {
	match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| media::load_clip(clip_id, path))) {
		Ok(Ok(loaded)) => Ok(loaded),
		Ok(Err(error)) => Err(error.to_string()),
		Err(_) => Err(format!("loader panicked while reading `{}`", path.display())),
	}
}

/// Longest time quitting waits for the state writer to flush its last snapshot.
const STATE_WRITER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);

/// Writes state snapshots off the render thread so `fsync` never drops a frame.
struct StateWriter {
	sender: Option<Sender<(PathBuf, PersistedState)>>,
	errors: Receiver<String>,
	thread: Option<std::thread::JoinHandle<()>>,
}

impl StateWriter {
	fn spawn() -> Self {
		let (sender, receiver) = crossbeam_channel::unbounded::<(PathBuf, PersistedState)>();
		let (error_tx, errors) = crossbeam_channel::unbounded::<String>();
		let thread = std::thread::Builder::new()
			.name("mur-state-writer".to_owned())
			.spawn(move || {
				while let Ok(mut job) = receiver.recv() {
					// Coalesce bursts: only the newest snapshot matters.
					while let Ok(newer) = receiver.try_recv() {
						job = newer;
					}
					let (path, snapshot) = job;
					let result =
						std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| save_state_atomic(&path, &snapshot)));
					let message = match result {
						Ok(Ok(())) => continue,
						Ok(Err(error)) => error.to_string(),
						Err(_) => "state writer panicked".to_owned(),
					};
					let _ = error_tx.send(message);
				}
			});
		match thread {
			Ok(handle) => Self {
				sender: Some(sender),
				errors,
				thread: Some(handle),
			},
			Err(error) => {
				tracing::error!(target: "app", "failed to spawn state writer: {error}; saving inline");
				Self {
					sender: None,
					errors,
					thread: None,
				}
			}
		}
	}

	/// Queues `snapshot`; falls back to an inline save when the writer is gone.
	fn submit(&self, path: &Path, snapshot: PersistedState) -> Result<(), ConfigError> {
		let Some(sender) = self.sender.as_ref() else {
			return save_state_atomic(path, &snapshot);
		};
		match sender.send((path.to_path_buf(), snapshot)) {
			Ok(()) => Ok(()),
			Err(error) => {
				let (path, snapshot) = error.into_inner();
				save_state_atomic(&path, &snapshot)
			}
		}
	}

	fn take_error(&self) -> Option<String> {
		self.errors.try_recv().ok()
	}

	/// Stops the writer after it flushed pending snapshots.
	///
	/// Waits at most [`STATE_WRITER_SHUTDOWN_TIMEOUT`]: a save stuck in `fsync`
	/// (dying disk, hung network volume) is detached so quitting never hangs.
	fn shutdown(&mut self) {
		self.sender = None;
		if let Some(thread) = self.thread.take()
			&& !crate::thread_util::join_with_timeout(thread, STATE_WRITER_SHUTDOWN_TIMEOUT)
		{
			tracing::warn!(
				target: "app",
				timeout_ms = STATE_WRITER_SHUTDOWN_TIMEOUT.as_millis() as u64,
				"state writer did not finish in time; detaching it"
			);
		}
	}
}

impl Drop for StateWriter {
	fn drop(&mut self) {
		self.shutdown();
	}
}

// ================================================================ GPU plumbing

/// A wgpu surface plus its current configuration.
struct SurfaceSlot {
	surface: wgpu::Surface<'static>,
	config: Option<wgpu::SurfaceConfiguration>,
	needs_reconfigure: bool,
	/// `true` for the audience output (vsync), `false` for the control window, whose
	/// presentation must never block the output frame.
	prefer_vsync: bool,
}

/// Outcome of acquiring the next surface texture.
enum Acquire {
	/// A texture to draw into.
	Frame(wgpu::SurfaceTexture, wgpu::TextureFormat, u32, u32),
	/// Skip this frame (minimized, occluded, timeout, outdated, validation).
	Skip,
	/// The surface is lost and must be recreated.
	Lost,
}

impl SurfaceSlot {
	fn new(surface: wgpu::Surface<'static>, prefer_vsync: bool) -> Self {
		Self {
			surface,
			config: None,
			needs_reconfigure: true,
			prefer_vsync,
		}
	}

	fn acquire(&mut self, gpu: &GpuContext, size: PhysicalSize<u32>, label: &str) -> Acquire {
		if size.width == 0 || size.height == 0 {
			return Acquire::Skip;
		}
		let size_changed = self
			.config
			.as_ref()
			.is_none_or(|config| config.width != size.width || config.height != size.height);
		if self.needs_reconfigure || size_changed {
			match gpu.configure_surface_with_vsync(&self.surface, size.width, size.height, self.prefer_vsync) {
				Ok(config) => {
					self.config = Some(config);
					self.needs_reconfigure = false;
				}
				Err(error) => {
					tracing::warn!(target: "render", "{label}: surface configuration failed: {error}");
					self.needs_reconfigure = true;
					return Acquire::Skip;
				}
			}
		}
		let Some((format, width, height)) = self
			.config
			.as_ref()
			.map(|config| (config.format, config.width, config.height))
		else {
			return Acquire::Skip;
		};

		match self.surface.get_current_texture() {
			wgpu::CurrentSurfaceTexture::Success(texture) => Acquire::Frame(texture, format, width, height),
			wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
				self.needs_reconfigure = true;
				Acquire::Frame(texture, format, width, height)
			}
			wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => Acquire::Skip,
			wgpu::CurrentSurfaceTexture::Outdated => {
				self.needs_reconfigure = true;
				Acquire::Skip
			}
			wgpu::CurrentSurfaceTexture::Lost => {
				tracing::warn!(target: "render", "{label}: surface lost; recreating");
				Acquire::Lost
			}
			wgpu::CurrentSurfaceTexture::Validation => {
				tracing::warn!(target: "render", "{label}: validation error while acquiring the surface");
				self.needs_reconfigure = true;
				Acquire::Skip
			}
		}
	}
}

/// Device, renderers, and the control surface; rebuilt as a unit after device loss.
///
/// Field order is drop order: surfaces and renderers go before the device and instance.
struct GpuStack {
	egui_renderer: egui_wgpu::Renderer,
	master: MasterRenderer,
	control: SurfaceSlot,
	gpu: GpuContext,
}

impl GpuStack {
	fn create(window: &Arc<Window>) -> Result<Self, RenderError> {
		let instance = GpuContext::create_instance();
		let surface = instance
			.create_surface(Arc::clone(window))
			.map_err(|error| RenderError::SurfaceLost(format!("failed to create control surface: {error}")))?;
		let gpu = GpuContext::with_instance(instance, Some(&surface))?;
		let caps = surface.get_capabilities(&gpu.adapter);
		let format = select_surface_format(&caps.formats).ok_or_else(|| {
			RenderError::SurfaceLost("control surface is incompatible with the GPU adapter".to_owned())
		})?;
		let mut master = MasterRenderer::new(&gpu)?;
		master.set_cpu_fallback_decoder(|frame| crate::media::bc_decode::decode_frame_to_rgba8(frame).ok());
		let egui_renderer = egui_wgpu::Renderer::new(&gpu.device, format, egui_wgpu::RendererOptions::default());
		Ok(Self {
			egui_renderer,
			master,
			control: SurfaceSlot::new(surface, CONTROL_PREFERS_VSYNC),
			gpu,
		})
	}

	fn max_texture_side(&self) -> usize {
		usize::try_from(self.gpu.device.limits().max_texture_dimension_2d).unwrap_or(2048)
	}
}

/// The borderless output window and its surface.
struct OutputWindow {
	window: Arc<Window>,
	surface: Option<SurfaceSlot>,
	monitor_id: Option<String>,
	placement: Option<OutputPlacement>,
	occluded: bool,
}

/// Everything that only exists once the event loop is running.
struct Runtime {
	control: Arc<Window>,
	control_occluded: bool,
	control_monitor_id: Option<String>,
	egui_ctx: egui::Context,
	egui_state: egui_winit::State,
	gpu: Option<GpuStack>,
	gpu_lost: Option<String>,
	gpu_retry_at: Option<Instant>,
	mirror_texture_id: Option<egui::TextureId>,
	output: Option<OutputWindow>,
	monitors: Vec<MonitorDescriptor>,
	next_monitor_poll: Instant,
	needs_output_sync: bool,
	force_placement: bool,
	streamer: Option<LayerStreamer>,
	bound_clips: [Option<ClipId>; LAYER_COUNT],
	force_restart: [bool; LAYER_COUNT],
	midi: MidiWorkerHandle,
	loader: ClipLoader,
	dialog_open: Arc<AtomicBool>,
	modifiers: ModifiersState,
	last_tick: Instant,
	next_frame_at: Instant,
	last_gpu_error: Option<String>,
	last_render_error: Option<String>,
	/// Whether the control window title currently shows [`GPU_LOST_WINDOW_TITLE`].
	title_shows_gpu_lost: bool,
	/// Keeps macOS App Nap from throttling the tick while MUR runs.
	_app_nap: platform::AppNapGuard,
}

// ======================================================================== app

/// The winit application: session state plus the runtime created in `resumed`.
struct App {
	proxy: EventLoopProxy<AppEvent>,
	args: CliArgs,
	state: AppState,
	ui_state: UiState,
	library: ClipLibrary,
	state_path: PathBuf,
	saver: StateSaver,
	writer: StateWriter,
	save_error_reported: bool,
	restored: bool,
	startup_requests: Vec<LoadRequest>,
	runtime: Option<Runtime>,
	fatal_error: Option<MurError>,
	exit_saved: bool,
}

/// Runs MUR until the operator confirms exit.
///
/// Returns an error only when the event loop, the control window, or the initial GPU
/// device cannot be created; every later failure is reported in the UI instead.
pub fn run(args: CliArgs) -> Result<(), MurError> {
	let event_loop = EventLoop::<AppEvent>::with_user_event().build().map_err(|error| {
		MurError::Render(RenderError::InvalidTarget(format!(
			"failed to create event loop: {error}"
		)))
	})?;
	event_loop.set_control_flow(ControlFlow::Poll);

	let mut app = App::new(args, event_loop.create_proxy());
	event_loop
		.run_app(&mut app)
		.map_err(|error| MurError::Render(RenderError::InvalidTarget(format!("event loop failed: {error}"))))?;

	match app.fatal_error.take() {
		Some(error) => Err(error),
		None => Ok(()),
	}
}

impl App {
	fn new(args: CliArgs, proxy: EventLoopProxy<AppEvent>) -> Self {
		let paths = AppPaths::discover();
		let state_path = paths.resolve_state_file(args.state.as_deref());
		let mut state = AppState::default();
		let mut library = ClipLibrary::new();
		let mut startup_requests = Vec::new();
		let mut restored = false;
		let mut startup_notices: Vec<(ToastLevel, String, String)> = Vec::new();

		if !args.no_restore {
			match load_state(&state_path) {
				Ok(Some(persisted)) => {
					startup_requests.extend(apply_persisted_state(&mut state, &mut library, &persisted));
					restored = true;
					tracing::info!(target: "app", "restored state from {}", state_path.display());
				}
				Ok(None) => tracing::info!(target: "app", "no saved state at {}", state_path.display()),
				Err(error) => {
					startup_notices.push((ToastLevel::Warning, "Saved state ignored".to_owned(), error.to_string()))
				}
			}
		}

		if let Some(project_path) = args.project.as_deref() {
			match ProjectConfig::load(project_path) {
				Ok(project) => {
					let clip_paths = project_clip_paths(&project);
					let outcome = apply_project(&mut state, &mut library, &project, &clip_paths);
					startup_requests.extend(outcome.requests);
				}
				Err(error) => {
					startup_notices.push((ToastLevel::Error, "Project not loaded".to_owned(), error.to_string()));
				}
			}
		}

		if let Some(folder) = args.folder.as_deref() {
			let paths = media::scan_media_dir(folder);
			if paths.is_empty() {
				startup_notices.push((
					ToastLevel::Warning,
					"No media found".to_owned(),
					format!("`{}` contains no supported clips", folder.display()),
				));
			}
			for path in &paths {
				startup_requests.extend(library.enqueue(&mut state, path));
			}
		}

		for (level, title, message) in startup_notices {
			notify(&mut state, level, title, message);
		}
		// `state.dirty` stays set when a project or folder changed the session, so the
		// first autosave already captures it.

		Self {
			proxy,
			args,
			state,
			ui_state: UiState::new(),
			library,
			state_path,
			saver: StateSaver::with_default_debounce(),
			writer: StateWriter::spawn(),
			save_error_reported: false,
			restored,
			startup_requests,
			runtime: None,
			fatal_error: None,
			exit_saved: false,
		}
	}

	// ------------------------------------------------------------- startup

	fn start(&mut self, event_loop: &ActiveEventLoop) -> Result<(), MurError> {
		let attributes = WindowAttributes::default()
			.with_title(CONTROL_WINDOW_TITLE)
			.with_inner_size(LogicalSize::new(1440.0, 900.0))
			.with_min_inner_size(LogicalSize::new(960.0, 640.0))
			.with_visible(true);
		let control = Arc::new(event_loop.create_window(attributes).map_err(|error| {
			MurError::Render(RenderError::InvalidTarget(format!(
				"failed to create control window: {error}"
			)))
		})?);

		let waker_proxy = self.proxy.clone();
		platform::set_quit_waker(Box::new(move || {
			let _ = waker_proxy.send_event(AppEvent::QuitRequested);
		}));
		match platform::install_exit_guard() {
			Ok(status) => tracing::info!(
				target: "app",
				menu_items = status.intercepted_menu_items,
				terminate = status.intercepts_terminate,
				"exit guard installed"
			),
			Err(error) => {
				notify(
					&mut self.state,
					ToastLevel::Warning,
					"Exit guard incomplete",
					error.to_string(),
				);
			}
		}

		let gpu = GpuStack::create(&control)?;
		let egui_ctx = egui::Context::default();
		let egui_state = new_egui_state(&egui_ctx, &control, gpu.max_texture_side());

		let monitors = platform::collect_monitors(event_loop);
		self.state.available_monitors = platform::summaries(&monitors);
		if auto_select_output(&mut self.state.output, &self.state.available_monitors, self.restored) {
			tracing::info!(target: "app", monitor = ?self.state.output.monitor_name, "auto-selected output display");
			self.state.dirty = true;
		}

		let streamer = match LayerStreamer::new() {
			Ok(streamer) => Some(streamer),
			Err(error) => {
				notify(
					&mut self.state,
					ToastLevel::Error,
					"Playback unavailable",
					error.to_string(),
				);
				None
			}
		};
		let midi = MidiWorkerHandle::spawn(self.state.selected_midi_port.clone());

		let sink_proxy = self.proxy.clone();
		let loader = ClipLoader::spawn(Arc::new(move |clip_id, path, result| {
			sink_proxy
				.send_event(AppEvent::ClipLoaded { clip_id, path, result })
				.is_ok()
		}));

		let now = Instant::now();
		let control_monitor_id = platform::find_monitor_by_handle(&monitors, control.current_monitor().as_ref())
			.map(|monitor| monitor.summary.id.clone());
		self.runtime = Some(Runtime {
			control,
			control_occluded: false,
			control_monitor_id,
			egui_ctx,
			egui_state,
			gpu: Some(gpu),
			gpu_lost: None,
			gpu_retry_at: None,
			mirror_texture_id: None,
			output: None,
			monitors,
			next_monitor_poll: now + MONITOR_POLL_INTERVAL,
			needs_output_sync: true,
			force_placement: false,
			streamer,
			bound_clips: [None; LAYER_COUNT],
			force_restart: [false; LAYER_COUNT],
			midi,
			loader,
			dialog_open: Arc::new(AtomicBool::new(false)),
			modifiers: ModifiersState::empty(),
			last_tick: now,
			next_frame_at: now,
			last_gpu_error: None,
			last_render_error: None,
			title_shows_gpu_lost: false,
			_app_nap: platform::prevent_app_nap(),
		});

		let requests = std::mem::take(&mut self.startup_requests);
		self.submit_loads(requests);
		self.reconcile_output(event_loop);
		tracing::info!(
			target: "app",
			clips = self.state.clips.len(),
			project = ?self.args.project,
			folder = ?self.args.folder,
			"MUR started"
		);
		Ok(())
	}

	fn submit_loads(&mut self, requests: Vec<LoadRequest>) {
		let Some(rt) = self.runtime.as_mut() else {
			self.startup_requests.extend(requests);
			return;
		};
		for request in requests {
			if let Err(request) = rt.loader.request(request.clone()) {
				let outcome = self
					.library
					.fail(&mut self.state, &request, "clip loader is unavailable".to_owned());
				if outcome.accepted {
					notify(
						&mut self.state,
						ToastLevel::Error,
						"Clip not loaded",
						format!("`{}`: clip loader is unavailable", request.path.display()),
					);
				}
			}
		}
	}

	// ------------------------------------------------------------ commands

	fn handle_command(&mut self, event_loop: &ActiveEventLoop, cmd: &UiCommand) {
		let outcome = apply_session_command(&mut self.state, &mut self.library, cmd);
		self.apply_outcome(event_loop, &outcome);
	}

	fn apply_outcome(&mut self, event_loop: &ActiveEventLoop, outcome: &UiApplyOutcome) {
		if let Some(rt) = self.runtime.as_mut() {
			for layer in LayerIndex::ALL {
				let index = layer.index();
				if outcome.was_retriggered(layer) {
					set_flag(&mut rt.force_restart, index);
				}
				if outcome.was_cleared(layer) {
					if let Some(streamer) = rt.streamer.as_ref() {
						streamer.clear_layer(layer);
					}
					if let Some(bound) = rt.bound_clips.get_mut(index) {
						*bound = None;
					}
				}
			}
			if outcome.needs_output_sync {
				rt.needs_output_sync = true;
			}
			if let Some(port) = outcome.midi_port_changed.clone() {
				rt.midi.select_port(port);
			}
			if outcome.rescan_midi {
				rt.midi.rescan();
			}
			if let Some(clip_id) = outcome.remove_clip
				&& let Some(streamer) = rt.streamer.as_ref()
			{
				streamer.remove_asset(clip_id);
				for (bound, layer) in rt.bound_clips.iter_mut().zip(LayerIndex::ALL) {
					if *bound == Some(clip_id) {
						streamer.clear_layer(layer);
						*bound = None;
					}
				}
			}
		}
		if outcome.open_files {
			self.open_dialog(false);
		}
		if outcome.open_folder {
			self.open_dialog(true);
		}

		match ExitAction::from_outcome(outcome) {
			ExitAction::None => {}
			ExitAction::ShowGuard => {
				if let Some(rt) = self.runtime.as_ref() {
					rt.control.set_minimized(false);
					rt.control.focus_window();
					rt.control.request_user_attention(Some(UserAttentionType::Critical));
					rt.control.request_redraw();
				}
			}
			ExitAction::Quit => self.quit(event_loop),
		}
	}

	fn quit(&mut self, event_loop: &ActiveEventLoop) {
		self.save_now();
		if let Some(rt) = self.runtime.as_mut() {
			rt.output = None;
		}
		platform::allow_termination();
		tracing::info!(target: "app", "exit confirmed");
		event_loop.exit();
	}

	/// Synchronously saves the current state (exit path).
	fn save_now(&mut self) {
		if self.exit_saved {
			return;
		}
		self.writer.shutdown();
		let snapshot = capture_persisted(&self.state, &self.library);
		match save_state_atomic(&self.state_path, &snapshot) {
			Ok(()) => tracing::info!(target: "app", "state saved to {}", self.state_path.display()),
			Err(error) => tracing::error!(target: "app", "failed to save state on exit: {error}"),
		}
		self.exit_saved = true;
	}

	fn open_dialog(&mut self, folder: bool) {
		let Some(rt) = self.runtime.as_ref() else {
			return;
		};
		if rt.dialog_open.swap(true, Ordering::SeqCst) {
			return;
		}
		let proxy = self.proxy.clone();
		let open_flag = Arc::clone(&rt.dialog_open);
		let dialog = rfd::AsyncFileDialog::new().set_parent(rt.control.as_ref());

		// The future is created on the main thread: on macOS this begins a sheet attached
		// to the control window (non-blocking), elsewhere rfd spawns its own dialog
		// thread. Only the waiting happens on the helper thread below.
		let spawned = if folder {
			let future = dialog.set_title("Add media folder").pick_folder();
			std::thread::Builder::new()
				.name("mur-folder-dialog".to_owned())
				.spawn(move || {
					let picked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pollster::block_on(future)));
					open_flag.store(false, Ordering::SeqCst);
					match picked {
						Ok(Some(handle)) => {
							let _ = proxy.send_event(AppEvent::FolderPicked(handle.path().to_path_buf()));
						}
						Ok(None) => {}
						Err(_) => tracing::error!(target: "app", "folder dialog panicked"),
					}
				})
		} else {
			let future = dialog
				.set_title("Add media or project files")
				.add_filter("Media & projects", &DIALOG_EXTENSIONS)
				.pick_files();
			std::thread::Builder::new()
				.name("mur-files-dialog".to_owned())
				.spawn(move || {
					let picked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pollster::block_on(future)));
					open_flag.store(false, Ordering::SeqCst);
					match picked {
						Ok(Some(handles)) => {
							let paths = handles.iter().map(|handle| handle.path().to_path_buf()).collect();
							let _ = proxy.send_event(AppEvent::FilesPicked(paths));
						}
						Ok(None) => {}
						Err(_) => tracing::error!(target: "app", "file dialog panicked"),
					}
				})
		};
		if let Err(error) = spawned {
			rt.dialog_open.store(false, Ordering::SeqCst);
			notify(
				&mut self.state,
				ToastLevel::Error,
				"Dialog unavailable",
				error.to_string(),
			);
		}
	}

	// ------------------------------------------------------------ ingestion

	fn ingest_paths(&mut self, event_loop: &ActiveEventLoop, paths: &[PathBuf]) {
		let mut requests = Vec::new();
		let mut unsupported = Vec::new();
		for path in paths {
			match classify_ingest_path(path) {
				IngestKind::Folder => {
					let found = media::scan_media_dir(path);
					if found.is_empty() {
						notify(
							&mut self.state,
							ToastLevel::Warning,
							"No media found",
							format!("`{}` contains no supported clips", path.display()),
						);
					}
					for media_path in &found {
						requests.extend(self.library.enqueue(&mut self.state, media_path));
					}
				}
				IngestKind::Project => self.load_project(event_loop, path),
				IngestKind::Media => requests.extend(self.library.enqueue(&mut self.state, path)),
				IngestKind::Unsupported => unsupported.push(path.display().to_string()),
			}
		}
		if !unsupported.is_empty() {
			notify(
				&mut self.state,
				ToastLevel::Warning,
				"Unsupported files skipped",
				unsupported.join(", "),
			);
		}
		self.submit_loads(requests);
	}

	fn load_project(&mut self, event_loop: &ActiveEventLoop, path: &Path) {
		match ProjectConfig::load(path) {
			Ok(project) => {
				let clip_paths = project_clip_paths(&project);
				let outcome = apply_project(&mut self.state, &mut self.library, &project, &clip_paths);
				let clip_outcome = UiApplyOutcome {
					cleared_layers: outcome.cleared_layers,
					retriggered_layers: outcome.bound_layers,
					needs_output_sync: outcome.needs_output_sync,
					midi_port_changed: outcome.midi_port_changed.clone(),
					..UiApplyOutcome::default()
				};
				self.apply_outcome(event_loop, &clip_outcome);
				self.submit_loads(outcome.requests);
				notify(
					&mut self.state,
					ToastLevel::Info,
					"Project loaded",
					path.display().to_string(),
				);
			}
			Err(error) => notify(
				&mut self.state,
				ToastLevel::Error,
				"Project not loaded",
				error.to_string(),
			),
		}
	}

	fn on_clip_loaded(&mut self, clip_id: ClipId, path: &Path, result: ClipLoadResult) {
		let (info, asset) = match result {
			Ok((info, asset)) => (Ok(info), Some(asset)),
			Err(message) => (Err(message), None),
		};
		let outcome = self.library.complete(&mut self.state, clip_id, path, info);
		if !outcome.accepted {
			tracing::debug!(target: "app", clip = clip_id.0, "discarded load result of a removed clip");
			return;
		}
		if let Some(rt) = self.runtime.as_mut() {
			if let (Some(asset), Some(streamer)) = (asset, rt.streamer.as_ref()) {
				streamer.register_asset(asset);
			}
			for layer in LayerIndex::ALL {
				let index = layer.index();
				if outcome.layer_needs_restart(layer) {
					set_flag(&mut rt.force_restart, index);
				}
				if flag(&outcome.failed_layers, index)
					&& let Some(streamer) = rt.streamer.as_ref()
				{
					streamer.clear_layer(layer);
					if let Some(bound) = rt.bound_clips.get_mut(index) {
						*bound = None;
					}
				}
			}
		}
		if let Some(message) = outcome.error {
			let name = path
				.file_name()
				.map(|name| name.to_string_lossy().into_owned())
				.unwrap_or_else(|| path.display().to_string());
			notify(
				&mut self.state,
				ToastLevel::Error,
				format!("Clip `{name}` failed to load"),
				message,
			);
		}
	}

	// ------------------------------------------------------------- output

	fn set_output_status(&mut self, next: OutputRuntimeStatus) {
		if let Some((level, title, message)) = output_transition_notice(&self.state.output_status, &next) {
			notify(&mut self.state, level, title, message);
		}
		self.state.output_status = next;
	}

	fn refresh_monitors(&mut self, event_loop: &ActiveEventLoop) {
		let Some(rt) = self.runtime.as_mut() else {
			return;
		};
		let monitors = platform::collect_monitors(event_loop);
		let summaries = platform::summaries(&monitors);
		if summaries != self.state.available_monitors {
			tracing::info!(target: "app", count = summaries.len(), "display configuration changed");
			self.state.available_monitors = summaries;
			rt.needs_output_sync = true;
			rt.force_placement = true;
		}
		// Identify the control display through the fresh list, never by querying its
		// (possibly stale) handle directly.
		let control_monitor_id = platform::find_monitor_by_handle(&monitors, rt.control.current_monitor().as_ref())
			.map(|monitor| monitor.summary.id.clone());
		rt.monitors = monitors;
		if control_monitor_id != rt.control_monitor_id {
			rt.control_monitor_id = control_monitor_id;
			rt.force_placement = true;
			rt.needs_output_sync = true;
		}
		// The follow-up reconcile also retries surface creation after transient errors.
		rt.needs_output_sync = true;
	}

	fn drop_output(&mut self) {
		if let Some(rt) = self.runtime.as_mut()
			&& let Some(mut output) = rt.output.take()
		{
			output.surface = None;
			output.window.set_visible(false);
			tracing::info!(target: "app", "output window closed");
		}
	}

	fn reconcile_output(&mut self, event_loop: &ActiveEventLoop) {
		// Always decide on a fresh monitor list: a display unplugged since the last poll
		// must neither be targeted through a stale handle nor pull the output onto the
		// control display.
		self.refresh_monitors(event_loop);
		let Some(rt) = self.runtime.as_mut() else {
			return;
		};
		rt.needs_output_sync = false;

		if !self.state.output.enabled {
			self.drop_output();
			self.set_output_status(OutputRuntimeStatus::Disabled);
			return;
		}

		let Some(desc) = platform::find_matching_monitor_excluding(
			&rt.monitors,
			self.state.output.monitor_name.as_deref(),
			rt.control_monitor_id.as_deref(),
		)
		.cloned() else {
			self.drop_output();
			let expected_name = self.state.output.monitor_name.clone();
			self.set_output_status(OutputRuntimeStatus::WaitingForMonitor { expected_name });
			return;
		};

		if rt.output.is_none() {
			match create_output_window(event_loop, &desc) {
				Ok(window) => {
					rt.output = Some(OutputWindow {
						window,
						surface: None,
						monitor_id: None,
						placement: None,
						occluded: false,
					});
				}
				Err(message) => {
					self.set_output_status(OutputRuntimeStatus::Error(message));
					return;
				}
			}
		}

		let force_placement = std::mem::take(&mut rt.force_placement);
		let control_monitor = rt
			.control_monitor_id
			.as_deref()
			.and_then(|id| rt.monitors.iter().find(|monitor| monitor.summary.id == id))
			.map(|monitor| monitor.handle.clone());
		let Some(output) = rt.output.as_mut() else {
			return;
		};

		let needs_placement = force_placement
			|| output.placement.is_none()
			|| output.monitor_id.as_deref() != Some(desc.summary.id.as_str());
		if needs_placement {
			match platform::configure_output_window(&output.window, &desc.handle, control_monitor.as_ref()) {
				Ok(placement) => {
					#[cfg(not(target_os = "macos"))]
					{
						output.window.set_visible(true);
						rt.control.focus_window();
					}
					output.placement = Some(placement);
					output.monitor_id = Some(desc.summary.id.clone());
					if let Some(slot) = output.surface.as_mut() {
						slot.needs_reconfigure = true;
					}
					tracing::info!(
						target: "app",
						monitor = %desc.summary.name,
						width = placement.width,
						height = placement.height,
						"output window placed"
					);
				}
				Err(PlatformError::MonitorUnavailable(name)) => {
					tracing::warn!(target: "app", "output display `{name}` vanished during placement");
					self.drop_output();
					let expected_name = self.state.output.monitor_name.clone();
					self.set_output_status(OutputRuntimeStatus::WaitingForMonitor { expected_name });
					return;
				}
				Err(error) => {
					self.drop_output();
					self.set_output_status(OutputRuntimeStatus::Error(error.to_string()));
					return;
				}
			}
		}

		if output.surface.is_none() {
			let Some(stack) = rt.gpu.as_ref() else {
				return;
			};
			match stack.gpu.create_surface(Arc::clone(&output.window)) {
				Ok(surface) => output.surface = Some(SurfaceSlot::new(surface, OUTPUT_PREFERS_VSYNC)),
				Err(error) => {
					let message = error.to_string();
					self.set_output_status(OutputRuntimeStatus::Error(message));
					return;
				}
			}
		}

		let Some(placement) = output.placement else {
			return;
		};
		self.set_output_status(OutputRuntimeStatus::Active {
			monitor_name: desc.summary.name.clone(),
			width: placement.width,
			height: placement.height,
		});
	}

	// ---------------------------------------------------------------- GPU

	fn recover_gpu(&mut self) {
		let Some(rt) = self.runtime.as_mut() else {
			return;
		};
		let now = Instant::now();
		if rt.gpu_retry_at.is_some_and(|retry_at| now < retry_at) {
			return;
		}
		let reason = rt.gpu_lost.clone().unwrap_or_else(|| "GPU device lost".to_owned());
		let first_attempt = rt.gpu_retry_at.is_none();

		// Surfaces belong to the old instance; drop them before the device.
		if let Some(output) = rt.output.as_mut() {
			output.surface = None;
		}
		rt.mirror_texture_id = None;
		rt.gpu = None;
		rt.bound_clips = [None; LAYER_COUNT];

		match GpuStack::create(&rt.control) {
			Ok(stack) => {
				let egui_ctx = egui::Context::default();
				rt.egui_state = new_egui_state(&egui_ctx, &rt.control, stack.max_texture_side());
				rt.egui_ctx = egui_ctx;
				rt.gpu = Some(stack);
				rt.gpu_lost = None;
				rt.gpu_retry_at = None;
				rt.needs_output_sync = true;
				rt.force_restart = [true; LAYER_COUNT];
				let hovered = self.ui_state.hovered_files;
				self.ui_state = UiState::new();
				self.ui_state.hovered_files = hovered;
				notify(&mut self.state, ToastLevel::Warning, "GPU recovered", reason);
			}
			Err(error) => {
				rt.gpu_retry_at = Some(now + GPU_RETRY_INTERVAL);
				if first_attempt {
					notify(
						&mut self.state,
						ToastLevel::Error,
						"GPU recovery failed",
						format!("{reason}; retrying: {error}"),
					);
				} else {
					tracing::warn!(target: "render", "GPU recovery attempt failed: {error}");
				}
			}
		}
	}

	// --------------------------------------------------------------- tick

	fn tick(&mut self, event_loop: &ActiveEventLoop) {
		let started = Instant::now();
		let Some(rt) = self.runtime.as_mut() else {
			return;
		};
		let dt = started.saturating_duration_since(rt.last_tick);
		rt.last_tick = started;
		rt.next_frame_at += TARGET_FRAME;
		if rt.next_frame_at <= started {
			rt.next_frame_at = started + TARGET_FRAME;
		}
		let dt_secs = dt.as_secs_f32().min(MAX_TICK_DT_SECS);

		// 1. Native quit requests.
		if platform::poll_quit_requested() {
			let outcome = handle_native_quit_request(&mut self.state);
			self.apply_outcome(event_loop, &outcome);
		}

		// 2-3. Worker events.
		self.drain_midi();
		self.drain_streamer();

		// 4. Simulation.
		self.state.advance_layers(dt_secs);
		self.state.toasts.prune_expired(self.state.elapsed_secs);

		// Display hot-plug and output reconciliation.
		// `reconcile_output` re-enumerates the monitors itself, so a due poll only has to
		// request a reconcile.
		if let Some(rt) = self.runtime.as_mut()
			&& started >= rt.next_monitor_poll
		{
			rt.next_monitor_poll = started + MONITOR_POLL_INTERVAL;
			rt.needs_output_sync = true;
		}
		if self.runtime.as_ref().is_some_and(|rt| rt.needs_output_sync) {
			self.reconcile_output(event_loop);
		}

		// GPU loss recovery.
		if self
			.runtime
			.as_ref()
			.is_some_and(|rt| rt.gpu_lost.is_some() || rt.gpu.is_none())
		{
			if let Some(rt) = self.runtime.as_mut()
				&& rt.gpu_lost.is_none()
			{
				rt.gpu_lost = Some("GPU unavailable".to_owned());
			}
			self.recover_gpu();
		}

		// 5-7. Streaming, master render, presentation.
		let commands = self.render_frame_guarded();
		self.update_control_title();
		for command in &commands {
			self.handle_command(event_loop, command);
		}

		// 8. Performance and persistence.
		self.update_perf(dt, started.elapsed());
		self.persist();
	}

	fn drain_midi(&mut self) {
		let Some(rt) = self.runtime.as_mut() else {
			return;
		};
		let mut events = Vec::new();
		while events.len() < MAX_MIDI_EVENTS_PER_TICK {
			match rt.midi.try_recv() {
				Some(event) => events.push(event),
				None => break,
			}
		}
		for event in events {
			match event {
				MidiEvent::Action(action) => {
					let result = apply_midi_action(&mut self.state, action);
					for layer in result.layers() {
						set_flag(&mut rt.force_restart, layer.index());
						self.library.cancel_binding(layer);
					}
				}
				MidiEvent::StatusChanged(status) => {
					if let MidiConnectionStatus::Connected { port_name } = &status
						&& !self.state.midi_status.is_connected()
					{
						let message = format!("Connected to `{port_name}`");
						notify(&mut self.state, ToastLevel::Info, "MIDI connected", message);
					}
					self.state.midi_status = status;
				}
				MidiEvent::PortsUpdated(ports) => self.state.midi_ports = ports,
				MidiEvent::Warning(message) => notify(&mut self.state, ToastLevel::Warning, "MIDI", message),
			}
		}
	}

	fn drain_streamer(&mut self) {
		let Some(rt) = self.runtime.as_mut() else {
			return;
		};
		let Some(streamer) = rt.streamer.as_ref() else {
			return;
		};
		for event in streamer.drain_events() {
			match event {
				LayerStreamEvent::Error {
					layer,
					clip_id,
					generation,
					message,
				} => {
					let current = streamer.layer_generation(layer) == Some(generation);
					let showing = self.state.layer(layer).active_clip == Some(clip_id);
					if !current || !showing {
						continue;
					}
					if let Some(state_layer) = self.state.layer_mut(layer) {
						state_layer.active_clip = None;
						state_layer.playback_state = PlaybackState::Stopped;
						state_layer.playhead_frame = 0.0;
						state_layer.last_error = Some(message.clone());
					}
					streamer.clear_layer(layer);
					if let Some(bound) = rt.bound_clips.get_mut(layer.index()) {
						*bound = None;
					}
					notify(
						&mut self.state,
						ToastLevel::Error,
						format!("Layer {layer} stopped"),
						message,
					);
				}
				LayerStreamEvent::WorkerRestarted { layer, restarts } => notify(
					&mut self.state,
					ToastLevel::Warning,
					format!("Layer {layer} decoder restarted"),
					format!("decode worker recovered from a panic ({restarts} restarts)"),
				),
			}
		}
	}

	/// Runs [`App::render_frame`] with panic containment.
	///
	/// wgpu and egui-wgpu panic on some device-loss and validation paths. A panic inside
	/// the frame marks the GPU as lost instead of unwinding out of the event loop, so the
	/// regular recovery rebuilds the device (and the egui context) after
	/// [`GPU_RETRY_INTERVAL`], so a persistent panic backs off instead of spinning.
	fn render_frame_guarded(&mut self) -> Vec<UiCommand> {
		// A stack already marked lost is waiting out the retry back-off; rendering with it
		// again would re-panic every tick and keep pushing the retry out indefinitely.
		if self
			.runtime
			.as_ref()
			.is_some_and(|rt| rt.gpu.is_some() && rt.gpu_lost.is_some())
		{
			return Vec::new();
		}
		match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.render_frame())) {
			Ok(commands) => commands,
			Err(payload) => {
				let detail = panic_message(payload.as_ref());
				tracing::error!(target: "render", "panic while rendering a frame: {detail}");
				if let Some(rt) = self.runtime.as_mut() {
					rt.gpu_lost = Some(format!("GPU frame panicked: {detail}"));
					rt.gpu_retry_at = Some(Instant::now() + GPU_RETRY_INTERVAL);
				}
				Vec::new()
			}
		}
	}

	/// Shows [`GPU_LOST_WINDOW_TITLE`] on the control window while the GPU is unavailable.
	fn update_control_title(&mut self) {
		let Some(rt) = self.runtime.as_mut() else {
			return;
		};
		let lost = rt.gpu.is_none() || rt.gpu_lost.is_some();
		if lost != rt.title_shows_gpu_lost {
			rt.control.set_title(if lost {
				GPU_LOST_WINDOW_TITLE
			} else {
				CONTROL_WINDOW_TITLE
			});
			rt.title_shows_gpu_lost = lost;
		}
	}

	/// Syncs layers, renders the master frame, presents the output, then the control UI.
	///
	/// Returns the commands issued through the control UI this frame.
	fn render_frame(&mut self) -> Vec<UiCommand> {
		let Some(rt) = self.runtime.as_mut() else {
			return Vec::new();
		};

		// 5. Streamer sync (non-blocking).
		let frames = sync_layers(
			rt.streamer.as_ref(),
			&self.state,
			&self.library,
			&mut rt.bound_clips,
			&mut rt.force_restart,
		);

		let Some(stack) = rt.gpu.as_mut() else {
			return Vec::new();
		};
		let _ = stack.gpu.device.poll(wgpu::PollType::Poll);

		// 6. Master render.
		let inputs: [LayerRenderInput<'_>; LAYER_COUNT] = std::array::from_fn(|index| {
			layer_render_input(&self.state, index, frames.get(index).and_then(|frame| frame.as_deref()))
		});
		let time_secs = (self.state.elapsed_secs % SHADER_TIME_WRAP_SECS) as f32;
		match stack
			.master
			.render_master(&stack.gpu, &inputs, &self.state.effects, time_secs)
		{
			Ok(_) => rt.last_render_error = None,
			Err(RenderError::DeviceLost(message)) => {
				rt.gpu_lost = Some(message);
				return Vec::new();
			}
			Err(error) => {
				let message = error.to_string();
				if rt.last_render_error.as_deref() != Some(message.as_str()) {
					tracing::warn!(target: "render", "master render failed: {message}");
					rt.last_render_error = Some(message);
				}
			}
		}
		if stack.gpu.is_device_lost() {
			rt.gpu_lost = Some(
				stack
					.gpu
					.take_last_error()
					.unwrap_or_else(|| "GPU device lost".to_owned()),
			);
			return Vec::new();
		}
		if let Some(error) = stack.gpu.take_last_error()
			&& rt.last_gpu_error.as_deref() != Some(error.as_str())
		{
			tracing::warn!(target: "render", "GPU error: {error}");
			rt.last_gpu_error = Some(error);
		}

		// 7a. Output window first: the audience view has priority.
		if self.state.output_status.is_active()
			&& let Some(output) = rt.output.as_mut()
			&& !output.occluded
		{
			let size = output.window.inner_size();
			let lost = match output
				.surface
				.as_mut()
				.map(|slot| slot.acquire(&stack.gpu, size, "output"))
			{
				Some(Acquire::Frame(frame, format, width, height)) => {
					let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
					match stack.master.present_surface(&stack.gpu, &view, format, width, height) {
						Ok(()) => stack.gpu.queue.present(frame),
						Err(RenderError::DeviceLost(message)) => rt.gpu_lost = Some(message),
						Err(error) => tracing::warn!(target: "render", "output present failed: {error}"),
					}
					false
				}
				Some(Acquire::Lost) => true,
				Some(Acquire::Skip) | None => false,
			};
			if lost {
				output.surface = None;
				rt.needs_output_sync = true;
			}
		}
		if rt.gpu_lost.is_some() {
			return Vec::new();
		}

		// 7b. Control window with the live mirror.
		if rt.control_occluded {
			return Vec::new();
		}
		let size = rt.control.inner_size();
		let (frame, width, height) = match stack.control.acquire(&stack.gpu, size, "control") {
			Acquire::Frame(frame, _, width, height) => (frame, width, height),
			Acquire::Skip => return Vec::new(),
			Acquire::Lost => {
				match stack.gpu.create_surface(Arc::clone(&rt.control)) {
					Ok(surface) => stack.control = SurfaceSlot::new(surface, CONTROL_PREFERS_VSYNC),
					Err(error) => rt.gpu_lost = Some(error.to_string()),
				}
				return Vec::new();
			}
		};

		let mirror =
			stack
				.master
				.sync_mirror_texture(&stack.gpu.device, &mut stack.egui_renderer, &mut rt.mirror_texture_id);
		rt.mirror_texture_id = Some(mirror);

		let raw_input = rt.egui_state.take_egui_input(&rt.control);
		self.ui_state.sync_thumbnails(&rt.egui_ctx, &self.state.clips);
		let mut commands = Vec::new();
		let state = &self.state;
		let ui_state = &mut self.ui_state;
		let mirror_texture_id = rt.mirror_texture_id;
		let full_output = rt.egui_ctx.run_ui(raw_input, |ui| {
			commands.extend(crate::ui::layout::draw_control_ui_in(
				ui,
				state,
				ui_state,
				mirror_texture_id,
			));
		});
		rt.egui_state
			.handle_platform_output(&rt.control, full_output.platform_output);

		let pixels_per_point = full_output.pixels_per_point;
		let paint_jobs = rt.egui_ctx.tessellate(full_output.shapes, pixels_per_point);
		let screen = egui_wgpu::ScreenDescriptor {
			size_in_pixels: [width, height],
			pixels_per_point,
		};
		// The device may have been lost while egui ran; uploading textures or buffers to a
		// lost device can panic inside egui-wgpu, so bail out into recovery first.
		if stack.gpu.is_device_lost() {
			rt.gpu_lost = Some(
				stack
					.gpu
					.take_last_error()
					.unwrap_or_else(|| "GPU device lost".to_owned()),
			);
			return commands;
		}
		let device = Arc::clone(&stack.gpu.device);
		let queue = Arc::clone(&stack.gpu.queue);
		for (id, deltas) in &full_output.textures_delta.set {
			for delta in deltas {
				stack.egui_renderer.update_texture(&device, &queue, *id, delta);
			}
		}
		let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("mur.control.encoder"),
		});
		let mut buffers = stack
			.egui_renderer
			.update_buffers(&device, &queue, &mut encoder, &paint_jobs, &screen);
		let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
		{
			let mut pass = encoder
				.begin_render_pass(&wgpu::RenderPassDescriptor {
					label: Some("mur.control.egui"),
					color_attachments: &[Some(wgpu::RenderPassColorAttachment {
						view: &view,
						depth_slice: None,
						resolve_target: None,
						ops: wgpu::Operations {
							load: wgpu::LoadOp::Clear(CONTROL_CLEAR_COLOR),
							store: wgpu::StoreOp::Store,
						},
					})],
					depth_stencil_attachment: None,
					timestamp_writes: None,
					occlusion_query_set: None,
					multiview_mask: None,
				})
				.forget_lifetime();
			stack.egui_renderer.render(&mut pass, &paint_jobs, &screen);
		}
		buffers.push(encoder.finish());
		queue.submit(buffers);
		queue.present(frame);
		for id in &full_output.textures_delta.free {
			stack.egui_renderer.free_texture(id);
		}
		commands
	}

	fn update_perf(&mut self, dt: Duration, work: Duration) {
		let perf = &mut self.state.perf;
		perf.total_frames = perf.total_frames.saturating_add(1);
		let dt_secs = dt.as_secs_f32();
		if dt_secs > 0.0 && dt_secs.is_finite() {
			let fps = 1.0 / dt_secs;
			perf.fps = if perf.fps > 0.0 {
				perf.fps + (fps - perf.fps) * PERF_EMA_ALPHA
			} else {
				fps
			};
		}
		let work_ms = work.as_secs_f32() * 1000.0;
		perf.frame_time_ms = if perf.frame_time_ms > 0.0 {
			perf.frame_time_ms + (work_ms - perf.frame_time_ms) * PERF_EMA_ALPHA
		} else {
			work_ms
		};
		if perf.total_frames > 1 && dt > TARGET_FRAME + TARGET_FRAME / 2 {
			let missed = (dt.as_secs_f64() / TARGET_FRAME.as_secs_f64()).floor() as u64;
			perf.dropped_frames = perf.dropped_frames.saturating_add(missed.saturating_sub(1).max(1));
		}
	}

	fn persist(&mut self) {
		if self.exit_saved {
			return;
		}
		if self.state.dirty {
			self.saver.mark_dirty();
			self.state.dirty = false;
		}
		if let Some(error) = self.writer.take_error() {
			self.report_save_error(error);
		}
		if !take_due_save(&mut self.saver, Instant::now()) {
			return;
		}
		let snapshot = capture_persisted(&self.state, &self.library);
		match self.writer.submit(&self.state_path, snapshot) {
			Ok(()) => self.save_error_reported = false,
			Err(error) => self.report_save_error(error.to_string()),
		}
	}

	fn report_save_error(&mut self, message: String) {
		self.saver.mark_dirty();
		if !self.save_error_reported {
			self.save_error_reported = true;
			notify(&mut self.state, ToastLevel::Warning, "Autosave failed", message);
		} else {
			tracing::warn!(target: "app", "autosave failed: {message}");
		}
	}

	// ------------------------------------------------------------- events

	fn is_control(&self, window_id: WindowId) -> bool {
		self.runtime.as_ref().is_some_and(|rt| rt.control.id() == window_id)
	}

	fn is_output(&self, window_id: WindowId) -> bool {
		self.runtime
			.as_ref()
			.and_then(|rt| rt.output.as_ref())
			.is_some_and(|output| output.window.id() == window_id)
	}

	fn on_key(&mut self, event_loop: &ActiveEventLoop, event: &KeyEvent, from_control: bool) {
		if event.state != ElementState::Pressed {
			return;
		}
		let Some(rt) = self.runtime.as_ref() else {
			return;
		};
		let egui_wants_keyboard = from_control && rt.egui_ctx.text_edit_focused();
		let Some(command) = map_key_event(
			&event.logical_key,
			event.physical_key,
			rt.modifiers,
			self.state.focused_layer,
			self.state.exit_confirm_open,
			egui_wants_keyboard,
		) else {
			return;
		};
		if event.repeat && !command_allows_key_repeat(&command) {
			return;
		}
		self.handle_command(event_loop, &command);
	}
}

impl ApplicationHandler<AppEvent> for App {
	fn resumed(&mut self, event_loop: &ActiveEventLoop) {
		if self.runtime.is_some() || self.fatal_error.is_some() {
			return;
		}
		if let Err(error) = self.start(event_loop) {
			tracing::error!(target: "app", "startup failed: {error}");
			self.fatal_error = Some(error);
			event_loop.exit();
		}
	}

	fn user_event(&mut self, event_loop: &ActiveEventLoop, event: AppEvent) {
		match event {
			AppEvent::QuitRequested => {
				// Only the caller that consumes the native flag routes the request, so the
				// wake-up event and the next tick never handle one press twice.
				if platform::poll_quit_requested() {
					let outcome = handle_native_quit_request(&mut self.state);
					self.apply_outcome(event_loop, &outcome);
				}
			}
			AppEvent::ClipLoaded { clip_id, path, result } => self.on_clip_loaded(clip_id, &path, result),
			AppEvent::FilesPicked(paths) => self.ingest_paths(event_loop, &paths),
			AppEvent::FolderPicked(folder) => self.ingest_paths(event_loop, &[folder]),
		}
	}

	fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
		let from_control = self.is_control(window_id);
		let from_output = self.is_output(window_id);
		if !from_control && !from_output {
			return;
		}

		if from_control && let Some(rt) = self.runtime.as_mut() {
			let _ = rt.egui_state.on_window_event(&rt.control, &event);
		}

		match &event {
			WindowEvent::CloseRequested => {
				let outcome = request_exit(&mut self.state);
				self.apply_outcome(event_loop, &outcome);
			}
			WindowEvent::ModifiersChanged(modifiers) => {
				if let Some(rt) = self.runtime.as_mut() {
					rt.modifiers = modifiers.state();
				}
			}
			WindowEvent::KeyboardInput { event, .. } => self.on_key(event_loop, event, from_control),
			WindowEvent::HoveredFile(_) => {
				self.ui_state.hovered_files = self.ui_state.hovered_files.saturating_add(1);
			}
			WindowEvent::HoveredFileCancelled => self.ui_state.hovered_files = 0,
			WindowEvent::DroppedFile(path) => {
				self.ui_state.hovered_files = 0;
				let path = path.clone();
				self.ingest_paths(event_loop, &[path]);
			}
			WindowEvent::Occluded(occluded) => {
				if let Some(rt) = self.runtime.as_mut() {
					if from_control {
						rt.control_occluded = *occluded;
					} else if let Some(output) = rt.output.as_mut() {
						output.occluded = *occluded;
					}
				}
			}
			WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
				let scale_changed = matches!(event, WindowEvent::ScaleFactorChanged { .. });
				if let Some(rt) = self.runtime.as_mut() {
					if from_output {
						if let Some(slot) = rt.output.as_mut().and_then(|output| output.surface.as_mut()) {
							slot.needs_reconfigure = true;
						}
					} else {
						rt.control.request_redraw();
					}
				}
				// Output resizes and DPI changes usually mean the display layout changed
				// (hot-unplug, resolution switch): re-enumerate right away.
				if from_output || scale_changed {
					self.refresh_monitors(event_loop);
				}
			}
			WindowEvent::Moved(_) => {
				// Re-enumerating also re-identifies the control display through the fresh
				// monitor list and forces a re-placement when it changed.
				self.refresh_monitors(event_loop);
			}
			WindowEvent::RedrawRequested => {
				let due = self
					.runtime
					.as_ref()
					.is_some_and(|rt| rt.last_tick.elapsed() >= MIN_REDRAW_TICK_SPACING);
				if from_control && due {
					self.tick(event_loop);
				}
			}
			_ => {}
		}
	}

	fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
		let Some(next_frame_at) = self.runtime.as_ref().map(|rt| rt.next_frame_at) else {
			return;
		};
		// Ticks are driven by the clock rather than by `RedrawRequested`, so the output
		// keeps running at 60 fps even when the control window is occluded or minimized.
		if Instant::now() >= next_frame_at {
			self.tick(event_loop);
		}
		if let Some(rt) = self.runtime.as_ref() {
			event_loop.set_control_flow(ControlFlow::WaitUntil(rt.next_frame_at));
		}
	}

	fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
		self.save_now();
		self.drop_output();
	}
}

/// Extracts a readable message from a panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
	if let Some(message) = payload.downcast_ref::<&str>() {
		(*message).to_owned()
	} else if let Some(message) = payload.downcast_ref::<String>() {
		message.clone()
	} else {
		"unknown panic".to_owned()
	}
}

/// Creates the egui-winit input state for the control window.
fn new_egui_state(ctx: &egui::Context, window: &Window, max_texture_side: usize) -> egui_winit::State {
	egui_winit::State::new(
		ctx.clone(),
		egui::ViewportId::ROOT,
		window,
		Some(window.scale_factor() as f32),
		window.theme(),
		Some(max_texture_side),
	)
}

/// Creates the (initially hidden) output window on `desc`'s monitor.
fn create_output_window(event_loop: &ActiveEventLoop, desc: &MonitorDescriptor) -> Result<Arc<Window>, String> {
	// Cached geometry: the live handle may already be stale (and winit panics on Windows
	// when querying a stale `HMONITOR`).
	let (x, y) = desc.position;
	let (width, height) = (desc.summary.width, desc.summary.height);
	let attributes = WindowAttributes::default()
		.with_title(OUTPUT_WINDOW_TITLE)
		.with_decorations(false)
		.with_resizable(false)
		.with_active(false)
		.with_visible(false)
		.with_position(PhysicalPosition::new(x, y))
		.with_inner_size(PhysicalSize::new(width.max(1), height.max(1)));
	event_loop
		.create_window(attributes)
		.map(Arc::new)
		.map_err(|error| format!("failed to create output window: {error}"))
}

/// Publishes every layer's desired state to the streamer and polls its current frame.
fn sync_layers(
	streamer: Option<&LayerStreamer>,
	state: &AppState,
	library: &ClipLibrary,
	bound: &mut [Option<ClipId>; LAYER_COUNT],
	force_restart: &mut [bool; LAYER_COUNT],
) -> [Option<Arc<DecodedFrame>>; LAYER_COUNT] {
	let mut frames: [Option<Arc<DecodedFrame>>; LAYER_COUNT] = Default::default();
	let Some(streamer) = streamer else {
		return frames;
	};
	for (((layer, frame), bound_clip), restart) in LayerIndex::ALL
		.into_iter()
		.zip(frames.iter_mut())
		.zip(bound.iter_mut())
		.zip(force_restart.iter_mut())
	{
		let layer_state = state.layer(layer);
		let streamable = layer_state
			.active_clip
			.filter(|clip_id| !library.is_loading(*clip_id))
			.filter(|clip_id| state.clip_by_id(*clip_id).is_some_and(ClipInfo::is_ready));
		let Some(clip_id) = streamable else {
			if bound_clip.is_some() {
				streamer.clear_layer(layer);
				*bound_clip = None;
			}
			continue;
		};
		if std::mem::take(restart) && bound_clip.is_some() {
			streamer.clear_layer(layer);
			*bound_clip = None;
		}
		let desired = layer_state.current_frame_index();
		let accepted = streamer.request_layer_state(
			layer,
			clip_id,
			desired,
			layer_state.speed,
			layer_state.playback_mode,
			layer_state.playback_state.is_playing(),
		);
		*bound_clip = accepted.then_some(clip_id);
		if accepted {
			*frame = streamer.poll_frame(layer, clip_id, desired);
		}
	}
	frames
}

/// Builds the compositor input of layer `index`; stopped or empty layers draw nothing.
fn layer_render_input<'a>(state: &AppState, index: usize, frame: Option<&'a DecodedFrame>) -> LayerRenderInput<'a> {
	let Some(layer) = state.layers.get(index) else {
		return LayerRenderInput::EMPTY;
	};
	let Some(clip_id) = layer.active_clip else {
		return LayerRenderInput::EMPTY;
	};
	if layer.playback_state == PlaybackState::Stopped {
		return LayerRenderInput::EMPTY;
	}
	LayerRenderInput {
		frame,
		clip_id: Some(clip_id),
		opacity: layer.opacity,
		blend_mode: layer.blend_mode,
		is_opaque_margin: state.clip_by_id(clip_id).is_none_or(ClipInfo::is_opaque_margin),
	}
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;
	use crate::config::PersistedLayerState;
	use crate::model::{BlendMode, PlaybackMode};

	fn loaded_info(path: &str, frame_count: u32, fps: f32) -> ClipInfo {
		ClipInfo {
			id: ClipId(999),
			path: PathBuf::from(path),
			name: "clip.mov".to_owned(),
			kind: ClipKind::Hap,
			width: 1920,
			height: 1080,
			fps,
			frame_count,
			has_alpha: false,
			status: ClipStatus::Ready,
			thumbnail: None,
		}
	}

	fn monitor(id: &str, primary: bool) -> MonitorSummary {
		MonitorSummary {
			id: id.to_owned(),
			name: id.to_owned(),
			width: 1920,
			height: 1080,
			scale_factor: 1.0,
			is_primary: primary,
		}
	}

	fn temp_dir(name: &str) -> PathBuf {
		let unique = std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.map(|elapsed| elapsed.as_nanos())
			.unwrap_or_default();
		let dir = std::env::temp_dir().join(format!("mur-app-test-{}-{unique}-{name}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		dir
	}

	#[test]
	fn enqueue_deduplicates_and_inserts_loading_placeholder() {
		let mut state = AppState::default();
		let mut library = ClipLibrary::new();
		let first = library.enqueue(&mut state, Path::new("/media/a.mov")).unwrap();
		assert!(library.enqueue(&mut state, Path::new("/media/a.mov")).is_none());
		assert_eq!(state.clips.len(), 1);
		assert!(library.is_loading(first.clip_id));
		let placeholder = state.clip_by_id(first.clip_id).unwrap();
		assert_eq!(placeholder.frame_count, 0);
		assert!(placeholder.thumbnail.is_none());
		assert_eq!(placeholder.name, "a.mov");
		assert_eq!(library.loading_count(), 1);
	}

	#[test]
	fn restored_selection_stays_selection_only_and_survives_the_active_binding() {
		let mut persisted = PersistedState::default();
		// Layer 1: selected only. Layer 2: active A with B selected.
		persisted.layers[0].selected_clip_path = Some(PathBuf::from("/media/b.mov"));
		persisted.layers[1].clip_path = Some(PathBuf::from("/media/a.mov"));
		persisted.layers[1].selected_clip_path = Some(PathBuf::from("/media/b.mov"));

		let mut state = AppState::default();
		let mut library = ClipLibrary::new();
		let requests = apply_persisted_state(&mut state, &mut library, &persisted);
		assert_eq!(requests.len(), 2);
		let a = requests
			.iter()
			.find(|r| r.path == Path::new("/media/a.mov"))
			.unwrap()
			.clone();
		let b = requests
			.iter()
			.find(|r| r.path == Path::new("/media/b.mov"))
			.unwrap()
			.clone();
		assert_eq!(library.pending_bindings().len(), 1, "only the active clip is bound");
		assert_eq!(state.layers[0].selected_clip, Some(b.clip_id));
		assert_eq!(state.layers[0].active_clip, None);

		for (request, frames) in [(&b, 10), (&a, 20)] {
			let path = request.path.to_string_lossy().into_owned();
			library.complete(
				&mut state,
				request.clip_id,
				&request.path,
				Ok(loaded_info(&path, frames, 30.0)),
			);
		}
		assert_eq!(
			state.layers[0].active_clip, None,
			"selection-only clip is never activated"
		);
		assert!(!state.toggle_pause_layer(LayerIndex::ALL[0]));
		assert_eq!(state.layers[1].active_clip, Some(a.clip_id));
		assert_eq!(state.layers[1].selected_clip, Some(b.clip_id));

		let snapshot = capture_persisted(&state, &library);
		assert_eq!(snapshot.layers[0].clip_path, None);
		assert_eq!(
			snapshot.layers[0].selected_clip_path,
			Some(PathBuf::from("/media/b.mov"))
		);
	}

	#[test]
	fn persisted_layer_binding_is_fulfilled_and_hydrated_on_load() {
		let mut persisted = PersistedState {
			clip_paths: vec![PathBuf::from("/media/a.mov")],
			..PersistedState::default()
		};
		persisted.layers[1] = PersistedLayerState {
			clip_path: Some(PathBuf::from("/media/a.mov")),
			selected_clip_path: None,
			playback_mode: PlaybackMode::OneShot,
			speed: 2.0,
			opacity: 0.4,
			blend_mode: BlendMode::Screen,
			playhead_frame: 42.0,
		};
		persisted.effects.master_dimmer = 0.5;

		let mut state = AppState::default();
		let mut library = ClipLibrary::new();
		let requests = apply_persisted_state(&mut state, &mut library, &persisted);
		assert_eq!(requests.len(), 1, "layer clip path is deduplicated against clip_paths");
		assert_eq!(state.effects.master_dimmer, 0.5);
		assert_eq!(state.layers[1].opacity, 0.4);
		assert_eq!(state.layers[1].active_clip, None);
		assert_eq!(library.pending_bindings().len(), 1);

		let snapshot = capture_persisted(&state, &library);
		assert_eq!(snapshot.layers[1].clip_path, Some(PathBuf::from("/media/a.mov")));

		let request = &requests[0];
		let outcome = library.complete(
			&mut state,
			request.clip_id,
			&request.path,
			Ok(loaded_info("/media/a.mov", 120, 30.0)),
		);
		assert!(outcome.accepted);
		assert!(outcome.bound_layers[1]);
		assert!(outcome.layer_needs_restart(LayerIndex::ALL[1]));
		let layer = &state.layers[1];
		assert_eq!(layer.active_clip, Some(request.clip_id));
		assert_eq!(layer.selected_clip, Some(request.clip_id));
		assert_eq!(layer.frame_count, 120);
		assert_eq!(layer.clip_fps, 30.0);
		assert_eq!(layer.playhead_frame, 42.0);
		assert_eq!(layer.playback_state, PlaybackState::Stopped);
		assert_eq!(layer.playback_mode, PlaybackMode::OneShot);
		assert_eq!(layer.blend_mode, BlendMode::Screen);
		assert!(library.pending_bindings().is_empty());
		assert!(!library.is_loading(request.clip_id));
		assert_eq!(state.clip_by_id(request.clip_id).unwrap().frame_count, 120);
	}

	#[test]
	fn layer_triggered_while_loading_is_hydrated() {
		let mut state = AppState::default();
		let mut library = ClipLibrary::new();
		let request = library.enqueue(&mut state, Path::new("/media/b.mov")).unwrap();
		let outcome = apply_session_command(
			&mut state,
			&mut library,
			&UiCommand::TriggerClipOnLayer {
				layer: LayerIndex::ALL[0],
				clip_id: request.clip_id,
			},
		);
		assert!(outcome.was_retriggered(LayerIndex::ALL[0]));
		assert_eq!(state.layers[0].frame_count, 0);

		let outcome = library.complete(
			&mut state,
			request.clip_id,
			&request.path,
			Ok(loaded_info("/media/b.mov", 300, 0.0)),
		);
		assert!(outcome.hydrated_layers[0]);
		assert_eq!(state.layers[0].frame_count, 300);
		assert_eq!(state.layers[0].clip_fps, DEFAULT_CLIP_FPS, "invalid fps falls back");
		assert_eq!(state.layers[0].playback_state, PlaybackState::Playing);
	}

	#[test]
	fn failed_load_marks_only_its_slot_and_waiting_layers() {
		let mut state = AppState::default();
		let mut library = ClipLibrary::new();
		let good = library.enqueue(&mut state, Path::new("/media/good.mov")).unwrap();
		let bad = library.enqueue(&mut state, Path::new("/media/bad.mov")).unwrap();
		library.add_binding(PendingBinding {
			layer: LayerIndex::ALL[2],
			path: PathBuf::from("/media/bad.mov"),
			playhead_frame: 0.0,
			keep_selection: false,
		});
		state.trigger_clip_on_layer(LayerIndex::ALL[3], bad.clip_id);

		let outcome = library.complete(&mut state, bad.clip_id, &bad.path, Err("corrupt atom".to_owned()));
		assert!(outcome.accepted);
		assert_eq!(outcome.error.as_deref(), Some("corrupt atom"));
		assert!(outcome.failed_layers[2]);
		assert!(outcome.failed_layers[3]);
		assert_eq!(
			state.clip_by_id(bad.clip_id).unwrap().status,
			ClipStatus::Error("corrupt atom".to_owned())
		);
		assert_eq!(state.layers[3].active_clip, None);
		assert_eq!(state.layers[3].playback_state, PlaybackState::Stopped);
		assert_eq!(state.layers[2].last_error.as_deref(), Some("corrupt atom"));
		assert!(state.clip_by_id(good.clip_id).unwrap().is_ready());
		assert!(library.is_loading(good.clip_id));

		// A failed clip can be retried in the same slot.
		let retry = library.enqueue(&mut state, Path::new("/media/bad.mov")).unwrap();
		assert_eq!(retry.clip_id, bad.clip_id);
		assert_eq!(state.clips.len(), 2);
	}

	#[test]
	fn removed_clip_discards_late_load_result() {
		let mut state = AppState::default();
		let mut library = ClipLibrary::new();
		let request = library.enqueue(&mut state, Path::new("/media/c.mov")).unwrap();
		library.add_binding(PendingBinding {
			layer: LayerIndex::ALL[0],
			path: PathBuf::from("/media/c.mov"),
			playhead_frame: 0.0,
			keep_selection: false,
		});
		let outcome = apply_session_command(&mut state, &mut library, &UiCommand::RemoveClip(request.clip_id));
		assert_eq!(outcome.remove_clip, Some(request.clip_id));
		assert!(!library.is_loading(request.clip_id));
		assert!(library.pending_bindings().is_empty());

		let outcome = library.complete(
			&mut state,
			request.clip_id,
			&request.path,
			Ok(loaded_info("/media/c.mov", 10, 60.0)),
		);
		assert!(!outcome.accepted);
		assert!(state.clips.is_empty());
	}

	#[test]
	fn stale_load_result_keeps_the_binding_of_a_newer_load() {
		let mut state = AppState::default();
		let mut library = ClipLibrary::new();
		let path = Path::new("/media/x.mov");
		let old = library.enqueue(&mut state, path).unwrap();
		apply_session_command(&mut state, &mut library, &UiCommand::RemoveClip(old.clip_id));
		assert!(state.clip_by_id(old.clip_id).is_none());

		// A project binds the same file to layer 1; it is re-queued under a new id.
		let new = library.enqueue(&mut state, path).unwrap();
		assert_ne!(new.clip_id, old.clip_id);
		library.add_binding(PendingBinding {
			layer: LayerIndex::ALL[0],
			path: path.to_path_buf(),
			playhead_frame: 0.0,
			keep_selection: false,
		});

		// The stale result of the first load arrives first.
		let outcome = library.complete(&mut state, old.clip_id, path, Ok(loaded_info("/media/x.mov", 10, 60.0)));
		assert!(!outcome.accepted);
		assert_eq!(library.pending_bindings().len(), 1, "newer binding survives");

		let outcome = library.complete(&mut state, new.clip_id, path, Ok(loaded_info("/media/x.mov", 10, 60.0)));
		assert!(outcome.accepted);
		assert!(outcome.bound_layers[0]);
		assert_eq!(state.layers[0].active_clip, Some(new.clip_id));
	}

	#[test]
	fn binding_does_not_override_operator_choice() {
		let mut state = AppState::default();
		let mut library = ClipLibrary::new();
		let other = library.enqueue(&mut state, Path::new("/media/other.png")).unwrap();
		library.complete(
			&mut state,
			other.clip_id,
			&other.path,
			Ok(loaded_info("/media/other.png", 1, 60.0)),
		);
		let restored = library.enqueue(&mut state, Path::new("/media/restored.mov")).unwrap();
		library.add_binding(PendingBinding {
			layer: LayerIndex::ALL[0],
			path: PathBuf::from("/media/restored.mov"),
			playhead_frame: 3.0,
			keep_selection: false,
		});
		state.trigger_clip_on_layer(LayerIndex::ALL[0], other.clip_id);

		let outcome = library.complete(
			&mut state,
			restored.clip_id,
			&restored.path,
			Ok(loaded_info("/media/restored.mov", 50, 60.0)),
		);
		assert!(!outcome.bound_layers[0]);
		assert_eq!(state.layers[0].active_clip, Some(other.clip_id));
		assert!(library.pending_bindings().is_empty());
	}

	#[test]
	fn project_applies_settings_and_binds_loaded_or_pending_clips() {
		let mut state = AppState::default();
		let mut library = ClipLibrary::new();
		let loaded = library.enqueue(&mut state, Path::new("/show/loaded.mov")).unwrap();
		library.complete(
			&mut state,
			loaded.clip_id,
			&loaded.path,
			Ok(loaded_info("/show/loaded.mov", 90, 25.0)),
		);

		let project = ProjectConfig {
			clips: vec![PathBuf::from("/show/a.mov"), PathBuf::from("/show/loaded.mov")],
			preferred_midi_port: Some("Launch Control XL 3".to_owned()),
			output: Some(OutputDisplaySelection {
				monitor_name: Some("Projector".to_owned()),
				enabled: true,
			}),
			layers: vec![
				PersistedLayerState {
					clip_path: Some(PathBuf::from("/show/loaded.mov")),
					opacity: 0.25,
					..PersistedLayerState::default()
				},
				PersistedLayerState {
					clip_path: Some(PathBuf::from("/show/b.mov")),
					speed: -1.0,
					blend_mode: BlendMode::Add,
					..PersistedLayerState::default()
				},
			],
			..ProjectConfig::default()
		};
		let clip_paths = project_clip_paths(&project);
		let outcome = apply_project(&mut state, &mut library, &project, &clip_paths);

		assert_eq!(
			outcome.requests.len(),
			2,
			"a.mov and b.mov are new; loaded.mov is reused"
		);
		assert!(outcome.needs_output_sync);
		assert_eq!(outcome.midi_port_changed, Some(Some("Launch Control XL 3".to_owned())));
		assert!(outcome.bound_layers[0]);
		assert_eq!(state.layers[0].active_clip, Some(loaded.clip_id));
		assert_eq!(state.layers[0].frame_count, 90);
		assert_eq!(state.layers[0].opacity, 0.25);
		assert_eq!(state.layers[1].speed, -1.0);
		assert_eq!(state.layers[1].blend_mode, BlendMode::Add);
		assert_eq!(
			library.pending_binding(LayerIndex::ALL[1]).unwrap().path,
			Path::new("/show/b.mov")
		);
		assert_eq!(state.output.monitor_name.as_deref(), Some("Projector"));
	}

	#[test]
	fn exit_guard_request_cancel_confirm() {
		let mut state = AppState::default();
		let library_cmd = |state: &mut AppState, cmd: UiCommand| {
			let mut library = ClipLibrary::new();
			apply_session_command(state, &mut library, &cmd)
		};

		let requested = request_exit(&mut state);
		assert!(state.exit_confirm_open);
		assert_eq!(ExitAction::from_outcome(&requested), ExitAction::ShowGuard);

		// A second close request (e.g. Alt+F4 twice) keeps the guard open, never quits.
		let again = request_exit(&mut state);
		assert_eq!(ExitAction::from_outcome(&again), ExitAction::ShowGuard);

		let cancelled = library_cmd(&mut state, UiCommand::CancelExit);
		assert!(!state.exit_confirm_open);
		assert_eq!(ExitAction::from_outcome(&cancelled), ExitAction::None);

		request_exit(&mut state);
		let confirmed = library_cmd(&mut state, UiCommand::ConfirmExit);
		assert!(!state.exit_confirm_open);
		assert_eq!(ExitAction::from_outcome(&confirmed), ExitAction::Quit);
	}

	#[test]
	fn native_quit_request_opens_guard_then_confirms() {
		let mut state = AppState::default();

		let first = handle_native_quit_request(&mut state);
		assert!(state.exit_confirm_open);
		assert_eq!(ExitAction::from_outcome(&first), ExitAction::ShowGuard);

		let second = handle_native_quit_request(&mut state);
		assert!(!state.exit_confirm_open);
		assert_eq!(ExitAction::from_outcome(&second), ExitAction::Quit);
	}

	#[test]
	fn command_outcomes_cancel_pending_bindings_of_touched_layers() {
		let mut state = AppState::default();
		let mut library = ClipLibrary::new();
		library.add_binding(PendingBinding {
			layer: LayerIndex::ALL[1],
			path: PathBuf::from("/media/x.mov"),
			playhead_frame: 0.0,
			keep_selection: false,
		});
		let outcome = apply_session_command(&mut state, &mut library, &UiCommand::ClearLayer(LayerIndex::ALL[1]));
		assert!(outcome.was_cleared(LayerIndex::ALL[1]));
		assert!(library.pending_bindings().is_empty());

		let outcome = apply_session_command(&mut state, &mut library, &UiCommand::ToggleOutputEnabled);
		assert!(outcome.needs_output_sync);
		assert_eq!(ExitAction::from_outcome(&outcome), ExitAction::None);
		let outcome = apply_session_command(&mut state, &mut library, &UiCommand::OpenAddFilesDialog);
		assert!(outcome.open_files);
	}

	#[test]
	fn autosave_keeps_custom_debounce_across_saves() {
		let debounce = Duration::from_secs(30);
		let mut saver = StateSaver::new(debounce);
		let start = saver.last_save();
		saver.mark_dirty();
		assert!(!take_due_save(&mut saver, start + Duration::from_secs(29)));
		let first = start + debounce;
		assert!(take_due_save(&mut saver, first));
		assert!(!saver.is_dirty());
		assert_eq!(saver.last_save(), first);

		// The next save again waits the full custom interval, not the default one.
		saver.mark_dirty();
		assert!(!take_due_save(&mut saver, first + Duration::from_secs(29)));
		assert!(saver.is_dirty());
		assert!(take_due_save(&mut saver, first + debounce));
		assert!(!take_due_save(&mut saver, first + debounce + Duration::from_secs(60)));
	}

	#[test]
	fn sanitize_fps_uses_model_bounds() {
		assert_eq!(sanitize_fps(29.97), 29.97);
		assert_eq!(sanitize_fps(MIN_CLIP_FPS), MIN_CLIP_FPS);
		assert_eq!(sanitize_fps(MAX_CLIP_FPS), MAX_CLIP_FPS);
		assert_eq!(sanitize_fps(0.5), DEFAULT_CLIP_FPS);
		assert_eq!(sanitize_fps(5000.0), DEFAULT_CLIP_FPS);
		assert_eq!(sanitize_fps(f32::NAN), DEFAULT_CLIP_FPS);
		assert_eq!(sanitize_fps(-1.0), DEFAULT_CLIP_FPS);
	}

	#[test]
	fn key_repeat_only_for_continuous_commands() {
		assert!(command_allows_key_repeat(&UiCommand::AdjustMasterDimmer(0.05)));
		assert!(command_allows_key_repeat(&UiCommand::AdjustLayerOpacity {
			layer: LayerIndex::ALL[0],
			delta: 0.05
		}));
		assert!(!command_allows_key_repeat(&UiCommand::StepLayerClip {
			layer: LayerIndex::ALL[0],
			delta: 1
		}));
		assert!(!command_allows_key_repeat(&UiCommand::CycleFocusLayer(1)));
		assert!(!command_allows_key_repeat(&UiCommand::TriggerLayer(LayerIndex::ALL[0])));
		assert!(!command_allows_key_repeat(&UiCommand::ToggleBlackout));
		assert!(!command_allows_key_repeat(&UiCommand::RequestExit));
	}

	#[test]
	fn auto_select_prefers_external_and_never_enables_single_display() {
		let mut output = OutputDisplaySelection::default();
		assert!(!auto_select_output(&mut output, &[monitor("Built-in", true)], false));
		assert!(!output.enabled);
		assert!(output.monitor_name.is_none());

		let monitors = [monitor("Built-in", true), monitor("Projector", false)];
		let mut fresh = OutputDisplaySelection::default();
		assert!(auto_select_output(&mut fresh, &monitors, false));
		assert_eq!(fresh.monitor_name.as_deref(), Some("Projector"));
		assert!(fresh.enabled);

		let mut restored = OutputDisplaySelection::default();
		assert!(auto_select_output(&mut restored, &monitors, true));
		assert!(!restored.enabled, "a restored disabled output stays disabled");

		let mut chosen = OutputDisplaySelection {
			monitor_name: Some("Other".to_owned()),
			enabled: true,
		};
		assert!(!auto_select_output(&mut chosen, &monitors, true));
		assert_eq!(chosen.monitor_name.as_deref(), Some("Other"));
	}

	#[test]
	fn output_transitions_notify_loss_and_recovery() {
		let active = OutputRuntimeStatus::Active {
			monitor_name: "Projector".to_owned(),
			width: 1920,
			height: 1080,
		};
		let waiting = OutputRuntimeStatus::WaitingForMonitor {
			expected_name: Some("Projector".to_owned()),
		};
		let lost = output_transition_notice(&active, &waiting).unwrap();
		assert_eq!(lost.0, ToastLevel::Warning);
		let restored = output_transition_notice(&waiting, &active).unwrap();
		assert_eq!(restored.0, ToastLevel::Info);
		assert!(output_transition_notice(&waiting, &waiting).is_none());
		assert!(output_transition_notice(&OutputRuntimeStatus::Disabled, &active).is_none());
		let error = output_transition_notice(&active, &OutputRuntimeStatus::Error("boom".to_owned())).unwrap();
		assert_eq!(error.0, ToastLevel::Error);
	}

	#[test]
	fn ingest_classification() {
		let dir = temp_dir("ingest");
		assert_eq!(classify_ingest_path(&dir), IngestKind::Folder);
		assert_eq!(classify_ingest_path(Path::new("/x/show.TOML")), IngestKind::Project);
		assert_eq!(classify_ingest_path(Path::new("/x/show.json")), IngestKind::Project);
		assert_eq!(classify_ingest_path(Path::new("/x/clip.MOV")), IngestKind::Media);
		assert_eq!(classify_ingest_path(Path::new("/x/clip.jpeg")), IngestKind::Media);
		assert_eq!(classify_ingest_path(Path::new("/x/clip.mp4")), IngestKind::Unsupported);
		std::fs::remove_dir_all(&dir).ok();
	}

	#[test]
	fn layer_render_input_hides_stopped_layers() {
		let mut state = AppState::default();
		let mut library = ClipLibrary::new();
		let request = library.enqueue(&mut state, Path::new("/media/alpha.mov")).unwrap();
		let mut info = loaded_info("/media/alpha.mov", 10, 60.0);
		info.kind = ClipKind::HapAlpha;
		info.has_alpha = true;
		library.complete(&mut state, request.clip_id, &request.path, Ok(info));
		bind_layer_to_clip(&mut state, LayerIndex::ALL[0], request.clip_id, 0.0);
		assert!(layer_render_input(&state, 0, None).clip_id.is_none());

		state.toggle_pause_layer(LayerIndex::ALL[0]);
		let input = layer_render_input(&state, 0, None);
		assert_eq!(input.clip_id, Some(request.clip_id));
		assert!(!input.is_opaque_margin);
		assert!(layer_render_input(&state, 9, None).clip_id.is_none());
	}

	#[test]
	fn notify_records_toast_and_log() {
		let mut state = AppState::default();
		notify(&mut state, ToastLevel::Error, "Title", "Message");
		assert_eq!(state.toasts.toasts().len(), 1);
		assert_eq!(state.toasts.logs().len(), 1);
		assert_eq!(state.toasts.logs()[0].source, "Title");
	}

	#[test]
	fn clip_loader_reports_errors_without_panicking() {
		let (tx, rx) = crossbeam_channel::unbounded();
		let mut loader = ClipLoader::spawn(Arc::new(move |clip_id, path, result| {
			tx.send((clip_id, path, result.map(|(info, _)| info))).is_ok()
		}));
		let dir = temp_dir("loader");
		let corrupt = dir.join("corrupt.mov");
		std::fs::write(&corrupt, b"definitely not a movie").unwrap();
		loader
			.request(LoadRequest {
				clip_id: ClipId(7),
				path: corrupt.clone(),
			})
			.unwrap();
		let (clip_id, path, result) = rx.recv_timeout(Duration::from_secs(10)).unwrap();
		assert_eq!(clip_id, ClipId(7));
		assert_eq!(path, corrupt);
		assert!(result.is_err());
		std::fs::remove_dir_all(&dir).ok();
	}

	#[test]
	fn clip_loader_loads_fixture_movie() {
		let (tx, rx) = crossbeam_channel::unbounded();
		let mut loader = ClipLoader::spawn(Arc::new(move |clip_id, _path, result| {
			tx.send((clip_id, result.map(|(info, asset)| (info, asset.frame_count()))))
				.is_ok()
		}));
		let dir = temp_dir("fixture");
		let path = dir.join("fixture.mov");
		media::test_fixtures::HapMovFixture::new(ClipKind::Hap, 16, 8, 5)
			.write_to(&path)
			.unwrap();
		loader
			.request(LoadRequest {
				clip_id: ClipId(3),
				path: path.clone(),
			})
			.unwrap();
		let (clip_id, result) = rx.recv_timeout(Duration::from_secs(10)).unwrap();
		assert_eq!(clip_id, ClipId(3));
		let (info, frames) = result.unwrap();
		assert_eq!(info.frame_count, 5);
		assert_eq!(frames, 5);

		let mut state = AppState::default();
		let mut library = ClipLibrary::new();
		let request = library.enqueue(&mut state, &path).unwrap();
		let outcome = library.complete(&mut state, request.clip_id, &path, Ok(info));
		assert!(outcome.accepted);
		assert!(state.clip_by_id(request.clip_id).unwrap().is_ready());
		std::fs::remove_dir_all(&dir).ok();
	}
}
