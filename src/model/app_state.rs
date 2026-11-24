//! Aggregated runtime state shared by the control UI, renderer, and MIDI loop.

use serde::{Deserialize, Serialize};

use crate::error::{ToastLevel, ToastNotification, ToastQueue};

use super::blend::BlendMode;
use super::clip::{ClipId, ClipInfo};
use super::effect::MasterEffects;
use super::layer::{LayerIndex, LayerState, PlaybackMode, PlaybackState};

/// Output display chosen in the control UI.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputDisplaySelection {
	/// Name of the target monitor, `None` until the operator picks one.
	pub monitor_name: Option<String>,
	/// Whether the output window should be sending frames.
	pub enabled: bool,
}

/// Live status of the external output window.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputRuntimeStatus {
	/// Output is switched off by the operator.
	#[default]
	Disabled,
	/// Output is presenting frames on the given monitor.
	Active {
		/// Monitor currently receiving frames.
		monitor_name: String,
		/// Presented width in physical pixels.
		width: u32,
		/// Presented height in physical pixels.
		height: u32,
	},
	/// The selected monitor is missing; output resumes when it reappears.
	WaitingForMonitor {
		/// Monitor name the output is waiting for.
		expected_name: Option<String>,
	},
	/// Output could not be created or was lost.
	Error(String),
}

impl OutputRuntimeStatus {
	/// Returns `true` while frames are being presented.
	pub fn is_active(&self) -> bool {
		matches!(self, Self::Active { .. })
	}
}

/// Connection state of the MIDI input subsystem.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MidiConnectionStatus {
	/// No port is open and scanning has not started.
	#[default]
	Disconnected,
	/// Looking for the configured controller.
	Scanning,
	/// A port is open and delivering messages.
	Connected {
		/// Name of the open input port.
		port_name: String,
	},
	/// The port was lost and a background reconnect is in progress.
	Reconnecting {
		/// Port that was open before the disconnect, when known.
		last_port: Option<String>,
		/// Last reconnect failure, when known.
		error: Option<String>,
	},
}

impl MidiConnectionStatus {
	/// Returns `true` while a port is open.
	pub fn is_connected(&self) -> bool {
		matches!(self, Self::Connected { .. })
	}
}

/// Render loop performance counters shown in the status bar.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PerformanceStats {
	/// Measured frames per second.
	pub fps: f32,
	/// Average frame time in milliseconds.
	pub frame_time_ms: f32,
	/// Frames skipped because the deadline was missed.
	pub dropped_frames: u64,
	/// Frames presented since launch.
	pub total_frames: u64,
}

/// One display that the operator can send output to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonitorSummary {
	/// Platform specific stable identifier.
	pub id: String,
	/// Human readable monitor name.
	pub name: String,
	/// Monitor width in physical pixels.
	pub width: u32,
	/// Monitor height in physical pixels.
	pub height: u32,
	/// HiDPI scale factor reported by the OS.
	pub scale_factor: f64,
	/// Whether this is the OS primary display.
	pub is_primary: bool,
}

/// Full runtime state of one MUR session.
#[derive(Debug, Clone)]
pub struct AppState {
	/// The four performance layers.
	pub layers: [LayerState; 4],
	/// Layer receiving keyboard focus and single-target actions.
	pub focused_layer: LayerIndex,
	/// Every clip loaded in this session, in grid order.
	pub clips: Vec<ClipInfo>,
	/// Master effect parameters.
	pub effects: MasterEffects,
	/// Requested output display.
	pub output: OutputDisplaySelection,
	/// Current output window status.
	pub output_status: OutputRuntimeStatus,
	/// Displays discovered on this machine.
	pub available_monitors: Vec<MonitorSummary>,
	/// MIDI connection state.
	pub midi_status: MidiConnectionStatus,
	/// MIDI input ports discovered so far.
	pub midi_ports: Vec<String>,
	/// MIDI port the operator selected, if any.
	pub selected_midi_port: Option<String>,
	/// Render loop statistics.
	pub perf: PerformanceStats,
	/// Non-blocking notifications and rolling log history.
	pub toasts: ToastQueue,
	/// Whether the quit-confirmation dialog is open.
	pub exit_confirm_open: bool,
	/// Whether the keyboard shortcut overlay is open.
	pub show_shortcuts: bool,
	/// Whether the log panel is open.
	pub show_logs: bool,
	/// Seconds elapsed on the application clock, driven by [`AppState::advance_layers`].
	pub elapsed_secs: f64,
	/// Whether state changed since the last persist.
	pub dirty: bool,
	/// Next identifier handed out by [`AppState::allocate_clip_id`].
	pub next_clip_id: u64,
}

impl Default for AppState {
	fn default() -> Self {
		Self {
			layers: std::array::from_fn(|_| LayerState::default()),
			focused_layer: LayerIndex::default(),
			clips: Vec::new(),
			effects: MasterEffects::default(),
			output: OutputDisplaySelection::default(),
			output_status: OutputRuntimeStatus::default(),
			available_monitors: Vec::new(),
			midi_status: MidiConnectionStatus::default(),
			midi_ports: Vec::new(),
			selected_midi_port: None,
			perf: PerformanceStats::default(),
			toasts: ToastQueue::default(),
			exit_confirm_open: false,
			show_shortcuts: false,
			show_logs: false,
			elapsed_secs: 0.0,
			dirty: false,
			next_clip_id: 1,
		}
	}
}

impl AppState {
	/// Returns the layer at `index`, or a neutral layer for an out-of-range index.
	pub fn layer(&self, index: LayerIndex) -> &LayerState {
		self.layers.get(index.index()).unwrap_or(&FALLBACK_LAYER)
	}

	/// Returns the mutable layer at `index`, or `None` for an out-of-range index.
	pub fn layer_mut(&mut self, index: LayerIndex) -> Option<&mut LayerState> {
		self.layers.get_mut(index.index())
	}

	/// Looks a clip up by its session identifier.
	pub fn clip_by_id(&self, id: ClipId) -> Option<&ClipInfo> {
		self.clips.iter().find(|clip| clip.id == id)
	}

	/// Looks a clip up by its position in the clip grid.
	pub fn clip_by_index(&self, index: usize) -> Option<&ClipInfo> {
		self.clips.get(index)
	}

	/// Reserves a fresh clip identifier and advances the counter.
	pub fn allocate_clip_id(&mut self) -> ClipId {
		let id = ClipId(self.next_clip_id);
		self.next_clip_id = self.next_clip_id.saturating_add(1);
		id
	}

	/// Inserts `clip`, replacing an existing clip with the same identifier.
	pub fn upsert_clip(&mut self, clip: ClipInfo) {
		if let Some(existing) = self.clips.iter_mut().find(|existing| existing.id == clip.id) {
			*existing = clip;
		} else {
			self.clips.push(clip);
		}
		self.dirty = true;
	}

	/// Triggers `clip_id` on `layer` and returns `true` when playback started.
	///
	/// Missing or failed clips are reported on the layer and through a toast
	/// instead of being triggered, so a broken file never silences the show.
	pub fn trigger_clip_on_layer(&mut self, layer: LayerIndex, clip_id: ClipId) -> bool {
		let Some((name, frame_count, fps, ready)) = self
			.clip_by_id(clip_id)
			.map(|clip| (clip.name.clone(), clip.frame_count, clip.fps, clip.is_ready()))
		else {
			self.report_layer_error(layer, format!("clip {:?} is no longer loaded", clip_id));
			return false;
		};

		if !ready {
			self.report_layer_error(layer, format!("clip `{name}` failed to load"));
			return false;
		}

		let Some(state) = self.layer_mut(layer) else {
			return false;
		};
		state.trigger(clip_id, frame_count, fps);
		self.dirty = true;
		true
	}

	/// Selects the clip at `clip_index` for triggering on `layer`.
	pub fn select_clip_by_index(&mut self, layer: LayerIndex, clip_index: usize) -> Option<ClipId> {
		let clip_id = self.clip_by_index(clip_index)?.id;
		let state = self.layer_mut(layer)?;
		state.selected_clip = Some(clip_id);
		self.dirty = true;
		Some(clip_id)
	}

	/// Restarts the layer's active clip from its first frame.
	pub fn retrigger_layer(&mut self, layer: LayerIndex) -> bool {
		let Some(state) = self.layer_mut(layer) else {
			return false;
		};
		if state.active_clip.is_none() {
			return false;
		}
		state.retrigger();
		self.dirty = true;
		true
	}

	/// Toggles pause/resume on `layer`; a stopped layer resumes instead.
	///
	/// A layer without an active clip stays stopped and reports `false`, even when
	/// a clip is merely selected: only `Enter` / a trigger may start playback, so a
	/// stray pause keystroke cannot silently bring an empty layer to life.
	pub fn toggle_pause_layer(&mut self, layer: LayerIndex) -> bool {
		let Some(state) = self.layer_mut(layer) else {
			return false;
		};
		match state.playback_state {
			PlaybackState::Playing => state.playback_state = PlaybackState::Paused,
			PlaybackState::Paused => state.playback_state = PlaybackState::Playing,
			PlaybackState::Stopped => {
				if state.active_clip.is_none() {
					return false;
				}
				state.playback_state = PlaybackState::Playing;
			}
		}
		self.dirty = true;
		true
	}

	/// Resets `layer` to its default, empty state.
	pub fn clear_layer(&mut self, layer: LayerIndex) -> bool {
		let Some(state) = self.layer_mut(layer) else {
			return false;
		};
		*state = LayerState::default();
		self.dirty = true;
		true
	}

	/// Sets the layer opacity, clamped to `0.0..=1.0`.
	pub fn set_layer_opacity(&mut self, layer: LayerIndex, opacity: f32) -> bool {
		let Some(state) = self.layer_mut(layer) else {
			return false;
		};
		state.set_opacity(opacity);
		self.dirty = true;
		true
	}

	/// Sets the layer speed, clamped to `-2.0..=4.0`.
	pub fn set_layer_speed(&mut self, layer: LayerIndex, speed: f32) -> bool {
		let Some(state) = self.layer_mut(layer) else {
			return false;
		};
		state.set_speed(speed);
		self.dirty = true;
		true
	}

	/// Steps the layer's blend mode forward or backward through the cycle.
	pub fn cycle_blend_mode(&mut self, layer: LayerIndex, forward: bool) -> Option<BlendMode> {
		let blend_mode = {
			let state = self.layer_mut(layer)?;
			state.blend_mode = if forward {
				state.blend_mode.next()
			} else {
				state.blend_mode.prev()
			};
			state.blend_mode
		};
		self.dirty = true;
		Some(blend_mode)
	}

	/// Switches the layer between loop and one-shot playback.
	pub fn cycle_playback_mode(&mut self, layer: LayerIndex) -> Option<PlaybackMode> {
		let playback_mode = {
			let state = self.layer_mut(layer)?;
			state.playback_mode = state.playback_mode.toggled();
			state.playback_mode
		};
		self.dirty = true;
		Some(playback_mode)
	}

	/// Moves keyboard focus by `delta` layers, wrapping around.
	pub fn cycle_focused_layer(&mut self, delta: i32) {
		self.focused_layer = self.focused_layer.offset(delta);
		self.dirty = true;
	}

	/// Enqueues a toast using the level's default lifetime.
	pub fn push_toast(
		&mut self,
		level: ToastLevel,
		title: impl Into<String>,
		message: impl Into<String>,
		now_secs: f64,
	) -> ToastNotification {
		self.toasts.notify(level, title, message, now_secs)
	}

	/// Advances every layer's playhead and the application clock by `dt_secs`.
	pub fn advance_layers(&mut self, dt_secs: f32) {
		if dt_secs.is_finite() && dt_secs > 0.0 {
			self.elapsed_secs += f64::from(dt_secs);
		}
		for layer in &mut self.layers {
			layer.advance_playhead(dt_secs);
		}
	}

	/// Records a layer failure on the layer and as an error toast.
	fn report_layer_error(&mut self, layer: LayerIndex, message: String) {
		if let Some(state) = self.layer_mut(layer) {
			state.last_error = Some(message.clone());
		}
		let title = format!("Layer {layer} error");
		self.toasts.notify(ToastLevel::Error, title, message, self.elapsed_secs);
	}
}

/// Neutral layer returned by [`AppState::layer`] for out-of-range indices.
///
/// [`LayerIndex`] cannot address a missing layer, so this only guards against
/// future refactors; it keeps the accessor panic-free either way.
static FALLBACK_LAYER: LayerState = LayerState {
	selected_clip: None,
	active_clip: None,
	playback_state: PlaybackState::Stopped,
	playback_mode: PlaybackMode::Loop,
	speed: 1.0,
	opacity: 1.0,
	blend_mode: BlendMode::Normal,
	playhead_frame: 0.0,
	frame_count: 0,
	clip_fps: 60.0,
	last_error: None,
};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;
	use crate::model::clip::{ClipInfo, ClipKind, ClipStatus};
	use std::path::PathBuf;

	fn clip(id: u64, status: ClipStatus) -> ClipInfo {
		ClipInfo {
			id: ClipId(id),
			path: PathBuf::from(format!("/clips/{id}.mov")),
			name: format!("{id}.mov"),
			kind: ClipKind::Hap,
			width: 1920,
			height: 1080,
			fps: 30.0,
			frame_count: 30,
			has_alpha: false,
			status,
			thumbnail: None,
		}
	}

	#[test]
	fn defaults_are_safe() {
		let state = AppState::default();
		assert!(!state.dirty);
		assert_eq!(state.layers.len(), 4);
		assert_eq!(state.focused_layer, LayerIndex::ALL[0]);
		assert_eq!(state.output_status, OutputRuntimeStatus::Disabled);
		assert_eq!(state.midi_status, MidiConnectionStatus::Disconnected);
		assert_eq!(state.next_clip_id, 1);
		assert_eq!(state.elapsed_secs, 0.0);
		assert!(!state.exit_confirm_open);
	}

	#[test]
	fn clip_ids_are_allocated_monotonically() {
		let mut state = AppState::default();
		assert_eq!(state.allocate_clip_id(), ClipId(1));
		assert_eq!(state.allocate_clip_id(), ClipId(2));
		assert_eq!(state.next_clip_id, 3);
	}

	#[test]
	fn upsert_replaces_matching_id() {
		let mut state = AppState::default();
		state.upsert_clip(clip(1, ClipStatus::Ready));
		state.upsert_clip(clip(2, ClipStatus::Ready));
		let mut updated = clip(1, ClipStatus::Error("bad".into()));
		updated.name = "renamed".to_string();
		state.upsert_clip(updated);

		assert_eq!(state.clips.len(), 2);
		assert_eq!(
			state.clip_by_id(ClipId(1)).map(|clip| clip.name.as_str()),
			Some("renamed")
		);
		assert_eq!(state.clip_by_index(1).map(|clip| clip.id), Some(ClipId(2)));
		assert!(state.clip_by_index(2).is_none());
	}

	#[test]
	fn trigger_starts_playback_using_clip_metadata() {
		let mut state = AppState::default();
		state.upsert_clip(clip(1, ClipStatus::Ready));
		let target = LayerIndex::ALL[2];
		assert!(state.trigger_clip_on_layer(target, ClipId(1)));

		let layer = state.layer(target);
		assert_eq!(layer.active_clip, Some(ClipId(1)));
		assert_eq!(layer.selected_clip, Some(ClipId(1)));
		assert_eq!(layer.frame_count, 30);
		assert_eq!(layer.clip_fps, 30.0);
		assert_eq!(layer.playback_state, PlaybackState::Playing);
		assert!(state.toasts.toasts().is_empty());
	}

	#[test]
	fn trigger_rejects_missing_or_failed_clips() {
		let mut state = AppState::default();
		state.upsert_clip(clip(1, ClipStatus::Error("corrupt".into())));
		let target = LayerIndex::ALL[0];

		assert!(!state.trigger_clip_on_layer(target, ClipId(1)));
		assert!(!state.trigger_clip_on_layer(target, ClipId(99)));

		let layer = state.layer(target);
		assert!(layer.active_clip.is_none());
		assert!(layer.last_error.is_some());
		assert_eq!(state.toasts.toasts().len(), 2);
	}

	#[test]
	fn select_clip_by_index_updates_selection_only() {
		let mut state = AppState::default();
		state.upsert_clip(clip(1, ClipStatus::Ready));
		state.upsert_clip(clip(2, ClipStatus::Ready));

		assert_eq!(state.select_clip_by_index(LayerIndex::ALL[1], 1), Some(ClipId(2)));
		let layer = state.layer(LayerIndex::ALL[1]);
		assert_eq!(layer.selected_clip, Some(ClipId(2)));
		assert!(layer.active_clip.is_none());
		assert_eq!(state.select_clip_by_index(LayerIndex::ALL[1], 5), None);
	}

	#[test]
	fn retrigger_and_pause_toggle_follow_transport_rules() {
		let mut state = AppState::default();
		state.upsert_clip(clip(1, ClipStatus::Ready));
		let target = LayerIndex::ALL[0];

		assert!(!state.retrigger_layer(target));
		assert!(!state.toggle_pause_layer(target));

		state.trigger_clip_on_layer(target, ClipId(1));
		state.advance_layers(0.5);
		assert!(state.layer(target).playhead_frame > 0.0);

		state.toggle_pause_layer(target);
		assert_eq!(state.layer(target).playback_state, PlaybackState::Paused);
		state.toggle_pause_layer(target);
		assert_eq!(state.layer(target).playback_state, PlaybackState::Playing);

		assert!(state.retrigger_layer(target));
		assert_eq!(state.layer(target).playhead_frame, 0.0);
	}

	#[test]
	fn pause_toggle_ignores_layers_without_an_active_clip() {
		let mut state = AppState::default();
		state.upsert_clip(clip(1, ClipStatus::Ready));
		let target = LayerIndex::ALL[0];

		// Selected but never triggered: pause must not start playback.
		assert_eq!(state.select_clip_by_index(target, 0), Some(ClipId(1)));
		assert_eq!(state.layer(target).selected_clip, Some(ClipId(1)));
		assert!(state.layer(target).active_clip.is_none());
		assert!(!state.toggle_pause_layer(target));
		assert_eq!(state.layer(target).playback_state, PlaybackState::Stopped);

		// Once the clip is actually active, the toggle resumes as before.
		assert!(state.trigger_clip_on_layer(target, ClipId(1)));
		assert!(state.toggle_pause_layer(target));
		assert_eq!(state.layer(target).playback_state, PlaybackState::Paused);
		assert!(state.toggle_pause_layer(target));
		assert_eq!(state.layer(target).playback_state, PlaybackState::Playing);
	}

	#[test]
	fn clear_layer_resets_transport() {
		let mut state = AppState::default();
		state.upsert_clip(clip(1, ClipStatus::Ready));
		let target = LayerIndex::ALL[3];
		state.trigger_clip_on_layer(target, ClipId(1));
		state.set_layer_opacity(target, 0.25);
		state.set_layer_speed(target, 2.0);

		assert!(state.clear_layer(target));
		let layer = state.layer(target);
		assert_eq!(layer.active_clip, None);
		assert_eq!(layer.playback_state, PlaybackState::Stopped);
		assert_eq!(layer.opacity, 1.0);
		assert_eq!(layer.speed, 1.0);
	}

	#[test]
	fn blend_and_playback_modes_cycle() {
		let mut state = AppState::default();
		let target = LayerIndex::ALL[0];
		assert_eq!(state.cycle_blend_mode(target, true), Some(BlendMode::Add));
		assert_eq!(state.cycle_blend_mode(target, false), Some(BlendMode::Normal));
		assert_eq!(state.layer(target).blend_mode, BlendMode::Normal);

		assert_eq!(state.cycle_playback_mode(target), Some(PlaybackMode::OneShot));
		assert_eq!(state.cycle_playback_mode(target), Some(PlaybackMode::Loop));
		state.cycle_focused_layer(-1);
		assert_eq!(state.focused_layer, LayerIndex::ALL[3]);
	}

	#[test]
	fn toasts_are_exposed() {
		let mut state = AppState::default();
		let toast = state.push_toast(ToastLevel::Warning, "title", "message", 100.0);
		assert_eq!(toast.id, 1);
		assert_eq!(state.toasts.toasts().len(), 1);
		assert_eq!(state.toasts.prune_expired(100.5), 0);
		assert_eq!(state.toasts.prune_expired(100.0 + toast.ttl_secs), 1);
	}

	#[test]
	fn out_of_range_layer_access_is_safe() {
		let mut state = AppState::default();
		assert!(state.layer_mut(LayerIndex::ALL[3]).is_some());
		assert_eq!(state.layer(LayerIndex::ALL[3]).speed, 1.0);
		assert_eq!(state.cycle_blend_mode(LayerIndex::ALL[2], true), Some(BlendMode::Add));
	}

	#[test]
	fn advance_layers_moves_all_playing_layers() {
		let mut state = AppState::default();
		for id in [1_u64, 2, 3, 4] {
			state.upsert_clip(clip(id, ClipStatus::Ready));
		}
		for (index, layer) in LayerIndex::ALL.iter().enumerate() {
			assert!(state.trigger_clip_on_layer(*layer, ClipId(index as u64 + 1)));
		}
		state.advance_layers(0.1);
		assert!(state.elapsed_secs > 0.0);
		for layer in &state.layers {
			assert!(layer.playhead_frame > 0.0);
		}

		state.advance_layers(-1.0);
		assert!(state.elapsed_secs > 0.0);
	}
}
