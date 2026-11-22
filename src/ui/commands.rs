//! Command protocol shared by the control UI, the keyboard fallback, and `app.rs`.
//!
//! Every operator gesture in MUR — an egui widget, a mouse click, a keyboard
//! shortcut from [`crate::ui::shortcuts`](super::shortcuts), or the MIDI
//! equivalent decoded by `crate::midi::mapping` — is reduced to exactly one
//! [`UiCommand`] value. `app.rs` then feeds that value to [`apply_ui_command`],
//! which performs the state mutation on [`AppState`] and reports the side effects
//! that cannot be expressed in the state itself (opening a native file dialog,
//! resynchronising the external output window, restarting a media streamer, …).
//!
//! Keeping both halves of that contract here means SPEC §7.2 — "the keyboard and
//! mouse must be able to replace every MIDI control" — is verifiable with plain
//! unit tests, without a window, a GPU, or a MIDI controller.

use crate::model::layer::{LAYER_COUNT, PlaybackState};
use crate::model::{AppState, BlendMode, ClipId, LayerIndex, MidiConnectionStatus, PlaybackMode};

/// One semantic operator action, independent of its input device.
///
/// Variants are intentionally coarse: they describe *what* the operator asked
/// for, never how it was delivered, so the same value can come from a UI button,
/// a keyboard shortcut, or a MIDI message.
#[derive(Debug, Clone, PartialEq)]
pub enum UiCommand {
	// ---------------------------------------------------------------- layers
	/// Moves keyboard focus to a specific layer.
	FocusLayer(LayerIndex),
	/// Moves keyboard focus by `delta` layers, wrapping around the ring.
	CycleFocusLayer(i32),
	/// Fires the layer's selected clip, or retriggers it when already active.
	TriggerLayer(LayerIndex),
	/// Fires a specific clip on a specific layer.
	TriggerClipOnLayer {
		/// Layer that should show the clip.
		layer: LayerIndex,
		/// Clip to bind and play.
		clip_id: ClipId,
	},
	/// Restarts the layer's active clip from its first frame.
	RetriggerLayer(LayerIndex),
	/// Restarts every layer that currently has an active clip.
	RetriggerAllActiveLayers,
	/// Toggles pause/resume on a layer.
	TogglePauseLayer(LayerIndex),
	/// Returns a layer to its default, empty state.
	ClearLayer(LayerIndex),
	/// Sets a layer's opacity, clamped to `0.0..=1.0`.
	SetLayerOpacity {
		/// Layer to change.
		layer: LayerIndex,
		/// Absolute opacity in `0.0..=1.0`.
		opacity: f32,
	},
	/// Adds `delta` to a layer's opacity, clamped to `0.0..=1.0`.
	AdjustLayerOpacity {
		/// Layer to change.
		layer: LayerIndex,
		/// Signed opacity delta.
		delta: f32,
	},
	/// Sets a layer's playback speed, clamped to `-2.0..=4.0`.
	SetLayerSpeed {
		/// Layer to change.
		layer: LayerIndex,
		/// Absolute speed multiplier, `1.0` is real time.
		speed: f32,
	},
	/// Adds `delta` to a layer's playback speed, clamped to `-2.0..=4.0`.
	AdjustLayerSpeed {
		/// Layer to change.
		layer: LayerIndex,
		/// Signed speed delta.
		delta: f32,
	},
	/// Returns every layer to real-time (`1.0x`) playback.
	ResetAllLayerSpeeds,
	/// Sets a layer's blend mode directly.
	SetLayerBlendMode {
		/// Layer to change.
		layer: LayerIndex,
		/// Requested compositing mode.
		blend_mode: BlendMode,
	},
	/// Steps a layer's blend mode forward or backward through the five modes.
	CycleLayerBlendMode {
		/// Layer to change.
		layer: LayerIndex,
		/// `true` for the next mode, `false` for the previous one.
		forward: bool,
	},
	/// Sets a layer's loop/one-shot mode directly.
	SetLayerPlaybackMode {
		/// Layer to change.
		layer: LayerIndex,
		/// Requested playback mode.
		mode: PlaybackMode,
	},
	/// Switches a layer between loop and one-shot playback.
	ToggleLayerPlaybackMode(LayerIndex),
	/// Selects which loaded clip the layer will trigger next.
	SelectLayerClip {
		/// Layer whose selection changes.
		layer: LayerIndex,
		/// Clip to highlight; ignored when it is no longer loaded.
		clip_id: ClipId,
	},
	/// Steps the layer's clip selection by `delta` entries inside the clip grid.
	StepLayerClip {
		/// Layer whose selection moves.
		layer: LayerIndex,
		/// Signed number of entries to step, wrapping around the grid.
		delta: i32,
	},
	/// Turns the layer's trigger button into a clip-grid trigger.
	TriggerSelectedClip(LayerIndex),

	// --------------------------------------------------------- master effects
	/// Toggles the RGB shift (chromatic aberration) effect.
	ToggleRgbShift,
	/// Sets the RGB shift strength in `0.0..=1.0`.
	SetRgbShiftIntensity(f32),
	/// Adds `delta` to the RGB shift strength.
	AdjustRgbShiftIntensity(f32),
	/// Toggles the glitch/block-noise effect.
	ToggleGlitch,
	/// Sets the glitch strength in `0.0..=1.0`.
	SetGlitchIntensity(f32),
	/// Adds `delta` to the glitch strength.
	AdjustGlitchIntensity(f32),
	/// Toggles the strobe/color-invert effect.
	ToggleStrobe,
	/// Sets the strobe strength in `0.0..=1.0`.
	SetStrobeIntensity(f32),
	/// Adds `delta` to the strobe strength.
	AdjustStrobeIntensity(f32),
	/// Sets the strobe frequency in `1.0..=30.0` hertz.
	SetStrobeSpeedHz(f32),
	/// Adds `delta` hertz to the strobe frequency.
	AdjustStrobeSpeedHz(f32),
	/// Sets the brightness offset in `-1.0..=1.0`.
	SetBrightness(f32),
	/// Adds `delta` to the brightness offset.
	AdjustBrightness(f32),
	/// Sets the contrast gain in `0.0..=2.0`.
	SetContrast(f32),
	/// Adds `delta` to the contrast gain.
	AdjustContrast(f32),
	/// Sets the saturation gain in `0.0..=2.0`.
	SetSaturation(f32),
	/// Adds `delta` to the saturation gain.
	AdjustSaturation(f32),
	/// Sets the master output gain in `0.0..=1.0`.
	SetMasterDimmer(f32),
	/// Adds `delta` to the master output gain.
	AdjustMasterDimmer(f32),
	/// Toggles the emergency master blackout.
	ToggleBlackout,
	/// Restores every master effect to its neutral value.
	ResetAllEffects,

	// ------------------------------------------ display, MIDI, files, lifecycle
	/// Selects the external monitor that receives the output, `None` to unset.
	SelectOutputMonitor(Option<String>),
	/// Switches the external output window on or off.
	ToggleOutputEnabled,
	/// Selects the MIDI input port, `None` to disconnect.
	SelectMidiPort(Option<String>),
	/// Asks the MIDI worker to enumerate ports again.
	RescanMidiPorts,
	/// Opens the native "add media files" dialog.
	OpenAddFilesDialog,
	/// Opens the native "add folder" dialog.
	OpenAddFolderDialog,
	/// Unloads a clip and releases it from every layer that used it.
	RemoveClip(ClipId),
	/// Shows or hides the keyboard shortcut overlay.
	ToggleShortcutsOverlay,
	/// Shows or hides the log drawer.
	ToggleLogsDrawer,
	/// Closes the shortcut overlay and the log drawer when either is open.
	CloseOverlays,
	/// Dismisses every visible toast notification.
	ClearToasts,
	/// Opens the quit confirmation guard (SPEC §9.2).
	RequestExit,
	/// Accepts the quit confirmation and lets the application exit.
	ConfirmExit,
	/// Cancels the quit confirmation and keeps the show running.
	CancelExit,
}

/// Side effects produced by [`apply_ui_command`] that live outside [`AppState`].
///
/// `app.rs` reads this value after every command and performs the imperative
/// parts (native dialogs, output window resync, MIDI worker restarts, media
/// streamer restarts). Everything else is already reflected in the state.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UiApplyOutcome {
	/// Layers whose active clip must be restarted on their media streamer.
	pub retriggered_layers: [bool; LAYER_COUNT],
	/// Layers whose streamer must release its clip and stop decoding.
	pub cleared_layers: [bool; LAYER_COUNT],
	/// Whether the output window must be recreated or retargeted.
	pub needs_output_sync: bool,
	/// New MIDI port selection, when the command changed it.
	pub midi_port_changed: Option<Option<String>>,
	/// Whether the MIDI worker should scan for ports again.
	pub rescan_midi: bool,
	/// Whether the native "add files" dialog should be opened.
	pub open_files: bool,
	/// Whether the native "add folder" dialog should be opened.
	pub open_folder: bool,
	/// Whether the quit guard was accepted and the process may exit.
	pub confirm_quit: bool,
	/// Whether the quit guard was just opened.
	pub exit_requested: bool,
	/// Clip the operator asked to unload, already removed from the state.
	pub remove_clip: Option<ClipId>,
}

impl UiApplyOutcome {
	/// Flags `layer` as needing its media streamer restarted.
	pub fn mark_retriggered(&mut self, layer: LayerIndex) {
		if let Some(flag) = self.retriggered_layers.get_mut(layer.index()) {
			*flag = true;
		}
	}

	/// Flags `layer` as needing its media streamer released.
	pub fn mark_cleared(&mut self, layer: LayerIndex) {
		if let Some(flag) = self.cleared_layers.get_mut(layer.index()) {
			*flag = true;
		}
	}

	/// Returns `true` when `layer` must restart its media streamer.
	pub fn was_retriggered(&self, layer: LayerIndex) -> bool {
		self.retriggered_layers.get(layer.index()).copied().unwrap_or(false)
	}

	/// Returns `true` when `layer` must release its media streamer.
	pub fn was_cleared(&self, layer: LayerIndex) -> bool {
		self.cleared_layers.get(layer.index()).copied().unwrap_or(false)
	}

	/// Returns `true` when at least one layer must be resynchronised.
	pub fn any_layer_changed(&self) -> bool {
		self.retriggered_layers.iter().any(|flag| *flag) || self.cleared_layers.iter().any(|flag| *flag)
	}

	/// Returns `true` when the command only changed [`AppState`].
	///
	/// Such a command needs no follow-up work from `app.rs`: no dialog, no
	/// output resync, no MIDI rescan, and no streamer restart.
	pub fn has_no_side_effects(&self) -> bool {
		!self.any_layer_changed()
			&& !self.needs_output_sync
			&& self.midi_port_changed.is_none()
			&& !self.rescan_midi
			&& !self.open_files
			&& !self.open_folder
			&& !self.confirm_quit
			&& !self.exit_requested
			&& self.remove_clip.is_none()
	}
}

/// Applies `cmd` to `app` and reports the impure side effects it requires.
///
/// The function is deterministic and never touches the OS: opening dialogs,
/// moving the output window, and restarting MIDI or media workers are reported
/// through [`UiApplyOutcome`] instead of being executed inline. Missing clips and
/// out-of-range layers therefore only produce a no-op, which keeps the render
/// loop free of panic paths (SPEC §9.2).
pub fn apply_ui_command(app: &mut AppState, cmd: &UiCommand) -> UiApplyOutcome {
	let mut outcome = UiApplyOutcome::default();

	match cmd {
		UiCommand::FocusLayer(layer) => {
			app.focused_layer = *layer;
			app.dirty = true;
		}
		UiCommand::CycleFocusLayer(delta) => {
			app.cycle_focused_layer(*delta);
		}
		UiCommand::TriggerLayer(layer) => {
			if trigger_selected_on_layer(app, *layer) {
				outcome.mark_retriggered(*layer);
			}
		}
		UiCommand::TriggerClipOnLayer { layer, clip_id } => {
			if app.trigger_clip_on_layer(*layer, *clip_id) {
				outcome.mark_retriggered(*layer);
			}
		}
		UiCommand::RetriggerLayer(layer) => {
			if app.retrigger_layer(*layer) {
				outcome.mark_retriggered(*layer);
			}
		}
		UiCommand::RetriggerAllActiveLayers => {
			for layer in LayerIndex::ALL {
				if app.retrigger_layer(layer) {
					outcome.mark_retriggered(layer);
				}
			}
		}
		UiCommand::TogglePauseLayer(layer) => {
			app.toggle_pause_layer(*layer);
		}
		UiCommand::ClearLayer(layer) => {
			if app.clear_layer(*layer) {
				outcome.mark_cleared(*layer);
			}
		}
		UiCommand::SetLayerOpacity { layer, opacity } => {
			app.set_layer_opacity(*layer, *opacity);
		}
		UiCommand::AdjustLayerOpacity { layer, delta } => {
			let opacity = app.layer(*layer).opacity + delta;
			app.set_layer_opacity(*layer, opacity);
		}
		UiCommand::SetLayerSpeed { layer, speed } => {
			app.set_layer_speed(*layer, *speed);
		}
		UiCommand::AdjustLayerSpeed { layer, delta } => {
			let speed = app.layer(*layer).speed + delta;
			app.set_layer_speed(*layer, speed);
		}
		UiCommand::ResetAllLayerSpeeds => {
			for layer in LayerIndex::ALL {
				app.set_layer_speed(layer, 1.0);
			}
		}
		UiCommand::SetLayerBlendMode { layer, blend_mode } => {
			if let Some(state) = app.layer_mut(*layer) {
				state.blend_mode = *blend_mode;
				app.dirty = true;
			}
		}
		UiCommand::CycleLayerBlendMode { layer, forward } => {
			app.cycle_blend_mode(*layer, *forward);
		}
		UiCommand::SetLayerPlaybackMode { layer, mode } => {
			if let Some(state) = app.layer_mut(*layer) {
				state.playback_mode = *mode;
				app.dirty = true;
			}
		}
		UiCommand::ToggleLayerPlaybackMode(layer) => {
			app.cycle_playback_mode(*layer);
		}
		UiCommand::SelectLayerClip { layer, clip_id } => {
			if app.clip_by_id(*clip_id).is_some()
				&& let Some(state) = app.layer_mut(*layer)
			{
				state.selected_clip = Some(*clip_id);
				app.dirty = true;
			}
		}
		UiCommand::StepLayerClip { layer, delta } => {
			if let Some(index) = stepped_clip_index(app, *layer, *delta) {
				app.select_clip_by_index(*layer, index);
			}
		}
		UiCommand::TriggerSelectedClip(layer) => {
			if trigger_selected_on_layer(app, *layer) {
				outcome.mark_retriggered(*layer);
			}
		}

		UiCommand::ToggleRgbShift => {
			app.effects.rgb_shift_enabled = !app.effects.rgb_shift_enabled;
			app.dirty = true;
		}
		UiCommand::SetRgbShiftIntensity(intensity) => {
			app.effects.rgb_shift_intensity = *intensity;
			apply_effect_clamp(app);
		}
		UiCommand::AdjustRgbShiftIntensity(delta) => {
			app.effects.rgb_shift_intensity += delta;
			apply_effect_clamp(app);
		}
		UiCommand::ToggleGlitch => {
			app.effects.glitch_enabled = !app.effects.glitch_enabled;
			app.dirty = true;
		}
		UiCommand::SetGlitchIntensity(intensity) => {
			app.effects.glitch_intensity = *intensity;
			apply_effect_clamp(app);
		}
		UiCommand::AdjustGlitchIntensity(delta) => {
			app.effects.glitch_intensity += delta;
			apply_effect_clamp(app);
		}
		UiCommand::ToggleStrobe => {
			app.effects.strobe_enabled = !app.effects.strobe_enabled;
			app.dirty = true;
		}
		UiCommand::SetStrobeIntensity(intensity) => {
			app.effects.strobe_intensity = *intensity;
			apply_effect_clamp(app);
		}
		UiCommand::AdjustStrobeIntensity(delta) => {
			app.effects.strobe_intensity += delta;
			apply_effect_clamp(app);
		}
		UiCommand::SetStrobeSpeedHz(hz) => {
			app.effects.strobe_speed_hz = *hz;
			apply_effect_clamp(app);
		}
		UiCommand::AdjustStrobeSpeedHz(delta) => {
			app.effects.strobe_speed_hz += delta;
			apply_effect_clamp(app);
		}
		UiCommand::SetBrightness(brightness) => {
			app.effects.brightness = *brightness;
			apply_effect_clamp(app);
		}
		UiCommand::AdjustBrightness(delta) => {
			app.effects.brightness += delta;
			apply_effect_clamp(app);
		}
		UiCommand::SetContrast(contrast) => {
			app.effects.contrast = *contrast;
			apply_effect_clamp(app);
		}
		UiCommand::AdjustContrast(delta) => {
			app.effects.contrast += delta;
			apply_effect_clamp(app);
		}
		UiCommand::SetSaturation(saturation) => {
			app.effects.saturation = *saturation;
			apply_effect_clamp(app);
		}
		UiCommand::AdjustSaturation(delta) => {
			app.effects.saturation += delta;
			apply_effect_clamp(app);
		}
		UiCommand::SetMasterDimmer(dimmer) => {
			app.effects.master_dimmer = *dimmer;
			apply_effect_clamp(app);
		}
		UiCommand::AdjustMasterDimmer(delta) => {
			app.effects.master_dimmer += delta;
			apply_effect_clamp(app);
		}
		UiCommand::ToggleBlackout => {
			app.effects.blackout = !app.effects.blackout;
			app.dirty = true;
		}
		UiCommand::ResetAllEffects => {
			app.effects.reset_effects();
			app.dirty = true;
		}

		UiCommand::SelectOutputMonitor(name) => {
			app.output.monitor_name = name.clone();
			app.dirty = true;
			outcome.needs_output_sync = true;
		}
		UiCommand::ToggleOutputEnabled => {
			app.output.enabled = !app.output.enabled;
			app.dirty = true;
			outcome.needs_output_sync = true;
		}
		UiCommand::SelectMidiPort(port) => {
			app.selected_midi_port = port.clone();
			app.dirty = true;
			outcome.midi_port_changed = Some(port.clone());
		}
		UiCommand::RescanMidiPorts => {
			app.midi_status = MidiConnectionStatus::Scanning;
			app.dirty = true;
			outcome.rescan_midi = true;
		}
		UiCommand::OpenAddFilesDialog => {
			outcome.open_files = true;
		}
		UiCommand::OpenAddFolderDialog => {
			outcome.open_folder = true;
		}
		UiCommand::RemoveClip(clip_id) => {
			remove_clip(app, *clip_id, &mut outcome);
		}
		UiCommand::ToggleShortcutsOverlay => {
			app.show_shortcuts = !app.show_shortcuts;
			app.dirty = true;
		}
		UiCommand::ToggleLogsDrawer => {
			app.show_logs = !app.show_logs;
			app.dirty = true;
		}
		UiCommand::CloseOverlays => {
			if app.show_shortcuts {
				app.show_shortcuts = false;
				app.dirty = true;
			}
			if app.show_logs {
				app.show_logs = false;
				app.dirty = true;
			}
		}
		UiCommand::ClearToasts => {
			app.toasts.clear_toasts();
		}
		UiCommand::RequestExit => {
			app.exit_confirm_open = true;
			app.dirty = true;
			outcome.exit_requested = true;
		}
		UiCommand::ConfirmExit => {
			app.exit_confirm_open = false;
			outcome.confirm_quit = true;
		}
		UiCommand::CancelExit => {
			app.exit_confirm_open = false;
		}
	}

	outcome
}

/// Writes a clamped effect value back into the state and marks it dirty.
fn apply_effect_clamp(app: &mut AppState) {
	app.effects.clamp_all();
	app.dirty = true;
}

/// Fires the layer's selected clip, retriggering when it is already active.
///
/// Mirrors the MIDI trigger button (SPEC §6.3): pressing it while the selected
/// clip is already playing restarts that clip from its first frame instead of
/// rebinding it, so a live button cannot accidentally reload the media.
fn trigger_selected_on_layer(app: &mut AppState, layer: LayerIndex) -> bool {
	let selected = app.layer(layer).selected_clip;
	let active = app.layer(layer).active_clip;
	match selected {
		Some(clip) if active == Some(clip) => app.retrigger_layer(layer),
		Some(clip) => app.trigger_clip_on_layer(layer, clip),
		None => app.retrigger_layer(layer),
	}
}

/// Resolves the clip-grid index reached by stepping the layer selection.
///
/// Wraps around the grid in both directions, starts from the first clip when
/// moving forward without a selection and from the last clip when moving
/// backward, and returns `None` when no clip is loaded.
fn stepped_clip_index(app: &AppState, layer: LayerIndex, delta: i32) -> Option<usize> {
	let count = app.clips.len();
	if count == 0 || delta == 0 {
		return None;
	}

	let count_i64 = i64::try_from(count).unwrap_or(i64::MAX);
	let current = app
		.layer(layer)
		.selected_clip
		.and_then(|id| app.clips.iter().position(|clip| clip.id == id));

	let target = match current {
		Some(index) => {
			let index_i64 = i64::try_from(index).unwrap_or(0);
			(index_i64 + i64::from(delta)).rem_euclid(count_i64)
		}
		None if delta > 0 => 0,
		None => count_i64.saturating_sub(1),
	};

	usize::try_from(target).ok().filter(|index| *index < count)
}

/// Removes `clip_id` from the session and releases it from every layer.
fn remove_clip(app: &mut AppState, clip_id: ClipId, outcome: &mut UiApplyOutcome) {
	let before = app.clips.len();
	app.clips.retain(|clip| clip.id != clip_id);

	for layer in LayerIndex::ALL {
		let mut released = false;
		if let Some(state) = app.layer_mut(layer) {
			if state.active_clip == Some(clip_id) {
				state.active_clip = None;
				state.playback_state = PlaybackState::Stopped;
				state.frame_count = 0;
				state.playhead_frame = 0.0;
				state.last_error = None;
				released = true;
			}
			if state.selected_clip == Some(clip_id) {
				state.selected_clip = None;
			}
		}
		if released {
			outcome.mark_cleared(layer);
		}
	}

	if app.clips.len() != before {
		app.dirty = true;
	}
	outcome.remove_clip = Some(clip_id);
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use std::path::PathBuf;

	use super::*;
	use crate::model::clip::{ClipInfo, ClipKind, ClipStatus};

	fn clip(id: u64) -> ClipInfo {
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
			status: ClipStatus::Ready,
			thumbnail: None,
		}
	}

	fn layer(offset: u8) -> LayerIndex {
		LayerIndex::new(usize::from(offset)).unwrap()
	}

	fn state_with_clips(count: u64) -> AppState {
		let mut state = AppState::default();
		for id in 1..=count {
			state.upsert_clip(clip(id));
		}
		state
	}

	fn approx(actual: f32, expected: f32) {
		assert!((actual - expected).abs() < 1e-5, "expected {expected}, got {actual}");
	}

	#[test]
	fn focus_and_cycle_move_within_the_layer_ring() {
		let mut state = AppState::default();
		let outcome = apply_ui_command(&mut state, &UiCommand::FocusLayer(layer(2)));
		assert_eq!(state.focused_layer, layer(2));
		assert!(outcome.has_no_side_effects());

		apply_ui_command(&mut state, &UiCommand::CycleFocusLayer(1));
		assert_eq!(state.focused_layer, layer(3));
		apply_ui_command(&mut state, &UiCommand::CycleFocusLayer(-2));
		assert_eq!(state.focused_layer, layer(1));
	}

	#[test]
	fn trigger_layer_binds_selection_then_retriggers_it() {
		let mut state = state_with_clips(1);
		let target = layer(0);
		state.layers[0].selected_clip = Some(ClipId(1));

		let outcome = apply_ui_command(&mut state, &UiCommand::TriggerLayer(target));
		assert!(outcome.was_retriggered(target));
		assert_eq!(state.layer(target).active_clip, Some(ClipId(1)));

		state.advance_layers(0.5);
		assert!(state.layer(target).playhead_frame > 0.0);
		let outcome = apply_ui_command(&mut state, &UiCommand::TriggerLayer(target));
		assert!(outcome.was_retriggered(target));
		assert_eq!(state.layer(target).playhead_frame, 0.0);
	}

	#[test]
	fn trigger_layer_without_selection_is_a_noop() {
		let mut state = state_with_clips(1);
		let target = layer(1);
		let outcome = apply_ui_command(&mut state, &UiCommand::TriggerLayer(target));
		assert!(!outcome.any_layer_changed());
		assert!(state.layer(target).active_clip.is_none());
	}

	#[test]
	fn trigger_clip_on_layer_reports_missing_clips_without_panicking() {
		let mut state = state_with_clips(1);
		let target = layer(0);
		let outcome = apply_ui_command(
			&mut state,
			&UiCommand::TriggerClipOnLayer {
				layer: target,
				clip_id: ClipId(99),
			},
		);
		assert!(!outcome.any_layer_changed());
		assert!(state.layer(target).last_error.is_some());
		assert_eq!(state.toasts.toasts().len(), 1);
	}

	#[test]
	fn retrigger_all_active_layers_flags_only_playing_layers() {
		let mut state = state_with_clips(2);
		apply_ui_command(
			&mut state,
			&UiCommand::TriggerClipOnLayer {
				layer: layer(0),
				clip_id: ClipId(1),
			},
		);
		apply_ui_command(
			&mut state,
			&UiCommand::TriggerClipOnLayer {
				layer: layer(3),
				clip_id: ClipId(2),
			},
		);

		let outcome = apply_ui_command(&mut state, &UiCommand::RetriggerAllActiveLayers);
		assert!(outcome.was_retriggered(layer(0)));
		assert!(outcome.was_retriggered(layer(3)));
		assert!(!outcome.was_retriggered(layer(1)));
		assert!(!outcome.was_retriggered(layer(2)));
	}

	#[test]
	fn pause_clear_and_retrigger_single_layer() {
		let mut state = state_with_clips(1);
		let target = layer(2);
		apply_ui_command(
			&mut state,
			&UiCommand::TriggerClipOnLayer {
				layer: target,
				clip_id: ClipId(1),
			},
		);

		apply_ui_command(&mut state, &UiCommand::TogglePauseLayer(target));
		assert_eq!(state.layer(target).playback_state, PlaybackState::Paused);
		apply_ui_command(&mut state, &UiCommand::TogglePauseLayer(target));
		assert_eq!(state.layer(target).playback_state, PlaybackState::Playing);

		let outcome = apply_ui_command(&mut state, &UiCommand::RetriggerLayer(target));
		assert!(outcome.was_retriggered(target));

		let outcome = apply_ui_command(&mut state, &UiCommand::ClearLayer(target));
		assert!(outcome.was_cleared(target));
		assert!(!outcome.was_retriggered(target));
		assert_eq!(state.layer(target).active_clip, None);
	}

	#[test]
	fn opacity_and_speed_commands_clamp() {
		let mut state = AppState::default();
		let target = layer(0);

		apply_ui_command(
			&mut state,
			&UiCommand::SetLayerOpacity {
				layer: target,
				opacity: 0.5,
			},
		);
		approx(state.layer(target).opacity, 0.5);
		apply_ui_command(
			&mut state,
			&UiCommand::AdjustLayerOpacity {
				layer: target,
				delta: 0.75,
			},
		);
		approx(state.layer(target).opacity, 1.0);
		apply_ui_command(
			&mut state,
			&UiCommand::AdjustLayerOpacity {
				layer: target,
				delta: -5.0,
			},
		);
		approx(state.layer(target).opacity, 0.0);

		apply_ui_command(
			&mut state,
			&UiCommand::AdjustLayerSpeed {
				layer: target,
				delta: -1.5,
			},
		);
		approx(state.layer(target).speed, -0.5);
		apply_ui_command(
			&mut state,
			&UiCommand::SetLayerSpeed {
				layer: target,
				speed: 99.0,
			},
		);
		approx(state.layer(target).speed, 4.0);

		apply_ui_command(
			&mut state,
			&UiCommand::AdjustLayerSpeed {
				layer: target,
				delta: 0.25,
			},
		);
		approx(state.layer(target).speed, 4.0);

		let outcome = apply_ui_command(&mut state, &UiCommand::ResetAllLayerSpeeds);
		assert!(outcome.has_no_side_effects());
		for layer in LayerIndex::ALL {
			approx(state.layer(layer).speed, 1.0);
		}
	}

	#[test]
	fn blend_and_playback_mode_commands() {
		let mut state = AppState::default();
		let target = layer(1);

		apply_ui_command(
			&mut state,
			&UiCommand::SetLayerBlendMode {
				layer: target,
				blend_mode: BlendMode::Screen,
			},
		);
		assert_eq!(state.layer(target).blend_mode, BlendMode::Screen);

		apply_ui_command(
			&mut state,
			&UiCommand::CycleLayerBlendMode {
				layer: target,
				forward: true,
			},
		);
		assert_eq!(state.layer(target).blend_mode, BlendMode::Difference);
		apply_ui_command(
			&mut state,
			&UiCommand::CycleLayerBlendMode {
				layer: target,
				forward: false,
			},
		);
		assert_eq!(state.layer(target).blend_mode, BlendMode::Screen);

		apply_ui_command(
			&mut state,
			&UiCommand::SetLayerPlaybackMode {
				layer: target,
				mode: PlaybackMode::OneShot,
			},
		);
		assert_eq!(state.layer(target).playback_mode, PlaybackMode::OneShot);
		apply_ui_command(&mut state, &UiCommand::ToggleLayerPlaybackMode(target));
		assert_eq!(state.layer(target).playback_mode, PlaybackMode::Loop);
	}

	#[test]
	fn clip_selection_steps_through_the_grid_with_wrapping() {
		let mut state = state_with_clips(3);
		let target = layer(0);

		// No selection: forward starts at the first clip, backward at the last one.
		apply_ui_command(
			&mut state,
			&UiCommand::StepLayerClip {
				layer: target,
				delta: -1,
			},
		);
		assert_eq!(state.layer(target).selected_clip, Some(ClipId(3)));

		apply_ui_command(
			&mut state,
			&UiCommand::StepLayerClip {
				layer: target,
				delta: 1,
			},
		);
		assert_eq!(state.layer(target).selected_clip, Some(ClipId(1)));

		apply_ui_command(
			&mut state,
			&UiCommand::StepLayerClip {
				layer: target,
				delta: -1,
			},
		);
		assert_eq!(state.layer(target).selected_clip, Some(ClipId(3)));

		// Zero delta and an empty grid are no-ops.
		apply_ui_command(
			&mut state,
			&UiCommand::StepLayerClip {
				layer: target,
				delta: 0,
			},
		);
		assert_eq!(state.layer(target).selected_clip, Some(ClipId(3)));

		let mut empty = AppState::default();
		apply_ui_command(
			&mut empty,
			&UiCommand::StepLayerClip {
				layer: target,
				delta: 1,
			},
		);
		assert_eq!(empty.layer(target).selected_clip, None);
	}

	#[test]
	fn select_layer_clip_ignores_unloaded_clips() {
		let mut state = state_with_clips(2);
		let target = layer(0);

		apply_ui_command(
			&mut state,
			&UiCommand::SelectLayerClip {
				layer: target,
				clip_id: ClipId(2),
			},
		);
		assert_eq!(state.layer(target).selected_clip, Some(ClipId(2)));

		apply_ui_command(
			&mut state,
			&UiCommand::SelectLayerClip {
				layer: target,
				clip_id: ClipId(42),
			},
		);
		assert_eq!(state.layer(target).selected_clip, Some(ClipId(2)));
	}

	#[test]
	fn trigger_selected_clip_reports_streamer_restart() {
		let mut state = state_with_clips(1);
		let target = layer(2);
		state.layers[2].selected_clip = Some(ClipId(1));

		let outcome = apply_ui_command(&mut state, &UiCommand::TriggerSelectedClip(target));
		assert!(outcome.was_retriggered(target));
		assert_eq!(state.layer(target).active_clip, Some(ClipId(1)));
	}

	#[test]
	fn effect_toggles_flip_once_per_command() {
		let mut state = AppState::default();

		for _ in 0..2 {
			apply_ui_command(&mut state, &UiCommand::ToggleRgbShift);
			apply_ui_command(&mut state, &UiCommand::ToggleGlitch);
			apply_ui_command(&mut state, &UiCommand::ToggleStrobe);
			apply_ui_command(&mut state, &UiCommand::ToggleBlackout);
		}
		assert!(!state.effects.rgb_shift_enabled);
		assert!(!state.effects.glitch_enabled);
		assert!(!state.effects.strobe_enabled);
		assert!(!state.effects.blackout);

		apply_ui_command(&mut state, &UiCommand::ToggleBlackout);
		assert!(state.effects.blackout);
	}

	#[test]
	fn effect_intensity_commands_are_clamped() {
		let mut state = AppState::default();

		apply_ui_command(&mut state, &UiCommand::AdjustRgbShiftIntensity(0.4));
		approx(state.effects.rgb_shift_intensity, 0.4);
		apply_ui_command(&mut state, &UiCommand::AdjustRgbShiftIntensity(9.0));
		approx(state.effects.rgb_shift_intensity, 1.0);
		apply_ui_command(&mut state, &UiCommand::SetRgbShiftIntensity(-3.0));
		approx(state.effects.rgb_shift_intensity, 0.0);

		apply_ui_command(&mut state, &UiCommand::AdjustGlitchIntensity(0.25));
		approx(state.effects.glitch_intensity, 0.25);
		apply_ui_command(&mut state, &UiCommand::SetGlitchIntensity(5.0));
		approx(state.effects.glitch_intensity, 1.0);

		apply_ui_command(&mut state, &UiCommand::AdjustStrobeIntensity(0.3));
		approx(state.effects.strobe_intensity, 0.3);
		apply_ui_command(&mut state, &UiCommand::SetStrobeIntensity(f32::NAN));
		approx(state.effects.strobe_intensity, 0.0);

		apply_ui_command(&mut state, &UiCommand::AdjustStrobeSpeedHz(4.0));
		approx(state.effects.strobe_speed_hz, 12.0);
		apply_ui_command(&mut state, &UiCommand::AdjustStrobeSpeedHz(100.0));
		approx(state.effects.strobe_speed_hz, 30.0);
		apply_ui_command(&mut state, &UiCommand::SetStrobeSpeedHz(0.0));
		approx(state.effects.strobe_speed_hz, 1.0);

		apply_ui_command(&mut state, &UiCommand::AdjustBrightness(0.2));
		approx(state.effects.brightness, 0.2);
		apply_ui_command(&mut state, &UiCommand::SetBrightness(-9.0));
		approx(state.effects.brightness, -1.0);

		apply_ui_command(&mut state, &UiCommand::AdjustContrast(0.5));
		approx(state.effects.contrast, 1.5);
		apply_ui_command(&mut state, &UiCommand::SetContrast(f32::INFINITY));
		approx(state.effects.contrast, 1.0);

		apply_ui_command(&mut state, &UiCommand::AdjustSaturation(-2.0));
		approx(state.effects.saturation, 0.0);
		apply_ui_command(&mut state, &UiCommand::SetSaturation(9.0));
		approx(state.effects.saturation, 2.0);

		apply_ui_command(&mut state, &UiCommand::AdjustMasterDimmer(-0.75));
		approx(state.effects.master_dimmer, 0.25);
		apply_ui_command(&mut state, &UiCommand::SetMasterDimmer(9.0));
		approx(state.effects.master_dimmer, 1.0);
		apply_ui_command(&mut state, &UiCommand::AdjustMasterDimmer(-9.0));
		approx(state.effects.master_dimmer, 0.0);
	}

	#[test]
	fn reset_all_effects_keeps_live_safety_values() {
		let mut state = AppState::default();
		apply_ui_command(&mut state, &UiCommand::SetRgbShiftIntensity(0.8));
		apply_ui_command(&mut state, &UiCommand::SetMasterDimmer(0.3));
		apply_ui_command(&mut state, &UiCommand::ToggleBlackout);

		apply_ui_command(&mut state, &UiCommand::ResetAllEffects);
		approx(state.effects.rgb_shift_intensity, 0.0);
		approx(state.effects.master_dimmer, 0.3);
		assert!(state.effects.blackout);
	}

	#[test]
	fn output_commands_request_a_window_sync() {
		let mut state = AppState::default();
		let outcome = apply_ui_command(&mut state, &UiCommand::SelectOutputMonitor(Some("DELL".into())));
		assert!(outcome.needs_output_sync);
		assert_eq!(state.output.monitor_name.as_deref(), Some("DELL"));

		let outcome = apply_ui_command(&mut state, &UiCommand::ToggleOutputEnabled);
		assert!(outcome.needs_output_sync);
		assert!(state.output.enabled);
		apply_ui_command(&mut state, &UiCommand::ToggleOutputEnabled);
		assert!(!state.output.enabled);

		let outcome = apply_ui_command(&mut state, &UiCommand::SelectOutputMonitor(None));
		assert!(outcome.needs_output_sync);
		assert!(state.output.monitor_name.is_none());
	}

	#[test]
	fn midi_commands_report_port_change_and_rescan() {
		let mut state = AppState::default();
		let outcome = apply_ui_command(&mut state, &UiCommand::SelectMidiPort(Some("LCXL".into())));
		assert_eq!(outcome.midi_port_changed, Some(Some("LCXL".to_string())));
		assert_eq!(state.selected_midi_port.as_deref(), Some("LCXL"));

		let outcome = apply_ui_command(&mut state, &UiCommand::SelectMidiPort(None));
		assert_eq!(outcome.midi_port_changed, Some(None));
		assert!(state.selected_midi_port.is_none());

		let outcome = apply_ui_command(&mut state, &UiCommand::RescanMidiPorts);
		assert!(outcome.rescan_midi);
		assert_eq!(state.midi_status, MidiConnectionStatus::Scanning);
	}

	#[test]
	fn dialog_commands_only_report_side_effects() {
		let mut state = AppState::default();
		let files = apply_ui_command(&mut state, &UiCommand::OpenAddFilesDialog);
		assert!(files.open_files);
		assert!(!files.open_folder);
		let folder = apply_ui_command(&mut state, &UiCommand::OpenAddFolderDialog);
		assert!(folder.open_folder);
		assert!(!folder.open_files);
	}

	#[test]
	fn remove_clip_releases_every_reference() {
		let mut state = state_with_clips(2);
		apply_ui_command(
			&mut state,
			&UiCommand::TriggerClipOnLayer {
				layer: layer(1),
				clip_id: ClipId(1),
			},
		);
		state.layers[2].selected_clip = Some(ClipId(1));

		let outcome = apply_ui_command(&mut state, &UiCommand::RemoveClip(ClipId(1)));
		assert_eq!(outcome.remove_clip, Some(ClipId(1)));
		assert!(outcome.was_cleared(layer(1)));
		assert_eq!(state.clips.len(), 1);
		assert!(state.layer(layer(1)).active_clip.is_none());
		assert_eq!(state.layer(layer(1)).playback_state, PlaybackState::Stopped);
		assert_eq!(state.layer(layer(2)).selected_clip, None);
		assert_eq!(state.layer(layer(0)).active_clip, None);
	}

	#[test]
	fn remove_unknown_clip_is_harmless() {
		let mut state = state_with_clips(1);
		let outcome = apply_ui_command(&mut state, &UiCommand::RemoveClip(ClipId(77)));
		assert_eq!(outcome.remove_clip, Some(ClipId(77)));
		assert!(!outcome.any_layer_changed());
		assert_eq!(state.clips.len(), 1);
	}

	#[test]
	fn overlay_and_toast_commands() {
		let mut state = AppState::default();

		apply_ui_command(&mut state, &UiCommand::ToggleShortcutsOverlay);
		assert!(state.show_shortcuts);
		apply_ui_command(&mut state, &UiCommand::ToggleLogsDrawer);
		assert!(state.show_logs);

		apply_ui_command(&mut state, &UiCommand::CloseOverlays);
		assert!(!state.show_shortcuts);
		assert!(!state.show_logs);
		// Closing again is a no-op.
		apply_ui_command(&mut state, &UiCommand::CloseOverlays);
		assert!(!state.show_logs);

		state.push_toast(crate::error::ToastLevel::Info, "title", "message", 1.0);
		assert_eq!(state.toasts.toasts().len(), 1);
		apply_ui_command(&mut state, &UiCommand::ClearToasts);
		assert!(state.toasts.toasts().is_empty());
	}

	#[test]
	fn exit_guard_commands() {
		let mut state = AppState::default();

		let outcome = apply_ui_command(&mut state, &UiCommand::RequestExit);
		assert!(outcome.exit_requested);
		assert!(!outcome.confirm_quit);
		assert!(state.exit_confirm_open);

		let outcome = apply_ui_command(&mut state, &UiCommand::CancelExit);
		assert!(!outcome.confirm_quit);
		assert!(!state.exit_confirm_open);

		apply_ui_command(&mut state, &UiCommand::RequestExit);
		let outcome = apply_ui_command(&mut state, &UiCommand::ConfirmExit);
		assert!(outcome.confirm_quit);
		assert!(!state.exit_confirm_open);
	}

	#[test]
	fn layered_commands_mark_state_dirty() {
		let mut state = AppState::default();
		assert!(!state.dirty);
		apply_ui_command(&mut state, &UiCommand::FocusLayer(layer(2)));
		assert!(state.dirty);

		let mut state = AppState::default();
		apply_ui_command(
			&mut state,
			&UiCommand::SelectLayerClip {
				layer: layer(0),
				clip_id: ClipId(1),
			},
		);
		// The clip is not loaded, so nothing changed.
		assert!(!state.dirty);
		assert_eq!(state.layer(layer(0)).selected_clip, None);
	}
}
