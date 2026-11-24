//! Exact Novation Launch Control XL 3 mapping for MIDI Channel 1.
//!
//! The mapping mirrors SPEC §6.3 and is intentionally fixed: MUR owns the custom
//! template, so every fader, encoder, and button keeps the same meaning for the
//! whole session. Only Channel 1 messages are accepted; anything on Channels
//! 2–16 is ignored so other controllers on the same bus cannot disturb the show.
//!
//! | Control | Message | Number | MUR action |
//! | --- | --- | --- | --- |
//! | Faders 1–4 | CC | `1`–`4` | Layer 1–4 opacity |
//! | Encoders Row 1 (1–4) | CC | `11`–`14` | Layer 1–4 playback speed |
//! | Encoders Row 2 (9–12) | CC | `21`–`24` | Layer 1–4 blend mode |
//! | Encoders Row 3 (17–20) | CC | `31`–`34` | Layer 1–4 clip selection |
//! | Faders 5–8 | CC | `5`–`8` | RGB shift / glitch / strobe intensity, master dimmer |
//! | Encoders Row 1 (5–8) | CC | `15`–`18` | Brightness, contrast, saturation, strobe speed |
//! | Buttons Row 1 (1–4) | Note | `36`–`39` | Layer 1–4 trigger / retrigger |
//! | Buttons Row 2 (9–12) | Note | `44`–`47` | Layer 1–4 pause / resume toggle |
//! | Buttons Row 1 (5–8) | Note | `40`–`43` | RGB shift, glitch, strobe, blackout toggles |
//! | Buttons Row 2 (13–16) | Note | `48`–`51` | Effect reset, retrigger all active layers, focused playback mode, speed reset |
//! | Reserved | CC | `25`–`28`, `35`–`38` | Unassigned |

use crate::model::effect::{cc_to_brightness, cc_to_factor_centered, cc_to_strobe_hz, cc_to_unit};
use crate::model::layer::LAYER_COUNT;
use crate::model::{AppState, BlendMode, LayerIndex, LayerState};

/// Status byte of a Control Change message on MIDI Channel 1.
const STATUS_CONTROL_CHANGE: u8 = 0xB0;
/// Status byte of a Note On message on MIDI Channel 1.
const STATUS_NOTE_ON: u8 = 0x90;
/// Status byte of a Note Off message on MIDI Channel 1.
const STATUS_NOTE_OFF: u8 = 0x80;

/// One semantic control action decoded from a single MIDI message.
///
/// Values are already converted into their model units (`0.0..=1.0` for
/// intensities, `-2.0..=4.0` for speed, and so on), so consumers never see raw
/// 7-bit controller values except in [`MidiAction::SelectLayerClip`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MidiAction {
	/// Sets a layer's opacity in `0.0..=1.0`.
	SetLayerOpacity {
		/// Layer controlled by the fader.
		layer: LayerIndex,
		/// Requested opacity, converted with [`cc_to_unit`].
		opacity: f32,
	},
	/// Sets a layer's playback speed in `-2.0..=4.0`.
	SetLayerSpeed {
		/// Layer controlled by the encoder.
		layer: LayerIndex,
		/// Requested speed, converted with [`LayerState::speed_from_midi_cc`].
		speed: f32,
	},
	/// Switches a layer's blend mode.
	SetLayerBlendMode {
		/// Layer controlled by the encoder.
		layer: LayerIndex,
		/// Mode selected by [`BlendMode::from_midi_cc`].
		blend_mode: BlendMode,
	},
	/// Selects which loaded clip the layer's trigger button will fire.
	SelectLayerClip {
		/// Layer whose selection changes.
		layer: LayerIndex,
		/// Raw 7-bit controller value, resolved later via [`resolve_clip_index`].
		cc_value: u8,
	},
	/// Fires the layer's selected clip, or restarts it when already active.
	TriggerLayer(LayerIndex),
	/// Toggles pause/resume on a layer.
	TogglePauseLayer(LayerIndex),
	/// Sets the RGB shift intensity in `0.0..=1.0`.
	SetRgbShiftIntensity(f32),
	/// Sets the glitch strength in `0.0..=1.0`.
	SetGlitchIntensity(f32),
	/// Sets the strobe intensity in `0.0..=1.0`.
	SetStrobeIntensity(f32),
	/// Sets the master output gain in `0.0..=1.0`.
	SetMasterDimmer(f32),
	/// Sets the brightness offset in `-1.0..=1.0`.
	SetBrightness(f32),
	/// Sets the contrast gain in `0.0..=2.0`.
	SetContrast(f32),
	/// Sets the saturation gain in `0.0..=2.0`.
	SetSaturation(f32),
	/// Sets the strobe frequency in `1.0..=30.0` hertz.
	SetStrobeSpeedHz(f32),
	/// Toggles the RGB shift effect.
	ToggleRgbShift,
	/// Toggles the glitch effect.
	ToggleGlitch,
	/// Toggles the strobe effect.
	ToggleStrobe,
	/// Toggles the emergency master blackout.
	ToggleBlackout,
	/// Restores every master effect to its neutral value.
	ResetAllEffects,
	/// Restarts every layer that currently has an active clip.
	RetriggerAllActiveLayers,
	/// Switches the focused layer between loop and one-shot playback.
	ToggleFocusedLayerPlaybackMode,
	/// Returns every layer to real-time (`1.0x`) playback speed.
	ResetAllLayerSpeeds,
}

/// Decodes one raw MIDI message into a [`MidiAction`].
///
/// `bytes` is expected to hold exactly one complete MIDI message as delivered by
/// a `midir` input callback (for example `[0xB0, 1, 64]`). Concatenated packets
/// and running-status streams are not decoded: only the first three bytes are
/// read, so callers must split multi-message buffers before calling this
/// function.
///
/// The function is deliberately total: short packets, System messages, wrong
/// channels, Note Off messages, and unassigned controller numbers all yield
/// `None` instead of an error, so the MIDI thread never allocates or logs per
/// message. Note On with velocity `0` is treated as a release (SPEC §6.1 keeps
/// every button momentary), which makes toggles fire exactly once per press.
pub fn parse_midi_message(bytes: &[u8]) -> Option<MidiAction> {
	let status = *bytes.first()?;
	if bytes.len() < 3 || status & 0x0F != 0 {
		return None;
	}
	let data1 = *bytes.get(1)?;
	let data2 = *bytes.get(2)?;

	match status {
		STATUS_CONTROL_CHANGE => parse_control_change(data1, data2),
		STATUS_NOTE_ON => parse_note_on(data1, data2),
		STATUS_NOTE_OFF => None,
		_ => None,
	}
}

/// Decodes a Channel 1 Control Change message.
fn parse_control_change(controller: u8, value: u8) -> Option<MidiAction> {
	let controller = controller.min(127);
	let value = value.min(127);

	match controller {
		1..=4 => Some(MidiAction::SetLayerOpacity {
			layer: layer_at(controller - 1)?,
			opacity: cc_to_unit(value),
		}),
		11..=14 => Some(MidiAction::SetLayerSpeed {
			layer: layer_at(controller - 11)?,
			speed: LayerState::speed_from_midi_cc(value),
		}),
		21..=24 => Some(MidiAction::SetLayerBlendMode {
			layer: layer_at(controller - 21)?,
			blend_mode: BlendMode::from_midi_cc(value),
		}),
		31..=34 => Some(MidiAction::SelectLayerClip {
			layer: layer_at(controller - 31)?,
			cc_value: value,
		}),
		5 => Some(MidiAction::SetRgbShiftIntensity(cc_to_unit(value))),
		6 => Some(MidiAction::SetGlitchIntensity(cc_to_unit(value))),
		7 => Some(MidiAction::SetStrobeIntensity(cc_to_unit(value))),
		8 => Some(MidiAction::SetMasterDimmer(cc_to_unit(value))),
		15 => Some(MidiAction::SetBrightness(cc_to_brightness(value))),
		16 => Some(MidiAction::SetContrast(cc_to_factor_centered(value))),
		17 => Some(MidiAction::SetSaturation(cc_to_factor_centered(value))),
		18 => Some(MidiAction::SetStrobeSpeedHz(cc_to_strobe_hz(value))),
		_ => None,
	}
}

/// Decodes a Channel 1 Note On message, ignoring releases (`velocity == 0`).
fn parse_note_on(note: u8, velocity: u8) -> Option<MidiAction> {
	if velocity == 0 {
		return None;
	}
	let note = note.min(127);

	match note {
		36..=39 => Some(MidiAction::TriggerLayer(layer_at(note - 36)?)),
		44..=47 => Some(MidiAction::TogglePauseLayer(layer_at(note - 44)?)),
		40 => Some(MidiAction::ToggleRgbShift),
		41 => Some(MidiAction::ToggleGlitch),
		42 => Some(MidiAction::ToggleStrobe),
		43 => Some(MidiAction::ToggleBlackout),
		48 => Some(MidiAction::ResetAllEffects),
		49 => Some(MidiAction::RetriggerAllActiveLayers),
		50 => Some(MidiAction::ToggleFocusedLayerPlaybackMode),
		51 => Some(MidiAction::ResetAllLayerSpeeds),
		_ => None,
	}
}

/// Converts a zero-based layer offset into a [`LayerIndex`].
fn layer_at(offset: u8) -> Option<LayerIndex> {
	LayerIndex::new(usize::from(offset))
}

/// Resolves a clip-selection controller value against the loaded clip count.
///
/// Returns `None` when no clip is loaded. Up to 128 clips, the controller's full
/// travel is spread evenly across the list (`resolve_clip_index(127, 4)` picks
/// index `3`); beyond 128 clips the value addresses the first 128 entries
/// directly, which keeps every reachable slot stable while the grid scrolls.
pub fn resolve_clip_index(cc_value: u8, clip_count: usize) -> Option<usize> {
	if clip_count == 0 {
		return None;
	}
	let last = clip_count - 1;
	let value = usize::from(cc_value.min(127));
	if clip_count <= 128 {
		Some(value.saturating_mul(clip_count).saturating_div(128).min(last))
	} else {
		Some(value.min(last))
	}
}

/// Layers whose streamer must be (re)activated after a MIDI action was applied.
///
/// [`apply_midi_action`] never touches the renderer directly; instead it flags
/// the affected layers so `app.rs` can bind the clip and restart that layer's
/// media streamer in the correct order on the render thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MidiApplyResult {
	/// One flag per layer, indexed like [`LayerIndex`].
	pub retriggered_layers: [bool; LAYER_COUNT],
}

impl MidiApplyResult {
	/// Flags `layer` as needing a streamer restart.
	pub fn mark(&mut self, layer: LayerIndex) {
		if let Some(flag) = self.retriggered_layers.get_mut(layer.index()) {
			*flag = true;
		}
	}

	/// Returns `true` when at least one layer must be restarted.
	pub fn any(&self) -> bool {
		self.retriggered_layers.iter().any(|flag| *flag)
	}

	/// Returns `true` when `layer` was triggered or retriggered.
	pub fn contains(&self, layer: LayerIndex) -> bool {
		self.retriggered_layers.get(layer.index()).copied().unwrap_or(false)
	}

	/// Iterates over the layers that must be restarted, in ascending order.
	pub fn layers(&self) -> impl Iterator<Item = LayerIndex> + '_ {
		LayerIndex::ALL.into_iter().filter(|layer| self.contains(*layer))
	}
}

/// Applies a decoded MIDI action to the session state.
///
/// Every state change goes through the [`AppState`] helpers, so clamping,
/// dirty-tracking, and toast reporting stay in one place. The return value lists
/// the layers that must be resynchronized with their media streamer.
pub fn apply_midi_action(app: &mut AppState, action: MidiAction) -> MidiApplyResult {
	let mut result = MidiApplyResult::default();

	match action {
		MidiAction::SetLayerOpacity { layer, opacity } => {
			app.set_layer_opacity(layer, opacity);
		}
		MidiAction::SetLayerSpeed { layer, speed } => {
			app.set_layer_speed(layer, speed);
		}
		MidiAction::SetLayerBlendMode { layer, blend_mode } => {
			let updated = app.layer_mut(layer).map(|state| state.blend_mode = blend_mode);
			if updated.is_some() {
				app.dirty = true;
			}
		}
		MidiAction::SelectLayerClip { layer, cc_value } => {
			if let Some(index) = resolve_clip_index(cc_value, app.clips.len()) {
				app.select_clip_by_index(layer, index);
			}
		}
		MidiAction::TriggerLayer(layer) => {
			let selected = app.layer(layer).selected_clip;
			let active = app.layer(layer).active_clip;
			match selected {
				Some(clip) if active == Some(clip) => {
					if app.retrigger_layer(layer) {
						result.mark(layer);
					}
				}
				Some(clip) => {
					if app.trigger_clip_on_layer(layer, clip) {
						result.mark(layer);
					}
				}
				None => {
					if app.retrigger_layer(layer) {
						result.mark(layer);
					}
				}
			}
		}
		MidiAction::TogglePauseLayer(layer) => {
			app.toggle_pause_layer(layer);
		}
		MidiAction::SetRgbShiftIntensity(intensity) => {
			app.effects.rgb_shift_intensity = intensity;
			app.effects.clamp_all();
			app.dirty = true;
		}
		MidiAction::SetGlitchIntensity(intensity) => {
			app.effects.glitch_intensity = intensity;
			app.effects.clamp_all();
			app.dirty = true;
		}
		MidiAction::SetStrobeIntensity(intensity) => {
			app.effects.strobe_intensity = intensity;
			app.effects.clamp_all();
			app.dirty = true;
		}
		MidiAction::SetMasterDimmer(dimmer) => {
			app.effects.master_dimmer = dimmer;
			app.effects.clamp_all();
			app.dirty = true;
		}
		MidiAction::SetBrightness(brightness) => {
			app.effects.brightness = brightness;
			app.effects.clamp_all();
			app.dirty = true;
		}
		MidiAction::SetContrast(contrast) => {
			app.effects.contrast = contrast;
			app.effects.clamp_all();
			app.dirty = true;
		}
		MidiAction::SetSaturation(saturation) => {
			app.effects.saturation = saturation;
			app.effects.clamp_all();
			app.dirty = true;
		}
		MidiAction::SetStrobeSpeedHz(hz) => {
			app.effects.strobe_speed_hz = hz;
			app.effects.clamp_all();
			app.dirty = true;
		}
		MidiAction::ToggleRgbShift => {
			app.effects.rgb_shift_enabled = !app.effects.rgb_shift_enabled;
			app.dirty = true;
		}
		MidiAction::ToggleGlitch => {
			app.effects.glitch_enabled = !app.effects.glitch_enabled;
			app.dirty = true;
		}
		MidiAction::ToggleStrobe => {
			app.effects.strobe_enabled = !app.effects.strobe_enabled;
			app.dirty = true;
		}
		MidiAction::ToggleBlackout => {
			app.effects.blackout = !app.effects.blackout;
			app.dirty = true;
		}
		MidiAction::ResetAllEffects => {
			app.effects.reset_effects();
			app.dirty = true;
		}
		MidiAction::RetriggerAllActiveLayers => {
			for layer in LayerIndex::ALL {
				if app.retrigger_layer(layer) {
					result.mark(layer);
				}
			}
		}
		MidiAction::ToggleFocusedLayerPlaybackMode => {
			let focused = app.focused_layer;
			app.cycle_playback_mode(focused);
		}
		MidiAction::ResetAllLayerSpeeds => {
			for layer in LayerIndex::ALL {
				app.set_layer_speed(layer, 1.0);
			}
		}
	}

	result
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use std::path::PathBuf;

	use super::*;
	use crate::model::clip::{ClipId, ClipInfo, ClipKind, ClipStatus};
	use crate::model::layer::{LAYER_COUNT, PlaybackMode, PlaybackState};

	fn cc(controller: u8, value: u8) -> Option<MidiAction> {
		parse_midi_message(&[0xB0, controller, value])
	}

	fn note(number: u8, velocity: u8) -> Option<MidiAction> {
		parse_midi_message(&[0x90, number, velocity])
	}

	fn layer(offset: u8) -> LayerIndex {
		LayerIndex::new(usize::from(offset)).unwrap()
	}

	fn approx(actual: f32, expected: f32) {
		assert!((actual - expected).abs() < 1e-5, "expected {expected}, got {actual}");
	}

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

	fn state_with_clips(count: u64) -> AppState {
		let mut state = AppState::default();
		for id in 1..=count {
			state.upsert_clip(clip(id));
		}
		state
	}

	// ----------------------------------------------------------------- conv

	#[test]
	fn every_cc_number_maps_per_spec() {
		for controller in 0..=127u8 {
			let value = 100u8;
			let expected = match controller {
				1..=4 => Some(MidiAction::SetLayerOpacity {
					layer: layer(controller - 1),
					opacity: cc_to_unit(value),
				}),
				5 => Some(MidiAction::SetRgbShiftIntensity(cc_to_unit(value))),
				6 => Some(MidiAction::SetGlitchIntensity(cc_to_unit(value))),
				7 => Some(MidiAction::SetStrobeIntensity(cc_to_unit(value))),
				8 => Some(MidiAction::SetMasterDimmer(cc_to_unit(value))),
				11..=14 => Some(MidiAction::SetLayerSpeed {
					layer: layer(controller - 11),
					speed: LayerState::speed_from_midi_cc(value),
				}),
				15 => Some(MidiAction::SetBrightness(cc_to_brightness(value))),
				16 => Some(MidiAction::SetContrast(cc_to_factor_centered(value))),
				17 => Some(MidiAction::SetSaturation(cc_to_factor_centered(value))),
				18 => Some(MidiAction::SetStrobeSpeedHz(cc_to_strobe_hz(value))),
				21..=24 => Some(MidiAction::SetLayerBlendMode {
					layer: layer(controller - 21),
					blend_mode: BlendMode::from_midi_cc(value),
				}),
				31..=34 => Some(MidiAction::SelectLayerClip {
					layer: layer(controller - 31),
					cc_value: value,
				}),
				_ => None,
			};
			assert_eq!(cc(controller, value), expected, "cc {controller}");
		}
	}

	#[test]
	fn cc_layer_groups_address_every_layer() {
		for offset in 0..LAYER_COUNT as u8 {
			assert_eq!(
				cc(1 + offset, 64),
				Some(MidiAction::SetLayerOpacity {
					layer: layer(offset),
					opacity: cc_to_unit(64),
				})
			);
			assert_eq!(
				cc(11 + offset, 64),
				Some(MidiAction::SetLayerSpeed {
					layer: layer(offset),
					speed: LayerState::speed_from_midi_cc(64),
				})
			);
			assert_eq!(
				cc(21 + offset, 64),
				Some(MidiAction::SetLayerBlendMode {
					layer: layer(offset),
					blend_mode: BlendMode::from_midi_cc(64),
				})
			);
			assert_eq!(
				cc(31 + offset, 64),
				Some(MidiAction::SelectLayerClip {
					layer: layer(offset),
					cc_value: 64,
				})
			);
		}
	}

	#[test]
	fn cc_boundary_values_are_converted_and_clamped() {
		for value in [0u8, 1, 32, 63, 64, 65, 96, 126, 127, 200, 255] {
			assert_eq!(
				cc(1, value),
				Some(MidiAction::SetLayerOpacity {
					layer: layer(0),
					opacity: cc_to_unit(value),
				})
			);
			assert_eq!(
				cc(11, value),
				Some(MidiAction::SetLayerSpeed {
					layer: layer(0),
					speed: LayerState::speed_from_midi_cc(value),
				})
			);
			assert_eq!(
				cc(21, value),
				Some(MidiAction::SetLayerBlendMode {
					layer: layer(0),
					blend_mode: BlendMode::from_midi_cc(value),
				})
			);
			// Above 127 the controller value is clamped back into the 7-bit range.
			assert_eq!(
				cc(31, value),
				Some(MidiAction::SelectLayerClip {
					layer: layer(0),
					cc_value: value.min(127),
				})
			);
		}
	}

	#[test]
	fn opacity_speed_and_blend_anchors_match_the_spec() {
		assert_eq!(
			cc(1, 0),
			Some(MidiAction::SetLayerOpacity {
				layer: layer(0),
				opacity: 0.0,
			})
		);
		assert_eq!(
			cc(4, 127),
			Some(MidiAction::SetLayerOpacity {
				layer: layer(3),
				opacity: 1.0,
			})
		);
		assert_eq!(
			cc(11, 64),
			Some(MidiAction::SetLayerSpeed {
				layer: layer(0),
				speed: 1.0,
			})
		);
		assert_eq!(
			cc(14, 0),
			Some(MidiAction::SetLayerSpeed {
				layer: layer(3),
				speed: -2.0,
			})
		);
		assert_eq!(
			cc(24, 127),
			Some(MidiAction::SetLayerBlendMode {
				layer: layer(3),
				blend_mode: BlendMode::Difference,
			})
		);
	}

	#[test]
	fn reserved_cc_numbers_are_unassigned() {
		for controller in [
			0u8, 9, 10, 19, 20, 25, 26, 27, 28, 29, 30, 35, 36, 37, 38, 39, 64, 100, 127,
		] {
			assert_eq!(cc(controller, 64), None, "cc {controller} must stay unassigned");
		}
	}

	#[test]
	fn every_note_number_maps_per_spec() {
		for number in 0..=127u8 {
			let expected = match number {
				36..=39 => Some(MidiAction::TriggerLayer(layer(number - 36))),
				44..=47 => Some(MidiAction::TogglePauseLayer(layer(number - 44))),
				40 => Some(MidiAction::ToggleRgbShift),
				41 => Some(MidiAction::ToggleGlitch),
				42 => Some(MidiAction::ToggleStrobe),
				43 => Some(MidiAction::ToggleBlackout),
				48 => Some(MidiAction::ResetAllEffects),
				49 => Some(MidiAction::RetriggerAllActiveLayers),
				50 => Some(MidiAction::ToggleFocusedLayerPlaybackMode),
				51 => Some(MidiAction::ResetAllLayerSpeeds),
				_ => None,
			};
			assert_eq!(note(number, 127), expected, "note {number}");
			assert_eq!(note(number, 1), expected, "note {number} with minimal velocity");
			assert_eq!(note(number, 0), None, "note {number} release must be ignored");
		}
	}

	#[test]
	fn note_layer_groups_address_every_layer() {
		for offset in 0..LAYER_COUNT as u8 {
			assert_eq!(note(36 + offset, 100), Some(MidiAction::TriggerLayer(layer(offset))));
			assert_eq!(
				note(44 + offset, 100),
				Some(MidiAction::TogglePauseLayer(layer(offset)))
			);
		}
	}

	#[test]
	fn note_off_and_velocity_zero_are_ignored() {
		assert_eq!(parse_midi_message(&[0x80, 36, 127]), None);
		assert_eq!(parse_midi_message(&[0x80, 40, 127]), None);
		assert_eq!(parse_midi_message(&[0x90, 36, 0]), None);
		assert_eq!(parse_midi_message(&[0x90, 51, 0]), None);
	}

	#[test]
	fn non_channel_one_messages_are_ignored() {
		for channel in 1..=15u8 {
			assert_eq!(parse_midi_message(&[0xB0 | channel, 1, 64]), None, "cc ch {channel}");
			assert_eq!(
				parse_midi_message(&[0x90 | channel, 36, 127]),
				None,
				"note ch {channel}"
			);
			assert_eq!(
				parse_midi_message(&[0x80 | channel, 36, 127]),
				None,
				"note off ch {channel}"
			);
		}
		assert_eq!(parse_midi_message(&[0xB0, 1, 64]), cc(1, 64));
		assert_eq!(
			parse_midi_message(&[0x90, 36, 127]),
			Some(MidiAction::TriggerLayer(layer(0)))
		);
	}

	#[test]
	fn malformed_and_unsupported_messages_are_ignored() {
		assert_eq!(parse_midi_message(&[]), None);
		assert_eq!(parse_midi_message(&[0xB0]), None);
		assert_eq!(parse_midi_message(&[0xB0, 1]), None);
		assert_eq!(parse_midi_message(&[0x90]), None);
		assert_eq!(parse_midi_message(&[0x90, 36]), None);
		assert_eq!(parse_midi_message(&[0xF8]), None);
		assert_eq!(parse_midi_message(&[0xF0, 0x7E, 0x7F]), None);
		assert_eq!(parse_midi_message(&[0xE0, 0x00, 0x40]), None);
		assert_eq!(parse_midi_message(&[0xC0, 0x05, 0x00]), None);
		assert_eq!(parse_midi_message(&[0xA0, 36, 127]), None);
	}

	#[test]
	fn longer_packets_use_the_first_three_bytes() {
		assert_eq!(parse_midi_message(&[0xB0, 1, 64, 0, 0]), cc(1, 64));
		assert_eq!(parse_midi_message(&[0x90, 37, 100, 1]), note(37, 100));
	}

	// ------------------------------------------------------- resolve_clip_index

	#[test]
	fn resolve_clip_index_rejects_empty_library() {
		assert_eq!(resolve_clip_index(0, 0), None);
		assert_eq!(resolve_clip_index(127, 0), None);
	}

	#[test]
	fn resolve_clip_index_spreads_small_libraries_evenly() {
		assert_eq!(resolve_clip_index(0, 1), Some(0));
		assert_eq!(resolve_clip_index(127, 1), Some(0));

		for value in 0..=63u8 {
			assert_eq!(resolve_clip_index(value, 2), Some(0), "cc {value}");
		}
		for value in 64..=127u8 {
			assert_eq!(resolve_clip_index(value, 2), Some(1), "cc {value}");
		}

		assert_eq!(resolve_clip_index(0, 4), Some(0));
		assert_eq!(resolve_clip_index(31, 4), Some(0));
		assert_eq!(resolve_clip_index(32, 4), Some(1));
		assert_eq!(resolve_clip_index(64, 4), Some(2));
		assert_eq!(resolve_clip_index(96, 4), Some(3));
		assert_eq!(resolve_clip_index(127, 4), Some(3));

		assert_eq!(resolve_clip_index(0, 5), Some(0));
		assert_eq!(resolve_clip_index(25, 5), Some(0));
		assert_eq!(resolve_clip_index(26, 5), Some(1));
		assert_eq!(resolve_clip_index(127, 5), Some(4));
	}

	#[test]
	fn resolve_clip_index_covers_exactly_128_clips() {
		for value in 0..=127u8 {
			assert_eq!(resolve_clip_index(value, 128), Some(usize::from(value)));
		}
	}

	#[test]
	fn resolve_clip_index_addresses_large_libraries_directly() {
		assert_eq!(resolve_clip_index(0, 200), Some(0));
		assert_eq!(resolve_clip_index(127, 200), Some(127));
		assert_eq!(resolve_clip_index(200, 200), Some(127));
		assert_eq!(resolve_clip_index(0, 129), Some(0));
		assert_eq!(resolve_clip_index(127, 129), Some(127));
	}

	// --------------------------------------------------------- apply_midi_action

	#[test]
	fn apply_layer_parameter_actions() {
		let mut state = AppState::default();

		apply_midi_action(
			&mut state,
			MidiAction::SetLayerOpacity {
				layer: layer(1),
				opacity: 0.25,
			},
		);
		approx(state.layer(layer(1)).opacity, 0.25);

		apply_midi_action(
			&mut state,
			MidiAction::SetLayerSpeed {
				layer: layer(2),
				speed: -1.5,
			},
		);
		approx(state.layer(layer(2)).speed, -1.5);

		apply_midi_action(
			&mut state,
			MidiAction::SetLayerBlendMode {
				layer: layer(3),
				blend_mode: BlendMode::Screen,
			},
		);
		assert_eq!(state.layer(layer(3)).blend_mode, BlendMode::Screen);
		assert!(state.dirty);

		// Out-of-range values are clamped instead of corrupting the state.
		apply_midi_action(
			&mut state,
			MidiAction::SetLayerOpacity {
				layer: layer(0),
				opacity: 5.0,
			},
		);
		approx(state.layer(layer(0)).opacity, 1.0);
		apply_midi_action(
			&mut state,
			MidiAction::SetLayerSpeed {
				layer: layer(0),
				speed: f32::NAN,
			},
		);
		approx(state.layer(layer(0)).speed, 1.0);
	}

	#[test]
	fn apply_select_layer_clip_uses_the_library_size() {
		let mut state = state_with_clips(4);
		apply_midi_action(
			&mut state,
			MidiAction::SelectLayerClip {
				layer: layer(0),
				cc_value: 127,
			},
		);
		assert_eq!(state.layer(layer(0)).selected_clip, Some(ClipId(4)));
		assert!(state.layer(layer(0)).active_clip.is_none());

		apply_midi_action(
			&mut state,
			MidiAction::SelectLayerClip {
				layer: layer(1),
				cc_value: 0,
			},
		);
		assert_eq!(state.layer(layer(1)).selected_clip, Some(ClipId(1)));

		// An empty library leaves the selection untouched.
		let mut empty = AppState::default();
		apply_midi_action(
			&mut empty,
			MidiAction::SelectLayerClip {
				layer: layer(0),
				cc_value: 64,
			},
		);
		assert_eq!(empty.layer(layer(0)).selected_clip, None);
	}

	#[test]
	fn apply_trigger_layer_starts_and_then_retriggers() {
		let mut state = state_with_clips(2);
		state.select_clip_by_index(layer(0), 1);

		let result = apply_midi_action(&mut state, MidiAction::TriggerLayer(layer(0)));
		assert!(result.any());
		assert!(result.contains(layer(0)));
		assert!(!result.contains(layer(1)));
		assert_eq!(result.layers().count(), 1);
		assert_eq!(state.layer(layer(0)).active_clip, Some(ClipId(2)));
		assert_eq!(state.layer(layer(0)).playback_state, PlaybackState::Playing);

		state.advance_layers(0.5);
		assert!(state.layer(layer(0)).playhead_frame > 0.0);

		let result = apply_midi_action(&mut state, MidiAction::TriggerLayer(layer(0)));
		assert!(result.contains(layer(0)));
		approx(state.layer(layer(0)).playhead_frame as f32, 0.0);
	}

	#[test]
	fn apply_trigger_layer_without_selection_is_safe() {
		let mut state = state_with_clips(1);
		let result = apply_midi_action(&mut state, MidiAction::TriggerLayer(layer(0)));
		assert!(!result.any());
		assert!(state.layer(layer(0)).active_clip.is_none());

		let empty = apply_midi_action(&mut AppState::default(), MidiAction::TriggerLayer(layer(3)));
		assert!(!empty.any());
	}

	#[test]
	fn apply_toggle_pause_layer_switches_transport() {
		let mut state = state_with_clips(1);
		state.select_clip_by_index(layer(0), 0);
		apply_midi_action(&mut state, MidiAction::TriggerLayer(layer(0)));

		let result = apply_midi_action(&mut state, MidiAction::TogglePauseLayer(layer(0)));
		assert!(!result.any());
		assert_eq!(state.layer(layer(0)).playback_state, PlaybackState::Paused);

		apply_midi_action(&mut state, MidiAction::TogglePauseLayer(layer(0)));
		assert_eq!(state.layer(layer(0)).playback_state, PlaybackState::Playing);
	}

	#[test]
	fn apply_effect_intensity_actions_clamp_values() {
		let mut state = AppState::default();

		apply_midi_action(&mut state, MidiAction::SetRgbShiftIntensity(0.4));
		approx(state.effects.rgb_shift_intensity, 0.4);
		apply_midi_action(&mut state, MidiAction::SetGlitchIntensity(0.6));
		approx(state.effects.glitch_intensity, 0.6);
		apply_midi_action(&mut state, MidiAction::SetStrobeIntensity(0.8));
		approx(state.effects.strobe_intensity, 0.8);
		apply_midi_action(&mut state, MidiAction::SetMasterDimmer(0.5));
		approx(state.effects.master_dimmer, 0.5);

		apply_midi_action(&mut state, MidiAction::SetRgbShiftIntensity(9.0));
		approx(state.effects.rgb_shift_intensity, 1.0);
		apply_midi_action(&mut state, MidiAction::SetGlitchIntensity(-3.0));
		approx(state.effects.glitch_intensity, 0.0);
		apply_midi_action(&mut state, MidiAction::SetStrobeIntensity(f32::NAN));
		approx(state.effects.strobe_intensity, 0.0);
		apply_midi_action(&mut state, MidiAction::SetMasterDimmer(f32::INFINITY));
		approx(state.effects.master_dimmer, 1.0);
	}

	#[test]
	fn apply_color_actions_clamp_values() {
		let mut state = AppState::default();

		apply_midi_action(&mut state, MidiAction::SetBrightness(-0.5));
		approx(state.effects.brightness, -0.5);
		apply_midi_action(&mut state, MidiAction::SetContrast(1.5));
		approx(state.effects.contrast, 1.5);
		apply_midi_action(&mut state, MidiAction::SetSaturation(0.25));
		approx(state.effects.saturation, 0.25);
		apply_midi_action(&mut state, MidiAction::SetStrobeSpeedHz(12.0));
		approx(state.effects.strobe_speed_hz, 12.0);

		apply_midi_action(&mut state, MidiAction::SetBrightness(-9.0));
		approx(state.effects.brightness, -1.0);
		apply_midi_action(&mut state, MidiAction::SetContrast(f32::INFINITY));
		approx(state.effects.contrast, 1.0);
		apply_midi_action(&mut state, MidiAction::SetSaturation(9.0));
		approx(state.effects.saturation, 2.0);
		apply_midi_action(&mut state, MidiAction::SetStrobeSpeedHz(0.0));
		approx(state.effects.strobe_speed_hz, 1.0);
	}

	#[test]
	fn apply_toggles_flip_one_effect_each() {
		let mut state = AppState::default();

		apply_midi_action(&mut state, MidiAction::ToggleRgbShift);
		assert!(state.effects.rgb_shift_enabled);
		assert!(!state.effects.glitch_enabled);
		apply_midi_action(&mut state, MidiAction::ToggleRgbShift);
		assert!(!state.effects.rgb_shift_enabled);

		apply_midi_action(&mut state, MidiAction::ToggleGlitch);
		assert!(state.effects.glitch_enabled);
		apply_midi_action(&mut state, MidiAction::ToggleStrobe);
		assert!(state.effects.strobe_enabled);
		apply_midi_action(&mut state, MidiAction::ToggleBlackout);
		assert!(state.effects.blackout);
		apply_midi_action(&mut state, MidiAction::ToggleBlackout);
		assert!(!state.effects.blackout);
	}

	#[test]
	fn apply_reset_all_effects_preserves_live_safety_fields() {
		let mut state = AppState::default();
		state.effects.rgb_shift_enabled = true;
		state.effects.rgb_shift_intensity = 0.9;
		state.effects.brightness = 0.5;
		state.effects.master_dimmer = 0.4;
		state.effects.blackout = true;

		apply_midi_action(&mut state, MidiAction::ResetAllEffects);

		assert!(!state.effects.rgb_shift_enabled);
		approx(state.effects.rgb_shift_intensity, 0.0);
		approx(state.effects.brightness, 0.0);
		approx(state.effects.master_dimmer, 0.4);
		assert!(state.effects.blackout);
	}

	#[test]
	fn apply_retrigger_all_active_layers_flags_exactly_those_layers() {
		let mut state = state_with_clips(2);
		state.select_clip_by_index(layer(0), 0);
		state.select_clip_by_index(layer(2), 1);
		apply_midi_action(&mut state, MidiAction::TriggerLayer(layer(0)));
		apply_midi_action(&mut state, MidiAction::TriggerLayer(layer(2)));
		state.advance_layers(0.5);

		let result = apply_midi_action(&mut state, MidiAction::RetriggerAllActiveLayers);

		assert_eq!(
			result.retriggered_layers,
			[true, false, true, false],
			"only active layers are restarted"
		);
		assert!(result.contains(layer(0)));
		assert!(!result.contains(layer(1)));
		assert_eq!(result.layers().collect::<Vec<_>>(), vec![layer(0), layer(2)]);
		approx(state.layer(layer(0)).playhead_frame as f32, 0.0);
		approx(state.layer(layer(2)).playhead_frame as f32, 0.0);
	}

	#[test]
	fn apply_toggle_focused_layer_playback_mode() {
		let mut state = AppState {
			focused_layer: layer(2),
			..AppState::default()
		};

		apply_midi_action(&mut state, MidiAction::ToggleFocusedLayerPlaybackMode);
		assert_eq!(state.layer(layer(2)).playback_mode, PlaybackMode::OneShot);

		apply_midi_action(&mut state, MidiAction::ToggleFocusedLayerPlaybackMode);
		assert_eq!(state.layer(layer(2)).playback_mode, PlaybackMode::Loop);
	}

	#[test]
	fn apply_reset_all_layer_speeds_returns_to_real_time() {
		let mut state = AppState::default();
		for layer in LayerIndex::ALL {
			state.set_layer_speed(layer, -2.0);
		}

		apply_midi_action(&mut state, MidiAction::ResetAllLayerSpeeds);

		for layer in LayerIndex::ALL {
			approx(state.layer(layer).speed, 1.0);
		}
	}

	#[test]
	fn parse_then_apply_round_trip() {
		let mut state = state_with_clips(2);
		state.select_clip_by_index(layer(1), 0);

		for bytes in [
			[0xB0u8, 2, 64],
			[0xB0, 12, 0],
			[0xB0, 22, 127],
			[0xB0, 32, 127],
			[0xB0, 8, 64],
			[0xB0, 16, 64],
			[0x90, 39, 127],
			[0x90, 36, 127],
		] {
			let action = parse_midi_message(&bytes).unwrap();
			apply_midi_action(&mut state, action);
		}

		approx(state.layer(layer(1)).opacity, cc_to_unit(64));
		approx(state.layer(layer(1)).speed, -2.0);
		assert_eq!(state.layer(layer(1)).blend_mode, BlendMode::Difference);
		assert_eq!(state.layer(layer(1)).selected_clip, Some(ClipId(2)));
		apply_midi_action(&mut state, parse_midi_message(&[0x90, 37, 127]).unwrap());
		assert_eq!(state.layer(layer(1)).active_clip, Some(ClipId(2)));
		// The layer runs at -2.0x, so a fresh trigger starts on the last frame.
		approx(state.layer(layer(1)).playhead_frame as f32, 29.0);
		apply_midi_action(&mut state, parse_midi_message(&[0x90, 45, 127]).unwrap());
		assert_eq!(state.layer(layer(1)).playback_state, PlaybackState::Paused);
		approx(state.effects.master_dimmer, cc_to_unit(64));
		approx(state.effects.contrast, 1.0);
		assert!(!state.effects.rgb_shift_enabled);
	}

	#[test]
	fn apply_result_defaults_to_no_layers() {
		let result = MidiApplyResult::default();
		assert!(!result.any());
		assert_eq!(result.retriggered_layers, [false; LAYER_COUNT]);
		assert_eq!(result.layers().count(), 0);
		for layer in LayerIndex::ALL {
			assert!(!result.contains(layer));
		}
	}
}
