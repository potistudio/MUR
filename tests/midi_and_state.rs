//! Integration tests for the Launch Control XL 3 mapping, the keyboard fallback,
//! project files, and crash-recovery state restoration.
//!
//! These tests link the exact production mapping (`parse_midi_message` /
//! `apply_midi_action`), the keyboard translation (`map_key_event`), the command
//! layer (`apply_ui_command`), and the TOML/JSON/state loaders together, without
//! opening windows, MIDI ports, or a GPU.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::{Path, PathBuf};

use mur::config::{PersistedLayerState, PersistedState, ProjectConfig, STATE_VERSION, load_state, save_state_atomic};
use mur::media::test_fixtures::TempDir;
use mur::midi::{MidiAction, apply_midi_action, parse_midi_message, resolve_clip_index};
use mur::model::effect::{cc_to_brightness, cc_to_factor_centered, cc_to_strobe_hz, cc_to_unit};
use mur::model::{
	AppState, BlendMode, ClipId, ClipInfo, ClipKind, ClipStatus, LayerIndex, LayerState, MasterEffects,
	OutputDisplaySelection, PlaybackMode, PlaybackState,
};
use mur::ui::{
	UiApplyOutcome, UiCommand, apply_ui_command, map_key_event,
	shortcuts::{COLOR_GRADE_STEP, LAYER_OPACITY_STEP, LAYER_SPEED_STEP, MASTER_DIMMER_STEP, STROBE_SPEED_STEP_HZ},
};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn approx(actual: f32, expected: f32) {
	assert!((actual - expected).abs() < 1e-5, "expected {expected}, got {actual}");
}

fn approx_f64(actual: f64, expected: f64) {
	assert!((actual - expected).abs() < 1e-9, "expected {expected}, got {actual}");
}

fn clip(id: u64, path: &Path) -> ClipInfo {
	ClipInfo {
		id: ClipId(id),
		path: path.to_path_buf(),
		name: path
			.file_name()
			.map(|name| name.to_string_lossy().into_owned())
			.unwrap_or_else(|| format!("{id}.mov")),
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
		state.upsert_clip(clip(id, &PathBuf::from(format!("/clips/{id}.mov"))));
	}
	state
}

fn cc(controller: u8, value: u8) -> Option<MidiAction> {
	parse_midi_message(&[0xB0, controller, value])
}

fn note(number: u8, velocity: u8) -> Option<MidiAction> {
	parse_midi_message(&[0x90, number, velocity])
}

fn by_char(text: &str, code: KeyCode, mods: ModifiersState, focused: LayerIndex) -> Option<UiCommand> {
	map_key_event(
		&Key::Character(text.into()),
		PhysicalKey::Code(code),
		mods,
		focused,
		false,
		false,
	)
}

fn by_named(named: NamedKey, code: KeyCode, mods: ModifiersState, focused: LayerIndex) -> Option<UiCommand> {
	map_key_event(&Key::Named(named), PhysicalKey::Code(code), mods, focused, false, false)
}

fn apply_key(app: &mut AppState, command: Option<UiCommand>) -> UiApplyOutcome {
	let command = command.expect("the key combination must map to a command");
	apply_ui_command(app, &command)
}

const DIGITS: [(KeyCode, u8); 4] = [
	(KeyCode::Digit1, 1),
	(KeyCode::Digit2, 2),
	(KeyCode::Digit3, 3),
	(KeyCode::Digit4, 4),
];

fn layer(number: u8) -> LayerIndex {
	LayerIndex::from_number(number).unwrap()
}

// ---------------------------------------------------------------------------
// Launch Control XL 3: layer CCs
// ---------------------------------------------------------------------------

#[test]
fn launch_control_layer_ccs_cover_every_band() {
	let mut app = state_with_clips(6);
	let all_modes = BlendMode::ALL;
	let blend_values = [0_u8, 25, 26, 51, 52, 76, 77, 102, 103, 127];
	let mut seen_blend = Vec::new();

	for (offset, target) in LayerIndex::ALL.iter().enumerate() {
		let offset = offset as u8;

		// Faders 1-4: opacity 0..127 -> 0.0..1.0.
		for value in [0_u8, 1, 32, 64, 100, 127] {
			let action = cc(1 + offset, value).unwrap();
			assert_eq!(
				action,
				MidiAction::SetLayerOpacity {
					layer: *target,
					opacity: cc_to_unit(value),
				}
			);
			let _ = apply_midi_action(&mut app, action);
			approx(app.layer(*target).opacity, cc_to_unit(value));
		}

		// Encoders row 1: speed -2.0..=4.0, centre detent 64 = 1.0.
		for value in [0_u8, 16, 32, 64, 96, 127] {
			let expected = LayerState::speed_from_midi_cc(value);
			let action = cc(11 + offset, value).unwrap();
			assert_eq!(
				action,
				MidiAction::SetLayerSpeed {
					layer: *target,
					speed: expected,
				}
			);
			let _ = apply_midi_action(&mut app, action);
			approx(app.layer(*target).speed, expected);
		}
		approx(app.layer(*target).speed.clamp(-2.0, 4.0), app.layer(*target).speed);
		approx(LayerState::speed_from_midi_cc(0), -2.0);
		approx(LayerState::speed_from_midi_cc(64), 1.0);
		approx(LayerState::speed_from_midi_cc(127), 4.0);

		// Encoders row 2: five evenly split blend bands.
		for value in blend_values {
			let expected = BlendMode::from_midi_cc(value);
			let action = cc(21 + offset, value).unwrap();
			assert_eq!(
				action,
				MidiAction::SetLayerBlendMode {
					layer: *target,
					blend_mode: expected,
				}
			);
			let _ = apply_midi_action(&mut app, action);
			assert_eq!(app.layer(*target).blend_mode, expected);
			seen_blend.push(expected);
		}

		// Encoders row 3: clip selection across the grid.
		for value in [0_u8, 16, 64, 100, 127] {
			let expected_index = resolve_clip_index(value, app.clips.len()).unwrap();
			let expected_id = app.clips[expected_index].id;
			let action = cc(31 + offset, value).unwrap();
			assert_eq!(
				action,
				MidiAction::SelectLayerClip {
					layer: *target,
					cc_value: value,
				}
			);
			let _ = apply_midi_action(&mut app, action);
			assert_eq!(app.layer(*target).selected_clip, Some(expected_id));
		}
	}

	for mode in all_modes {
		assert!(seen_blend.contains(&mode), "{mode:?} was never reachable from CC");
	}
}

// ---------------------------------------------------------------------------
// Launch Control XL 3: master CCs
// ---------------------------------------------------------------------------

#[test]
fn launch_control_master_ccs_and_reserved_controllers_match_the_spec() {
	let mut app = AppState::default();

	for value in [0_u8, 32, 64, 100, 127] {
		let unit = cc_to_unit(value);
		for (controller, field) in [(5_u8, 0_usize), (6, 1), (7, 2), (8, 3)] {
			let action = cc(controller, value).unwrap();
			let _ = apply_midi_action(&mut app, action);
			let actual = match field {
				0 => app.effects.rgb_shift_intensity,
				1 => app.effects.glitch_intensity,
				2 => app.effects.strobe_intensity,
				_ => app.effects.master_dimmer,
			};
			approx(actual, unit);
		}
	}
	assert_eq!(cc(5, 64), Some(MidiAction::SetRgbShiftIntensity(cc_to_unit(64))));
	assert_eq!(cc(6, 64), Some(MidiAction::SetGlitchIntensity(cc_to_unit(64))));
	assert_eq!(cc(7, 64), Some(MidiAction::SetStrobeIntensity(cc_to_unit(64))));
	assert_eq!(cc(8, 64), Some(MidiAction::SetMasterDimmer(cc_to_unit(64))));

	// Encoders row 1, right strip: colour grade and strobe frequency.
	for value in [0_u8, 32, 64, 96, 127] {
		let brightness = cc_to_brightness(value);
		let contrast = cc_to_factor_centered(value);
		let saturation = cc_to_factor_centered(value);
		let hz = cc_to_strobe_hz(value);

		assert_eq!(cc(15, value), Some(MidiAction::SetBrightness(brightness)));
		let _ = apply_midi_action(&mut app, cc(15, value).unwrap());
		approx(app.effects.brightness, brightness);

		assert_eq!(cc(16, value), Some(MidiAction::SetContrast(contrast)));
		let _ = apply_midi_action(&mut app, cc(16, value).unwrap());
		approx(app.effects.contrast, contrast);

		assert_eq!(cc(17, value), Some(MidiAction::SetSaturation(saturation)));
		let _ = apply_midi_action(&mut app, cc(17, value).unwrap());
		approx(app.effects.saturation, saturation);

		assert_eq!(cc(18, value), Some(MidiAction::SetStrobeSpeedHz(hz)));
		let _ = apply_midi_action(&mut app, cc(18, value).unwrap());
		approx(app.effects.strobe_speed_hz, hz);
	}
	approx(cc_to_brightness(64), 0.0);
	approx(cc_to_factor_centered(64), 1.0);
	approx(cc_to_strobe_hz(0), 1.0);
	approx(cc_to_strobe_hz(127), 30.0);

	for controller in (25_u8..=28).chain(35_u8..=38) {
		for value in [0_u8, 64, 127] {
			assert_eq!(cc(controller, value), None, "CC {controller} must stay unassigned");
		}
	}
}

// ---------------------------------------------------------------------------
// Launch Control XL 3: notes
// ---------------------------------------------------------------------------

#[test]
fn launch_control_note_mapping_covers_triggers_pauses_toggles_and_resets() {
	let mut app = state_with_clips(4);
	for (index, target) in LayerIndex::ALL.iter().enumerate() {
		assert_eq!(note(36 + index as u8, 100), Some(MidiAction::TriggerLayer(*target)));
		app.layer_mut(*target).unwrap().selected_clip = Some(ClipId(index as u64 + 1));
	}

	// Note 36-39: trigger / retrigger.
	for (index, target) in LayerIndex::ALL.iter().enumerate() {
		let action = note(36 + index as u8, 127).unwrap();
		let result = apply_midi_action(&mut app, action);
		assert!(result.contains(*target));
		assert_eq!(app.layer(*target).active_clip, Some(ClipId(index as u64 + 1)));
		assert_eq!(app.layer(*target).playback_state, PlaybackState::Playing);
	}

	// Note 44-47: pause / resume toggles.
	for (index, target) in LayerIndex::ALL.iter().enumerate() {
		assert_eq!(note(44 + index as u8, 100), Some(MidiAction::TogglePauseLayer(*target)));
		let _ = apply_midi_action(&mut app, note(44 + index as u8, 100).unwrap());
		assert_eq!(app.layer(*target).playback_state, PlaybackState::Paused);
		let _ = apply_midi_action(&mut app, note(44 + index as u8, 100).unwrap());
		assert_eq!(app.layer(*target).playback_state, PlaybackState::Playing);
	}

	// Note 40-43: effect and blackout toggles.
	for (number, expected) in [
		(40_u8, MidiAction::ToggleRgbShift),
		(41, MidiAction::ToggleGlitch),
		(42, MidiAction::ToggleStrobe),
		(43, MidiAction::ToggleBlackout),
	] {
		assert_eq!(note(number, 100), Some(expected));
		let before = match number {
			40 => app.effects.rgb_shift_enabled,
			41 => app.effects.glitch_enabled,
			42 => app.effects.strobe_enabled,
			_ => app.effects.blackout,
		};
		let _ = apply_midi_action(&mut app, note(number, 100).unwrap());
		let after = match number {
			40 => app.effects.rgb_shift_enabled,
			41 => app.effects.glitch_enabled,
			42 => app.effects.strobe_enabled,
			_ => app.effects.blackout,
		};
		assert_eq!(after, !before, "note {number} must toggle");
		let _ = apply_midi_action(&mut app, note(number, 100).unwrap());
	}

	// Note 48: reset all effects, preserving dimmer and blackout.
	app.effects.rgb_shift_enabled = true;
	app.effects.rgb_shift_intensity = 0.9;
	app.effects.glitch_enabled = true;
	app.effects.strobe_enabled = true;
	app.effects.brightness = 0.5;
	app.effects.contrast = 1.8;
	app.effects.saturation = 0.1;
	app.effects.master_dimmer = 0.4;
	app.effects.blackout = true;
	assert_eq!(note(48, 100), Some(MidiAction::ResetAllEffects));
	let _ = apply_midi_action(&mut app, note(48, 100).unwrap());
	assert!(!app.effects.rgb_shift_enabled);
	assert!(!app.effects.glitch_enabled);
	assert!(!app.effects.strobe_enabled);
	approx(app.effects.rgb_shift_intensity, 0.0);
	approx(app.effects.brightness, 0.0);
	approx(app.effects.contrast, 1.0);
	approx(app.effects.saturation, 1.0);
	approx(app.effects.master_dimmer, 0.4);
	assert!(app.effects.blackout);

	// Note 49: retrigger every active layer.
	app.effects.master_dimmer = 1.0;
	app.effects.blackout = false;
	app.advance_layers(0.5);
	assert_eq!(note(49, 100), Some(MidiAction::RetriggerAllActiveLayers));
	let result = apply_midi_action(&mut app, note(49, 100).unwrap());
	for target in LayerIndex::ALL {
		assert!(result.contains(target));
		assert_eq!(app.layer(target).playhead_frame, 0.0);
	}

	// Note 50: focused layer loop / one-shot.
	app.focused_layer = LayerIndex::ALL[2];
	app.layer_mut(app.focused_layer).unwrap().playback_mode = PlaybackMode::Loop;
	assert_eq!(note(50, 100), Some(MidiAction::ToggleFocusedLayerPlaybackMode));
	let _ = apply_midi_action(&mut app, note(50, 100).unwrap());
	assert_eq!(app.layer(app.focused_layer).playback_mode, PlaybackMode::OneShot);
	let _ = apply_midi_action(&mut app, note(50, 100).unwrap());
	assert_eq!(app.layer(app.focused_layer).playback_mode, PlaybackMode::Loop);

	// Note 51: reset every layer speed to 1.0x.
	for target in LayerIndex::ALL {
		app.set_layer_speed(target, 3.0);
	}
	assert_eq!(note(51, 100), Some(MidiAction::ResetAllLayerSpeeds));
	let _ = apply_midi_action(&mut app, note(51, 100).unwrap());
	for target in LayerIndex::ALL {
		approx(app.layer(target).speed, 1.0);
	}
}

#[test]
fn non_channel_one_and_malformed_midi_packets_are_ignored() {
	assert_eq!(parse_midi_message(&[]), None);
	assert_eq!(parse_midi_message(&[0xB0]), None);
	assert_eq!(parse_midi_message(&[0xB0, 1]), None);
	assert_eq!(parse_midi_message(&[0xB1, 1, 100]), None);
	assert_eq!(parse_midi_message(&[0x91, 36, 100]), None);
	assert_eq!(parse_midi_message(&[0x80, 36, 0]), None);
	assert_eq!(parse_midi_message(&[0x90, 36, 0]), None);
	assert_eq!(parse_midi_message(&[0xF8, 0, 0]), None);
	assert_eq!(parse_midi_message(&[0xB0, 25, 100]), None);

	for number in 36_u8..=51 {
		assert_eq!(note(number, 0), None, "note {number} release must be ignored");
	}
	for controller in 0_u8..=127 {
		if !matches!(controller, 1..=8 | 11..=18 | 21..=24 | 31..=34) {
			assert_eq!(cc(controller, 100), None, "CC {controller} must be unassigned");
		}
	}
}

// ---------------------------------------------------------------------------
// Keyboard fallback: layer controls
// ---------------------------------------------------------------------------

#[test]
fn keyboard_fallback_drives_every_layer_control() {
	let mut app = state_with_clips(4);
	for (index, target) in LayerIndex::ALL.iter().enumerate() {
		app.layer_mut(*target).unwrap().selected_clip = Some(ClipId(index as u64 + 1));
	}

	// Plain digits focus layers 1-4.
	for (code, number) in DIGITS {
		let expected = UiCommand::FocusLayer(layer(number));
		let command = by_char(&number.to_string(), code, ModifiersState::empty(), app.focused_layer);
		assert_eq!(command, Some(expected));
		apply_key(&mut app, command);
		assert_eq!(app.focused_layer, layer(number));
	}

	// Shift + digits trigger layers 1-4.
	for (code, number) in DIGITS {
		let expected = UiCommand::TriggerLayer(layer(number));
		let command = by_char(&number.to_string(), code, ModifiersState::SHIFT, app.focused_layer);
		assert_eq!(command, Some(expected));
		let outcome = apply_key(&mut app, command);
		assert!(outcome.was_retriggered(layer(number)));
		assert_eq!(app.layer(layer(number)).active_clip, Some(ClipId(u64::from(number))));
		assert_eq!(app.layer(layer(number)).playback_state, PlaybackState::Playing);
	}

	// Alt + digits pause / resume.
	for (code, number) in DIGITS {
		let command = by_char(&number.to_string(), code, ModifiersState::ALT, app.focused_layer);
		assert_eq!(command, Some(UiCommand::TogglePauseLayer(layer(number))));
		apply_key(&mut app, command);
		assert_eq!(app.layer(layer(number)).playback_state, PlaybackState::Paused);
		let focused = app.focused_layer;
		apply_key(
			&mut app,
			by_char(&number.to_string(), code, ModifiersState::ALT, focused),
		);
		assert_eq!(app.layer(layer(number)).playback_state, PlaybackState::Playing);
	}

	// Tab / Shift + Tab cycle focus.
	app.focused_layer = LayerIndex::ALL[0];
	let command = by_named(NamedKey::Tab, KeyCode::Tab, ModifiersState::empty(), app.focused_layer);
	assert_eq!(command, Some(UiCommand::CycleFocusLayer(1)));
	apply_key(&mut app, command);
	assert_eq!(app.focused_layer, LayerIndex::ALL[1]);
	let command = by_named(NamedKey::Tab, KeyCode::Tab, ModifiersState::SHIFT, app.focused_layer);
	assert_eq!(command, Some(UiCommand::CycleFocusLayer(-1)));
	apply_key(&mut app, command);
	assert_eq!(app.focused_layer, LayerIndex::ALL[0]);

	// R / Shift + R retrigger one or all layers.
	app.advance_layers(0.5);
	let command = by_char("r", KeyCode::KeyR, ModifiersState::empty(), app.focused_layer);
	assert_eq!(command, Some(UiCommand::RetriggerLayer(app.focused_layer)));
	apply_key(&mut app, command);
	assert_eq!(app.layer(app.focused_layer).playhead_frame, 0.0);

	app.advance_layers(0.5);
	let command = by_char("r", KeyCode::KeyR, ModifiersState::SHIFT, app.focused_layer);
	assert_eq!(command, Some(UiCommand::RetriggerAllActiveLayers));
	let outcome = apply_key(&mut app, command);
	for target in LayerIndex::ALL {
		assert!(outcome.was_retriggered(target));
		assert_eq!(app.layer(target).playhead_frame, 0.0);
	}

	// Space pauses / resumes, L toggles the playback mode, X clears the layer.
	let command = by_named(
		NamedKey::Space,
		KeyCode::Space,
		ModifiersState::empty(),
		app.focused_layer,
	);
	assert_eq!(command, Some(UiCommand::TogglePauseLayer(app.focused_layer)));
	apply_key(&mut app, command.clone());
	assert_eq!(app.layer(app.focused_layer).playback_state, PlaybackState::Paused);
	apply_key(&mut app, command);
	assert_eq!(app.layer(app.focused_layer).playback_state, PlaybackState::Playing);

	let command = by_char("l", KeyCode::KeyL, ModifiersState::empty(), app.focused_layer);
	assert_eq!(command, Some(UiCommand::ToggleLayerPlaybackMode(app.focused_layer)));
	apply_key(&mut app, command);
	assert_eq!(app.layer(app.focused_layer).playback_mode, PlaybackMode::OneShot);

	// Q / W adjust opacity in 0.05 steps and clamp.
	app.set_layer_opacity(app.focused_layer, 1.0);
	let command = by_char("q", KeyCode::KeyQ, ModifiersState::empty(), app.focused_layer);
	assert_eq!(
		command,
		Some(UiCommand::AdjustLayerOpacity {
			layer: app.focused_layer,
			delta: -LAYER_OPACITY_STEP,
		})
	);
	apply_key(&mut app, command);
	approx(app.layer(app.focused_layer).opacity, 1.0 - LAYER_OPACITY_STEP);
	let focused = app.focused_layer;
	apply_key(&mut app, by_char("w", KeyCode::KeyW, ModifiersState::empty(), focused));
	approx(app.layer(app.focused_layer).opacity, 1.0);
	app.set_layer_opacity(app.focused_layer, 0.02);
	let focused = app.focused_layer;
	apply_key(&mut app, by_char("q", KeyCode::KeyQ, ModifiersState::empty(), focused));
	approx(app.layer(app.focused_layer).opacity, 0.0);

	// A / D adjust speed in 0.1 steps and clamp.
	app.set_layer_speed(app.focused_layer, 1.0);
	let command = by_char("a", KeyCode::KeyA, ModifiersState::empty(), app.focused_layer);
	assert_eq!(
		command,
		Some(UiCommand::AdjustLayerSpeed {
			layer: app.focused_layer,
			delta: -LAYER_SPEED_STEP,
		})
	);
	apply_key(&mut app, command);
	approx(app.layer(app.focused_layer).speed, 1.0 - LAYER_SPEED_STEP);
	let focused = app.focused_layer;
	apply_key(&mut app, by_char("d", KeyCode::KeyD, ModifiersState::empty(), focused));
	app.set_layer_speed(app.focused_layer, -2.0);
	let focused = app.focused_layer;
	apply_key(&mut app, by_char("a", KeyCode::KeyA, ModifiersState::empty(), focused));
	approx(app.layer(app.focused_layer).speed, -2.0);

	// S / Shift + S reset one or every layer speed.
	app.set_layer_speed(app.focused_layer, 2.5);
	let command = by_char("s", KeyCode::KeyS, ModifiersState::empty(), app.focused_layer);
	assert_eq!(
		command,
		Some(UiCommand::SetLayerSpeed {
			layer: app.focused_layer,
			speed: 1.0,
		})
	);
	apply_key(&mut app, command);
	approx(app.layer(app.focused_layer).speed, 1.0);
	for target in LayerIndex::ALL {
		app.set_layer_speed(target, 3.0);
	}
	let command = by_char("s", KeyCode::KeyS, ModifiersState::SHIFT, app.focused_layer);
	assert_eq!(command, Some(UiCommand::ResetAllLayerSpeeds));
	apply_key(&mut app, command);
	for target in LayerIndex::ALL {
		approx(app.layer(target).speed, 1.0);
	}

	// B / Shift + B cycle blend modes.
	app.layer_mut(app.focused_layer).unwrap().blend_mode = BlendMode::Normal;
	let command = by_char("b", KeyCode::KeyB, ModifiersState::empty(), app.focused_layer);
	assert_eq!(
		command,
		Some(UiCommand::CycleLayerBlendMode {
			layer: app.focused_layer,
			forward: true,
		})
	);
	apply_key(&mut app, command);
	assert_eq!(app.layer(app.focused_layer).blend_mode, BlendMode::Add);
	let command = by_char("b", KeyCode::KeyB, ModifiersState::SHIFT, app.focused_layer);
	assert_eq!(
		command,
		Some(UiCommand::CycleLayerBlendMode {
			layer: app.focused_layer,
			forward: false,
		})
	);
	apply_key(&mut app, command);
	assert_eq!(app.layer(app.focused_layer).blend_mode, BlendMode::Normal);

	// [ / ] step through the clip grid.
	app.layer_mut(app.focused_layer).unwrap().selected_clip = None;
	let command = by_char("]", KeyCode::BracketRight, ModifiersState::empty(), app.focused_layer);
	assert_eq!(
		command,
		Some(UiCommand::StepLayerClip {
			layer: app.focused_layer,
			delta: 1,
		})
	);
	apply_key(&mut app, command);
	assert_eq!(app.layer(app.focused_layer).selected_clip, Some(ClipId(1)));
	let command = by_char("[", KeyCode::BracketLeft, ModifiersState::empty(), app.focused_layer);
	assert_eq!(
		command,
		Some(UiCommand::StepLayerClip {
			layer: app.focused_layer,
			delta: -1,
		})
	);
	apply_key(&mut app, command);
	assert_eq!(app.layer(app.focused_layer).selected_clip, Some(ClipId(4)));

	// Enter fires the selected clip on the focused layer.
	app.layer_mut(app.focused_layer).unwrap().selected_clip = Some(ClipId(2));
	let command = by_named(
		NamedKey::Enter,
		KeyCode::Enter,
		ModifiersState::empty(),
		app.focused_layer,
	);
	assert_eq!(command, Some(UiCommand::TriggerSelectedClip(app.focused_layer)));
	let outcome = apply_key(&mut app, command);
	assert!(outcome.was_retriggered(app.focused_layer));
	assert_eq!(app.layer(app.focused_layer).active_clip, Some(ClipId(2)));

	// X clears the focused layer.
	let command = by_char("x", KeyCode::KeyX, ModifiersState::empty(), app.focused_layer);
	assert_eq!(command, Some(UiCommand::ClearLayer(app.focused_layer)));
	apply_key(&mut app, command);
	assert_eq!(app.layer(app.focused_layer).active_clip, None);
	assert_eq!(app.layer(app.focused_layer).playback_state, PlaybackState::Stopped);
}

// ---------------------------------------------------------------------------
// Keyboard fallback: master effects, output, overlays
// ---------------------------------------------------------------------------

#[test]
fn keyboard_fallback_drives_master_effects_output_and_overlays() {
	let mut app = AppState::default();
	let focused = app.focused_layer;

	// F1-F3 toggle the three shader effects.
	for (named, code, field) in [
		(NamedKey::F1, KeyCode::F1, 0_u8),
		(NamedKey::F2, KeyCode::F2, 1),
		(NamedKey::F3, KeyCode::F3, 2),
	] {
		let command = by_named(named, code, ModifiersState::empty(), focused);
		assert_eq!(
			command,
			Some(match field {
				0 => UiCommand::ToggleRgbShift,
				1 => UiCommand::ToggleGlitch,
				_ => UiCommand::ToggleStrobe,
			})
		);
		apply_key(&mut app, command);
		let enabled = match field {
			0 => app.effects.rgb_shift_enabled,
			1 => app.effects.glitch_enabled,
			_ => app.effects.strobe_enabled,
		};
		assert!(enabled);
	}

	// Shift / Alt + F1-F3 adjust the matching intensity by 0.1.
	for (named, code, base) in [
		(NamedKey::F1, KeyCode::F1, 0.0_f32),
		(NamedKey::F2, KeyCode::F2, 0.0),
		(NamedKey::F3, KeyCode::F3, 0.0),
	] {
		let command = by_named(named, code, ModifiersState::SHIFT, focused);
		assert_eq!(
			command,
			Some(match code {
				KeyCode::F1 => UiCommand::AdjustRgbShiftIntensity(0.1),
				KeyCode::F2 => UiCommand::AdjustGlitchIntensity(0.1),
				_ => UiCommand::AdjustStrobeIntensity(0.1),
			})
		);
		apply_key(&mut app, command);
		let increased = match code {
			KeyCode::F1 => app.effects.rgb_shift_intensity,
			KeyCode::F2 => app.effects.glitch_intensity,
			_ => app.effects.strobe_intensity,
		};
		approx(increased, base + 0.1);
		let command = by_named(named, code, ModifiersState::ALT, focused);
		apply_key(&mut app, command);
		let decreased = match code {
			KeyCode::F1 => app.effects.rgb_shift_intensity,
			KeyCode::F2 => app.effects.glitch_intensity,
			_ => app.effects.strobe_intensity,
		};
		approx(decreased, base);
	}

	// F4 and ` toggle blackout.
	let command = by_named(NamedKey::F4, KeyCode::F4, ModifiersState::empty(), focused);
	assert_eq!(command, Some(UiCommand::ToggleBlackout));
	apply_key(&mut app, command);
	assert!(app.effects.blackout);
	let command = by_char("`", KeyCode::Backquote, ModifiersState::empty(), focused);
	assert_eq!(command, Some(UiCommand::ToggleBlackout));
	apply_key(&mut app, command);
	assert!(!app.effects.blackout);

	// F5 / F6 dimmer.
	let command = by_named(NamedKey::F5, KeyCode::F5, ModifiersState::empty(), focused);
	assert_eq!(command, Some(UiCommand::AdjustMasterDimmer(-MASTER_DIMMER_STEP)));
	apply_key(&mut app, command);
	approx(app.effects.master_dimmer, 1.0 - MASTER_DIMMER_STEP);
	let command = by_named(NamedKey::F6, KeyCode::F6, ModifiersState::empty(), focused);
	assert_eq!(command, Some(UiCommand::AdjustMasterDimmer(MASTER_DIMMER_STEP)));
	apply_key(&mut app, command);
	approx(app.effects.master_dimmer, 1.0);

	// F7-F9 / Shift + F7-F9 colour grading.
	for (named, code, shift) in [
		(NamedKey::F7, KeyCode::F7, false),
		(NamedKey::F8, KeyCode::F8, false),
		(NamedKey::F9, KeyCode::F9, false),
		(NamedKey::F7, KeyCode::F7, true),
		(NamedKey::F8, KeyCode::F8, true),
		(NamedKey::F9, KeyCode::F9, true),
	] {
		let mods = if shift {
			ModifiersState::SHIFT
		} else {
			ModifiersState::empty()
		};
		let command = by_named(named, code, mods, focused);
		assert!(command.is_some());
		apply_key(&mut app, command);
	}
	approx(app.effects.brightness, 0.0);
	approx(app.effects.contrast, 1.0);
	approx(app.effects.saturation, 1.0);
	assert!((COLOR_GRADE_STEP - 0.05).abs() < 1e-6);

	let command = by_named(NamedKey::F10, KeyCode::F10, ModifiersState::empty(), focused);
	assert_eq!(command, Some(UiCommand::AdjustStrobeSpeedHz(STROBE_SPEED_STEP_HZ)));
	let before = app.effects.strobe_speed_hz;
	apply_key(&mut app, command);
	approx(app.effects.strobe_speed_hz, before + STROBE_SPEED_STEP_HZ);
	let command = by_named(NamedKey::F10, KeyCode::F10, ModifiersState::SHIFT, focused);
	apply_key(&mut app, command);
	approx(app.effects.strobe_speed_hz, before);

	// F12 resets the effects but keeps the live safety fields.
	app.effects.master_dimmer = 0.6;
	app.effects.blackout = true;
	let command = by_named(NamedKey::F12, KeyCode::F12, ModifiersState::empty(), focused);
	assert_eq!(command, Some(UiCommand::ResetAllEffects));
	apply_key(&mut app, command);
	assert!(!app.effects.rgb_shift_enabled);
	assert!(!app.effects.glitch_enabled);
	assert!(!app.effects.strobe_enabled);
	approx(app.effects.master_dimmer, 0.6);
	assert!(app.effects.blackout);
	app.effects.blackout = false;

	// Ctrl + 1-4 toggle the effects and blackout.
	for (number, expected) in [
		(1_u8, UiCommand::ToggleRgbShift),
		(2, UiCommand::ToggleGlitch),
		(3, UiCommand::ToggleStrobe),
		(4, UiCommand::ToggleBlackout),
	] {
		let command = by_char(
			&number.to_string(),
			DIGITS[usize::from(number - 1)].0,
			ModifiersState::CONTROL,
			focused,
		);
		assert_eq!(command, Some(expected));
		apply_key(&mut app, command);
	}
	assert!(app.effects.rgb_shift_enabled);
	assert!(app.effects.glitch_enabled);
	assert!(app.effects.strobe_enabled);
	assert!(app.effects.blackout);

	// Ctrl + 5-9 and Ctrl + 0 adjust then reset the master parameters.
	app.effects.blackout = false;
	app.effects.master_dimmer = 1.0;
	app.effects.brightness = 0.0;
	app.effects.contrast = 1.0;
	app.effects.saturation = 1.0;
	app.effects.strobe_speed_hz = 8.0;
	for (number, code, expected) in [
		(5_u8, KeyCode::Digit5, UiCommand::AdjustMasterDimmer(0.05)),
		(6, KeyCode::Digit6, UiCommand::AdjustBrightness(0.05)),
		(7, KeyCode::Digit7, UiCommand::AdjustContrast(0.05)),
		(8, KeyCode::Digit8, UiCommand::AdjustSaturation(0.05)),
		(9, KeyCode::Digit9, UiCommand::AdjustStrobeSpeedHz(2.0)),
		(0, KeyCode::Digit0, UiCommand::ResetAllEffects),
	] {
		let command = by_char(&number.to_string(), code, ModifiersState::CONTROL, focused);
		assert_eq!(command, Some(expected));
		apply_key(&mut app, command);
	}

	// Ctrl / Cmd + O toggle the external output.
	let command = by_char("o", KeyCode::KeyO, ModifiersState::CONTROL, focused);
	assert_eq!(command, Some(UiCommand::ToggleOutputEnabled));
	let outcome = apply_key(&mut app, command);
	assert!(app.output.enabled);
	assert!(outcome.needs_output_sync);
	let command = by_char("o", KeyCode::KeyO, ModifiersState::SUPER, focused);
	assert_eq!(command, Some(UiCommand::ToggleOutputEnabled));
	apply_key(&mut app, command);
	assert!(!app.output.enabled);

	// H / ? toggle the shortcut overlay, G the log drawer, Escape closes both.
	let command = by_char("h", KeyCode::KeyH, ModifiersState::empty(), focused);
	assert_eq!(command, Some(UiCommand::ToggleShortcutsOverlay));
	apply_key(&mut app, command);
	assert!(app.show_shortcuts);
	// `?` is the shifted `/` key on a real US keyboard, not a bare `h`.
	let command = by_char("?", KeyCode::Slash, ModifiersState::SHIFT, focused);
	assert_eq!(command, Some(UiCommand::ToggleShortcutsOverlay));
	apply_key(&mut app, command);
	assert!(!app.show_shortcuts);
	// `~` is Shift + `` ` ``; the backquote alias keeps the blackout reachable on
	// layouts where the physical key prints `~`.
	let command = by_char("~", KeyCode::Backquote, ModifiersState::SHIFT, focused);
	assert_eq!(command, Some(UiCommand::ToggleBlackout));
	apply_key(&mut app, command);
	assert!(app.effects.blackout);
	apply_key(&mut app, Some(UiCommand::ToggleBlackout));
	assert!(!app.effects.blackout);
	let command = by_char("g", KeyCode::KeyG, ModifiersState::empty(), focused);
	assert_eq!(command, Some(UiCommand::ToggleLogsDrawer));
	apply_key(&mut app, command);
	assert!(app.show_logs);
	app.show_shortcuts = true;
	let command = by_named(NamedKey::Escape, KeyCode::Escape, ModifiersState::empty(), focused);
	assert_eq!(command, Some(UiCommand::CloseOverlays));
	apply_key(&mut app, command);
	assert!(!app.show_shortcuts);
	assert!(!app.show_logs);
}

#[test]
fn exit_guard_accepts_only_its_own_answers() {
	let mut app = AppState::default();
	let focused = app.focused_layer;

	// Ctrl+Q (H1 on Windows/Linux) opens the guard but never quits.
	let command = by_char("q", KeyCode::KeyQ, ModifiersState::CONTROL, focused);
	assert_eq!(command, Some(UiCommand::RequestExit));
	let outcome = apply_key(&mut app, command);
	assert!(outcome.exit_requested);
	assert!(!outcome.confirm_quit);
	assert!(app.exit_confirm_open);

	// Cmd+Q (Claude M5 on macOS) does exactly the same.
	apply_key(&mut app, Some(UiCommand::CancelExit));
	assert!(!app.exit_confirm_open);
	let command = by_char("q", KeyCode::KeyQ, ModifiersState::SUPER, focused);
	assert_eq!(command, Some(UiCommand::RequestExit));
	apply_key(&mut app, command);
	assert!(app.exit_confirm_open);

	// While the guard is open every other key is swallowed, even when egui
	// claims the keyboard (a focused text field must not bypass the guard).
	let swallowed = map_key_event(
		&Key::Character("x".into()),
		PhysicalKey::Code(KeyCode::KeyX),
		ModifiersState::empty(),
		focused,
		true,
		true,
	);
	assert_eq!(swallowed, None);
	let swallowed_digit = map_key_event(
		&Key::Character("1".into()),
		PhysicalKey::Code(KeyCode::Digit1),
		ModifiersState::empty(),
		focused,
		true,
		false,
	);
	assert_eq!(swallowed_digit, None);

	// Enter is the most common live trigger and must never confirm the guard.
	for (key, code) in [
		(Key::Named(NamedKey::Enter), KeyCode::Enter),
		(Key::Named(NamedKey::Enter), KeyCode::NumpadEnter),
	] {
		let swallowed = map_key_event(
			&key,
			PhysicalKey::Code(code),
			ModifiersState::empty(),
			focused,
			true,
			true,
		);
		assert_eq!(swallowed, None, "{code:?} must be swallowed while the guard is open");
	}

	// Escape cancels without quitting.
	let cancel = map_key_event(
		&Key::Named(NamedKey::Escape),
		PhysicalKey::Code(KeyCode::Escape),
		ModifiersState::empty(),
		focused,
		true,
		true,
	);
	assert_eq!(cancel, Some(UiCommand::CancelExit));

	// Repeating a quit gesture is the only way to confirm the exit.
	for (key, code, mods) in [
		(Key::Character("q".into()), KeyCode::KeyQ, ModifiersState::SUPER),
		(Key::Character("q".into()), KeyCode::KeyQ, ModifiersState::CONTROL),
		(Key::Named(NamedKey::F4), KeyCode::F4, ModifiersState::ALT),
	] {
		let confirm = map_key_event(&key, PhysicalKey::Code(code), mods, focused, true, false);
		assert_eq!(confirm, Some(UiCommand::ConfirmExit), "{code:?} must confirm the guard");
	}

	// Cancel keeps the show running and restores normal mapping.
	let outcome = apply_key(&mut app, cancel);
	assert!(!outcome.confirm_quit);
	assert!(!outcome.exit_requested);
	assert!(!app.exit_confirm_open);
	let command = by_char("x", KeyCode::KeyX, ModifiersState::empty(), focused);
	assert_eq!(command, Some(UiCommand::ClearLayer(focused)));

	// Confirming leaves the guard closed and requests the actual quit.
	apply_key(&mut app, by_char("q", KeyCode::KeyQ, ModifiersState::CONTROL, focused));
	assert!(app.exit_confirm_open);
	let outcome = apply_key(
		&mut app,
		map_key_event(
			&Key::Character("q".into()),
			PhysicalKey::Code(KeyCode::KeyQ),
			ModifiersState::SUPER,
			focused,
			true,
			false,
		),
	);
	assert!(outcome.confirm_quit);
	assert!(!app.exit_confirm_open);

	// Alt + F4 also routes through the guard.
	let command = by_named(NamedKey::F4, KeyCode::F4, ModifiersState::ALT, focused);
	assert_eq!(command, Some(UiCommand::RequestExit));
}

// ---------------------------------------------------------------------------
// Project files
// ---------------------------------------------------------------------------

/// A path that is absolute on the host. `/absolute/b.mov` is only
/// root-relative on Windows, where joining it keeps the base dir's drive.
#[cfg(windows)]
const ABSOLUTE_CLIP: &str = "C:/absolute/b.mov";
#[cfg(not(windows))]
const ABSOLUTE_CLIP: &str = "/absolute/b.mov";

#[test]
fn toml_and_json_project_files_resolve_relative_media_paths() {
	let dir = TempDir::new("project_files").unwrap();

	let toml_path = dir.path().join("show.toml");
	std::fs::write(
		&toml_path,
		format!(
			r#"
clip_folder = "media"
clips = ["clips/a.mov", "{ABSOLUTE_CLIP}"]
preferred_midi_port = "Launch Control XL 3"

[output]
monitor_name = "Studio Display"
enabled = true

[effects]
master_dimmer = 0.4
brightness = 0.2
saturation = 9.0

[[layers]]
clip_path = "layers/one.mov"
opacity = 0.5
speed = 2.0
blend_mode = "Screen"
playback_mode = "OneShot"

[[layers]]
clip_path = "layers/two.mov"
"#
		),
	)
	.unwrap();
	let toml = ProjectConfig::load(&toml_path).unwrap();
	assert_eq!(toml.clip_folder, Some(dir.path().join("media")));
	assert_eq!(
		toml.explicit_clips().unwrap(),
		[dir.path().join("clips/a.mov"), PathBuf::from(ABSOLUTE_CLIP)]
	);
	assert_eq!(toml.layers.len(), 2);
	assert_eq!(toml.layers[0].clip_path, Some(dir.path().join("layers/one.mov")));
	approx(toml.layers[0].opacity, 0.5);
	approx(toml.layers[0].speed, 2.0);
	assert_eq!(toml.layers[0].blend_mode, BlendMode::Screen);
	assert_eq!(toml.layers[0].playback_mode, PlaybackMode::OneShot);
	assert_eq!(toml.layers[1].clip_path, Some(dir.path().join("layers/two.mov")));
	assert_eq!(toml.layers[1].blend_mode, BlendMode::Normal);
	assert_eq!(toml.layers[1].playback_mode, PlaybackMode::Loop);
	let effects = toml.effects.as_ref().unwrap();
	approx(effects.master_dimmer, 0.4);
	approx(effects.brightness, 0.2);
	approx(effects.saturation, 2.0);
	assert_eq!(
		toml.output.as_ref().unwrap().monitor_name.as_deref(),
		Some("Studio Display")
	);
	assert!(toml.output.as_ref().unwrap().enabled);
	assert_eq!(toml.preferred_midi_port.as_deref(), Some("Launch Control XL 3"));

	let json_path = dir.path().join("show.json");
	std::fs::write(
		&json_path,
		r#"{
			"clips": ["media/intro.mov"],
			"layers": [
				{
					"clip_path": "media/loop.mov",
					"opacity": 0.25,
					"blend_mode": "Difference",
					"playback_mode": "Loop"
				}
			],
			"output": {"monitor_name": "HDMI-1", "enabled": false},
			"effects": {"saturation": 0.5},
			"preferred_midi_port": "LCXL3"
		}"#,
	)
	.unwrap();
	let json = ProjectConfig::load(&json_path).unwrap();
	assert_eq!(json.clip_folder, None);
	assert_eq!(json.explicit_clips().unwrap(), [dir.path().join("media/intro.mov")]);
	assert_eq!(json.layers.len(), 1);
	assert_eq!(json.layers[0].clip_path, Some(dir.path().join("media/loop.mov")));
	approx(json.layers[0].opacity, 0.25);
	assert_eq!(json.layers[0].blend_mode, BlendMode::Difference);
	assert_eq!(json.layers[0].playback_mode, PlaybackMode::Loop);
	approx(json.effects.as_ref().unwrap().saturation, 0.5);
	assert_eq!(json.output.as_ref().unwrap().monitor_name.as_deref(), Some("HDMI-1"));
	assert!(!json.output.as_ref().unwrap().enabled);
	assert_eq!(json.preferred_midi_port.as_deref(), Some("LCXL3"));

	let empty = ProjectConfig::default();
	assert!(empty.explicit_clips().is_none());
}

// ---------------------------------------------------------------------------
// Persisted session state
// ---------------------------------------------------------------------------

#[test]
fn persisted_state_round_trips_atomically_and_restores_stopped_layers() {
	let dir = TempDir::new("persisted_state").unwrap();
	let mut app = AppState::default();
	for id in 1_u64..=4 {
		app.upsert_clip(clip(id, &dir.path().join(format!("clip{id}.mov"))));
	}

	let speeds = [1.0_f32, 0.5, -1.0, 2.0];
	for (index, target) in LayerIndex::ALL.iter().enumerate() {
		assert!(app.trigger_clip_on_layer(*target, ClipId(index as u64 + 1)));
		app.set_layer_opacity(*target, 0.8 - index as f32 * 0.1);
		app.set_layer_speed(*target, speeds[index]);
		let state = app.layer_mut(*target).unwrap();
		state.blend_mode = BlendMode::ALL[index];
		state.playback_mode = if index % 2 == 0 {
			PlaybackMode::Loop
		} else {
			PlaybackMode::OneShot
		};
	}
	app.advance_layers(0.3);
	app.effects = MasterEffects {
		rgb_shift_enabled: true,
		rgb_shift_intensity: 0.6,
		glitch_enabled: true,
		glitch_intensity: 0.4,
		strobe_enabled: true,
		strobe_intensity: 0.3,
		strobe_speed_hz: 18.0,
		brightness: 0.25,
		contrast: 1.4,
		saturation: 0.8,
		master_dimmer: 0.55,
		blackout: false,
	};
	app.output = OutputDisplaySelection {
		monitor_name: Some("Studio Display".to_string()),
		enabled: true,
	};
	app.selected_midi_port = Some("Launch Control XL 3".to_string());
	app.focused_layer = LayerIndex::ALL[2];

	let snapshot = PersistedState::capture(&app);
	assert_eq!(snapshot.version, STATE_VERSION);
	assert_eq!(snapshot.focused_layer, 3);
	assert_eq!(snapshot.clip_paths.len(), 4);
	for (index, layer) in snapshot.layers.iter().enumerate() {
		let expected = dir.path().join(format!("clip{}.mov", index + 1));
		assert_eq!(layer.clip_path.as_deref(), Some(expected.as_path()));
	}

	let state_path = dir.path().join("state.json");
	save_state_atomic(&state_path, &snapshot).unwrap();
	assert!(state_path.exists());
	let leftovers: Vec<String> = std::fs::read_dir(dir.path())
		.unwrap()
		.filter_map(Result::ok)
		.map(|entry| entry.file_name().to_string_lossy().into_owned())
		.filter(|name| name.contains(".tmp"))
		.collect();
	assert!(leftovers.is_empty(), "atomic save left temporary files: {leftovers:?}");

	let loaded = load_state(&state_path).unwrap().unwrap();
	assert_eq!(loaded, snapshot);

	let mut restored = AppState::default();
	for id in 101_u64..=104 {
		restored.upsert_clip(clip(id, &dir.path().join(format!("clip{}.mov", id - 100))));
	}
	let mapping: Vec<(PathBuf, ClipId)> = restored.clips.iter().map(|clip| (clip.path.clone(), clip.id)).collect();
	loaded.restore_into(&mut restored, |path| {
		mapping
			.iter()
			.find(|(candidate, _)| candidate == path)
			.map(|(_, id)| *id)
	});

	assert!(!restored.dirty);
	assert_eq!(restored.focused_layer, LayerIndex::ALL[2]);
	assert_eq!(restored.effects, snapshot.effects);
	approx(restored.effects.master_dimmer, 0.55);
	assert_eq!(restored.output, snapshot.output);
	assert_eq!(restored.selected_midi_port, snapshot.selected_midi_port);

	for (index, target) in LayerIndex::ALL.iter().enumerate() {
		let layer = restored.layer(*target);
		let persisted = &snapshot.layers[index];
		assert_eq!(
			layer.playback_state,
			PlaybackState::Paused,
			"on-air layers come back visible but never start moving on their own (SPEC §7.3/§8)"
		);
		// The playhead itself is preserved: the operator returns to the frame they
		// left, only the transport stays paused until they resume it.
		approx_f64(layer.playhead_frame, persisted.playhead_frame);
		assert_eq!(layer.playback_mode, persisted.playback_mode);
		assert_eq!(layer.blend_mode, persisted.blend_mode);
		approx(layer.speed, persisted.speed);
		approx(layer.opacity, persisted.opacity);
		assert!(
			persisted.playhead_frame > 0.0,
			"`advance_layers(0.3)` must have moved every layer off frame 0"
		);
		let clip_id = layer.active_clip.expect("each layer had a persisted clip");
		let path = restored.clip_by_id(clip_id).unwrap().path.clone();
		assert_eq!(Some(path), persisted.clip_path);
	}

	// A persisted path that no longer exists leaves the layer empty and stopped,
	// while its saved settings are still applied.
	let mut missing = snapshot.clone();
	missing.layers[1].clip_path = Some(dir.path().join("missing.mov"));
	let mut fresh = AppState::default();
	for id in 201_u64..=204 {
		fresh.upsert_clip(clip(id, &dir.path().join(format!("clip{}.mov", id - 200))));
	}
	let mapping: Vec<(PathBuf, ClipId)> = fresh.clips.iter().map(|clip| (clip.path.clone(), clip.id)).collect();
	missing.restore_into(&mut fresh, |path| {
		mapping
			.iter()
			.find(|(candidate, _)| candidate == path)
			.map(|(_, id)| *id)
	});
	let missing_layer = fresh.layer(LayerIndex::ALL[1]);
	assert_eq!(missing_layer.active_clip, None);
	assert_eq!(missing_layer.playback_state, PlaybackState::Stopped);
	approx(missing_layer.speed, missing.layers[1].speed);
	approx(missing_layer.opacity, missing.layers[1].opacity);
	assert_eq!(missing_layer.playback_mode, missing.layers[1].playback_mode);

	// Missing, malformed, and future-version state files stay total.
	assert!(load_state(&dir.path().join("absent.json")).unwrap().is_none());
	let malformed = dir.path().join("malformed.json");
	std::fs::write(&malformed, "not json").unwrap();
	assert!(load_state(&malformed).is_err());
	let future = dir.path().join("future.json");
	std::fs::write(&future, format!(r#"{{"version":{}}}"#, STATE_VERSION + 1)).unwrap();
	assert!(load_state(&future).unwrap().is_none());
}

#[test]
fn state_files_tolerate_one_three_and_five_layer_entries() {
	let dir = TempDir::new("lenient_layers").unwrap();

	// One layer entry: the rest is padded with defaults, and the missing
	// `effects` block falls back to the neutral master chain.
	let one = dir.path().join("one.json");
	std::fs::write(
		&one,
		r#"{"version":1,"layers":[{"opacity":0.25,"blend_mode":"Screen"}]}"#,
	)
	.unwrap();
	let loaded = load_state(&one).unwrap().unwrap();
	assert_eq!(loaded.version, STATE_VERSION);
	approx(loaded.layers[0].opacity, 0.25);
	assert_eq!(loaded.layers[0].blend_mode, BlendMode::Screen);
	for layer in &loaded.layers[1..] {
		assert_eq!(*layer, PersistedLayerState::default());
	}
	assert_eq!(loaded.effects, MasterEffects::default());
	assert!(loaded.clip_paths.is_empty());
	assert_eq!(loaded.focused_layer, 1);

	// Three entries with a partial effects block.
	let three = dir.path().join("three.json");
	std::fs::write(
		&three,
		r#"{
			"version": 1,
			"layers": [
				{"opacity": 0.11},
				{"opacity": 0.22, "speed": 2.0},
				{"opacity": 0.33, "playback_mode": "OneShot"}
			],
			"effects": {"master_dimmer": 0.5}
		}"#,
	)
	.unwrap();
	let loaded = load_state(&three).unwrap().unwrap();
	approx(loaded.layers[0].opacity, 0.11);
	approx(loaded.layers[1].opacity, 0.22);
	approx(loaded.layers[1].speed, 2.0);
	approx(loaded.layers[2].opacity, 0.33);
	assert_eq!(loaded.layers[2].playback_mode, PlaybackMode::OneShot);
	assert_eq!(loaded.layers[3], PersistedLayerState::default());
	approx(loaded.effects.master_dimmer, 0.5);
	approx(loaded.effects.contrast, 1.0);

	// Five entries: the fifth layer has no slot in a four-layer rig and is dropped.
	let five = dir.path().join("five.json");
	std::fs::write(
		&five,
		r#"{
			"version": 1,
			"layers": [
				{"opacity": 0.11},
				{"opacity": 0.22},
				{"opacity": 0.33},
				{"opacity": 0.44},
				{"opacity": 0.55}
			]
		}"#,
	)
	.unwrap();
	let loaded = load_state(&five).unwrap().unwrap();
	for (index, expected) in [0.11_f32, 0.22, 0.33, 0.44].into_iter().enumerate() {
		approx(loaded.layers[index].opacity, expected);
	}
	assert!(
		loaded.layers.iter().all(|layer| (layer.opacity - 0.55).abs() > 1e-5),
		"the fifth layer entry must be ignored"
	);

	// Zero entries and a completely empty document still load.
	let empty = dir.path().join("empty.json");
	std::fs::write(&empty, "{}").unwrap();
	let loaded = load_state(&empty).unwrap().unwrap();
	assert_eq!(loaded.layers, PersistedState::default().layers);
}

// ---------------------------------------------------------------------------
// SPEC §6.3 MIDI <-> UI parity
// ---------------------------------------------------------------------------

/// One row of the MIDI -> UI parity table.
struct ParityCase {
	/// Control being exercised, used in assertion messages.
	name: &'static str,
	/// Action produced by the Launch Control XL 3 mapping.
	midi: MidiAction,
	/// Equivalent command produced by the keyboard/UI layer.
	ui: UiCommand,
}

/// Initial state every parity row starts from.
///
/// All four layers are playing a distinct clip part-way through, so trigger,
/// retrigger and pause rows exercise real transport changes, and the master
/// effects sit at non-neutral values.
fn parity_state() -> AppState {
	let mut app = state_with_clips(4);
	for (index, target) in LayerIndex::ALL.iter().enumerate() {
		let id = ClipId(index as u64 + 1);
		let layer = app.layer_mut(*target).unwrap();
		layer.selected_clip = Some(id);
		layer.active_clip = Some(id);
		layer.playback_state = PlaybackState::Playing;
		layer.playhead_frame = 7.0;
		layer.frame_count = 30;
		layer.clip_fps = 30.0;
		layer.speed = 1.0 + index as f32 * 0.25;
		layer.opacity = 0.8;
		layer.blend_mode = BlendMode::ALL[index];
	}
	app.effects = MasterEffects {
		rgb_shift_enabled: false,
		rgb_shift_intensity: 0.2,
		glitch_enabled: true,
		glitch_intensity: 0.3,
		strobe_enabled: false,
		strobe_intensity: 0.4,
		strobe_speed_hz: 6.0,
		brightness: 0.1,
		contrast: 1.2,
		saturation: 0.9,
		master_dimmer: 0.85,
		blackout: false,
	};
	app.focused_layer = LayerIndex::ALL[1];
	app
}

/// Every mapped MIDI control paired with its UI equivalent (SPEC §6.3).
fn parity_cases() -> Vec<ParityCase> {
	let first = LayerIndex::ALL[0];
	let second = LayerIndex::ALL[1];
	let selected_id = ClipId(resolve_clip_index(100, 4).unwrap() as u64 + 1);
	vec![
		ParityCase {
			name: "SetLayerOpacity",
			midi: MidiAction::SetLayerOpacity {
				layer: first,
				opacity: 0.42,
			},
			ui: UiCommand::SetLayerOpacity {
				layer: first,
				opacity: 0.42,
			},
		},
		ParityCase {
			name: "SetLayerSpeed",
			midi: MidiAction::SetLayerSpeed {
				layer: second,
				speed: 2.5,
			},
			ui: UiCommand::SetLayerSpeed {
				layer: second,
				speed: 2.5,
			},
		},
		ParityCase {
			name: "SetLayerBlendMode",
			midi: MidiAction::SetLayerBlendMode {
				layer: first,
				blend_mode: BlendMode::from_midi_cc(64),
			},
			ui: UiCommand::SetLayerBlendMode {
				layer: first,
				blend_mode: BlendMode::from_midi_cc(64),
			},
		},
		ParityCase {
			name: "SelectLayerClip",
			midi: MidiAction::SelectLayerClip {
				layer: second,
				cc_value: 100,
			},
			ui: UiCommand::SelectLayerClip {
				layer: second,
				clip_id: selected_id,
			},
		},
		ParityCase {
			name: "TriggerLayer",
			midi: MidiAction::TriggerLayer(first),
			ui: UiCommand::TriggerLayer(first),
		},
		ParityCase {
			name: "TogglePauseLayer",
			midi: MidiAction::TogglePauseLayer(second),
			ui: UiCommand::TogglePauseLayer(second),
		},
		ParityCase {
			name: "SetRgbShiftIntensity",
			midi: MidiAction::SetRgbShiftIntensity(0.7),
			ui: UiCommand::SetRgbShiftIntensity(0.7),
		},
		ParityCase {
			name: "SetGlitchIntensity",
			midi: MidiAction::SetGlitchIntensity(0.55),
			ui: UiCommand::SetGlitchIntensity(0.55),
		},
		ParityCase {
			name: "SetStrobeIntensity",
			midi: MidiAction::SetStrobeIntensity(0.9),
			ui: UiCommand::SetStrobeIntensity(0.9),
		},
		ParityCase {
			name: "SetMasterDimmer",
			midi: MidiAction::SetMasterDimmer(0.35),
			ui: UiCommand::SetMasterDimmer(0.35),
		},
		ParityCase {
			name: "SetBrightness",
			midi: MidiAction::SetBrightness(-0.2),
			ui: UiCommand::SetBrightness(-0.2),
		},
		ParityCase {
			name: "SetContrast",
			midi: MidiAction::SetContrast(1.6),
			ui: UiCommand::SetContrast(1.6),
		},
		ParityCase {
			name: "SetSaturation",
			midi: MidiAction::SetSaturation(0.4),
			ui: UiCommand::SetSaturation(0.4),
		},
		ParityCase {
			name: "SetStrobeSpeedHz",
			midi: MidiAction::SetStrobeSpeedHz(12.0),
			ui: UiCommand::SetStrobeSpeedHz(12.0),
		},
		ParityCase {
			name: "ToggleRgbShift",
			midi: MidiAction::ToggleRgbShift,
			ui: UiCommand::ToggleRgbShift,
		},
		ParityCase {
			name: "ToggleGlitch",
			midi: MidiAction::ToggleGlitch,
			ui: UiCommand::ToggleGlitch,
		},
		ParityCase {
			name: "ToggleStrobe",
			midi: MidiAction::ToggleStrobe,
			ui: UiCommand::ToggleStrobe,
		},
		ParityCase {
			name: "ToggleBlackout",
			midi: MidiAction::ToggleBlackout,
			ui: UiCommand::ToggleBlackout,
		},
		ParityCase {
			name: "ResetAllEffects",
			midi: MidiAction::ResetAllEffects,
			ui: UiCommand::ResetAllEffects,
		},
		ParityCase {
			name: "RetriggerAllActiveLayers",
			midi: MidiAction::RetriggerAllActiveLayers,
			ui: UiCommand::RetriggerAllActiveLayers,
		},
		ParityCase {
			name: "ToggleFocusedLayerPlaybackMode",
			midi: MidiAction::ToggleFocusedLayerPlaybackMode,
			ui: UiCommand::ToggleLayerPlaybackMode(second),
		},
		ParityCase {
			name: "ResetAllLayerSpeeds",
			midi: MidiAction::ResetAllLayerSpeeds,
			ui: UiCommand::ResetAllLayerSpeeds,
		},
	]
}

#[test]
fn midi_actions_and_ui_commands_mutate_identical_state() {
	let cases = parity_cases();
	assert_eq!(cases.len(), 22, "SPEC §6.3 maps 22 distinct MIDI controls");

	for case in cases {
		let initial = parity_state();
		let mut midi_app = parity_state();
		let mut ui_app = parity_state();

		let midi_outcome = apply_midi_action(&mut midi_app, case.midi);
		let ui_outcome = apply_ui_command(&mut ui_app, &case.ui);

		assert!(
			midi_app.layers != initial.layers
				|| midi_app.focused_layer != initial.focused_layer
				|| midi_app.effects != initial.effects
				|| midi_app.dirty != initial.dirty
				|| LayerIndex::ALL.into_iter().any(|t| midi_outcome.contains(t)),
			"`{}`: parity case did not mutate any state",
			case.name
		);
		assert_eq!(midi_app.layers, ui_app.layers, "`{}`: layer state diverged", case.name);
		assert_eq!(
			midi_app.focused_layer, ui_app.focused_layer,
			"`{}`: focus diverged",
			case.name
		);
		assert_eq!(midi_app.effects, ui_app.effects, "`{}`: effects diverged", case.name);
		assert_eq!(midi_app.dirty, ui_app.dirty, "`{}`: dirty flag diverged", case.name);
		for target in LayerIndex::ALL {
			assert_eq!(
				midi_outcome.contains(target),
				ui_outcome.was_retriggered(target),
				"`{}`: retrigger flag for {target} diverged",
				case.name
			);
		}
	}
}
