//! Keyboard fallback covering every MIDI-mappable control (SPEC §7.2).
//!
//! The fixed Launch Control XL 3 template is the primary instrument, but the
//! operator must be able to run the whole show from a bare keyboard when the
//! controller is unplugged, busy, or broken. [`map_key_event`] translates one
//! winit key event into a [`UiCommand`] and never performs I/O, so the whole
//! fallback map is unit-testable.
//!
//! Two rules keep the map safe on stage:
//!
//! - While the quit confirmation guard is open (SPEC §9.2) the map answers only
//!   three gestures: pressing the quit shortcut again confirms, `Escape`
//!   cancels, and everything else — including `Enter`, which triggers clips
//!   during a set — is swallowed (`None`). That check runs first, which keeps the
//!   dialog answerable even when an egui widget holds keyboard focus.
//! - While egui is capturing keyboard input (a text field has focus) nothing is
//!   mapped at all.
//!
//! The number row is resolved through [`PhysicalKey`] first, because `Shift + 1`
//! produces `!` on US and Japanese layouts alike; the printed character is only
//! used as a fallback when the platform reports no physical key code.

use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};

use crate::model::LayerIndex;

use super::commands::UiCommand;

/// Opacity change applied by `Q` and `W`.
pub const LAYER_OPACITY_STEP: f32 = 0.05;
/// Playback-speed change applied by `A` and `D`.
pub const LAYER_SPEED_STEP: f32 = 0.1;
/// Intensity change applied by `Shift + F1`…`Shift + F3` and `Alt + F1`…`Alt + F3`.
pub const EFFECT_INTENSITY_STEP: f32 = 0.1;
/// Color-grade change applied by the `F7`…`F9` keys.
pub const COLOR_GRADE_STEP: f32 = 0.05;
/// Master dimmer change applied by `F5` and `F6`.
pub const MASTER_DIMMER_STEP: f32 = 0.05;
/// Strobe frequency change applied by `F10` and `Shift + F10`.
pub const STROBE_SPEED_STEP_HZ: f32 = 2.0;

/// One row of the keyboard help overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortcutEntry {
	/// Key or key combination, written the way the operator presses it.
	pub keys: &'static str,
	/// What the combination does.
	pub description: &'static str,
}

/// Every keyboard fallback, in overlay order.
pub const SHORTCUT_LEGEND: &[ShortcutEntry] = &[
	ShortcutEntry {
		keys: "1 / 2 / 3 / 4",
		description: "Focus layer 1-4",
	},
	ShortcutEntry {
		keys: "Shift + 1-4",
		description: "Trigger layer 1-4 (selected clip)",
	},
	ShortcutEntry {
		keys: "Alt + 1-4",
		description: "Pause / resume layer 1-4",
	},
	ShortcutEntry {
		keys: "Tab / Shift + Tab",
		description: "Focus next / previous layer",
	},
	ShortcutEntry {
		keys: "R",
		description: "Retrigger focused layer",
	},
	ShortcutEntry {
		keys: "Shift + R",
		description: "Retrigger all active layers",
	},
	ShortcutEntry {
		keys: "Space",
		description: "Pause / resume focused layer",
	},
	ShortcutEntry {
		keys: "L",
		description: "Toggle loop / one-shot on focused layer",
	},
	ShortcutEntry {
		keys: "X",
		description: "Clear focused layer",
	},
	ShortcutEntry {
		keys: "Q / W",
		description: "Layer opacity - / +",
	},
	ShortcutEntry {
		keys: "A / D",
		description: "Layer speed - / +",
	},
	ShortcutEntry {
		keys: "S",
		description: "Reset focused layer speed to 1x",
	},
	ShortcutEntry {
		keys: "Shift + S",
		description: "Reset every layer speed to 1x",
	},
	ShortcutEntry {
		keys: "B / Shift + B",
		description: "Next / previous blend mode",
	},
	ShortcutEntry {
		keys: "[ / ]",
		description: "Previous / next clip in the grid",
	},
	ShortcutEntry {
		keys: "Enter",
		description: "Trigger selected clip on focused layer",
	},
	ShortcutEntry {
		keys: "F1",
		description: "Toggle RGB shift",
	},
	ShortcutEntry {
		keys: "Shift + F1 / Alt + F1",
		description: "RGB shift intensity + / -",
	},
	ShortcutEntry {
		keys: "F2",
		description: "Toggle glitch",
	},
	ShortcutEntry {
		keys: "Shift + F2 / Alt + F2",
		description: "Glitch intensity + / -",
	},
	ShortcutEntry {
		keys: "F3",
		description: "Toggle strobe",
	},
	ShortcutEntry {
		keys: "Shift + F3 / Alt + F3",
		description: "Strobe intensity + / -",
	},
	ShortcutEntry {
		keys: "F4 or `",
		description: "Toggle master blackout",
	},
	ShortcutEntry {
		keys: "F5 / F6",
		description: "Master dimmer - / +",
	},
	ShortcutEntry {
		keys: "F7 / Shift + F7",
		description: "Brightness + / -",
	},
	ShortcutEntry {
		keys: "F8 / Shift + F8",
		description: "Contrast + / -",
	},
	ShortcutEntry {
		keys: "F9 / Shift + F9",
		description: "Saturation + / -",
	},
	ShortcutEntry {
		keys: "F10 / Shift + F10",
		description: "Strobe speed + / -",
	},
	ShortcutEntry {
		keys: "F12 (Ctrl/Cmd + F12 reserved)",
		description: "Reset all effects",
	},
	ShortcutEntry {
		keys: "Ctrl + 1 / 2 / 3 / 4",
		description: "Toggle RGB shift / glitch / strobe / blackout",
	},
	ShortcutEntry {
		keys: "Ctrl + 5-9, Ctrl + 0",
		description: "Dimmer, brightness, contrast, saturation, strobe speed, reset (Shift = -)",
	},
	ShortcutEntry {
		keys: "Ctrl + O or Cmd + O",
		description: "Toggle output on / off",
	},
	ShortcutEntry {
		keys: "H or ?",
		description: "Show / hide this shortcut overlay",
	},
	ShortcutEntry {
		keys: "G",
		description: "Show / hide the log drawer",
	},
	ShortcutEntry {
		keys: "Escape",
		description: "Close overlays",
	},
	ShortcutEntry {
		keys: "Cmd + Q / Ctrl + Q / Alt + F4",
		description: "Request exit (confirmation guard)",
	},
	ShortcutEntry {
		keys: "Repeat quit key / Escape (in dialog)",
		description: "Confirm / cancel exit",
	},
];

/// Translates one winit key press into the command it should run.
///
/// `physical` is preferred over the printed character for the number row so that
/// `Shift + 1`…`Shift + 4` and `Alt + 1`…`Alt + 4` work on every layout, where
/// `Shift + 1` yields `!` (US/JP) instead of the digit itself. Everything else
/// follows the character the operator actually sees, which keeps the legend
/// truthful on non-QWERTY keyboards.
///
/// Returns `None` for unmapped keys, while the quit guard is open (except for the
/// gestures that answer it), and while egui has keyboard focus. Key releases are
/// the caller's responsibility: pass only `ElementState::Pressed` events in.
pub fn map_key_event(
	key: &Key,
	physical: PhysicalKey,
	modifiers: ModifiersState,
	focused_layer: LayerIndex,
	exit_confirm_open: bool,
	egui_wants_keyboard: bool,
) -> Option<UiCommand> {
	if exit_confirm_open {
		// SPEC §9.2: the guard only accepts its own answers, otherwise a stray
		// keystroke would change the live output behind it. `Enter` is deliberately
		// swallowed: triggering clips with `Enter` is the most common live gesture,
		// and it must never end the show.
		let is_enter = matches!(key, Key::Named(NamedKey::Enter))
			|| matches!(physical, PhysicalKey::Code(KeyCode::Enter | KeyCode::NumpadEnter));
		if is_enter {
			return None;
		}
		let is_escape =
			matches!(key, Key::Named(NamedKey::Escape)) || matches!(physical, PhysicalKey::Code(KeyCode::Escape));
		if is_escape {
			return Some(UiCommand::CancelExit);
		}
		// Only a second quit gesture confirms; every other key is swallowed.
		return is_quit_gesture(key, physical, KeyMods::from_state(modifiers)).then_some(UiCommand::ConfirmExit);
	}

	if egui_wants_keyboard {
		return None;
	}

	let mods = KeyMods::from_state(modifiers);

	match resolve_logical_key(key, physical) {
		LogicalKey::Letter(letter) => map_letter(letter, mods, focused_layer),
		LogicalKey::Digit(number) => map_digit(number, mods),
		LogicalKey::Function(number) => map_function(number, mods),
		LogicalKey::BracketLeft => map_bracket(-1, mods, focused_layer),
		LogicalKey::BracketRight => map_bracket(1, mods, focused_layer),
		LogicalKey::Backquote => map_backquote(mods),
		LogicalKey::Question => map_overlay_toggle(mods),
		LogicalKey::Space => map_space(mods, focused_layer),
		LogicalKey::Tab => map_tab(mods),
		LogicalKey::Enter => map_enter(mods, focused_layer),
		LogicalKey::Escape => map_escape(mods),
		LogicalKey::Other => None,
	}
}

// ------------------------------------------------------------------ quit guard

/// Returns `true` when the pressed key is one of the quit shortcuts.
///
/// The same predicate drives the initial [`UiCommand::RequestExit`] and the
/// second press that confirms the dialog, so the two can never disagree.
fn is_quit_gesture(key: &Key, physical: PhysicalKey, mods: KeyMods) -> bool {
	match resolve_logical_key(key, physical) {
		// `Cmd + Q` / `Ctrl + Q` (Shift is tolerated, `Alt` is not).
		LogicalKey::Letter('q') => {
			(mods.meta && !mods.control && !mods.alt) || (mods.control && !mods.alt && !mods.meta)
		}
		// `Alt + F4`.
		LogicalKey::Function(4) => mods.alt && !mods.control && !mods.meta,
		_ => false,
	}
}

// --------------------------------------------------------------------- modifiers

/// Modifier flags with the predicates the mapping needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KeyMods {
	/// Shift is held.
	shift: bool,
	/// Alt / Option is held.
	alt: bool,
	/// Control is held.
	control: bool,
	/// Cmd (macOS) or the Windows key is held.
	meta: bool,
}

impl KeyMods {
	/// Captures the current modifier state.
	fn from_state(state: ModifiersState) -> Self {
		Self {
			shift: state.shift_key(),
			alt: state.alt_key(),
			control: state.control_key(),
			meta: state.super_key(),
		}
	}

	/// Returns `true` when no modifier is held.
	fn plain(self) -> bool {
		!self.shift && !self.alt && !self.control && !self.meta
	}

	/// Returns `true` when only Shift is held.
	fn shift_only(self) -> bool {
		self.shift && !self.alt && !self.control && !self.meta
	}

	/// Returns `true` when only Alt is held.
	fn alt_only(self) -> bool {
		self.alt && !self.shift && !self.control && !self.meta
	}

	/// Returns `true` when Control is held without Alt or Cmd.
	fn control_only(self) -> bool {
		self.control && !self.alt && !self.meta
	}

	/// Returns `true` when Control or Cmd is held.
	fn command(self) -> bool {
		self.control || self.meta
	}
}

// ---------------------------------------------------------------------- key ids

/// Platform-independent identity of the pressed key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogicalKey {
	/// A latin letter, already lowercased.
	Letter(char),
	/// A number-row digit in `0..=9`.
	Digit(u8),
	/// `[` / `{`.
	BracketLeft,
	/// `]` / `}`.
	BracketRight,
	/// `` ` `` / `~`.
	Backquote,
	/// `?`, the shifted form of `/`.
	Question,
	/// Space bar.
	Space,
	/// Tab.
	Tab,
	/// Enter / Return, including the numpad key.
	Enter,
	/// Escape.
	Escape,
	/// Function keys `F1..=F12`.
	Function(u8),
	/// Anything the mapping does not use.
	Other,
}

/// Resolves the key identity, preferring the layout-independent physical code
/// for the number row.
fn resolve_logical_key(key: &Key, physical: PhysicalKey) -> LogicalKey {
	let from_physical = logical_from_physical(physical);
	if matches!(from_physical, LogicalKey::Digit(_)) {
		return from_physical;
	}
	match logical_from_key(key) {
		LogicalKey::Other => from_physical,
		logical => logical,
	}
}

/// Derives the key identity from the printed character.
fn logical_from_key(key: &Key) -> LogicalKey {
	match key {
		Key::Character(text) => text.chars().next().map_or(LogicalKey::Other, logical_from_char),
		Key::Named(named) => logical_from_named(*named),
		_ => LogicalKey::Other,
	}
}

/// Maps a printed character onto its key identity.
///
/// Only the unambiguous shifted symbols of the first four number-row keys are
/// recognised; the remaining digits rely on the physical code, which every
/// desktop platform reports for the number row.
fn logical_from_char(ch: char) -> LogicalKey {
	match ch {
		'0' => LogicalKey::Digit(0),
		'1' => LogicalKey::Digit(1),
		'2' => LogicalKey::Digit(2),
		'3' => LogicalKey::Digit(3),
		'4' => LogicalKey::Digit(4),
		'5' => LogicalKey::Digit(5),
		'6' => LogicalKey::Digit(6),
		'7' => LogicalKey::Digit(7),
		'8' => LogicalKey::Digit(8),
		'9' => LogicalKey::Digit(9),
		'a'..='z' => LogicalKey::Letter(ch),
		'A'..='Z' => LogicalKey::Letter(ch.to_ascii_lowercase()),
		'!' => LogicalKey::Digit(1),
		'@' => LogicalKey::Digit(2),
		'"' => LogicalKey::Digit(2),
		'#' => LogicalKey::Digit(3),
		'$' => LogicalKey::Digit(4),
		'[' | '{' => LogicalKey::BracketLeft,
		']' | '}' => LogicalKey::BracketRight,
		'`' | '~' => LogicalKey::Backquote,
		'?' => LogicalKey::Question,
		' ' => LogicalKey::Space,
		_ => LogicalKey::Other,
	}
}

/// Maps a winit named key onto its key identity.
fn logical_from_named(named: NamedKey) -> LogicalKey {
	match named {
		NamedKey::Space => LogicalKey::Space,
		NamedKey::Tab => LogicalKey::Tab,
		NamedKey::Enter => LogicalKey::Enter,
		NamedKey::Escape => LogicalKey::Escape,
		NamedKey::F1 => LogicalKey::Function(1),
		NamedKey::F2 => LogicalKey::Function(2),
		NamedKey::F3 => LogicalKey::Function(3),
		NamedKey::F4 => LogicalKey::Function(4),
		NamedKey::F5 => LogicalKey::Function(5),
		NamedKey::F6 => LogicalKey::Function(6),
		NamedKey::F7 => LogicalKey::Function(7),
		NamedKey::F8 => LogicalKey::Function(8),
		NamedKey::F9 => LogicalKey::Function(9),
		NamedKey::F10 => LogicalKey::Function(10),
		NamedKey::F11 => LogicalKey::Function(11),
		NamedKey::F12 => LogicalKey::Function(12),
		_ => LogicalKey::Other,
	}
}

/// Maps a physical key code onto its key identity, used as a fallback and as the
/// authoritative source for the number row.
fn logical_from_physical(physical: PhysicalKey) -> LogicalKey {
	match physical {
		PhysicalKey::Code(code) => logical_from_code(code),
		PhysicalKey::Unidentified(_) => LogicalKey::Other,
	}
}

/// Maps a winit key code onto its key identity.
fn logical_from_code(code: KeyCode) -> LogicalKey {
	match code {
		KeyCode::Digit0 => LogicalKey::Digit(0),
		KeyCode::Digit1 => LogicalKey::Digit(1),
		KeyCode::Digit2 => LogicalKey::Digit(2),
		KeyCode::Digit3 => LogicalKey::Digit(3),
		KeyCode::Digit4 => LogicalKey::Digit(4),
		KeyCode::Digit5 => LogicalKey::Digit(5),
		KeyCode::Digit6 => LogicalKey::Digit(6),
		KeyCode::Digit7 => LogicalKey::Digit(7),
		KeyCode::Digit8 => LogicalKey::Digit(8),
		KeyCode::Digit9 => LogicalKey::Digit(9),
		KeyCode::KeyA => LogicalKey::Letter('a'),
		KeyCode::KeyB => LogicalKey::Letter('b'),
		KeyCode::KeyC => LogicalKey::Letter('c'),
		KeyCode::KeyD => LogicalKey::Letter('d'),
		KeyCode::KeyE => LogicalKey::Letter('e'),
		KeyCode::KeyF => LogicalKey::Letter('f'),
		KeyCode::KeyG => LogicalKey::Letter('g'),
		KeyCode::KeyH => LogicalKey::Letter('h'),
		KeyCode::KeyI => LogicalKey::Letter('i'),
		KeyCode::KeyJ => LogicalKey::Letter('j'),
		KeyCode::KeyK => LogicalKey::Letter('k'),
		KeyCode::KeyL => LogicalKey::Letter('l'),
		KeyCode::KeyM => LogicalKey::Letter('m'),
		KeyCode::KeyN => LogicalKey::Letter('n'),
		KeyCode::KeyO => LogicalKey::Letter('o'),
		KeyCode::KeyP => LogicalKey::Letter('p'),
		KeyCode::KeyQ => LogicalKey::Letter('q'),
		KeyCode::KeyR => LogicalKey::Letter('r'),
		KeyCode::KeyS => LogicalKey::Letter('s'),
		KeyCode::KeyT => LogicalKey::Letter('t'),
		KeyCode::KeyU => LogicalKey::Letter('u'),
		KeyCode::KeyV => LogicalKey::Letter('v'),
		KeyCode::KeyW => LogicalKey::Letter('w'),
		KeyCode::KeyX => LogicalKey::Letter('x'),
		KeyCode::KeyY => LogicalKey::Letter('y'),
		KeyCode::KeyZ => LogicalKey::Letter('z'),
		KeyCode::BracketLeft => LogicalKey::BracketLeft,
		KeyCode::BracketRight => LogicalKey::BracketRight,
		KeyCode::Backquote => LogicalKey::Backquote,
		KeyCode::Space => LogicalKey::Space,
		KeyCode::Tab => LogicalKey::Tab,
		KeyCode::Enter => LogicalKey::Enter,
		KeyCode::NumpadEnter => LogicalKey::Enter,
		KeyCode::Escape => LogicalKey::Escape,
		KeyCode::F1 => LogicalKey::Function(1),
		KeyCode::F2 => LogicalKey::Function(2),
		KeyCode::F3 => LogicalKey::Function(3),
		KeyCode::F4 => LogicalKey::Function(4),
		KeyCode::F5 => LogicalKey::Function(5),
		KeyCode::F6 => LogicalKey::Function(6),
		KeyCode::F7 => LogicalKey::Function(7),
		KeyCode::F8 => LogicalKey::Function(8),
		KeyCode::F9 => LogicalKey::Function(9),
		KeyCode::F10 => LogicalKey::Function(10),
		KeyCode::F11 => LogicalKey::Function(11),
		KeyCode::F12 => LogicalKey::Function(12),
		_ => LogicalKey::Other,
	}
}

// ------------------------------------------------------------------- sub-maps

/// Maps the number row: focus, trigger, pause, or the Ctrl master-effect block.
fn map_digit(number: u8, mods: KeyMods) -> Option<UiCommand> {
	if mods.control_only() {
		return map_control_digit(number, mods.shift);
	}
	if let Some(layer) = LayerIndex::from_number(number) {
		if mods.alt_only() {
			return Some(UiCommand::TogglePauseLayer(layer));
		}
		if mods.shift_only() {
			return Some(UiCommand::TriggerLayer(layer));
		}
		if mods.plain() {
			return Some(UiCommand::FocusLayer(layer));
		}
	}
	None
}

/// Ctrl + number-row block, a macOS-friendly alternative to the `F` keys.
///
/// `1`…`4` toggle the three master effects and the blackout; `5`…`9` and `0`
/// adjust the parameters that the `F5`…`F12` keys also reach, where `Shift`
/// inverts the direction.
fn map_control_digit(number: u8, shift: bool) -> Option<UiCommand> {
	let sign = if shift { -1.0 } else { 1.0 };
	Some(match number {
		1 => UiCommand::ToggleRgbShift,
		2 => UiCommand::ToggleGlitch,
		3 => UiCommand::ToggleStrobe,
		4 => UiCommand::ToggleBlackout,
		5 => UiCommand::AdjustMasterDimmer(sign * MASTER_DIMMER_STEP),
		6 => UiCommand::AdjustBrightness(sign * COLOR_GRADE_STEP),
		7 => UiCommand::AdjustContrast(sign * COLOR_GRADE_STEP),
		8 => UiCommand::AdjustSaturation(sign * COLOR_GRADE_STEP),
		9 => UiCommand::AdjustStrobeSpeedHz(sign * STROBE_SPEED_STEP_HZ),
		0 => UiCommand::ResetAllEffects,
		_ => return None,
	})
}

/// Maps the letter keys that act on the focused layer and the master effects.
fn map_letter(letter: char, mods: KeyMods, focused_layer: LayerIndex) -> Option<UiCommand> {
	if mods.command() {
		return match letter {
			'o' if !mods.alt => Some(UiCommand::ToggleOutputEnabled),
			// `Cmd + Q` and `Ctrl + Q` both request the guarded exit; `Alt` is
			// excluded so an Option/Alt chord cannot quit a live set.
			'q' if (mods.meta && !mods.control && !mods.alt) || mods.control_only() => Some(UiCommand::RequestExit),
			_ => None,
		};
	}
	if mods.alt {
		return None;
	}

	let shift = mods.shift;
	match letter {
		'r' => Some(if shift {
			UiCommand::RetriggerAllActiveLayers
		} else {
			UiCommand::RetriggerLayer(focused_layer)
		}),
		's' => Some(if shift {
			UiCommand::ResetAllLayerSpeeds
		} else {
			UiCommand::SetLayerSpeed {
				layer: focused_layer,
				speed: 1.0,
			}
		}),
		'b' => Some(UiCommand::CycleLayerBlendMode {
			layer: focused_layer,
			forward: !shift,
		}),
		'q' if !shift => Some(UiCommand::AdjustLayerOpacity {
			layer: focused_layer,
			delta: -LAYER_OPACITY_STEP,
		}),
		'w' if !shift => Some(UiCommand::AdjustLayerOpacity {
			layer: focused_layer,
			delta: LAYER_OPACITY_STEP,
		}),
		'a' if !shift => Some(UiCommand::AdjustLayerSpeed {
			layer: focused_layer,
			delta: -LAYER_SPEED_STEP,
		}),
		'd' if !shift => Some(UiCommand::AdjustLayerSpeed {
			layer: focused_layer,
			delta: LAYER_SPEED_STEP,
		}),
		'l' if !shift => Some(UiCommand::ToggleLayerPlaybackMode(focused_layer)),
		'x' if !shift => Some(UiCommand::ClearLayer(focused_layer)),
		'g' if !shift => Some(UiCommand::ToggleLogsDrawer),
		'h' => Some(UiCommand::ToggleShortcutsOverlay),
		_ => None,
	}
}

/// Maps the function row onto the master effects and the quit guard.
///
/// `F4` (blackout) and `F12` (reset all effects) only fire unmodified: the `F12`
/// chord is reserved for the OS, and a stray modifier must not blank the output
/// mid-set.
fn map_function(number: u8, mods: KeyMods) -> Option<UiCommand> {
	if mods.command() {
		return None;
	}
	let shift = mods.shift;

	match number {
		1 if mods.alt => Some(UiCommand::AdjustRgbShiftIntensity(-EFFECT_INTENSITY_STEP)),
		1 if shift => Some(UiCommand::AdjustRgbShiftIntensity(EFFECT_INTENSITY_STEP)),
		1 => Some(UiCommand::ToggleRgbShift),
		2 if mods.alt => Some(UiCommand::AdjustGlitchIntensity(-EFFECT_INTENSITY_STEP)),
		2 if shift => Some(UiCommand::AdjustGlitchIntensity(EFFECT_INTENSITY_STEP)),
		2 => Some(UiCommand::ToggleGlitch),
		3 if mods.alt => Some(UiCommand::AdjustStrobeIntensity(-EFFECT_INTENSITY_STEP)),
		3 if shift => Some(UiCommand::AdjustStrobeIntensity(EFFECT_INTENSITY_STEP)),
		3 => Some(UiCommand::ToggleStrobe),
		4 if mods.alt => Some(UiCommand::RequestExit),
		4 if mods.plain() => Some(UiCommand::ToggleBlackout),
		5 => Some(UiCommand::AdjustMasterDimmer(-MASTER_DIMMER_STEP)),
		6 => Some(UiCommand::AdjustMasterDimmer(MASTER_DIMMER_STEP)),
		7 if shift => Some(UiCommand::AdjustBrightness(-COLOR_GRADE_STEP)),
		7 => Some(UiCommand::AdjustBrightness(COLOR_GRADE_STEP)),
		8 if shift => Some(UiCommand::AdjustContrast(-COLOR_GRADE_STEP)),
		8 => Some(UiCommand::AdjustContrast(COLOR_GRADE_STEP)),
		9 if shift => Some(UiCommand::AdjustSaturation(-COLOR_GRADE_STEP)),
		9 => Some(UiCommand::AdjustSaturation(COLOR_GRADE_STEP)),
		10 if shift => Some(UiCommand::AdjustStrobeSpeedHz(-STROBE_SPEED_STEP_HZ)),
		10 => Some(UiCommand::AdjustStrobeSpeedHz(STROBE_SPEED_STEP_HZ)),
		12 if mods.plain() => Some(UiCommand::ResetAllEffects),
		_ => None,
	}
}

/// Maps `[` / `]` onto clip selection stepping.
fn map_bracket(delta: i32, mods: KeyMods, focused_layer: LayerIndex) -> Option<UiCommand> {
	if mods.alt || mods.command() {
		return None;
	}
	Some(UiCommand::StepLayerClip {
		layer: focused_layer,
		delta,
	})
}

/// Maps the backquote key, the ergonomic alternative to `F4`.
///
/// `Shift + backquote` stays bound because `` ` `` and `~` live on the same key
/// on most layouts; `Alt` and `Cmd`/`Ctrl` chords are left to the application.
fn map_backquote(mods: KeyMods) -> Option<UiCommand> {
	if mods.alt || mods.command() {
		return None;
	}
	Some(UiCommand::ToggleBlackout)
}

/// Maps `?` onto the shortcut overlay toggle.
fn map_overlay_toggle(mods: KeyMods) -> Option<UiCommand> {
	if mods.alt || mods.command() {
		return None;
	}
	Some(UiCommand::ToggleShortcutsOverlay)
}

/// Maps the space bar onto pause/resume of the focused layer.
fn map_space(mods: KeyMods, focused_layer: LayerIndex) -> Option<UiCommand> {
	if mods.shift || mods.alt || mods.command() {
		return None;
	}
	Some(UiCommand::TogglePauseLayer(focused_layer))
}

/// Maps `Tab` / `Shift + Tab` onto focus cycling.
fn map_tab(mods: KeyMods) -> Option<UiCommand> {
	if mods.alt || mods.command() {
		return None;
	}
	Some(UiCommand::CycleFocusLayer(if mods.shift { -1 } else { 1 }))
}

/// Maps `Enter` onto the selected-clip trigger.
fn map_enter(mods: KeyMods, focused_layer: LayerIndex) -> Option<UiCommand> {
	if mods.alt || mods.command() {
		return None;
	}
	Some(UiCommand::TriggerSelectedClip(focused_layer))
}

/// Maps `Escape` onto closing the overlays.
fn map_escape(mods: KeyMods) -> Option<UiCommand> {
	if mods.command() {
		return None;
	}
	Some(UiCommand::CloseOverlays)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use std::path::PathBuf;

	use super::*;
	use crate::model::AppState;
	use crate::model::clip::{ClipId, ClipInfo, ClipKind, ClipStatus};
	use crate::ui::commands::apply_ui_command;

	fn focused() -> LayerIndex {
		LayerIndex::new(1).unwrap()
	}

	/// Maps a character key that also has a physical code.
	fn by_char(text: &str, code: KeyCode, mods: ModifiersState) -> Option<UiCommand> {
		map_key_event(
			&Key::Character(text.into()),
			PhysicalKey::Code(code),
			mods,
			focused(),
			false,
			false,
		)
	}

	/// Maps a character key with no physical code (worst case fallback).
	fn by_char_only(text: &str, mods: ModifiersState) -> Option<UiCommand> {
		map_key_event(
			&Key::Character(text.into()),
			PhysicalKey::Unidentified(winit::keyboard::NativeKeyCode::Unidentified),
			mods,
			focused(),
			false,
			false,
		)
	}

	/// Maps a named key with its physical code.
	fn by_named(named: NamedKey, code: KeyCode, mods: ModifiersState) -> Option<UiCommand> {
		map_key_event(
			&Key::Named(named),
			PhysicalKey::Code(code),
			mods,
			focused(),
			false,
			false,
		)
	}

	/// Maps a key that is only reachable through its physical code.
	fn by_code(code: KeyCode, mods: ModifiersState) -> Option<UiCommand> {
		map_key_event(
			&Key::Unidentified(winit::keyboard::NativeKey::Unidentified),
			PhysicalKey::Code(code),
			mods,
			focused(),
			false,
			false,
		)
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

	#[test]
	fn plain_digits_focus_every_layer() {
		for (code, number) in DIGITS {
			let expected = UiCommand::FocusLayer(layer(number));
			assert_eq!(
				by_code(code, ModifiersState::empty()),
				Some(expected.clone()),
				"{code:?}"
			);
			assert_eq!(
				by_char(&number.to_string(), code, ModifiersState::empty()),
				Some(expected),
				"{code:?}"
			);
		}
	}

	#[test]
	fn shift_digits_trigger_every_layer_via_physical_key() {
		for (code, number) in DIGITS {
			let expected = UiCommand::TriggerLayer(layer(number));
			// US/JP layouts report `!` for Shift+1; the physical code still wins.
			let shifted_symbol = match number {
				1 => "!",
				2 => "@",
				3 => "#",
				_ => "$",
			};
			assert_eq!(
				by_char(shifted_symbol, code, ModifiersState::SHIFT),
				Some(expected.clone()),
				"{code:?}"
			);
			// Alt plus Shift is deliberately unbound, so a stuck modifier cannot
			// turn a layer trigger into a pause toggle mid-set.
			assert_eq!(
				by_char(shifted_symbol, code, ModifiersState::SHIFT | ModifiersState::ALT),
				None,
				"Shift + Alt stays free on {code:?}"
			);
		}
	}

	#[test]
	fn shift_digit_symbol_falls_back_to_the_printed_character() {
		assert_eq!(
			by_char_only("!", ModifiersState::SHIFT),
			Some(UiCommand::TriggerLayer(layer(1)))
		);
		assert_eq!(
			by_char_only("#", ModifiersState::SHIFT),
			Some(UiCommand::TriggerLayer(layer(3)))
		);
		// Unknown characters stay unmapped.
		assert_eq!(by_char_only("§", ModifiersState::SHIFT), None);
	}

	#[test]
	fn alt_digits_toggle_pause() {
		for (code, number) in DIGITS {
			assert_eq!(
				by_code(code, ModifiersState::ALT),
				Some(UiCommand::TogglePauseLayer(layer(number)))
			);
		}
	}

	#[test]
	fn digits_outside_the_layer_range_are_unmapped_without_modifiers() {
		for code in [
			KeyCode::Digit5,
			KeyCode::Digit6,
			KeyCode::Digit7,
			KeyCode::Digit8,
			KeyCode::Digit9,
			KeyCode::Digit0,
		] {
			assert_eq!(by_code(code, ModifiersState::empty()), None, "{code:?}");
			assert_eq!(by_code(code, ModifiersState::SHIFT), None, "{code:?}");
			assert_eq!(by_code(code, ModifiersState::ALT), None, "{code:?}");
		}
	}

	#[test]
	fn ctrl_digit_block_covers_the_master_effects() {
		let ctrl = ModifiersState::CONTROL;
		let ctrl_shift = ModifiersState::CONTROL | ModifiersState::SHIFT;

		assert_eq!(by_code(KeyCode::Digit1, ctrl), Some(UiCommand::ToggleRgbShift));
		assert_eq!(by_code(KeyCode::Digit2, ctrl), Some(UiCommand::ToggleGlitch));
		assert_eq!(by_code(KeyCode::Digit3, ctrl), Some(UiCommand::ToggleStrobe));
		assert_eq!(by_code(KeyCode::Digit4, ctrl), Some(UiCommand::ToggleBlackout));
		assert_eq!(by_code(KeyCode::Digit0, ctrl), Some(UiCommand::ResetAllEffects));

		assert_eq!(
			by_code(KeyCode::Digit5, ctrl),
			Some(UiCommand::AdjustMasterDimmer(MASTER_DIMMER_STEP))
		);
		assert_eq!(
			by_code(KeyCode::Digit5, ctrl_shift),
			Some(UiCommand::AdjustMasterDimmer(-MASTER_DIMMER_STEP))
		);
		assert_eq!(
			by_code(KeyCode::Digit6, ctrl),
			Some(UiCommand::AdjustBrightness(COLOR_GRADE_STEP))
		);
		assert_eq!(
			by_code(KeyCode::Digit6, ctrl_shift),
			Some(UiCommand::AdjustBrightness(-COLOR_GRADE_STEP))
		);
		assert_eq!(
			by_code(KeyCode::Digit7, ctrl),
			Some(UiCommand::AdjustContrast(COLOR_GRADE_STEP))
		);
		assert_eq!(
			by_code(KeyCode::Digit7, ctrl_shift),
			Some(UiCommand::AdjustContrast(-COLOR_GRADE_STEP))
		);
		assert_eq!(
			by_code(KeyCode::Digit8, ctrl),
			Some(UiCommand::AdjustSaturation(COLOR_GRADE_STEP))
		);
		assert_eq!(
			by_code(KeyCode::Digit8, ctrl_shift),
			Some(UiCommand::AdjustSaturation(-COLOR_GRADE_STEP))
		);
		assert_eq!(
			by_code(KeyCode::Digit9, ctrl),
			Some(UiCommand::AdjustStrobeSpeedHz(STROBE_SPEED_STEP_HZ))
		);
		assert_eq!(
			by_code(KeyCode::Digit9, ctrl_shift),
			Some(UiCommand::AdjustStrobeSpeedHz(-STROBE_SPEED_STEP_HZ))
		);
	}

	#[test]
	fn layer_letter_shortcuts() {
		let empty = ModifiersState::empty();
		let shift = ModifiersState::SHIFT;

		assert_eq!(
			by_char("r", KeyCode::KeyR, empty),
			Some(UiCommand::RetriggerLayer(focused()))
		);
		assert_eq!(
			by_char("R", KeyCode::KeyR, shift),
			Some(UiCommand::RetriggerAllActiveLayers)
		);
		assert_eq!(
			by_char("l", KeyCode::KeyL, empty),
			Some(UiCommand::ToggleLayerPlaybackMode(focused()))
		);
		assert_eq!(
			by_char("x", KeyCode::KeyX, empty),
			Some(UiCommand::ClearLayer(focused()))
		);
		assert_eq!(
			by_char("q", KeyCode::KeyQ, empty),
			Some(UiCommand::AdjustLayerOpacity {
				layer: focused(),
				delta: -LAYER_OPACITY_STEP,
			})
		);
		assert_eq!(
			by_char("w", KeyCode::KeyW, empty),
			Some(UiCommand::AdjustLayerOpacity {
				layer: focused(),
				delta: LAYER_OPACITY_STEP,
			})
		);
		assert_eq!(
			by_char("a", KeyCode::KeyA, empty),
			Some(UiCommand::AdjustLayerSpeed {
				layer: focused(),
				delta: -LAYER_SPEED_STEP,
			})
		);
		assert_eq!(
			by_char("d", KeyCode::KeyD, empty),
			Some(UiCommand::AdjustLayerSpeed {
				layer: focused(),
				delta: LAYER_SPEED_STEP,
			})
		);
		assert_eq!(
			by_char("s", KeyCode::KeyS, empty),
			Some(UiCommand::SetLayerSpeed {
				layer: focused(),
				speed: 1.0,
			})
		);
		assert_eq!(by_char("S", KeyCode::KeyS, shift), Some(UiCommand::ResetAllLayerSpeeds));
		assert_eq!(
			by_char("b", KeyCode::KeyB, empty),
			Some(UiCommand::CycleLayerBlendMode {
				layer: focused(),
				forward: true,
			})
		);
		assert_eq!(
			by_char("B", KeyCode::KeyB, shift),
			Some(UiCommand::CycleLayerBlendMode {
				layer: focused(),
				forward: false,
			})
		);
	}

	#[test]
	fn caps_lock_style_uppercase_letters_still_map() {
		// Caps Lock reports an uppercase character without the Shift modifier.
		assert_eq!(
			by_char("G", KeyCode::KeyG, ModifiersState::empty()),
			Some(UiCommand::ToggleLogsDrawer)
		);
		assert_eq!(
			by_char("R", KeyCode::KeyR, ModifiersState::empty()),
			Some(UiCommand::RetriggerLayer(focused()))
		);
	}

	#[test]
	fn unused_letters_and_modifier_combinations_stay_unmapped() {
		for (text, code) in [
			("e", KeyCode::KeyE),
			("f", KeyCode::KeyF),
			("y", KeyCode::KeyY),
			("z", KeyCode::KeyZ),
			("o", KeyCode::KeyO),
		] {
			assert_eq!(by_char(text, code, ModifiersState::empty()), None, "{text}");
		}
		assert_eq!(by_char("q", KeyCode::KeyQ, ModifiersState::SHIFT), None);
		assert_eq!(by_char("x", KeyCode::KeyX, ModifiersState::SHIFT), None);
		assert_eq!(by_char("l", KeyCode::KeyL, ModifiersState::ALT), None);
		assert_eq!(by_char("g", KeyCode::KeyG, ModifiersState::CONTROL), None);
	}

	#[test]
	fn overlay_and_drawer_letters() {
		assert_eq!(
			by_char("h", KeyCode::KeyH, ModifiersState::empty()),
			Some(UiCommand::ToggleShortcutsOverlay)
		);
		assert_eq!(
			by_char("H", KeyCode::KeyH, ModifiersState::SHIFT),
			Some(UiCommand::ToggleShortcutsOverlay)
		);
		assert_eq!(
			by_char("?", KeyCode::Slash, ModifiersState::SHIFT),
			Some(UiCommand::ToggleShortcutsOverlay)
		);
		assert_eq!(
			by_char("g", KeyCode::KeyG, ModifiersState::empty()),
			Some(UiCommand::ToggleLogsDrawer)
		);
	}

	#[test]
	fn output_toggle_needs_control_or_command() {
		assert_eq!(
			by_char("o", KeyCode::KeyO, ModifiersState::CONTROL),
			Some(UiCommand::ToggleOutputEnabled)
		);
		assert_eq!(
			by_char("o", KeyCode::KeyO, ModifiersState::SUPER),
			Some(UiCommand::ToggleOutputEnabled)
		);
		assert_eq!(by_char("o", KeyCode::KeyO, ModifiersState::empty()), None);
	}

	#[test]
	fn quit_request_goes_through_the_guard() {
		assert_eq!(
			by_char("q", KeyCode::KeyQ, ModifiersState::SUPER),
			Some(UiCommand::RequestExit)
		);
		assert_eq!(
			by_char("q", KeyCode::KeyQ, ModifiersState::CONTROL),
			Some(UiCommand::RequestExit),
			"Ctrl + Q must request exit, not layer opacity"
		);
		assert_eq!(
			by_char("q", KeyCode::KeyQ, ModifiersState::CONTROL | ModifiersState::SHIFT),
			Some(UiCommand::RequestExit),
			"Ctrl + Shift + Q is still a quit request"
		);
		assert_eq!(
			by_char("q", KeyCode::KeyQ, ModifiersState::SUPER | ModifiersState::ALT),
			None
		);
		assert_eq!(
			by_named(NamedKey::F4, KeyCode::F4, ModifiersState::ALT),
			Some(UiCommand::RequestExit)
		);
		assert_eq!(by_code(KeyCode::NumpadEnter, ModifiersState::ALT), None);
	}

	#[test]
	fn plain_function_keys_ignore_stray_modifiers() {
		// F4 / F12 only fire unmodified; Alt + F4 requests exit instead.
		assert_eq!(by_code(KeyCode::F4, ModifiersState::SHIFT), None);
		assert_eq!(by_code(KeyCode::F12, ModifiersState::SHIFT), None);
		assert_eq!(by_code(KeyCode::F12, ModifiersState::CONTROL), None);
		assert_eq!(
			by_code(KeyCode::F4, ModifiersState::empty()),
			Some(UiCommand::ToggleBlackout)
		);
		assert_eq!(
			by_code(KeyCode::F12, ModifiersState::empty()),
			Some(UiCommand::ResetAllEffects)
		);
	}

	#[test]
	fn function_keys_cover_the_master_effects() {
		let empty = ModifiersState::empty();
		let shift = ModifiersState::SHIFT;
		let alt = ModifiersState::ALT;

		assert_eq!(by_code(KeyCode::F1, empty), Some(UiCommand::ToggleRgbShift));
		assert_eq!(
			by_code(KeyCode::F1, shift),
			Some(UiCommand::AdjustRgbShiftIntensity(EFFECT_INTENSITY_STEP))
		);
		assert_eq!(
			by_code(KeyCode::F1, alt),
			Some(UiCommand::AdjustRgbShiftIntensity(-EFFECT_INTENSITY_STEP))
		);
		assert_eq!(by_code(KeyCode::F2, empty), Some(UiCommand::ToggleGlitch));
		assert_eq!(
			by_code(KeyCode::F2, shift),
			Some(UiCommand::AdjustGlitchIntensity(EFFECT_INTENSITY_STEP))
		);
		assert_eq!(
			by_code(KeyCode::F2, alt),
			Some(UiCommand::AdjustGlitchIntensity(-EFFECT_INTENSITY_STEP))
		);
		assert_eq!(by_code(KeyCode::F3, empty), Some(UiCommand::ToggleStrobe));
		assert_eq!(
			by_code(KeyCode::F3, shift),
			Some(UiCommand::AdjustStrobeIntensity(EFFECT_INTENSITY_STEP))
		);
		assert_eq!(
			by_code(KeyCode::F3, alt),
			Some(UiCommand::AdjustStrobeIntensity(-EFFECT_INTENSITY_STEP))
		);
		assert_eq!(by_code(KeyCode::F4, empty), Some(UiCommand::ToggleBlackout));
		assert_eq!(
			by_code(KeyCode::F5, empty),
			Some(UiCommand::AdjustMasterDimmer(-MASTER_DIMMER_STEP))
		);
		assert_eq!(
			by_code(KeyCode::F6, empty),
			Some(UiCommand::AdjustMasterDimmer(MASTER_DIMMER_STEP))
		);
		assert_eq!(
			by_code(KeyCode::F7, empty),
			Some(UiCommand::AdjustBrightness(COLOR_GRADE_STEP))
		);
		assert_eq!(
			by_code(KeyCode::F7, shift),
			Some(UiCommand::AdjustBrightness(-COLOR_GRADE_STEP))
		);
		assert_eq!(
			by_code(KeyCode::F8, empty),
			Some(UiCommand::AdjustContrast(COLOR_GRADE_STEP))
		);
		assert_eq!(
			by_code(KeyCode::F8, shift),
			Some(UiCommand::AdjustContrast(-COLOR_GRADE_STEP))
		);
		assert_eq!(
			by_code(KeyCode::F9, empty),
			Some(UiCommand::AdjustSaturation(COLOR_GRADE_STEP))
		);
		assert_eq!(
			by_code(KeyCode::F9, shift),
			Some(UiCommand::AdjustSaturation(-COLOR_GRADE_STEP))
		);
		assert_eq!(
			by_code(KeyCode::F10, empty),
			Some(UiCommand::AdjustStrobeSpeedHz(STROBE_SPEED_STEP_HZ))
		);
		assert_eq!(
			by_code(KeyCode::F10, shift),
			Some(UiCommand::AdjustStrobeSpeedHz(-STROBE_SPEED_STEP_HZ))
		);
		assert_eq!(by_code(KeyCode::F11, empty), None);
		assert_eq!(by_code(KeyCode::F12, empty), Some(UiCommand::ResetAllEffects));
		assert_eq!(
			by_code(KeyCode::F12, ModifiersState::CONTROL),
			None,
			"Cmd/Ctrl + F12 is reserved for the OS"
		);
	}

	#[test]
	fn backquote_is_an_alternative_blackout_key() {
		assert_eq!(
			by_char("`", KeyCode::Backquote, ModifiersState::empty()),
			Some(UiCommand::ToggleBlackout)
		);
		assert_eq!(
			by_code(KeyCode::Backquote, ModifiersState::SHIFT),
			Some(UiCommand::ToggleBlackout)
		);
		assert_eq!(by_code(KeyCode::Backquote, ModifiersState::ALT), None);
	}

	#[test]
	fn bracket_keys_step_the_clip_selection() {
		assert_eq!(
			by_char("[", KeyCode::BracketLeft, ModifiersState::empty()),
			Some(UiCommand::StepLayerClip {
				layer: focused(),
				delta: -1,
			})
		);
		assert_eq!(
			by_char("]", KeyCode::BracketRight, ModifiersState::empty()),
			Some(UiCommand::StepLayerClip {
				layer: focused(),
				delta: 1,
			})
		);
		// Shifted forms and physical-only reports agree.
		assert_eq!(
			by_char("{", KeyCode::BracketLeft, ModifiersState::SHIFT),
			Some(UiCommand::StepLayerClip {
				layer: focused(),
				delta: -1,
			})
		);
		assert_eq!(
			by_code(KeyCode::BracketRight, ModifiersState::empty()),
			Some(UiCommand::StepLayerClip {
				layer: focused(),
				delta: 1,
			})
		);
		assert_eq!(by_code(KeyCode::BracketLeft, ModifiersState::CONTROL), None);
	}

	#[test]
	fn transport_named_keys() {
		assert_eq!(
			by_named(NamedKey::Space, KeyCode::Space, ModifiersState::empty()),
			Some(UiCommand::TogglePauseLayer(focused()))
		);
		assert_eq!(by_named(NamedKey::Space, KeyCode::Space, ModifiersState::SHIFT), None);
		assert_eq!(
			by_named(NamedKey::Enter, KeyCode::Enter, ModifiersState::empty()),
			Some(UiCommand::TriggerSelectedClip(focused()))
		);
		assert_eq!(
			by_code(KeyCode::NumpadEnter, ModifiersState::empty()),
			Some(UiCommand::TriggerSelectedClip(focused()))
		);
		assert_eq!(
			by_named(NamedKey::Tab, KeyCode::Tab, ModifiersState::empty()),
			Some(UiCommand::CycleFocusLayer(1))
		);
		assert_eq!(
			by_named(NamedKey::Tab, KeyCode::Tab, ModifiersState::SHIFT),
			Some(UiCommand::CycleFocusLayer(-1))
		);
		assert_eq!(
			by_named(NamedKey::Escape, KeyCode::Escape, ModifiersState::empty()),
			Some(UiCommand::CloseOverlays)
		);
	}

	#[test]
	fn unmapped_keys_and_modifiers_return_none() {
		assert_eq!(
			by_code(KeyCode::KeyM, ModifiersState::CONTROL | ModifiersState::SHIFT),
			None
		);
		assert_eq!(by_code(KeyCode::ArrowUp, ModifiersState::empty()), None);
		assert_eq!(by_code(KeyCode::CapsLock, ModifiersState::empty()), None);
		assert_eq!(
			map_key_event(
				&Key::Dead(Some('^')),
				PhysicalKey::Code(KeyCode::Equal),
				ModifiersState::empty(),
				focused(),
				false,
				false,
			),
			None
		);
	}

	/// Maps a key while the quit confirmation dialog is open.
	fn guarded(key: &Key, physical: PhysicalKey, mods: ModifiersState) -> Option<UiCommand> {
		map_key_event(key, physical, mods, focused(), true, false)
	}

	#[test]
	fn exit_guard_swallows_everything_but_the_quit_key_and_escape() {
		// `Enter` triggers clips all night; it must never close the show.
		for (key, code) in [
			(Key::Named(NamedKey::Enter), KeyCode::Enter),
			(Key::Named(NamedKey::Enter), KeyCode::NumpadEnter),
			(Key::Character("\r".into()), KeyCode::Enter),
			(Key::Character("\r".into()), KeyCode::NumpadEnter),
		] {
			assert_eq!(
				guarded(&key, PhysicalKey::Code(code), ModifiersState::empty()),
				None,
				"{code:?} must not confirm the exit dialog"
			);
		}

		// `Escape` cancels.
		assert_eq!(
			guarded(
				&Key::Named(NamedKey::Escape),
				PhysicalKey::Code(KeyCode::Escape),
				ModifiersState::empty()
			),
			Some(UiCommand::CancelExit)
		);

		// Repeating the quit gesture confirms.
		for (key, code, mods) in [
			(Key::Character("q".into()), KeyCode::KeyQ, ModifiersState::SUPER),
			(Key::Character("q".into()), KeyCode::KeyQ, ModifiersState::CONTROL),
			(Key::Named(NamedKey::F4), KeyCode::F4, ModifiersState::ALT),
		] {
			assert_eq!(
				guarded(&key, PhysicalKey::Code(code), mods),
				Some(UiCommand::ConfirmExit),
				"{code:?} + {mods:?} must confirm"
			);
		}

		// A bare `Q` (opacity) is not a quit gesture, so it is swallowed.
		assert_eq!(
			guarded(
				&Key::Character("q".into()),
				PhysicalKey::Code(KeyCode::KeyQ),
				ModifiersState::empty()
			),
			None
		);

		// Live controls must not leak through the guard.
		for code in [
			KeyCode::Digit1,
			KeyCode::KeyR,
			KeyCode::F4,
			KeyCode::F12,
			KeyCode::Space,
			KeyCode::Backquote,
			KeyCode::KeyX,
		] {
			assert_eq!(
				guarded(
					&Key::Unidentified(winit::keyboard::NativeKey::Unidentified),
					PhysicalKey::Code(code),
					ModifiersState::empty()
				),
				None,
				"{code:?} must be blocked by the exit guard"
			);
		}
	}

	#[test]
	fn egui_keyboard_capture_suppresses_every_shortcut() {
		for code in [KeyCode::Digit1, KeyCode::KeyR, KeyCode::F1, KeyCode::Space] {
			assert_eq!(
				map_key_event(
					&Key::Unidentified(winit::keyboard::NativeKey::Unidentified),
					PhysicalKey::Code(code),
					ModifiersState::empty(),
					focused(),
					false,
					true,
				),
				None,
				"{code:?}"
			);
		}
		// The exit guard is checked first, so its dialog stays answerable.
		assert_eq!(
			map_key_event(
				&Key::Named(NamedKey::Escape),
				PhysicalKey::Code(KeyCode::Escape),
				ModifiersState::empty(),
				focused(),
				true,
				true,
			),
			Some(UiCommand::CancelExit),
			"the exit guard outranks egui capture"
		);
	}

	#[test]
	fn shortcuts_never_shadow_each_other() {
		// Every distinct plain key must produce a distinct command.
		let letters = [
			(KeyCode::KeyR, "r"),
			(KeyCode::KeyS, "s"),
			(KeyCode::KeyB, "b"),
			(KeyCode::KeyQ, "q"),
			(KeyCode::KeyW, "w"),
			(KeyCode::KeyA, "a"),
			(KeyCode::KeyD, "d"),
			(KeyCode::KeyL, "l"),
			(KeyCode::KeyX, "x"),
			(KeyCode::KeyG, "g"),
			(KeyCode::KeyH, "h"),
		];
		let mut seen: Vec<UiCommand> = Vec::new();
		for (code, text) in letters {
			let command = by_char(text, code, ModifiersState::empty()).unwrap();
			assert!(!seen.contains(&command), "{text} collides with another shortcut");
			seen.push(command);
		}
		assert_eq!(seen.len(), letters.len());
	}

	#[test]
	fn legend_is_complete_and_readable() {
		assert!(SHORTCUT_LEGEND.len() >= 30);
		for entry in SHORTCUT_LEGEND {
			assert!(!entry.keys.trim().is_empty());
			assert!(!entry.description.trim().is_empty());
		}
		let mut keys: Vec<&str> = SHORTCUT_LEGEND.iter().map(|entry| entry.keys).collect();
		keys.sort_unstable();
		keys.dedup();
		assert_eq!(keys.len(), SHORTCUT_LEGEND.len(), "duplicate legend rows");
	}

	/// Commands produced by every gesture documented in one `SHORTCUT_LEGEND` row.
	///
	/// `None` means the test does not know the row yet, which makes adding a legend
	/// entry without a real mapping fail [`every_legend_row_maps_to_a_command`].
	fn legend_commands(keys: &str) -> Option<Vec<Option<UiCommand>>> {
		let plain = ModifiersState::empty();
		let shift = ModifiersState::SHIFT;
		let alt = ModifiersState::ALT;
		let control = ModifiersState::CONTROL;
		let command = ModifiersState::SUPER;

		let chars = |pairs: &[(&str, KeyCode, ModifiersState)]| -> Vec<Option<UiCommand>> {
			pairs
				.iter()
				.map(|(text, code, mods)| by_char(text, *code, *mods))
				.collect()
		};
		let codes = |pairs: &[(KeyCode, ModifiersState)]| -> Vec<Option<UiCommand>> {
			pairs.iter().map(|(code, mods)| by_code(*code, *mods)).collect()
		};
		let layers = |mods: ModifiersState| -> Vec<Option<UiCommand>> {
			DIGITS.iter().map(|(code, _)| by_code(*code, mods)).collect()
		};

		let commands = match keys {
			"1 / 2 / 3 / 4" => layers(plain),
			"Shift + 1-4" => layers(shift),
			"Alt + 1-4" => layers(alt),
			"Tab / Shift + Tab" => vec![
				by_named(NamedKey::Tab, KeyCode::Tab, plain),
				by_named(NamedKey::Tab, KeyCode::Tab, shift),
			],
			"R" => chars(&[("r", KeyCode::KeyR, plain)]),
			"Shift + R" => chars(&[("R", KeyCode::KeyR, shift)]),
			"Space" => vec![by_named(NamedKey::Space, KeyCode::Space, plain)],
			"L" => chars(&[("l", KeyCode::KeyL, plain)]),
			"X" => chars(&[("x", KeyCode::KeyX, plain)]),
			"Q / W" => chars(&[("q", KeyCode::KeyQ, plain), ("w", KeyCode::KeyW, plain)]),
			"A / D" => chars(&[("a", KeyCode::KeyA, plain), ("d", KeyCode::KeyD, plain)]),
			"S" => chars(&[("s", KeyCode::KeyS, plain)]),
			"Shift + S" => chars(&[("S", KeyCode::KeyS, shift)]),
			"B / Shift + B" => chars(&[("b", KeyCode::KeyB, plain), ("B", KeyCode::KeyB, shift)]),
			"[ / ]" => chars(&[("[", KeyCode::BracketLeft, plain), ("]", KeyCode::BracketRight, plain)]),
			"Enter" => vec![
				by_named(NamedKey::Enter, KeyCode::Enter, plain),
				by_code(KeyCode::NumpadEnter, plain),
			],
			"F1" => codes(&[(KeyCode::F1, plain)]),
			"Shift + F1 / Alt + F1" => codes(&[(KeyCode::F1, shift), (KeyCode::F1, alt)]),
			"F2" => codes(&[(KeyCode::F2, plain)]),
			"Shift + F2 / Alt + F2" => codes(&[(KeyCode::F2, shift), (KeyCode::F2, alt)]),
			"F3" => codes(&[(KeyCode::F3, plain)]),
			"Shift + F3 / Alt + F3" => codes(&[(KeyCode::F3, shift), (KeyCode::F3, alt)]),
			"F4 or `" => vec![by_code(KeyCode::F4, plain), by_char("`", KeyCode::Backquote, plain)],
			"F5 / F6" => codes(&[(KeyCode::F5, plain), (KeyCode::F6, plain)]),
			"F7 / Shift + F7" => codes(&[(KeyCode::F7, plain), (KeyCode::F7, shift)]),
			"F8 / Shift + F8" => codes(&[(KeyCode::F8, plain), (KeyCode::F8, shift)]),
			"F9 / Shift + F9" => codes(&[(KeyCode::F9, plain), (KeyCode::F9, shift)]),
			"F10 / Shift + F10" => codes(&[(KeyCode::F10, plain), (KeyCode::F10, shift)]),
			"F12 (Ctrl/Cmd + F12 reserved)" => codes(&[(KeyCode::F12, plain)]),
			"Ctrl + 1 / 2 / 3 / 4" => codes(&[
				(KeyCode::Digit1, control),
				(KeyCode::Digit2, control),
				(KeyCode::Digit3, control),
				(KeyCode::Digit4, control),
			]),
			"Ctrl + 5-9, Ctrl + 0" => codes(&[
				(KeyCode::Digit5, control),
				(KeyCode::Digit6, control),
				(KeyCode::Digit7, control),
				(KeyCode::Digit8, control),
				(KeyCode::Digit9, control),
				(KeyCode::Digit0, control),
			]),
			"Ctrl + O or Cmd + O" => chars(&[("o", KeyCode::KeyO, control), ("o", KeyCode::KeyO, command)]),
			"H or ?" => chars(&[("h", KeyCode::KeyH, plain), ("?", KeyCode::Slash, shift)]),
			"G" => chars(&[("g", KeyCode::KeyG, plain)]),
			"Escape" => vec![by_named(NamedKey::Escape, KeyCode::Escape, plain)],
			"Cmd + Q / Ctrl + Q / Alt + F4" => vec![
				by_char("q", KeyCode::KeyQ, command),
				by_char("q", KeyCode::KeyQ, control),
				by_named(NamedKey::F4, KeyCode::F4, alt),
			],
			"Repeat quit key / Escape (in dialog)" => vec![
				guarded(&Key::Character("q".into()), PhysicalKey::Code(KeyCode::KeyQ), command),
				guarded(&Key::Named(NamedKey::Escape), PhysicalKey::Code(KeyCode::Escape), plain),
			],
			_ => return None,
		};
		Some(commands)
	}

	#[test]
	fn every_legend_row_maps_to_a_command() {
		let mut uncovered: Vec<&str> = Vec::new();
		for entry in SHORTCUT_LEGEND {
			let Some(commands) = legend_commands(entry.keys) else {
				uncovered.push(entry.keys);
				continue;
			};
			assert!(!commands.is_empty(), "{} has no probe", entry.keys);
			for (index, command) in commands.iter().enumerate() {
				assert!(
					command.is_some(),
					"`{}` gesture #{index} is documented but not mapped (`{}`)",
					entry.keys,
					entry.description
				);
			}
		}
		assert!(uncovered.is_empty(), "undocumented probes: {uncovered:?}");
	}

	#[test]
	fn mapped_commands_apply_cleanly_to_the_state() {
		let mut state = AppState::default();
		state.upsert_clip(ClipInfo {
			id: ClipId(1),
			path: PathBuf::from("/clips/one.mov"),
			name: "one.mov".to_string(),
			kind: ClipKind::Hap,
			width: 1920,
			height: 1080,
			fps: 30.0,
			frame_count: 30,
			has_alpha: false,
			status: ClipStatus::Ready,
			thumbnail: None,
		});

		let target = focused();
		let sequence = [
			("]", KeyCode::BracketRight, ModifiersState::empty()),
			("q", KeyCode::KeyQ, ModifiersState::empty()),
			("d", KeyCode::KeyD, ModifiersState::empty()),
			("b", KeyCode::KeyB, ModifiersState::empty()),
			("`", KeyCode::Backquote, ModifiersState::empty()),
			("1", KeyCode::Digit1, ModifiersState::empty()),
		];
		for (text, code, mods) in sequence {
			let command = by_char(text, code, mods).unwrap();
			let outcome = apply_ui_command(&mut state, &command);
			assert!(
				outcome.has_no_side_effects(),
				"{text} should not need a streamer restart"
			);
		}

		assert_eq!(state.layer(target).selected_clip, Some(ClipId(1)));
		assert_eq!(state.layer(target).active_clip, None);
		assert!((state.layer(target).opacity - 0.95).abs() < 1e-5);
		assert!((state.layer(target).speed - 1.1).abs() < 1e-5);
		assert_eq!(state.layer(target).blend_mode, crate::model::BlendMode::Add);
		assert!(state.effects.blackout);
		assert_eq!(state.focused_layer, layer(1));

		// The trigger key then starts the selected clip for real.
		let trigger = by_named(NamedKey::Enter, KeyCode::Enter, ModifiersState::empty()).unwrap();
		let outcome = apply_ui_command(&mut state, &trigger);
		assert!(outcome.was_retriggered(target));
		assert_eq!(state.layer(target).active_clip, Some(ClipId(1)));
	}
}
