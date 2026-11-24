//! Master shader effect parameters and their MIDI CC conversions.

use serde::{Deserialize, Serialize};

/// Default strobe frequency in hertz.
pub const DEFAULT_STROBE_HZ: f32 = 8.0;
/// Slowest strobe frequency reachable from MIDI.
pub const MIN_STROBE_HZ: f32 = 1.0;
/// Fastest strobe frequency reachable from MIDI.
pub const MAX_STROBE_HZ: f32 = 30.0;

/// Master effects applied to the composited output of all layers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MasterEffects {
	/// Whether the RGB shift effect is active.
	pub rgb_shift_enabled: bool,
	/// RGB shift strength in `0.0..=1.0`.
	pub rgb_shift_intensity: f32,
	/// Whether the glitch/block-noise effect is active.
	pub glitch_enabled: bool,
	/// Glitch strength in `0.0..=1.0`.
	pub glitch_intensity: f32,
	/// Whether the strobe/color-invert effect is active.
	pub strobe_enabled: bool,
	/// Strobe intensity in `0.0..=1.0`.
	pub strobe_intensity: f32,
	/// Strobe frequency in hertz, `1.0..=30.0`.
	pub strobe_speed_hz: f32,
	/// Brightness offset in `-1.0..=1.0`, `0.0` is neutral.
	pub brightness: f32,
	/// Contrast gain in `0.0..=2.0`, `1.0` is neutral.
	pub contrast: f32,
	/// Saturation gain in `0.0..=2.0`, `1.0` is neutral.
	pub saturation: f32,
	/// Master output gain in `0.0..=1.0`.
	pub master_dimmer: f32,
	/// Emergency blackout; silences the output without touching other values.
	pub blackout: bool,
}

impl Default for MasterEffects {
	fn default() -> Self {
		Self {
			rgb_shift_enabled: false,
			rgb_shift_intensity: 0.0,
			glitch_enabled: false,
			glitch_intensity: 0.0,
			strobe_enabled: false,
			strobe_intensity: 0.0,
			strobe_speed_hz: DEFAULT_STROBE_HZ,
			brightness: 0.0,
			contrast: 1.0,
			saturation: 1.0,
			master_dimmer: 1.0,
			blackout: false,
		}
	}
}

impl MasterEffects {
	/// Restores every effect parameter to its neutral value.
	///
	/// [`MasterEffects::master_dimmer`] and [`MasterEffects::blackout`] are
	/// deliberately preserved so a panic hit on the reset button cannot black out
	/// the output or blast it at full gain during a live set.
	pub fn reset_effects(&mut self) {
		self.rgb_shift_enabled = false;
		self.rgb_shift_intensity = 0.0;
		self.glitch_enabled = false;
		self.glitch_intensity = 0.0;
		self.strobe_enabled = false;
		self.strobe_intensity = 0.0;
		self.strobe_speed_hz = DEFAULT_STROBE_HZ;
		self.brightness = 0.0;
		self.contrast = 1.0;
		self.saturation = 1.0;
	}

	/// Clamps every parameter into its valid range, replacing non-finite values.
	pub fn clamp_all(&mut self) {
		self.rgb_shift_intensity = sanitize(self.rgb_shift_intensity, 0.0, 1.0, 0.0);
		self.glitch_intensity = sanitize(self.glitch_intensity, 0.0, 1.0, 0.0);
		self.strobe_intensity = sanitize(self.strobe_intensity, 0.0, 1.0, 0.0);
		self.strobe_speed_hz = sanitize(self.strobe_speed_hz, MIN_STROBE_HZ, MAX_STROBE_HZ, DEFAULT_STROBE_HZ);
		self.brightness = sanitize(self.brightness, -1.0, 1.0, 0.0);
		self.contrast = sanitize(self.contrast, 0.0, 2.0, 1.0);
		self.saturation = sanitize(self.saturation, 0.0, 2.0, 1.0);
		self.master_dimmer = sanitize(self.master_dimmer, 0.0, 1.0, 1.0);
	}
}

/// Clamps `value` into `min..=max`, substituting `fallback` for NaN/infinity.
fn sanitize(value: f32, min: f32, max: f32, fallback: f32) -> f32 {
	if value.is_finite() {
		value.clamp(min, max)
	} else {
		fallback
	}
}

/// Converts a 7-bit controller value into `0.0..=1.0`.
///
/// Values above `127` (impossible for standard MIDI but common in tests and
/// malformed input) are clamped to `127`.
pub fn cc_to_unit(value: u8) -> f32 {
	f32::from(value.min(127)) / 127.0
}

/// Converts a controller value into a brightness offset in `-1.0..=1.0`.
///
/// `0` maps to `-1.0`, `64` to the neutral `0.0`, and `127` to `+1.0`.
pub fn cc_to_brightness(value: u8) -> f32 {
	let value = value.min(127);
	let raw = f32::from(value);
	if value <= 64 {
		-1.0 + raw / 64.0
	} else {
		(raw - 64.0) / 63.0
	}
}

/// Converts a controller value into a centered gain factor in `0.0..=2.0`.
///
/// Used for contrast and saturation: `0` maps to `0.0`, `64` to the neutral
/// `1.0`, and `127` to `2.0`.
pub fn cc_to_factor_centered(value: u8) -> f32 {
	let value = value.min(127);
	let raw = f32::from(value);
	if value <= 64 {
		raw / 64.0
	} else {
		1.0 + (raw - 64.0) / 63.0
	}
}

/// Converts a controller value into a strobe frequency in `1.0..=30.0` hertz.
pub fn cc_to_strobe_hz(value: u8) -> f32 {
	MIN_STROBE_HZ * (MAX_STROBE_HZ / MIN_STROBE_HZ).powf(cc_to_unit(value))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;

	fn approx(actual: f32, expected: f32) {
		assert!((actual - expected).abs() < 1e-5, "expected {expected}, got {actual}");
	}

	#[test]
	fn defaults_are_neutral() {
		let effects = MasterEffects::default();
		assert!(!effects.blackout);
		approx(effects.master_dimmer, 1.0);
		approx(effects.brightness, 0.0);
		approx(effects.contrast, 1.0);
		approx(effects.saturation, 1.0);
		approx(effects.strobe_speed_hz, 8.0);
		assert!(!effects.rgb_shift_enabled);
	}

	#[test]
	fn unit_conversion_clamps_above_127() {
		approx(cc_to_unit(0), 0.0);
		approx(cc_to_unit(127), 1.0);
		approx(cc_to_unit(255), 1.0);
		approx(cc_to_unit(64), 64.0 / 127.0);
	}

	#[test]
	fn brightness_matches_spec_anchors() {
		approx(cc_to_brightness(0), -1.0);
		approx(cc_to_brightness(64), 0.0);
		approx(cc_to_brightness(127), 1.0);
		approx(cc_to_brightness(255), 1.0);
		assert!(cc_to_brightness(32) < 0.0);
		assert!(cc_to_brightness(96) > 0.0);
	}

	#[test]
	fn centered_factor_matches_spec_anchors() {
		approx(cc_to_factor_centered(0), 0.0);
		approx(cc_to_factor_centered(64), 1.0);
		approx(cc_to_factor_centered(127), 2.0);
		approx(cc_to_factor_centered(255), 2.0);
	}

	#[test]
	fn strobe_frequency_spans_one_to_thirty_hz() {
		approx(cc_to_strobe_hz(0), MIN_STROBE_HZ);
		approx(cc_to_strobe_hz(127), MAX_STROBE_HZ);
		approx(cc_to_strobe_hz(255), MAX_STROBE_HZ);
		assert!(cc_to_strobe_hz(64) > MIN_STROBE_HZ && cc_to_strobe_hz(64) < MAX_STROBE_HZ);
	}

	#[test]
	fn reset_clears_effects_but_preserves_live_safety_fields() {
		let mut effects = MasterEffects {
			rgb_shift_enabled: true,
			rgb_shift_intensity: 0.8,
			glitch_enabled: true,
			glitch_intensity: 0.5,
			strobe_enabled: true,
			strobe_intensity: 0.9,
			strobe_speed_hz: 24.0,
			brightness: 0.4,
			contrast: 1.8,
			saturation: 0.2,
			master_dimmer: 0.35,
			blackout: true,
		};

		effects.reset_effects();

		assert!(!effects.rgb_shift_enabled);
		approx(effects.rgb_shift_intensity, 0.0);
		assert!(!effects.glitch_enabled);
		approx(effects.glitch_intensity, 0.0);
		assert!(!effects.strobe_enabled);
		approx(effects.strobe_intensity, 0.0);
		approx(effects.strobe_speed_hz, DEFAULT_STROBE_HZ);
		approx(effects.brightness, 0.0);
		approx(effects.contrast, 1.0);
		approx(effects.saturation, 1.0);
		approx(effects.master_dimmer, 0.35);
		assert!(effects.blackout);
	}

	#[test]
	fn clamp_all_bounds_and_sanitizes() {
		let mut effects = MasterEffects {
			rgb_shift_intensity: 5.0,
			glitch_intensity: -2.0,
			strobe_intensity: f32::NAN,
			strobe_speed_hz: 1_000.0,
			brightness: -9.0,
			contrast: f32::INFINITY,
			saturation: 7.0,
			master_dimmer: -0.5,
			..MasterEffects::default()
		};
		effects.clamp_all();

		approx(effects.rgb_shift_intensity, 1.0);
		approx(effects.glitch_intensity, 0.0);
		approx(effects.strobe_intensity, 0.0);
		approx(effects.strobe_speed_hz, MAX_STROBE_HZ);
		approx(effects.brightness, -1.0);
		approx(effects.contrast, 1.0);
		approx(effects.saturation, 2.0);
		approx(effects.master_dimmer, 0.0);
	}

	#[test]
	fn deserializing_partial_object_uses_neutral_defaults() {
		let effects: MasterEffects = serde_json::from_str(r#"{"blackout":true}"#).unwrap();
		assert!(effects.blackout);
		approx(effects.master_dimmer, 1.0);
		approx(effects.contrast, 1.0);
		approx(effects.strobe_speed_hz, DEFAULT_STROBE_HZ);
	}
}
