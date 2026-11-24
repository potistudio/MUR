//! Layer blend modes and their MIDI CC mapping.

use serde::{Deserialize, Serialize};

/// Compositing mode applied when a layer is blended onto the master output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum BlendMode {
	/// Standard source-over compositing.
	#[default]
	Normal,
	/// Additive blending, brightening the result.
	Add,
	/// Multiplicative blending, darkening the result.
	Multiply,
	/// Inverse-multiply blending, lifting shadows.
	Screen,
	/// Absolute difference between source and destination.
	Difference,
}

impl BlendMode {
	/// Every blend mode in MIDI CC order.
	pub const ALL: [Self; 5] = [Self::Normal, Self::Add, Self::Multiply, Self::Screen, Self::Difference];

	/// Maps a controller value in `0..=127` onto one of the five modes.
	///
	/// The range is split evenly: `0..=25` Normal, `26..=51` Add, `52..=76`
	/// Multiply, `77..=102` Screen, and every larger value Difference.
	pub fn from_midi_cc(value: u8) -> Self {
		match value {
			0..=25 => Self::Normal,
			26..=51 => Self::Add,
			52..=76 => Self::Multiply,
			77..=102 => Self::Screen,
			_ => Self::Difference,
		}
	}

	/// Stable discriminator used for indexing and persisting.
	pub fn as_u32(self) -> u32 {
		match self {
			Self::Normal => 0,
			Self::Add => 1,
			Self::Multiply => 2,
			Self::Screen => 3,
			Self::Difference => 4,
		}
	}

	/// Human readable name for the control UI.
	pub fn label(self) -> &'static str {
		match self {
			Self::Normal => "Normal",
			Self::Add => "Add",
			Self::Multiply => "Multiply",
			Self::Screen => "Screen",
			Self::Difference => "Difference",
		}
	}

	/// Next mode in the cycle, wrapping back to [`BlendMode::Normal`].
	pub fn next(self) -> Self {
		let count = Self::ALL.len();
		let index = (self.as_u32() as usize + 1) % count;
		Self::ALL.get(index).copied().unwrap_or(self)
	}

	/// Previous mode in the cycle, wrapping around to [`BlendMode::Difference`].
	pub fn prev(self) -> Self {
		let count = Self::ALL.len();
		let index = (self.as_u32() as usize + count - 1) % count;
		Self::ALL.get(index).copied().unwrap_or(self)
	}
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;

	#[test]
	fn default_is_normal() {
		assert_eq!(BlendMode::default(), BlendMode::Normal);
	}

	#[test]
	fn midi_mapping_matches_spec_boundaries() {
		let cases = [
			(0, BlendMode::Normal),
			(25, BlendMode::Normal),
			(26, BlendMode::Add),
			(51, BlendMode::Add),
			(52, BlendMode::Multiply),
			(76, BlendMode::Multiply),
			(77, BlendMode::Screen),
			(102, BlendMode::Screen),
			(103, BlendMode::Difference),
			(127, BlendMode::Difference),
			(255, BlendMode::Difference),
		];
		for (value, expected) in cases {
			assert_eq!(BlendMode::from_midi_cc(value), expected, "cc {value}");
		}
	}

	#[test]
	fn all_table_is_in_discriminator_order() {
		for (index, mode) in BlendMode::ALL.iter().enumerate() {
			assert_eq!(mode.as_u32() as usize, index);
		}
	}

	#[test]
	fn next_and_prev_cycle_through_every_mode() {
		let mut mode = BlendMode::Normal;
		for _ in 0..BlendMode::ALL.len() {
			mode = mode.next();
		}
		assert_eq!(mode, BlendMode::Normal);

		for _ in 0..BlendMode::ALL.len() {
			mode = mode.prev();
		}
		assert_eq!(mode, BlendMode::Normal);

		assert_eq!(BlendMode::Normal.prev(), BlendMode::Difference);
		assert_eq!(BlendMode::Difference.next(), BlendMode::Normal);
	}

	#[test]
	fn labels_are_unique() {
		let mut labels: Vec<&str> = BlendMode::ALL.iter().map(|mode| mode.label()).collect();
		labels.sort_unstable();
		labels.dedup();
		assert_eq!(labels.len(), BlendMode::ALL.len());
	}
}
