//! Layer identity, playback state, and playhead advancement.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::blend::BlendMode;
use super::clip::ClipId;

/// Number of performance layers in MUR.
pub const LAYER_COUNT: usize = 4;
/// Slowest supported playback speed (reverse, double speed).
pub const MIN_SPEED: f32 = -2.0;
/// Fastest supported playback speed (quadruple speed).
pub const MAX_SPEED: f32 = 4.0;
/// Frame rate substituted when a clip reports an unusable fps value.
pub const DEFAULT_CLIP_FPS: f32 = 60.0;
/// Lower opacity bound.
pub const MIN_OPACITY: f32 = 0.0;
/// Upper opacity bound.
pub const MAX_OPACITY: f32 = 1.0;

/// Zero-based index of one of the four performance layers.
///
/// The inner value is private so instances always address an existing layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct LayerIndex(u8);

impl LayerIndex {
	/// All four layers in ascending order.
	pub const ALL: [Self; LAYER_COUNT] = [Self(0), Self(1), Self(2), Self(3)];

	/// Creates an index from a zero-based position, or `None` when out of range.
	pub fn new(index: usize) -> Option<Self> {
		if index < LAYER_COUNT {
			u8::try_from(index).ok().map(Self)
		} else {
			None
		}
	}

	/// Creates an index from the one-based layer number (`1..=4`).
	pub fn from_number(number: u8) -> Option<Self> {
		number
			.checked_sub(1)
			.and_then(|zero_based| Self::new(usize::from(zero_based)))
	}

	/// Zero-based position of the layer (`0..=3`).
	pub fn index(self) -> usize {
		usize::from(self.0)
	}

	/// One-based layer number shown to the operator (`1..=4`).
	pub fn number(self) -> u8 {
		self.0.saturating_add(1)
	}

	/// Offsets the index by `delta`, wrapping around the layer ring.
	pub fn offset(self, delta: i32) -> Self {
		let count = i32::try_from(LAYER_COUNT).unwrap_or(i32::MAX);
		let shifted = (i32::from(self.0) + delta).rem_euclid(count);
		Self(u8::try_from(shifted).unwrap_or(0))
	}
}

impl fmt::Display for LayerIndex {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(formatter, "L{}", self.number())
	}
}

/// How a layer behaves when the playhead reaches the end of a clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum PlaybackMode {
	/// Wrap around and keep playing forever.
	#[default]
	Loop,
	/// Stop on the first frame after the clip finished.
	OneShot,
}

impl PlaybackMode {
	/// Human readable name for the control UI.
	pub fn label(self) -> &'static str {
		match self {
			Self::Loop => "Loop",
			Self::OneShot => "One Shot",
		}
	}

	/// Returns the other mode.
	pub fn toggled(self) -> Self {
		match self {
			Self::Loop => Self::OneShot,
			Self::OneShot => Self::Loop,
		}
	}
}

/// Transport state of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum PlaybackState {
	/// No playhead movement; the last visible frame stays on screen.
	#[default]
	Stopped,
	/// The playhead advances every frame.
	Playing,
	/// The playhead is frozen but the layer still composites.
	Paused,
}

impl PlaybackState {
	/// Returns `true` while the playhead advances.
	pub fn is_playing(self) -> bool {
		matches!(self, Self::Playing)
	}
}

/// Complete playback and compositing state of one layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayerState {
	/// Clip highlighted in the UI and targeted by the trigger button.
	pub selected_clip: Option<ClipId>,
	/// Clip currently bound to the layer for playback and compositing.
	pub active_clip: Option<ClipId>,
	/// Transport state of the layer.
	pub playback_state: PlaybackState,
	/// Loop or one-shot behaviour.
	pub playback_mode: PlaybackMode,
	/// Playback speed in `-2.0..=4.0`, `1.0` is real time.
	pub speed: f32,
	/// Layer opacity in `0.0..=1.0`.
	pub opacity: f32,
	/// Compositing mode used when blending this layer.
	pub blend_mode: BlendMode,
	/// Current playhead position in frames; the integer part selects the frame.
	pub playhead_frame: f64,
	/// Number of frames in the active clip.
	pub frame_count: u32,
	/// Frame rate of the active clip.
	pub clip_fps: f32,
	/// Last error raised while triggering or decoding this layer.
	pub last_error: Option<String>,
}

impl Default for LayerState {
	fn default() -> Self {
		Self {
			selected_clip: None,
			active_clip: None,
			playback_state: PlaybackState::Stopped,
			playback_mode: PlaybackMode::Loop,
			speed: 1.0,
			opacity: 1.0,
			blend_mode: BlendMode::Normal,
			playhead_frame: 0.0,
			frame_count: 0,
			clip_fps: DEFAULT_CLIP_FPS,
			last_error: None,
		}
	}
}

impl LayerState {
	/// Returns `true` when the layer has an active clip.
	pub fn has_clip(&self) -> bool {
		self.active_clip.is_some()
	}

	/// Stores `speed` clamped into `-2.0..=4.0`, ignoring non-finite input.
	pub fn set_speed(&mut self, speed: f32) {
		if speed.is_finite() {
			self.speed = speed.clamp(MIN_SPEED, MAX_SPEED);
		}
	}

	/// Stores `opacity` clamped into `0.0..=1.0`, ignoring non-finite input.
	pub fn set_opacity(&mut self, opacity: f32) {
		if opacity.is_finite() {
			self.opacity = opacity.clamp(MIN_OPACITY, MAX_OPACITY);
		}
	}

	/// Maps playback-speed CC values onto `-2.0..=4.0`.
	///
	/// `0` maps to `-2.0`, the centre detent `64` to `+1.0`, and `127` to `+4.0`.
	pub fn speed_from_midi_cc(value: u8) -> f32 {
		let value = value.min(127);
		let raw = f32::from(value);
		if value <= 64 {
			MIN_SPEED + (raw / 64.0) * 3.0
		} else {
			1.0 + ((raw - 64.0) / 63.0) * 3.0
		}
	}

	/// Advances the playhead by `dt_secs` seconds of wall-clock time.
	///
	/// Static clips (`frame_count <= 1`) hold frame `0` so triggered images stay
	/// visible without being auto-stopped. Looping clips wrap with `rem_euclid`,
	/// which makes reverse playback and arbitrary speeds behave correctly.
	/// One-shot clips stop on frame `0` after their direction's end is reached.
	pub fn advance_playhead(&mut self, dt_secs: f32) {
		if !self.playback_state.is_playing()
			|| !dt_secs.is_finite()
			|| dt_secs <= 0.0
			|| !self.speed.is_finite()
			|| !self.clip_fps.is_finite()
			|| self.clip_fps <= 0.0
		{
			return;
		}

		if self.frame_count == 0 {
			return;
		}
		if self.frame_count == 1 {
			self.playhead_frame = 0.0;
			return;
		}

		let frames = f64::from(self.frame_count);
		let delta = f64::from(self.speed) * f64::from(self.clip_fps) * f64::from(dt_secs);
		let next = self.playhead_frame + delta;

		match self.playback_mode {
			PlaybackMode::Loop => {
				self.playhead_frame = next.rem_euclid(frames);
			}
			PlaybackMode::OneShot => {
				let finished = if self.speed < 0.0 { next < 0.0 } else { next >= frames };
				if finished {
					self.playhead_frame = 0.0;
					self.playback_state = PlaybackState::Stopped;
				} else {
					self.playhead_frame = next.max(0.0);
				}
			}
		}
	}

	/// Integer frame index of the current playhead, clamped to the clip length.
	pub fn current_frame_index(&self) -> u32 {
		let last_frame = self.frame_count.saturating_sub(1);
		let frame = self.playhead_frame.floor();
		if !frame.is_finite() || frame <= 0.0 {
			return 0;
		}
		(frame as u32).min(last_frame)
	}

	/// Binds `clip_id` to the layer and starts playback from its first frame.
	pub fn trigger(&mut self, clip_id: ClipId, frame_count: u32, fps: f32) {
		self.selected_clip = Some(clip_id);
		self.active_clip = Some(clip_id);
		self.frame_count = frame_count;
		self.clip_fps = if fps.is_finite() && fps > 0.0 {
			fps
		} else {
			DEFAULT_CLIP_FPS
		};
		self.last_error = None;
		self.playhead_frame = self.start_frame();
		self.playback_state = PlaybackState::Playing;
	}

	/// Restarts the active clip from its first frame, or from its last frame when
	/// playing in reverse. No-op when the layer has no active clip.
	pub fn retrigger(&mut self) {
		if self.active_clip.is_none() {
			return;
		}
		self.playhead_frame = self.start_frame();
		self.last_error = None;
		self.playback_state = PlaybackState::Playing;
	}

	/// Frame the playhead should show when (re)starting, honouring reverse speed.
	fn start_frame(&self) -> f64 {
		if self.speed < 0.0 {
			f64::from(self.frame_count.saturating_sub(1))
		} else {
			0.0
		}
	}
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;

	fn playing_layer(frame_count: u32, fps: f32) -> LayerState {
		let mut layer = LayerState::default();
		layer.trigger(ClipId(1), frame_count, fps);
		layer
	}

	#[test]
	fn index_round_trips_through_numbers() {
		for (index, layer) in LayerIndex::ALL.iter().enumerate() {
			assert_eq!(layer.index(), index);
			assert_eq!(layer.number(), index as u8 + 1);
			assert_eq!(LayerIndex::new(index), Some(*layer));
			assert_eq!(LayerIndex::from_number(index as u8 + 1), Some(*layer));
		}
		assert_eq!(LayerIndex::new(4), None);
		assert_eq!(LayerIndex::new(usize::MAX), None);
		assert_eq!(LayerIndex::from_number(0), None);
		assert_eq!(LayerIndex::from_number(5), None);
		assert_eq!(LayerIndex::default(), LayerIndex::ALL[0]);
	}

	#[test]
	fn display_uses_one_based_label() {
		let labels: Vec<String> = LayerIndex::ALL.iter().map(ToString::to_string).collect();
		assert_eq!(labels, ["L1", "L2", "L3", "L4"]);
	}

	#[test]
	fn offset_wraps_in_both_directions() {
		let first = LayerIndex::ALL[0];
		let last = LayerIndex::ALL[3];
		assert_eq!(last.offset(1), first);
		assert_eq!(first.offset(-1), last);
		assert_eq!(first.offset(-9), LayerIndex::ALL[3]);
		assert_eq!(first.offset(0), first);
	}

	#[test]
	fn speed_mapping_matches_spec_anchors() {
		assert_eq!(LayerState::speed_from_midi_cc(0), MIN_SPEED);
		assert_eq!(LayerState::speed_from_midi_cc(64), 1.0);
		assert_eq!(LayerState::speed_from_midi_cc(127), 4.0);
		assert_eq!(LayerState::speed_from_midi_cc(255), 4.0);
		assert!(LayerState::speed_from_midi_cc(32) < 0.0);
		assert!(LayerState::speed_from_midi_cc(96) > 1.0);
	}

	#[test]
	fn set_speed_and_opacity_clamp_and_ignore_nan() {
		let mut layer = LayerState::default();
		layer.set_speed(99.0);
		assert_eq!(layer.speed, MAX_SPEED);
		layer.set_speed(-99.0);
		assert_eq!(layer.speed, MIN_SPEED);
		layer.set_speed(f32::NAN);
		assert_eq!(layer.speed, MIN_SPEED);

		layer.set_opacity(2.0);
		assert_eq!(layer.opacity, MAX_OPACITY);
		layer.set_opacity(-1.0);
		assert_eq!(layer.opacity, MIN_OPACITY);
		layer.set_opacity(f32::NAN);
		assert_eq!(layer.opacity, MIN_OPACITY);
	}

	#[test]
	fn loop_wraps_forward_and_reverse() {
		let mut layer = playing_layer(10, 10.0);
		layer.playback_mode = PlaybackMode::Loop;
		layer.playhead_frame = 9.0;
		layer.advance_playhead(0.1);
		assert!(layer.playhead_frame.abs() < 1e-5);
		assert_eq!(layer.playback_state, PlaybackState::Playing);

		layer.playhead_frame = 0.0;
		layer.speed = -1.0;
		layer.advance_playhead(0.1);
		assert!((layer.playhead_frame - 9.0).abs() < 1e-5);
		assert_eq!(layer.playback_state, PlaybackState::Playing);

		layer.speed = -1.0;
		layer.advance_playhead(5.0);
		assert!(layer.playhead_frame >= 0.0 && layer.playhead_frame < 10.0);
	}

	#[test]
	fn one_shot_forward_stops_after_last_frame() {
		let mut layer = playing_layer(10, 10.0);
		layer.playback_mode = PlaybackMode::OneShot;
		layer.playhead_frame = 9.0;
		layer.advance_playhead(0.1);
		assert_eq!(layer.playhead_frame, 0.0);
		assert_eq!(layer.playback_state, PlaybackState::Stopped);
	}

	#[test]
	fn one_shot_reverse_stops_at_start() {
		let mut layer = playing_layer(10, 10.0);
		layer.playback_mode = PlaybackMode::OneShot;
		layer.speed = -1.0;
		layer.playhead_frame = 0.0;
		layer.advance_playhead(0.1);
		assert_eq!(layer.playhead_frame, 0.0);
		assert_eq!(layer.playback_state, PlaybackState::Stopped);
	}

	#[test]
	fn static_clips_hold_frame_zero_without_stopping() {
		let mut layer = playing_layer(1, 60.0);
		layer.playback_mode = PlaybackMode::OneShot;
		layer.advance_playhead(10.0);
		assert_eq!(layer.playhead_frame, 0.0);
		assert_eq!(layer.playback_state, PlaybackState::Playing);
		assert_eq!(layer.current_frame_index(), 0);
	}

	#[test]
	fn advance_guards_bad_input() {
		let mut layer = playing_layer(10, 60.0);
		layer.advance_playhead(0.0);
		assert_eq!(layer.playhead_frame, 0.0);
		layer.advance_playhead(-1.0);
		assert_eq!(layer.playhead_frame, 0.0);
		layer.advance_playhead(f32::NAN);
		assert_eq!(layer.playhead_frame, 0.0);

		layer.playback_state = PlaybackState::Stopped;
		layer.advance_playhead(1.0);
		assert_eq!(layer.playhead_frame, 0.0);

		layer.playback_state = PlaybackState::Playing;
		layer.speed = f32::NAN;
		layer.advance_playhead(1.0);
		assert_eq!(layer.playhead_frame, 0.0);

		layer.speed = 1.0;
		layer.clip_fps = 0.0;
		layer.advance_playhead(1.0);
		assert_eq!(layer.playhead_frame, 0.0);
	}

	#[test]
	fn current_frame_index_clamps_to_clip_range() {
		let mut layer = playing_layer(10, 60.0);
		layer.playhead_frame = 5.7;
		assert_eq!(layer.current_frame_index(), 5);

		layer.playhead_frame = 999.0;
		assert_eq!(layer.current_frame_index(), 9);

		layer.playhead_frame = -4.0;
		assert_eq!(layer.current_frame_index(), 0);

		layer.playhead_frame = f64::NAN;
		assert_eq!(layer.current_frame_index(), 0);

		layer.frame_count = 0;
		assert_eq!(layer.current_frame_index(), 0);
	}

	#[test]
	fn trigger_sets_playback_and_retrigger_handles_direction() {
		let mut layer = LayerState {
			last_error: Some("old".to_string()),
			playback_mode: PlaybackMode::OneShot,
			..LayerState::default()
		};
		layer.trigger(ClipId(3), 24, 24.0);
		assert_eq!(layer.selected_clip, Some(ClipId(3)));
		assert_eq!(layer.active_clip, Some(ClipId(3)));
		assert_eq!(layer.frame_count, 24);
		assert_eq!(layer.clip_fps, 24.0);
		assert_eq!(layer.playhead_frame, 0.0);
		assert!(layer.last_error.is_none());
		assert!(layer.playback_state.is_playing());

		layer.playhead_frame = 12.0;
		layer.retrigger();
		assert_eq!(layer.playhead_frame, 0.0);
		assert!(layer.playback_state.is_playing());

		layer.speed = -1.0;
		layer.playhead_frame = 12.0;
		layer.retrigger();
		assert_eq!(layer.playhead_frame, 23.0);
	}

	#[test]
	fn trigger_replaces_unusable_fps_with_default() {
		let mut layer = LayerState::default();
		layer.trigger(ClipId(1), 10, f32::NAN);
		assert_eq!(layer.clip_fps, DEFAULT_CLIP_FPS);
		layer.trigger(ClipId(1), 10, 0.0);
		assert_eq!(layer.clip_fps, DEFAULT_CLIP_FPS);
	}

	#[test]
	fn retrigger_without_active_clip_is_a_noop() {
		let mut layer = LayerState {
			playhead_frame: 3.0,
			..LayerState::default()
		};
		layer.retrigger();
		assert_eq!(layer.playhead_frame, 3.0);
		assert_eq!(layer.playback_state, PlaybackState::Stopped);
	}
}
