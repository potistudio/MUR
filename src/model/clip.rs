//! Clip metadata, decode payloads, and thumbnail containers.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Stable identifier of a loaded clip for the lifetime of one session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ClipId(pub u64);

/// Media kind of a clip, which determines how its frames are uploaded to the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClipKind {
	/// HAP with DXT1/BC1 blocks, fully opaque.
	Hap,
	/// HAP Alpha with DXT5/BC3 blocks including an alpha channel.
	HapAlpha,
	/// HAP Q with DXT5-YCoCg blocks, converted to RGB in the shader.
	HapQ,
	/// Still image decoded to RGBA8.
	ImageRgba,
}

impl ClipKind {
	/// Short display name of the codec.
	pub fn label(self) -> &'static str {
		match self {
			Self::Hap => "HAP",
			Self::HapAlpha => "HAP Alpha",
			Self::HapQ => "HAP Q",
			Self::ImageRgba => "Image",
		}
	}

	/// Compact badge describing the GPU texture format.
	pub fn badge(self) -> &'static str {
		match self {
			Self::Hap => "DXT1",
			Self::HapAlpha => "DXT5",
			Self::HapQ => "YCoCg",
			Self::ImageRgba => "RGBA",
		}
	}

	/// Returns `true` for animated HAP clips and `false` for still images.
	pub fn is_video(self) -> bool {
		!matches!(self, Self::ImageRgba)
	}
}

/// Load/decode status of a clip as shown in the clip grid.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClipStatus {
	/// The clip is decoded and safe to trigger.
	Ready,
	/// Loading or decoding failed; the message is shown to the operator.
	Error(String),
}

/// Decoded RGBA8 preview used by the clip grid and by first-frame display.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThumbnailImage {
	/// Pixel width in pixels.
	pub width: u32,
	/// Pixel height in pixels.
	pub height: u32,
	/// Tightly packed RGBA8 pixels, `width * height * 4` bytes.
	pub rgba: Vec<u8>,
}

impl ThumbnailImage {
	/// Returns `true` when the pixel buffer length matches the declared geometry.
	pub fn is_valid(&self) -> bool {
		let expected = (self.width as usize)
			.saturating_mul(self.height as usize)
			.saturating_mul(4);
		self.rgba.len() == expected
	}
}

/// A decoded frame payload together with the format the GPU expects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelPayload {
	/// BC1/DXT1 compressed blocks.
	Bc1(Vec<u8>),
	/// BC3/DXT5 compressed blocks including alpha.
	Bc3(Vec<u8>),
	/// BC3 compressed YCoCg blocks, converted to RGB on the GPU.
	Bc3YCoCg(Vec<u8>),
	/// Uncompressed RGBA8 pixels.
	Rgba8(Vec<u8>),
}

impl PixelPayload {
	/// Number of bytes in the payload buffer.
	pub fn byte_len(&self) -> usize {
		match self {
			Self::Bc1(bytes) | Self::Bc3(bytes) | Self::Bc3YCoCg(bytes) | Self::Rgba8(bytes) => bytes.len(),
		}
	}
}

/// One decoded video frame ready for texture upload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodedFrame {
	/// Zero-based frame index within the clip.
	pub frame_index: u32,
	/// Frame width in pixels.
	pub width: u32,
	/// Frame height in pixels.
	pub height: u32,
	/// Encoded or raw pixel data.
	pub payload: PixelPayload,
}

/// Everything the UI needs to know about a loaded clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipInfo {
	/// Session-unique identifier.
	pub id: ClipId,
	/// Absolute path of the source file.
	pub path: PathBuf,
	/// Display name derived from the file name.
	pub name: String,
	/// Media kind of the clip.
	pub kind: ClipKind,
	/// Native pixel width.
	pub width: u32,
	/// Native pixel height.
	pub height: u32,
	/// Native frame rate in frames per second.
	pub fps: f32,
	/// Number of frames; `1` for still images.
	pub frame_count: u32,
	/// Whether the clip carries a usable alpha channel.
	pub has_alpha: bool,
	/// Load status of the clip.
	pub status: ClipStatus,
	/// First-frame preview shared by the UI and the zero-latency trigger path.
	pub thumbnail: Option<ThumbnailImage>,
}

impl ClipInfo {
	/// Returns `true` when the clip has neither an alpha channel nor an alpha codec.
	pub fn is_opaque_margin(&self) -> bool {
		!self.has_alpha && !matches!(self.kind, ClipKind::HapAlpha)
	}

	/// Returns `true` when the clip finished loading without errors.
	pub fn is_ready(&self) -> bool {
		matches!(self.status, ClipStatus::Ready)
	}
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;

	fn clip(kind: ClipKind, has_alpha: bool, status: ClipStatus) -> ClipInfo {
		ClipInfo {
			id: ClipId(7),
			path: PathBuf::from("/clips/test.mov"),
			name: "test.mov".to_string(),
			kind,
			width: 1920,
			height: 1080,
			fps: 60.0,
			frame_count: 120,
			has_alpha,
			status,
			thumbnail: None,
		}
	}

	#[test]
	fn payload_lengths_are_reported() {
		assert_eq!(PixelPayload::Bc1(vec![0; 8]).byte_len(), 8);
		assert_eq!(PixelPayload::Bc3(vec![0; 16]).byte_len(), 16);
		assert_eq!(PixelPayload::Bc3YCoCg(vec![0; 32]).byte_len(), 32);
		assert_eq!(PixelPayload::Rgba8(vec![0; 64]).byte_len(), 64);
	}

	#[test]
	fn thumbnail_validity_checks_buffer_length() {
		let valid = ThumbnailImage {
			width: 2,
			height: 2,
			rgba: vec![0; 16],
		};
		let invalid = ThumbnailImage {
			width: 2,
			height: 2,
			rgba: vec![0; 15],
		};
		assert!(valid.is_valid());
		assert!(!invalid.is_valid());
	}

	#[test]
	fn opaque_margin_follows_alpha_and_kind() {
		assert!(clip(ClipKind::Hap, false, ClipStatus::Ready).is_opaque_margin());
		assert!(!clip(ClipKind::HapAlpha, true, ClipStatus::Ready).is_opaque_margin());
		// A HAP Alpha clip that reports no alpha still reserves a transparent margin.
		assert!(!clip(ClipKind::HapAlpha, false, ClipStatus::Ready).is_opaque_margin());
		assert!(!clip(ClipKind::ImageRgba, true, ClipStatus::Ready).is_opaque_margin());
	}

	#[test]
	fn readiness_tracks_status() {
		assert!(clip(ClipKind::Hap, false, ClipStatus::Ready).is_ready());
		assert!(!clip(ClipKind::Hap, false, ClipStatus::Error("boom".into())).is_ready());
	}

	#[test]
	fn kind_labels_and_badges_are_stable() {
		assert_eq!(ClipKind::Hap.badge(), "DXT1");
		assert_eq!(ClipKind::HapAlpha.badge(), "DXT5");
		assert_eq!(ClipKind::HapQ.badge(), "YCoCg");
		assert_eq!(ClipKind::ImageRgba.badge(), "RGBA");
		assert!(!ClipKind::ImageRgba.is_video());
		assert!(ClipKind::Hap.is_video());
		assert_eq!(ClipKind::HapQ.label(), "HAP Q");
	}
}
