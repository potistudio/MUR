//! Media loading, decoding, and per-layer streaming.
//!
//! The media subsystem is entirely pure Rust:
//!
//! - [`mov`] parses QuickTime `.mov` box structure without reading `mdat` payloads.
//! - [`hap`] unwraps HAP frames (raw, Snappy, and chunked Snappy) into BC blocks.
//! - [`bc_decode`] converts BC1/BC3/BC3-YCoCg blocks to RGBA8 for thumbnails and
//!   for GPUs without BC texture support.
//! - [`image_loader`] decodes PNG/JPEG still images.
//! - [`streamer`] runs four isolated, panic-supervised decode workers, one per layer.
//! - [`test_fixtures`] builds synthetic HAP `.mov` files for unit and integration tests.
//!
//! Every entry point validates its input and returns [`MediaError`] instead of
//! panicking, so a corrupt file only ever affects its own clip slot or layer.

pub mod bc_decode;
pub mod hap;
pub mod image_loader;
pub mod mov;
pub mod streamer;
pub mod test_fixtures;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use bc_decode::{
	DEFAULT_THUMBNAIL_MAX_HEIGHT, DEFAULT_THUMBNAIL_MAX_WIDTH, create_thumbnail, decode_bc1_opaque_to_rgba8,
	decode_bc1_to_rgba8, decode_bc3_to_rgba8, decode_bc3_ycocg_to_rgba8, decode_frame_to_rgba8,
};
pub use hap::{decode_hap_frame, decode_hap_payload, expected_hap_bytes};
pub use image_loader::{LoadedImage, load_image};
pub use mov::{MovIndex, MovSample, parse_mov_from_file, read_mov_sample, read_mov_sample_into};
pub use streamer::{LayerStreamEvent, LayerStreamer, PREFETCH_AHEAD, RING_CAPACITY, prefetch_window};

use crate::error::MediaError;
use crate::model::{ClipId, ClipInfo, ClipKind, ClipStatus, DecodedFrame};

/// Largest accepted frame width or height in pixels.
pub const MAX_DIMENSION: u32 = 8192;

/// File extensions (lowercase, without dot) that MUR can load.
pub const SUPPORTED_EXTENSIONS: [&str; 4] = ["mov", "png", "jpg", "jpeg"];

/// Returns `true` when `width` and `height` are both within `1..=MAX_DIMENSION`.
pub fn dimensions_are_valid(width: u32, height: u32) -> bool {
	(1..=MAX_DIMENSION).contains(&width) && (1..=MAX_DIMENSION).contains(&height)
}

/// Where the frames of a loaded clip come from.
#[derive(Debug, Clone)]
pub enum ClipBacking {
	/// A still image whose single frame is kept resident in memory.
	StaticImage,
	/// A HAP QuickTime movie streamed from disk on demand.
	HapMov {
		/// Path of the movie file.
		path: PathBuf,
		/// Parsed sample index shared by every layer playing the clip.
		index: Arc<MovIndex>,
	},
}

/// A validated clip ready to be registered with a [`LayerStreamer`].
#[derive(Debug, Clone)]
pub struct ClipAsset {
	/// Metadata shown by the UI.
	pub info: Arc<ClipInfo>,
	/// Resident first frame used for zero-latency triggers and thumbnails.
	pub first_frame: Arc<DecodedFrame>,
	/// Frame source of the clip.
	pub backing: ClipBacking,
}

impl ClipAsset {
	/// Identifier of the clip.
	pub fn id(&self) -> ClipId {
		self.info.id
	}

	/// Number of frames in the clip (`1` for still images).
	pub fn frame_count(&self) -> u32 {
		self.info.frame_count
	}
}

/// Returns `true` when `path` has a supported media extension (case-insensitive).
pub fn is_supported_media_path(path: &Path) -> bool {
	path.extension()
		.and_then(|ext| ext.to_str())
		.map(|ext| {
			let lower = ext.to_ascii_lowercase();
			SUPPORTED_EXTENSIONS.contains(&lower.as_str())
		})
		.unwrap_or(false)
}

/// Lists supported media files directly inside `dir`, sorted by path.
///
/// Unreadable directories or entries are skipped and yield an empty or partial
/// list instead of an error, so a bad folder never blocks startup.
pub fn scan_media_dir(dir: &Path) -> Vec<PathBuf> {
	let Ok(entries) = std::fs::read_dir(dir) else {
		tracing::warn!(dir = %dir.display(), "media directory could not be read");
		return Vec::new();
	};
	let mut paths: Vec<PathBuf> = entries
		.filter_map(Result::ok)
		.map(|entry| entry.path())
		.filter(|path| path.is_file() && is_supported_media_path(path))
		.collect();
	paths.sort();
	paths
}

/// Loads, validates, and decodes frame 0 of the clip at `path`.
///
/// The returned [`ClipInfo`] carries a thumbnail, and the [`ClipAsset`] keeps the
/// decoded first frame resident. Corrupt or unsupported files return an error;
/// any unexpected panic inside a third-party decoder is caught and reported as
/// [`MediaError::Corrupt`].
pub fn load_clip(id: ClipId, path: &Path) -> Result<(ClipInfo, ClipAsset), MediaError> {
	match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load_clip_inner(id, path))) {
		Ok(result) => result,
		Err(_) => Err(MediaError::Corrupt {
			path: path.display().to_string(),
			detail: "decoder panicked while loading the clip".to_string(),
		}),
	}
}

fn load_clip_inner(id: ClipId, path: &Path) -> Result<(ClipInfo, ClipAsset), MediaError> {
	let extension = path
		.extension()
		.and_then(|ext| ext.to_str())
		.map(str::to_ascii_lowercase)
		.unwrap_or_default();
	let name = path
		.file_name()
		.map(|name| name.to_string_lossy().into_owned())
		.unwrap_or_else(|| path.display().to_string());

	match extension.as_str() {
		"mov" => load_hap_clip(id, path, name),
		"png" | "jpg" | "jpeg" => {
			let loaded = load_image(path)?;
			let info = ClipInfo {
				id,
				path: path.to_path_buf(),
				name,
				kind: ClipKind::ImageRgba,
				width: loaded.width,
				height: loaded.height,
				fps: crate::model::layer::DEFAULT_CLIP_FPS,
				frame_count: 1,
				has_alpha: loaded.has_alpha,
				status: ClipStatus::Ready,
				thumbnail: Some(loaded.thumbnail),
			};
			let asset = ClipAsset {
				info: Arc::new(info.clone()),
				first_frame: loaded.frame,
				backing: ClipBacking::StaticImage,
			};
			Ok((info, asset))
		}
		other => Err(MediaError::UnsupportedFormat(format!(
			"unsupported file extension `{other}` for `{}`",
			path.display()
		))),
	}
}

fn load_hap_clip(id: ClipId, path: &Path, name: String) -> Result<(ClipInfo, ClipAsset), MediaError> {
	let index = parse_mov_from_file(path)?;
	let first_sample = index.samples.first().ok_or_else(|| MediaError::Corrupt {
		path: path.display().to_string(),
		detail: "video track contains no samples".to_string(),
	})?;
	let mut file = File::open(path).map_err(|source| MediaError::Io {
		path: path.display().to_string(),
		source,
	})?;
	let bytes = read_mov_sample(&mut file, first_sample)?;
	let first_frame = decode_hap_frame(&bytes, index.kind, index.width, index.height, 0)?;
	let rgba = decode_frame_to_rgba8(&first_frame)?;
	let thumbnail = create_thumbnail(
		&rgba,
		index.width,
		index.height,
		DEFAULT_THUMBNAIL_MAX_WIDTH,
		DEFAULT_THUMBNAIL_MAX_HEIGHT,
	);
	let frame_count = u32::try_from(index.samples.len()).map_err(|_| MediaError::Corrupt {
		path: path.display().to_string(),
		detail: "too many samples".to_string(),
	})?;
	let info = ClipInfo {
		id,
		path: path.to_path_buf(),
		name,
		kind: index.kind,
		width: index.width,
		height: index.height,
		fps: index.fps,
		frame_count,
		has_alpha: matches!(index.kind, ClipKind::HapAlpha),
		status: ClipStatus::Ready,
		thumbnail: Some(thumbnail),
	};
	let asset = ClipAsset {
		info: Arc::new(info.clone()),
		first_frame: Arc::new(first_frame),
		backing: ClipBacking::HapMov {
			path: path.to_path_buf(),
			index: Arc::new(index),
		},
	};
	Ok((info, asset))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::test_fixtures::{FixtureCompression, HapMovFixture, TempDir};
	use super::*;
	use crate::model::PixelPayload;

	#[test]
	fn supported_paths_are_case_insensitive() {
		assert!(is_supported_media_path(Path::new("a/b/clip.MOV")));
		assert!(is_supported_media_path(Path::new("x.png")));
		assert!(is_supported_media_path(Path::new("x.JpEg")));
		assert!(is_supported_media_path(Path::new("x.jpg")));
		assert!(!is_supported_media_path(Path::new("x.mp4")));
		assert!(!is_supported_media_path(Path::new("noext")));
	}

	#[test]
	fn dimension_validation_bounds() {
		assert!(dimensions_are_valid(1, 1));
		assert!(dimensions_are_valid(8192, 8192));
		assert!(!dimensions_are_valid(0, 10));
		assert!(!dimensions_are_valid(10, 8193));
	}

	#[test]
	fn scan_media_dir_filters_and_sorts() {
		let dir = TempDir::new("scan").unwrap();
		for name in ["b.mov", "a.PNG", "c.jpeg", "notes.txt", "d.mp4"] {
			std::fs::write(dir.path().join(name), b"x").unwrap();
		}
		std::fs::create_dir(dir.path().join("sub.mov")).unwrap();
		let found: Vec<String> = scan_media_dir(dir.path())
			.iter()
			.map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
			.collect();
		assert_eq!(found, ["a.PNG", "b.mov", "c.jpeg"]);
		assert!(scan_media_dir(&dir.path().join("missing")).is_empty());
	}

	#[test]
	fn load_clip_decodes_hap_first_frame() {
		let dir = TempDir::new("load_hap").unwrap();
		let path = dir.path().join("clip.mov");
		let mut fixture = HapMovFixture::new(ClipKind::HapAlpha, 40, 20, 5);
		fixture.compression = FixtureCompression::Snappy;
		fixture.write_to(&path).unwrap();

		let (info, asset) = load_clip(ClipId(9), &path).unwrap();
		assert_eq!(info.kind, ClipKind::HapAlpha);
		assert_eq!(info.frame_count, 5);
		assert_eq!((info.width, info.height), (40, 20));
		assert!(info.has_alpha);
		assert!(info.is_ready());
		assert_eq!(info.name, "clip.mov");
		assert!(info.thumbnail.as_ref().unwrap().is_valid());
		assert_eq!(asset.id(), ClipId(9));
		assert!(matches!(asset.first_frame.payload, PixelPayload::Bc3(_)));
		assert!(matches!(asset.backing, ClipBacking::HapMov { .. }));
	}

	#[test]
	fn load_clip_rejects_bad_inputs_without_panicking() {
		let dir = TempDir::new("load_bad").unwrap();
		let garbage = dir.path().join("garbage.mov");
		std::fs::write(&garbage, [0xFFu8; 64]).unwrap();
		assert!(load_clip(ClipId(1), &garbage).is_err());

		let bad_png = dir.path().join("bad.png");
		std::fs::write(&bad_png, b"not a png").unwrap();
		assert!(load_clip(ClipId(2), &bad_png).is_err());

		let txt = dir.path().join("a.txt");
		std::fs::write(&txt, b"x").unwrap();
		assert!(matches!(
			load_clip(ClipId(3), &txt),
			Err(MediaError::UnsupportedFormat(_))
		));

		assert!(matches!(
			load_clip(ClipId(4), &dir.path().join("missing.mov")),
			Err(MediaError::Io { .. })
		));
	}

	#[test]
	fn load_clip_loads_still_images() {
		let dir = TempDir::new("load_png").unwrap();
		let path = dir.path().join("still.png");
		image::RgbaImage::from_pixel(8, 4, image::Rgba([1, 2, 3, 255]))
			.save(&path)
			.unwrap();
		let (info, asset) = load_clip(ClipId(5), &path).unwrap();
		assert_eq!(info.kind, ClipKind::ImageRgba);
		assert_eq!(info.frame_count, 1);
		assert!(!info.has_alpha);
		assert!(matches!(asset.backing, ClipBacking::StaticImage));
		assert_eq!(asset.frame_count(), 1);
	}
}
