//! PNG / JPEG still-image loading via the `image` crate.

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use crate::error::MediaError;
use crate::model::{DecodedFrame, PixelPayload, ThumbnailImage};

use super::bc_decode::{DEFAULT_THUMBNAIL_MAX_HEIGHT, DEFAULT_THUMBNAIL_MAX_WIDTH, create_thumbnail};
use super::{MAX_DIMENSION, dimensions_are_valid};

/// Largest still-image file accepted, in bytes.
pub const MAX_IMAGE_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// A decoded still image ready to become a clip.
#[derive(Debug, Clone)]
pub struct LoadedImage {
	/// Image width in pixels.
	pub width: u32,
	/// Image height in pixels.
	pub height: u32,
	/// Whether any pixel is not fully opaque.
	pub has_alpha: bool,
	/// The image as a single resident RGBA8 frame.
	pub frame: Arc<DecodedFrame>,
	/// Downscaled preview for the clip grid.
	pub thumbnail: ThumbnailImage,
}

/// Reads and decodes the PNG or JPEG file at `path`.
pub fn load_image(path: &Path) -> Result<LoadedImage, MediaError> {
	let io = |source| MediaError::Io {
		path: path.display().to_string(),
		source,
	};
	let length = std::fs::metadata(path).map_err(io)?.len();
	if length > MAX_IMAGE_FILE_BYTES {
		return Err(MediaError::Corrupt {
			path: path.display().to_string(),
			detail: format!("image file of {length} bytes exceeds the limit"),
		});
	}
	let bytes = std::fs::read(path).map_err(io)?;
	load_image_from_memory(&bytes, path)
}

/// Decodes PNG or JPEG `bytes`; `path` is only used for error messages.
pub fn load_image_from_memory(bytes: &[u8], path: &Path) -> Result<LoadedImage, MediaError> {
	let corrupt = |detail: String| MediaError::Corrupt {
		path: path.display().to_string(),
		detail,
	};
	let reader = image::ImageReader::new(Cursor::new(bytes))
		.with_guessed_format()
		.map_err(|error| corrupt(format!("cannot detect image format: {error}")))?;
	match reader.format() {
		Some(image::ImageFormat::Png | image::ImageFormat::Jpeg) => {}
		Some(other) => {
			return Err(MediaError::UnsupportedFormat(format!(
				"image format {other:?} in `{}`",
				path.display()
			)));
		}
		None => return Err(corrupt("unrecognized image data".to_string())),
	}

	// Check the header dimensions before decoding any pixels.
	let (width, height) = reader
		.into_dimensions()
		.map_err(|error| corrupt(format!("cannot read image header: {error}")))?;
	if !dimensions_are_valid(width, height) {
		return Err(corrupt(format!(
			"image size {width}x{height} is outside 1..={MAX_DIMENSION}"
		)));
	}

	let mut reader = image::ImageReader::new(Cursor::new(bytes))
		.with_guessed_format()
		.map_err(|error| corrupt(format!("cannot detect image format: {error}")))?;
	let mut limits = image::Limits::default();
	limits.max_image_width = Some(MAX_DIMENSION);
	limits.max_image_height = Some(MAX_DIMENSION);
	reader.limits(limits);
	let decoded = reader
		.decode()
		.map_err(|error| corrupt(format!("cannot decode image: {error}")))?;
	let rgba = decoded.to_rgba8();
	let (width, height) = rgba.dimensions();
	if !dimensions_are_valid(width, height) {
		return Err(corrupt(format!("decoded image size {width}x{height} is invalid")));
	}
	let pixels = rgba.into_raw();
	let has_alpha = pixels.as_chunks::<4>().0.iter().any(|&[_, _, _, alpha]| alpha < 255);
	let thumbnail = create_thumbnail(
		&pixels,
		width,
		height,
		DEFAULT_THUMBNAIL_MAX_WIDTH,
		DEFAULT_THUMBNAIL_MAX_HEIGHT,
	);
	Ok(LoadedImage {
		width,
		height,
		has_alpha,
		frame: Arc::new(DecodedFrame {
			frame_index: 0,
			width,
			height,
			payload: PixelPayload::Rgba8(pixels),
		}),
		thumbnail,
	})
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;
	use crate::media::test_fixtures::TempDir;

	#[test]
	fn loads_png_with_and_without_alpha() {
		let dir = TempDir::new("img_png").unwrap();
		let opaque = dir.path().join("opaque.png");
		image::RgbaImage::from_pixel(400, 200, image::Rgba([10, 20, 30, 255]))
			.save(&opaque)
			.unwrap();
		let loaded = load_image(&opaque).unwrap();
		assert_eq!((loaded.width, loaded.height), (400, 200));
		assert!(!loaded.has_alpha);
		assert_eq!((loaded.thumbnail.width, loaded.thumbnail.height), (192, 96));
		assert!(loaded.thumbnail.is_valid());
		assert_eq!(loaded.frame.payload.byte_len(), 400 * 200 * 4);

		let mut translucent = image::RgbaImage::from_pixel(3, 3, image::Rgba([1, 1, 1, 255]));
		translucent.put_pixel(2, 2, image::Rgba([1, 1, 1, 10]));
		let path = dir.path().join("alpha.png");
		translucent.save(&path).unwrap();
		assert!(load_image(&path).unwrap().has_alpha);
	}

	#[test]
	fn loads_jpeg() {
		let dir = TempDir::new("img_jpg").unwrap();
		let path = dir.path().join("photo.jpg");
		image::RgbImage::from_pixel(64, 48, image::Rgb([200, 100, 50]))
			.save(&path)
			.unwrap();
		let loaded = load_image(&path).unwrap();
		assert_eq!((loaded.width, loaded.height), (64, 48));
		assert!(!loaded.has_alpha);
		assert!(matches!(loaded.frame.payload, PixelPayload::Rgba8(_)));
	}

	#[test]
	fn rejects_corrupt_and_oversized_images() {
		let path = Path::new("mem.png");
		assert!(load_image_from_memory(b"definitely not an image", path).is_err());
		assert!(load_image_from_memory(&[], path).is_err());

		let mut png = Vec::new();
		image::RgbaImage::from_pixel(4, 4, image::Rgba([0, 0, 0, 255]))
			.write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
			.unwrap();
		for cut in [8, 20, png.len() / 2, png.len() - 1] {
			let _ = load_image_from_memory(&png[..cut], path);
		}
		for position in 0..png.len() {
			let mut mutated = png.clone();
			mutated[position] ^= 0x5A;
			let _ = load_image_from_memory(&mutated, path);
		}

		// Patch the IHDR width to 9000 (> 8192): rejected before decoding.
		let mut wide = png.clone();
		wide[16..20].copy_from_slice(&9000u32.to_be_bytes());
		assert!(matches!(
			load_image_from_memory(&wide, path),
			Err(MediaError::Corrupt { .. })
		));
		assert!(matches!(
			load_image(Path::new("/nonexistent/x.png")),
			Err(MediaError::Io { .. })
		));
	}
}
