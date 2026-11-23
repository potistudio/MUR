//! CPU decoders for BC1, BC3, and BC3-YCoCg blocks plus thumbnail downsampling.
//!
//! These are used to build UI thumbnails from the first HAP frame and as an
//! RGBA8 fallback for GPUs that lack BC texture compression support.

use crate::error::MediaError;
use crate::model::{DecodedFrame, PixelPayload, ThumbnailImage};

use super::dimensions_are_valid;

/// Default maximum thumbnail width in pixels.
pub const DEFAULT_THUMBNAIL_MAX_WIDTH: u32 = 192;
/// Default maximum thumbnail height in pixels.
pub const DEFAULT_THUMBNAIL_MAX_HEIGHT: u32 = 108;

/// A decoded 4×4 block: 4 rows of 4 RGBA pixels.
type Block = [[u8; 16]; 4];

fn invalid(detail: impl Into<String>) -> MediaError {
	MediaError::Decode {
		frame_index: 0,
		detail: detail.into(),
	}
}

/// Number of RGBA8 bytes for an image, with overflow checking.
fn rgba_len(width: u32, height: u32) -> Result<usize, MediaError> {
	usize::try_from(u64::from(width).saturating_mul(u64::from(height)).saturating_mul(4))
		.map_err(|_| invalid("image too large"))
}

/// Expands an RGB565 color to RGB888.
fn expand_565(color: u16) -> [u8; 3] {
	let r = ((color >> 11) & 0x1F) as u8;
	let g = ((color >> 5) & 0x3F) as u8;
	let b = (color & 0x1F) as u8;
	[(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)]
}

/// Decodes the 8-byte BC1 color part of a block into a 4-entry RGBA palette.
///
/// `force_four_color` is set for BC3 color blocks, which never use the
/// three-color + transparent mode.
fn color_palette(block: &[u8; 8], force_four_color: bool) -> ([[u8; 4]; 4], u32) {
	let [c0l, c0h, c1l, c1h, i0, i1, i2, i3] = *block;
	let c0 = u16::from_le_bytes([c0l, c0h]);
	let c1 = u16::from_le_bytes([c1l, c1h]);
	let indices = u32::from_le_bytes([i0, i1, i2, i3]);
	let [r0, g0, b0] = expand_565(c0).map(u16::from);
	let [r1, g1, b1] = expand_565(c1).map(u16::from);
	let mix = |a: u16, b: u16, wa: u16, wb: u16, div: u16| ((a * wa + b * wb) / div) as u8;
	let p0 = [r0 as u8, g0 as u8, b0 as u8, 255];
	let p1 = [r1 as u8, g1 as u8, b1 as u8, 255];
	let palette = if force_four_color || c0 > c1 {
		[
			p0,
			p1,
			[mix(r0, r1, 2, 1, 3), mix(g0, g1, 2, 1, 3), mix(b0, b1, 2, 1, 3), 255],
			[mix(r0, r1, 1, 2, 3), mix(g0, g1, 1, 2, 3), mix(b0, b1, 1, 2, 3), 255],
		]
	} else {
		[
			p0,
			p1,
			[mix(r0, r1, 1, 1, 2), mix(g0, g1, 1, 1, 2), mix(b0, b1, 1, 1, 2), 255],
			[0, 0, 0, 0],
		]
	};
	(palette, indices)
}

/// Decodes the 8-byte BC3 alpha part of a block into 16 alpha values.
fn alpha_values(block: &[u8; 8]) -> [u8; 16] {
	let [a0, a1, b0, b1, b2, b3, b4, b5] = *block;
	let (w0, w1) = (u16::from(a0), u16::from(a1));
	let mut table = [0u8; 8];
	table[0] = a0;
	table[1] = a1;
	if a0 > a1 {
		for step in 1u16..7 {
			if let Some(slot) = table.get_mut(usize::from(step) + 1) {
				*slot = ((w0 * (7 - step) + w1 * step) / 7) as u8;
			}
		}
	} else {
		for step in 1u16..5 {
			if let Some(slot) = table.get_mut(usize::from(step) + 1) {
				*slot = ((w0 * (5 - step) + w1 * step) / 5) as u8;
			}
		}
		table[6] = 0;
		table[7] = 255;
	}
	let bits = u64::from_le_bytes([b0, b1, b2, b3, b4, b5, 0, 0]);
	let mut out = [0u8; 16];
	for (pixel, slot) in out.iter_mut().enumerate() {
		let index = ((bits >> (3 * pixel)) & 0x7) as usize;
		*slot = table.get(index).copied().unwrap_or(0);
	}
	out
}

fn split8(bytes: &[u8]) -> Option<[u8; 8]> {
	<[u8; 8]>::try_from(bytes).ok()
}

fn decode_bc1_block(bytes: &[u8]) -> Block {
	let mut block = [[0u8; 16]; 4];
	let Some(color) = split8(bytes) else {
		return block;
	};
	let (palette, indices) = color_palette(&color, false);
	write_block(&mut block, |pixel| {
		let index = ((indices >> (2 * pixel)) & 0x3) as usize;
		palette.get(index).copied().unwrap_or([0, 0, 0, 0])
	});
	block
}

/// Decodes a BC1 block of RGB-only data (HAP RGB), forcing every pixel opaque.
///
/// HAP RGB has no alpha channel, so index 3 of a three-color block is plain
/// black rather than BC1's punch-through transparent black.
fn decode_bc1_opaque_block(bytes: &[u8]) -> Block {
	let mut block = decode_bc1_block(bytes);
	for row in &mut block {
		for pixel in row.as_chunks_mut::<4>().0 {
			pixel[3] = 255;
		}
	}
	block
}

fn decode_bc3_block(bytes: &[u8]) -> Block {
	let mut block = [[0u8; 16]; 4];
	let (Some(alpha), Some(color)) = (bytes.get(..8).and_then(split8), bytes.get(8..16).and_then(split8)) else {
		return block;
	};
	let alphas = alpha_values(&alpha);
	let (palette, indices) = color_palette(&color, true);
	write_block(&mut block, |pixel| {
		let index = ((indices >> (2 * pixel)) & 0x3) as usize;
		let [r, g, b, _] = palette.get(index).copied().unwrap_or([0, 0, 0, 255]);
		[r, g, b, alphas.get(pixel).copied().unwrap_or(255)]
	});
	block
}

/// Converts one scaled-YCoCg pixel (`r=Co`, `g=Cg`, `b=scale`, `a=Y`) to RGBA.
///
/// The scale uses the same continuous formula as the GPU shader and the HAP
/// reference `hapqconversion.frag` (`scale = b * 255 / 8 + 1` on normalized
/// `b`, i.e. `b / 8 + 1` on the byte), so the CPU fallback and thumbnails
/// match the GPU path exactly for every expanded 5-bit blue value.
pub fn ycocg_to_rgba(pixel: [u8; 4]) -> [u8; 4] {
	let [co_raw, cg_raw, scale_raw, y_raw] = pixel;
	let scale = f32::from(scale_raw) / 8.0 + 1.0;
	let co = (f32::from(co_raw) - 128.0) / scale;
	let cg = (f32::from(cg_raw) - 128.0) / scale;
	let y = f32::from(y_raw);
	let clamp = |value: f32| value.round().clamp(0.0, 255.0) as u8;
	[clamp(y + co - cg), clamp(y + cg), clamp(y - co - cg), 255]
}

fn decode_bc3_ycocg_block(bytes: &[u8]) -> Block {
	let mut block = decode_bc3_block(bytes);
	for row in &mut block {
		for pixel in row.as_chunks_mut::<4>().0 {
			*pixel = ycocg_to_rgba(*pixel);
		}
	}
	block
}

/// Fills a block row-major (pixel 0 = top-left) from a per-pixel function.
fn write_block(block: &mut Block, mut pixel_at: impl FnMut(usize) -> [u8; 4]) {
	for (row_index, row) in block.iter_mut().enumerate() {
		for (column, out) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
			*out = pixel_at(row_index * 4 + column);
		}
	}
}

/// Decodes a full image of blocks of `block_bytes` each with `decode_block`.
fn decode_blocks(
	data: &[u8],
	width: u32,
	height: u32,
	block_bytes: usize,
	decode_block: fn(&[u8]) -> Block,
) -> Result<Vec<u8>, MediaError> {
	if !dimensions_are_valid(width, height) {
		return Err(invalid(format!("invalid dimensions {width}x{height}")));
	}
	let blocks_x = usize::try_from(width.div_ceil(4)).map_err(|_| invalid("width overflow"))?;
	let blocks_y = usize::try_from(height.div_ceil(4)).map_err(|_| invalid("height overflow"))?;
	let expected = blocks_x
		.checked_mul(blocks_y)
		.and_then(|blocks| blocks.checked_mul(block_bytes))
		.ok_or_else(|| invalid("block data size overflow"))?;
	if data.len() != expected {
		return Err(invalid(format!(
			"block data has {} bytes, expected {expected} for {width}x{height}",
			data.len()
		)));
	}
	let mut rgba = vec![0u8; rgba_len(width, height)?];
	let width = usize::try_from(width).map_err(|_| invalid("width overflow"))?;
	let height = usize::try_from(height).map_err(|_| invalid("height overflow"))?;
	let row_stride = width.saturating_mul(4);

	for (block_index, bytes) in data.chunks_exact(block_bytes).enumerate() {
		let block_x = block_index % blocks_x;
		let block_y = block_index / blocks_x;
		let block = decode_block(bytes);
		let x0 = block_x * 4;
		let visible = width.saturating_sub(x0).min(4);
		let visible_bytes = visible * 4;
		for (row_offset, row) in block.iter().enumerate() {
			let y = block_y * 4 + row_offset;
			if y >= height {
				break;
			}
			let start = y * row_stride + x0 * 4;
			if let (Some(destination), Some(source)) =
				(rgba.get_mut(start..start + visible_bytes), row.get(..visible_bytes))
			{
				destination.copy_from_slice(source);
			}
		}
	}
	Ok(rgba)
}

/// Decodes BC1/DXT1 blocks to tightly packed RGBA8.
pub fn decode_bc1_to_rgba8(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, MediaError> {
	decode_blocks(data, width, height, 8, decode_bc1_block)
}

/// Decodes RGB-only BC1 blocks (HAP RGB) to tightly packed, fully opaque RGBA8.
///
/// Unlike [`decode_bc1_to_rgba8`], index 3 of three-color blocks decodes as
/// opaque black, matching the GPU path which ignores BC1 alpha for HAP RGB.
pub fn decode_bc1_opaque_to_rgba8(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, MediaError> {
	decode_blocks(data, width, height, 8, decode_bc1_opaque_block)
}

/// Decodes BC3/DXT5 blocks to tightly packed RGBA8.
pub fn decode_bc3_to_rgba8(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, MediaError> {
	decode_blocks(data, width, height, 16, decode_bc3_block)
}

/// Decodes HAP Q scaled-YCoCg BC3 blocks to opaque RGBA8.
///
/// Per pixel: `scale = b / 8 + 1`, `co = (r - 128) / scale`,
/// `cg = (g - 128) / scale`, `y = a`, then `R = y + co - cg`, `G = y + cg`,
/// `B = y - co - cg`, `A = 255`.
pub fn decode_bc3_ycocg_to_rgba8(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, MediaError> {
	decode_blocks(data, width, height, 16, decode_bc3_ycocg_block)
}

/// Converts any decoded frame payload to RGBA8.
pub fn decode_frame_to_rgba8(frame: &DecodedFrame) -> Result<Vec<u8>, MediaError> {
	let result = match &frame.payload {
		// BC1 payloads only come from HAP RGB, which carries no alpha.
		PixelPayload::Bc1(data) => decode_bc1_opaque_to_rgba8(data, frame.width, frame.height),
		PixelPayload::Bc3(data) => decode_bc3_to_rgba8(data, frame.width, frame.height),
		PixelPayload::Bc3YCoCg(data) => decode_bc3_ycocg_to_rgba8(data, frame.width, frame.height),
		PixelPayload::Rgba8(data) => {
			if !dimensions_are_valid(frame.width, frame.height) {
				Err(invalid("invalid dimensions"))
			} else if data.len() != rgba_len(frame.width, frame.height)? {
				Err(invalid("RGBA8 buffer length does not match dimensions"))
			} else {
				Ok(data.clone())
			}
		}
	};
	result.map_err(|error| match error {
		MediaError::Decode { detail, .. } => MediaError::Decode {
			frame_index: frame.frame_index,
			detail,
		},
		other => other,
	})
}

/// Builds an aspect-preserving box-filtered thumbnail no larger than `max_w`×`max_h`.
///
/// Images smaller than the bounds keep their size. Invalid input (zero
/// dimensions or a buffer shorter than `width * height * 4`) yields a single
/// opaque black pixel so the UI always has something to draw.
pub fn create_thumbnail(rgba: &[u8], width: u32, height: u32, max_w: u32, max_h: u32) -> ThumbnailImage {
	let fallback = ThumbnailImage {
		width: 1,
		height: 1,
		rgba: vec![0, 0, 0, 255],
	};
	let Ok(needed) = rgba_len(width, height) else {
		return fallback;
	};
	if width == 0 || height == 0 || max_w == 0 || max_h == 0 || rgba.len() < needed {
		return fallback;
	}
	let scale = (f64::from(max_w) / f64::from(width))
		.min(f64::from(max_h) / f64::from(height))
		.min(1.0);
	let thumb_w = ((f64::from(width) * scale).round() as u32).clamp(1, max_w.min(width));
	let thumb_h = ((f64::from(height) * scale).round() as u32).clamp(1, max_h.min(height));

	let (src_w, src_h) = (width as usize, height as usize);
	let (dst_w, dst_h) = (thumb_w as usize, thumb_h as usize);
	let mut out = Vec::with_capacity(dst_w.saturating_mul(dst_h).saturating_mul(4));
	for dy in 0..dst_h {
		let y0 = dy * src_h / dst_h;
		let y1 = ((dy + 1) * src_h / dst_h).max(y0 + 1).min(src_h);
		for dx in 0..dst_w {
			let x0 = dx * src_w / dst_w;
			let x1 = ((dx + 1) * src_w / dst_w).max(x0 + 1).min(src_w);
			let mut sum = [0u64; 4];
			let mut count = 0u64;
			for y in y0..y1 {
				let row_start = (y * src_w + x0) * 4;
				let row_end = (y * src_w + x1) * 4;
				if let Some(row) = rgba.get(row_start..row_end) {
					for pixel in row.as_chunks::<4>().0 {
						for (total, &channel) in sum.iter_mut().zip(pixel) {
							*total += u64::from(channel);
						}
						count += 1;
					}
				}
			}
			let count = count.max(1);
			out.extend(sum.iter().map(|&total| ((total + count / 2) / count) as u8));
		}
	}
	ThumbnailImage {
		width: thumb_w,
		height: thumb_h,
		rgba: out,
	}
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;
	use crate::media::test_fixtures::{bc1_solid_block, bc3_solid_block, bc3_ycocg_solid_block, rgb_to_565};

	fn pixel(rgba: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
		let start = ((y * width + x) * 4) as usize;
		rgba[start..start + 4].try_into().unwrap()
	}

	#[test]
	fn bc1_solid_and_interpolated() {
		let block = bc1_solid_block([255, 0, 0]);
		let rgba = decode_bc1_to_rgba8(&block, 4, 4).unwrap();
		assert!(rgba.as_chunks::<4>().0.iter().all(|p| *p == [255, 0, 0, 255]));

		// c0 = white, c1 = black, four-color mode; indices 0,1,2,3 on the first row.
		let mut block = [0u8; 8];
		block[..2].copy_from_slice(&0xFFFFu16.to_le_bytes());
		block[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
		block[4] = 0b11_10_01_00;
		let rgba = decode_bc1_to_rgba8(&block, 4, 4).unwrap();
		assert_eq!(pixel(&rgba, 4, 0, 0), [255, 255, 255, 255]);
		assert_eq!(pixel(&rgba, 4, 1, 0), [0, 0, 0, 255]);
		assert_eq!(pixel(&rgba, 4, 2, 0), [170, 170, 170, 255]);
		assert_eq!(pixel(&rgba, 4, 3, 0), [85, 85, 85, 255]);
		assert_eq!(pixel(&rgba, 4, 0, 1), [255, 255, 255, 255]);
	}

	#[test]
	fn bc1_three_color_mode_has_transparent_black() {
		let mut block = [0u8; 8];
		block[..2].copy_from_slice(&0x0000u16.to_le_bytes());
		block[2..4].copy_from_slice(&0xFFFFu16.to_le_bytes());
		block[4] = 0b11_10_01_00;
		let rgba = decode_bc1_to_rgba8(&block, 4, 4).unwrap();
		assert_eq!(pixel(&rgba, 4, 2, 0), [127, 127, 127, 255]);
		assert_eq!(pixel(&rgba, 4, 3, 0), [0, 0, 0, 0]);
	}

	#[test]
	fn hap_rgb_three_color_black_is_opaque() {
		let mut block = [0u8; 8];
		block[..2].copy_from_slice(&0x0000u16.to_le_bytes());
		block[2..4].copy_from_slice(&0xFFFFu16.to_le_bytes());
		block[4] = 0b11_10_01_00;
		let rgba = decode_bc1_opaque_to_rgba8(&block, 4, 4).unwrap();
		assert_eq!(pixel(&rgba, 4, 2, 0), [127, 127, 127, 255]);
		assert_eq!(pixel(&rgba, 4, 3, 0), [0, 0, 0, 255]);

		// Frame dispatch treats BC1 payloads as HAP RGB.
		let frame = DecodedFrame {
			frame_index: 0,
			width: 4,
			height: 4,
			payload: PixelPayload::Bc1(block.to_vec()),
		};
		let rgba = decode_frame_to_rgba8(&frame).unwrap();
		assert!(rgba.as_chunks::<4>().0.iter().all(|p| p[3] == 255));
	}

	#[test]
	fn bc3_alpha_modes() {
		let block = bc3_solid_block([0, 255, 0, 128]);
		let rgba = decode_bc3_to_rgba8(&block, 4, 4).unwrap();
		assert!(rgba.as_chunks::<4>().0.iter().all(|p| *p == [0, 255, 0, 128]));

		// Eight-alpha mode: a0=255, a1=0; pixel1 index 1 (0), pixel2 index 2 (6/7*255).
		let mut block = bc3_solid_block([0, 0, 255, 0]);
		block[0] = 255;
		block[1] = 0;
		let bits: u64 = (1 << 3) | (2 << 6) | (7 << 9);
		block[2..8].copy_from_slice(&bits.to_le_bytes()[..6]);
		let rgba = decode_bc3_to_rgba8(&block, 4, 4).unwrap();
		assert_eq!(pixel(&rgba, 4, 0, 0)[3], 255);
		assert_eq!(pixel(&rgba, 4, 1, 0)[3], 0);
		assert_eq!(pixel(&rgba, 4, 2, 0)[3], 218);
		assert_eq!(pixel(&rgba, 4, 3, 0)[3], 36);

		// Six-alpha mode: a0 <= a1, indices 6 and 7 are 0 and 255.
		block[0] = 10;
		block[1] = 20;
		let bits: u64 = (6 << 3) | (7 << 6) | (2 << 9);
		block[2..8].copy_from_slice(&bits.to_le_bytes()[..6]);
		let rgba = decode_bc3_to_rgba8(&block, 4, 4).unwrap();
		assert_eq!(pixel(&rgba, 4, 0, 0)[3], 10);
		assert_eq!(pixel(&rgba, 4, 1, 0)[3], 0);
		assert_eq!(pixel(&rgba, 4, 2, 0)[3], 255);
		assert_eq!(pixel(&rgba, 4, 3, 0)[3], 12);
	}

	#[test]
	fn bc3_color_block_is_always_four_color() {
		let mut block = [0u8; 16];
		block[0] = 255;
		block[1] = 255;
		// c0 < c1 would be three-color in BC1, but BC3 always interpolates four colors.
		block[8..10].copy_from_slice(&0x0000u16.to_le_bytes());
		block[10..12].copy_from_slice(&0xFFFFu16.to_le_bytes());
		block[12] = 0b11_00_00_00;
		let rgba = decode_bc3_to_rgba8(&block, 4, 4).unwrap();
		assert_eq!(pixel(&rgba, 4, 3, 0), [170, 170, 170, 255]);
	}

	#[test]
	fn ycocg_formula_matches_spec() {
		// Co = 132 (5-bit 16), Cg = 130 (6-bit 32), scale byte 0 → scale 1, Y = 100.
		let mut block = [0u8; 16];
		block[0] = 100;
		block[1] = 100;
		let color = (16u16 << 11) | (32 << 5);
		block[8..10].copy_from_slice(&color.to_le_bytes());
		block[10..12].copy_from_slice(&color.to_le_bytes());
		let rgba = decode_bc3_ycocg_to_rgba8(&block, 4, 4).unwrap();
		assert_eq!(pixel(&rgba, 4, 0, 0), [102, 102, 94, 255]);

		// Scale byte 255 → 255 / 8 + 1 = 32.875 (continuous, like the GPU shader).
		assert_eq!(ycocg_to_rgba([128 + 66, 128, 255, 100]), [102, 100, 98, 255]);
		// Clamping.
		assert_eq!(ycocg_to_rgba([255, 0, 0, 255]), [255, 127, 255, 255]);
		assert_eq!(ycocg_to_rgba([0, 0, 0, 0]), [0, 0, 255, 255]);
	}

	#[test]
	fn ycocg_scale_matches_gpu_formula_for_non_multiple_of_eight_blue() {
		// 5-bit blue 7 expands to 57, which is not a multiple of 8: the GPU uses
		// 57 / 8 + 1 = 8.125, whereas `(57 >> 3) + 1` would truncate to 8.
		let block = bc3_solid_block([128 + 65, 128 - 65, 57, 120]);
		let scale_byte = {
			let rgba = decode_bc3_to_rgba8(&block, 4, 4).unwrap();
			pixel(&rgba, 4, 0, 0)
		};
		assert_eq!(scale_byte[2], 57);
		let rgba = decode_bc3_ycocg_to_rgba8(&block, 4, 4).unwrap();
		let out = pixel(&rgba, 4, 0, 0);
		let [co_raw, cg_raw, scale_raw, y_raw] = scale_byte.map(f32::from);
		let scale = scale_raw / 8.0 + 1.0;
		let (co, cg) = ((co_raw - 128.0) / scale, (cg_raw - 128.0) / scale);
		let expected = [y_raw + co - cg, y_raw + cg, y_raw - co - cg].map(|v| v.round().clamp(0.0, 255.0) as u8);
		assert_eq!(&out[..3], &expected[..]);
		assert_eq!(out[3], 255);

		// Every expanded 5-bit blue value uses the continuous scale.
		for b5 in 0u8..32 {
			let byte = (b5 << 3) | (b5 >> 2);
			let scale = f32::from(byte) / 8.0 + 1.0;
			let co = (f32::from(200u8) - 128.0) / scale;
			let expected_r = (100.0 + co - (f32::from(128u8) - 128.0) / scale)
				.round()
				.clamp(0.0, 255.0) as u8;
			assert_eq!(ycocg_to_rgba([200, 128, byte, 100])[0], expected_r, "b5={b5}");
		}
	}

	#[test]
	fn ycocg_fixture_blocks_decode_close_to_source() {
		for rgb in [
			[200, 40, 90],
			[10, 220, 30],
			[128, 128, 128],
			[0, 0, 0],
			[255, 255, 255],
		] {
			let block = bc3_ycocg_solid_block(rgb);
			let rgba = decode_bc3_ycocg_to_rgba8(&block, 4, 4).unwrap();
			let out = pixel(&rgba, 4, 0, 0);
			for channel in 0..3 {
				let diff = (i32::from(out[channel]) - i32::from(rgb[channel])).abs();
				assert!(diff <= 12, "{rgb:?} -> {out:?}");
			}
			assert_eq!(out[3], 255);
		}
	}

	#[test]
	fn partial_blocks_are_cropped() {
		// 6x5 image = 2x2 blocks; each block a different color.
		let colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]];
		let data: Vec<u8> = colors.iter().flat_map(|&c| bc1_solid_block(c)).collect();
		let rgba = decode_bc1_to_rgba8(&data, 6, 5).unwrap();
		assert_eq!(rgba.len(), 6 * 5 * 4);
		assert_eq!(pixel(&rgba, 6, 0, 0), [255, 0, 0, 255]);
		assert_eq!(pixel(&rgba, 6, 5, 0), [0, 255, 0, 255]);
		assert_eq!(pixel(&rgba, 6, 0, 4), [0, 0, 255, 255]);
		assert_eq!(pixel(&rgba, 6, 5, 4), [255, 255, 255, 255]);
	}

	#[test]
	fn rejects_wrong_lengths_and_dimensions() {
		assert!(decode_bc1_to_rgba8(&[0; 7], 4, 4).is_err());
		assert!(decode_bc3_to_rgba8(&[0; 17], 4, 4).is_err());
		assert!(decode_bc3_ycocg_to_rgba8(&[0; 16], 0, 4).is_err());
		assert!(decode_bc1_to_rgba8(&[0; 8], 9000, 1).is_err());
	}

	#[test]
	fn frame_dispatch_and_rgba_passthrough() {
		let frame = DecodedFrame {
			frame_index: 4,
			width: 4,
			height: 4,
			payload: PixelPayload::Bc1(bc1_solid_block([0, 0, 255]).to_vec()),
		};
		let rgba = decode_frame_to_rgba8(&frame).unwrap();
		assert_eq!(pixel(&rgba, 4, 3, 3), [0, 0, 255, 255]);

		let rgba_frame = DecodedFrame {
			frame_index: 1,
			width: 1,
			height: 1,
			payload: PixelPayload::Rgba8(vec![1, 2, 3, 4]),
		};
		assert_eq!(decode_frame_to_rgba8(&rgba_frame).unwrap(), vec![1, 2, 3, 4]);
		let bad = DecodedFrame {
			payload: PixelPayload::Bc3(vec![0; 3]),
			..frame
		};
		assert!(matches!(
			decode_frame_to_rgba8(&bad),
			Err(MediaError::Decode { frame_index: 4, .. })
		));
		assert_eq!(rgb_to_565([255, 255, 255]), 0xFFFF);
	}

	#[test]
	fn thumbnails_preserve_aspect_and_average() {
		let rgba: Vec<u8> = (0..1920 * 1080).flat_map(|_| [10u8, 20, 30, 255]).collect();
		let thumb = create_thumbnail(&rgba, 1920, 1080, 192, 108);
		assert_eq!((thumb.width, thumb.height), (192, 108));
		assert!(thumb.is_valid());
		assert!(thumb.rgba.as_chunks::<4>().0.iter().all(|p| *p == [10, 20, 30, 255]));

		let tall = vec![255u8; 100 * 400 * 4];
		let thumb = create_thumbnail(&tall, 100, 400, 192, 108);
		assert_eq!((thumb.width, thumb.height), (27, 108));
		assert!(thumb.is_valid());

		// Smaller than bounds: unchanged size.
		let small = vec![7u8; 8 * 2 * 4];
		let thumb = create_thumbnail(&small, 8, 2, 192, 108);
		assert_eq!((thumb.width, thumb.height), (8, 2));
		assert_eq!(thumb.rgba, small);

		// Box filter averages a 2x1 black/white image into gray.
		let pair = [0, 0, 0, 255, 255, 255, 255, 255];
		let thumb = create_thumbnail(&pair, 2, 1, 1, 1);
		assert_eq!(thumb.rgba, vec![128, 128, 128, 255]);
	}

	#[test]
	fn thumbnail_handles_invalid_input() {
		let thumb = create_thumbnail(&[0; 3], 4, 4, 192, 108);
		assert_eq!((thumb.width, thumb.height), (1, 1));
		assert!(thumb.is_valid());
		assert!(create_thumbnail(&[], 0, 0, 192, 108).is_valid());
		assert!(create_thumbnail(&[0; 16], 2, 2, 0, 0).is_valid());
	}
}
