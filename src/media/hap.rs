//! Pure-Rust HAP frame parser.
//!
//! A HAP frame is a sequence of sections, each introduced by a 4-byte header
//! (`u24 LE` length + section type) or, when the 24-bit length is zero, an 8-byte
//! header with a `u32 LE` length. The top-level section type encodes both the
//! texture format (low nibble) and the compressor (high nibble):
//!
//! | high nibble | compressor                         |
//! | ----------- | ---------------------------------- |
//! | `0xA`       | none                               |
//! | `0xB`       | Snappy                             |
//! | `0xC`       | chunked ("complex"), see below     |
//!
//! Chunked frames start with a Decode Instructions container (`0x01`) holding a
//! Compressor Table (`0x02`, one byte per chunk: `0x0A` none / `0x0B` Snappy), a
//! Chunk Size Table (`0x03`, `u32 LE` per chunk), and an optional Chunk Offset
//! Table (`0x04`, `u32 LE` per chunk, relative to the first byte after the
//! container). The compressed chunks follow; their decompressed bytes are
//! concatenated in table order.

use crate::error::MediaError;
use crate::model::{ClipKind, DecodedFrame, PixelPayload};

use super::dimensions_are_valid;

/// Texture format nibble for RGB DXT1 (`Hap1`).
const FORMAT_RGB_DXT1: u8 = 0x0B;
/// Texture format nibble for RGBA DXT5 (`Hap5`).
const FORMAT_RGBA_DXT5: u8 = 0x0E;
/// Texture format nibble for scaled YCoCg DXT5 (`HapY`).
const FORMAT_YCOCG_DXT5: u8 = 0x0F;

/// Compressor nibble: uncompressed.
const COMPRESSOR_NONE: u8 = 0x0A;
/// Compressor nibble: single Snappy stream.
const COMPRESSOR_SNAPPY: u8 = 0x0B;
/// Compressor nibble: chunked with decode instructions.
const COMPRESSOR_COMPLEX: u8 = 0x0C;

/// Decode Instructions container section.
pub const SECTION_DECODE_INSTRUCTIONS: u8 = 0x01;
/// Compressor Table section.
pub const SECTION_COMPRESSOR_TABLE: u8 = 0x02;
/// Chunk Size Table section.
pub const SECTION_CHUNK_SIZE_TABLE: u8 = 0x03;
/// Chunk Offset Table section.
pub const SECTION_CHUNK_OFFSET_TABLE: u8 = 0x04;
/// Multiple-image section used by HAP Q Alpha (unsupported).
const SECTION_MULTIPLE_IMAGES: u8 = 0x0D;

/// Chunk compressor byte: stored uncompressed.
pub const CHUNK_COMPRESSOR_NONE: u8 = 0x0A;
/// Chunk compressor byte: Snappy compressed.
pub const CHUNK_COMPRESSOR_SNAPPY: u8 = 0x0B;

/// Upper bound on the chunk count of a single frame.
pub const MAX_CHUNKS: usize = 4096;

/// A parsed section header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionHeader {
	/// Section type byte.
	pub section_type: u8,
	/// Header length in bytes (4 or 8).
	pub header_len: usize,
	/// Length of the section body in bytes.
	pub body_len: usize,
}

impl SectionHeader {
	/// Total size of header plus body, or `None` on overflow.
	pub fn total_len(&self) -> Option<usize> {
		self.header_len.checked_add(self.body_len)
	}
}

fn decode_error(frame_index: u32, detail: impl Into<String>) -> MediaError {
	MediaError::Decode {
		frame_index,
		detail: detail.into(),
	}
}

/// Parses the section header at the start of `bytes`.
pub fn parse_section_header(bytes: &[u8]) -> Option<SectionHeader> {
	let header = bytes.get(..4)?;
	let [l0, l1, l2, section_type] = <[u8; 4]>::try_from(header).ok()?;
	let short_len = u32::from_le_bytes([l0, l1, l2, 0]);
	if short_len != 0 {
		return Some(SectionHeader {
			section_type,
			header_len: 4,
			body_len: usize::try_from(short_len).ok()?,
		});
	}
	let extended = bytes.get(4..8)?;
	let long_len = u32::from_le_bytes(<[u8; 4]>::try_from(extended).ok()?);
	Some(SectionHeader {
		section_type,
		header_len: 8,
		body_len: usize::try_from(long_len).ok()?,
	})
}

/// Splits one section off the front of `bytes`, returning `(header, body, rest)`.
fn split_section(bytes: &[u8]) -> Option<(SectionHeader, &[u8], &[u8])> {
	let header = parse_section_header(bytes)?;
	let total = header.total_len()?;
	let body = bytes.get(header.header_len..total)?;
	let rest = bytes.get(total..)?;
	Some((header, body, rest))
}

/// Number of BC bytes a HAP frame of `kind` and size `width`×`height` decodes to.
///
/// Each 4×4 block takes 8 bytes for `Hap` (BC1) and 16 bytes for `HapAlpha` and
/// `HapQ` (BC3). Partial edge blocks are rounded up.
pub fn expected_hap_bytes(kind: ClipKind, width: u32, height: u32) -> Result<usize, MediaError> {
	let block_bytes: u64 = match kind {
		ClipKind::Hap => 8,
		ClipKind::HapAlpha | ClipKind::HapQ => 16,
		ClipKind::ImageRgba => {
			return Err(MediaError::UnsupportedFormat("still images are not HAP".into()));
		}
	};
	if !dimensions_are_valid(width, height) {
		return Err(MediaError::UnsupportedFormat(format!(
			"invalid HAP frame dimensions {width}x{height}"
		)));
	}
	let blocks = u64::from(width.div_ceil(4)).saturating_mul(u64::from(height.div_ceil(4)));
	usize::try_from(blocks.saturating_mul(block_bytes))
		.map_err(|_| MediaError::UnsupportedFormat("HAP frame too large".into()))
}

fn format_nibble(kind: ClipKind) -> Option<u8> {
	match kind {
		ClipKind::Hap => Some(FORMAT_RGB_DXT1),
		ClipKind::HapAlpha => Some(FORMAT_RGBA_DXT5),
		ClipKind::HapQ => Some(FORMAT_YCOCG_DXT5),
		ClipKind::ImageRgba => None,
	}
}

/// Top-level section type for `kind` combined with the compressor nibble.
///
/// `compressor` is `0xA` (none), `0xB` (Snappy), or `0xC` (chunked).
pub fn section_type_for(kind: ClipKind, compressor: u8) -> Option<u8> {
	let format = format_nibble(kind)?;
	match compressor {
		COMPRESSOR_NONE | COMPRESSOR_SNAPPY | COMPRESSOR_COMPLEX => Some((compressor << 4) | format),
		_ => None,
	}
}

/// Decodes a HAP frame and wraps it in a [`DecodedFrame`].
pub fn decode_hap_frame(
	data: &[u8],
	kind: ClipKind,
	width: u32,
	height: u32,
	frame_index: u32,
) -> Result<DecodedFrame, MediaError> {
	let payload = decode_hap_payload(data, kind, width, height, frame_index)?;
	Ok(DecodedFrame {
		frame_index,
		width,
		height,
		payload,
	})
}

/// Decodes the HAP sample `data` into BC blocks matching `kind`.
///
/// The output size is checked against [`expected_hap_bytes`] before any
/// allocation, and Snappy lengths are validated with
/// `snap::raw::decompress_len`, so hostile frames cannot trigger large
/// allocations or out-of-bounds writes.
pub fn decode_hap_payload(
	data: &[u8],
	kind: ClipKind,
	width: u32,
	height: u32,
	frame_index: u32,
) -> Result<PixelPayload, MediaError> {
	let expected = expected_hap_bytes(kind, width, height)?;
	let expected_format = format_nibble(kind).ok_or_else(|| decode_error(frame_index, "not a HAP clip kind"))?;
	let (header, body, _trailing) =
		split_section(data).ok_or_else(|| decode_error(frame_index, "truncated HAP section header"))?;

	if header.section_type == SECTION_MULTIPLE_IMAGES {
		return Err(MediaError::UnsupportedFormat(
			"multi-image HAP frames (HAP Q Alpha) are not supported".into(),
		));
	}
	let format = header.section_type & 0x0F;
	let compressor = header.section_type >> 4;
	if format != expected_format {
		return Err(decode_error(
			frame_index,
			format!(
				"section type 0x{:02X} does not match {} texture format",
				header.section_type,
				kind.label()
			),
		));
	}

	let blocks = match compressor {
		COMPRESSOR_NONE => {
			if body.len() != expected {
				return Err(decode_error(
					frame_index,
					format!("uncompressed frame has {} bytes, expected {expected}", body.len()),
				));
			}
			body.to_vec()
		}
		COMPRESSOR_SNAPPY => {
			let mut output = vec![0u8; expected];
			let written = decompress_snappy_into(body, &mut output, frame_index)?;
			if written != expected {
				return Err(decode_error(
					frame_index,
					format!("Snappy frame decoded to {written} bytes, expected {expected}"),
				));
			}
			output
		}
		COMPRESSOR_COMPLEX => decode_chunked(body, expected, frame_index)?,
		other => {
			return Err(decode_error(frame_index, format!("unknown HAP compressor 0x{other:X}")));
		}
	};

	Ok(match kind {
		ClipKind::Hap => PixelPayload::Bc1(blocks),
		ClipKind::HapAlpha => PixelPayload::Bc3(blocks),
		ClipKind::HapQ => PixelPayload::Bc3YCoCg(blocks),
		ClipKind::ImageRgba => PixelPayload::Rgba8(blocks),
	})
}

/// Decompresses a Snappy stream into `output`, which must be large enough.
///
/// Returns the number of bytes written.
fn decompress_snappy_into(input: &[u8], output: &mut [u8], frame_index: u32) -> Result<usize, MediaError> {
	let length = snap::raw::decompress_len(input)
		.map_err(|error| decode_error(frame_index, format!("invalid Snappy header: {error}")))?;
	if length > output.len() {
		return Err(decode_error(
			frame_index,
			format!("Snappy data claims {length} bytes, only {} expected", output.len()),
		));
	}
	let target = output
		.get_mut(..length)
		.ok_or_else(|| decode_error(frame_index, "Snappy output out of range"))?;
	snap::raw::Decoder::new()
		.decompress(input, target)
		.map_err(|error| decode_error(frame_index, format!("Snappy decode failed: {error}")))
}

/// Chunk layout extracted from the decode instructions.
#[derive(Debug, Default)]
struct DecodeInstructions<'a> {
	compressors: Option<&'a [u8]>,
	sizes: Option<Vec<usize>>,
	offsets: Option<Vec<usize>>,
}

impl<'a> DecodeInstructions<'a> {
	/// Records one instruction sub-section; unknown types are ignored.
	fn absorb(&mut self, header: SectionHeader, body: &'a [u8], frame_index: u32) -> Result<(), MediaError> {
		match header.section_type {
			SECTION_COMPRESSOR_TABLE => self.compressors = Some(body),
			SECTION_CHUNK_SIZE_TABLE => self.sizes = Some(read_u32_table(body, frame_index, "chunk size")?),
			SECTION_CHUNK_OFFSET_TABLE => self.offsets = Some(read_u32_table(body, frame_index, "chunk offset")?),
			_ => {}
		}
		Ok(())
	}

	/// Parses every sub-section of a Decode Instructions container body.
	fn parse_container(mut body: &'a [u8], frame_index: u32) -> Result<Self, MediaError> {
		let mut instructions = Self::default();
		while !body.is_empty() {
			let (header, section, rest) =
				split_section(body).ok_or_else(|| decode_error(frame_index, "truncated decode instruction section"))?;
			instructions.absorb(header, section, frame_index)?;
			body = rest;
		}
		Ok(instructions)
	}
}

fn read_u32_table(body: &[u8], frame_index: u32, name: &str) -> Result<Vec<usize>, MediaError> {
	if !body.len().is_multiple_of(4) {
		return Err(decode_error(
			frame_index,
			format!("{name} table length is not a multiple of 4"),
		));
	}
	if body.len() / 4 > MAX_CHUNKS {
		return Err(decode_error(frame_index, format!("{name} table has too many entries")));
	}
	body.as_chunks::<4>()
		.0
		.iter()
		.map(|bytes| {
			usize::try_from(u32::from_le_bytes(*bytes)).map_err(|_| decode_error(frame_index, "table entry overflow"))
		})
		.collect()
}

/// Decodes the body of a chunked (`0xC?`) section.
fn decode_chunked(body: &[u8], expected: usize, frame_index: u32) -> Result<Vec<u8>, MediaError> {
	let (first, first_body, after_first) =
		split_section(body).ok_or_else(|| decode_error(frame_index, "missing decode instructions"))?;

	let (instructions, frame_data) = if first.section_type == SECTION_DECODE_INSTRUCTIONS {
		(
			DecodeInstructions::parse_container(first_body, frame_index)?,
			after_first,
		)
	} else if matches!(
		first.section_type,
		SECTION_COMPRESSOR_TABLE | SECTION_CHUNK_SIZE_TABLE | SECTION_CHUNK_OFFSET_TABLE
	) {
		// Tolerate encoders that emit the tables without the 0x01 wrapper: consume
		// consecutive table sections; chunk data starts after the last one.
		let mut instructions = DecodeInstructions::default();
		instructions.absorb(first, first_body, frame_index)?;
		let mut rest = after_first;
		while let Some((header, section, next)) = split_section(rest) {
			if !matches!(
				header.section_type,
				SECTION_COMPRESSOR_TABLE | SECTION_CHUNK_SIZE_TABLE | SECTION_CHUNK_OFFSET_TABLE
			) {
				break;
			}
			instructions.absorb(header, section, frame_index)?;
			rest = next;
		}
		(instructions, rest)
	} else {
		return Err(decode_error(
			frame_index,
			format!(
				"chunked frame starts with unexpected section 0x{:02X}",
				first.section_type
			),
		));
	};

	let compressors = instructions
		.compressors
		.ok_or_else(|| decode_error(frame_index, "chunked frame lacks a compressor table"))?;
	let sizes = instructions
		.sizes
		.ok_or_else(|| decode_error(frame_index, "chunked frame lacks a chunk size table"))?;
	let chunk_count = compressors.len();
	if chunk_count == 0 || chunk_count > MAX_CHUNKS {
		return Err(decode_error(frame_index, format!("invalid chunk count {chunk_count}")));
	}
	if sizes.len() != chunk_count {
		return Err(decode_error(
			frame_index,
			"chunk size table does not match compressor table",
		));
	}
	if let Some(offsets) = &instructions.offsets
		&& offsets.len() != chunk_count
	{
		return Err(decode_error(
			frame_index,
			"chunk offset table does not match compressor table",
		));
	}

	let mut output = vec![0u8; expected];
	let mut written = 0usize;
	let mut sequential_offset = 0usize;
	for (chunk_index, (&compressor, &size)) in compressors.iter().zip(&sizes).enumerate() {
		let start = match &instructions.offsets {
			Some(offsets) => *offsets
				.get(chunk_index)
				.ok_or_else(|| decode_error(frame_index, "missing chunk offset"))?,
			None => sequential_offset,
		};
		let end = start
			.checked_add(size)
			.ok_or_else(|| decode_error(frame_index, "chunk range overflow"))?;
		sequential_offset = end;
		let input = frame_data
			.get(start..end)
			.ok_or_else(|| decode_error(frame_index, format!("chunk {chunk_index} lies outside the frame")))?;
		let target = output
			.get_mut(written..)
			.ok_or_else(|| decode_error(frame_index, "chunk output overflow"))?;
		let produced = match compressor {
			CHUNK_COMPRESSOR_NONE => {
				let destination = target.get_mut(..input.len()).ok_or_else(|| {
					decode_error(frame_index, format!("chunk {chunk_index} overflows the frame buffer"))
				})?;
				destination.copy_from_slice(input);
				input.len()
			}
			CHUNK_COMPRESSOR_SNAPPY => decompress_snappy_into(input, target, frame_index)?,
			other => {
				return Err(decode_error(
					frame_index,
					format!("unknown chunk compressor 0x{other:02X}"),
				));
			}
		};
		written = written
			.checked_add(produced)
			.ok_or_else(|| decode_error(frame_index, "chunk output overflow"))?;
	}
	if written != expected {
		return Err(decode_error(
			frame_index,
			format!("chunked frame decoded to {written} bytes, expected {expected}"),
		));
	}
	Ok(output)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;
	use crate::media::test_fixtures::{FixtureCompression, encode_hap_frame, fixture_frame_payload, section_header};

	const KINDS: [ClipKind; 3] = [ClipKind::Hap, ClipKind::HapAlpha, ClipKind::HapQ];

	fn all_compressions() -> Vec<FixtureCompression> {
		vec![
			FixtureCompression::Uncompressed,
			FixtureCompression::Snappy,
			FixtureCompression::Chunked {
				chunks: 4,
				mixed: false,
				offset_table: false,
			},
			FixtureCompression::Chunked {
				chunks: 3,
				mixed: true,
				offset_table: true,
			},
			FixtureCompression::Chunked {
				chunks: 1,
				mixed: false,
				offset_table: true,
			},
		]
	}

	#[test]
	fn expected_sizes_round_up_blocks() {
		assert_eq!(expected_hap_bytes(ClipKind::Hap, 1920, 1080).unwrap(), 480 * 270 * 8);
		assert_eq!(expected_hap_bytes(ClipKind::HapAlpha, 5, 5).unwrap(), 4 * 16);
		assert_eq!(expected_hap_bytes(ClipKind::HapQ, 4, 4).unwrap(), 16);
		assert!(expected_hap_bytes(ClipKind::ImageRgba, 4, 4).is_err());
		assert!(expected_hap_bytes(ClipKind::Hap, 0, 4).is_err());
		assert!(expected_hap_bytes(ClipKind::Hap, 9000, 4).is_err());
	}

	#[test]
	fn section_headers_short_and_extended() {
		let short = section_header(0xBB, 10, false);
		assert_eq!(
			parse_section_header(&short),
			Some(SectionHeader {
				section_type: 0xBB,
				header_len: 4,
				body_len: 10
			})
		);
		let long = section_header(0xAB, 10, true);
		assert_eq!(
			parse_section_header(&long),
			Some(SectionHeader {
				section_type: 0xAB,
				header_len: 8,
				body_len: 10
			})
		);
		assert_eq!(parse_section_header(&[1, 0]), None);
		assert_eq!(parse_section_header(&[0, 0, 0, 0xAB, 1]), None);
	}

	#[test]
	fn round_trips_every_kind_and_compression() {
		for kind in KINDS {
			for compression in all_compressions() {
				for extended in [false, true] {
					let (width, height) = (37, 19);
					let raw = fixture_frame_payload(kind, width, height, 3).unwrap();
					let frame = encode_hap_frame(kind, &raw, compression, extended).unwrap();
					let decoded = decode_hap_frame(&frame, kind, width, height, 3).unwrap();
					assert_eq!(decoded.frame_index, 3);
					assert_eq!(decoded.payload.byte_len(), raw.len());
					let bytes = match (&decoded.payload, kind) {
						(PixelPayload::Bc1(b), ClipKind::Hap) => b,
						(PixelPayload::Bc3(b), ClipKind::HapAlpha) => b,
						(PixelPayload::Bc3YCoCg(b), ClipKind::HapQ) => b,
						other => panic!("unexpected payload {other:?}"),
					};
					assert_eq!(bytes, &raw, "{kind:?} {compression:?} ext={extended}");
				}
			}
		}
	}

	#[test]
	fn unwrapped_chunk_tables_are_accepted() {
		let kind = ClipKind::Hap;
		let raw = fixture_frame_payload(kind, 16, 16, 0).unwrap();
		let half = raw.len() / 2;
		let chunks = [
			snap::raw::Encoder::new().compress_vec(&raw[..half]).unwrap(),
			raw[half..].to_vec(),
		];
		let mut body = Vec::new();
		body.extend(section_header(SECTION_COMPRESSOR_TABLE, 2, false));
		body.extend([CHUNK_COMPRESSOR_SNAPPY, CHUNK_COMPRESSOR_NONE]);
		body.extend(section_header(SECTION_CHUNK_SIZE_TABLE, 8, false));
		for chunk in &chunks {
			body.extend((chunk.len() as u32).to_le_bytes());
		}
		for chunk in &chunks {
			body.extend(chunk);
		}
		let mut frame = section_header(0xCB, body.len(), false);
		frame.extend(body);
		let decoded = decode_hap_payload(&frame, kind, 16, 16, 0).unwrap();
		assert_eq!(decoded, PixelPayload::Bc1(raw));
	}

	#[test]
	fn unknown_instruction_sections_are_skipped() {
		let kind = ClipKind::HapQ;
		let raw = fixture_frame_payload(kind, 8, 8, 1).unwrap();
		let mut instructions = Vec::new();
		instructions.extend(section_header(0x7E, 3, false));
		instructions.extend([9, 9, 9]);
		instructions.extend(section_header(SECTION_COMPRESSOR_TABLE, 1, false));
		instructions.push(CHUNK_COMPRESSOR_NONE);
		instructions.extend(section_header(SECTION_CHUNK_SIZE_TABLE, 4, true));
		instructions.extend((raw.len() as u32).to_le_bytes());
		let mut body = section_header(SECTION_DECODE_INSTRUCTIONS, instructions.len(), false);
		body.extend(instructions);
		body.extend(&raw);
		let mut frame = section_header(0xCF, body.len(), false);
		frame.extend(body);
		assert_eq!(
			decode_hap_payload(&frame, kind, 8, 8, 1).unwrap(),
			PixelPayload::Bc3YCoCg(raw)
		);
	}

	#[test]
	fn rejects_mismatched_format_and_sizes() {
		let raw = fixture_frame_payload(ClipKind::Hap, 8, 8, 0).unwrap();
		let frame = encode_hap_frame(ClipKind::Hap, &raw, FixtureCompression::Snappy, false).unwrap();
		// Wrong kind for the section type.
		assert!(decode_hap_payload(&frame, ClipKind::HapAlpha, 8, 8, 0).is_err());
		// Wrong dimensions → wrong expected size.
		assert!(decode_hap_payload(&frame, ClipKind::Hap, 16, 8, 0).is_err());
		assert!(decode_hap_payload(&frame, ClipKind::Hap, 4, 8, 0).is_err());

		let mut short = section_header(0xAB, raw.len() - 1, false);
		short.extend(&raw[1..]);
		assert!(decode_hap_payload(&short, ClipKind::Hap, 8, 8, 0).is_err());

		let mut multi = section_header(0x0D, 4, false);
		multi.extend([0; 4]);
		assert!(matches!(
			decode_hap_payload(&multi, ClipKind::HapQ, 8, 8, 0),
			Err(MediaError::UnsupportedFormat(_))
		));
		assert!(decode_hap_payload(&[], ClipKind::Hap, 8, 8, 0).is_err());
	}

	#[test]
	fn rejects_hostile_snappy_lengths() {
		// A Snappy stream claiming ~4 GiB of output must be rejected before allocating it.
		let mut body = vec![0xFF, 0xFF, 0xFF, 0xFF, 0x0F];
		body.extend([0; 8]);
		let mut frame = section_header(0xBB, body.len(), false);
		frame.extend(body);
		let error = decode_hap_payload(&frame, ClipKind::Hap, 8, 8, 5).unwrap_err();
		assert!(matches!(error, MediaError::Decode { frame_index: 5, .. }));
	}

	#[test]
	fn fuzzed_frames_never_panic() {
		for kind in KINDS {
			for compression in all_compressions() {
				let raw = fixture_frame_payload(kind, 12, 8, 2).unwrap();
				let frame = encode_hap_frame(kind, &raw, compression, false).unwrap();
				for position in 0..frame.len() {
					for flip in [0x01u8, 0x80, 0xFF] {
						let mut mutated = frame.clone();
						mutated[position] ^= flip;
						let _ = decode_hap_payload(&mutated, kind, 12, 8, 0);
					}
					let _ = decode_hap_payload(&frame[..position], kind, 12, 8, 0);
				}
			}
		}
	}

	#[test]
	fn section_types_combine_nibbles() {
		assert_eq!(section_type_for(ClipKind::Hap, 0xA), Some(0xAB));
		assert_eq!(section_type_for(ClipKind::HapAlpha, 0xB), Some(0xBE));
		assert_eq!(section_type_for(ClipKind::HapQ, 0xC), Some(0xCF));
		assert_eq!(section_type_for(ClipKind::ImageRgba, 0xA), None);
		assert_eq!(section_type_for(ClipKind::Hap, 0x3), None);
	}
}
