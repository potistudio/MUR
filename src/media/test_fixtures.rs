//! Pure-Rust builders for synthetic HAP `.mov` files and BC blocks.
//!
//! These helpers power the unit tests in [`crate::media`] as well as the
//! integration and performance tests under `tests/`. They never panic: every
//! fallible step returns [`MediaError`].

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::MediaError;
use crate::model::ClipKind;

use super::hap::{
	CHUNK_COMPRESSOR_NONE, CHUNK_COMPRESSOR_SNAPPY, SECTION_CHUNK_OFFSET_TABLE, SECTION_CHUNK_SIZE_TABLE,
	SECTION_COMPRESSOR_TABLE, SECTION_DECODE_INSTRUCTIONS, expected_hap_bytes, section_type_for,
};

fn fixture_error(detail: impl Into<String>) -> MediaError {
	MediaError::Corrupt {
		path: "<fixture>".to_string(),
		detail: detail.into(),
	}
}

fn len_u32(len: usize) -> Result<u32, MediaError> {
	u32::try_from(len).map_err(|_| fixture_error("length exceeds u32"))
}

// ---------------------------------------------------------------------------
// Temporary directories
// ---------------------------------------------------------------------------

/// A uniquely named directory under the system temp dir, removed on drop.
#[derive(Debug)]
pub struct TempDir {
	path: PathBuf,
}

impl TempDir {
	/// Creates a fresh directory whose name starts with `mur-{prefix}`.
	pub fn new(prefix: &str) -> std::io::Result<Self> {
		static COUNTER: AtomicU64 = AtomicU64::new(0);
		let count = COUNTER.fetch_add(1, Ordering::Relaxed);
		let nanos = std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.map(|elapsed| elapsed.as_nanos())
			.unwrap_or(0);
		let path = std::env::temp_dir().join(format!("mur-{prefix}-{}-{nanos}-{count}", std::process::id()));
		std::fs::create_dir_all(&path)?;
		Ok(Self { path })
	}

	/// Path of the directory.
	pub fn path(&self) -> &Path {
		&self.path
	}
}

impl Drop for TempDir {
	fn drop(&mut self) {
		let _ = std::fs::remove_dir_all(&self.path);
	}
}

// ---------------------------------------------------------------------------
// BC blocks
// ---------------------------------------------------------------------------

/// Quantizes an RGB888 color to RGB565 with rounding.
pub fn rgb_to_565(rgb: [u8; 3]) -> u16 {
	let [r, g, b] = rgb.map(u16::from);
	let r5 = (r * 31 + 127) / 255;
	let g6 = (g * 63 + 127) / 255;
	let b5 = (b * 31 + 127) / 255;
	(r5 << 11) | (g6 << 5) | b5
}

/// A BC1 block where every pixel has color `rgb` (565-quantized).
pub fn bc1_solid_block(rgb: [u8; 3]) -> [u8; 8] {
	let [lo, hi] = rgb_to_565(rgb).to_le_bytes();
	[lo, hi, lo, hi, 0, 0, 0, 0]
}

/// A BC3 block where every pixel has color `rgba` (RGB 565-quantized).
pub fn bc3_solid_block(rgba: [u8; 4]) -> [u8; 16] {
	let [r, g, b, a] = rgba;
	let [c0, c1, c2, c3, c4, c5, c6, c7] = bc1_solid_block([r, g, b]);
	[a, a, 0, 0, 0, 0, 0, 0, c0, c1, c2, c3, c4, c5, c6, c7]
}

/// A HAP Q (scaled YCoCg, scale 1) BC3 block approximating `rgb`.
pub fn bc3_ycocg_solid_block(rgb: [u8; 3]) -> [u8; 16] {
	let [r, g, b] = rgb.map(f32::from);
	let y = (r + 2.0 * g + b) / 4.0;
	let co = (r - b) / 2.0;
	let cg = (-r + 2.0 * g - b) / 4.0;
	let to_byte = |value: f32| value.round().clamp(0.0, 255.0) as u8;
	bc3_solid_block([to_byte(co + 128.0), to_byte(cg + 128.0), 0, to_byte(y)])
}

/// Deterministic pseudo-random BC payload for `frame_index` of a `kind` clip.
///
/// Runs of four identical blocks keep the data Snappy-compressible, while the
/// frame index seeds every block so frames are distinguishable byte for byte.
pub fn fixture_frame_payload(kind: ClipKind, width: u32, height: u32, frame_index: u32) -> Result<Vec<u8>, MediaError> {
	let total = expected_hap_bytes(kind, width, height)?;
	let block_bytes = if matches!(kind, ClipKind::Hap) { 8 } else { 16 };
	let mut out = Vec::with_capacity(total);
	let mut block_index = 0u64;
	while out.len() < total {
		let mut state = (u64::from(frame_index) << 32) ^ (block_index / 4) ^ 0x9E37_79B9_7F4A_7C15;
		let mut block = [0u8; 16];
		for byte in &mut block {
			state ^= state << 13;
			state ^= state >> 7;
			state ^= state << 17;
			*byte = (state >> 24) as u8;
		}
		out.extend(block.iter().take(block_bytes));
		block_index = block_index.saturating_add(1);
	}
	out.truncate(total);
	Ok(out)
}

// ---------------------------------------------------------------------------
// HAP frame encoding
// ---------------------------------------------------------------------------

/// How fixture HAP frames are compressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureCompression {
	/// Raw BC blocks (`0xA?`).
	Uncompressed,
	/// A single Snappy stream (`0xB?`).
	Snappy,
	/// Chunked frame with decode instructions (`0xC?`).
	Chunked {
		/// Number of chunks (at least 1).
		chunks: usize,
		/// Store every other chunk uncompressed (`0x0A`) instead of Snappy.
		mixed: bool,
		/// Emit a Chunk Offset Table (`0x04`).
		offset_table: bool,
	},
}

/// Builds a HAP section header for a body of `body_len` bytes.
///
/// With `extended`, or when the length does not fit in 24 bits, the 8-byte
/// form (zero 24-bit length followed by a `u32 LE`) is used.
pub fn section_header(section_type: u8, body_len: usize, extended: bool) -> Vec<u8> {
	let len = u32::try_from(body_len).unwrap_or(u32::MAX);
	if extended || len == 0 || len > 0x00FF_FFFF {
		let [l0, l1, l2, l3] = len.to_le_bytes();
		vec![0, 0, 0, section_type, l0, l1, l2, l3]
	} else {
		let [l0, l1, l2, _] = len.to_le_bytes();
		vec![l0, l1, l2, section_type]
	}
}

fn section(section_type: u8, body: &[u8], extended: bool) -> Vec<u8> {
	let mut out = section_header(section_type, body.len(), extended);
	out.extend_from_slice(body);
	out
}

fn snappy(data: &[u8]) -> Result<Vec<u8>, MediaError> {
	snap::raw::Encoder::new()
		.compress_vec(data)
		.map_err(|error| fixture_error(format!("snappy: {error}")))
}

/// Wraps raw BC `blocks` into a HAP frame of `kind` using `compression`.
pub fn encode_hap_frame(
	kind: ClipKind,
	blocks: &[u8],
	compression: FixtureCompression,
	extended: bool,
) -> Result<Vec<u8>, MediaError> {
	let type_for = |compressor| section_type_for(kind, compressor).ok_or_else(|| fixture_error("not a HAP kind"));
	match compression {
		FixtureCompression::Uncompressed => Ok(section(type_for(0x0A)?, blocks, extended)),
		FixtureCompression::Snappy => Ok(section(type_for(0x0B)?, &snappy(blocks)?, extended)),
		FixtureCompression::Chunked {
			chunks,
			mixed,
			offset_table,
		} => {
			let chunk_count = chunks.max(1);
			let chunk_len = blocks.len().div_ceil(chunk_count).max(1);
			let mut compressors = Vec::new();
			let mut payloads = Vec::new();
			for (index, chunk) in blocks.chunks(chunk_len).enumerate() {
				if mixed && index % 2 == 1 {
					compressors.push(CHUNK_COMPRESSOR_NONE);
					payloads.push(chunk.to_vec());
				} else {
					compressors.push(CHUNK_COMPRESSOR_SNAPPY);
					payloads.push(snappy(chunk)?);
				}
			}
			let mut sizes = Vec::new();
			let mut offsets = Vec::new();
			let mut running = 0u32;
			for payload in &payloads {
				let size = len_u32(payload.len())?;
				sizes.extend(size.to_le_bytes());
				offsets.extend(running.to_le_bytes());
				running = running
					.checked_add(size)
					.ok_or_else(|| fixture_error("chunk overflow"))?;
			}
			let mut instructions = section(SECTION_COMPRESSOR_TABLE, &compressors, extended);
			instructions.extend(section(SECTION_CHUNK_SIZE_TABLE, &sizes, extended));
			if offset_table {
				instructions.extend(section(SECTION_CHUNK_OFFSET_TABLE, &offsets, extended));
			}
			let mut body = section(SECTION_DECODE_INSTRUCTIONS, &instructions, extended);
			for payload in &payloads {
				body.extend_from_slice(payload);
			}
			Ok(section(type_for(0x0C)?, &body, extended))
		}
	}
}

// ---------------------------------------------------------------------------
// QuickTime boxes
// ---------------------------------------------------------------------------

/// Builds a box with a 32-bit size header.
pub fn mov_box(box_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
	let size = u32::try_from(payload.len().saturating_add(8)).unwrap_or(u32::MAX);
	let mut out = Vec::with_capacity(payload.len().saturating_add(8));
	out.extend(size.to_be_bytes());
	out.extend(box_type);
	out.extend_from_slice(payload);
	out
}

fn full_box(box_type: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
	let [_, f1, f2, f3] = flags.to_be_bytes();
	let mut payload = vec![version, f1, f2, f3];
	payload.extend_from_slice(body);
	mov_box(box_type, &payload)
}

fn concat(parts: &[Vec<u8>]) -> Vec<u8> {
	parts.iter().flatten().copied().collect()
}

/// Configuration of a synthetic HAP QuickTime movie.
#[derive(Debug, Clone)]
pub struct HapMovFixture {
	/// HAP flavour of the video track.
	pub kind: ClipKind,
	/// Frame width in pixels.
	pub width: u32,
	/// Frame height in pixels.
	pub height: u32,
	/// Number of video frames.
	pub frame_count: u32,
	/// Media timescale written to `mdhd`.
	pub timescale: u32,
	/// Duration of every sample in timescale units.
	pub sample_duration: u32,
	/// Frame compression.
	pub compression: FixtureCompression,
	/// Use 8-byte HAP section headers everywhere.
	pub extended_section_header: bool,
	/// Write chunk offsets as `co64` instead of `stco`.
	pub use_co64: bool,
	/// `mdhd` version (0 or 1).
	pub mdhd_version: u8,
	/// Add a dummy `soun` track before the video track.
	pub include_audio_track: bool,
	/// Leave out the video track entirely.
	pub omit_video_track: bool,
	/// Place `moov` before `mdat`.
	pub moov_first: bool,
	/// Write `mdat` with size 0 (extends to end of file); implies `moov_first`.
	pub mdat_size_zero: bool,
	/// Write `mdat` with a 64-bit extended size header.
	pub extended_mdat_header: bool,
	/// Samples per `stsc` chunk (at least 1).
	pub samples_per_chunk: u32,
	/// Replace the `stsd` FourCC, e.g. with `avc1`.
	pub codec_override: Option<[u8; 4]>,
	/// Replace the dimensions written into `stsd`.
	pub declared_size_override: Option<(u32, u32)>,
	/// Value added to every video chunk offset (to fake out-of-bounds tables).
	pub chunk_offset_bias: u64,
	/// Override the `first_chunk` value of the first `stsc` entry (default `1`).
	pub stsc_first_chunk: Option<u32>,
	/// Leave the `stts` box out of the video sample table entirely.
	pub omit_stts: bool,
	/// Override every `stsz` sample size with this value.
	pub sample_size_override: Option<u32>,
	/// Insert a sample description with this FourCC before the HAP one, making
	/// HAP the second `stsd` entry (and the default `stsc` description index 2).
	pub leading_stsd_entry: Option<[u8; 4]>,
	/// Override the `stsc` sample description index of every chunk run.
	pub stsc_description_index: Option<u32>,
}

impl HapMovFixture {
	/// A Snappy-compressed 30 fps clip with sensible defaults.
	pub fn new(kind: ClipKind, width: u32, height: u32, frame_count: u32) -> Self {
		Self {
			kind,
			width,
			height,
			frame_count,
			timescale: 30_000,
			sample_duration: 1_000,
			compression: FixtureCompression::Snappy,
			extended_section_header: false,
			use_co64: false,
			mdhd_version: 0,
			include_audio_track: false,
			omit_video_track: false,
			moov_first: false,
			mdat_size_zero: false,
			extended_mdat_header: false,
			samples_per_chunk: 1,
			codec_override: None,
			declared_size_override: None,
			chunk_offset_bias: 0,
			stsc_first_chunk: None,
			omit_stts: false,
			sample_size_override: None,
			leading_stsd_entry: None,
			stsc_description_index: None,
		}
	}

	/// FourCC written into the sample description.
	pub fn fourcc(&self) -> [u8; 4] {
		if let Some(code) = self.codec_override {
			return code;
		}
		match self.kind {
			ClipKind::Hap => *b"Hap1",
			ClipKind::HapAlpha => *b"Hap5",
			ClipKind::HapQ | ClipKind::ImageRgba => *b"HapY",
		}
	}

	/// The raw BC payload of frame `index`.
	pub fn frame_payload(&self, index: u32) -> Result<Vec<u8>, MediaError> {
		fixture_frame_payload(self.kind, self.width, self.height, index)
	}

	/// Every encoded HAP sample in order.
	pub fn encoded_frames(&self) -> Result<Vec<Vec<u8>>, MediaError> {
		(0..self.frame_count)
			.map(|index| {
				let raw = self.frame_payload(index)?;
				encode_hap_frame(self.kind, &raw, self.compression, self.extended_section_header)
			})
			.collect()
	}

	/// Serializes the movie into memory.
	pub fn build(&self) -> Result<Vec<u8>, MediaError> {
		build_hap_mov_bytes(self)
	}

	/// Writes the movie to `path`.
	pub fn write_to(&self, path: &Path) -> Result<(), MediaError> {
		let bytes = self.build()?;
		std::fs::write(path, bytes).map_err(|source| MediaError::Io {
			path: path.display().to_string(),
			source,
		})
	}
}

/// Audio sample bytes stored after the video frames.
const AUDIO_SAMPLE: [u8; 16] = [0x11; 16];

/// Serializes `fixture` into a complete QuickTime movie.
pub fn build_hap_mov_bytes(fixture: &HapMovFixture) -> Result<Vec<u8>, MediaError> {
	let frames = fixture.encoded_frames()?;
	let mut mdat_payload: Vec<u8> = frames.iter().flatten().copied().collect();
	let audio_offset_in_mdat = mdat_payload.len();
	if fixture.include_audio_track {
		mdat_payload.extend_from_slice(&AUDIO_SAMPLE);
	}

	let ftyp = mov_box(b"ftyp", b"qt  \x20\x05\x03\x00qt  ");
	let moov_first = fixture.moov_first || fixture.mdat_size_zero;
	let mdat_header_len: u64 = if fixture.extended_mdat_header { 16 } else { 8 };
	let mdat_header = |payload_len: usize| -> Result<Vec<u8>, MediaError> {
		let payload_len = u64::try_from(payload_len).map_err(|_| fixture_error("mdat too large"))?;
		let total = payload_len.saturating_add(mdat_header_len);
		let mut header = Vec::new();
		if fixture.mdat_size_zero {
			header.extend(0u32.to_be_bytes());
			header.extend(b"mdat");
		} else if fixture.extended_mdat_header {
			header.extend(1u32.to_be_bytes());
			header.extend(b"mdat");
			header.extend(total.to_be_bytes());
		} else {
			header.extend(
				u32::try_from(total)
					.map_err(|_| fixture_error("mdat too large"))?
					.to_be_bytes(),
			);
			header.extend(b"mdat");
		}
		Ok(header)
	};
	let mdat = concat(&[mdat_header(mdat_payload.len())?, mdat_payload]);

	let ftyp_len = u64::try_from(ftyp.len()).map_err(|_| fixture_error("ftyp too large"))?;
	let payload_start = |moov_len: u64| {
		if moov_first {
			ftyp_len + moov_len + mdat_header_len
		} else {
			ftyp_len + mdat_header_len
		}
	};

	// Box sizes do not depend on offset values, so measure first, then rebuild.
	let probe = build_moov(fixture, &frames, payload_start(0), audio_offset_in_mdat)?;
	let probe_len = u64::try_from(probe.len()).map_err(|_| fixture_error("moov too large"))?;
	let moov = build_moov(fixture, &frames, payload_start(probe_len), audio_offset_in_mdat)?;

	Ok(if moov_first {
		concat(&[ftyp, moov, mdat])
	} else {
		concat(&[ftyp, mdat, moov])
	})
}

fn build_moov(
	fixture: &HapMovFixture,
	frames: &[Vec<u8>],
	payload_start: u64,
	audio_offset_in_mdat: usize,
) -> Result<Vec<u8>, MediaError> {
	let mut mvhd_body = vec![0u8; 96];
	if let Some(rate) = mvhd_body.get_mut(12..16) {
		rate.copy_from_slice(&0x0001_0000u32.to_be_bytes());
	}
	let mut parts = vec![full_box(b"mvhd", 0, 0, &mvhd_body)];

	if fixture.include_audio_track {
		let offset = payload_start
			.checked_add(u64::try_from(audio_offset_in_mdat).map_err(|_| fixture_error("offset overflow"))?)
			.ok_or_else(|| fixture_error("offset overflow"))?;
		parts.push(audio_trak(fixture, offset)?);
	}
	if !fixture.omit_video_track {
		parts.push(video_trak(fixture, frames, payload_start)?);
	}
	Ok(mov_box(b"moov", &concat(&parts)))
}

fn hdlr(subtype: &[u8; 4]) -> Vec<u8> {
	let mut body = Vec::new();
	body.extend(b"mhlr");
	body.extend(subtype);
	body.extend([0u8; 12]);
	body.push(0);
	full_box(b"hdlr", 0, 0, &body)
}

fn mdhd(fixture: &HapMovFixture, duration: u64) -> Vec<u8> {
	let mut body = Vec::new();
	if fixture.mdhd_version == 1 {
		body.extend([0u8; 16]);
		body.extend(fixture.timescale.to_be_bytes());
		body.extend(duration.to_be_bytes());
	} else {
		body.extend([0u8; 8]);
		body.extend(fixture.timescale.to_be_bytes());
		body.extend(u32::try_from(duration).unwrap_or(u32::MAX).to_be_bytes());
	}
	body.extend([0u8; 4]);
	full_box(b"mdhd", fixture.mdhd_version, 0, &body)
}

fn tkhd(track_id: u32, width: u32, height: u32) -> Vec<u8> {
	let mut body = vec![0u8; 80];
	if let Some(id) = body.get_mut(8..12) {
		id.copy_from_slice(&track_id.to_be_bytes());
	}
	if let Some(w) = body.get_mut(72..76) {
		w.copy_from_slice(&(width << 16).to_be_bytes());
	}
	if let Some(h) = body.get_mut(76..80) {
		h.copy_from_slice(&(height << 16).to_be_bytes());
	}
	full_box(b"tkhd", 0, 0x0F, &body)
}

fn dinf() -> Vec<u8> {
	let url = full_box(b"url ", 0, 1, &[]);
	let mut dref_body = 1u32.to_be_bytes().to_vec();
	dref_body.extend(url);
	mov_box(b"dinf", &full_box(b"dref", 0, 0, &dref_body))
}

fn stsd(entry_type: &[u8; 4], entry_body: &[u8]) -> Vec<u8> {
	stsd_entries(&[(*entry_type, entry_body)])
}

fn stsd_entries(entries: &[([u8; 4], &[u8])]) -> Vec<u8> {
	let mut body = u32::try_from(entries.len()).unwrap_or(0).to_be_bytes().to_vec();
	for (entry_type, entry_body) in entries {
		body.extend(mov_box(entry_type, entry_body));
	}
	full_box(b"stsd", 0, 0, &body)
}

fn stts(count: u32, delta: u32) -> Vec<u8> {
	let mut body = Vec::new();
	if count == 0 {
		body.extend(0u32.to_be_bytes());
	} else {
		body.extend(1u32.to_be_bytes());
		body.extend(count.to_be_bytes());
		body.extend(delta.to_be_bytes());
	}
	full_box(b"stts", 0, 0, &body)
}

fn stsz(sizes: &[u32]) -> Result<Vec<u8>, MediaError> {
	let mut body = Vec::new();
	let count = len_u32(sizes.len())?;
	match sizes.first() {
		Some(&first) if sizes.iter().all(|&size| size == first) => {
			body.extend(first.to_be_bytes());
			body.extend(count.to_be_bytes());
		}
		_ => {
			body.extend(0u32.to_be_bytes());
			body.extend(count.to_be_bytes());
			for size in sizes {
				body.extend(size.to_be_bytes());
			}
		}
	}
	Ok(full_box(b"stsz", 0, 0, &body))
}

fn stsc(sample_count: u32, samples_per_chunk: u32, first_chunk: u32, description_index: u32) -> (Vec<u8>, u32) {
	let per_chunk = samples_per_chunk.max(1);
	let chunk_count = sample_count.div_ceil(per_chunk);
	let remainder = sample_count % per_chunk;
	let mut entries: Vec<(u32, u32)> = Vec::new();
	if chunk_count == 1 {
		entries.push((first_chunk, sample_count));
	} else if chunk_count > 1 {
		entries.push((first_chunk, per_chunk));
		if remainder != 0 {
			entries.push((chunk_count, remainder));
		}
	}
	let mut body = u32::try_from(entries.len()).unwrap_or(0).to_be_bytes().to_vec();
	for (first, per) in entries {
		body.extend(first.to_be_bytes());
		body.extend(per.to_be_bytes());
		body.extend(description_index.to_be_bytes());
	}
	(full_box(b"stsc", 0, 0, &body), chunk_count)
}

fn chunk_offset_box(offsets: &[u64], use_co64: bool) -> Result<Vec<u8>, MediaError> {
	let mut body = len_u32(offsets.len())?.to_be_bytes().to_vec();
	if use_co64 {
		for offset in offsets {
			body.extend(offset.to_be_bytes());
		}
		Ok(full_box(b"co64", 0, 0, &body))
	} else {
		for &offset in offsets {
			let offset = u32::try_from(offset).map_err(|_| fixture_error("stco offset exceeds u32"))?;
			body.extend(offset.to_be_bytes());
		}
		Ok(full_box(b"stco", 0, 0, &body))
	}
}

fn video_trak(fixture: &HapMovFixture, frames: &[Vec<u8>], payload_start: u64) -> Result<Vec<u8>, MediaError> {
	let sizes = frames
		.iter()
		.map(|frame| match fixture.sample_size_override {
			Some(size) => Ok(size),
			None => len_u32(frame.len()),
		})
		.collect::<Result<Vec<u32>, MediaError>>()?;
	let count = len_u32(sizes.len())?;
	let description_index = fixture
		.stsc_description_index
		.unwrap_or(if fixture.leading_stsd_entry.is_some() { 2 } else { 1 });
	let (stsc_box, chunk_count) = stsc(
		count,
		fixture.samples_per_chunk,
		fixture.stsc_first_chunk.unwrap_or(1),
		description_index,
	);
	let per_chunk = usize::try_from(fixture.samples_per_chunk.max(1)).unwrap_or(1);
	let mut offsets = Vec::new();
	let mut running = payload_start;
	for chunk in sizes.chunks(per_chunk) {
		offsets.push(running.saturating_add(fixture.chunk_offset_bias));
		running = running.saturating_add(chunk.iter().map(|&size| u64::from(size)).sum::<u64>());
	}
	if len_u32(offsets.len())? != chunk_count {
		return Err(fixture_error("chunk count mismatch"));
	}

	let (declared_w, declared_h) = fixture
		.declared_size_override
		.unwrap_or((fixture.width, fixture.height));
	let mut entry = Vec::new();
	entry.extend([0u8; 6]);
	entry.extend(1u16.to_be_bytes());
	entry.extend([0u8; 4]); // version + revision
	entry.extend(b"MUR ");
	entry.extend([0u8; 8]); // temporal + spatial quality
	entry.extend(u16::try_from(declared_w).unwrap_or(u16::MAX).to_be_bytes());
	entry.extend(u16::try_from(declared_h).unwrap_or(u16::MAX).to_be_bytes());
	entry.extend(0x0048_0000u32.to_be_bytes());
	entry.extend(0x0048_0000u32.to_be_bytes());
	entry.extend(0u32.to_be_bytes());
	entry.extend(1u16.to_be_bytes());
	let mut name = [0u8; 32];
	let label = b"Hap";
	if let Some(slot) = name.get_mut(1..=label.len()) {
		slot.copy_from_slice(label);
	}
	name[0] = 3;
	entry.extend(name);
	entry.extend(24u16.to_be_bytes());
	entry.extend(0xFFFFu16.to_be_bytes());

	let stsd_box = match fixture.leading_stsd_entry {
		Some(leading) => stsd_entries(&[(leading, &entry), (fixture.fourcc(), &entry)]),
		None => stsd(&fixture.fourcc(), &entry),
	};
	let mut stbl_parts = vec![
		stsd_box,
		stsc_box,
		stsz(&sizes)?,
		chunk_offset_box(&offsets, fixture.use_co64)?,
	];
	if !fixture.omit_stts {
		// `stts` precedes `stsc` in the canonical layout; order is irrelevant to
		// the parser, but keep the table together for readability.
		stbl_parts.insert(1, stts(count, fixture.sample_duration));
	}
	let stbl = mov_box(b"stbl", &concat(&stbl_parts));
	let vmhd = full_box(b"vmhd", 0, 1, &[0u8; 8]);
	let minf = mov_box(b"minf", &concat(&[vmhd, hdlr(b"alis"), dinf(), stbl]));
	let duration = u64::from(count).saturating_mul(u64::from(fixture.sample_duration));
	let mdia = mov_box(b"mdia", &concat(&[mdhd(fixture, duration), hdlr(b"vide"), minf]));
	Ok(mov_box(
		b"trak",
		&concat(&[tkhd(2, fixture.width, fixture.height), mdia]),
	))
}

fn audio_trak(fixture: &HapMovFixture, offset: u64) -> Result<Vec<u8>, MediaError> {
	let mut entry = Vec::new();
	entry.extend([0u8; 6]);
	entry.extend(1u16.to_be_bytes());
	entry.extend([0u8; 8]); // version, revision, vendor
	entry.extend(2u16.to_be_bytes()); // channels
	entry.extend(16u16.to_be_bytes()); // sample size
	entry.extend([0u8; 4]); // compression id + packet size
	entry.extend((48_000u32 << 16).to_be_bytes());
	let stbl = mov_box(
		b"stbl",
		&concat(&[
			stsd(b"sowt", &entry),
			stts(1, 4),
			stsc(1, 1, 1, 1).0,
			stsz(&[len_u32(AUDIO_SAMPLE.len())?])?,
			chunk_offset_box(&[offset], fixture.use_co64)?,
		]),
	);
	let smhd = full_box(b"smhd", 0, 0, &[0u8; 4]);
	let minf = mov_box(b"minf", &concat(&[smhd, dinf(), stbl]));
	let audio_mdhd = HapMovFixture {
		timescale: 48_000,
		..fixture.clone()
	};
	let mdia = mov_box(b"mdia", &concat(&[mdhd(&audio_mdhd, 4), hdlr(b"soun"), minf]));
	Ok(mov_box(b"trak", &concat(&[tkhd(1, 0, 0), mdia])))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;
	use crate::media::mov::parse_mov_from_bytes;

	#[test]
	fn fixture_payloads_are_deterministic_and_distinct() {
		let a = fixture_frame_payload(ClipKind::Hap, 32, 32, 1).unwrap();
		let b = fixture_frame_payload(ClipKind::Hap, 32, 32, 1).unwrap();
		let c = fixture_frame_payload(ClipKind::Hap, 32, 32, 2).unwrap();
		assert_eq!(a, b);
		assert_ne!(a, c);
		assert_eq!(a.len(), 8 * 8 * 8);
		assert_eq!(fixture_frame_payload(ClipKind::HapQ, 5, 5, 0).unwrap().len(), 4 * 16);
	}

	#[test]
	fn section_header_switches_to_extended_for_large_bodies() {
		assert_eq!(section_header(0xAB, 0x0100_0000, false).len(), 8);
		assert_eq!(section_header(0xAB, 0x00FF_FFFF, false).len(), 4);
		assert_eq!(section_header(0xAB, 0, false).len(), 8);
	}

	#[test]
	fn every_fixture_variant_parses() {
		for kind in [ClipKind::Hap, ClipKind::HapAlpha, ClipKind::HapQ] {
			for per_chunk in [1, 2, 4, 10] {
				let mut fixture = HapMovFixture::new(kind, 20, 12, 7);
				fixture.samples_per_chunk = per_chunk;
				fixture.include_audio_track = per_chunk % 2 == 0;
				fixture.use_co64 = per_chunk > 2;
				fixture.compression = FixtureCompression::Chunked {
					chunks: 2,
					mixed: true,
					offset_table: per_chunk == 4,
				};
				let index = parse_mov_from_bytes(&fixture.build().unwrap()).unwrap();
				assert_eq!(index.kind, kind);
				assert_eq!(index.frame_count(), 7);
				assert!((index.fps - 30.0).abs() < 0.001);
			}
		}
	}

	#[test]
	fn zero_frame_fixture_parses_without_samples() {
		let fixture = HapMovFixture::new(ClipKind::Hap, 8, 8, 0);
		let index = parse_mov_from_bytes(&fixture.build().unwrap()).unwrap();
		assert_eq!(index.frame_count(), 0);
	}
}
