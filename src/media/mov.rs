//! Pure-Rust QuickTime `.mov` / ISO-BMFF box parser for HAP video tracks.
//!
//! Only the box structure is read: top-level boxes are skipped with `Seek`, and
//! only the `moov` box is loaded into memory. `mdat` payloads stay on disk and
//! are fetched sample by sample through [`read_mov_sample`].

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::MediaError;
use crate::model::ClipKind;

use super::dimensions_are_valid;

/// Maximum number of samples accepted in a single video track.
pub const MAX_SAMPLE_COUNT: usize = 10_000_000;

/// Maximum size of a single sample in bytes (HAP frames of 8192² BC3 fit easily).
pub const MAX_SAMPLE_BYTES: u32 = 256 * 1024 * 1024;

/// Maximum size of the `moov` box loaded into memory.
pub const MAX_MOOV_BYTES: u64 = 512 * 1024 * 1024;

/// Frame rate used when `stts`/`mdhd` do not yield a usable value.
pub const FALLBACK_FPS: f32 = 60.0;

/// Maximum number of top-level boxes scanned before giving up.
const MAX_TOP_LEVEL_BOXES: usize = 100_000;

/// Location of one encoded video frame inside the movie file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MovSample {
	/// Absolute byte offset of the sample in the file.
	pub offset: u64,
	/// Sample size in bytes.
	pub size: u32,
}

/// Parsed description of the first supported HAP video track of a movie.
#[derive(Debug, Clone, PartialEq)]
pub struct MovIndex {
	/// HAP flavour of the track.
	pub kind: ClipKind,
	/// Frame width in pixels.
	pub width: u32,
	/// Frame height in pixels.
	pub height: u32,
	/// Frame rate derived from `mdhd` timescale and `stts` durations.
	pub fps: f32,
	/// Every sample of the track in decode order.
	pub samples: Vec<MovSample>,
}

impl MovIndex {
	/// Number of frames in the track.
	pub fn frame_count(&self) -> usize {
		self.samples.len()
	}
}

/// Internal parse failure, converted into [`MediaError`] with the file path attached.
#[derive(Debug)]
enum ParseError {
	Corrupt(String),
	Unsupported(String),
}

type ParseResult<T> = Result<T, ParseError>;

fn corrupt<T>(detail: impl Into<String>) -> ParseResult<T> {
	Err(ParseError::Corrupt(detail.into()))
}

fn to_media_error(path: &Path, error: ParseError) -> MediaError {
	match error {
		ParseError::Corrupt(detail) => MediaError::Corrupt {
			path: path.display().to_string(),
			detail,
		},
		ParseError::Unsupported(detail) => MediaError::UnsupportedFormat(format!("{detail} in `{}`", path.display())),
	}
}

fn io_error(path: &Path, source: std::io::Error) -> MediaError {
	MediaError::Io {
		path: path.display().to_string(),
		source,
	}
}

/// Parses the box structure of the movie at `path` and indexes its HAP video track.
///
/// Audio, timecode, and other non-video tracks are ignored. Non-HAP video codecs
/// and HAP variants outside the 11/28 scope (`HapM`, `HapA`) are rejected with
/// [`MediaError::UnsupportedFormat`]. Every sample is bounds-checked against the
/// file length.
pub fn parse_mov_from_file(path: &Path) -> Result<MovIndex, MediaError> {
	let file = File::open(path).map_err(|source| io_error(path, source))?;
	let file_len = file.metadata().map_err(|source| io_error(path, source))?.len();
	let mut reader = BufReader::new(file);
	let moov = read_moov(&mut reader, file_len).map_err(|error| match error {
		ReadMoovError::Io(source) => io_error(path, source),
		ReadMoovError::Parse(parse) => to_media_error(path, parse),
	})?;
	parse_moov(&moov, file_len).map_err(|error| to_media_error(path, error))
}

/// Parses an in-memory movie, used by tests and fixture validation.
pub fn parse_mov_from_bytes(bytes: &[u8]) -> Result<MovIndex, MediaError> {
	let path = Path::new("<memory>");
	let file_len =
		u64::try_from(bytes.len()).map_err(|_| to_media_error(path, ParseError::Corrupt("too large".into())))?;
	let mut cursor = std::io::Cursor::new(bytes);
	let moov = read_moov(&mut cursor, file_len).map_err(|error| match error {
		ReadMoovError::Io(source) => io_error(path, source),
		ReadMoovError::Parse(parse) => to_media_error(path, parse),
	})?;
	parse_moov(&moov, file_len).map_err(|error| to_media_error(path, error))
}

/// Reads the bytes of `sample` from an open movie file into a fresh buffer.
pub fn read_mov_sample<R: Read + Seek>(reader: &mut R, sample: &MovSample) -> Result<Vec<u8>, MediaError> {
	let mut buffer = Vec::new();
	read_mov_sample_into(reader, sample, &mut buffer)?;
	Ok(buffer)
}

/// Reads the bytes of `sample` into `buffer`, reusing its allocation.
///
/// A file that was truncated or replaced after indexing produces
/// [`MediaError::Io`] (typically `UnexpectedEof`) instead of a panic.
pub fn read_mov_sample_into<R: Read + Seek>(
	reader: &mut R,
	sample: &MovSample,
	buffer: &mut Vec<u8>,
) -> Result<(), MediaError> {
	let path = Path::new("<media sample>");
	if sample.size > MAX_SAMPLE_BYTES {
		return Err(to_media_error(
			path,
			ParseError::Corrupt(format!("sample of {} bytes exceeds the limit", sample.size)),
		));
	}
	let size = usize::try_from(sample.size)
		.map_err(|_| to_media_error(path, ParseError::Corrupt("sample size overflow".into())))?;
	reader
		.seek(SeekFrom::Start(sample.offset))
		.map_err(|source| io_error(path, source))?;
	buffer.clear();
	buffer.resize(size, 0);
	reader.read_exact(buffer).map_err(|source| io_error(path, source))?;
	Ok(())
}

// ---------------------------------------------------------------------------
// Top-level scan
// ---------------------------------------------------------------------------

enum ReadMoovError {
	Io(std::io::Error),
	Parse(ParseError),
}

impl From<std::io::Error> for ReadMoovError {
	fn from(error: std::io::Error) -> Self {
		Self::Io(error)
	}
}

/// Walks the top-level boxes with `Seek` and returns the payload of `moov`.
fn read_moov<R: Read + Seek>(reader: &mut R, file_len: u64) -> Result<Vec<u8>, ReadMoovError> {
	let mut position = 0u64;
	for _ in 0..MAX_TOP_LEVEL_BOXES {
		let remaining = file_len.saturating_sub(position);
		if remaining < 8 {
			break;
		}
		reader.seek(SeekFrom::Start(position))?;
		let mut header = [0u8; 8];
		reader.read_exact(&mut header)?;
		let [s0, s1, s2, s3, t0, t1, t2, t3] = header;
		let size32 = u32::from_be_bytes([s0, s1, s2, s3]);
		let box_type = [t0, t1, t2, t3];
		let (box_size, header_len) = match size32 {
			0 => (remaining, 8u64),
			1 => {
				if remaining < 16 {
					return Err(ReadMoovError::Parse(ParseError::Corrupt(
						"truncated 64-bit box header".into(),
					)));
				}
				let mut large = [0u8; 8];
				reader.read_exact(&mut large)?;
				(u64::from_be_bytes(large), 16u64)
			}
			size => (u64::from(size), 8u64),
		};
		if box_size < header_len {
			return Err(ReadMoovError::Parse(ParseError::Corrupt(format!(
				"top-level box `{}` has invalid size {box_size}",
				fourcc_str(box_type)
			))));
		}
		if box_size > remaining {
			return Err(ReadMoovError::Parse(ParseError::Corrupt(format!(
				"top-level box `{}` extends past the end of the file",
				fourcc_str(box_type)
			))));
		}
		if &box_type == b"moov" {
			let payload_len = box_size.saturating_sub(header_len);
			if payload_len > MAX_MOOV_BYTES {
				return Err(ReadMoovError::Parse(ParseError::Corrupt(
					"moov box is too large".into(),
				)));
			}
			let payload_len = usize::try_from(payload_len)
				.map_err(|_| ReadMoovError::Parse(ParseError::Corrupt("moov box is too large".into())))?;
			let mut payload = vec![0u8; payload_len];
			reader.read_exact(&mut payload)?;
			return Ok(payload);
		}
		if &box_type == b"cmov" {
			return Err(ReadMoovError::Parse(ParseError::Unsupported(
				"compressed movie headers (cmov)".into(),
			)));
		}
		position = match position.checked_add(box_size) {
			Some(next) => next,
			None => {
				return Err(ReadMoovError::Parse(ParseError::Corrupt("box offset overflow".into())));
			}
		};
	}
	Err(ReadMoovError::Parse(ParseError::Corrupt("no moov box found".into())))
}

// ---------------------------------------------------------------------------
// In-memory box iteration
// ---------------------------------------------------------------------------

/// Iterates the child boxes of an in-memory container payload.
struct BoxIter<'a> {
	data: &'a [u8],
	position: usize,
	failed: bool,
}

impl<'a> BoxIter<'a> {
	fn new(data: &'a [u8]) -> Self {
		Self {
			data,
			position: 0,
			failed: false,
		}
	}
}

impl<'a> Iterator for BoxIter<'a> {
	type Item = ParseResult<([u8; 4], &'a [u8])>;

	fn next(&mut self) -> Option<Self::Item> {
		if self.failed {
			return None;
		}
		let rest = self.data.get(self.position..)?;
		// Fewer than 8 bytes cannot hold a box; QuickTime allows a 4-byte zero terminator.
		if rest.len() < 8 {
			return None;
		}
		let result = (|| {
			let size32 = read_u32(rest, 0)?;
			let box_type = read_fourcc(rest, 4)?;
			let (box_size, header_len) = match size32 {
				0 => (rest.len(), 8usize),
				1 => {
					let large = read_u64(rest, 8)?;
					let size = usize::try_from(large).map_err(|_| ParseError::Corrupt("box size overflow".into()))?;
					(size, 16usize)
				}
				size => (
					usize::try_from(size).map_err(|_| ParseError::Corrupt("box size overflow".into()))?,
					8usize,
				),
			};
			if box_size < header_len {
				return corrupt(format!("box `{}` has invalid size {box_size}", fourcc_str(box_type)));
			}
			let payload = rest
				.get(header_len..box_size)
				.ok_or_else(|| ParseError::Corrupt(format!("box `{}` is truncated", fourcc_str(box_type))))?;
			Ok((box_type, payload, box_size))
		})();
		match result {
			Ok((box_type, payload, box_size)) => {
				self.position = self.position.saturating_add(box_size);
				Some(Ok((box_type, payload)))
			}
			Err(error) => {
				self.failed = true;
				Some(Err(error))
			}
		}
	}
}

/// Returns the payload of the first child box of type `wanted`.
fn find_child<'a>(data: &'a [u8], wanted: &[u8; 4]) -> ParseResult<Option<&'a [u8]>> {
	for item in BoxIter::new(data) {
		let (box_type, payload) = item?;
		if &box_type == wanted {
			return Ok(Some(payload));
		}
	}
	Ok(None)
}

fn require_child<'a>(data: &'a [u8], wanted: &[u8; 4], parent: &str) -> ParseResult<&'a [u8]> {
	find_child(data, wanted)?
		.ok_or_else(|| ParseError::Corrupt(format!("`{parent}` is missing `{}`", fourcc_str(*wanted))))
}

// ---------------------------------------------------------------------------
// Byte helpers
// ---------------------------------------------------------------------------

fn read_bytes<const N: usize>(data: &[u8], offset: usize) -> ParseResult<[u8; N]> {
	let end = offset
		.checked_add(N)
		.ok_or_else(|| ParseError::Corrupt("offset overflow".into()))?;
	let slice = data
		.get(offset..end)
		.ok_or_else(|| ParseError::Corrupt("unexpected end of box".into()))?;
	let mut out = [0u8; N];
	out.copy_from_slice(slice);
	Ok(out)
}

fn read_u16(data: &[u8], offset: usize) -> ParseResult<u16> {
	read_bytes::<2>(data, offset).map(u16::from_be_bytes)
}

fn read_u32(data: &[u8], offset: usize) -> ParseResult<u32> {
	read_bytes::<4>(data, offset).map(u32::from_be_bytes)
}

fn read_u64(data: &[u8], offset: usize) -> ParseResult<u64> {
	read_bytes::<8>(data, offset).map(u64::from_be_bytes)
}

fn read_fourcc(data: &[u8], offset: usize) -> ParseResult<[u8; 4]> {
	read_bytes::<4>(data, offset)
}

fn fourcc_str(code: [u8; 4]) -> String {
	code.iter()
		.map(|&byte| {
			if byte.is_ascii_graphic() || byte == b' ' {
				char::from(byte)
			} else {
				'?'
			}
		})
		.collect()
}

/// Validates a full-box table `entry_count` against the bytes available.
fn table_entries(payload: &[u8], header: usize, entry_size: usize, count: u32, name: &str) -> ParseResult<usize> {
	let count = usize::try_from(count).map_err(|_| ParseError::Corrupt(format!("{name} count overflow")))?;
	if count > MAX_SAMPLE_COUNT {
		return corrupt(format!("{name} has too many entries ({count})"));
	}
	let needed = count
		.checked_mul(entry_size)
		.and_then(|bytes| bytes.checked_add(header))
		.ok_or_else(|| ParseError::Corrupt(format!("{name} size overflow")))?;
	if payload.len() < needed {
		return corrupt(format!("{name} is truncated"));
	}
	Ok(count)
}

// ---------------------------------------------------------------------------
// moov / trak parsing
// ---------------------------------------------------------------------------

fn parse_moov(moov: &[u8], file_len: u64) -> ParseResult<MovIndex> {
	let mut unsupported: Option<ParseError> = None;
	let mut saw_video = false;
	for item in BoxIter::new(moov) {
		let (box_type, payload) = item?;
		if &box_type != b"trak" {
			continue;
		}
		match parse_trak(payload, file_len)? {
			TrackOutcome::Ignored => {}
			TrackOutcome::Video(index) => return Ok(index),
			TrackOutcome::UnsupportedVideo(error) => {
				saw_video = true;
				if unsupported.is_none() {
					unsupported = Some(error);
				}
			}
		}
	}
	match unsupported {
		Some(error) => Err(error),
		None if saw_video => Err(ParseError::Unsupported("unsupported video track".into())),
		None => Err(ParseError::Unsupported("no video track".into())),
	}
}

enum TrackOutcome {
	Ignored,
	Video(MovIndex),
	UnsupportedVideo(ParseError),
}

fn parse_trak(trak: &[u8], file_len: u64) -> ParseResult<TrackOutcome> {
	let Some(mdia) = find_child(trak, b"mdia")? else {
		return Ok(TrackOutcome::Ignored);
	};
	let Some(hdlr) = find_child(mdia, b"hdlr")? else {
		return Ok(TrackOutcome::Ignored);
	};
	// FullBox header (4) + pre_defined / component type (4) + handler subtype (4).
	let handler = read_fourcc(hdlr, 8)?;
	if &handler != b"vide" {
		return Ok(TrackOutcome::Ignored);
	}

	let mdhd = require_child(mdia, b"mdhd", "mdia")?;
	let (timescale, _duration) = parse_mdhd(mdhd)?;
	let minf = require_child(mdia, b"minf", "mdia")?;
	let stbl = require_child(minf, b"stbl", "minf")?;

	let stsd = require_child(stbl, b"stsd", "stbl")?;
	let (kind, width, height) = match parse_stsd(stsd) {
		Ok(description) => description,
		Err(error @ ParseError::Unsupported(_)) => return Ok(TrackOutcome::UnsupportedVideo(error)),
		Err(error) => return Err(error),
	};

	let sample_sizes = parse_stsz(require_child(stbl, b"stsz", "stbl")?)?;
	let chunk_offsets = match find_child(stbl, b"stco")? {
		Some(stco) => parse_stco(stco)?,
		None => match find_child(stbl, b"co64")? {
			Some(co64) => parse_co64(co64)?,
			None => return corrupt("`stbl` is missing `stco`/`co64`"),
		},
	};
	let stsc = parse_stsc(require_child(stbl, b"stsc", "stbl")?)?;
	let samples = build_samples(&sample_sizes, &chunk_offsets, &stsc, file_len)?;
	let fps = match find_child(stbl, b"stts")? {
		Some(stts) => compute_fps(timescale, stts)?,
		None => FALLBACK_FPS,
	};

	Ok(TrackOutcome::Video(MovIndex {
		kind,
		width,
		height,
		fps,
		samples,
	}))
}

/// Returns `(timescale, duration)` from an `mdhd` box of version 0 or 1.
fn parse_mdhd(mdhd: &[u8]) -> ParseResult<(u32, u64)> {
	let version = read_bytes::<1>(mdhd, 0)?;
	match version {
		[0] => Ok((read_u32(mdhd, 12)?, u64::from(read_u32(mdhd, 16)?))),
		[1] => Ok((read_u32(mdhd, 20)?, read_u64(mdhd, 24)?)),
		[other] => corrupt(format!("unknown mdhd version {other}")),
	}
}

/// Maps a sample-description FourCC to the HAP flavour it encodes.
pub fn clip_kind_from_fourcc(code: [u8; 4]) -> Result<ClipKind, MediaError> {
	match &code {
		b"Hap1" => Ok(ClipKind::Hap),
		b"Hap5" => Ok(ClipKind::HapAlpha),
		b"HapY" => Ok(ClipKind::HapQ),
		b"HapM" => Err(MediaError::UnsupportedFormat(
			"HAP Q Alpha (HapM) is not supported".into(),
		)),
		b"HapA" => Err(MediaError::UnsupportedFormat(
			"HAP Alpha-only (HapA) is not supported".into(),
		)),
		_ => Err(MediaError::UnsupportedFormat(format!(
			"video codec `{}` is not HAP",
			fourcc_str(code)
		))),
	}
}

fn parse_stsd(stsd: &[u8]) -> ParseResult<(ClipKind, u32, u32)> {
	let entry_count = read_u32(stsd, 4)?;
	if entry_count == 0 {
		return corrupt("stsd has no sample descriptions");
	}
	let entries = stsd
		.get(8..)
		.ok_or_else(|| ParseError::Corrupt("stsd is truncated".into()))?;
	let Some(first) = BoxIter::new(entries).next() else {
		return corrupt("stsd has no sample descriptions");
	};
	let (format, description) = first?;
	let kind = clip_kind_from_fourcc(format).map_err(|error| match error {
		MediaError::UnsupportedFormat(detail) => ParseError::Unsupported(detail),
		other => ParseError::Corrupt(other.to_string()),
	})?;
	// reserved(6) dref(2) version(2) revision(2) vendor(4) temporal(4) spatial(4) width(2) height(2)
	let width = u32::from(read_u16(description, 24)?);
	let height = u32::from(read_u16(description, 26)?);
	if !dimensions_are_valid(width, height) {
		return corrupt(format!("invalid frame dimensions {width}x{height}"));
	}
	Ok((kind, width, height))
}

fn parse_stsz(stsz: &[u8]) -> ParseResult<Vec<u32>> {
	let uniform = read_u32(stsz, 4)?;
	let count = read_u32(stsz, 8)?;
	let count_usize = usize::try_from(count).map_err(|_| ParseError::Corrupt("stsz count overflow".into()))?;
	if count_usize > MAX_SAMPLE_COUNT {
		return corrupt(format!("stsz has too many samples ({count})"));
	}
	let sizes = if uniform > 0 {
		vec![uniform; count_usize]
	} else {
		let count = table_entries(stsz, 12, 4, count, "stsz")?;
		let table = stsz
			.get(12..)
			.ok_or_else(|| ParseError::Corrupt("stsz is truncated".into()))?;
		table
			.as_chunks::<4>()
			.0
			.iter()
			.take(count)
			.map(|bytes| read_u32(bytes, 0))
			.collect::<ParseResult<Vec<u32>>>()?
	};
	if let Some(too_big) = sizes.iter().find(|&&size| size > MAX_SAMPLE_BYTES) {
		return corrupt(format!("sample of {too_big} bytes exceeds the limit"));
	}
	Ok(sizes)
}

fn parse_stco(stco: &[u8]) -> ParseResult<Vec<u64>> {
	let count = table_entries(stco, 8, 4, read_u32(stco, 4)?, "stco")?;
	let table = stco
		.get(8..)
		.ok_or_else(|| ParseError::Corrupt("stco is truncated".into()))?;
	table
		.as_chunks::<4>()
		.0
		.iter()
		.take(count)
		.map(|bytes| read_u32(bytes, 0).map(u64::from))
		.collect()
}

fn parse_co64(co64: &[u8]) -> ParseResult<Vec<u64>> {
	let count = table_entries(co64, 8, 8, read_u32(co64, 4)?, "co64")?;
	let table = co64
		.get(8..)
		.ok_or_else(|| ParseError::Corrupt("co64 is truncated".into()))?;
	table
		.as_chunks::<8>()
		.0
		.iter()
		.take(count)
		.map(|bytes| read_u64(bytes, 0))
		.collect()
}

/// One `stsc` run: chunks starting at `first_chunk` (1-based) hold `samples_per_chunk`.
#[derive(Debug, Clone, Copy)]
struct StscEntry {
	first_chunk: u32,
	samples_per_chunk: u32,
}

fn parse_stsc(stsc: &[u8]) -> ParseResult<Vec<StscEntry>> {
	let count = table_entries(stsc, 8, 12, read_u32(stsc, 4)?, "stsc")?;
	let table = stsc
		.get(8..)
		.ok_or_else(|| ParseError::Corrupt("stsc is truncated".into()))?;
	let entries = table
		.as_chunks::<12>()
		.0
		.iter()
		.take(count)
		.map(|bytes| {
			Ok(StscEntry {
				first_chunk: read_u32(bytes, 0)?,
				samples_per_chunk: read_u32(bytes, 4)?,
			})
		})
		.collect::<ParseResult<Vec<StscEntry>>>()?;
	let mut previous = 0u32;
	for entry in &entries {
		if entry.first_chunk == 0 || entry.first_chunk <= previous {
			return corrupt("stsc chunk numbers are not strictly increasing");
		}
		previous = entry.first_chunk;
	}
	// `first_chunk` is 1-based in ISO BMFF: a gap before the first chunk would
	// shift the whole sample-to-chunk mapping, so treat it as corruption.
	if entries.first().is_some_and(|entry| entry.first_chunk != 1) {
		return corrupt("first stsc entry must start at chunk 1");
	}
	Ok(entries)
}

/// Expands chunk/sample tables into absolute sample locations.
fn build_samples(
	sizes: &[u32],
	chunk_offsets: &[u64],
	stsc: &[StscEntry],
	file_len: u64,
) -> ParseResult<Vec<MovSample>> {
	let total = sizes.len();
	let mut samples = Vec::with_capacity(total);
	let chunk_count = chunk_offsets.len();

	for (entry_index, entry) in stsc.iter().enumerate() {
		if samples.len() >= total {
			break;
		}
		let first = usize::try_from(entry.first_chunk)
			.map_err(|_| ParseError::Corrupt("stsc chunk overflow".into()))?
			.saturating_sub(1);
		let end = match stsc.get(entry_index.saturating_add(1)) {
			Some(next) => usize::try_from(next.first_chunk)
				.map_err(|_| ParseError::Corrupt("stsc chunk overflow".into()))?
				.saturating_sub(1)
				.min(chunk_count),
			None => chunk_count,
		};
		for chunk_index in first..end {
			if samples.len() >= total {
				break;
			}
			let mut offset = *chunk_offsets
				.get(chunk_index)
				.ok_or_else(|| ParseError::Corrupt("stsc references a missing chunk".into()))?;
			for _ in 0..entry.samples_per_chunk {
				let Some(&size) = sizes.get(samples.len()) else {
					break;
				};
				let end_offset = offset
					.checked_add(u64::from(size))
					.ok_or_else(|| ParseError::Corrupt("sample offset overflow".into()))?;
				if end_offset > file_len {
					return corrupt(format!(
						"sample {} at offset {offset} (+{size}) lies outside the file",
						samples.len()
					));
				}
				samples.push(MovSample { offset, size });
				offset = end_offset;
			}
		}
	}
	if samples.len() != total {
		return corrupt(format!("sample table describes {} of {total} samples", samples.len()));
	}
	Ok(samples)
}

/// Computes frames per second from the media timescale and `stts` durations.
///
/// Results outside [`MIN_CLIP_FPS`](crate::model::layer::MIN_CLIP_FPS) `..=`
/// [`MAX_CLIP_FPS`](crate::model::layer::MAX_CLIP_FPS) are replaced with
/// [`FALLBACK_FPS`], so a corrupt timescale cannot produce a playhead that jumps
/// through the clip every frame.
fn compute_fps(timescale: u32, stts: &[u8]) -> ParseResult<f32> {
	let count = table_entries(stts, 8, 8, read_u32(stts, 4)?, "stts")?;
	let table = stts
		.get(8..)
		.ok_or_else(|| ParseError::Corrupt("stts is truncated".into()))?;
	let mut total_samples = 0u64;
	let mut total_duration = 0u64;
	for bytes in table.as_chunks::<8>().0.iter().take(count) {
		let sample_count = u64::from(read_u32(bytes, 0)?);
		let delta = u64::from(read_u32(bytes, 4)?);
		total_samples = total_samples.saturating_add(sample_count);
		total_duration = total_duration.saturating_add(sample_count.saturating_mul(delta));
	}
	if timescale == 0 || total_samples == 0 || total_duration == 0 {
		return Ok(FALLBACK_FPS);
	}
	let average = total_duration as f64 / total_samples as f64;
	let fps = f64::from(timescale) / average;
	let (min, max) = (
		f64::from(crate::model::layer::MIN_CLIP_FPS),
		f64::from(crate::model::layer::MAX_CLIP_FPS),
	);
	if fps.is_finite() && (min..=max).contains(&fps) {
		Ok(fps as f32)
	} else {
		Ok(FALLBACK_FPS)
	}
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;
	use crate::media::test_fixtures::{FixtureCompression, HapMovFixture, TempDir, mov_box};

	fn parse(fixture: &HapMovFixture) -> Result<MovIndex, MediaError> {
		parse_mov_from_bytes(&fixture.build().unwrap())
	}

	#[test]
	fn parses_all_hap_fourccs() {
		for (kind, fps) in [
			(ClipKind::Hap, 30.0),
			(ClipKind::HapAlpha, 25.0),
			(ClipKind::HapQ, 60.0),
		] {
			let mut fixture = HapMovFixture::new(kind, 64, 32, 7);
			fixture.timescale = 600;
			fixture.sample_duration = (600.0 / fps) as u32;
			let index = parse(&fixture).unwrap();
			assert_eq!(index.kind, kind);
			assert_eq!((index.width, index.height), (64, 32));
			assert_eq!(index.frame_count(), 7);
			assert!((index.fps - fps).abs() < 0.01, "{} vs {fps}", index.fps);
		}
	}

	#[test]
	fn sample_offsets_point_at_frame_data() {
		let mut fixture = HapMovFixture::new(ClipKind::Hap, 16, 16, 5);
		fixture.samples_per_chunk = 2;
		fixture.compression = FixtureCompression::Snappy;
		let bytes = fixture.build().unwrap();
		let index = parse_mov_from_bytes(&bytes).unwrap();
		let frames = fixture.encoded_frames().unwrap();
		assert_eq!(index.samples.len(), frames.len());
		for (sample, expected) in index.samples.iter().zip(&frames) {
			let start = sample.offset as usize;
			assert_eq!(&bytes[start..start + sample.size as usize], expected.as_slice());
		}
	}

	#[test]
	fn handles_co64_mdhd_v1_audio_and_moov_first() {
		let mut fixture = HapMovFixture::new(ClipKind::HapQ, 8, 8, 4);
		fixture.use_co64 = true;
		fixture.mdhd_version = 1;
		fixture.include_audio_track = true;
		fixture.moov_first = true;
		fixture.samples_per_chunk = 3;
		let index = parse(&fixture).unwrap();
		assert_eq!(index.kind, ClipKind::HapQ);
		assert_eq!(index.frame_count(), 4);
	}

	#[test]
	fn handles_extended_and_zero_sized_mdat() {
		let mut fixture = HapMovFixture::new(ClipKind::Hap, 8, 8, 3);
		fixture.extended_mdat_header = true;
		assert_eq!(parse(&fixture).unwrap().frame_count(), 3);

		let mut fixture = HapMovFixture::new(ClipKind::Hap, 8, 8, 3);
		fixture.moov_first = true;
		fixture.mdat_size_zero = true;
		assert_eq!(parse(&fixture).unwrap().frame_count(), 3);
	}

	#[test]
	fn uniform_stsz_is_supported() {
		let mut fixture = HapMovFixture::new(ClipKind::HapAlpha, 8, 8, 6);
		fixture.compression = FixtureCompression::Uncompressed;
		let bytes = fixture.build().unwrap();
		let index = parse_mov_from_bytes(&bytes).unwrap();
		assert!(index.samples.windows(2).all(|w| w[0].size == w[1].size));
		assert_eq!(index.frame_count(), 6);
	}

	#[test]
	fn rejects_unsupported_codecs() {
		for code in [*b"HapM", *b"HapA", *b"avc1", *b"hvc1", *b"apch"] {
			let mut fixture = HapMovFixture::new(ClipKind::Hap, 8, 8, 2);
			fixture.codec_override = Some(code);
			assert!(
				matches!(parse(&fixture), Err(MediaError::UnsupportedFormat(_))),
				"{code:?}"
			);
		}
	}

	#[test]
	fn rejects_audio_only_and_missing_moov() {
		let mut fixture = HapMovFixture::new(ClipKind::Hap, 8, 8, 2);
		fixture.include_audio_track = true;
		fixture.omit_video_track = true;
		assert!(matches!(parse(&fixture), Err(MediaError::UnsupportedFormat(_))));

		let only_ftyp = mov_box(b"ftyp", b"qt  \0\0\0\0qt  ");
		assert!(matches!(
			parse_mov_from_bytes(&only_ftyp),
			Err(MediaError::Corrupt { .. })
		));
		assert!(parse_mov_from_bytes(&[]).is_err());
	}

	#[test]
	fn rejects_invalid_dimensions() {
		for (w, h) in [(0, 8), (8, 0), (8193, 8), (8, 9000)] {
			let mut fixture = HapMovFixture::new(ClipKind::Hap, 8, 8, 1);
			fixture.declared_size_override = Some((w, h));
			assert!(matches!(parse(&fixture), Err(MediaError::Corrupt { .. })), "{w}x{h}");
		}
	}

	#[test]
	fn rejects_out_of_bounds_samples() {
		// A valid file keeps every sample inside the file.
		let bytes = HapMovFixture::new(ClipKind::Hap, 8, 8, 3).build().unwrap();
		let index = parse_mov_from_bytes(&bytes).unwrap();
		let last = index.samples.last().unwrap();
		assert!(last.offset + u64::from(last.size) <= bytes.len() as u64);

		// moov comes first, so truncating the file leaves the last sample past EOF.
		let mut moov_first = HapMovFixture::new(ClipKind::Hap, 8, 8, 3);
		moov_first.moov_first = true;
		let mut bytes = moov_first.build().unwrap();
		bytes.truncate(bytes.len() - 4);
		assert!(matches!(parse_mov_from_bytes(&bytes), Err(MediaError::Corrupt { .. })));

		// Chunk offsets pointing far beyond the file are rejected.
		let mut lying = HapMovFixture::new(ClipKind::Hap, 8, 8, 3);
		lying.chunk_offset_bias = 1 << 30;
		assert!(matches!(parse(&lying), Err(MediaError::Corrupt { .. })));
	}

	#[test]
	fn rejects_bad_box_sizes_without_looping() {
		// size 4 (< 8) at top level.
		let mut bytes = vec![0, 0, 0, 4];
		bytes.extend_from_slice(b"free");
		bytes.extend_from_slice(&[0; 16]);
		assert!(matches!(parse_mov_from_bytes(&bytes), Err(MediaError::Corrupt { .. })));

		// moov containing a child with size 3.
		let mut child = vec![0, 0, 0, 3];
		child.extend_from_slice(b"trak");
		let moov = mov_box(b"moov", &child);
		assert!(matches!(parse_mov_from_bytes(&moov), Err(MediaError::Corrupt { .. })));

		// box larger than file.
		let mut huge = vec![0x7F, 0xFF, 0xFF, 0xFF];
		huge.extend_from_slice(b"moov");
		assert!(matches!(parse_mov_from_bytes(&huge), Err(MediaError::Corrupt { .. })));

		// 64-bit size smaller than header.
		let mut ext = vec![0, 0, 0, 1];
		ext.extend_from_slice(b"wide");
		ext.extend_from_slice(&8u64.to_be_bytes());
		ext.extend_from_slice(&[0; 8]);
		assert!(matches!(parse_mov_from_bytes(&ext), Err(MediaError::Corrupt { .. })));
	}

	#[test]
	fn corrupt_tables_never_panic() {
		let bytes = HapMovFixture::new(ClipKind::Hap, 16, 8, 4).build().unwrap();
		// Flip each byte of the file in turn; parsing must never panic.
		for position in 0..bytes.len() {
			let mut mutated = bytes.clone();
			mutated[position] ^= 0xA5;
			let _ = parse_mov_from_bytes(&mutated);
			let mut truncated = bytes.clone();
			truncated.truncate(position);
			let _ = parse_mov_from_bytes(&truncated);
		}
	}

	#[test]
	fn fps_falls_back_when_durations_are_zero() {
		let mut fixture = HapMovFixture::new(ClipKind::Hap, 8, 8, 2);
		fixture.sample_duration = 0;
		assert_eq!(parse(&fixture).unwrap().fps, FALLBACK_FPS);
		let mut fixture = HapMovFixture::new(ClipKind::Hap, 8, 8, 2);
		fixture.timescale = 0;
		assert_eq!(parse(&fixture).unwrap().fps, FALLBACK_FPS);
	}

	#[test]
	fn fps_outside_the_clip_bounds_falls_back() {
		// timescale 1 with 100_000-unit frames means 1e-5 fps, far below the floor.
		let mut slow = HapMovFixture::new(ClipKind::Hap, 8, 8, 2);
		slow.timescale = 1;
		slow.sample_duration = 100_000;
		assert_eq!(parse(&slow).unwrap().fps, FALLBACK_FPS);

		// A huge timescale produces 10_000 fps, far above the ceiling.
		let mut fast = HapMovFixture::new(ClipKind::Hap, 8, 8, 2);
		fast.timescale = 10_000_000;
		fast.sample_duration = 1_000;
		assert_eq!(parse(&fast).unwrap().fps, FALLBACK_FPS);
	}

	#[test]
	fn missing_stts_falls_back_to_the_default_fps() {
		let mut fixture = HapMovFixture::new(ClipKind::Hap, 16, 16, 4);
		fixture.omit_stts = true;
		let index = parse(&fixture).unwrap();
		assert_eq!(index.fps, FALLBACK_FPS);
		assert_eq!(index.frame_count(), 4);
	}

	#[test]
	fn rejects_stsc_that_does_not_start_at_chunk_one() {
		let mut fixture = HapMovFixture::new(ClipKind::Hap, 8, 8, 3);
		fixture.stsc_first_chunk = Some(2);
		assert!(matches!(parse(&fixture), Err(MediaError::Corrupt { .. })));
	}

	#[test]
	fn rejects_oversized_sample_sizes() {
		let mut fixture = HapMovFixture::new(ClipKind::Hap, 8, 8, 3);
		fixture.sample_size_override = Some(300 * 1024 * 1024);
		assert!(matches!(parse(&fixture), Err(MediaError::Corrupt { .. })));
	}

	#[test]
	fn file_parsing_and_sample_reads() {
		let dir = TempDir::new("mov_file").unwrap();
		let path = dir.path().join("clip.mov");
		let fixture = HapMovFixture::new(ClipKind::Hap, 16, 16, 3);
		fixture.write_to(&path).unwrap();
		let index = parse_mov_from_file(&path).unwrap();
		let mut file = File::open(&path).unwrap();
		let frames = fixture.encoded_frames().unwrap();
		let mut scratch = Vec::new();
		for (sample, expected) in index.samples.iter().zip(&frames) {
			read_mov_sample_into(&mut file, sample, &mut scratch).unwrap();
			assert_eq!(&scratch, expected);
			assert_eq!(&read_mov_sample(&mut file, sample).unwrap(), expected);
		}
		let beyond = MovSample {
			offset: 1 << 40,
			size: 16,
		};
		assert!(matches!(
			read_mov_sample(&mut file, &beyond),
			Err(MediaError::Io { .. })
		));
		assert!(matches!(
			parse_mov_from_file(&dir.path().join("missing.mov")),
			Err(MediaError::Io { .. })
		));
	}

	#[test]
	fn fourcc_mapping() {
		assert_eq!(clip_kind_from_fourcc(*b"Hap1").unwrap(), ClipKind::Hap);
		assert_eq!(clip_kind_from_fourcc(*b"Hap5").unwrap(), ClipKind::HapAlpha);
		assert_eq!(clip_kind_from_fourcc(*b"HapY").unwrap(), ClipKind::HapQ);
		assert!(clip_kind_from_fourcc(*b"\0\x01ab").is_err());
	}
}
