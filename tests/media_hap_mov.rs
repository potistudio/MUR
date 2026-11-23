//! End-to-end media tests for HAP movies, still images, and the four-layer streamer.
//!
//! These tests exercise the exact load/scan/stream path used by the application:
//! synthetic QuickTime movies are written to disk, validated, decoded, and then
//! played through the isolated per-layer workers. Nothing here touches the GPU,
//! so the tests also double as a regression suite for the pure-Rust decoders.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use mur::error::MediaError;
use mur::media::mov::FALLBACK_FPS;
use mur::media::test_fixtures::{FixtureCompression, HapMovFixture, TempDir};
use mur::media::{ClipBacking, LayerStreamEvent, LayerStreamer, expected_hap_bytes, load_clip, scan_media_dir};
use mur::model::{ClipId, ClipKind, DecodedFrame, LayerIndex, LayerState, PixelPayload, PlaybackMode, PlaybackState};

const L1: LayerIndex = LayerIndex::ALL[0];
const L2: LayerIndex = LayerIndex::ALL[1];
const L3: LayerIndex = LayerIndex::ALL[2];
const L4: LayerIndex = LayerIndex::ALL[3];

/// Polls `condition` until it returns `true` or the timeout elapses.
fn wait_until(mut condition: impl FnMut() -> bool) -> bool {
	let start = Instant::now();
	while start.elapsed() < Duration::from_secs(10) {
		if condition() {
			return true;
		}
		std::thread::sleep(Duration::from_millis(2));
	}
	false
}

/// Builds a synthetic HAP movie description.
fn movie(
	kind: ClipKind,
	width: u32,
	height: u32,
	frame_count: u32,
	compression: FixtureCompression,
	use_co64: bool,
	with_audio: bool,
) -> HapMovFixture {
	let mut fixture = HapMovFixture::new(kind, width, height, frame_count);
	fixture.compression = compression;
	fixture.use_co64 = use_co64;
	fixture.include_audio_track = with_audio;
	fixture
}

/// Writes a synthetic HAP movie and returns its path.
fn write_movie(dir: &Path, name: &str, fixture: HapMovFixture) -> PathBuf {
	let path = dir.join(name);
	fixture.write_to(&path).unwrap();
	path
}

/// Loads a clip and registers it with a streamer.
fn load_and_register(streamer: &LayerStreamer, dir: &Path, name: &str, id: u64, kind: ClipKind) -> PathBuf {
	let fixture = movie(
		kind,
		48,
		32,
		40,
		FixtureCompression::Chunked {
			chunks: 2,
			mixed: true,
			offset_table: false,
		},
		kind == ClipKind::HapQ,
		kind == ClipKind::HapAlpha,
	);
	let path = write_movie(dir, name, fixture);
	let (_, asset) = load_clip(ClipId(id), &path).unwrap();
	streamer.register_asset(asset);
	path
}

fn payload_bytes(frame: &DecodedFrame) -> &[u8] {
	match &frame.payload {
		PixelPayload::Bc1(bytes) | PixelPayload::Bc3(bytes) | PixelPayload::Bc3YCoCg(bytes) => bytes,
		PixelPayload::Rgba8(bytes) => bytes,
	}
}

// ---------------------------------------------------------------------------
// HAP codecs and container variants
// ---------------------------------------------------------------------------

#[test]
fn all_hap_codecs_load_across_compression_and_container_variants() {
	let dir = TempDir::new("hap_variants").unwrap();
	let compressions = [
		FixtureCompression::Uncompressed,
		FixtureCompression::Snappy,
		FixtureCompression::Chunked {
			chunks: 2,
			mixed: false,
			offset_table: false,
		},
		FixtureCompression::Chunked {
			chunks: 3,
			mixed: true,
			offset_table: true,
		},
	];

	let mut id = 1_u64;
	for kind in [ClipKind::Hap, ClipKind::HapAlpha, ClipKind::HapQ] {
		for (variant, compression) in compressions.into_iter().enumerate() {
			let name = format!("{}_{}.mov", kind.label().replace(' ', "_"), variant);
			let fixture = movie(kind, 24, 16, 5, compression, variant % 2 == 1, variant == 1);
			let path = write_movie(dir.path(), &name, fixture);
			let (info, asset) = load_clip(ClipId(id), &path).unwrap();

			assert_eq!(info.kind, kind, "{name}");
			assert_eq!((info.width, info.height), (24, 16), "{name}");
			assert_eq!(info.frame_count, 5, "{name}");
			assert!((info.fps - 30.0).abs() < 0.001, "{name}");
			assert!(info.is_ready(), "{name}");
			assert_eq!(info.name, name);
			assert_eq!(info.has_alpha, kind == ClipKind::HapAlpha, "{name}");
			if kind == ClipKind::HapAlpha {
				assert!(!info.is_opaque_margin());
			} else {
				assert!(info.is_opaque_margin());
			}
			let expected_blocks = expected_hap_bytes(kind, 24, 16).unwrap();
			assert_eq!(asset.first_frame.width, 24);
			assert_eq!(asset.first_frame.height, 16);
			let payload = payload_bytes(&asset.first_frame);
			assert_eq!(payload.len(), expected_blocks, "{name}");
			match kind {
				ClipKind::Hap => assert!(matches!(&asset.first_frame.payload, PixelPayload::Bc1(_))),
				ClipKind::HapAlpha => assert!(matches!(&asset.first_frame.payload, PixelPayload::Bc3(_))),
				ClipKind::HapQ => assert!(matches!(&asset.first_frame.payload, PixelPayload::Bc3YCoCg(_))),
				ClipKind::ImageRgba => panic!("unexpected still image"),
			}
			let fixture = HapMovFixture::new(kind, 24, 16, 5);
			assert_eq!(payload, fixture.frame_payload(0).unwrap().as_slice());
			assert!(matches!(asset.backing, ClipBacking::HapMov { .. }));
			assert!(info.thumbnail.as_ref().is_some_and(|thumb| thumb.is_valid()));

			id += 1;
		}
	}
}

#[test]
fn audio_track_is_ignored_and_co64_offsets_index_correctly() {
	let dir = TempDir::new("hap_audio_co64").unwrap();
	let path = write_movie(
		dir.path(),
		"audio_co64.mov",
		movie(ClipKind::HapAlpha, 32, 24, 7, FixtureCompression::Snappy, true, true),
	);
	let (info, asset) = load_clip(ClipId(1), &path).unwrap();
	assert_eq!(info.kind, ClipKind::HapAlpha);
	assert_eq!(info.frame_count, 7);
	let fixture = HapMovFixture::new(ClipKind::HapAlpha, 32, 24, 7);
	assert_eq!(
		payload_bytes(&asset.first_frame),
		fixture.frame_payload(0).unwrap().as_slice()
	);

	let ClipBacking::HapMov { index, .. } = &asset.backing else {
		panic!("HAP movie must stream from disk");
	};
	assert_eq!(index.samples.len(), 7);
	for sample in &index.samples {
		assert!(sample.offset + u64::from(sample.size) <= std::fs::metadata(&path).unwrap().len());
	}
}

// ---------------------------------------------------------------------------
// Still images and directory scanning
// ---------------------------------------------------------------------------

#[test]
fn still_images_report_alpha_and_scan_media_dir_filters_and_sorts() {
	let dir = TempDir::new("still_images").unwrap();

	let alpha_path = dir.path().join("alpha.png");
	let mut alpha_image = image::RgbaImage::from_pixel(8, 6, image::Rgba([40, 80, 120, 255]));
	alpha_image.put_pixel(0, 0, image::Rgba([40, 80, 120, 0]));
	alpha_image.save(&alpha_path).unwrap();
	let (alpha_info, alpha_asset) = load_clip(ClipId(1), &alpha_path).unwrap();
	assert_eq!(alpha_info.kind, ClipKind::ImageRgba);
	assert!(alpha_info.has_alpha);
	assert!(!alpha_info.is_opaque_margin());
	assert_eq!(alpha_info.frame_count, 1);
	assert_eq!((alpha_info.width, alpha_info.height), (8, 6));
	assert!(matches!(alpha_asset.backing, ClipBacking::StaticImage));
	assert!(matches!(alpha_asset.first_frame.payload, PixelPayload::Rgba8(_)));

	let opaque_path = dir.path().join("opaque.PNG");
	image::RgbaImage::from_pixel(5, 4, image::Rgba([200, 10, 30, 255]))
		.save(&opaque_path)
		.unwrap();
	let (opaque_info, _) = load_clip(ClipId(2), &opaque_path).unwrap();
	assert_eq!(opaque_info.kind, ClipKind::ImageRgba);
	assert!(!opaque_info.has_alpha);
	assert!(opaque_info.is_opaque_margin());

	let jpeg_path = dir.path().join("photo.jpeg");
	image::RgbImage::from_pixel(9, 7, image::Rgb([12, 240, 90]))
		.save(&jpeg_path)
		.unwrap();
	let (jpeg_info, jpeg_asset) = load_clip(ClipId(3), &jpeg_path).unwrap();
	assert_eq!(jpeg_info.kind, ClipKind::ImageRgba);
	assert!(!jpeg_info.has_alpha);
	assert_eq!(jpeg_asset.frame_count(), 1);
	assert_eq!((jpeg_info.width, jpeg_info.height), (9, 7));

	// Filtering: supported extensions only, files only, sorted by full path.
	write_movie(
		dir.path(),
		"b.mov",
		movie(ClipKind::Hap, 8, 8, 1, FixtureCompression::Snappy, false, false),
	);
	std::fs::write(dir.path().join("c.jpg"), b"placeholder").unwrap();
	std::fs::write(dir.path().join("notes.txt"), b"ignored").unwrap();
	std::fs::write(dir.path().join("clip.mp4"), b"ignored").unwrap();
	std::fs::create_dir(dir.path().join("sub.mov")).unwrap();
	let found: Vec<String> = scan_media_dir(dir.path())
		.iter()
		.map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
		.collect();
	assert_eq!(found, ["alpha.png", "b.mov", "c.jpg", "opaque.PNG", "photo.jpeg"]);
	assert!(scan_media_dir(&dir.path().join("missing")).is_empty());
}

// ---------------------------------------------------------------------------
// Corrupt input
// ---------------------------------------------------------------------------

#[test]
fn corrupt_truncated_and_garbage_media_is_rejected_without_panicking() {
	let dir = TempDir::new("corrupt_media").unwrap();

	let zero_mov = dir.path().join("zero.mov");
	std::fs::write(&zero_mov, b"").unwrap();
	assert!(matches!(
		load_clip(ClipId(1), &zero_mov),
		Err(MediaError::Corrupt { .. })
	));

	let garbage_mov = dir.path().join("garbage.mov");
	std::fs::write(&garbage_mov, [0xFF_u8; 96]).unwrap();
	assert!(matches!(
		load_clip(ClipId(2), &garbage_mov),
		Err(MediaError::Corrupt { .. })
	));

	let valid_path = write_movie(
		dir.path(),
		"valid.mov",
		movie(ClipKind::Hap, 16, 16, 4, FixtureCompression::Snappy, false, false),
	);
	let full = std::fs::read(&valid_path).unwrap();
	let truncated = dir.path().join("truncated.mov");
	std::fs::write(&truncated, &full[..full.len() / 3]).unwrap();
	assert!(matches!(
		load_clip(ClipId(3), &truncated),
		Err(MediaError::Corrupt { .. })
	));

	let zero_png = dir.path().join("zero.png");
	std::fs::write(&zero_png, b"").unwrap();
	assert!(matches!(
		load_clip(ClipId(4), &zero_png),
		Err(MediaError::Corrupt { .. })
	));

	let garbage_png = dir.path().join("garbage.png");
	std::fs::write(&garbage_png, [0x42_u8; 96]).unwrap();
	assert!(matches!(
		load_clip(ClipId(5), &garbage_png),
		Err(MediaError::Corrupt { .. })
	));

	let zero_jpg = dir.path().join("zero.jpg");
	std::fs::write(&zero_jpg, b"").unwrap();
	assert!(matches!(
		load_clip(ClipId(6), &zero_jpg),
		Err(MediaError::Corrupt { .. })
	));

	let garbage_jpg = dir.path().join("garbage.jpg");
	std::fs::write(&garbage_jpg, [0x5A_u8; 64]).unwrap();
	assert!(matches!(
		load_clip(ClipId(7), &garbage_jpg),
		Err(MediaError::Corrupt { .. })
	));

	let truncated_png = dir.path().join("truncated.png");
	let mut png = Vec::new();
	image::RgbaImage::from_pixel(16, 16, image::Rgba([1, 2, 3, 255]))
		.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
		.unwrap();
	std::fs::write(&truncated_png, &png[..png.len() / 2]).unwrap();
	assert!(matches!(
		load_clip(ClipId(8), &truncated_png),
		Err(MediaError::Corrupt { .. })
	));

	let unsupported = dir.path().join("clip.mp4");
	std::fs::write(&unsupported, b"not supported").unwrap();
	assert!(matches!(
		load_clip(ClipId(9), &unsupported),
		Err(MediaError::UnsupportedFormat(_))
	));

	assert!(matches!(
		load_clip(ClipId(10), &dir.path().join("missing.mov")),
		Err(MediaError::Io { .. })
	));
}

// ---------------------------------------------------------------------------
// Container edge cases
// ---------------------------------------------------------------------------

/// Extracts the human readable detail of a [`MediaError::Corrupt`].
fn corrupt_detail(error: MediaError) -> String {
	match error {
		MediaError::Corrupt { detail, .. } => detail,
		other => panic!("expected MediaError::Corrupt, got {other:?}"),
	}
}

#[test]
fn zero_sample_movie_is_rejected_with_a_clear_error() {
	let dir = TempDir::new("zero_samples").unwrap();
	let path = write_movie(
		dir.path(),
		"zero.mov",
		movie(ClipKind::Hap, 32, 32, 0, FixtureCompression::Snappy, false, false),
	);
	let error = load_clip(ClipId(1), &path).unwrap_err();
	assert!(matches!(error, MediaError::Corrupt { .. }), "{error:?}");
	assert!(
		corrupt_detail(error).contains("video track contains no samples"),
		"a zero-sample movie must say why it was rejected"
	);
}

#[test]
fn single_frame_movies_play_in_loop_and_one_shot() {
	let dir = TempDir::new("single_frame").unwrap();
	let path = write_movie(
		dir.path(),
		"single.mov",
		movie(ClipKind::HapAlpha, 32, 24, 1, FixtureCompression::Snappy, false, false),
	);
	let (info, asset) = load_clip(ClipId(1), &path).unwrap();
	assert_eq!(info.frame_count, 1);
	assert_eq!(info.kind, ClipKind::HapAlpha);
	let fixture = HapMovFixture::new(ClipKind::HapAlpha, 32, 24, 1);
	assert_eq!(
		payload_bytes(&asset.first_frame),
		fixture.frame_payload(0).unwrap().as_slice()
	);

	let streamer = LayerStreamer::new().unwrap();
	streamer.register_asset(asset);

	for mode in [PlaybackMode::Loop, PlaybackMode::OneShot] {
		let mut layer = LayerState::default();
		layer.trigger(ClipId(1), 1, info.fps);
		layer.playback_mode = mode;
		// A one-frame clip stays on frame 0 and never stops on its own.
		for _ in 0..3 {
			layer.advance_playhead(0.5);
			assert_eq!(layer.playhead_frame, 0.0, "{mode:?}");
			assert_eq!(layer.current_frame_index(), 0, "{mode:?}");
		}
		assert_eq!(layer.playback_state, PlaybackState::Playing, "{mode:?}");

		let desired = layer.current_frame_index();
		assert!(streamer.request_layer_state(
			L1,
			ClipId(1),
			desired,
			layer.speed,
			mode,
			layer.playback_state.is_playing(),
		));
		let frame = streamer.poll_frame(L1, ClipId(1), desired).unwrap();
		assert_eq!(frame.frame_index, 0, "{mode:?}");
		assert_eq!((frame.width, frame.height), (32, 24), "{mode:?}");
	}
	assert!(streamer.drain_events().is_empty(), "a one-frame clip must never fail");
}

#[test]
fn missing_or_absurd_sample_timing_falls_back_to_sixty_fps() {
	let dir = TempDir::new("fallback_fps").unwrap();
	let mut omitted = movie(ClipKind::Hap, 24, 16, 4, FixtureCompression::Snappy, false, false);
	omitted.omit_stts = true;
	let mut slow = movie(ClipKind::Hap, 24, 16, 4, FixtureCompression::Snappy, false, false);
	slow.timescale = 1;
	slow.sample_duration = 100_000;

	for (name, fixture) in [("omitted_stts.mov", omitted), ("slow_motion.mov", slow)] {
		let path = write_movie(dir.path(), name, fixture);
		let (info, asset) = load_clip(ClipId(1), &path).unwrap();
		assert_eq!(info.frame_count, 4, "{name}");
		assert!(
			(info.fps - FALLBACK_FPS).abs() < 1e-6,
			"`{name}` must fall back to {FALLBACK_FPS} fps, got {}",
			info.fps
		);

		let streamer = LayerStreamer::new().unwrap();
		streamer.register_asset(asset);
		assert!(streamer.request_layer_state(L1, ClipId(1), 2, 1.0, PlaybackMode::Loop, true));
		assert!(
			wait_until(|| streamer
				.poll_frame(L1, ClipId(1), 2)
				.is_some_and(|frame| frame.frame_index == 2)),
			"`{name}` never played frame 2"
		);
		assert!(streamer.drain_events().is_empty(), "`{name}` reported errors");
	}
}

#[test]
fn corrupt_sample_tables_are_rejected() {
	let dir = TempDir::new("corrupt_tables").unwrap();

	let mut misplaced = movie(ClipKind::Hap, 16, 16, 4, FixtureCompression::Snappy, false, false);
	misplaced.stsc_first_chunk = Some(2);
	let path = write_movie(dir.path(), "stsc_gap.mov", misplaced);
	let error = load_clip(ClipId(1), &path).unwrap_err();
	assert!(matches!(error, MediaError::Corrupt { .. }), "{error:?}");
	let detail = corrupt_detail(error);
	assert!(detail.contains("chunk 1"), "{detail}");

	let mut oversized = movie(ClipKind::Hap, 16, 16, 4, FixtureCompression::Snappy, false, false);
	oversized.sample_size_override = Some(300 * 1024 * 1024);
	let path = write_movie(dir.path(), "oversized.mov", oversized);
	let error = load_clip(ClipId(2), &path).unwrap_err();
	assert!(matches!(error, MediaError::Corrupt { .. }), "{error:?}");
	let detail = corrupt_detail(error);
	assert!(detail.contains("exceeds the limit"), "{detail}");
}

#[test]
fn clear_layer_stops_serving_frames_immediately() {
	let dir = TempDir::new("clear_layer").unwrap();
	let streamer = LayerStreamer::new().unwrap();
	load_and_register(&streamer, dir.path(), "clear.mov", 1, ClipKind::Hap);
	assert!(streamer.request_layer_state(L1, ClipId(1), 3, 1.0, PlaybackMode::Loop, true));
	assert!(
		wait_until(|| streamer
			.poll_frame(L1, ClipId(1), 3)
			.is_some_and(|frame| frame.frame_index == 3)),
		"frame 3 was never decoded"
	);

	let before = streamer.layer_generation(L1).unwrap();
	streamer.clear_layer(L1);
	// No sleep and no polling window: clearing bumps the generation and drops the
	// ring synchronously, so stale frames can never leak onto the live output.
	assert!(streamer.layer_generation(L1).unwrap() > before);
	assert!(
		streamer.poll_frame(L1, ClipId(1), 3).is_none(),
		"a cleared layer must not serve buffered frames"
	);
	assert!(streamer.poll_frame(L1, ClipId(1), 4).is_none());
	assert!(streamer.buffered_frames(L1).is_empty());
}

// ---------------------------------------------------------------------------
// Four-layer streamer fault isolation
// ---------------------------------------------------------------------------

#[test]
fn four_layer_streamer_isolates_a_truncated_layer() {
	let dir = TempDir::new("stream_fault").unwrap();
	let streamer = LayerStreamer::new().unwrap();
	let mut paths = Vec::with_capacity(4);
	for (offset, layer) in LayerIndex::ALL.iter().enumerate() {
		let id = offset as u64 + 1;
		let path = load_and_register(&streamer, dir.path(), &format!("layer{id}.mov"), id, ClipKind::Hap);
		paths.push(path);
		assert!(streamer.request_layer_state(*layer, ClipId(id), 1, 1.0, PlaybackMode::Loop, true));
	}
	for (offset, layer) in LayerIndex::ALL.iter().enumerate() {
		let id = ClipId(offset as u64 + 1);
		assert!(
			wait_until(|| streamer
				.poll_frame(*layer, id, 1)
				.is_some_and(|frame| frame.frame_index == 1)),
			"layer {layer} never delivered frame 1"
		);
	}

	// Truncate layer 2's file mid-playback. The worker holds the file open, so
	// shortening the underlying inode makes the next unfetched read fail.
	let victim = paths.get(1).unwrap();
	std::fs::OpenOptions::new()
		.write(true)
		.open(victim)
		.unwrap()
		.set_len(64)
		.unwrap();

	for desired in 2..25 {
		for (offset, layer) in LayerIndex::ALL.iter().enumerate() {
			let id = offset as u64 + 1;
			streamer.request_layer_state(*layer, ClipId(id), desired, 1.0, PlaybackMode::Loop, true);
		}
		std::thread::sleep(Duration::from_millis(1));
	}

	let mut events = Vec::new();
	assert!(
		wait_until(|| {
			events.extend(streamer.drain_events());
			events
				.iter()
				.any(|event| matches!(event, LayerStreamEvent::Error { layer, clip_id: ClipId(2), .. } if *layer == L2))
		}),
		"layer 2 never reported its I/O failure"
	);
	for event in &events {
		if let LayerStreamEvent::Error { layer, .. } = event {
			assert_eq!(*layer, L2, "only layer 2 may fail: {events:?}");
		}
	}
	assert!(
		events
			.iter()
			.any(|event| matches!(event, LayerStreamEvent::Error { message, .. } if message.contains("layer2.mov")))
	);
	assert_eq!(streamer.worker_restarts(L1), 0);
	assert_eq!(streamer.worker_restarts(L2), 0);
	assert_eq!(streamer.worker_restarts(L3), 0);
	assert_eq!(streamer.worker_restarts(L4), 0);

	// The surviving layers must keep decoding new frames after the failure.
	for (offset, layer) in [L1, L3, L4].into_iter().enumerate() {
		let id = ClipId([1_u64, 3, 4][offset]);
		streamer.request_layer_state(layer, id, 33, 1.0, PlaybackMode::Loop, true);
		assert!(
			wait_until(|| streamer
				.poll_frame(layer, id, 33)
				.is_some_and(|frame| frame.frame_index == 33)),
			"layer {layer} stalled after layer 2 failed"
		);
	}
	let late = streamer.drain_events();
	assert!(
		late.iter().all(|event| !matches!(
			event,
			LayerStreamEvent::Error { layer, .. } if *layer != L2
		)),
		"unexpected fault on a healthy layer: {late:?}"
	);
}

// ---------------------------------------------------------------------------
// Playback sequencing with the streamer
// ---------------------------------------------------------------------------

#[test]
fn playback_sequencing_loop_oneshot_and_retrigger_match_the_streamer() {
	let dir = TempDir::new("playback_sequence").unwrap();
	let streamer = LayerStreamer::new().unwrap();
	let path = write_movie(
		dir.path(),
		"sequence.mov",
		movie(ClipKind::Hap, 32, 32, 12, FixtureCompression::Snappy, false, false),
	);
	let (_, asset) = load_clip(ClipId(1), &path).unwrap();
	streamer.register_asset(asset);

	// 2 fps with dt = 0.5 advances exactly one frame per tick.
	let mut layer = LayerState::default();
	layer.trigger(ClipId(1), 12, 2.0);
	layer.playback_mode = PlaybackMode::Loop;
	assert_eq!(layer.playback_state, PlaybackState::Playing);

	let request_and_wait = |layer: &LayerState, streamer: &LayerStreamer| {
		let desired = layer.current_frame_index();
		assert!(streamer.request_layer_state(
			L1,
			ClipId(1),
			desired,
			layer.speed,
			layer.playback_mode,
			layer.playback_state.is_playing(),
		));
		if desired == 0 {
			assert!(streamer.poll_frame(L1, ClipId(1), 0).is_some());
		} else {
			assert!(
				wait_until(|| streamer
					.poll_frame(L1, ClipId(1), desired)
					.is_some_and(|frame| frame.frame_index == desired)),
				"frame {desired} was never decoded"
			);
		}
	};

	// Forward loop: 1..=11 then wrap to 0.
	let mut expected = Vec::new();
	for _ in 0..14 {
		layer.advance_playhead(0.5);
		expected.push(layer.current_frame_index());
		request_and_wait(&layer, &streamer);
	}
	assert_eq!(expected, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2]);
	assert_eq!(layer.playback_state, PlaybackState::Playing);

	// Reverse loop wraps 0 -> 11.
	layer.speed = -1.0;
	layer.playhead_frame = 0.0;
	layer.playback_state = PlaybackState::Playing;
	layer.advance_playhead(0.5);
	assert_eq!(layer.current_frame_index(), 11);
	request_and_wait(&layer, &streamer);
	layer.advance_playhead(0.5);
	assert_eq!(layer.current_frame_index(), 10);
	request_and_wait(&layer, &streamer);

	// One-shot forward stops cleanly on frame 0 at the clip end.
	layer.speed = 1.0;
	layer.playback_mode = PlaybackMode::OneShot;
	layer.playhead_frame = 11.0;
	layer.advance_playhead(0.5);
	assert_eq!(layer.playback_state, PlaybackState::Stopped);
	assert_eq!(layer.playhead_frame, 0.0);
	assert!(streamer.request_layer_state(L1, ClipId(1), 0, layer.speed, layer.playback_mode, false));
	assert!(streamer.poll_frame(L1, ClipId(1), 0).is_some());

	// One-shot reverse stops cleanly on frame 0 at the clip start.
	layer.playback_state = PlaybackState::Playing;
	layer.playback_mode = PlaybackMode::OneShot;
	layer.speed = -1.0;
	layer.playhead_frame = 0.0;
	layer.advance_playhead(0.5);
	assert_eq!(layer.playback_state, PlaybackState::Stopped);
	assert_eq!(layer.playhead_frame, 0.0);

	// Retrigger jumps to frame 0 when moving forward and to the last frame in reverse.
	layer.playback_state = PlaybackState::Playing;
	layer.speed = 1.0;
	layer.playhead_frame = 7.0;
	layer.retrigger();
	assert_eq!(layer.playhead_frame, 0.0);
	request_and_wait(&layer, &streamer);

	layer.speed = -1.0;
	layer.playhead_frame = 3.0;
	layer.retrigger();
	assert_eq!(layer.playhead_frame, 11.0);
	assert_eq!(layer.current_frame_index(), 11);
	request_and_wait(&layer, &streamer);
}

// ---------------------------------------------------------------------------
// Misc invariants
// ---------------------------------------------------------------------------

#[test]
fn static_image_assets_never_need_a_decode_worker() {
	let dir = TempDir::new("static_asset").unwrap();
	let path = dir.path().join("still.png");
	image::RgbaImage::from_pixel(4, 4, image::Rgba([7, 8, 9, 255]))
		.save(&path)
		.unwrap();
	let (info, asset) = load_clip(ClipId(9), &path).unwrap();
	assert_eq!(info.frame_count, 1);
	assert!(matches!(asset.backing, ClipBacking::StaticImage));

	let streamer = LayerStreamer::new().unwrap();
	let resident = Arc::clone(&asset.first_frame);
	streamer.register_asset(asset);
	assert!(streamer.request_layer_state(L1, ClipId(9), 0, 1.0, PlaybackMode::Loop, true));
	let served = streamer.poll_frame(L1, ClipId(9), 5).unwrap();
	assert!(Arc::ptr_eq(&resident, &served));
}
