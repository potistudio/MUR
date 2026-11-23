//! Headless GPU integration tests for the 1920x1080 master pipeline.
//!
//! Every test creates a real `wgpu` device (Metal on macOS), renders through the
//! production [`MasterRenderer`], and reads pixels back from the master output.
//! There is no mock: blend maths, HAP Q YCoCg conversion, aspect margins, master
//! effects, presentation letterboxing, and the sustained four-layer frame
//! budget are all verified against real GPU output.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use mur::media::test_fixtures::{
	FixtureCompression, HapMovFixture, TempDir, bc1_solid_block, bc3_solid_block, bc3_ycocg_solid_block, rgb_to_565,
};
use mur::media::{ClipAsset, decode_frame_to_rgba8, load_clip};
use mur::model::{
	BlendMode, ClipId, ClipKind, DecodedFrame, LayerIndex, LayerState, MasterEffects, PixelPayload, PlaybackMode,
};
use mur::render::texture_cache::plan_upload;
use mur::render::{GpuContext, INTERNAL_HEIGHT, INTERNAL_WIDTH, LayerRenderInput, MasterRenderer, fit_rect};

/// Serializes GPU tests so the sustained frame-budget test is not disturbed by
/// other 1080p readbacks running on the same device.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn gpu_lock() -> MutexGuard<'static, ()> {
	GPU_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Creates a headless GPU context, skipping gracefully on non-macOS machines
/// without an adapter. On macOS a Metal adapter is mandatory, so a missing GPU
/// is a hard failure instead of a silent skip.
fn gpu_context(label: &str) -> Option<GpuContext> {
	match GpuContext::new(None) {
		Ok(ctx) => {
			let info = ctx.adapter.get_info();
			eprintln!(
				"[{label}] adapter=`{}` backend={:?} bc_textures={}",
				info.name, info.backend, ctx.supports_bc
			);
			if cfg!(target_os = "macos") {
				assert_eq!(
					info.backend,
					wgpu::Backend::Metal,
					"[{label}] GPU tests must execute on Metal on macOS"
				);
			}
			Some(ctx)
		}
		Err(error) => {
			if cfg!(target_os = "macos") {
				panic!("[{label}] no GPU adapter on macOS: {error}");
			}
			eprintln!("[{label}] no GPU adapter, skipping: {error}");
			None
		}
	}
}

/// Copies an arbitrary RGBA8 texture back to tightly packed pixels.
fn read_texture_rgba8(ctx: &GpuContext, texture: &wgpu::Texture, width: u32, height: u32) -> Vec<u8> {
	let unpadded_row = width * 4;
	let padded_row = unpadded_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
	let buffer_size = u64::from(padded_row) * u64::from(height);
	let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
		label: Some("mur.test.readback"),
		size: buffer_size,
		usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
		mapped_at_creation: false,
	});
	let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
		label: Some("mur.test.readback"),
	});
	encoder.copy_texture_to_buffer(
		wgpu::TexelCopyTextureInfo {
			texture,
			mip_level: 0,
			origin: wgpu::Origin3d::ZERO,
			aspect: wgpu::TextureAspect::All,
		},
		wgpu::TexelCopyBufferInfo {
			buffer: &staging,
			layout: wgpu::TexelCopyBufferLayout {
				offset: 0,
				bytes_per_row: Some(padded_row),
				rows_per_image: Some(height),
			},
		},
		wgpu::Extent3d {
			width,
			height,
			depth_or_array_layers: 1,
		},
	);
	ctx.queue.submit(std::iter::once(encoder.finish()));

	let (sender, receiver) = mpsc::channel();
	staging.map_async(wgpu::MapMode::Read, .., move |result| {
		let _ = sender.send(result);
	});
	ctx.device
		.poll(wgpu::PollType::Wait {
			submission_index: None,
			timeout: None,
		})
		.expect("GPU poll for readback");
	receiver
		.recv()
		.expect("readback callback dropped")
		.expect("readback mapping failed");

	let mut pixels = Vec::with_capacity((unpadded_row as usize).saturating_mul(height as usize));
	{
		let mapped = staging.get_mapped_range(..).expect("mapped readback range");
		for row in mapped.chunks_exact(padded_row as usize) {
			pixels.extend_from_slice(row.get(..unpadded_row as usize).expect("readback row"));
		}
	}
	staging.unmap();
	pixels
}

fn pixel_at(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
	let index = ((y * width + x) * 4) as usize;
	[pixels[index], pixels[index + 1], pixels[index + 2], pixels[index + 3]]
}

fn master_pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
	pixel_at(pixels, INTERNAL_WIDTH, x, y)
}

fn assert_color(actual: [u8; 4], expected: [f32; 3], tolerance: f32, context: &str) {
	for (channel, value) in actual.iter().take(3).enumerate() {
		let expected_byte = expected.get(channel).copied().unwrap_or(0.0) * 255.0;
		let delta = (f32::from(*value) - expected_byte).abs();
		assert!(
			delta <= tolerance,
			"{context}: channel {channel} was {value} (expected {expected_byte:.1}, delta {delta:.1}), full pixel {actual:?}"
		);
	}
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
	assert_eq!(a.len(), b.len());
	let mut sum = 0_u64;
	let mut count = 0_u64;
	for y in (0..INTERNAL_HEIGHT).step_by(16) {
		for x in (0..INTERNAL_WIDTH).step_by(16) {
			let index = ((y * INTERNAL_WIDTH + x) * 4) as usize;
			for channel in 0..3 {
				sum += u64::from(a[index + channel].abs_diff(b[index + channel]));
				count += 1;
			}
		}
	}
	sum as f64 / count as f64
}

/// Repeats a 4x4 BC block across the whole frame.
fn solid_frame(kind: ClipKind, width: u32, height: u32, rgb: [u8; 3], alpha: u8) -> DecodedFrame {
	let block: Vec<u8> = match kind {
		ClipKind::Hap => bc1_solid_block(rgb).to_vec(),
		ClipKind::HapAlpha => bc3_solid_block([rgb[0], rgb[1], rgb[2], alpha]).to_vec(),
		ClipKind::HapQ => bc3_ycocg_solid_block(rgb).to_vec(),
		ClipKind::ImageRgba => {
			let mut rgba = Vec::with_capacity((width * height * 4) as usize);
			for _ in 0..(width * height) {
				rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], alpha]);
			}
			return DecodedFrame {
				frame_index: 0,
				width,
				height,
				payload: PixelPayload::Rgba8(rgba),
			};
		}
	};
	let block_count = width.div_ceil(4) * height.div_ceil(4);
	let mut data = Vec::with_capacity(block_count as usize * block.len());
	for _ in 0..block_count {
		data.extend_from_slice(&block);
	}
	let payload = match kind {
		ClipKind::Hap => PixelPayload::Bc1(data),
		ClipKind::HapAlpha => PixelPayload::Bc3(data),
		ClipKind::HapQ => PixelPayload::Bc3YCoCg(data),
		ClipKind::ImageRgba => panic!("still images are handled before the block path"),
	};
	DecodedFrame {
		frame_index: 0,
		width,
		height,
		payload,
	}
}

/// A full-frame high-contrast pattern so spatial effects produce measurable deltas.
fn pattern_frame(width: u32, height: u32) -> DecodedFrame {
	let mut rgba = Vec::with_capacity((width * height * 4) as usize);
	for y in 0..height {
		for x in 0..width {
			let checker = ((x / 48) + (y / 48)) % 2 == 0;
			let color = if checker {
				[240, 40, 20, 255]
			} else {
				[20, 60, 220, 255]
			};
			rgba.extend_from_slice(&color);
		}
	}
	DecodedFrame {
		frame_index: 0,
		width,
		height,
		payload: PixelPayload::Rgba8(rgba),
	}
}

/// RGB565-expanded normalized RGB exactly as the BC1 decoder produces it.
fn quantized(rgb: [u8; 3]) -> [f32; 3] {
	let color = rgb_to_565(rgb);
	let r5 = u16::from(((color >> 11) & 0x1F) as u8);
	let g6 = u16::from(((color >> 5) & 0x3F) as u8);
	let b5 = u16::from((color & 0x1F) as u8);
	let expand5 = |value: u16| -> f32 { (((value << 3) | (value >> 2)) as f32) / 255.0 };
	let expand6 = |value: u16| -> f32 { (((value << 2) | (value >> 4)) as f32) / 255.0 };
	[expand5(r5), expand6(g6), expand5(b5)]
}

fn blend_expected(mode: BlendMode, dst: [f32; 3], src: [f32; 3]) -> [f32; 3] {
	std::array::from_fn(|channel| {
		let d = dst.get(channel).copied().unwrap_or(0.0);
		let s = src.get(channel).copied().unwrap_or(0.0);
		match mode {
			BlendMode::Normal => s,
			BlendMode::Add => (d + s).clamp(0.0, 1.0),
			BlendMode::Multiply => d * s,
			BlendMode::Screen => 1.0 - (1.0 - d) * (1.0 - s),
			BlendMode::Difference => (d - s).abs(),
		}
	})
}

fn empty_input() -> LayerRenderInput<'static> {
	LayerRenderInput {
		frame: None,
		clip_id: None,
		opacity: 0.0,
		blend_mode: BlendMode::Normal,
		is_opaque_margin: true,
	}
}

fn single_layer_inputs(frame: &DecodedFrame, clip_id: ClipId) -> [LayerRenderInput<'_>; 4] {
	[
		LayerRenderInput {
			frame: Some(frame),
			clip_id: Some(clip_id),
			opacity: 1.0,
			blend_mode: BlendMode::Normal,
			is_opaque_margin: true,
		},
		empty_input(),
		empty_input(),
		empty_input(),
	]
}

// ---------------------------------------------------------------------------
// Blend modes
// ---------------------------------------------------------------------------

#[test]
fn all_five_blend_modes_produce_expected_composites() {
	let _guard = gpu_lock();
	let Some(ctx) = gpu_context("blend_modes") else {
		return;
	};
	let mut renderer = MasterRenderer::new(&ctx).unwrap();

	let bottom = solid_frame(ClipKind::Hap, 4, 4, [64, 128, 255], 255);
	let top = solid_frame(ClipKind::Hap, 4, 4, [255, 128, 0], 255);
	let bottom_rgb = quantized([64, 128, 255]);
	let top_rgb = quantized([255, 128, 0]);

	let mut seen = Vec::new();
	for mode in BlendMode::ALL {
		let layers = [
			LayerRenderInput {
				frame: Some(&bottom),
				clip_id: Some(ClipId(1)),
				opacity: 1.0,
				blend_mode: BlendMode::Normal,
				is_opaque_margin: true,
			},
			LayerRenderInput {
				frame: Some(&top),
				clip_id: Some(ClipId(2)),
				opacity: 1.0,
				blend_mode: mode,
				is_opaque_margin: true,
			},
			empty_input(),
			empty_input(),
		];
		let report = renderer
			.render_master(&ctx, &layers, &MasterEffects::default(), 0.0)
			.unwrap();
		assert_eq!(report.layers_composited, 2, "{mode:?}");
		let pixels = renderer.readback_master_rgba8(&ctx).unwrap();
		let actual = master_pixel(&pixels, INTERNAL_WIDTH / 2, INTERNAL_HEIGHT / 2);
		let expected = blend_expected(mode, bottom_rgb, top_rgb);
		assert_color(actual, expected, 2.0, &format!("blend {mode:?}"));
		assert_eq!(actual[3], 255, "{mode:?} master output is opaque");
		seen.push(actual);
		eprintln!("blend {:?}: {:?} (expected {:?})", mode, actual, expected);
	}
	assert_eq!(seen.len(), 5);
	for (index, first) in seen.iter().enumerate() {
		for second in seen.iter().skip(index + 1) {
			assert_ne!(first, second, "every blend mode must produce a distinct composite");
		}
	}
}

// ---------------------------------------------------------------------------
// HAP Q YCoCg
// ---------------------------------------------------------------------------

fn cpu_bc_decoder(frame: &DecodedFrame) -> Option<Vec<u8>> {
	decode_frame_to_rgba8(frame).ok()
}

#[test]
fn hapq_gpu_ycocg_matches_cpu_decode_and_cpu_fallback_path() {
	let _guard = gpu_lock();
	let Some(ctx) = gpu_context("hapq_ycocg") else {
		return;
	};
	if cfg!(target_os = "macos") {
		assert!(ctx.supports_bc, "Metal must expose BC texture sampling");
	}

	let source_rgb = [200_u8, 120, 40];
	let frame = solid_frame(ClipKind::HapQ, 8, 8, source_rgb, 255);
	let cpu_pixels = decode_frame_to_rgba8(&frame).unwrap();
	let cpu_center = pixel_at(&cpu_pixels, 8, 4, 4);
	assert_color(
		cpu_center,
		[
			f32::from(source_rgb[0]) / 255.0,
			f32::from(source_rgb[1]) / 255.0,
			f32::from(source_rgb[2]) / 255.0,
		],
		12.0,
		"CPU YCoCg decode",
	);

	// Without BC support the frame must be decoded to RGBA8 on the CPU first.
	let fallback_plan = plan_upload(&frame, false, 8192).unwrap();
	assert!(fallback_plan.needs_cpu_decode);
	assert_eq!(fallback_plan.format, wgpu::TextureFormat::Rgba8Unorm);
	let gpu_plan = plan_upload(&frame, true, 8192).unwrap();
	assert_eq!(gpu_plan.format, wgpu::TextureFormat::Bc3RgbaUnorm);
	assert!(gpu_plan.is_ycocg);
	assert!(!gpu_plan.needs_cpu_decode);

	let mut renderer = MasterRenderer::new(&ctx).unwrap();
	renderer.set_cpu_fallback_decoder(cpu_bc_decoder);

	let gpu_center = if ctx.supports_bc {
		let layers = single_layer_inputs(&frame, ClipId(11));
		renderer
			.render_master(&ctx, &layers, &MasterEffects::default(), 0.0)
			.unwrap();
		let pixels = renderer.readback_master_rgba8(&ctx).unwrap();
		let center = master_pixel(&pixels, INTERNAL_WIDTH / 2, INTERNAL_HEIGHT / 2);
		assert_color(
			center,
			[
				f32::from(cpu_center[0]) / 255.0,
				f32::from(cpu_center[1]) / 255.0,
				f32::from(cpu_center[2]) / 255.0,
			],
			2.0,
			"GPU YCoCg conversion",
		);
		eprintln!("HAP Q GPU center={center:?} CPU center={cpu_center:?}");
		center
	} else {
		eprintln!("BC textures unavailable; skipping the GPU YCoCg half of the test");
		cpu_center
	};

	// The CPU fallback creates a plain RGBA8 frame; uploading it must render the
	// same colours as the GPU's on-the-fly YCoCg conversion.
	let cpu_frame = DecodedFrame {
		frame_index: 0,
		width: 8,
		height: 8,
		payload: PixelPayload::Rgba8(cpu_pixels),
	};
	let layers = single_layer_inputs(&cpu_frame, ClipId(12));
	renderer
		.render_master(&ctx, &layers, &MasterEffects::default(), 0.0)
		.unwrap();
	let pixels = renderer.readback_master_rgba8(&ctx).unwrap();
	let fallback_center = master_pixel(&pixels, INTERNAL_WIDTH / 2, INTERNAL_HEIGHT / 2);
	assert_color(
		fallback_center,
		[
			f32::from(gpu_center[0]) / 255.0,
			f32::from(gpu_center[1]) / 255.0,
			f32::from(gpu_center[2]) / 255.0,
		],
		2.0,
		"CPU fallback RGBA upload",
	);
}

// ---------------------------------------------------------------------------
// Aspect-ratio margins
// ---------------------------------------------------------------------------

#[test]
fn aspect_margins_are_black_for_opaque_clips_and_transparent_for_alpha_clips() {
	let _guard = gpu_lock();
	let Some(ctx) = gpu_context("aspect_margins") else {
		return;
	};
	let mut renderer = MasterRenderer::new(&ctx).unwrap();

	let lower = solid_frame(ClipKind::Hap, 64, 36, [0, 255, 0], 255);
	let mut opaque_top = solid_frame(ClipKind::Hap, 64, 48, [255, 255, 0], 255);
	opaque_top.frame_index = 0;
	let mut alpha_top = solid_frame(ClipKind::HapAlpha, 64, 48, [255, 0, 255], 255);
	alpha_top.frame_index = 1;

	let render = |renderer: &mut MasterRenderer, top: &DecodedFrame, opaque: bool| -> Vec<u8> {
		let layers = [
			LayerRenderInput {
				frame: Some(&lower),
				clip_id: Some(ClipId(21)),
				opacity: 1.0,
				blend_mode: BlendMode::Normal,
				is_opaque_margin: true,
			},
			LayerRenderInput {
				frame: Some(top),
				clip_id: Some(ClipId(22)),
				opacity: 1.0,
				blend_mode: BlendMode::Normal,
				is_opaque_margin: opaque,
			},
			empty_input(),
			empty_input(),
		];
		renderer
			.render_master(&ctx, &layers, &MasterEffects::default(), 0.0)
			.unwrap();
		renderer.readback_master_rgba8(&ctx).unwrap()
	};

	// 4:3 inside 16:9 is pillarboxed: 240 px bars on each side.
	let rect = fit_rect(64, 48, INTERNAL_WIDTH, INTERNAL_HEIGHT);
	assert!((rect[0] - 0.125).abs() < 1e-6, "left bar: {rect:?}");
	assert!((rect[2] - 0.875).abs() < 1e-6, "right bar: {rect:?}");
	let margin_x = 120;
	let center_x = INTERNAL_WIDTH / 2;
	let center_y = INTERNAL_HEIGHT / 2;

	let opaque_pixels = render(&mut renderer, &opaque_top, true);
	let opaque_margin = master_pixel(&opaque_pixels, margin_x, center_y);
	let opaque_center = master_pixel(&opaque_pixels, center_x, center_y);
	assert_eq!(opaque_margin, [0, 0, 0, 255], "opaque clip margins are black");
	assert_color(opaque_center, [1.0, 1.0, 0.0], 2.0, "opaque clip centre");

	let alpha_pixels = render(&mut renderer, &alpha_top, false);
	let alpha_margin = master_pixel(&alpha_pixels, margin_x, center_y);
	let alpha_center = master_pixel(&alpha_pixels, center_x, center_y);
	assert_color(
		alpha_margin,
		[0.0, 1.0, 0.0],
		2.0,
		"alpha clip margin shows lower layer",
	);
	assert_color(alpha_center, [1.0, 0.0, 1.0], 2.0, "alpha clip centre");
	assert_ne!(opaque_margin, alpha_margin);
}

// ---------------------------------------------------------------------------
// Master effects
// ---------------------------------------------------------------------------

#[test]
fn master_effects_dimmer_and_blackout_alter_the_output_measurably() {
	let _guard = gpu_lock();
	let Some(ctx) = gpu_context("master_effects") else {
		return;
	};
	let mut renderer = MasterRenderer::new(&ctx).unwrap();
	let frame = pattern_frame(INTERNAL_WIDTH, INTERNAL_HEIGHT);
	let layers = [
		LayerRenderInput {
			frame: Some(&frame),
			clip_id: Some(ClipId(31)),
			opacity: 1.0,
			blend_mode: BlendMode::Normal,
			is_opaque_margin: true,
		},
		empty_input(),
		empty_input(),
		empty_input(),
	];

	let render = |renderer: &mut MasterRenderer, effects: &MasterEffects, time: f32| -> Vec<u8> {
		renderer.render_master(&ctx, &layers, effects, time).unwrap();
		renderer.readback_master_rgba8(&ctx).unwrap()
	};

	let neutral = render(&mut renderer, &MasterEffects::default(), 0.0);

	let rgb_shift = MasterEffects {
		rgb_shift_enabled: true,
		rgb_shift_intensity: 1.0,
		..MasterEffects::default()
	};
	let shifted = render(&mut renderer, &rgb_shift, 0.0);
	let shift_diff = mean_abs_diff(&neutral, &shifted);
	assert!(shift_diff > 2.0, "RGB shift changed too little: {shift_diff}");
	eprintln!("RGB shift mean abs diff: {shift_diff:.2}");

	let glitch = MasterEffects {
		glitch_enabled: true,
		glitch_intensity: 1.0,
		..MasterEffects::default()
	};
	let glitched = render(&mut renderer, &glitch, 1.0);
	let glitch_diff = mean_abs_diff(&neutral, &glitched);
	assert!(glitch_diff > 1.0, "glitch changed too little: {glitch_diff}");
	eprintln!("glitch mean abs diff: {glitch_diff:.2}");

	let strobe = MasterEffects {
		strobe_enabled: true,
		strobe_intensity: 1.0,
		..MasterEffects::default()
	};
	let strobed = render(&mut renderer, &strobe, 0.0);
	let strobe_diff = mean_abs_diff(&neutral, &strobed);
	assert!(strobe_diff > 20.0, "strobe/invert changed too little: {strobe_diff}");
	eprintln!("strobe mean abs diff: {strobe_diff:.2}");

	let brightness = MasterEffects {
		brightness: 0.3,
		..MasterEffects::default()
	};
	let brightened = render(&mut renderer, &brightness, 0.0);
	let brightness_diff = mean_abs_diff(&neutral, &brightened);
	assert!(
		brightness_diff > 10.0,
		"brightness changed too little: {brightness_diff}"
	);

	let contrast = MasterEffects {
		contrast: 1.8,
		..MasterEffects::default()
	};
	let contrasted = render(&mut renderer, &contrast, 0.0);
	let contrast_diff = mean_abs_diff(&neutral, &contrasted);
	assert!(contrast_diff > 5.0, "contrast changed too little: {contrast_diff}");

	let saturation = MasterEffects {
		saturation: 0.0,
		..MasterEffects::default()
	};
	let desaturated = render(&mut renderer, &saturation, 0.0);
	let saturation_diff = mean_abs_diff(&neutral, &desaturated);
	assert!(
		saturation_diff > 3.0,
		"saturation changed too little: {saturation_diff}"
	);

	let dimmer = MasterEffects {
		master_dimmer: 0.0,
		..MasterEffects::default()
	};
	let dimmed = render(&mut renderer, &dimmer, 0.0);
	for y in (0..INTERNAL_HEIGHT).step_by(64) {
		for x in (0..INTERNAL_WIDTH).step_by(64) {
			let pixel = master_pixel(&dimmed, x, y);
			assert_eq!(pixel, [0, 0, 0, 255], "master dimmer 0.0 must be black at {x},{y}");
		}
	}

	let blackout = MasterEffects {
		rgb_shift_enabled: true,
		rgb_shift_intensity: 1.0,
		glitch_enabled: true,
		glitch_intensity: 1.0,
		strobe_enabled: true,
		strobe_intensity: 1.0,
		brightness: 1.0,
		contrast: 2.0,
		saturation: 2.0,
		master_dimmer: 1.0,
		blackout: true,
		strobe_speed_hz: 30.0,
	};
	let blacked_out = render(&mut renderer, &blackout, 0.25);
	assert!(
		blacked_out
			.as_chunks::<4>()
			.0
			.iter()
			.all(|pixel| *pixel == [0, 0, 0, 255]),
		"blackout must force every output pixel to opaque black"
	);
}

// ---------------------------------------------------------------------------
// Presentation
// ---------------------------------------------------------------------------

#[test]
fn present_surface_letterboxes_a_16_by_9_output_on_a_4_by_3_target() {
	let _guard = gpu_lock();
	let Some(ctx) = gpu_context("present") else {
		return;
	};
	let mut renderer = MasterRenderer::new(&ctx).unwrap();
	let frame = solid_frame(ClipKind::Hap, 64, 36, [255, 0, 0], 255);
	let layers = [
		LayerRenderInput {
			frame: Some(&frame),
			clip_id: Some(ClipId(41)),
			opacity: 1.0,
			blend_mode: BlendMode::Normal,
			is_opaque_margin: true,
		},
		empty_input(),
		empty_input(),
		empty_input(),
	];
	renderer
		.render_master(&ctx, &layers, &MasterEffects::default(), 0.0)
		.unwrap();

	let (target_w, target_h) = (800_u32, 600_u32);
	let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
		label: Some("mur.test.present_target"),
		size: wgpu::Extent3d {
			width: target_w,
			height: target_h,
			depth_or_array_layers: 1,
		},
		mip_level_count: 1,
		sample_count: 1,
		dimension: wgpu::TextureDimension::D2,
		format: wgpu::TextureFormat::Rgba8Unorm,
		usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
		view_formats: &[],
	});
	let view = target.create_view(&wgpu::TextureViewDescriptor::default());
	renderer
		.present_surface(&ctx, &view, wgpu::TextureFormat::Rgba8Unorm, target_w, target_h)
		.unwrap();
	let pixels = read_texture_rgba8(&ctx, &target, target_w, target_h);

	let rect = fit_rect(INTERNAL_WIDTH, INTERNAL_HEIGHT, target_w, target_h);
	assert!((rect[0] - 0.0).abs() < 1e-6, "{rect:?}");
	assert!((rect[1] - 0.125).abs() < 1e-6, "{rect:?}");
	assert!((rect[2] - 1.0).abs() < 1e-6, "{rect:?}");
	assert!((rect[3] - 0.875).abs() < 1e-6, "{rect:?}");

	let center = pixel_at(&pixels, target_w, target_w / 2, target_h / 2);
	assert_color(center, [1.0, 0.0, 0.0], 2.0, "present centre");
	let top_bar = pixel_at(&pixels, target_w, target_w / 2, 20);
	let bottom_bar = pixel_at(&pixels, target_w, target_w / 2, target_h - 20);
	assert_eq!(top_bar, [0, 0, 0, 255], "top letterbox bar");
	assert_eq!(bottom_bar, [0, 0, 0, 255], "bottom letterbox bar");
	let left = pixel_at(&pixels, target_w, 5, target_h / 2);
	let right = pixel_at(&pixels, target_w, target_w - 5, target_h / 2);
	assert_color(left, [1.0, 0.0, 0.0], 2.0, "present left edge");
	assert_color(right, [1.0, 0.0, 0.0], 2.0, "present right edge");
}

// ---------------------------------------------------------------------------
// Sustained four-layer 1080p performance
// ---------------------------------------------------------------------------

/// Builds four distinct 1920x1080 clips (HAP, HAP Alpha, HAP Q, and a PNG).
fn build_fhd_clips(dir: &TempDir) -> Vec<(ClipId, mur::model::ClipInfo, ClipAsset)> {
	let mut clips = Vec::with_capacity(4);
	let kinds = [ClipKind::Hap, ClipKind::HapAlpha, ClipKind::HapQ];
	for (index, kind) in kinds.iter().enumerate() {
		let path = dir.path().join(format!("perf_{index}.mov"));
		let mut fixture = HapMovFixture::new(*kind, INTERNAL_WIDTH, INTERNAL_HEIGHT, 6);
		fixture.compression = FixtureCompression::Snappy;
		fixture.use_co64 = index == 2;
		fixture.include_audio_track = index == 1;
		fixture.write_to(&path).unwrap();
		let id = ClipId(index as u64 + 1);
		let (info, asset) = load_clip(id, &path).unwrap();
		clips.push((id, info, asset));
	}

	let png_path = dir.path().join("perf_still.png");
	let image = image::RgbaImage::from_fn(INTERNAL_WIDTH, INTERNAL_HEIGHT, |x, y| {
		let checker = ((x / 96) + (y / 96)) % 2 == 0;
		if checker {
			image::Rgba([30, 200, 160, 255])
		} else {
			image::Rgba([220, 40, 90, 255])
		}
	});
	image.save(&png_path).unwrap();
	let id = ClipId(4);
	let (info, asset) = load_clip(id, &png_path).unwrap();
	clips.push((id, info, asset));
	clips
}

#[test]
fn four_layer_fhd_sustained_60fps_budget() {
	let _guard = gpu_lock();
	let Some(ctx) = gpu_context("four_layer_fhd") else {
		return;
	};
	let dir = TempDir::new("perf_fhd").unwrap();
	let clips = build_fhd_clips(&dir);
	assert_eq!(clips.len(), 4);

	let streamer = mur::media::LayerStreamer::new().unwrap();
	for (_, info, asset) in &clips {
		assert_eq!(info.width, INTERNAL_WIDTH);
		assert_eq!(info.height, INTERNAL_HEIGHT);
		streamer.register_asset(asset.clone());
	}

	let modes = [
		BlendMode::Add,
		BlendMode::Multiply,
		BlendMode::Screen,
		BlendMode::Difference,
	];
	let mut states: [LayerState; 4] = std::array::from_fn(|index| {
		let (id, info, _) = &clips[index];
		let mut state = LayerState::default();
		state.trigger(*id, info.frame_count.max(1), info.fps);
		state.playback_mode = PlaybackMode::Loop;
		state.set_opacity(0.95 - index as f32 * 0.1);
		state.set_speed(1.0);
		state.blend_mode = modes[index];
		state
	});
	let effects = MasterEffects {
		rgb_shift_enabled: true,
		rgb_shift_intensity: 0.4,
		glitch_enabled: true,
		glitch_intensity: 0.35,
		strobe_enabled: true,
		strobe_intensity: 0.25,
		strobe_speed_hz: 8.0,
		brightness: 0.05,
		contrast: 1.1,
		saturation: 1.15,
		master_dimmer: 0.9,
		blackout: false,
	};

	// Prime the workers so the measured loop only covers steady-state playback.
	for (index, layer) in LayerIndex::ALL.iter().enumerate() {
		let (id, info, _) = &clips[index];
		let desired = if info.frame_count > 1 { 1 } else { 0 };
		streamer.request_layer_state(
			*layer,
			*id,
			desired,
			1.0,
			PlaybackMode::Loop,
			states[index].playback_state.is_playing(),
		);
		if info.frame_count > 1 {
			let ready = {
				let start = Instant::now();
				let mut ready = false;
				while start.elapsed() < std::time::Duration::from_secs(15) {
					if streamer.poll_frame(*layer, *id, desired).is_some() {
						ready = true;
						break;
					}
					std::thread::sleep(std::time::Duration::from_millis(2));
				}
				ready
			};
			assert!(ready, "layer {layer} did not prime frame {desired}");
		}
	}

	let mut renderer = MasterRenderer::new(&ctx).unwrap();
	let dt = 1.0_f32 / 60.0;
	let frame_budget = 180_u32;
	let mut gpu_slots: [Option<Arc<DecodedFrame>>; 4] = std::array::from_fn(|_| None);

	let start = Instant::now();
	for tick in 0..frame_budget {
		for state in &mut states {
			state.advance_playhead(dt);
		}

		for (index, layer) in LayerIndex::ALL.iter().enumerate() {
			let state = &states[index];
			let (id, _, _) = &clips[index];
			let desired = state.current_frame_index();
			streamer.request_layer_state(
				*layer,
				*id,
				desired,
				state.speed,
				state.playback_mode,
				state.playback_state.is_playing(),
			);
			gpu_slots[index] = streamer.poll_frame(*layer, *id, desired);
		}

		let inputs: [LayerRenderInput<'_>; 4] = std::array::from_fn(|index| {
			let (id, info, _) = &clips[index];
			LayerRenderInput {
				frame: gpu_slots[index].as_deref(),
				clip_id: Some(*id),
				opacity: states[index].opacity,
				blend_mode: states[index].blend_mode,
				is_opaque_margin: info.is_opaque_margin(),
			}
		});
		let report = renderer
			.render_master(&ctx, &inputs, &effects, tick as f32 * dt)
			.unwrap();
		assert_eq!(
			report.layers_composited, 4,
			"tick {tick} must composite all four layers"
		);

		if tick % 30 == 0 {
			ctx.device
				.poll(wgpu::PollType::Wait {
					submission_index: None,
					timeout: None,
				})
				.unwrap();
		}
	}
	ctx.device
		.poll(wgpu::PollType::Wait {
			submission_index: None,
			timeout: None,
		})
		.unwrap();
	let elapsed = start.elapsed();
	let avg_ms = elapsed.as_secs_f64() * 1000.0 / f64::from(frame_budget);
	let fps = 1000.0 / avg_ms;
	eprintln!(
		"4-layer 1080p x180 avg frame time: {avg_ms:.3} ms ({fps:.1} fps) \
		 [adapter={}, backend={:?}, bc={}]",
		ctx.adapter.get_info().name,
		ctx.adapter.get_info().backend,
		ctx.supports_bc
	);

	assert!(
		ctx.take_last_error().is_none(),
		"the sustained GPU loop produced validation errors"
	);

	#[cfg(not(debug_assertions))]
	assert!(
		avg_ms < 16.67,
		"release build missed the 60 fps budget: {avg_ms:.3} ms per frame ({fps:.1} fps)"
	);
	#[cfg(debug_assertions)]
	assert!(
		avg_ms < 100.0,
		"debug build exceeded the sanity budget: {avg_ms:.3} ms per frame ({fps:.1} fps)"
	);
}
