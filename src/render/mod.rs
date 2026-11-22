//! GPU renderer: 4-layer compositor, master effects, presentation, and mirror.
//!
//! Every frame flows through the same fixed pipeline:
//!
//! 1. [`texture_cache`] uploads changed layer frames (BC1 / BC3 / HAP Q YCoCg
//!    blocks, or RGBA8) into per-layer textures.
//! 2. [`compositor`] fits each layer into the 1920x1080 frame with aspect-ratio
//!    margins and blends it with one of five blend modes.
//! 3. [`effects`] applies the master shader effects, dimmer, and blackout into
//!    the `master_output` texture.
//! 4. [`presenter`] scales `master_output` onto output surfaces, while the
//!    control UI mirrors the same texture through `egui_wgpu` with no CPU
//!    readback.
//!
//! GPU validation errors are routed to [`GpuContext::take_last_error`] instead
//! of panicking, and device loss is reported by [`GpuContext::is_device_lost`].

pub mod compositor;
pub mod effects;
pub mod presenter;
pub mod texture_cache;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use crate::error::RenderError;
use crate::model::layer::LAYER_COUNT;
use crate::model::{BlendMode, ClipId, DecodedFrame, MasterEffects};

use self::compositor::{LayerCompositor, LayerDraw};
use self::effects::{EffectsUniform, MasterEffectsPass};
use self::presenter::SurfacePresenter;
use self::texture_cache::{CpuFallbackDecoder, LayerTextureCache, LayerUploadStatus, UploadContext};

/// Internal render width in pixels.
pub const INTERNAL_WIDTH: u32 = 1920;
/// Internal render height in pixels.
pub const INTERNAL_HEIGHT: u32 = 1080;
/// Texture format of every internal render target and of `master_output`.
pub const INTERNAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Row pitch alignment required for texture-to-buffer copies.
const COPY_ROW_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// Shared wgpu instance, adapter, device, and queue plus error tracking.
#[derive(Debug)]
pub struct GpuContext {
	/// The wgpu instance every surface must be created from.
	pub instance: wgpu::Instance,
	/// The selected GPU adapter.
	pub adapter: wgpu::Adapter,
	/// The logical device.
	pub device: Arc<wgpu::Device>,
	/// The device's submission queue.
	pub queue: Arc<wgpu::Queue>,
	/// Whether BC1/BC3 textures can be sampled directly.
	pub supports_bc: bool,
	device_lost: Arc<AtomicBool>,
	last_gpu_error: Arc<Mutex<Option<String>>>,
}

impl GpuContext {
	/// Creates a context on a new [`wgpu::Instance`].
	///
	/// Pass `None` for headless use. A provided surface must have been created
	/// from the same instance; use [`GpuContext::with_instance`] together with
	/// [`GpuContext::create_instance`] when a surface must exist first.
	pub fn new(compatible_surface: Option<&wgpu::Surface<'_>>) -> Result<Self, RenderError> {
		Self::with_instance(Self::create_instance(), compatible_surface)
	}

	/// Creates the instance used by [`GpuContext::new`] (primary backends).
	pub fn create_instance() -> wgpu::Instance {
		let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
		descriptor.backends = wgpu::Backends::PRIMARY;
		wgpu::Instance::new(descriptor)
	}

	/// Creates a context on an existing instance, preferring a high-performance adapter.
	pub fn with_instance(
		instance: wgpu::Instance,
		compatible_surface: Option<&wgpu::Surface<'_>>,
	) -> Result<Self, RenderError> {
		let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
			power_preference: wgpu::PowerPreference::HighPerformance,
			force_fallback_adapter: false,
			compatible_surface,
			apply_limit_buckets: false,
		}))
		.map_err(|err| RenderError::DeviceCreation(format!("no suitable GPU adapter: {err}")))?;

		let supports_bc = adapter.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC);
		let required_features = if supports_bc {
			wgpu::Features::TEXTURE_COMPRESSION_BC
		} else {
			wgpu::Features::empty()
		};
		let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
			label: Some("mur.device"),
			required_features,
			required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
			experimental_features: wgpu::ExperimentalFeatures::disabled(),
			memory_hints: wgpu::MemoryHints::Performance,
			trace: wgpu::Trace::Off,
		}))
		.map_err(|err| RenderError::DeviceCreation(err.to_string()))?;

		let device_lost = Arc::new(AtomicBool::new(false));
		let last_gpu_error = Arc::new(Mutex::new(None));

		let error_slot = Arc::clone(&last_gpu_error);
		device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
			let message = error.to_string();
			tracing::error!(target: "render", "uncaptured GPU error: {message}");
			*error_slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(message);
		}));

		let lost_flag = Arc::clone(&device_lost);
		let lost_slot = Arc::clone(&last_gpu_error);
		device.set_device_lost_callback(move |reason, message| {
			lost_flag.store(true, Ordering::SeqCst);
			let message = format!("device lost ({reason:?}): {message}");
			tracing::error!(target: "render", "{message}");
			*lost_slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(message);
		});

		let info = adapter.get_info();
		tracing::info!(
			target: "render",
			"GPU adapter `{}` ({:?}), BC textures: {supports_bc}",
			info.name,
			info.backend
		);

		Ok(Self {
			instance,
			adapter,
			device: Arc::new(device),
			queue: Arc::new(queue),
			supports_bc,
			device_lost,
			last_gpu_error,
		})
	}

	/// Returns `true` once the device was lost and must be recreated.
	pub fn is_device_lost(&self) -> bool {
		self.device_lost.load(Ordering::SeqCst)
	}

	/// Takes the most recent uncaptured GPU error or device-loss message.
	pub fn take_last_error(&self) -> Option<String> {
		self.last_gpu_error
			.lock()
			.unwrap_or_else(PoisonError::into_inner)
			.take()
	}

	/// Creates a surface for `window` from this context's instance.
	pub fn create_surface(&self, window: Arc<winit::window::Window>) -> Result<wgpu::Surface<'static>, RenderError> {
		self.instance
			.create_surface(window)
			.map_err(|err| RenderError::SurfaceLost(format!("failed to create surface: {err}")))
	}

	/// Configures `surface` for `width`x`height` with a non-sRGB format and vsync.
	pub fn configure_surface(
		&self,
		surface: &wgpu::Surface<'_>,
		width: u32,
		height: u32,
	) -> Result<wgpu::SurfaceConfiguration, RenderError> {
		if width == 0 || height == 0 {
			return Err(RenderError::InvalidTarget(format!(
				"surface size must be non-zero, got {width}x{height}"
			)));
		}
		let caps = surface.get_capabilities(&self.adapter);
		let format = select_surface_format(&caps.formats).ok_or_else(|| {
			RenderError::SurfaceLost("surface is incompatible with the selected GPU adapter".to_owned())
		})?;
		let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
			wgpu::PresentMode::Fifo
		} else {
			wgpu::PresentMode::AutoVsync
		};
		let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
			wgpu::CompositeAlphaMode::Opaque
		} else {
			caps.alpha_modes
				.first()
				.copied()
				.unwrap_or(wgpu::CompositeAlphaMode::Auto)
		};
		let config = wgpu::SurfaceConfiguration {
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
			format,
			color_space: wgpu::SurfaceColorSpace::default(),
			width,
			height,
			present_mode,
			desired_maximum_frame_latency: 2,
			alpha_mode,
			view_formats: Vec::new(),
		};

		let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
		surface.configure(&self.device, &config);
		if let Some(error) = pollster::block_on(scope.pop()) {
			return Err(RenderError::SurfaceLost(format!(
				"failed to configure surface: {error}"
			)));
		}
		Ok(config)
	}
}

/// Picks `Bgra8Unorm`, then `Rgba8Unorm`, then any non-sRGB, then any format.
pub fn select_surface_format(formats: &[wgpu::TextureFormat]) -> Option<wgpu::TextureFormat> {
	[wgpu::TextureFormat::Bgra8Unorm, wgpu::TextureFormat::Rgba8Unorm]
		.into_iter()
		.find(|preferred| formats.contains(preferred))
		.or_else(|| formats.iter().copied().find(|format| !format.is_srgb()))
		.or_else(|| formats.first().copied())
}

/// Per-layer input for [`MasterRenderer::render_master`].
#[derive(Debug, Clone, Copy)]
pub struct LayerRenderInput<'a> {
	/// Newly decoded frame, or `None` to hold the last frame of `clip_id`.
	pub frame: Option<&'a DecodedFrame>,
	/// Clip shown on the layer; `None` means the layer is empty.
	pub clip_id: Option<ClipId>,
	/// Layer opacity in `0.0..=1.0`.
	pub opacity: f32,
	/// Blend mode applied onto the layers below.
	pub blend_mode: BlendMode,
	/// `true` for opaque clips (black margins), `false` for alpha clips (transparent margins).
	pub is_opaque_margin: bool,
}

impl LayerRenderInput<'_> {
	/// An empty layer that draws nothing.
	pub const EMPTY: LayerRenderInput<'static> = LayerRenderInput {
		frame: None,
		clip_id: None,
		opacity: 0.0,
		blend_mode: BlendMode::Normal,
		is_opaque_margin: true,
	};
}

/// Per-frame summary returned by [`MasterRenderer::render_master`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderFrameReport {
	/// Texture preparation outcome per layer.
	pub layers: [LayerUploadStatus; LAYER_COUNT],
	/// Number of layers blended into the composite.
	pub layers_composited: u32,
}

impl RenderFrameReport {
	/// Number of layers whose texture was written this frame.
	pub fn uploads(&self) -> usize {
		self.layers
			.iter()
			.filter(|status| matches!(status, LayerUploadStatus::Uploaded))
			.count()
	}
}

/// Owns the full master render pipeline and its `master_output` texture.
#[derive(Debug)]
pub struct MasterRenderer {
	cache: LayerTextureCache,
	compositor: LayerCompositor,
	effects: MasterEffectsPass,
	presenter: SurfacePresenter,
	cpu_fallback: Option<CpuFallbackDecoder>,
	mirror_id: Mutex<Option<egui::TextureId>>,
}

impl MasterRenderer {
	/// Builds every pipeline and render target on `ctx`'s device.
	pub fn new(ctx: &GpuContext) -> Result<Self, RenderError> {
		let device = ctx.device.as_ref();
		let cache = LayerTextureCache::new(device);
		let compositor = LayerCompositor::new(device, &ctx.queue, &cache)?;
		let effects = MasterEffectsPass::new(device, &compositor)?;
		let presenter = SurfacePresenter::new(device, effects.output_view())?;
		Ok(Self {
			cache,
			compositor,
			effects,
			presenter,
			cpu_fallback: None,
			mirror_id: Mutex::new(None),
		})
	}

	/// Installs the CPU decoder used for BC payloads when `!ctx.supports_bc`.
	pub fn set_cpu_fallback_decoder(&mut self, decoder: CpuFallbackDecoder) {
		self.cpu_fallback = Some(decoder);
	}

	/// The per-layer texture cache.
	pub fn texture_cache(&self) -> &LayerTextureCache {
		&self.cache
	}

	/// Uploads changed layer frames, composites all layers, and applies the
	/// master effects into `master_output`.
	///
	/// Layers are blended bottom (index 0) to top (index 3).
	pub fn render_master(
		&mut self,
		ctx: &GpuContext,
		layers: &[LayerRenderInput<'_>; LAYER_COUNT],
		effects: &MasterEffects,
		time_secs: f32,
	) -> Result<RenderFrameReport, RenderError> {
		if ctx.is_device_lost() {
			return Err(RenderError::DeviceLost(
				"render skipped because the GPU device was lost".to_owned(),
			));
		}
		let upload_ctx = UploadContext {
			device: &ctx.device,
			queue: &ctx.queue,
			supports_bc: ctx.supports_bc,
			cpu_fallback: self.cpu_fallback,
		};

		let mut report = RenderFrameReport {
			layers: [LayerUploadStatus::Inactive; LAYER_COUNT],
			layers_composited: 0,
		};
		let mut draws: [Option<LayerDraw>; LAYER_COUNT] = [None; LAYER_COUNT];
		for (index, ((input, draw), status)) in layers
			.iter()
			.zip(draws.iter_mut())
			.zip(report.layers.iter_mut())
			.enumerate()
		{
			let opacity = if input.opacity.is_finite() {
				input.opacity.clamp(0.0, 1.0)
			} else {
				0.0
			};
			if opacity <= 0.0 {
				continue;
			}
			*status = self.cache.prepare(&upload_ctx, index, input.frame, input.clip_id);
			if !status.is_drawable() {
				continue;
			}
			if let Some(slot) = self.cache.slot(index) {
				*draw = Some(LayerDraw::new(
					index,
					slot.content_size(),
					slot.uv_max(),
					opacity,
					input.blend_mode,
					slot.is_ycocg(),
					input.is_opaque_margin,
				));
				report.layers_composited = report.layers_composited.saturating_add(1);
			}
		}

		let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("mur.master.encoder"),
		});
		let composite = self.compositor.encode(&ctx.queue, &mut encoder, &self.cache, &draws);
		let uniform = EffectsUniform::from_effects(effects, time_secs);
		self.effects.encode(&ctx.queue, &mut encoder, composite, &uniform);
		ctx.queue.submit(std::iter::once(encoder.finish()));
		Ok(report)
	}

	/// View of the 1920x1080 `Rgba8Unorm` master output.
	pub fn master_view(&self) -> &wgpu::TextureView {
		self.effects.output_view()
	}

	/// The 1920x1080 `Rgba8Unorm` master output texture.
	pub fn master_texture(&self) -> &wgpu::Texture {
		self.effects.output()
	}

	/// Blits `master_output` aspect-fit into `target_view` with black bars.
	///
	/// sRGB target formats receive linearized values so colors match the
	/// non-sRGB path.
	pub fn present_surface(
		&self,
		ctx: &GpuContext,
		target_view: &wgpu::TextureView,
		target_format: wgpu::TextureFormat,
		target_w: u32,
		target_h: u32,
	) -> Result<(), RenderError> {
		if ctx.is_device_lost() {
			return Err(RenderError::DeviceLost(
				"present skipped because the GPU device was lost".to_owned(),
			));
		}
		self.presenter
			.present(&ctx.device, &ctx.queue, target_view, target_format, target_w, target_h)
	}

	/// Registers `master_view()` with `egui_renderer` (or rebinds `existing`)
	/// and returns the texture id to draw the mirror preview with.
	///
	/// The texture is shared on the GPU, so no CPU readback happens. Calling it
	/// every frame is cheap: once registered, the id is returned unchanged.
	pub fn sync_mirror_texture(
		&self,
		device: &wgpu::Device,
		egui_renderer: &mut egui_wgpu::Renderer,
		existing: &mut Option<egui::TextureId>,
	) -> egui::TextureId {
		let mut registered = self.mirror_id.lock().unwrap_or_else(PoisonError::into_inner);
		match *existing {
			Some(id) if *registered == Some(id) && egui_renderer.texture(&id).is_some() => id,
			Some(id) if egui_renderer.texture(&id).is_some() => {
				egui_renderer.update_egui_texture_from_wgpu_texture(
					device,
					self.master_view(),
					wgpu::FilterMode::Linear,
					id,
				);
				*registered = Some(id);
				id
			}
			_ => {
				let id = egui_renderer.register_native_texture(device, self.master_view(), wgpu::FilterMode::Linear);
				*existing = Some(id);
				*registered = Some(id);
				id
			}
		}
	}

	/// Copies `master_output` back to the CPU as tightly packed RGBA8.
	///
	/// Blocks until the GPU finishes; intended for tests and diagnostics.
	pub fn readback_master_rgba8(&self, ctx: &GpuContext) -> Result<Vec<u8>, RenderError> {
		let unpadded_row = INTERNAL_WIDTH * 4;
		let padded_row = unpadded_row.div_ceil(COPY_ROW_ALIGNMENT) * COPY_ROW_ALIGNMENT;
		let buffer_size = u64::from(padded_row) * u64::from(INTERNAL_HEIGHT);
		let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("mur.readback.staging"),
			size: buffer_size,
			usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
			mapped_at_creation: false,
		});

		let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("mur.readback.encoder"),
		});
		encoder.copy_texture_to_buffer(
			wgpu::TexelCopyTextureInfo {
				texture: self.master_texture(),
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			wgpu::TexelCopyBufferInfo {
				buffer: &staging,
				layout: wgpu::TexelCopyBufferLayout {
					offset: 0,
					bytes_per_row: Some(padded_row),
					rows_per_image: Some(INTERNAL_HEIGHT),
				},
			},
			wgpu::Extent3d {
				width: INTERNAL_WIDTH,
				height: INTERNAL_HEIGHT,
				depth_or_array_layers: 1,
			},
		);
		ctx.queue.submit(std::iter::once(encoder.finish()));

		let (sender, receiver) = std::sync::mpsc::channel();
		staging.map_async(wgpu::MapMode::Read, .., move |result| {
			let _ = sender.send(result);
		});
		ctx.device
			.poll(wgpu::PollType::Wait {
				submission_index: None,
				timeout: None,
			})
			.map_err(|err| RenderError::DeviceLost(format!("readback poll failed: {err}")))?;
		receiver
			.recv()
			.map_err(|_| RenderError::DeviceLost("readback callback was dropped".to_owned()))?
			.map_err(|err| RenderError::DeviceLost(format!("readback mapping failed: {err}")))?;

		let mut pixels = Vec::with_capacity((unpadded_row as usize).saturating_mul(INTERNAL_HEIGHT as usize));
		{
			let mapped = staging
				.get_mapped_range(..)
				.map_err(|err| RenderError::DeviceLost(format!("readback range failed: {err}")))?;
			for row in mapped.chunks_exact(padded_row as usize) {
				pixels.extend_from_slice(row.get(..unpadded_row as usize).unwrap_or(&[]));
			}
		}
		staging.unmap();
		Ok(pixels)
	}
}

/// Largest centered rectangle of `src` aspect inside `dst`, in `dst` UV space.
///
/// Returns `[min_u, min_v, max_u, max_v]`; degenerate sizes yield the full rect.
pub fn fit_rect(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> [f32; 4] {
	if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
		return [0.0, 0.0, 1.0, 1.0];
	}
	let src_aspect = src_w as f64 / src_h as f64;
	let dst_aspect = dst_w as f64 / dst_h as f64;
	let (width, height) = if src_aspect > dst_aspect {
		(1.0, dst_aspect / src_aspect)
	} else {
		(src_aspect / dst_aspect, 1.0)
	};
	let min_u = (1.0 - width) * 0.5;
	let min_v = (1.0 - height) * 0.5;
	[
		min_u as f32,
		min_v as f32,
		(min_u + width) as f32,
		(min_v + height) as f32,
	]
}

/// Uniform buffer binding visible to the fragment stage.
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
	wgpu::BindGroupLayoutEntry {
		binding,
		visibility: wgpu::ShaderStages::FRAGMENT,
		ty: wgpu::BindingType::Buffer {
			ty: wgpu::BufferBindingType::Uniform,
			has_dynamic_offset: false,
			min_binding_size: None,
		},
		count: None,
	}
}

/// Filterable 2D float texture binding visible to the fragment stage.
fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
	wgpu::BindGroupLayoutEntry {
		binding,
		visibility: wgpu::ShaderStages::FRAGMENT,
		ty: wgpu::BindingType::Texture {
			sample_type: wgpu::TextureSampleType::Float { filterable: true },
			view_dimension: wgpu::TextureViewDimension::D2,
			multisampled: false,
		},
		count: None,
	}
}

/// Filtering sampler binding visible to the fragment stage.
fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
	wgpu::BindGroupLayoutEntry {
		binding,
		visibility: wgpu::ShaderStages::FRAGMENT,
		ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
		count: None,
	}
}

/// Bilinear sampler clamped to the texture edge.
fn linear_clamp_sampler(device: &wgpu::Device, label: &str) -> wgpu::Sampler {
	device.create_sampler(&wgpu::SamplerDescriptor {
		label: Some(label),
		address_mode_u: wgpu::AddressMode::ClampToEdge,
		address_mode_v: wgpu::AddressMode::ClampToEdge,
		address_mode_w: wgpu::AddressMode::ClampToEdge,
		mag_filter: wgpu::FilterMode::Linear,
		min_filter: wgpu::FilterMode::Linear,
		mipmap_filter: wgpu::MipmapFilterMode::Nearest,
		..Default::default()
	})
}

/// Creates a 1920x1080 internal render target that can also be sampled and copied.
fn create_internal_target(device: &wgpu::Device, label: &str) -> wgpu::Texture {
	device.create_texture(&wgpu::TextureDescriptor {
		label: Some(label),
		size: wgpu::Extent3d {
			width: INTERNAL_WIDTH,
			height: INTERNAL_HEIGHT,
			depth_or_array_layers: 1,
		},
		mip_level_count: 1,
		sample_count: 1,
		dimension: wgpu::TextureDimension::D2,
		format: INTERNAL_FORMAT,
		usage: wgpu::TextureUsages::RENDER_ATTACHMENT
			| wgpu::TextureUsages::TEXTURE_BINDING
			| wgpu::TextureUsages::COPY_SRC,
		view_formats: &[],
	})
}

/// Color attachment cleared to opaque black and stored.
fn color_attachment(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
	wgpu::RenderPassColorAttachment {
		view,
		depth_slice: None,
		resolve_target: None,
		ops: wgpu::Operations {
			load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
			store: wgpu::StoreOp::Store,
		},
	}
}

/// Compiles WGSL, turning validation failures into [`RenderError::PipelineCreation`].
fn create_shader(device: &wgpu::Device, name: &str, source: &str) -> Result<wgpu::ShaderModule, RenderError> {
	let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
	let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
		label: Some(name),
		source: wgpu::ShaderSource::Wgsl(source.into()),
	});
	match pollster::block_on(scope.pop()) {
		Some(error) => Err(RenderError::PipelineCreation {
			name: name.to_owned(),
			detail: error.to_string(),
		}),
		None => Ok(module),
	}
}

/// Builds a fullscreen-triangle pipeline (`vs_main`/`fs_main`) for `format`.
fn build_pipeline(
	device: &wgpu::Device,
	name: &str,
	shader: &wgpu::ShaderModule,
	layout: &wgpu::PipelineLayout,
	format: wgpu::TextureFormat,
) -> Result<wgpu::RenderPipeline, RenderError> {
	let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
	let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
		label: Some(name),
		layout: Some(layout),
		vertex: wgpu::VertexState {
			module: shader,
			entry_point: Some("vs_main"),
			compilation_options: wgpu::PipelineCompilationOptions::default(),
			buffers: &[],
		},
		primitive: wgpu::PrimitiveState::default(),
		depth_stencil: None,
		multisample: wgpu::MultisampleState::default(),
		fragment: Some(wgpu::FragmentState {
			module: shader,
			entry_point: Some("fs_main"),
			compilation_options: wgpu::PipelineCompilationOptions::default(),
			targets: &[Some(wgpu::ColorTargetState {
				format,
				blend: None,
				write_mask: wgpu::ColorWrites::ALL,
			})],
		}),
		multiview_mask: None,
		cache: None,
	});
	match pollster::block_on(scope.pop()) {
		Some(error) => Err(RenderError::PipelineCreation {
			name: name.to_owned(),
			detail: error.to_string(),
		}),
		None => Ok(pipeline),
	}
}

/// Compiles `source` and builds its fullscreen pipeline in one step.
fn create_fullscreen_pipeline(
	device: &wgpu::Device,
	name: &str,
	source: &str,
	layout: &wgpu::PipelineLayout,
	format: wgpu::TextureFormat,
) -> Result<wgpu::RenderPipeline, RenderError> {
	let shader = create_shader(device, name, source)?;
	build_pipeline(device, name, &shader, layout, format)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::texture_cache::UploadRejection;
	use super::*;
	use crate::model::PixelPayload;

	/// Creates a headless context, or `None` when the machine has no GPU.
	fn gpu() -> Option<GpuContext> {
		match GpuContext::new(None) {
			Ok(ctx) => Some(ctx),
			Err(err) => {
				eprintln!("skipping GPU test: {err}");
				None
			}
		}
	}

	fn solid_rgba(width: u32, height: u32, rgba: [u8; 4]) -> DecodedFrame {
		DecodedFrame {
			frame_index: 0,
			width,
			height,
			payload: PixelPayload::Rgba8(rgba.repeat((width * height) as usize)),
		}
	}

	fn layer(frame: &DecodedFrame, clip: u64, blend_mode: BlendMode, opaque: bool) -> LayerRenderInput<'_> {
		LayerRenderInput {
			frame: Some(frame),
			clip_id: Some(ClipId(clip)),
			opacity: 1.0,
			blend_mode,
			is_opaque_margin: opaque,
		}
	}

	fn pixel(data: &[u8], x: u32, y: u32) -> [u8; 4] {
		let offset = ((y * INTERNAL_WIDTH + x) * 4) as usize;
		[data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]
	}

	fn assert_close(actual: [u8; 4], expected: [u8; 4], tolerance: u8) {
		for (a, e) in actual.iter().zip(expected.iter()) {
			assert!(
				a.abs_diff(*e) <= tolerance,
				"pixel {actual:?} differs from {expected:?}"
			);
		}
	}

	fn render(
		ctx: &GpuContext,
		renderer: &mut MasterRenderer,
		layers: [LayerRenderInput<'_>; LAYER_COUNT],
		effects: &MasterEffects,
	) -> (RenderFrameReport, Vec<u8>) {
		let report = renderer.render_master(ctx, &layers, effects, 0.25).unwrap();
		let pixels = renderer.readback_master_rgba8(ctx).unwrap();
		assert_eq!(pixels.len(), (INTERNAL_WIDTH * INTERNAL_HEIGHT * 4) as usize);
		assert_eq!(ctx.take_last_error(), None);
		(report, pixels)
	}

	/// One BC3 block whose alpha is `alpha` and whose RGB565 color is `color`.
	fn bc3_block(alpha: u8, color: u16) -> [u8; 16] {
		let [lo, hi] = color.to_le_bytes();
		[alpha, alpha, 0, 0, 0, 0, 0, 0, lo, hi, lo, hi, 0, 0, 0, 0]
	}

	#[test]
	fn fit_rect_letterbox_and_pillarbox() {
		assert_eq!(fit_rect(1920, 1080, 1920, 1080), [0.0, 0.0, 1.0, 1.0]);
		let [min_u, _, max_u, _] = fit_rect(1440, 1080, 1920, 1080);
		assert!((min_u - 0.125).abs() < 1e-6 && (max_u - 0.875).abs() < 1e-6);
		let [_, min_v, _, max_v] = fit_rect(3840, 1080, 1920, 1080);
		assert!((min_v - 0.25).abs() < 1e-6 && (max_v - 0.75).abs() < 1e-6);
		assert_eq!(fit_rect(0, 10, 1920, 1080), [0.0, 0.0, 1.0, 1.0]);
	}

	#[test]
	fn surface_format_prefers_non_srgb() {
		use wgpu::TextureFormat as F;
		assert_eq!(
			select_surface_format(&[F::Bgra8UnormSrgb, F::Bgra8Unorm]),
			Some(F::Bgra8Unorm)
		);
		assert_eq!(
			select_surface_format(&[F::Rgba8UnormSrgb, F::Rgba8Unorm]),
			Some(F::Rgba8Unorm)
		);
		assert_eq!(select_surface_format(&[F::Bgra8UnormSrgb]), Some(F::Bgra8UnormSrgb));
		assert_eq!(select_surface_format(&[]), None);
	}

	#[test]
	fn gpu_empty_layers_render_black() {
		let Some(ctx) = gpu() else { return };
		let mut renderer = MasterRenderer::new(&ctx).unwrap();
		let layers = [LayerRenderInput::EMPTY; LAYER_COUNT];
		let (report, pixels) = render(&ctx, &mut renderer, layers, &MasterEffects::default());
		assert_eq!(report.layers_composited, 0);
		assert_close(pixel(&pixels, 960, 540), [0, 0, 0, 255], 0);
		assert_close(pixel(&pixels, 0, 0), [0, 0, 0, 255], 0);
	}

	#[test]
	fn gpu_four_layers_and_upload_skip() {
		let Some(ctx) = gpu() else { return };
		let mut renderer = MasterRenderer::new(&ctx).unwrap();
		let red = solid_rgba(16, 9, [255, 0, 0, 255]);
		let green = solid_rgba(16, 9, [0, 255, 0, 255]);
		let blue = solid_rgba(32, 18, [0, 0, 255, 255]);
		let half = solid_rgba(8, 8, [255, 255, 255, 128]);
		let layers = [
			layer(&red, 1, BlendMode::Normal, true),
			layer(&green, 2, BlendMode::Add, true),
			layer(&blue, 3, BlendMode::Screen, true),
			layer(&half, 4, BlendMode::Multiply, false),
		];
		let (report, pixels) = render(&ctx, &mut renderer, layers, &MasterEffects::default());
		assert_eq!(report.layers_composited, 4);
		assert_eq!(report.uploads(), 4);
		// Red + green + blue screened = white; multiplying by white keeps white.
		assert_close(pixel(&pixels, 960, 540), [255, 255, 255, 255], 2);

		let (report, _) = render(&ctx, &mut renderer, layers, &MasterEffects::default());
		assert_eq!(report.uploads(), 0);
		assert!(report.layers.iter().all(|status| *status == LayerUploadStatus::Reused));

		// No new frame: the last frame of the same clip is held.
		let mut held = layers;
		held[0].frame = None;
		let (report, _) = render(&ctx, &mut renderer, held, &MasterEffects::default());
		assert_eq!(report.layers[0], LayerUploadStatus::Held);
	}

	#[test]
	fn gpu_all_blend_modes() {
		let Some(ctx) = gpu() else { return };
		let mut renderer = MasterRenderer::new(&ctx).unwrap();
		let base = solid_rgba(16, 9, [128, 128, 128, 255]);
		let top = solid_rgba(16, 9, [64, 64, 64, 255]);
		let cases = [
			(BlendMode::Normal, 64),
			(BlendMode::Add, 192),
			(BlendMode::Multiply, 32),
			(BlendMode::Screen, 160),
			(BlendMode::Difference, 64),
		];
		for (mode, expected) in cases {
			let layers = [
				layer(&base, 1, BlendMode::Normal, true),
				layer(&top, 2, mode, true),
				LayerRenderInput::EMPTY,
				LayerRenderInput::EMPTY,
			];
			let (_, pixels) = render(&ctx, &mut renderer, layers, &MasterEffects::default());
			assert_close(pixel(&pixels, 960, 540), [expected, expected, expected, 255], 2);
		}

		let mut faded = layer(&top, 2, BlendMode::Normal, true);
		faded.opacity = 0.5;
		let layers = [
			layer(&base, 1, BlendMode::Normal, true),
			faded,
			LayerRenderInput::EMPTY,
			LayerRenderInput::EMPTY,
		];
		let (_, pixels) = render(&ctx, &mut renderer, layers, &MasterEffects::default());
		assert_close(pixel(&pixels, 960, 540), [96, 96, 96, 255], 2);
	}

	#[test]
	fn gpu_aspect_margins_follow_opacity_mode() {
		let Some(ctx) = gpu() else { return };
		let mut renderer = MasterRenderer::new(&ctx).unwrap();
		let white = solid_rgba(16, 9, [255, 255, 255, 255]);
		let red_4_3 = solid_rgba(8, 6, [255, 0, 0, 255]);
		for (opaque, margin) in [(true, [0, 0, 0, 255]), (false, [255, 255, 255, 255])] {
			let layers = [
				layer(&white, 1, BlendMode::Normal, true),
				layer(&red_4_3, 2, BlendMode::Normal, opaque),
				LayerRenderInput::EMPTY,
				LayerRenderInput::EMPTY,
			];
			let (_, pixels) = render(&ctx, &mut renderer, layers, &MasterEffects::default());
			assert_close(pixel(&pixels, 960, 540), [255, 0, 0, 255], 1);
			assert_close(pixel(&pixels, 100, 540), margin, 1);
			assert_close(pixel(&pixels, 1820, 540), margin, 1);
		}
	}

	#[test]
	fn gpu_hap_bc1_and_hap_q_ycocg() {
		let Some(ctx) = gpu() else { return };
		let mut renderer = MasterRenderer::new(&ctx).unwrap();
		if !ctx.supports_bc {
			eprintln!("skipping BC test: adapter lacks TEXTURE_COMPRESSION_BC");
			return;
		}
		// 8x8 = 2x2 blocks. BC1 pure red (RGB565 0xF800), 4-color mode index 0.
		let bc1_block = [0x00, 0xF8, 0x00, 0x00, 0, 0, 0, 0];
		let bc1 = DecodedFrame {
			frame_index: 0,
			width: 7,
			height: 7,
			payload: PixelPayload::Bc1(bc1_block.repeat(4)),
		};
		let layers = [
			layer(&bc1, 1, BlendMode::Normal, true),
			LayerRenderInput::EMPTY,
			LayerRenderInput::EMPTY,
			LayerRenderInput::EMPTY,
		];
		let (report, pixels) = render(&ctx, &mut renderer, layers, &MasterEffects::default());
		assert_eq!(report.layers[0], LayerUploadStatus::Uploaded);
		assert_close(pixel(&pixels, 960, 540), [255, 0, 0, 255], 1);

		// HAP Q: Co = R, Cg = G, scale = B, Y = A. Neutral chroma around 128 with Y = 200.
		let color = (16u16 << 11) | (32u16 << 5);
		let ycocg = DecodedFrame {
			frame_index: 0,
			width: 8,
			height: 8,
			payload: PixelPayload::Bc3YCoCg(bc3_block(200, color).repeat(4)),
		};
		let layers = [
			layer(&ycocg, 2, BlendMode::Normal, true),
			LayerRenderInput::EMPTY,
			LayerRenderInput::EMPTY,
			LayerRenderInput::EMPTY,
		];
		let (_, pixels) = render(&ctx, &mut renderer, layers, &MasterEffects::default());
		// R ≈ Y + Co - Cg, G ≈ Y + Cg, B ≈ Y - Co - Cg (565 rounding shifts Co/Cg slightly).
		assert_close(pixel(&pixels, 960, 540), [202, 202, 194, 255], 4);
	}

	#[test]
	fn gpu_cpu_fallback_and_malformed_payloads() {
		let Some(ctx) = gpu() else { return };
		let mut renderer = MasterRenderer::new(&ctx).unwrap();
		fn green_decoder(frame: &DecodedFrame) -> Option<Vec<u8>> {
			Some([0u8, 255, 0, 255].repeat((frame.width * frame.height) as usize))
		}
		renderer.set_cpu_fallback_decoder(green_decoder);

		let truncated = DecodedFrame {
			frame_index: 0,
			width: 16,
			height: 9,
			payload: if ctx.supports_bc {
				PixelPayload::Bc3(vec![0; 10])
			} else {
				PixelPayload::Rgba8(vec![0; 10])
			},
		};
		let layers = [
			layer(&truncated, 1, BlendMode::Normal, true),
			LayerRenderInput::EMPTY,
			LayerRenderInput::EMPTY,
			LayerRenderInput::EMPTY,
		];
		let (report, pixels) = render(&ctx, &mut renderer, layers, &MasterEffects::default());
		assert!(matches!(
			report.layers[0],
			LayerUploadStatus::Rejected {
				reason: UploadRejection::LengthMismatch { .. },
				holding: false
			}
		));
		assert_close(pixel(&pixels, 960, 540), [0, 0, 0, 255], 0);

		if !ctx.supports_bc {
			let bc = DecodedFrame {
				frame_index: 0,
				width: 4,
				height: 4,
				payload: PixelPayload::Bc1(vec![0; 8]),
			};
			let layers = [
				layer(&bc, 2, BlendMode::Normal, true),
				LayerRenderInput::EMPTY,
				LayerRenderInput::EMPTY,
				LayerRenderInput::EMPTY,
			];
			let (_, pixels) = render(&ctx, &mut renderer, layers, &MasterEffects::default());
			assert_close(pixel(&pixels, 960, 540), [0, 255, 0, 255], 1);
		}
	}

	#[test]
	fn gpu_master_effects_dimmer_and_blackout() {
		let Some(ctx) = gpu() else { return };
		let mut renderer = MasterRenderer::new(&ctx).unwrap();
		let red = solid_rgba(16, 9, [255, 0, 0, 255]);
		let layers = [
			layer(&red, 1, BlendMode::Normal, true),
			LayerRenderInput::EMPTY,
			LayerRenderInput::EMPTY,
			LayerRenderInput::EMPTY,
		];

		let dimmed = MasterEffects {
			master_dimmer: 0.5,
			..MasterEffects::default()
		};
		let (_, pixels) = render(&ctx, &mut renderer, layers, &dimmed);
		assert_close(pixel(&pixels, 960, 540), [128, 0, 0, 255], 2);

		let desaturated = MasterEffects {
			saturation: 0.0,
			..MasterEffects::default()
		};
		let (_, pixels) = render(&ctx, &mut renderer, layers, &desaturated);
		let [r, g, b, _] = pixel(&pixels, 960, 540);
		assert!(r == g && g == b, "saturation 0 must be grey, got {r} {g} {b}");

		let all_on = MasterEffects {
			rgb_shift_enabled: true,
			rgb_shift_intensity: 1.0,
			glitch_enabled: true,
			glitch_intensity: 1.0,
			strobe_enabled: true,
			strobe_intensity: 1.0,
			brightness: 0.2,
			contrast: 1.5,
			..MasterEffects::default()
		};
		let (_, _) = render(&ctx, &mut renderer, layers, &all_on);

		let blackout = MasterEffects {
			blackout: true,
			..all_on
		};
		let (_, pixels) = render(&ctx, &mut renderer, layers, &blackout);
		assert!(pixels.as_chunks::<4>().0.iter().all(|px| *px == [0, 0, 0, 255]));
	}

	#[test]
	fn gpu_present_to_offscreen_targets() {
		let Some(ctx) = gpu() else { return };
		let mut renderer = MasterRenderer::new(&ctx).unwrap();
		let layers = [LayerRenderInput::EMPTY; LAYER_COUNT];
		renderer
			.render_master(&ctx, &layers, &MasterEffects::default(), 0.0)
			.unwrap();
		for (format, width, height) in [
			(wgpu::TextureFormat::Bgra8Unorm, 1280, 1024),
			(wgpu::TextureFormat::Bgra8UnormSrgb, 3840, 1600),
			(wgpu::TextureFormat::Rgba8Unorm, 640, 360),
		] {
			let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
				label: Some("test.present_target"),
				size: wgpu::Extent3d {
					width,
					height,
					depth_or_array_layers: 1,
				},
				mip_level_count: 1,
				sample_count: 1,
				dimension: wgpu::TextureDimension::D2,
				format,
				usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
				view_formats: &[],
			});
			let view = target.create_view(&wgpu::TextureViewDescriptor::default());
			renderer.present_surface(&ctx, &view, format, width, height).unwrap();
		}
		assert!(matches!(
			renderer.present_surface(&ctx, renderer.master_view(), INTERNAL_FORMAT, 0, 10),
			Err(RenderError::InvalidTarget(_))
		));
		ctx.device
			.poll(wgpu::PollType::Wait {
				submission_index: None,
				timeout: None,
			})
			.unwrap();
		assert_eq!(ctx.take_last_error(), None);
	}

	#[test]
	fn gpu_mirror_registers_once() {
		let Some(ctx) = gpu() else { return };
		let renderer = MasterRenderer::new(&ctx).unwrap();
		let mut egui_renderer = egui_wgpu::Renderer::new(
			&ctx.device,
			wgpu::TextureFormat::Bgra8Unorm,
			egui_wgpu::RendererOptions::default(),
		);
		let mut existing = None;
		let first = renderer.sync_mirror_texture(&ctx.device, &mut egui_renderer, &mut existing);
		let second = renderer.sync_mirror_texture(&ctx.device, &mut egui_renderer, &mut existing);
		assert_eq!(first, second);
		assert_eq!(existing, Some(first));
		assert_eq!(ctx.take_last_error(), None);
	}
}
