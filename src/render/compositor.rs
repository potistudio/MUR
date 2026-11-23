//! Four-layer ping-pong compositor producing the 1920x1080 master image.
//!
//! The compositor clears `target_a` to opaque black, then blends every
//! drawable layer in order by reading the previous target and writing the
//! other one. All uniform buffers and bind groups are created up front, so the
//! render loop performs no allocations of its own.

use bytemuck::{Pod, Zeroable};

use super::texture_cache::LayerTextureCache;
use super::{INTERNAL_FORMAT, INTERNAL_HEIGHT, INTERNAL_WIDTH};
use crate::error::RenderError;
use crate::model::BlendMode;
use crate::model::layer::LAYER_COUNT;

/// WGSL source of the layer compositing shader.
pub const LAYER_COMPOSITE_WGSL: &str = include_str!("shaders/layer_composite.wgsl");

/// How the compositing shader interprets a layer texture's texels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexelEncoding {
	/// Straight RGBA: the texel alpha is the layer alpha.
	Rgba,
	/// RGB without alpha (HAP RGB / BC1): the texel alpha is ignored.
	OpaqueRgb,
	/// DXT5-YCoCg (HAP Q), converted to opaque RGB in the shader.
	YCoCg,
}

/// Per-layer uniform block mirrored by `LayerUniform` in `layer_composite.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct LayerUniform {
	/// Fitted rectangle in output UV space: `[min_u, min_v, max_u, max_v]`.
	pub src_rect: [f32; 4],
	/// UV bound of the meaningful texels, cropping BC block padding.
	pub uv_max: [f32; 2],
	/// Layer opacity in `0.0..=1.0`.
	pub opacity: f32,
	/// Blend mode discriminator from [`BlendMode::as_u32`].
	pub blend_mode: u32,
	/// `1` for DXT5-YCoCg (HAP Q) textures, `0` otherwise.
	pub is_ycocg: u32,
	/// `1` when margins are opaque black, `0` when transparent.
	pub opaque_margin: u32,
	/// `1` when the texel alpha is ignored (HAP RGB / BC1), `0` otherwise.
	pub force_opaque: u32,
	/// Padding to a 16-byte multiple.
	pub pad: u32,
}

/// Parameters of one layer draw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerDraw {
	/// Index of the layer slot in the texture cache.
	pub slot: usize,
	/// Uniform values for the draw.
	pub uniform: LayerUniform,
}

impl LayerDraw {
	/// Builds the uniform for a layer of `content_size` pixels.
	pub fn new(
		slot: usize,
		content_size: (u32, u32),
		uv_max: [f32; 2],
		opacity: f32,
		blend_mode: BlendMode,
		encoding: TexelEncoding,
		opaque_margin: bool,
	) -> Self {
		let opacity = if opacity.is_finite() {
			opacity.clamp(0.0, 1.0)
		} else {
			0.0
		};
		Self {
			slot,
			uniform: LayerUniform {
				src_rect: super::fit_rect(content_size.0, content_size.1, INTERNAL_WIDTH, INTERNAL_HEIGHT),
				uv_max,
				opacity,
				blend_mode: blend_mode.as_u32(),
				is_ycocg: u32::from(encoding == TexelEncoding::YCoCg),
				opaque_margin: u32::from(opaque_margin),
				force_opaque: u32::from(encoding == TexelEncoding::OpaqueRgb),
				pad: 0,
			},
		}
	}
}

/// Which ping-pong target holds a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeTarget {
	/// The first ping-pong target.
	A,
	/// The second ping-pong target.
	B,
}

impl CompositeTarget {
	/// The other ping-pong target.
	pub fn other(self) -> Self {
		match self {
			Self::A => Self::B,
			Self::B => Self::A,
		}
	}
}

/// Preallocated uniform buffer and bind group of one layer.
#[derive(Debug)]
struct LayerUniformSlot {
	buffer: wgpu::Buffer,
	bind_group: wgpu::BindGroup,
}

/// GPU objects of one ping-pong target.
#[derive(Debug)]
struct PingPong {
	view: wgpu::TextureView,
	read_bind_group: wgpu::BindGroup,
}

/// Fullscreen layer compositor.
#[derive(Debug)]
pub struct LayerCompositor {
	pipeline: wgpu::RenderPipeline,
	layer_uniforms: [LayerUniformSlot; LAYER_COUNT],
	target_a: PingPong,
	target_b: PingPong,
	_fallback: wgpu::Texture,
	fallback_bind_group: wgpu::BindGroup,
}

impl LayerCompositor {
	/// Creates the pipeline, the ping-pong targets, and every per-layer resource.
	pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, cache: &LayerTextureCache) -> Result<Self, RenderError> {
		let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("mur.composite.uniform_layout"),
			entries: &[super::uniform_entry(0)],
		});
		let dst_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("mur.composite.dst_layout"),
			entries: &[super::texture_entry(0)],
		});
		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("mur.composite.pipeline_layout"),
			bind_group_layouts: &[
				Some(&uniform_layout),
				Some(&dst_layout),
				Some(cache.bind_group_layout()),
			],
			immediate_size: 0,
		});
		let pipeline = super::create_fullscreen_pipeline(
			device,
			"mur.composite",
			LAYER_COMPOSITE_WGSL,
			&pipeline_layout,
			INTERNAL_FORMAT,
		)?;

		let layer_uniforms: [LayerUniformSlot; LAYER_COUNT] = std::array::from_fn(|_| {
			let buffer = device.create_buffer(&wgpu::BufferDescriptor {
				label: Some("mur.composite.uniform"),
				size: std::mem::size_of::<LayerUniform>() as u64,
				usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
				mapped_at_creation: false,
			});
			let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
				label: Some("mur.composite.uniform_bind_group"),
				layout: &uniform_layout,
				entries: &[wgpu::BindGroupEntry {
					binding: 0,
					resource: buffer.as_entire_binding(),
				}],
			});
			LayerUniformSlot { buffer, bind_group }
		});

		let target_a = create_ping_pong(device, &dst_layout, "mur.composite.target_a");
		let target_b = create_ping_pong(device, &dst_layout, "mur.composite.target_b");

		let fallback = device.create_texture(&wgpu::TextureDescriptor {
			label: Some("mur.composite.fallback_black"),
			size: wgpu::Extent3d {
				width: 1,
				height: 1,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::Rgba8Unorm,
			usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
			view_formats: &[],
		});
		queue.write_texture(
			wgpu::TexelCopyTextureInfo {
				texture: &fallback,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			&[0, 0, 0, 255],
			wgpu::TexelCopyBufferLayout {
				offset: 0,
				bytes_per_row: Some(4),
				rows_per_image: Some(1),
			},
			wgpu::Extent3d {
				width: 1,
				height: 1,
				depth_or_array_layers: 1,
			},
		);
		let fallback_view = fallback.create_view(&wgpu::TextureViewDescriptor::default());
		let fallback_bind_group = cache.create_bind_group(device, &fallback_view, "mur.composite.fallback_bind_group");

		Ok(Self {
			pipeline,
			layer_uniforms,
			target_a,
			target_b,
			_fallback: fallback,
			fallback_bind_group,
		})
	}

	/// View of a ping-pong target.
	pub fn target_view(&self, target: CompositeTarget) -> &wgpu::TextureView {
		match target {
			CompositeTarget::A => &self.target_a.view,
			CompositeTarget::B => &self.target_b.view,
		}
	}

	/// Encodes the clear pass followed by one blend pass per draw.
	///
	/// `draws` must be ordered bottom layer first. Returns the target holding
	/// the final composite.
	pub fn encode(
		&self,
		queue: &wgpu::Queue,
		encoder: &mut wgpu::CommandEncoder,
		cache: &LayerTextureCache,
		draws: &[Option<LayerDraw>; LAYER_COUNT],
	) -> CompositeTarget {
		{
			let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("mur.composite.clear"),
				color_attachments: &[Some(super::color_attachment(&self.target_a.view))],
				depth_stencil_attachment: None,
				timestamp_writes: None,
				occlusion_query_set: None,
				multiview_mask: None,
			});
		}

		let mut current = CompositeTarget::A;
		for (index, draw) in draws.iter().enumerate() {
			let Some(draw) = draw else { continue };
			if draw.uniform.opacity <= 0.0 {
				continue;
			}
			let Some(uniforms) = self.layer_uniforms.get(index) else {
				continue;
			};
			let src_bind_group = cache
				.slot(draw.slot)
				.and_then(|slot| slot.bind_group())
				.unwrap_or(&self.fallback_bind_group);
			queue.write_buffer(&uniforms.buffer, 0, bytemuck::bytes_of(&draw.uniform));

			let read = self.ping_pong(current);
			let next = current.other();
			let write = self.ping_pong(next);
			let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("mur.composite.layer"),
				color_attachments: &[Some(super::color_attachment(&write.view))],
				depth_stencil_attachment: None,
				timestamp_writes: None,
				occlusion_query_set: None,
				multiview_mask: None,
			});
			pass.set_pipeline(&self.pipeline);
			pass.set_bind_group(0, &uniforms.bind_group, &[]);
			pass.set_bind_group(1, &read.read_bind_group, &[]);
			pass.set_bind_group(2, src_bind_group, &[]);
			pass.draw(0..3, 0..1);
			current = next;
		}
		current
	}

	fn ping_pong(&self, target: CompositeTarget) -> &PingPong {
		match target {
			CompositeTarget::A => &self.target_a,
			CompositeTarget::B => &self.target_b,
		}
	}
}

/// Creates one 1920x1080 ping-pong target and its read bind group.
fn create_ping_pong(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, label: &str) -> PingPong {
	let texture = super::create_internal_target(device, label);
	let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
	let read_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
		label: Some(label),
		layout,
		entries: &[wgpu::BindGroupEntry {
			binding: 0,
			resource: wgpu::BindingResource::TextureView(&view),
		}],
	});
	PingPong { view, read_bind_group }
}
