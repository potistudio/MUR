//! Aspect-ratio-preserving blit of the master output onto a window surface.
//!
//! The 1920x1080 master image is scaled into the largest centered 16:9
//! rectangle of the target and the remaining area is filled with opaque black.
//! Pipelines are built lazily per target format, so a surface that changes
//! format (for example after moving to another display) keeps presenting.

use std::sync::{Mutex, PoisonError};

use bytemuck::{Pod, Zeroable};

use super::{INTERNAL_HEIGHT, INTERNAL_WIDTH};
use crate::error::RenderError;

/// WGSL source of the presentation shader.
pub const PRESENT_WGSL: &str = include_str!("shaders/present.wgsl");

/// Uniform block mirrored by `PresentUniform` in `present.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct PresentUniform {
	/// Fitted rectangle in target UV space: `[min_u, min_v, max_u, max_v]`.
	pub dst_rect: [f32; 4],
	/// `1` when the target is an sRGB format that expects linear values.
	pub to_linear: u32,
	/// Padding to a 16-byte multiple.
	pub pad: [u32; 3],
}

impl PresentUniform {
	/// Builds the uniform for a `width`x`height` target of `format`.
	pub fn new(format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
		Self {
			dst_rect: super::fit_rect(INTERNAL_WIDTH, INTERNAL_HEIGHT, width, height),
			to_linear: u32::from(format.is_srgb()),
			pad: [0; 3],
		}
	}
}

/// Blits the master output onto arbitrary render targets.
#[derive(Debug)]
pub struct SurfacePresenter {
	shader: wgpu::ShaderModule,
	pipeline_layout: wgpu::PipelineLayout,
	uniform_buffer: wgpu::Buffer,
	bind_group: wgpu::BindGroup,
	pipelines: Mutex<Vec<(wgpu::TextureFormat, wgpu::RenderPipeline)>>,
}

impl SurfacePresenter {
	/// Creates the presenter reading from `master_view`.
	pub fn new(device: &wgpu::Device, master_view: &wgpu::TextureView) -> Result<Self, RenderError> {
		let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("mur.present.layout"),
			entries: &[
				super::uniform_entry(0),
				super::texture_entry(1),
				super::sampler_entry(2),
			],
		});
		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("mur.present.pipeline_layout"),
			bind_group_layouts: &[Some(&layout)],
			immediate_size: 0,
		});
		let shader = super::create_shader(device, "mur.present", PRESENT_WGSL)?;
		let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("mur.present.uniform"),
			size: std::mem::size_of::<PresentUniform>() as u64,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let sampler = super::linear_clamp_sampler(device, "mur.present.sampler");
		let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("mur.present.bind_group"),
			layout: &layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: uniform_buffer.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::TextureView(master_view),
				},
				wgpu::BindGroupEntry {
					binding: 2,
					resource: wgpu::BindingResource::Sampler(&sampler),
				},
			],
		});
		Ok(Self {
			shader,
			pipeline_layout,
			uniform_buffer,
			bind_group,
			pipelines: Mutex::new(Vec::with_capacity(4)),
		})
	}

	/// Returns the cached pipeline for `format`, building it on first use.
	pub fn pipeline_for(
		&self,
		device: &wgpu::Device,
		format: wgpu::TextureFormat,
	) -> Result<wgpu::RenderPipeline, RenderError> {
		let mut pipelines = self.pipelines.lock().unwrap_or_else(PoisonError::into_inner);
		if let Some((_, pipeline)) = pipelines.iter().find(|(cached, _)| *cached == format) {
			return Ok(pipeline.clone());
		}
		let pipeline = super::build_pipeline(device, "mur.present", &self.shader, &self.pipeline_layout, format)?;
		pipelines.push((format, pipeline.clone()));
		Ok(pipeline)
	}

	/// Number of target formats with a cached pipeline.
	pub fn cached_format_count(&self) -> usize {
		self.pipelines.lock().unwrap_or_else(PoisonError::into_inner).len()
	}

	/// Encodes and submits the blit of the master output into `target_view`.
	pub fn present(
		&self,
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		target_view: &wgpu::TextureView,
		target_format: wgpu::TextureFormat,
		target_width: u32,
		target_height: u32,
	) -> Result<(), RenderError> {
		if target_width == 0 || target_height == 0 {
			return Err(RenderError::InvalidTarget(format!(
				"present target has zero size ({target_width}x{target_height})"
			)));
		}
		let pipeline = self.pipeline_for(device, target_format)?;
		let uniform = PresentUniform::new(target_format, target_width, target_height);
		queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));

		let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("mur.present.encoder"),
		});
		{
			let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("mur.present"),
				color_attachments: &[Some(super::color_attachment(target_view))],
				depth_stencil_attachment: None,
				timestamp_writes: None,
				occlusion_query_set: None,
				multiview_mask: None,
			});
			pass.set_pipeline(&pipeline);
			pass.set_bind_group(0, &self.bind_group, &[]);
			pass.draw(0..3, 0..1);
		}
		queue.submit(std::iter::once(encoder.finish()));
		Ok(())
	}
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;

	#[test]
	fn uniform_flags_srgb_targets() {
		assert_eq!(
			PresentUniform::new(wgpu::TextureFormat::Bgra8UnormSrgb, 1920, 1080).to_linear,
			1
		);
		assert_eq!(
			PresentUniform::new(wgpu::TextureFormat::Bgra8Unorm, 1920, 1080).to_linear,
			0
		);
	}

	#[test]
	fn letterbox_on_4_3_target() {
		let uniform = PresentUniform::new(wgpu::TextureFormat::Bgra8Unorm, 1600, 1200);
		let [min_u, min_v, max_u, max_v] = uniform.dst_rect;
		assert!((min_u - 0.0).abs() < 1e-6 && (max_u - 1.0).abs() < 1e-6);
		assert!((min_v - 0.125).abs() < 1e-6 && (max_v - 0.875).abs() < 1e-6);
	}
}
