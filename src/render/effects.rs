//! Master shader effects: glitch, RGB shift, color grading, strobe, dimmer, blackout.
//!
//! Reads the final composite (one of the compositor's ping-pong targets) and
//! renders it into the 1920x1080 `master_output` texture, which is what the
//! presenter, the egui mirror and readback consume.

use bytemuck::{Pod, Zeroable};

use super::compositor::{CompositeTarget, LayerCompositor};
use super::{INTERNAL_FORMAT, INTERNAL_HEIGHT, INTERNAL_WIDTH};
use crate::error::RenderError;
use crate::model::MasterEffects;

/// WGSL source of the master effects shader.
pub const MASTER_EFFECTS_WGSL: &str = include_str!("shaders/master_effects.wgsl");

/// Period in seconds after which shader time wraps to keep `f32` precision.
pub const TIME_WRAP_SECS: f32 = 3600.0;

/// Uniform block mirrored by `EffectsUniform` in `master_effects.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct EffectsUniform {
	/// RGB shift strength, `0.0` when disabled.
	pub rgb_shift: f32,
	/// Glitch strength, `0.0` when disabled.
	pub glitch: f32,
	/// Strobe/invert strength, `0.0` when disabled.
	pub strobe: f32,
	/// Strobe frequency in hertz.
	pub strobe_speed_hz: f32,
	/// Brightness offset.
	pub brightness: f32,
	/// Contrast gain.
	pub contrast: f32,
	/// Saturation gain.
	pub saturation: f32,
	/// Master output gain.
	pub master_dimmer: f32,
	/// Animation time in seconds, wrapped at [`TIME_WRAP_SECS`].
	pub time_secs: f32,
	/// `1` forces opaque black output.
	pub blackout: u32,
	/// Padding to a 16-byte multiple.
	pub pad: [u32; 2],
}

impl EffectsUniform {
	/// Converts effect parameters into shader values, sanitizing every input.
	pub fn from_effects(effects: &MasterEffects, time_secs: f32) -> Self {
		let mut fx = effects.clone();
		fx.clamp_all();
		let gated = |enabled: bool, value: f32| if enabled { value } else { 0.0 };
		let time_secs = if time_secs.is_finite() {
			time_secs.rem_euclid(TIME_WRAP_SECS)
		} else {
			0.0
		};
		Self {
			rgb_shift: gated(fx.rgb_shift_enabled, fx.rgb_shift_intensity),
			glitch: gated(fx.glitch_enabled, fx.glitch_intensity),
			strobe: gated(fx.strobe_enabled, fx.strobe_intensity),
			strobe_speed_hz: fx.strobe_speed_hz,
			brightness: fx.brightness,
			contrast: fx.contrast,
			saturation: fx.saturation,
			master_dimmer: fx.master_dimmer,
			time_secs,
			blackout: u32::from(fx.blackout),
			pad: [0; 2],
		}
	}
}

/// Fullscreen master effects pass owning the `master_output` texture.
#[derive(Debug)]
pub struct MasterEffectsPass {
	pipeline: wgpu::RenderPipeline,
	uniform_buffer: wgpu::Buffer,
	bind_group_a: wgpu::BindGroup,
	bind_group_b: wgpu::BindGroup,
	output: wgpu::Texture,
	output_view: wgpu::TextureView,
}

impl MasterEffectsPass {
	/// Creates the pipeline, the output texture, and bind groups for both
	/// compositor targets.
	pub fn new(device: &wgpu::Device, compositor: &LayerCompositor) -> Result<Self, RenderError> {
		let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("mur.effects.layout"),
			entries: &[
				super::uniform_entry(0),
				super::texture_entry(1),
				super::sampler_entry(2),
			],
		});
		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("mur.effects.pipeline_layout"),
			bind_group_layouts: &[Some(&layout)],
			immediate_size: 0,
		});
		let pipeline = super::create_fullscreen_pipeline(
			device,
			"mur.effects",
			MASTER_EFFECTS_WGSL,
			&pipeline_layout,
			INTERNAL_FORMAT,
		)?;
		let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("mur.effects.uniform"),
			size: std::mem::size_of::<EffectsUniform>() as u64,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let sampler = super::linear_clamp_sampler(device, "mur.effects.sampler");
		let make_bind_group = |view: &wgpu::TextureView, label: &str| {
			device.create_bind_group(&wgpu::BindGroupDescriptor {
				label: Some(label),
				layout: &layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						resource: uniform_buffer.as_entire_binding(),
					},
					wgpu::BindGroupEntry {
						binding: 1,
						resource: wgpu::BindingResource::TextureView(view),
					},
					wgpu::BindGroupEntry {
						binding: 2,
						resource: wgpu::BindingResource::Sampler(&sampler),
					},
				],
			})
		};
		let bind_group_a = make_bind_group(compositor.target_view(CompositeTarget::A), "mur.effects.read_a");
		let bind_group_b = make_bind_group(compositor.target_view(CompositeTarget::B), "mur.effects.read_b");
		let output = super::create_internal_target(device, "mur.master_output");
		let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
		Ok(Self {
			pipeline,
			uniform_buffer,
			bind_group_a,
			bind_group_b,
			output,
			output_view,
		})
	}

	/// The 1920x1080 `Rgba8Unorm` master output texture.
	pub fn output(&self) -> &wgpu::Texture {
		&self.output
	}

	/// View of the master output texture.
	pub fn output_view(&self) -> &wgpu::TextureView {
		&self.output_view
	}

	/// Output dimensions in pixels.
	pub fn output_size(&self) -> (u32, u32) {
		(INTERNAL_WIDTH, INTERNAL_HEIGHT)
	}

	/// Writes `uniform` and encodes the effects pass reading `input`.
	pub fn encode(
		&self,
		queue: &wgpu::Queue,
		encoder: &mut wgpu::CommandEncoder,
		input: CompositeTarget,
		uniform: &EffectsUniform,
	) {
		queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniform));
		let bind_group = match input {
			CompositeTarget::A => &self.bind_group_a,
			CompositeTarget::B => &self.bind_group_b,
		};
		let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
			label: Some("mur.effects"),
			color_attachments: &[Some(super::color_attachment(&self.output_view))],
			depth_stencil_attachment: None,
			timestamp_writes: None,
			occlusion_query_set: None,
			multiview_mask: None,
		});
		pass.set_pipeline(&self.pipeline);
		pass.set_bind_group(0, bind_group, &[]);
		pass.draw(0..3, 0..1);
	}
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;

	#[test]
	fn disabled_effects_are_zeroed() {
		let fx = MasterEffects {
			rgb_shift_intensity: 0.8,
			glitch_intensity: 0.5,
			..MasterEffects::default()
		};
		let uniform = EffectsUniform::from_effects(&fx, 1.0);
		assert_eq!(uniform.rgb_shift, 0.0);
		assert_eq!(uniform.glitch, 0.0);
		assert_eq!(uniform.master_dimmer, 1.0);
		assert_eq!(uniform.blackout, 0);
	}

	#[test]
	fn sanitizes_values_and_time() {
		let fx = MasterEffects {
			strobe_enabled: true,
			strobe_intensity: f32::NAN,
			master_dimmer: 4.0,
			blackout: true,
			..MasterEffects::default()
		};
		let uniform = EffectsUniform::from_effects(&fx, f32::INFINITY);
		assert_eq!(uniform.strobe, 0.0);
		assert_eq!(uniform.master_dimmer, 1.0);
		assert_eq!(uniform.time_secs, 0.0);
		assert_eq!(uniform.blackout, 1);
		let wrapped = EffectsUniform::from_effects(&fx, TIME_WRAP_SECS + 2.5);
		assert!((wrapped.time_secs - 2.5).abs() < 1e-3);
	}

	#[test]
	fn uniform_is_16_byte_aligned() {
		assert_eq!(std::mem::size_of::<EffectsUniform>() % 16, 0);
	}
}
