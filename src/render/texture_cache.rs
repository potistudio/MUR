//! Per-layer GPU texture slots for decoded clip frames.
//!
//! Each of the four layers owns one [`LayerTextureSlot`]. Frames are uploaded
//! as compressed BC1/BC3 blocks when the device supports
//! [`wgpu::Features::TEXTURE_COMPRESSION_BC`], or as RGBA8 otherwise. Uploads
//! are skipped when the `(clip, frame, size, format)` key is unchanged, and
//! payload lengths are validated before `queue.write_texture` so malformed
//! data never reaches wgpu validation.

use std::borrow::Cow;

use crate::model::layer::LAYER_COUNT;
use crate::model::{ClipId, DecodedFrame, PixelPayload};

/// CPU decoder used when the GPU cannot sample BC textures.
///
/// The callback must return tightly packed, display-ready RGBA8 pixels of
/// exactly `width * height * 4` bytes (HAP Q frames already converted from
/// YCoCg to RGB), or `None` when the frame cannot be decoded.
pub type CpuFallbackDecoder = fn(&DecodedFrame) -> Option<Vec<u8>>;

/// Bytes per 4x4 block of BC1 data.
const BC1_BLOCK_BYTES: u32 = 8;
/// Bytes per 4x4 block of BC3 data.
const BC3_BLOCK_BYTES: u32 = 16;
/// Bytes per RGBA8 texel.
const RGBA8_TEXEL_BYTES: u32 = 4;

/// Identity of the content currently stored in a slot, used to skip uploads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UploadKey {
	/// Clip the frame belongs to; `None` disables upload skipping.
	pub clip_id: Option<ClipId>,
	/// Zero-based index of the frame within the clip.
	pub frame_index: u32,
	/// Frame width in pixels.
	pub width: u32,
	/// Frame height in pixels.
	pub height: u32,
	/// GPU texture format the frame was uploaded with.
	pub format: wgpu::TextureFormat,
	/// Whether the texels hold DXT5-YCoCg data.
	pub is_ycocg: bool,
}

/// Reason a frame was refused before reaching the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadRejection {
	/// The frame declared a zero width or height.
	ZeroSize,
	/// The frame exceeds the device's maximum 2D texture dimension.
	TooLarge {
		/// Largest supported edge length in texels.
		max_dimension: u32,
	},
	/// The payload length does not match the declared geometry.
	LengthMismatch {
		/// Number of bytes the geometry requires.
		expected: usize,
		/// Number of bytes actually provided.
		actual: usize,
	},
	/// BC data was supplied but the device cannot sample BC textures and no
	/// CPU fallback decoder is installed.
	BcUnsupported,
	/// The CPU fallback decoder failed to produce pixels.
	FallbackDecodeFailed,
}

impl std::fmt::Display for UploadRejection {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::ZeroSize => write!(f, "frame has zero width or height"),
			Self::TooLarge { max_dimension } => write!(f, "frame exceeds the {max_dimension}px GPU texture limit"),
			Self::LengthMismatch { expected, actual } => {
				write!(f, "payload is {actual} bytes but {expected} bytes were expected")
			}
			Self::BcUnsupported => write!(f, "GPU lacks BC texture support and no CPU fallback is installed"),
			Self::FallbackDecodeFailed => write!(f, "CPU fallback decoder failed"),
		}
	}
}

/// Outcome of preparing one layer's texture for the current frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerUploadStatus {
	/// The layer has nothing to draw this frame.
	Inactive,
	/// A new frame was written to the GPU.
	Uploaded,
	/// The frame was already resident; the upload was skipped.
	Reused,
	/// No new frame arrived, so the last frame of the same clip is held.
	Held,
	/// The frame was refused before upload.
	Rejected {
		/// Why the frame was refused.
		reason: UploadRejection,
		/// Whether the previous frame of the same clip is still shown.
		holding: bool,
	},
}

impl LayerUploadStatus {
	/// Returns `true` when the slot holds content that should be composited.
	pub fn is_drawable(self) -> bool {
		match self {
			Self::Uploaded | Self::Reused | Self::Held => true,
			Self::Rejected { holding, .. } => holding,
			Self::Inactive => false,
		}
	}
}

/// Texture geometry and layout derived from a decoded frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UploadPlan {
	/// GPU texture format.
	pub format: wgpu::TextureFormat,
	/// Allocated texture width, padded to whole blocks for BC formats.
	pub alloc_width: u32,
	/// Allocated texture height, padded to whole blocks for BC formats.
	pub alloc_height: u32,
	/// Bytes per row of texels (or of 4x4 blocks for BC formats).
	pub bytes_per_row: u32,
	/// Rows (or block rows) per image.
	pub rows_per_image: u32,
	/// Exact payload byte length required by the geometry.
	pub expected_len: usize,
	/// UV bound of the meaningful texels, excluding block padding.
	pub uv_max: [f32; 2],
	/// Whether the texels hold DXT5-YCoCg data.
	pub is_ycocg: bool,
	/// Whether the payload must be decoded to RGBA8 on the CPU first.
	pub needs_cpu_decode: bool,
}

/// Computes the upload layout for `frame` without touching the GPU.
///
/// Returns an [`UploadRejection`] for zero-sized or oversized frames. The
/// payload length itself is validated separately once the bytes that will
/// actually be uploaded are known.
pub fn plan_upload(frame: &DecodedFrame, supports_bc: bool, max_dimension: u32) -> Result<UploadPlan, UploadRejection> {
	let (width, height) = (frame.width, frame.height);
	if width == 0 || height == 0 {
		return Err(UploadRejection::ZeroSize);
	}
	if width > max_dimension || height > max_dimension {
		return Err(UploadRejection::TooLarge { max_dimension });
	}

	let block = match &frame.payload {
		PixelPayload::Bc1(_) => Some((wgpu::TextureFormat::Bc1RgbaUnorm, BC1_BLOCK_BYTES, false)),
		PixelPayload::Bc3(_) => Some((wgpu::TextureFormat::Bc3RgbaUnorm, BC3_BLOCK_BYTES, false)),
		PixelPayload::Bc3YCoCg(_) => Some((wgpu::TextureFormat::Bc3RgbaUnorm, BC3_BLOCK_BYTES, true)),
		PixelPayload::Rgba8(_) => None,
	};

	match block {
		Some((format, block_bytes, is_ycocg)) if supports_bc => {
			let padded_w = width.div_ceil(4).saturating_mul(4);
			let padded_h = height.div_ceil(4).saturating_mul(4);
			if padded_w > max_dimension || padded_h > max_dimension {
				return Err(UploadRejection::TooLarge { max_dimension });
			}
			let bytes_per_row = (padded_w / 4).saturating_mul(block_bytes);
			let rows_per_image = padded_h / 4;
			Ok(UploadPlan {
				format,
				alloc_width: padded_w,
				alloc_height: padded_h,
				bytes_per_row,
				rows_per_image,
				expected_len: (bytes_per_row as usize).saturating_mul(rows_per_image as usize),
				uv_max: [width as f32 / padded_w as f32, height as f32 / padded_h as f32],
				is_ycocg,
				needs_cpu_decode: false,
			})
		}
		other => {
			let bytes_per_row = width.saturating_mul(RGBA8_TEXEL_BYTES);
			Ok(UploadPlan {
				format: wgpu::TextureFormat::Rgba8Unorm,
				alloc_width: width,
				alloc_height: height,
				bytes_per_row,
				rows_per_image: height,
				expected_len: (bytes_per_row as usize).saturating_mul(height as usize),
				uv_max: [1.0, 1.0],
				is_ycocg: false,
				needs_cpu_decode: other.is_some(),
			})
		}
	}
}

/// GPU objects backing one slot.
#[derive(Debug)]
struct SlotTexture {
	texture: wgpu::Texture,
	bind_group: wgpu::BindGroup,
	width: u32,
	height: u32,
	format: wgpu::TextureFormat,
}

/// GPU texture slot for one layer.
#[derive(Debug, Default)]
pub struct LayerTextureSlot {
	gpu: Option<SlotTexture>,
	content: Option<UploadKey>,
	uv_max: [f32; 2],
	content_width: u32,
	content_height: u32,
}

impl LayerTextureSlot {
	/// Key of the frame currently resident in the slot, if any.
	pub fn content_key(&self) -> Option<UploadKey> {
		self.content
	}

	/// UV bound of the meaningful texels of the resident frame.
	pub fn uv_max(&self) -> [f32; 2] {
		self.uv_max
	}

	/// Whether the resident frame holds DXT5-YCoCg data.
	pub fn is_ycocg(&self) -> bool {
		self.content.is_some_and(|key| key.is_ycocg)
	}

	/// Pixel size of the resident frame, excluding block padding.
	pub fn content_size(&self) -> (u32, u32) {
		(self.content_width, self.content_height)
	}

	/// Cached texture/sampler bind group, available once a texture exists.
	pub fn bind_group(&self) -> Option<&wgpu::BindGroup> {
		self.gpu.as_ref().map(|gpu| &gpu.bind_group)
	}

	/// Whether the slot holds a frame of `clip_id`.
	fn holds_clip(&self, clip_id: Option<ClipId>) -> bool {
		clip_id.is_some() && self.content.is_some_and(|key| key.clip_id == clip_id) && self.gpu.is_some()
	}

	/// Forgets the resident frame while keeping the texture allocation.
	fn clear_content(&mut self) {
		self.content = None;
	}
}

/// GPU resources shared by every slot.
pub struct UploadContext<'a> {
	/// Device used to (re)create textures and bind groups.
	pub device: &'a wgpu::Device,
	/// Queue receiving texture writes.
	pub queue: &'a wgpu::Queue,
	/// Whether BC textures can be sampled.
	pub supports_bc: bool,
	/// Optional CPU decoder for BC payloads on devices without BC support.
	pub cpu_fallback: Option<CpuFallbackDecoder>,
}

/// Four per-layer texture slots plus the bind group layout they share.
#[derive(Debug)]
pub struct LayerTextureCache {
	slots: [LayerTextureSlot; LAYER_COUNT],
	layout: wgpu::BindGroupLayout,
	sampler: wgpu::Sampler,
}

impl LayerTextureCache {
	/// Creates empty slots and the shared texture bind group layout.
	pub fn new(device: &wgpu::Device) -> Self {
		let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("mur.layer_src.layout"),
			entries: &[super::texture_entry(0), super::sampler_entry(1)],
		});
		let sampler = super::linear_clamp_sampler(device, "mur.layer_src.sampler");
		Self {
			slots: Default::default(),
			layout,
			sampler,
		}
	}

	/// Bind group layout of a slot: binding 0 texture, binding 1 sampler.
	pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
		&self.layout
	}

	/// Builds a texture/sampler bind group compatible with the slot layout.
	pub fn create_bind_group(&self, device: &wgpu::Device, view: &wgpu::TextureView, label: &str) -> wgpu::BindGroup {
		device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some(label),
			layout: &self.layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: wgpu::BindingResource::TextureView(view),
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::Sampler(&self.sampler),
				},
			],
		})
	}

	/// Slot for layer `index`, if the index is in range.
	pub fn slot(&self, index: usize) -> Option<&LayerTextureSlot> {
		self.slots.get(index)
	}

	/// Prepares layer `index` for compositing and reports what happened.
	///
	/// * A new `frame` is uploaded unless the same key is already resident.
	/// * Without a frame, the last frame of the same `clip_id` is held.
	/// * A rejected frame of the same clip keeps holding the previous frame.
	pub fn prepare(
		&mut self,
		ctx: &UploadContext<'_>,
		index: usize,
		frame: Option<&DecodedFrame>,
		clip_id: Option<ClipId>,
	) -> LayerUploadStatus {
		let max_dimension = ctx.device.limits().max_texture_dimension_2d;
		let Some(slot) = self.slots.get(index) else {
			return LayerUploadStatus::Inactive;
		};
		let holding = slot.holds_clip(clip_id);

		let Some(frame) = frame else {
			if holding {
				return LayerUploadStatus::Held;
			}
			if let Some(slot) = self.slots.get_mut(index) {
				slot.clear_content();
			}
			return LayerUploadStatus::Inactive;
		};

		match self.upload(ctx, index, frame, clip_id, max_dimension) {
			Ok(status) => status,
			Err(reason) => {
				if !holding && let Some(slot) = self.slots.get_mut(index) {
					slot.clear_content();
				}
				LayerUploadStatus::Rejected { reason, holding }
			}
		}
	}

	/// Validates and uploads `frame` into slot `index`.
	fn upload(
		&mut self,
		ctx: &UploadContext<'_>,
		index: usize,
		frame: &DecodedFrame,
		clip_id: Option<ClipId>,
		max_dimension: u32,
	) -> Result<LayerUploadStatus, UploadRejection> {
		let plan = plan_upload(frame, ctx.supports_bc, max_dimension)?;
		let key = UploadKey {
			clip_id,
			frame_index: frame.frame_index,
			width: frame.width,
			height: frame.height,
			format: plan.format,
			is_ycocg: plan.is_ycocg,
		};

		let resident = self
			.slots
			.get(index)
			.is_some_and(|slot| clip_id.is_some() && slot.content == Some(key) && slot.gpu.is_some());
		if resident {
			return Ok(LayerUploadStatus::Reused);
		}

		let bytes: Cow<'_, [u8]> = if plan.needs_cpu_decode {
			let decoder = ctx.cpu_fallback.ok_or(UploadRejection::BcUnsupported)?;
			Cow::Owned(decoder(frame).ok_or(UploadRejection::FallbackDecodeFailed)?)
		} else {
			Cow::Borrowed(payload_bytes(&frame.payload))
		};
		if bytes.len() != plan.expected_len {
			return Err(UploadRejection::LengthMismatch {
				expected: plan.expected_len,
				actual: bytes.len(),
			});
		}

		self.ensure_texture(ctx.device, index, &plan);
		let Some(slot) = self.slots.get_mut(index) else {
			return Ok(LayerUploadStatus::Inactive);
		};
		let Some(gpu) = slot.gpu.as_ref() else {
			return Ok(LayerUploadStatus::Inactive);
		};

		ctx.queue.write_texture(
			wgpu::TexelCopyTextureInfo {
				texture: &gpu.texture,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			&bytes,
			wgpu::TexelCopyBufferLayout {
				offset: 0,
				bytes_per_row: Some(plan.bytes_per_row),
				rows_per_image: Some(plan.rows_per_image),
			},
			wgpu::Extent3d {
				width: plan.alloc_width,
				height: plan.alloc_height,
				depth_or_array_layers: 1,
			},
		);

		slot.content = Some(key);
		slot.uv_max = plan.uv_max;
		slot.content_width = frame.width;
		slot.content_height = frame.height;
		Ok(LayerUploadStatus::Uploaded)
	}

	/// (Re)creates the slot texture when its size or format changed.
	fn ensure_texture(&mut self, device: &wgpu::Device, index: usize, plan: &UploadPlan) {
		let matches = self
			.slots
			.get(index)
			.and_then(|slot| slot.gpu.as_ref())
			.is_some_and(|gpu| {
				gpu.width == plan.alloc_width && gpu.height == plan.alloc_height && gpu.format == plan.format
			});
		if matches {
			return;
		}

		let texture = device.create_texture(&wgpu::TextureDescriptor {
			label: Some("mur.layer_src.texture"),
			size: wgpu::Extent3d {
				width: plan.alloc_width,
				height: plan.alloc_height,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: plan.format,
			usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
			view_formats: &[],
		});
		let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
		let bind_group = self.create_bind_group(device, &view, "mur.layer_src.bind_group");
		if let Some(slot) = self.slots.get_mut(index) {
			if let Some(old) = slot.gpu.take() {
				old.texture.destroy();
			}
			slot.gpu = Some(SlotTexture {
				texture,
				bind_group,
				width: plan.alloc_width,
				height: plan.alloc_height,
				format: plan.format,
			});
			slot.content = None;
		}
	}
}

/// Raw bytes of a payload regardless of its encoding.
fn payload_bytes(payload: &PixelPayload) -> &[u8] {
	match payload {
		PixelPayload::Bc1(bytes)
		| PixelPayload::Bc3(bytes)
		| PixelPayload::Bc3YCoCg(bytes)
		| PixelPayload::Rgba8(bytes) => bytes,
	}
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;

	fn frame(width: u32, height: u32, payload: PixelPayload) -> DecodedFrame {
		DecodedFrame {
			frame_index: 0,
			width,
			height,
			payload,
		}
	}

	#[test]
	fn bc1_plan_pads_to_blocks() {
		let plan = plan_upload(&frame(1918, 1078, PixelPayload::Bc1(Vec::new())), true, 8192).unwrap();
		assert_eq!(plan.format, wgpu::TextureFormat::Bc1RgbaUnorm);
		assert_eq!((plan.alloc_width, plan.alloc_height), (1920, 1080));
		assert_eq!(plan.bytes_per_row, 480 * 8);
		assert_eq!(plan.rows_per_image, 270);
		assert_eq!(plan.expected_len, 480 * 8 * 270);
		assert!((plan.uv_max[0] - 1918.0 / 1920.0).abs() < 1e-6);
		assert!(!plan.needs_cpu_decode);
	}

	#[test]
	fn bc3_ycocg_plan_marks_ycocg() {
		let plan = plan_upload(&frame(6, 5, PixelPayload::Bc3YCoCg(Vec::new())), true, 8192).unwrap();
		assert_eq!(plan.format, wgpu::TextureFormat::Bc3RgbaUnorm);
		assert_eq!((plan.alloc_width, plan.alloc_height), (8, 8));
		assert_eq!(plan.bytes_per_row, 2 * 16);
		assert_eq!(plan.expected_len, 2 * 16 * 2);
		assert!(plan.is_ycocg);
	}

	#[test]
	fn bc_without_support_falls_back_to_rgba() {
		let plan = plan_upload(&frame(6, 5, PixelPayload::Bc1(Vec::new())), false, 8192).unwrap();
		assert_eq!(plan.format, wgpu::TextureFormat::Rgba8Unorm);
		assert_eq!(plan.expected_len, 6 * 5 * 4);
		assert!(plan.needs_cpu_decode);
		assert!(!plan.is_ycocg);
	}

	#[test]
	fn rejects_degenerate_frames() {
		assert_eq!(
			plan_upload(&frame(0, 4, PixelPayload::Rgba8(Vec::new())), true, 8192),
			Err(UploadRejection::ZeroSize)
		);
		assert_eq!(
			plan_upload(&frame(9000, 4, PixelPayload::Rgba8(Vec::new())), true, 8192),
			Err(UploadRejection::TooLarge { max_dimension: 8192 })
		);
	}

	#[test]
	fn drawable_statuses() {
		assert!(LayerUploadStatus::Uploaded.is_drawable());
		assert!(LayerUploadStatus::Held.is_drawable());
		assert!(!LayerUploadStatus::Inactive.is_drawable());
		assert!(
			!LayerUploadStatus::Rejected {
				reason: UploadRejection::ZeroSize,
				holding: false
			}
			.is_drawable()
		);
	}
}
