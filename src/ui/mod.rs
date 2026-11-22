//! Control UI for the VJ booth: command protocol, keyboard fallback, and egui layout.
//!
//! The UI never mutates [`AppState`](crate::model::AppState) directly. Every
//! widget gesture is reduced to a [`UiCommand`] by [`draw_control_ui`], keyboard
//! fallbacks are reduced by [`map_key_event`], and `app.rs` applies both through
//! [`apply_ui_command`]. The only state owned here is [`UiState`]: GPU-side
//! thumbnail textures and transient drag-and-drop feedback, none of which needs
//! to be persisted.

pub mod commands;
pub mod layout;
pub mod shortcuts;

use std::collections::{HashMap, HashSet};
use std::fmt;

pub use commands::{UiApplyOutcome, UiCommand, apply_ui_command};
pub use layout::{apply_vj_theme, draw_control_ui};
pub use shortcuts::{SHORTCUT_LEGEND, ShortcutEntry, map_key_event};

use crate::model::{ClipId, ClipInfo, ThumbnailImage};

/// Transient, non-persisted state owned by the control UI.
///
/// Holds one egui texture per clip thumbnail so the clip grid and the layer
/// rack can show first frames without re-uploading pixels every frame, plus the
/// drag-and-drop hover counter that drives the drop highlight.
#[derive(Default)]
pub struct UiState {
	/// Uploaded thumbnail textures keyed by clip.
	thumbnails: HashMap<ClipId, egui::TextureHandle>,
	/// Number of files currently hovering over the control window.
	///
	/// Incremented on winit `HoveredFile`, reset on `HoveredFileCancelled` and
	/// `DroppedFile`. Any value above zero highlights the media library.
	pub hovered_files: usize,
	/// Whether [`apply_vj_theme`] already ran on the current context.
	theme_applied: bool,
}

impl UiState {
	/// Creates an empty UI state; the theme is applied on first use.
	pub fn new() -> Self {
		Self::default()
	}

	/// Returns the uploaded thumbnail texture of `clip_id`, if any.
	pub fn thumbnail(&self, clip_id: ClipId) -> Option<&egui::TextureHandle> {
		self.thumbnails.get(&clip_id)
	}

	/// Number of thumbnail textures currently resident on the GPU.
	pub fn thumbnail_count(&self) -> usize {
		self.thumbnails.len()
	}

	/// Returns `true` once [`apply_vj_theme`] has been applied by this state.
	pub fn theme_applied(&self) -> bool {
		self.theme_applied
	}

	/// Applies the VJ booth theme to `ctx` the first time it is called.
	pub fn ensure_theme(&mut self, ctx: &egui::Context) {
		if !self.theme_applied {
			apply_vj_theme(ctx);
			self.theme_applied = true;
		}
	}

	/// Uploads missing thumbnails and releases textures of clips that went away.
	///
	/// Only valid [`ThumbnailImage`]s that fit into the backend's maximum texture
	/// side are uploaded, so malformed decoder output can never trip egui's size
	/// assertions. Textures whose clip was removed, failed, or lost its thumbnail
	/// are dropped, which frees the GPU memory. The call is cheap when nothing
	/// changed and is safe to run every frame.
	pub fn sync_thumbnails(&mut self, ctx: &egui::Context, clips: &[ClipInfo]) {
		self.ensure_theme(ctx);

		let live: HashSet<ClipId> = clips
			.iter()
			.filter(|clip| usable_thumbnail(clip).is_some())
			.map(|clip| clip.id)
			.collect();
		self.thumbnails.retain(|clip_id, _| live.contains(clip_id));

		let max_side = ctx.input(|input| input.max_texture_side);
		for clip in clips {
			if self.thumbnails.contains_key(&clip.id) {
				continue;
			}
			let Some(image) = usable_thumbnail(clip) else {
				continue;
			};
			let width = image.width as usize;
			let height = image.height as usize;
			if width > max_side || height > max_side {
				continue;
			}
			let color_image = egui::ColorImage::from_rgba_unmultiplied([width, height], &image.rgba);
			let handle = ctx.load_texture(
				format!("mur-thumbnail-{}", clip.id.0),
				color_image,
				egui::TextureOptions::LINEAR,
			);
			self.thumbnails.insert(clip.id, handle);
		}
	}
}

impl fmt::Debug for UiState {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("UiState")
			.field("thumbnails", &self.thumbnails.len())
			.field("hovered_files", &self.hovered_files)
			.field("theme_applied", &self.theme_applied)
			.finish()
	}
}

/// Returns the clip's thumbnail when it is present, non-empty, and well formed.
fn usable_thumbnail(clip: &ClipInfo) -> Option<&ThumbnailImage> {
	clip.thumbnail
		.as_ref()
		.filter(|thumbnail| thumbnail.is_valid() && thumbnail.width > 0 && thumbnail.height > 0)
}
