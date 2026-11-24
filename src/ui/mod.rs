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
use std::time::{Duration, Instant};

pub use commands::{UiApplyOutcome, UiCommand, apply_ui_command};
pub use layout::{apply_vj_theme, draw_control_ui};
pub use shortcuts::{SHORTCUT_LEGEND, ShortcutEntry, map_key_event};

use crate::model::{ClipId, ClipInfo, ThumbnailImage};

/// How long an armed "unload live clip" confirmation stays valid.
pub const REMOVE_CONFIRM_WINDOW: Duration = Duration::from_secs(3);

/// Transient, non-persisted state owned by the control UI.
///
/// Holds one egui texture per clip thumbnail so the clip grid and the layer
/// rack can show first frames without re-uploading pixels every frame, the
/// drag-and-drop hover counter that drives the drop highlight, and the pending
/// two-click confirmation for unloading a clip that is live on a layer.
#[derive(Default)]
pub struct UiState {
	/// Uploaded thumbnail textures keyed by clip.
	thumbnails: HashMap<ClipId, egui::TextureHandle>,
	/// Number of files currently hovering over the control window.
	///
	/// Incremented on winit `HoveredFile`, reset on `HoveredFileCancelled` and
	/// `DroppedFile`. Any value above zero highlights the media library.
	pub hovered_files: usize,
	/// Whether [`apply_vj_theme`] has been applied at least once by this state.
	theme_applied: bool,
	/// Clip whose "×" button was clicked once while it is live, and when.
	///
	/// A second click within [`REMOVE_CONFIRM_WINDOW`] unloads the clip; the
	/// confirmation expires afterwards. Clips that are not live on any layer
	/// unload on a single click and never set this.
	pub pending_remove_clip: Option<(ClipId, Instant)>,
	/// Scratch set reused by [`UiState::sync_thumbnails`] to avoid a per-frame allocation.
	live_clip_ids: HashSet<ClipId>,
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

	/// Applies the VJ booth theme to `ctx` once per context.
	pub fn ensure_theme(&mut self, ctx: &egui::Context) {
		let key = egui::Id::unique("mur_vj_theme_applied");
		let already_applied = ctx.data(|data| data.get_temp::<bool>(key).unwrap_or(false));
		if !already_applied {
			apply_vj_theme(ctx);
			ctx.data_mut(|data| data.insert_temp(key, true));
		}
		self.theme_applied = true;
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

		let live = &mut self.live_clip_ids;
		live.clear();
		live.extend(
			clips
				.iter()
				.filter(|clip| usable_thumbnail(clip).is_some())
				.map(|clip| clip.id),
		);
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
			.field("pending_remove_clip", &self.pending_remove_clip)
			.finish_non_exhaustive()
	}
}

/// Returns `true` while the unload confirmation of `clip_id` is armed at `now`.
pub fn remove_confirmation_armed(pending: Option<(ClipId, Instant)>, clip_id: ClipId, now: Instant) -> bool {
	pending.is_some_and(|(pending_id, armed_at)| {
		pending_id == clip_id && now.saturating_duration_since(armed_at) < REMOVE_CONFIRM_WINDOW
	})
}

/// Handles a click on a clip tile's "×" button; returns `true` when the clip should unload.
///
/// Clips that are not live unload immediately. A live clip first arms the
/// confirmation (stored in `pending`); a second click within
/// [`REMOVE_CONFIRM_WINDOW`] unloads it and clears the confirmation.
pub fn handle_remove_click(
	pending: &mut Option<(ClipId, Instant)>,
	clip_id: ClipId,
	is_live: bool,
	now: Instant,
) -> bool {
	if !is_live {
		if pending.is_some_and(|(pending_id, _)| pending_id == clip_id) {
			*pending = None;
		}
		return true;
	}
	if remove_confirmation_armed(*pending, clip_id, now) {
		*pending = None;
		return true;
	}
	*pending = Some((clip_id, now));
	false
}

/// Returns the clip's thumbnail when it is present, non-empty, and well formed.
fn usable_thumbnail(clip: &ClipInfo) -> Option<&ThumbnailImage> {
	clip.thumbnail
		.as_ref()
		.filter(|thumbnail| thumbnail.is_valid() && thumbnail.width > 0 && thumbnail.height > 0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;

	#[test]
	fn inactive_clips_unload_on_a_single_click() {
		let now = Instant::now();
		let mut pending = None;
		assert!(handle_remove_click(&mut pending, ClipId(1), false, now));
		assert!(pending.is_none());
	}

	#[test]
	fn live_clips_need_a_second_click_within_the_window() {
		let start = Instant::now();
		let mut pending = None;
		assert!(!handle_remove_click(&mut pending, ClipId(1), true, start));
		assert!(remove_confirmation_armed(pending, ClipId(1), start));
		assert!(!remove_confirmation_armed(pending, ClipId(2), start));
		let second = start + Duration::from_secs(1);
		assert!(handle_remove_click(&mut pending, ClipId(1), true, second));
		assert!(pending.is_none());
	}

	#[test]
	fn live_clip_confirmation_expires_and_rearms() {
		let start = Instant::now();
		let mut pending = None;
		assert!(!handle_remove_click(&mut pending, ClipId(1), true, start));
		let late = start + REMOVE_CONFIRM_WINDOW;
		assert!(!remove_confirmation_armed(pending, ClipId(1), late));
		assert!(
			!handle_remove_click(&mut pending, ClipId(1), true, late),
			"an expired confirmation re-arms"
		);
		assert!(handle_remove_click(
			&mut pending,
			ClipId(1),
			true,
			late + Duration::from_millis(500)
		));
	}

	#[test]
	fn arming_another_clip_replaces_the_confirmation() {
		let start = Instant::now();
		let mut pending = None;
		assert!(!handle_remove_click(&mut pending, ClipId(1), true, start));
		assert!(!handle_remove_click(&mut pending, ClipId(2), true, start));
		assert!(!remove_confirmation_armed(pending, ClipId(1), start));
		assert!(
			!handle_remove_click(&mut pending, ClipId(1), true, start),
			"clip 1 must be armed again"
		);
	}
}
