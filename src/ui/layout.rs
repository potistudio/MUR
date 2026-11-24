//! egui layout of the VJ booth control window (SPEC §7.1).
//!
//! [`draw_control_ui`] renders the whole control surface from a read-only
//! [`AppState`] and returns the operator's gestures as [`UiCommand`]s; it never
//! mutates the state itself, so the same command path serves the mouse, the
//! keyboard fallback, and MIDI. The window is split into:
//!
//! 1. a status bar (output display, MIDI port, performance, tool toggles),
//! 2. a stage with the 16:9 master mirror and the master effect strip,
//! 3. the four-layer mixer rack,
//! 4. the media library clip grid (drag-and-drop target), and
//! 5. overlays: log drawer, shortcut reference, toast stack, and the exit guard.
//!
//! While the exit guard is open every live control behind it is disabled, so a
//! stray click can never reach the performance (SPEC §9.2).

use std::time::Instant;

use egui::{
	Align, Align2, Button, Color32, ComboBox, CornerRadius, FontId, Frame, Id, Layout, Margin, Order, Pos2,
	ProgressBar, Rect, RichText, Sense, Slider, Stroke, StrokeKind, TextStyle, TextureId, Theme, Ui, Vec2, vec2,
};

use crate::error::ToastLevel;
use crate::model::effect::{MAX_STROBE_HZ, MIN_STROBE_HZ};
use crate::model::layer::{MAX_SPEED, MIN_SPEED};
use crate::model::{
	AppState, BlendMode, ClipId, ClipInfo, ClipKind, ClipStatus, LayerIndex, MidiConnectionStatus, OutputRuntimeStatus,
	PlaybackMode,
};

use super::commands::UiCommand;
use super::shortcuts::SHORTCUT_LEGEND;
use super::{REMOVE_CONFIRM_WINDOW, UiState, handle_remove_click, remove_confirmation_armed};

// ------------------------------------------------------------------ palette

/// Deepest background, used for the mirror frame and text fields.
const BG_DEEP: Color32 = Color32::from_rgb(0x10, 0x12, 0x16);
/// Panel background.
const BG_PANEL: Color32 = Color32::from_rgb(0x16, 0x19, 0x20);
/// Elevated card surface.
const CARD: Color32 = Color32::from_rgb(0x1D, 0x21, 0x2A);
/// Raised widget surface.
const CARD_RAISED: Color32 = Color32::from_rgb(0x26, 0x2B, 0x36);
/// Hovered widget surface.
const CARD_HOVER: Color32 = Color32::from_rgb(0x2D, 0x34, 0x42);
/// Hairline border colour.
const HAIRLINE: Color32 = Color32::from_rgb(0x2F, 0x35, 0x42);
/// Primary text colour.
const TEXT: Color32 = Color32::from_rgb(0xE5, 0xE9, 0xF0);
/// Secondary text colour.
const TEXT_DIM: Color32 = Color32::from_rgb(0x8B, 0x93, 0xA1);
/// Focus and selection accent (sky blue).
const ACCENT: Color32 = Color32::from_rgb(0x38, 0xBD, 0xF8);
/// Secondary accent (cyan), used for live/active markers.
const ACCENT_CYAN: Color32 = Color32::from_rgb(0x22, 0xD3, 0xEE);
/// Dark accent fill behind selected widgets.
const ACCENT_DEEP: Color32 = Color32::from_rgb(0x0C, 0x3B, 0x52);
/// Healthy status colour.
const GOOD: Color32 = Color32::from_rgb(0x22, 0xC5, 0x5E);
/// Warning colour.
const WARN: Color32 = Color32::from_rgb(0xF5, 0x9E, 0x0B);
/// Blackout and error colour.
const DANGER: Color32 = Color32::from_rgb(0xEF, 0x44, 0x44);
/// Dark red fill for the idle blackout button.
const DANGER_DEEP: Color32 = Color32::from_rgb(0x5A, 0x16, 0x16);
/// Neutral/disabled status colour.
const NEUTRAL: Color32 = Color32::from_rgb(0x6B, 0x72, 0x80);

/// Width of one clip-grid tile.
const TILE_WIDTH: f32 = 150.0;
/// Height of the thumbnail inside a clip-grid tile (16:9).
const TILE_THUMB_HEIGHT: f32 = TILE_WIDTH * 9.0 / 16.0;
/// Width of the master effect strip next to the mirror.
const FX_STRIP_WIDTH: f32 = 360.0;

// -------------------------------------------------------------------- theme

/// Applies the dark live-performance palette and default fonts to `ctx`.
///
/// Installs `egui`'s embedded default fonts with `Hack` added as a fallback
/// under [`egui::FontFamily::Proportional`] so UI symbols (`●`, `▸`) render
/// without missing-glyph boxes, then restyles the dark theme: obsidian panels,
/// slate cards with hairline borders, a sky-blue focus accent, amber warnings,
/// red errors, and slightly larger monospace readouts for numeric values.
/// Idempotent.
pub fn apply_vj_theme(ctx: &egui::Context) {
	let mut fonts = egui::FontDefinitions::default();
	if let Some(proportional) = fonts.families.get_mut(&egui::FontFamily::Proportional)
		&& fonts.font_data.contains_key("Hack")
		&& !proportional.iter().any(|name| name == "Hack")
	{
		// Keep Ubuntu-Light primary and fall back to Hack for symbols (`●`, `▸`)
		// that Ubuntu-Light does not provide.
		let insert_at = usize::from(!proportional.is_empty());
		proportional.insert(insert_at, "Hack".to_owned());
	}
	ctx.set_fonts(fonts);
	ctx.set_theme(Theme::Dark);
	ctx.style_mut_of(Theme::Dark, configure_style);
}

/// Writes the VJ booth palette, spacing, and typography into `style`.
fn configure_style(style: &mut egui::Style) {
	let radius = CornerRadius::same(4);
	let visuals = &mut style.visuals;
	visuals.dark_mode = true;
	visuals.override_text_color = None;
	visuals.panel_fill = BG_PANEL;
	visuals.window_fill = CARD;
	visuals.window_stroke = Stroke::new(1.0, HAIRLINE);
	visuals.window_corner_radius = CornerRadius::same(6);
	visuals.menu_corner_radius = radius;
	visuals.extreme_bg_color = BG_DEEP;
	visuals.faint_bg_color = Color32::from_rgb(0x1A, 0x1E, 0x26);
	visuals.code_bg_color = BG_DEEP;
	visuals.hyperlink_color = ACCENT;
	visuals.warn_fg_color = WARN;
	visuals.error_fg_color = DANGER;
	visuals.slider_trailing_fill = true;
	visuals.selection.bg_fill = ACCENT_DEEP;
	visuals.selection.stroke = Stroke::new(1.0, ACCENT);

	let widgets = &mut visuals.widgets;
	widgets.noninteractive.bg_fill = CARD;
	widgets.noninteractive.weak_bg_fill = CARD;
	widgets.noninteractive.bg_stroke = Stroke::new(1.0, HAIRLINE);
	widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
	widgets.noninteractive.corner_radius = radius;

	widgets.inactive.bg_fill = CARD_RAISED;
	widgets.inactive.weak_bg_fill = CARD_RAISED;
	widgets.inactive.bg_stroke = Stroke::new(1.0, HAIRLINE);
	widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
	widgets.inactive.corner_radius = radius;

	widgets.hovered.bg_fill = CARD_HOVER;
	widgets.hovered.weak_bg_fill = CARD_HOVER;
	widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT.gamma_multiply(0.6));
	widgets.hovered.fg_stroke = Stroke::new(1.5, TEXT);
	widgets.hovered.corner_radius = radius;

	widgets.active.bg_fill = ACCENT_DEEP;
	widgets.active.weak_bg_fill = ACCENT_DEEP;
	widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
	widgets.active.fg_stroke = Stroke::new(2.0, Color32::WHITE);
	widgets.active.corner_radius = radius;

	widgets.open.bg_fill = CARD_RAISED;
	widgets.open.weak_bg_fill = CARD_RAISED;
	widgets.open.bg_stroke = Stroke::new(1.0, ACCENT);
	widgets.open.fg_stroke = Stroke::new(1.0, TEXT);
	widgets.open.corner_radius = radius;

	style.spacing.item_spacing = vec2(6.0, 5.0);
	style.spacing.button_padding = vec2(8.0, 3.0);
	style.spacing.slider_width = 150.0;
	style.spacing.interact_size.y = 20.0;

	style.text_styles.insert(TextStyle::Small, FontId::proportional(10.5));
	style.text_styles.insert(TextStyle::Body, FontId::proportional(13.0));
	style.text_styles.insert(TextStyle::Button, FontId::proportional(13.0));
	style.text_styles.insert(TextStyle::Heading, FontId::proportional(17.0));
	style.text_styles.insert(TextStyle::Monospace, FontId::monospace(12.5));
}

// --------------------------------------------------------------- entrypoints

/// Draws the full control window and returns the commands the operator issued.
///
/// Must run inside an egui pass started with [`egui::Context::run_pass`] (or
/// `begin_pass`/`end_pass`), because it draws onto the pass's root [`Ui`].
/// When the caller already holds a `&mut Ui` (for example inside
/// [`egui::Context::run_ui`]), use [`draw_control_ui_in`] instead.
///
/// `mirror_texture` is the registered egui texture of the downscaled master
/// output; `None` shows a placeholder until the renderer produced a frame.
pub fn draw_control_ui(
	ctx: &egui::Context,
	app: &AppState,
	ui_state: &mut UiState,
	mirror_texture: Option<TextureId>,
) -> Vec<UiCommand> {
	ui_state.ensure_theme(ctx);
	ctx.root_ui(|ui| draw_control_ui_in(ui, app, ui_state, mirror_texture))
}

/// Draws the full control window into `ui` and returns the issued commands.
///
/// Same as [`draw_control_ui`] but for callers that already own a [`Ui`]
/// covering the window, such as the callback of [`egui::Context::run_ui`].
pub fn draw_control_ui_in(
	ui: &mut Ui,
	app: &AppState,
	ui_state: &mut UiState,
	mirror_texture: Option<TextureId>,
) -> Vec<UiCommand> {
	ui_state.ensure_theme(ui.ctx());
	let mut out = Vec::new();
	let live_enabled = !app.exit_confirm_open;
	let now = Instant::now();
	// Expired confirmations are dropped before drawing.
	let mut pending_remove = ui_state
		.pending_remove_clip
		.take()
		.filter(|&(clip_id, armed_at)| remove_confirmation_armed(Some((clip_id, armed_at)), clip_id, now));
	let state: &UiState = ui_state;

	ui.add_enabled_ui(live_enabled, |ui| {
		egui::Panel::top("mur_status_bar")
			.frame(
				Frame::new()
					.fill(BG_DEEP)
					.inner_margin(Margin::symmetric(10, 6))
					.stroke(Stroke::new(1.0, HAIRLINE)),
			)
			.show(ui, |ui| draw_status_bar(ui, app, &mut out));

		egui::Panel::bottom("mur_media_library")
			.resizable(true)
			.default_size(260.0)
			.min_size(150.0)
			.frame(library_frame(state.hovered_files > 0))
			.show(ui, |ui| {
				draw_media_library(ui, app, state, &mut pending_remove, now, &mut out)
			});

		egui::CentralPanel::default()
			.frame(Frame::new().fill(BG_PANEL).inner_margin(Margin::same(10)))
			.show(ui, |ui| {
				draw_stage(ui, app, mirror_texture, &mut out);
				ui.add_space(8.0);
				egui::ScrollArea::vertical()
					.id_salt("mur_mixer_rack")
					.auto_shrink([false, true])
					.show(ui, |ui| draw_mixer_rack(ui, app, state, &mut out));
			});
	});

	let ctx = ui.ctx().clone();
	if live_enabled {
		draw_toasts(&ctx, app, &mut out);
		if app.show_logs {
			draw_log_drawer(&ctx, app, &mut out);
		}
		if app.show_shortcuts {
			draw_shortcut_overlay(&ctx, &mut out);
		}
	} else {
		draw_exit_modal(&ctx, &mut out);
	}

	ui_state.pending_remove_clip = pending_remove;
	if ui_state.pending_remove_clip.is_some() {
		// Repaint so the armed "×?" reverts once the confirmation window expires.
		ctx.request_repaint_after(REMOVE_CONFIRM_WINDOW);
	}
	out
}

// ---------------------------------------------------------------- status bar

/// Renders the top status bar: output display, MIDI, performance, tool toggles.
fn draw_status_bar(ui: &mut Ui, app: &AppState, out: &mut Vec<UiCommand>) {
	ui.horizontal(|ui| {
		ui.label(RichText::new("MUR").strong().size(16.0).color(ACCENT));
		ui.separator();

		ui.label(RichText::new("DISPLAY").small().color(TEXT_DIM));
		draw_display_selector(ui, app, out);
		let output_label = if app.output.enabled { "Output ON" } else { "Output OFF" };
		if ui
			.add(Button::new(output_label).selected(app.output.enabled))
			.on_hover_text("Switch the external output window on or off")
			.clicked()
		{
			out.push(UiCommand::ToggleOutputEnabled);
		}
		output_status_pill(ui, &app.output_status);
		ui.separator();

		ui.label(RichText::new("MIDI").small().color(TEXT_DIM));
		draw_midi_selector(ui, app, out);
		if ui.button("Rescan").on_hover_text("Scan MIDI ports again").clicked() {
			out.push(UiCommand::RescanMidiPorts);
		}
		midi_status_pill(ui, &app.midi_status);

		ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
			if ui
				.add(Button::new("Shortcuts (?)").selected(app.show_shortcuts))
				.clicked()
			{
				out.push(UiCommand::ToggleShortcutsOverlay);
			}
			if ui.add(Button::new("Logs").selected(app.show_logs)).clicked() {
				out.push(UiCommand::ToggleLogsDrawer);
			}
			ui.separator();
			if app.perf.dropped_frames > 0 {
				pill(ui, &format!("drop: {}", app.perf.dropped_frames), BG_DEEP, WARN);
			}
			ui.label(
				RichText::new(format!(
					"{:.1} fps · {:.1} ms",
					finite_or_zero(app.perf.fps),
					finite_or_zero(app.perf.frame_time_ms)
				))
				.monospace()
				.color(fps_color(app.perf.fps)),
			);
		});
	});
}

/// Output display picker over "Auto / None" plus every discovered monitor.
fn draw_display_selector(ui: &mut Ui, app: &AppState, out: &mut Vec<UiCommand>) {
	let selected = app.output.monitor_name.as_deref();
	// Selections are stored as monitor ids (unique even for identical models);
	// plain names from older state files or projects still match by name.
	let is_selected = |monitor: &crate::model::MonitorSummary| {
		selected.is_some_and(|selected| monitor.id == selected || monitor.name == selected)
	};
	let selected_text = match selected {
		None => "Auto / None".to_owned(),
		Some(name) => match app.available_monitors.iter().find(|monitor| is_selected(monitor)) {
			Some(monitor) => monitor_label(monitor),
			None => format!("{} (missing)", display_name_of(name)),
		},
	};

	ComboBox::from_id_salt("mur_display_selector")
		.selected_text(selected_text)
		.width(230.0)
		.show_ui(ui, |ui| {
			if ui.selectable_label(selected.is_none(), "Auto / None").clicked() && selected.is_some() {
				out.push(UiCommand::SelectOutputMonitor(None));
			}
			for monitor in &app.available_monitors {
				let is_selected = is_selected(monitor);
				if ui.selectable_label(is_selected, monitor_label(monitor)).clicked() && !is_selected {
					out.push(UiCommand::SelectOutputMonitor(Some(monitor.id.clone())));
				}
			}
		});
}

/// Display name part of a stored monitor selection (`name|WxH|x,y[|cg:N]` or a plain name).
fn display_name_of(selection: &str) -> &str {
	selection.split('|').next().unwrap_or(selection)
}

/// Human readable monitor entry, e.g. `DELL (1920×1080) [Primary]`.
fn monitor_label(monitor: &crate::model::MonitorSummary) -> String {
	let primary = if monitor.is_primary { " [Primary]" } else { "" };
	format!("{} ({}×{}){primary}", monitor.name, monitor.width, monitor.height)
}

/// MIDI port picker over "Auto (LCXL3)" plus every discovered port.
fn draw_midi_selector(ui: &mut Ui, app: &AppState, out: &mut Vec<UiCommand>) {
	let selected = app.selected_midi_port.as_deref();
	ComboBox::from_id_salt("mur_midi_selector")
		.selected_text(selected.unwrap_or("Auto (LCXL3)"))
		.width(200.0)
		.show_ui(ui, |ui| {
			if ui.selectable_label(selected.is_none(), "Auto (LCXL3)").clicked() && selected.is_some() {
				out.push(UiCommand::SelectMidiPort(None));
			}
			for port in &app.midi_ports {
				let is_selected = selected == Some(port.as_str());
				if ui.selectable_label(is_selected, port.as_str()).clicked() && !is_selected {
					out.push(UiCommand::SelectMidiPort(Some(port.clone())));
				}
			}
		});
}

/// Colour-coded badge describing the external output window.
fn output_status_pill(ui: &mut Ui, status: &OutputRuntimeStatus) {
	match status {
		OutputRuntimeStatus::Active {
			monitor_name,
			width,
			height,
		} => {
			pill(ui, &format!("● LIVE {monitor_name} {width}×{height}"), BG_DEEP, GOOD);
		}
		OutputRuntimeStatus::WaitingForMonitor { expected_name } => {
			let name = expected_name.as_deref().unwrap_or("monitor");
			pill(ui, &format!("WAITING {name}"), BG_DEEP, WARN)
				.on_hover_text("The selected display is missing; output resumes when it reconnects");
		}
		OutputRuntimeStatus::Disabled => {
			pill(ui, "OFF", BG_DEEP, NEUTRAL);
		}
		OutputRuntimeStatus::Error(message) => {
			pill(ui, "OUTPUT ERROR", BG_DEEP, DANGER).on_hover_text(message.as_str());
		}
	}
}

/// Colour-coded badge describing the MIDI connection.
fn midi_status_pill(ui: &mut Ui, status: &MidiConnectionStatus) {
	match status {
		MidiConnectionStatus::Connected { port_name } => {
			pill(ui, &format!("● {port_name}"), BG_DEEP, GOOD);
		}
		MidiConnectionStatus::Scanning => {
			pill(ui, "Scanning…", BG_DEEP, ACCENT);
		}
		MidiConnectionStatus::Reconnecting { last_port, error } => {
			let port = last_port.as_deref().unwrap_or("controller");
			let response = pill(ui, &format!("Reconnecting {port}"), BG_DEEP, WARN);
			if let Some(error) = error {
				response.on_hover_text(error.as_str());
			}
		}
		MidiConnectionStatus::Disconnected => {
			pill(ui, "Disconnected", BG_DEEP, NEUTRAL);
		}
	}
}

// --------------------------------------------------------------------- stage

/// Renders the master mirror on the left and the master effect strip on the right.
fn draw_stage(ui: &mut Ui, app: &AppState, mirror_texture: Option<TextureId>, out: &mut Vec<UiCommand>) {
	let available = ui.available_size();
	let spacing = ui.spacing().item_spacing.x;
	let max_height = (available.y * 0.5).clamp(140.0, 440.0);
	let mirror_width = (available.x - FX_STRIP_WIDTH - spacing * 3.0)
		.min(max_height * 16.0 / 9.0)
		.max(160.0);

	ui.horizontal_top(|ui| {
		draw_mirror(ui, app, mirror_texture, vec2(mirror_width, mirror_width * 9.0 / 16.0));
		card_frame(false).show(ui, |ui| {
			ui.set_width(FX_STRIP_WIDTH - 24.0);
			draw_fx_strip(ui, app, out);
		});
	});
}

/// Paints the 16:9 master output mirror with blackout and output-state overlays.
fn draw_mirror(ui: &mut Ui, app: &AppState, mirror_texture: Option<TextureId>, size: Vec2) {
	let (rect, _response) = ui.allocate_exact_size(size, Sense::hover());
	let painter = ui.painter_at(rect);
	painter.rect_filled(rect, CornerRadius::same(4), Color32::BLACK);

	match mirror_texture {
		Some(texture) => {
			let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
			painter.image(texture, rect, uv, Color32::WHITE);
		}
		None => {
			painter.text(
				rect.center(),
				Align2::CENTER_CENTER,
				"Waiting for first GPU frame",
				FontId::proportional(14.0),
				TEXT_DIM,
			);
		}
	}

	painter.text(
		rect.left_top() + vec2(8.0, 6.0),
		Align2::LEFT_TOP,
		"MASTER · 1920×1080",
		FontId::monospace(11.0),
		TEXT_DIM,
	);

	if app.effects.blackout {
		painter.rect_filled(rect, CornerRadius::same(4), Color32::from_black_alpha(160));
		let badge = Rect::from_center_size(rect.center(), vec2(200.0_f32.min(rect.width() - 8.0), 44.0));
		painter.rect_filled(badge, CornerRadius::same(6), DANGER);
		painter.text(
			badge.center(),
			Align2::CENTER_CENTER,
			"BLACKOUT",
			FontId::proportional(24.0),
			Color32::WHITE,
		);
	}

	let banner = match &app.output_status {
		OutputRuntimeStatus::WaitingForMonitor { expected_name } => Some((
			format!(
				"Output: WaitingForMonitor ({})",
				expected_name.as_deref().unwrap_or("no display")
			),
			WARN,
		)),
		OutputRuntimeStatus::Error(message) => Some((format!("Output error: {message}"), DANGER)),
		OutputRuntimeStatus::Active { .. } | OutputRuntimeStatus::Disabled => None,
	};
	if let Some((text, color)) = banner {
		let strip = Rect::from_min_max(Pos2::new(rect.left(), rect.bottom() - 26.0), rect.right_bottom());
		painter.rect_filled(strip, CornerRadius::ZERO, color.gamma_multiply(0.85));
		painter.text(
			strip.center(),
			Align2::CENTER_CENTER,
			text,
			FontId::proportional(13.0),
			Color32::BLACK,
		);
	}

	let border = if app.effects.blackout { DANGER } else { HAIRLINE };
	painter.rect_stroke(
		rect,
		CornerRadius::same(4),
		Stroke::new(1.0, border),
		StrokeKind::Inside,
	);
}

/// Master effects, colour grade, dimmer, reset, and the BLACKOUT button.
fn draw_fx_strip(ui: &mut Ui, app: &AppState, out: &mut Vec<UiCommand>) {
	let fx = &app.effects;
	section_title(ui, "MASTER FX");

	egui::Grid::new("mur_fx_grid")
		.num_columns(2)
		.spacing(vec2(8.0, 4.0))
		.show(ui, |ui| {
			if fx_toggle(ui, "RGB SHIFT", fx.rgb_shift_enabled) {
				out.push(UiCommand::ToggleRgbShift);
			}
			if let Some(value) = unit_slider(ui, fx.rgb_shift_intensity, 0.0..=1.0, "") {
				out.push(UiCommand::SetRgbShiftIntensity(value));
			}
			ui.end_row();

			if fx_toggle(ui, "GLITCH", fx.glitch_enabled) {
				out.push(UiCommand::ToggleGlitch);
			}
			if let Some(value) = unit_slider(ui, fx.glitch_intensity, 0.0..=1.0, "") {
				out.push(UiCommand::SetGlitchIntensity(value));
			}
			ui.end_row();

			if fx_toggle(ui, "STROBE", fx.strobe_enabled) {
				out.push(UiCommand::ToggleStrobe);
			}
			if let Some(value) = unit_slider(ui, fx.strobe_intensity, 0.0..=1.0, "") {
				out.push(UiCommand::SetStrobeIntensity(value));
			}
			ui.end_row();

			dim_label(ui, "Speed");
			if let Some(value) = unit_slider(ui, fx.strobe_speed_hz, MIN_STROBE_HZ..=MAX_STROBE_HZ, " Hz") {
				out.push(UiCommand::SetStrobeSpeedHz(value));
			}
			ui.end_row();

			dim_label(ui, "Brightness");
			if let Some(value) = unit_slider(ui, fx.brightness, -1.0..=1.0, "") {
				out.push(UiCommand::SetBrightness(value));
			}
			ui.end_row();

			dim_label(ui, "Contrast");
			if let Some(value) = unit_slider(ui, fx.contrast, 0.0..=2.0, "") {
				out.push(UiCommand::SetContrast(value));
			}
			ui.end_row();

			dim_label(ui, "Saturation");
			if let Some(value) = unit_slider(ui, fx.saturation, 0.0..=2.0, "") {
				out.push(UiCommand::SetSaturation(value));
			}
			ui.end_row();

			ui.label(RichText::new("DIMMER").strong().color(TEXT));
			if let Some(value) = unit_slider(ui, fx.master_dimmer, 0.0..=1.0, "") {
				out.push(UiCommand::SetMasterDimmer(value));
			}
			ui.end_row();
		});

	ui.add_space(4.0);
	if ui
		.add(Button::new("Reset FX").min_size(vec2(ui.available_width(), 22.0)))
		.on_hover_text("Restore every effect to neutral (dimmer and blackout are kept)")
		.clicked()
	{
		out.push(UiCommand::ResetAllEffects);
	}

	let (label, fill) = if fx.blackout {
		("BLACKOUT ACTIVE · click to restore", DANGER)
	} else {
		("BLACKOUT", DANGER_DEEP)
	};
	let blackout = Button::new(RichText::new(label).strong().size(17.0).color(Color32::WHITE))
		.fill(fill)
		.stroke(Stroke::new(1.5, DANGER))
		.min_size(vec2(ui.available_width(), 40.0));
	if ui.add(blackout).clicked() {
		out.push(UiCommand::ToggleBlackout);
	}
}

// ---------------------------------------------------------------- mixer rack

/// Renders the four layer columns side by side.
fn draw_mixer_rack(ui: &mut Ui, app: &AppState, ui_state: &UiState, out: &mut Vec<UiCommand>) {
	ui.columns_const(|columns: &mut [Ui; 4]| {
		for (column, layer) in columns.iter_mut().zip(LayerIndex::ALL) {
			draw_layer_column(column, app, ui_state, layer, out);
		}
	});
}

/// Renders one layer strip: header, active clip, transport, and mix controls.
fn draw_layer_column(ui: &mut Ui, app: &AppState, ui_state: &UiState, layer: LayerIndex, out: &mut Vec<UiCommand>) {
	let state = app.layer(layer);
	let focused = app.focused_layer == layer;

	card_frame(focused).show(ui, |ui| {
		ui.set_width(ui.available_width());

		ui.horizontal(|ui| {
			let badge = RichText::new(format!(" {layer} "))
				.strong()
				.monospace()
				.color(if focused { BG_DEEP } else { TEXT });
			let header = Button::new(badge)
				.fill(if focused { ACCENT } else { CARD_RAISED })
				.selected(focused);
			if ui.add(header).on_hover_text("Focus this layer").clicked() && !focused {
				out.push(UiCommand::FocusLayer(layer));
			}
			let (state_text, color) = match state.playback_state {
				crate::model::PlaybackState::Playing => ("PLAYING", GOOD),
				crate::model::PlaybackState::Paused => ("PAUSED", WARN),
				crate::model::PlaybackState::Stopped => ("STOPPED", NEUTRAL),
			};
			ui.label(RichText::new(state_text).small().strong().color(color));
			if focused {
				ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
					ui.label(RichText::new("FOCUS").small().color(ACCENT));
				});
			}
		});

		draw_layer_clip(ui, app, ui_state, state.active_clip, state.selected_clip);

		if let Some(error) = &state.last_error {
			Frame::new()
				.fill(DANGER.gamma_multiply(0.18))
				.stroke(Stroke::new(1.0, DANGER))
				.corner_radius(CornerRadius::same(3))
				.inner_margin(Margin::symmetric(6, 3))
				.show(ui, |ui| {
					ui.set_width(ui.available_width());
					ui.label(RichText::new(error.as_str()).small().color(DANGER));
				});
		}

		ui.horizontal_wrapped(|ui| {
			let has_clip = state.has_clip();
			let playing = state.playback_state.is_playing();
			let play_label = if has_clip && playing { "Pause" } else { "Play" };
			let can_play = has_clip || state.selected_clip.is_some();
			if ui.add_enabled(can_play, Button::new(play_label)).clicked() {
				out.push(if has_clip {
					UiCommand::TogglePauseLayer(layer)
				} else {
					UiCommand::TriggerLayer(layer)
				});
			}
			if ui.add_enabled(has_clip, Button::new("Retrigger")).clicked() {
				out.push(UiCommand::RetriggerLayer(layer));
			}
			let mode_label = match state.playback_mode {
				PlaybackMode::Loop => "Loop",
				PlaybackMode::OneShot => "1-Shot",
			};
			if ui
				.add(Button::new(mode_label).selected(state.playback_mode == PlaybackMode::OneShot))
				.on_hover_text("Toggle loop / one-shot")
				.clicked()
			{
				out.push(UiCommand::ToggleLayerPlaybackMode(layer));
			}
			if ui.button("Clear").clicked() {
				out.push(UiCommand::ClearLayer(layer));
			}
		});

		let slider_width = (ui.available_width() - 90.0).max(60.0);
		ui.spacing_mut().slider_width = slider_width;

		ui.horizontal(|ui| {
			dim_label(ui, "Opacity");
			if let Some(opacity) = unit_slider(ui, state.opacity, 0.0..=1.0, "") {
				out.push(UiCommand::SetLayerOpacity { layer, opacity });
			}
		});
		ui.horizontal(|ui| {
			dim_label(ui, "Speed  ");
			ui.spacing_mut().slider_width = (slider_width - 34.0).max(40.0);
			if let Some(speed) = unit_slider(ui, state.speed, MIN_SPEED..=MAX_SPEED, "x") {
				out.push(UiCommand::SetLayerSpeed { layer, speed });
			}
			if ui.small_button("1x").on_hover_text("Reset to real time").clicked() {
				out.push(UiCommand::SetLayerSpeed { layer, speed: 1.0 });
			}
		});
		ui.horizontal(|ui| {
			dim_label(ui, "Blend  ");
			ComboBox::from_id_salt(("mur_layer_blend", layer.index()))
				.selected_text(state.blend_mode.label())
				.width(slider_width)
				.show_ui(ui, |ui| {
					for mode in BlendMode::ALL {
						let is_selected = state.blend_mode == mode;
						if ui.selectable_label(is_selected, mode.label()).clicked() && !is_selected {
							out.push(UiCommand::SetLayerBlendMode {
								layer,
								blend_mode: mode,
							});
						}
					}
				});
		});

		let progress = playback_progress(state.playhead_frame, state.frame_count);
		let frame_text = if state.frame_count > 0 {
			format!(
				"{} / {}",
				state.current_frame_index().saturating_add(1),
				state.frame_count
			)
		} else {
			"— / —".to_owned()
		};
		ui.add(
			ProgressBar::new(progress)
				.desired_height(12.0)
				.fill(if state.playback_state.is_playing() {
					ACCENT
				} else {
					NEUTRAL
				})
				.text(RichText::new(frame_text).monospace().size(10.0)),
		);
	});
}

/// Shows the layer's active clip (thumbnail, name, format) or an empty placeholder.
fn draw_layer_clip(ui: &mut Ui, app: &AppState, ui_state: &UiState, active: Option<ClipId>, selected: Option<ClipId>) {
	let thumb_size = vec2(96.0, 54.0);
	ui.horizontal(|ui| {
		let (rect, _) = ui.allocate_exact_size(thumb_size, Sense::hover());
		let clip = active.and_then(|id| app.clip_by_id(id));
		paint_thumbnail(ui, rect, clip, ui_state, active.is_some());

		ui.vertical(|ui| match (active, clip) {
			(Some(_), Some(clip)) => {
				ui.label(RichText::new(clip.name.as_str()).strong().color(TEXT))
					.on_hover_text(clip.path.display().to_string());
				ui.horizontal(|ui| {
					format_badge(ui, clip.kind);
					ui.label(RichText::new(clip.kind.badge()).small().color(TEXT_DIM));
				});
				ui.label(
					RichText::new(format!(
						"{}×{} @ {:.0}",
						clip.width,
						clip.height,
						finite_or_zero(clip.fps)
					))
					.monospace()
					.size(11.0)
					.color(TEXT_DIM),
				);
			}
			(Some(_), None) => {
				ui.label(RichText::new("Clip unloaded").color(WARN));
			}
			(None, _) => {
				ui.label(RichText::new("Empty Layer").color(TEXT_DIM));
			}
		});
	});

	if let Some(next) = selected.filter(|id| Some(*id) != active)
		&& let Some(clip) = app.clip_by_id(next)
	{
		ui.label(RichText::new(format!("Next ▸ {}", clip.name)).small().color(ACCENT));
	}
}

// ------------------------------------------------------------ media library

/// Frame of the media library panel, highlighted while files hover the window.
fn library_frame(drop_hover: bool) -> Frame {
	let stroke = if drop_hover {
		Stroke::new(2.0, ACCENT)
	} else {
		Stroke::new(1.0, HAIRLINE)
	};
	Frame::new()
		.fill(if drop_hover {
			ACCENT_DEEP.gamma_multiply(0.6)
		} else {
			BG_DEEP
		})
		.stroke(stroke)
		.inner_margin(Margin::symmetric(10, 8))
}

/// Header bar plus the scrollable clip grid.
fn draw_media_library(
	ui: &mut Ui,
	app: &AppState,
	ui_state: &UiState,
	pending_remove: &mut Option<(ClipId, Instant)>,
	now: Instant,
	out: &mut Vec<UiCommand>,
) {
	ui.horizontal(|ui| {
		section_title(ui, "MEDIA LIBRARY");
		if ui.button("+ Files").clicked() {
			out.push(UiCommand::OpenAddFilesDialog);
		}
		if ui.button("+ Folder").clicked() {
			out.push(UiCommand::OpenAddFolderDialog);
		}
		ui.label(
			RichText::new(format!("{} clips", app.clips.len()))
				.monospace()
				.color(TEXT_DIM),
		);
		ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
			pill(ui, &format!("Target {}", app.focused_layer), ACCENT, BG_DEEP)
				.on_hover_text("Clicking a tile selects it on this layer; double-click fires it");
			ui.label(
				RichText::new("Drop .mov / .png / .jpg / .toml here")
					.small()
					.color(TEXT_DIM),
			);
		});
	});

	if ui_state.hovered_files > 0 {
		Frame::new()
			.fill(ACCENT_DEEP)
			.stroke(Stroke::new(1.0, ACCENT))
			.corner_radius(CornerRadius::same(4))
			.inner_margin(Margin::symmetric(8, 4))
			.show(ui, |ui| {
				ui.set_width(ui.available_width());
				ui.label(
					RichText::new(format!(
						"Release to add {} file(s) to the library",
						ui_state.hovered_files
					))
					.strong()
					.color(ACCENT),
				);
			});
	}

	ui.add_space(4.0);
	egui::ScrollArea::vertical()
		.id_salt("mur_clip_grid")
		.auto_shrink([false, false])
		.show(ui, |ui| {
			if app.clips.is_empty() {
				ui.add_space(12.0);
				ui.vertical_centered(|ui| {
					ui.label(RichText::new("No clips loaded").color(TEXT_DIM));
					ui.label(
						RichText::new("Use + Files / + Folder or drag media onto this window")
							.small()
							.color(TEXT_DIM),
					);
				});
				return;
			}
			ui.horizontal_wrapped(|ui| {
				ui.spacing_mut().item_spacing = vec2(8.0, 8.0);
				for clip in &app.clips {
					draw_clip_tile(ui, app, ui_state, clip, pending_remove, now, out);
				}
			});
		});
}

/// One clip tile: thumbnail, badge, name, per-layer triggers, and unload button.
///
/// Unloading a clip that is live on any layer takes two clicks within
/// [`REMOVE_CONFIRM_WINDOW`](super::REMOVE_CONFIRM_WINDOW): the first turns "×"
/// into a warning-colored "×?".
fn draw_clip_tile(
	ui: &mut Ui,
	app: &AppState,
	ui_state: &UiState,
	clip: &ClipInfo,
	pending_remove: &mut Option<(ClipId, Instant)>,
	now: Instant,
	out: &mut Vec<UiCommand>,
) {
	let focused = app.focused_layer;
	let selected_on_focus = app.layer(focused).selected_clip == Some(clip.id);
	let live_layers: Vec<LayerIndex> = LayerIndex::ALL
		.into_iter()
		.filter(|layer| app.layer(*layer).active_clip == Some(clip.id))
		.collect();
	let stroke = if selected_on_focus {
		Stroke::new(2.0, ACCENT)
	} else if !live_layers.is_empty() {
		Stroke::new(1.5, ACCENT_CYAN.gamma_multiply(0.8))
	} else {
		Stroke::new(1.0, HAIRLINE)
	};

	ui.push_id(("mur_clip_tile", clip.id.0), |ui| {
		Frame::new()
			.fill(CARD)
			.stroke(stroke)
			.corner_radius(CornerRadius::same(5))
			.inner_margin(Margin::same(5))
			.show(ui, |ui| {
				ui.set_width(TILE_WIDTH);
				ui.vertical(|ui| {
					let (rect, response) = ui.allocate_exact_size(vec2(TILE_WIDTH, TILE_THUMB_HEIGHT), Sense::click());
					paint_thumbnail(ui, rect, Some(clip), ui_state, true);
					paint_tile_overlays(ui, rect, clip, &live_layers);

					let response = response.on_hover_ui(|ui| clip_tooltip(ui, clip));
					if response.double_clicked() {
						out.push(UiCommand::TriggerClipOnLayer {
							layer: focused,
							clip_id: clip.id,
						});
					} else if response.clicked() && !selected_on_focus {
						out.push(UiCommand::SelectLayerClip {
							layer: focused,
							clip_id: clip.id,
						});
					}

					let name_color = if clip.is_ready() { TEXT } else { DANGER };
					ui.add(egui::Label::new(RichText::new(clip.name.as_str()).small().color(name_color)).truncate());

					ui.horizontal(|ui| {
						ui.spacing_mut().item_spacing.x = 3.0;
						ui.spacing_mut().button_padding = vec2(5.0, 1.0);
						for layer in LayerIndex::ALL {
							let live = live_layers.contains(&layer);
							let button = Button::new(RichText::new(format!("{}", layer.number())).monospace())
								.selected(live)
								.min_size(vec2(22.0, 18.0));
							if ui
								.add_enabled(clip.is_ready(), button)
								.on_hover_text(format!("Fire on {layer}"))
								.clicked()
							{
								out.push(UiCommand::TriggerClipOnLayer {
									layer,
									clip_id: clip.id,
								});
							}
						}
						ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
							let is_live = !live_layers.is_empty();
							let armed = is_live && remove_confirmation_armed(*pending_remove, clip.id, now);
							let (remove, hint) = if armed {
								(
									Button::new(RichText::new("×?").strong().color(BG_DEEP))
										.fill(WARN)
										.min_size(vec2(26.0, 18.0)),
									"Click again to unload live clip",
								)
							} else if is_live {
								(
									Button::new(RichText::new("×").color(DANGER)).min_size(vec2(20.0, 18.0)),
									"Unload live clip (click twice)",
								)
							} else {
								(
									Button::new(RichText::new("×").color(DANGER)).min_size(vec2(20.0, 18.0)),
									"Unload clip",
								)
							};
							if ui.add(remove).on_hover_text(hint).clicked()
								&& handle_remove_click(pending_remove, clip.id, is_live, now)
							{
								out.push(UiCommand::RemoveClip(clip.id));
							}
						});
					});
				});
			});
	});
}

/// Format badge and LIVE markers painted over a tile thumbnail.
fn paint_tile_overlays(ui: &Ui, rect: Rect, clip: &ClipInfo, live_layers: &[LayerIndex]) {
	let painter = ui.painter_at(rect);
	let badge_text = format_badge_text(clip.kind);
	let badge_rect = Rect::from_min_size(rect.left_top() + vec2(4.0, 4.0), vec2(44.0, 15.0));
	painter.rect_filled(badge_rect, CornerRadius::same(3), Color32::from_black_alpha(200));
	painter.text(
		badge_rect.center(),
		Align2::CENTER_CENTER,
		badge_text,
		FontId::monospace(9.5),
		ACCENT,
	);

	if !live_layers.is_empty() {
		let label = live_layers
			.iter()
			.map(ToString::to_string)
			.collect::<Vec<_>>()
			.join(" ");
		let live_rect = Rect::from_min_size(rect.right_top() + vec2(-70.0, 4.0), vec2(66.0, 15.0));
		painter.rect_filled(live_rect, CornerRadius::same(3), ACCENT_CYAN);
		painter.text(
			live_rect.center(),
			Align2::CENTER_CENTER,
			label,
			FontId::monospace(9.5),
			BG_DEEP,
		);
	}
}

/// Hover tooltip with the clip's path, geometry, timing, and load status.
fn clip_tooltip(ui: &mut Ui, clip: &ClipInfo) {
	ui.label(RichText::new(clip.name.as_str()).strong());
	ui.label(RichText::new(clip.path.display().to_string()).small().color(TEXT_DIM));
	ui.label(
		RichText::new(format!(
			"{} · {} · {}×{} · {:.2} fps · {} frames",
			clip.kind.label(),
			clip.kind.badge(),
			clip.width,
			clip.height,
			finite_or_zero(clip.fps),
			clip.frame_count
		))
		.monospace(),
	);
	if clip.has_alpha {
		ui.label(RichText::new("Alpha channel").small().color(ACCENT));
	}
	if let ClipStatus::Error(message) = &clip.status {
		ui.label(RichText::new(format!("Error: {message}")).color(DANGER));
	}
	ui.label(
		RichText::new("Click: select on focused layer · Double-click: fire")
			.small()
			.color(TEXT_DIM),
	);
}

// ------------------------------------------------------------------ overlays

/// Bottom-right stack of active toasts with a clear button.
fn draw_toasts(ctx: &egui::Context, app: &AppState, out: &mut Vec<UiCommand>) {
	let toasts = app.toasts.toasts();
	if toasts.is_empty() {
		return;
	}
	egui::Area::new(Id::unique("mur_toast_stack"))
		.anchor(Align2::RIGHT_BOTTOM, vec2(-16.0, -16.0))
		.order(Order::Foreground)
		.interactable(true)
		.show(ctx, |ui| {
			ui.set_max_width(340.0);
			ui.with_layout(Layout::top_down(Align::Max), |ui| {
				if ui.small_button("Clear notifications").clicked() {
					out.push(UiCommand::ClearToasts);
				}
				for toast in toasts {
					let color = level_color(toast.level);
					Frame::new()
						.fill(CARD)
						.stroke(Stroke::new(1.0, color))
						.corner_radius(CornerRadius::same(5))
						.inner_margin(Margin::symmetric(10, 6))
						.show(ui, |ui| {
							ui.set_width(320.0);
							ui.horizontal(|ui| {
								ui.label(RichText::new(toast.level.label()).small().strong().color(color));
								ui.label(RichText::new(toast.title.as_str()).strong().color(TEXT));
							});
							if !toast.message.is_empty() {
								ui.label(RichText::new(toast.message.as_str()).small().color(TEXT_DIM));
							}
						});
				}
			});
		});
}

/// Log drawer listing retained log entries, newest first.
fn draw_log_drawer(ctx: &egui::Context, app: &AppState, out: &mut Vec<UiCommand>) {
	let mut open = true;
	egui::Window::new("Logs")
		.id(Id::unique("mur_log_drawer"))
		.open(&mut open)
		.collapsible(false)
		.resizable(true)
		.default_width(460.0)
		.anchor(Align2::RIGHT_TOP, vec2(-12.0, 48.0))
		.show(ctx, |ui| {
			ui.horizontal(|ui| {
				ui.label(
					RichText::new(format!("{} entries", app.toasts.logs().len()))
						.small()
						.color(TEXT_DIM),
				);
				ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
					if ui.button("Close").clicked() {
						out.push(UiCommand::ToggleLogsDrawer);
					}
				});
			});
			ui.separator();
			egui::ScrollArea::vertical()
				.id_salt("mur_log_scroll")
				.max_height(380.0)
				.auto_shrink([false, true])
				.show(ui, |ui| {
					if app.toasts.logs().is_empty() {
						ui.label(RichText::new("No log entries yet").color(TEXT_DIM));
					}
					for entry in app.toasts.logs().iter().rev() {
						ui.horizontal_wrapped(|ui| {
							ui.label(
								RichText::new(entry.timestamp.as_str())
									.monospace()
									.size(11.0)
									.color(TEXT_DIM),
							);
							ui.label(
								RichText::new(entry.level.label())
									.monospace()
									.size(11.0)
									.strong()
									.color(level_color(entry.level)),
							);
							ui.label(
								RichText::new(entry.source.as_str())
									.monospace()
									.size(11.0)
									.color(ACCENT),
							);
							ui.label(RichText::new(entry.message.as_str()).color(TEXT));
						});
					}
				});
		});
	if !open {
		out.push(UiCommand::ToggleLogsDrawer);
	}
}

/// Centered reference table of every keyboard shortcut.
fn draw_shortcut_overlay(ctx: &egui::Context, out: &mut Vec<UiCommand>) {
	let mut open = true;
	egui::Window::new("Keyboard Shortcuts")
		.id(Id::unique("mur_shortcut_overlay"))
		.open(&mut open)
		.collapsible(false)
		.resizable(false)
		.anchor(Align2::CENTER_CENTER, Vec2::ZERO)
		.show(ctx, |ui| {
			egui::ScrollArea::vertical()
				.id_salt("mur_shortcut_scroll")
				.max_height(520.0)
				.show(ui, |ui| {
					egui::Grid::new("mur_shortcut_grid")
						.num_columns(2)
						.striped(true)
						.spacing(vec2(18.0, 4.0))
						.show(ui, |ui| {
							for entry in SHORTCUT_LEGEND {
								ui.label(RichText::new(entry.keys).monospace().color(ACCENT));
								ui.label(RichText::new(entry.description).color(TEXT));
								ui.end_row();
							}
						});
				});
			ui.separator();
			ui.vertical_centered(|ui| {
				if ui.button("Close").clicked() {
					out.push(UiCommand::ToggleShortcutsOverlay);
				}
			});
		});
	if !open {
		out.push(UiCommand::ToggleShortcutsOverlay);
	}
}

/// High-contrast quit guard over a dimmed, input-blocking backdrop.
fn draw_exit_modal(ctx: &egui::Context, out: &mut Vec<UiCommand>) {
	egui::Modal::new(Id::unique("mur_exit_confirm"))
		.backdrop_color(Color32::from_black_alpha(200))
		.frame(
			Frame::new()
				.fill(CARD)
				.stroke(Stroke::new(2.0, DANGER))
				.corner_radius(CornerRadius::same(8))
				.inner_margin(Margin::same(20)),
		)
		.show(ctx, |ui| {
			ui.set_width(380.0);
			ui.vertical_centered(|ui| {
				ui.label(RichText::new("Exit MUR?").strong().size(22.0).color(TEXT));
				ui.add_space(4.0);
				ui.label(RichText::new("Live VJ output will stop.").size(15.0).color(WARN));
				ui.add_space(6.0);
				ui.label(
					RichText::new("Press Cmd+Q / Ctrl+Q / Alt+F4 again or click Quit MUR to exit. Esc cancels.")
						.size(13.0)
						.color(TEXT_DIM),
				);
				ui.add_space(14.0);
			});
			ui.columns_const(|columns: &mut [Ui; 2]| {
				let [cancel_column, confirm_column] = columns;
				let cancel = Button::new(RichText::new("Cancel (Esc)").size(15.0))
					.min_size(vec2(cancel_column.available_width(), 36.0));
				if cancel_column.add(cancel).clicked() {
					out.push(UiCommand::CancelExit);
				}
				let confirm = Button::new(RichText::new("Quit MUR").strong().size(15.0).color(Color32::WHITE))
					.fill(DANGER)
					.min_size(vec2(confirm_column.available_width(), 36.0));
				if confirm_column.add(confirm).clicked() {
					out.push(UiCommand::ConfirmExit);
				}
			});
		});
}

// ------------------------------------------------------------------- helpers

/// Card frame used by the FX strip and layer columns; accent border when focused.
fn card_frame(focused: bool) -> Frame {
	Frame::new()
		.fill(CARD)
		.stroke(if focused {
			Stroke::new(2.0, ACCENT)
		} else {
			Stroke::new(1.0, HAIRLINE)
		})
		.corner_radius(CornerRadius::same(6))
		.inner_margin(Margin::same(10))
}

/// Small uppercase section heading.
fn section_title(ui: &mut Ui, text: &str) {
	ui.label(RichText::new(text).small().strong().color(TEXT_DIM));
}

/// Dimmed parameter label.
fn dim_label(ui: &mut Ui, text: &str) {
	ui.label(RichText::new(text).color(TEXT_DIM));
}

/// Compact rounded status badge with explicit text and fill colours.
fn pill(ui: &mut Ui, text: &str, fill: Color32, text_color: Color32) -> egui::Response {
	Frame::new()
		.fill(fill)
		.stroke(Stroke::new(1.0, text_color.gamma_multiply(0.7)))
		.corner_radius(CornerRadius::same(9))
		.inner_margin(Margin::symmetric(8, 2))
		.show(ui, |ui| {
			ui.label(RichText::new(text).small().strong().color(text_color));
		})
		.response
}

/// Effect on/off pill; returns `true` when it was clicked.
fn fx_toggle(ui: &mut Ui, label: &str, enabled: bool) -> bool {
	let text = RichText::new(label)
		.strong()
		.small()
		.color(if enabled { BG_DEEP } else { TEXT });
	let button = Button::new(text)
		.fill(if enabled { ACCENT } else { CARD_RAISED })
		.selected(enabled)
		.min_size(vec2(84.0, 20.0));
	ui.add(button).clicked()
}

/// Slider over an `f32`; returns the new value only when the operator changed it.
///
/// Non-finite inputs are displayed as the range start so a corrupted value can
/// never poison the widget, and no command is emitted until the slider moves.
fn unit_slider(ui: &mut Ui, value: f32, range: std::ops::RangeInclusive<f32>, suffix: &str) -> Option<f32> {
	let mut edited = if value.is_finite() {
		value.clamp(*range.start(), *range.end())
	} else {
		*range.start()
	};
	let response = ui.add(Slider::new(&mut edited, range).fixed_decimals(2).suffix(suffix));
	(response.changed() && edited.is_finite()).then_some(edited)
}

/// Paints a clip thumbnail (aspect-fit) or a status placeholder into `rect`.
fn paint_thumbnail(ui: &Ui, rect: Rect, clip: Option<&ClipInfo>, ui_state: &UiState, show_status: bool) {
	let painter = ui.painter_at(rect);
	painter.rect_filled(rect, CornerRadius::same(3), Color32::BLACK);

	match clip {
		Some(clip) => {
			if let Some(texture) = ui_state.thumbnail(clip.id) {
				let fitted = fit_rect(rect, texture.size_vec2());
				let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
				painter.image(texture.id(), fitted, uv, Color32::WHITE);
			} else if show_status {
				let (text, color) = match &clip.status {
					ClipStatus::Error(_) => ("Error", DANGER),
					ClipStatus::Ready => ("Loading…", TEXT_DIM),
				};
				painter.text(
					rect.center(),
					Align2::CENTER_CENTER,
					text,
					FontId::proportional(12.0),
					color,
				);
			}
			if !clip.is_ready() {
				painter.rect_stroke(
					rect,
					CornerRadius::same(3),
					Stroke::new(1.0, DANGER),
					StrokeKind::Inside,
				);
			}
		}
		None => {
			if show_status {
				painter.text(
					rect.center(),
					Align2::CENTER_CENTER,
					"Missing",
					FontId::proportional(11.0),
					WARN,
				);
			}
		}
	}
	painter.rect_stroke(
		rect,
		CornerRadius::same(3),
		Stroke::new(1.0, HAIRLINE),
		StrokeKind::Outside,
	);
}

/// Largest rectangle with the aspect ratio of `size` centered inside `outer`.
fn fit_rect(outer: Rect, size: Vec2) -> Rect {
	if !(size.x > 0.0 && size.y > 0.0) || !size.x.is_finite() || !size.y.is_finite() {
		return outer;
	}
	let scale = (outer.width() / size.x).min(outer.height() / size.y);
	if !scale.is_finite() || scale <= 0.0 {
		return outer;
	}
	Rect::from_center_size(outer.center(), size * scale)
}

/// Normalized playhead position in `0.0..=1.0`.
fn playback_progress(playhead_frame: f64, frame_count: u32) -> f32 {
	if frame_count == 0 || !playhead_frame.is_finite() {
		return 0.0;
	}
	(playhead_frame / f64::from(frame_count)).clamp(0.0, 1.0) as f32
}

/// Short codec badge shown on tiles and layer strips.
fn format_badge_text(kind: ClipKind) -> &'static str {
	match kind {
		ClipKind::Hap => "HAP",
		ClipKind::HapAlpha => "HAP A",
		ClipKind::HapQ => "HAP Q",
		ClipKind::ImageRgba => "IMG",
	}
}

/// Renders the codec badge as a pill.
fn format_badge(ui: &mut Ui, kind: ClipKind) {
	pill(ui, format_badge_text(kind), BG_DEEP, ACCENT);
}

/// Colour used for a toast or log severity.
fn level_color(level: ToastLevel) -> Color32 {
	match level {
		ToastLevel::Info => ACCENT,
		ToastLevel::Warning => WARN,
		ToastLevel::Error => DANGER,
	}
}

/// Colour of the FPS readout: green near 60, amber when degraded, red when failing.
fn fps_color(fps: f32) -> Color32 {
	if !fps.is_finite() || fps <= 0.0 {
		TEXT_DIM
	} else if fps >= 58.0 {
		GOOD
	} else if fps >= 50.0 {
		WARN
	} else {
		DANGER
	}
}

/// Replaces NaN/infinity with `0.0` so readouts never print garbage.
fn finite_or_zero(value: f32) -> f32 {
	if value.is_finite() { value } else { 0.0 }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use std::path::PathBuf;

	use super::*;
	use crate::model::{MonitorSummary, PlaybackState, ThumbnailImage};

	/// Runs `frames` headless passes and returns the commands of the last one.
	fn run_frames(app: &AppState, ui_state: &mut UiState, mirror: Option<TextureId>, frames: usize) -> Vec<UiCommand> {
		let ctx = egui::Context::default();
		run_frames_on(&ctx, app, ui_state, mirror, frames)
	}

	fn run_frames_on(
		ctx: &egui::Context,
		app: &AppState,
		ui_state: &mut UiState,
		mirror: Option<TextureId>,
		frames: usize,
	) -> Vec<UiCommand> {
		ctx.set_missing_glyph_policy(egui::MissingGlyphPolicy::Panic);
		let mut last = Vec::new();
		for _ in 0..frames {
			let input = egui::RawInput {
				screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1600.0, 1000.0))),
				..Default::default()
			};
			let mut output = ctx.run_pass(input, |ctx| {
				ui_state.sync_thumbnails(ctx, &app.clips);
				last = draw_control_ui(ctx, app, ui_state, mirror);
			});
			let primitives = ctx.tessellate(std::mem::take(&mut output.shapes), output.pixels_per_point);
			assert!(!primitives.is_empty(), "the control UI should paint something");
			output.drop_without_applying_deltas();
		}
		last
	}

	fn thumbnail(width: u32, height: u32) -> ThumbnailImage {
		ThumbnailImage {
			width,
			height,
			rgba: vec![128; (width * height * 4) as usize],
		}
	}

	fn clip(id: u64, kind: ClipKind, status: ClipStatus, thumb: Option<ThumbnailImage>) -> ClipInfo {
		ClipInfo {
			id: ClipId(id),
			path: PathBuf::from(format!("/clips/clip_{id}.mov")),
			name: format!("clip_{id}.mov"),
			kind,
			width: 1920,
			height: 1080,
			fps: 60.0,
			frame_count: 240,
			has_alpha: matches!(kind, ClipKind::HapAlpha),
			status,
			thumbnail: thumb,
		}
	}

	fn populated_state() -> AppState {
		let mut app = AppState {
			available_monitors: monitors(),
			..AppState::default()
		};
		app.output.monitor_name = Some("Projector".into());
		populate(&mut app);
		app
	}

	fn monitors() -> Vec<MonitorSummary> {
		vec![
			MonitorSummary {
				id: "0".into(),
				name: "Built-in Retina".into(),
				width: 3024,
				height: 1964,
				scale_factor: 2.0,
				is_primary: true,
			},
			MonitorSummary {
				id: "1".into(),
				name: "Projector".into(),
				width: 1920,
				height: 1080,
				scale_factor: 1.0,
				is_primary: false,
			},
		]
	}

	fn populate(app: &mut AppState) {
		app.output.enabled = true;
		app.output_status = OutputRuntimeStatus::Active {
			monitor_name: "Projector".into(),
			width: 1920,
			height: 1080,
		};
		app.midi_ports = vec!["LCXL3 MIDI In".into(), "IAC Bus 1".into()];
		app.selected_midi_port = Some("LCXL3 MIDI In".into());
		app.midi_status = MidiConnectionStatus::Connected {
			port_name: "LCXL3 MIDI In".into(),
		};
		app.perf.fps = 59.8;
		app.perf.frame_time_ms = 16.7;
		app.perf.dropped_frames = 3;

		app.clips = vec![
			clip(1, ClipKind::Hap, ClipStatus::Ready, Some(thumbnail(16, 9))),
			clip(2, ClipKind::HapAlpha, ClipStatus::Ready, Some(thumbnail(8, 8))),
			clip(3, ClipKind::HapQ, ClipStatus::Ready, None),
			clip(4, ClipKind::ImageRgba, ClipStatus::Error("corrupt atom".into()), None),
			clip(
				5,
				ClipKind::Hap,
				ClipStatus::Ready,
				Some(ThumbnailImage {
					width: 4,
					height: 4,
					rgba: vec![0; 3],
				}),
			),
		];

		assert!(app.trigger_clip_on_layer(LayerIndex::ALL[0], ClipId(1)));
		assert!(app.trigger_clip_on_layer(LayerIndex::ALL[1], ClipId(2)));
		app.layers[1].playback_state = PlaybackState::Paused;
		app.layers[1].playhead_frame = 120.0;
		app.layers[1].blend_mode = BlendMode::Screen;
		assert!(!app.trigger_clip_on_layer(LayerIndex::ALL[2], ClipId(4)));
		app.layers[3].selected_clip = Some(ClipId(3));
		app.focused_layer = LayerIndex::ALL[1];

		app.effects.blackout = true;
		app.effects.rgb_shift_enabled = true;
		app.effects.rgb_shift_intensity = 0.4;
		app.effects.strobe_speed_hz = f32::NAN;

		app.push_toast(ToastLevel::Info, "Loaded", "clip_1.mov", 0.0);
		app.push_toast(ToastLevel::Warning, "MIDI", "Reconnecting", 0.0);
		app.toasts.push_log(crate::error::LogEntry::new(
			"12:00:00",
			ToastLevel::Error,
			"decoder",
			"bad frame",
		));
	}

	#[test]
	fn empty_state_renders_without_commands() {
		let app = AppState::default();
		let mut ui_state = UiState::new();
		let commands = run_frames(&app, &mut ui_state, None, 3);
		assert!(commands.is_empty(), "unexpected commands: {commands:?}");
		assert!(ui_state.theme_applied());
		assert_eq!(ui_state.thumbnail_count(), 0);
	}

	#[test]
	fn populated_state_renders_and_syncs_thumbnails() {
		let mut app = populated_state();
		let mut ui_state = UiState::new();
		let ctx = egui::Context::default();
		let mirror = Some(TextureId::Managed(0));

		let commands = run_frames_on(&ctx, &app, &mut ui_state, mirror, 3);
		assert!(commands.is_empty(), "unexpected commands: {commands:?}");
		// Only clips 1 and 2 carry valid thumbnails; 5 is malformed and skipped.
		assert_eq!(ui_state.thumbnail_count(), 2);
		assert!(ui_state.thumbnail(ClipId(1)).is_some());
		assert!(ui_state.thumbnail(ClipId(2)).is_some());
		assert!(ui_state.thumbnail(ClipId(5)).is_none());

		let first_id = ui_state.thumbnail(ClipId(1)).map(egui::TextureHandle::id);
		app.clips.retain(|clip| clip.id != ClipId(2));
		run_frames_on(&ctx, &app, &mut ui_state, mirror, 1);
		assert_eq!(ui_state.thumbnail_count(), 1);
		assert!(ui_state.thumbnail(ClipId(2)).is_none());
		// Existing textures are reused rather than re-uploaded.
		assert_eq!(ui_state.thumbnail(ClipId(1)).map(egui::TextureHandle::id), first_id);

		app.clips.clear();
		run_frames_on(&ctx, &app, &mut ui_state, None, 1);
		assert_eq!(ui_state.thumbnail_count(), 0);
	}

	#[test]
	fn overlays_and_exit_guard_render() {
		let mut app = populated_state();
		app.show_shortcuts = true;
		app.show_logs = true;
		let mut ui_state = UiState::new();
		let commands = run_frames(&app, &mut ui_state, None, 3);
		assert!(commands.is_empty(), "unexpected commands: {commands:?}");

		app.exit_confirm_open = true;
		let commands = run_frames(&app, &mut ui_state, None, 3);
		assert!(commands.is_empty(), "unexpected commands: {commands:?}");

		let empty = AppState {
			exit_confirm_open: true,
			show_logs: true,
			..AppState::default()
		};
		let commands = run_frames(&empty, &mut UiState::new(), None, 2);
		assert!(commands.is_empty(), "unexpected commands: {commands:?}");
	}

	#[test]
	fn drop_hover_and_waiting_monitor_render() {
		let mut app = populated_state();
		app.output_status = OutputRuntimeStatus::WaitingForMonitor {
			expected_name: Some("Projector".into()),
		};
		app.output.monitor_name = Some("Unplugged HDMI".into());
		app.midi_status = MidiConnectionStatus::Reconnecting {
			last_port: Some("LCXL3 MIDI In".into()),
			error: Some("port vanished".into()),
		};
		let mut ui_state = UiState::new();
		ui_state.hovered_files = 3;
		let commands = run_frames(&app, &mut ui_state, None, 2);
		assert!(commands.is_empty(), "unexpected commands: {commands:?}");

		app.output_status = OutputRuntimeStatus::Error("surface lost".into());
		app.midi_status = MidiConnectionStatus::Scanning;
		app.clips.clear();
		let commands = run_frames(&app, &mut ui_state, None, 2);
		assert!(commands.is_empty(), "unexpected commands: {commands:?}");
	}

	#[test]
	fn helpers_are_panic_free_on_degenerate_input() {
		let outer = Rect::from_min_size(Pos2::ZERO, vec2(160.0, 90.0));
		assert_eq!(fit_rect(outer, Vec2::ZERO), outer);
		assert_eq!(fit_rect(outer, vec2(f32::NAN, 1.0)), outer);
		let square = fit_rect(outer, vec2(10.0, 10.0));
		assert!((square.width() - 90.0).abs() < 1e-3);
		assert!((square.height() - 90.0).abs() < 1e-3);

		assert_eq!(playback_progress(f64::NAN, 10), 0.0);
		assert_eq!(playback_progress(5.0, 0), 0.0);
		assert!((playback_progress(5.0, 10) - 0.5).abs() < 1e-6);
		assert_eq!(playback_progress(50.0, 10), 1.0);

		assert_eq!(format_badge_text(ClipKind::HapAlpha), "HAP A");
		assert_eq!(format_badge_text(ClipKind::ImageRgba), "IMG");
		assert_eq!(finite_or_zero(f32::INFINITY), 0.0);
		assert_eq!(fps_color(60.0), GOOD);
		assert_eq!(fps_color(f32::NAN), TEXT_DIM);
	}

	#[test]
	fn all_ui_text_and_symbols_have_installed_glyphs() {
		let defaults = egui::FontDefinitions::default();
		assert!(
			!defaults.font_data.is_empty(),
			"egui default_fonts must embed built-in font data"
		);

		let ctx = egui::Context::default();
		apply_vj_theme(&ctx);
		ctx.set_missing_glyph_policy(egui::MissingGlyphPolicy::Panic);

		let mut samples: Vec<String> =
			vec!["MUR ● LIVE 1920×1080 · 60.0 fps · 16.7 ms — / — Next ▸ clip × ×? Loading… –".to_owned()];
		for entry in SHORTCUT_LEGEND {
			samples.push(entry.keys.to_owned());
			samples.push(entry.description.to_owned());
		}

		let output = ctx.run_pass(egui::RawInput::default(), |ctx| {
			ctx.fonts_mut(|fonts| {
				for sample in &samples {
					let _ = fonts.layout_no_wrap(sample.clone(), FontId::proportional(14.0), TEXT);
					let _ = fonts.layout_no_wrap(sample.clone(), FontId::monospace(12.5), TEXT);
				}
			});
		});
		output.drop_without_applying_deltas();
	}
}
