//! Platform integration: display identity, output window placement, and the quit guard.
//!
//! The cross-platform part of this module turns winit [`MonitorHandle`]s into stable
//! [`MonitorSummary`] values, matches the operator's saved display choice against the
//! currently connected monitors, and exposes small facades that dispatch to the
//! `macos` and `windows` back ends:
//!
//! - [`configure_output_window`] places the output window borderless over a whole
//!   monitor (above the macOS menu bar / Dock, DPI-aware on Windows).
//! - [`install_exit_guard`] / [`poll_quit_requested`] turn `Cmd+Q` and the Dock "Quit"
//!   item on macOS into a request that the control UI confirms, instead of terminating
//!   the process immediately. On Windows `Alt+F4` and the close button already arrive as
//!   `WindowEvent::CloseRequested`, so nothing needs intercepting there.
//! - [`prevent_app_nap`] keeps macOS App Nap from throttling the render loop.
//!
//! Monitor handles are only queried through crash-safe paths: on Windows winit's
//! `MonitorHandle::name` / `size` panic for a display that was just unplugged, so the
//! Windows back end reads `GetMonitorInfoW` itself and disconnected monitors are skipped
//! before anything else touches them. [`MonitorDescriptor`] caches the geometry, so
//! callers never have to query a possibly stale handle again.

#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "windows")]
pub mod windows;

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event_loop::ActiveEventLoop;
use winit::monitor::MonitorHandle;
use winit::window::Window;

use crate::error::{MurError, RenderError};
use crate::model::MonitorSummary;

/// Errors raised by the platform integration layer.
#[derive(Debug, thiserror::Error)]
pub enum PlatformError {
	/// A native UI call was attempted off the main thread.
	#[error("platform window call must run on the main thread")]
	NotMainThread,
	/// The native window handle could not be obtained or has an unexpected type.
	#[error("native window handle unavailable: {0}")]
	WindowHandle(String),
	/// The requested monitor is no longer connected.
	#[error("target monitor `{0}` is not connected")]
	MonitorUnavailable(String),
	/// A native platform call failed.
	#[error("native platform call failed: {0}")]
	Native(String),
}

impl From<PlatformError> for RenderError {
	fn from(error: PlatformError) -> Self {
		Self::InvalidTarget(error.to_string())
	}
}

impl From<PlatformError> for MurError {
	fn from(error: PlatformError) -> Self {
		Self::Render(error.into())
	}
}

/// Access to the [`MonitorSummary`] of a monitor-like value, used by the matching helpers.
pub trait SummarizedMonitor {
	/// Returns the summary describing this monitor.
	fn summary(&self) -> &MonitorSummary;
}

impl SummarizedMonitor for MonitorSummary {
	fn summary(&self) -> &MonitorSummary {
		self
	}
}

/// Display name reported for a monitor that vanished while it was being queried.
pub const DISCONNECTED_DISPLAY_NAME: &str = "unnamed or disconnected display";

/// Separator plus prefix of the native display id component of a [`monitor_id`].
const NATIVE_ID_PREFIX: &str = "cg:";

/// A connected monitor: the live winit handle paired with its extracted summary.
#[derive(Debug, Clone, PartialEq)]
pub struct MonitorDescriptor {
	/// Live winit handle, valid while the monitor stays connected.
	pub handle: MonitorHandle,
	/// Serializable description shown in the control UI.
	pub summary: MonitorSummary,
	/// Top-left corner in physical desktop pixels, captured with [`MonitorDescriptor::summary`].
	///
	/// Use this (and `summary.width` / `summary.height`) instead of querying
	/// [`MonitorDescriptor::handle`], which may already be stale.
	pub position: (i32, i32),
}

impl MonitorDescriptor {
	/// Builds a descriptor for `handle`, or `None` when the monitor is no longer connected.
	///
	/// `primary` is only consulted on platforms where the back end cannot tell the primary
	/// display itself (neither macOS nor Windows).
	pub fn probe(handle: MonitorHandle, primary: Option<&MonitorHandle>) -> Option<Self> {
		let probe = probe_monitor(&handle)?;
		let is_primary = probe
			.is_primary
			.unwrap_or_else(|| primary.is_some_and(|primary| primary == &handle));
		let summary = probe.summary(is_primary);
		Some(Self {
			handle,
			summary,
			position: (probe.position.x, probe.position.y),
		})
	}

	/// Top-left corner in physical desktop pixels.
	pub fn position(&self) -> (i32, i32) {
		self.position
	}
}

impl SummarizedMonitor for MonitorDescriptor {
	fn summary(&self) -> &MonitorSummary {
		&self.summary
	}
}

/// Raw facts about a connected monitor gathered through crash-safe platform calls.
#[derive(Debug, Clone, PartialEq)]
struct MonitorProbe {
	/// OS display name, if any.
	name: Option<String>,
	/// Size in physical pixels.
	size: PhysicalSize<u32>,
	/// Top-left corner in physical desktop pixels.
	position: PhysicalPosition<i32>,
	/// HiDPI scale factor (already sanitized).
	scale_factor: f64,
	/// Primary flag when the back end knows it.
	is_primary: Option<bool>,
	/// Stable native display id (`CGDirectDisplayID` on macOS).
	native_id: Option<u32>,
}

impl MonitorProbe {
	fn summary(&self, is_primary: bool) -> MonitorSummary {
		let name = self.name.as_deref();
		MonitorSummary {
			id: monitor_id_with_native(name, self.size, self.position, self.native_id),
			name: monitor_display_name(name, self.size),
			width: self.size.width,
			height: self.size.height,
			scale_factor: self.scale_factor,
			is_primary,
		}
	}
}

/// Queries `handle` without ever touching winit calls that panic on stale handles.
///
/// Returns `None` when the monitor is known to be disconnected.
fn probe_monitor(handle: &MonitorHandle) -> Option<MonitorProbe> {
	#[cfg(target_os = "windows")]
	{
		// Bounds first: they double as the "still connected" check, and winit's own
		// `name()` / `size()` would unwrap a failed `GetMonitorInfoW`.
		let info = windows::monitor_info(handle)?;
		let (left, top, _, _) = info.bounds.monitor;
		Some(MonitorProbe {
			name: info.device_name,
			size: PhysicalSize::new(info.bounds.width(), info.bounds.height()),
			position: PhysicalPosition::new(left, top),
			scale_factor: sanitize_scale_factor(handle.scale_factor()),
			is_primary: Some(info.bounds.is_primary),
			native_id: None,
		})
	}
	#[cfg(target_os = "macos")]
	{
		use winit::platform::macos::MonitorHandleExtMacOS;
		let position = handle.position();
		let screen = macos::screen_info(handle);
		let name = screen
			.as_ref()
			.and_then(|screen| screen.localized_name.clone())
			.or_else(|| handle.name());
		let is_primary = screen.map_or(position == PhysicalPosition::new(0, 0), |screen| screen.is_primary);
		Some(MonitorProbe {
			name,
			size: handle.size(),
			position,
			scale_factor: sanitize_scale_factor(handle.scale_factor()),
			is_primary: Some(is_primary),
			native_id: Some(handle.native_id()),
		})
	}
	#[cfg(not(any(target_os = "macos", target_os = "windows")))]
	{
		Some(MonitorProbe {
			name: handle.name(),
			size: handle.size(),
			position: handle.position(),
			scale_factor: sanitize_scale_factor(handle.scale_factor()),
			is_primary: None,
			native_id: None,
		})
	}
}

/// Returns a usable scale factor, replacing non-finite or non-positive values with `1.0`.
pub fn sanitize_scale_factor(scale_factor: f64) -> f64 {
	if scale_factor.is_finite() && scale_factor > 0.0 {
		scale_factor
	} else {
		1.0
	}
}

/// Returns the trimmed OS monitor name, or `None` when it is missing or blank.
fn clean_monitor_name(name: Option<&str>) -> Option<&str> {
	name.map(str::trim).filter(|name| !name.is_empty())
}

/// Builds the human readable monitor name, falling back to `"Display WxH"`.
pub fn monitor_display_name(name: Option<&str>, size: PhysicalSize<u32>) -> String {
	match clean_monitor_name(name) {
		Some(name) => name.to_owned(),
		None => format!("Display {}x{}", size.width, size.height),
	}
}

/// Builds a stable monitor identifier of the form `name|WxH|x,y`.
///
/// The identifier distinguishes identical monitor models by their desktop position;
/// [`find_matching_monitor`] additionally falls back to the name so that a resolution
/// change or rearrangement still finds the operator's display.
pub fn monitor_id(name: Option<&str>, size: PhysicalSize<u32>, position: PhysicalPosition<i32>) -> String {
	monitor_id_with_native(name, size, position, None)
}

/// Builds a monitor identifier that also carries a stable native display id.
///
/// With `native_id = Some(id)` (the `CGDirectDisplayID` on macOS) the identifier is
/// `name|WxH|x,y|cg:<id>`, so the same physical display is still recognized after a
/// resolution change or rearrangement, and two displays of the same model never collide.
/// Without it this is exactly [`monitor_id`].
pub fn monitor_id_with_native(
	name: Option<&str>,
	size: PhysicalSize<u32>,
	position: PhysicalPosition<i32>,
	native_id: Option<u32>,
) -> String {
	let mut id = format!(
		"{}|{}x{}|{},{}",
		monitor_display_name(name, size),
		size.width,
		size.height,
		position.x,
		position.y
	);
	if let Some(native_id) = native_id {
		id.push('|');
		id.push_str(NATIVE_ID_PREFIX);
		id.push_str(&native_id.to_string());
	}
	id
}

/// Extracts the [`MonitorSummary`] of `handle`; `primary` is the OS primary monitor, if known.
///
/// Uses the same crash-safe queries as [`collect_monitors`]. A monitor that disconnected
/// in the meantime is summarized as [`DISCONNECTED_DISPLAY_NAME`] with a zero size.
pub fn monitor_summary(handle: &MonitorHandle, primary: Option<&MonitorHandle>) -> MonitorSummary {
	match MonitorDescriptor::probe(handle.clone(), primary) {
		Some(descriptor) => descriptor.summary,
		None => {
			let size = PhysicalSize::new(0, 0);
			MonitorSummary {
				id: monitor_id(Some(DISCONNECTED_DISPLAY_NAME), size, PhysicalPosition::new(0, 0)),
				name: DISCONNECTED_DISPLAY_NAME.to_owned(),
				width: 0,
				height: 0,
				scale_factor: 1.0,
				is_primary: false,
			}
		}
	}
}

/// Enumerates every connected monitor with its summary, in OS order.
///
/// Monitors that disconnect during enumeration are skipped. On macOS and Windows the
/// primary display is determined by the back end; `ActiveEventLoop::primary_monitor` is
/// only used elsewhere (it panics inside winit on macOS when the main display vanishes).
pub fn collect_monitors(event_loop: &ActiveEventLoop) -> Vec<MonitorDescriptor> {
	#[cfg(any(target_os = "macos", target_os = "windows"))]
	let primary: Option<MonitorHandle> = None;
	#[cfg(not(any(target_os = "macos", target_os = "windows")))]
	let primary = event_loop.primary_monitor();
	event_loop
		.available_monitors()
		.filter_map(|handle| MonitorDescriptor::probe(handle, primary.as_ref()))
		.collect()
}

/// Returns the descriptor in `monitors` whose live handle equals `handle`.
///
/// Used to identify the monitor of a window (`Window::current_monitor`) without querying
/// the handle itself, which may be stale.
pub fn find_monitor_by_handle<'a>(
	monitors: &'a [MonitorDescriptor],
	handle: Option<&MonitorHandle>,
) -> Option<&'a MonitorDescriptor> {
	let handle = handle?;
	monitors.iter().find(|monitor| &monitor.handle == handle)
}

/// Clones the summaries out of `monitors`, e.g. for `AppState::available_monitors`.
pub fn summaries<T: SummarizedMonitor>(monitors: &[T]) -> Vec<MonitorSummary> {
	monitors.iter().map(|monitor| monitor.summary().clone()).collect()
}

/// Finds the monitor the operator selected.
///
/// Same as [`find_matching_monitor_excluding`] without an excluded monitor.
pub fn find_matching_monitor<'a, T: SummarizedMonitor>(monitors: &'a [T], selected: Option<&str>) -> Option<&'a T> {
	find_matching_monitor_excluding(monitors, selected, None)
}

/// Finds the monitor the operator selected, never falling back onto `exclude_id`.
///
/// `selected` may be either a [`MonitorSummary::id`] or a [`MonitorSummary::name`]. The
/// lookup tries, in order:
///
/// 1. exact id — an intentional choice, honored even for the excluded monitor;
/// 2. the same native display id (see [`monitor_id_with_native`]) with a matching name,
///    i.e. the very same physical display after a resolution change or rearrangement;
/// 3. exact name, then case-insensitive name;
/// 4. for a stale id, its name part, preferring the monitor at the same desktop position.
///
/// Steps 3 and 4 skip the monitor whose id equals `exclude_id` (the control window's
/// display): when an external display that shares its model name with the control
/// display is unplugged, the output must wait for it instead of covering the control UI.
/// `None` is returned when nothing is selected or the selected display is disconnected,
/// so the caller can enter `WaitingForMonitor` and resume once the display reappears.
pub fn find_matching_monitor_excluding<'a, T: SummarizedMonitor>(
	monitors: &'a [T],
	selected: Option<&str>,
	exclude_id: Option<&str>,
) -> Option<&'a T> {
	let selected = selected.map(str::trim).filter(|selected| !selected.is_empty())?;
	if let Some(found) = monitors.iter().find(|monitor| monitor.summary().id == selected) {
		return Some(found);
	}
	let parsed = parse_monitor_id(selected);
	if let Some(parsed) = parsed.as_ref()
		&& let Some(native) = parsed.native
		&& let Some(found) = monitors.iter().find(|monitor| {
			let summary = monitor.summary();
			names_match(&summary.name, parsed.name)
				&& parse_monitor_id(&summary.id).is_some_and(|candidate| candidate.native == Some(native))
		}) {
		return Some(found);
	}

	let candidates: Vec<&'a T> = monitors
		.iter()
		.filter(|monitor| exclude_id.is_none_or(|exclude| monitor.summary().id != exclude))
		.collect();
	if let Some(found) = find_by_name(&candidates, selected) {
		return Some(found);
	}
	let parsed = parsed?;
	candidates
		.iter()
		.copied()
		.filter(|monitor| names_match(&monitor.summary().name, parsed.name))
		.find(|monitor| parse_monitor_id(&monitor.summary().id).is_some_and(|id| id.position == parsed.position))
		.or_else(|| find_by_name(&candidates, parsed.name))
}

/// Finds a monitor by exact name, then by case-insensitive name.
fn find_by_name<'a, T: SummarizedMonitor>(monitors: &[&'a T], name: &str) -> Option<&'a T> {
	monitors
		.iter()
		.copied()
		.find(|monitor| monitor.summary().name == name)
		.or_else(|| {
			monitors
				.iter()
				.copied()
				.find(|monitor| names_match(&monitor.summary().name, name))
		})
}

/// Case-insensitive, whitespace-trimmed name comparison.
fn names_match(left: &str, right: &str) -> bool {
	left.trim().to_lowercase() == right.trim().to_lowercase()
}

/// Components of a [`monitor_id`] / [`monitor_id_with_native`] string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParsedMonitorId<'a> {
	/// Display name part.
	name: &'a str,
	/// `x,y` position part.
	position: &'a str,
	/// Native display id digits, when present.
	native: Option<&'a str>,
}

/// Splits a monitor id into its parts; `None` for plain names.
fn parse_monitor_id(id: &str) -> Option<ParsedMonitorId<'_>> {
	let (rest, native) = match id.rsplit_once('|') {
		Some((rest, last)) => match last.strip_prefix(NATIVE_ID_PREFIX) {
			Some(native) => (rest, Some(native)),
			None => (id, None),
		},
		None => (id, None),
	};
	let (rest, position) = rest.rsplit_once('|')?;
	let (name, _size) = rest.rsplit_once('|')?;
	Some(ParsedMonitorId { name, position, native })
}

/// Picks the first non-primary display, used to auto-select an external output on first launch.
///
/// Returns `None` when only the primary display is connected, so MUR never covers the
/// control window by default.
pub fn pick_default_external_monitor<T: SummarizedMonitor>(monitors: &[T]) -> Option<&T> {
	monitors.iter().find(|monitor| !monitor.summary().is_primary)
}

/// Rectangle in logical points with a top-left origin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogicalRect {
	/// Left edge.
	pub x: f64,
	/// Top edge (or bottom edge in flipped, bottom-left-origin coordinates).
	pub y: f64,
	/// Width.
	pub width: f64,
	/// Height.
	pub height: f64,
}

impl LogicalRect {
	/// Converts a monitor's physical position and size into logical points.
	pub fn from_physical(position: PhysicalPosition<i32>, size: PhysicalSize<u32>, scale_factor: f64) -> Self {
		let scale = sanitize_scale_factor(scale_factor);
		Self {
			x: f64::from(position.x) / scale,
			y: f64::from(position.y) / scale,
			width: f64::from(size.width) / scale,
			height: f64::from(size.height) / scale,
		}
	}

	/// Mirrors the rectangle vertically inside a space of height `reference_height`.
	///
	/// This converts between top-left-origin desktop coordinates (CoreGraphics / winit)
	/// and bottom-left-origin AppKit screen coordinates, where `reference_height` is the
	/// height of the primary screen. The conversion is its own inverse.
	pub fn flipped(self, reference_height: f64) -> Self {
		Self {
			y: reference_height - (self.y + self.height),
			..self
		}
	}

	/// Returns `true` when every edge of both rectangles is within `tolerance` points.
	pub fn approx_eq(&self, other: &Self, tolerance: f64) -> bool {
		(self.x - other.x).abs() <= tolerance
			&& (self.y - other.y).abs() <= tolerance
			&& (self.width - other.width).abs() <= tolerance
			&& (self.height - other.height).abs() <= tolerance
	}

	/// Fraction of `self` covered by `other`, in `0.0..=1.0`.
	pub fn overlap_ratio(&self, other: &Self) -> f64 {
		let area = self.width * self.height;
		if !area.is_finite() || area <= 0.0 {
			return 0.0;
		}
		let width = (self.x + self.width).min(other.x + other.width) - self.x.max(other.x);
		let height = (self.y + self.height).min(other.y + other.height) - self.y.max(other.y);
		if width <= 0.0 || height <= 0.0 {
			return 0.0;
		}
		(width * height / area).clamp(0.0, 1.0)
	}
}

/// Where the output window ended up, in physical desktop pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutputPlacement {
	/// Left edge in physical pixels.
	pub x: i32,
	/// Top edge in physical pixels.
	pub y: i32,
	/// Width in physical pixels.
	pub width: u32,
	/// Height in physical pixels.
	pub height: u32,
	/// HiDPI scale factor of the target monitor.
	pub scale_factor: f64,
	/// Whether the window was raised above the menu bar, Dock, or taskbar.
	pub elevated: bool,
}

impl OutputPlacement {
	/// Placement covering the whole of `monitor` as reported by winit.
	pub fn from_monitor(monitor: &MonitorHandle, elevated: bool) -> Self {
		let position = monitor.position();
		let size = monitor.size();
		Self {
			x: position.x,
			y: position.y,
			width: size.width,
			height: size.height,
			scale_factor: sanitize_scale_factor(monitor.scale_factor()),
			elevated,
		}
	}
}

/// Returns `true` when the output window should be raised above system UI.
///
/// Elevating is skipped when the output shares the control window's monitor, so a
/// single-display development setup never hides the control UI behind the output.
pub fn should_elevate_output(output: &MonitorHandle, control: Option<&MonitorHandle>) -> bool {
	control != Some(output)
}

/// Places `window` borderless over the entire `monitor` and prepares it for live output.
///
/// `control_monitor` is the monitor currently showing the control window (see
/// [`should_elevate_output`]). On macOS this keeps the window visible while the control
/// window is focused, joins all Spaces, and raises it above the menu bar and Dock; on
/// Windows it uses DPI-aware borderless fullscreen on the monitor. Returns
/// [`PlatformError::MonitorUnavailable`] when the monitor vanished in the meantime.
pub fn configure_output_window(
	window: &Window,
	monitor: &MonitorHandle,
	control_monitor: Option<&MonitorHandle>,
) -> Result<OutputPlacement, PlatformError> {
	let elevate = should_elevate_output(monitor, control_monitor);
	#[cfg(target_os = "macos")]
	{
		macos::configure_output_window(window, monitor, elevate)
	}
	#[cfg(target_os = "windows")]
	{
		windows::configure_output_window(window, monitor, elevate)
	}
	#[cfg(not(any(target_os = "macos", target_os = "windows")))]
	{
		use winit::window::{Fullscreen, WindowLevel};
		window.set_decorations(false);
		window.set_window_level(if elevate {
			WindowLevel::AlwaysOnTop
		} else {
			WindowLevel::Normal
		});
		window.set_fullscreen(Some(Fullscreen::Borderless(Some(monitor.clone()))));
		Ok(OutputPlacement::from_monitor(monitor, elevate))
	}
}

/// Drops the output window back to the normal window level, e.g. while output is paused.
pub fn lower_output_window(window: &Window) -> Result<(), PlatformError> {
	#[cfg(target_os = "macos")]
	{
		macos::lower_output_window(window)
	}
	#[cfg(target_os = "windows")]
	{
		windows::lower_output_window(window)
	}
	#[cfg(not(any(target_os = "macos", target_os = "windows")))]
	{
		window.set_window_level(winit::window::WindowLevel::Normal);
		Ok(())
	}
}

/// Callback that wakes the winit event loop, typically wrapping `EventLoopProxy::send_event`.
pub type QuitWaker = Box<dyn Fn() + Send + Sync + 'static>;

/// What [`install_exit_guard`] managed to intercept.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExitGuardStatus {
	/// Number of "Quit" menu items (and their `Cmd+Q` shortcut) rerouted to the guard.
	pub intercepted_menu_items: usize,
	/// Whether app-level termination (Dock "Quit", `terminate:`) is routed to the guard.
	pub intercepts_terminate: bool,
}

static QUIT_REQUESTED: AtomicBool = AtomicBool::new(false);
static ALLOW_TERMINATION: AtomicBool = AtomicBool::new(false);
static QUIT_WAKER: Mutex<Option<QuitWaker>> = Mutex::new(None);

/// Registers the callback invoked whenever a native quit request is intercepted.
pub fn set_quit_waker(waker: QuitWaker) {
	let mut slot = QUIT_WAKER.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
	*slot = Some(waker);
}

/// Records a quit request and wakes the event loop; called by the native back ends.
pub fn request_quit() {
	QUIT_REQUESTED.store(true, Ordering::SeqCst);
	let slot = QUIT_WAKER.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
	if let Some(waker) = slot.as_ref() {
		waker();
	}
}

/// Returns `true` once per intercepted quit request, clearing the flag.
///
/// The app should open the exit confirmation overlay when this returns `true`. On
/// platforms without native interception it simply stays `false`.
pub fn poll_quit_requested() -> bool {
	QUIT_REQUESTED.swap(false, Ordering::SeqCst)
}

/// Lets subsequent native termination requests through, after the operator confirmed exit.
///
/// Call this right before `ActiveEventLoop::exit()` so a late `terminate:` (e.g. from
/// logout) is not cancelled again.
pub fn allow_termination() {
	ALLOW_TERMINATION.store(true, Ordering::SeqCst);
}

/// Returns `true` once [`allow_termination`] was called.
pub fn termination_allowed() -> bool {
	ALLOW_TERMINATION.load(Ordering::SeqCst)
}

/// Routes native quit commands to the exit confirmation guard.
///
/// On macOS this must run on the main thread after the event loop launched (e.g. in
/// `ApplicationHandler::resumed`), once winit installed its default menu. It never
/// replaces winit's application delegate. Elsewhere it is a no-op, because close
/// requests already arrive as `WindowEvent::CloseRequested`.
pub fn install_exit_guard() -> Result<ExitGuardStatus, PlatformError> {
	#[cfg(target_os = "macos")]
	{
		macos::install_exit_guard()
	}
	#[cfg(not(target_os = "macos"))]
	{
		Ok(ExitGuardStatus::default())
	}
}

/// Keeps macOS App Nap from throttling MUR while it is alive.
///
/// App Nap coalesces timers and slows the run loop of apps whose windows are hidden or
/// occluded, which would starve the output window's 60 fps tick when the control window
/// is in the background. On other platforms this is an empty value.
#[must_use = "App Nap is only prevented while the guard is alive"]
pub struct AppNapGuard {
	#[cfg(target_os = "macos")]
	_activity: macos::AppNapActivity,
}

impl std::fmt::Debug for AppNapGuard {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter.debug_struct("AppNapGuard").finish_non_exhaustive()
	}
}

/// Begins a latency-critical activity that disables App Nap until the guard is dropped.
///
/// A no-op on platforms without App Nap.
pub fn prevent_app_nap() -> AppNapGuard {
	#[cfg(target_os = "macos")]
	{
		AppNapGuard {
			_activity: macos::AppNapActivity::begin(),
		}
	}
	#[cfg(not(target_os = "macos"))]
	{
		AppNapGuard {}
	}
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;

	fn summary(id: &str, name: &str, is_primary: bool) -> MonitorSummary {
		MonitorSummary {
			id: id.to_owned(),
			name: name.to_owned(),
			width: 1920,
			height: 1080,
			scale_factor: 1.0,
			is_primary,
		}
	}

	fn monitors() -> Vec<MonitorSummary> {
		vec![
			summary("Built-in Retina Display|3024x1964|0,0", "Built-in Retina Display", true),
			summary("DELL U2720Q|3840x2160|3024,0", "DELL U2720Q", false),
			summary("DELL U2720Q|3840x2160|6864,0", "DELL U2720Q", false),
		]
	}

	#[test]
	fn id_is_stable_and_includes_geometry() {
		let size = PhysicalSize::new(3840, 2160);
		let position = PhysicalPosition::new(-1920, 0);
		let id = monitor_id(Some("  LG HDR 4K "), size, position);
		assert_eq!(id, "LG HDR 4K|3840x2160|-1920,0");
		assert_eq!(id, monitor_id(Some("LG HDR 4K"), size, position));
	}

	#[test]
	fn missing_name_falls_back_to_resolution() {
		let size = PhysicalSize::new(1920, 1080);
		assert_eq!(monitor_display_name(None, size), "Display 1920x1080");
		assert_eq!(monitor_display_name(Some("   "), size), "Display 1920x1080");
		assert_eq!(
			monitor_id(None, size, PhysicalPosition::new(0, 0)),
			"Display 1920x1080|1920x1080|0,0"
		);
	}

	#[test]
	fn matching_prefers_exact_id_over_name() {
		let monitors = monitors();
		let found = find_matching_monitor(&monitors, Some("DELL U2720Q|3840x2160|6864,0")).unwrap();
		assert_eq!(found.id, "DELL U2720Q|3840x2160|6864,0");
	}

	#[test]
	fn matching_falls_back_to_exact_then_case_insensitive_name() {
		let monitors = monitors();
		let exact = find_matching_monitor(&monitors, Some("DELL U2720Q")).unwrap();
		assert_eq!(exact.id, "DELL U2720Q|3840x2160|3024,0");
		let folded = find_matching_monitor(&monitors, Some("built-in retina display")).unwrap();
		assert!(folded.is_primary);
	}

	#[test]
	fn stale_id_matches_by_name_preferring_same_position() {
		let monitors = monitors();
		let same_position = find_matching_monitor(&monitors, Some("DELL U2720Q|1920x1080|6864,0")).unwrap();
		assert_eq!(same_position.id, "DELL U2720Q|3840x2160|6864,0");
		let moved = find_matching_monitor(&monitors, Some("dell u2720q|1920x1080|0,900")).unwrap();
		assert_eq!(moved.id, "DELL U2720Q|3840x2160|3024,0");
		assert!(find_matching_monitor(&monitors, Some("Projector|1920x1080|0,0")).is_none());
	}

	#[test]
	fn matching_returns_none_when_disconnected_or_unselected() {
		let monitors = monitors();
		assert!(find_matching_monitor(&monitors, Some("Projector")).is_none());
		assert!(find_matching_monitor(&monitors, None).is_none());
		assert!(find_matching_monitor(&monitors, Some("  ")).is_none());
		assert!(find_matching_monitor::<MonitorSummary>(&[], Some("DELL U2720Q")).is_none());
	}

	#[test]
	fn native_id_is_appended_and_parsed() {
		let size = PhysicalSize::new(3024, 1964);
		let id = monitor_id_with_native(
			Some("Built-in Retina Display"),
			size,
			PhysicalPosition::new(0, 0),
			Some(1),
		);
		assert_eq!(id, "Built-in Retina Display|3024x1964|0,0|cg:1");
		let parsed = parse_monitor_id(&id).unwrap();
		assert_eq!(parsed.name, "Built-in Retina Display");
		assert_eq!(parsed.position, "0,0");
		assert_eq!(parsed.native, Some("1"));
		let plain = parse_monitor_id("DELL|1920x1080|-1920,0").unwrap();
		assert_eq!(plain.native, None);
		assert_eq!(plain.position, "-1920,0");
		assert!(parse_monitor_id("DELL").is_none());
	}

	#[test]
	fn native_id_survives_resolution_change() {
		let monitors = vec![
			summary("Built-in|1512x982|0,0|cg:1", "Built-in", true),
			summary("LG HDR 4K|1920x1080|1512,0|cg:7", "LG HDR 4K", false),
		];
		let found = find_matching_monitor(&monitors, Some("LG HDR 4K|3840x2160|-3840,0|cg:7")).unwrap();
		assert_eq!(found.id, "LG HDR 4K|1920x1080|1512,0|cg:7");
	}

	fn same_named() -> Vec<MonitorSummary> {
		vec![
			summary("Monitor #1234|1920x1080|0,0|cg:1", "Monitor #1234", true),
			summary("Monitor #1234|1920x1080|1920,0|cg:2", "Monitor #1234", false),
		]
	}

	#[test]
	fn same_named_external_is_found_and_control_is_never_a_fallback() {
		let control = "Monitor #1234|1920x1080|0,0|cg:1";
		let external = "Monitor #1234|1920x1080|1920,0|cg:2";
		let monitors = same_named();

		let by_id = find_matching_monitor_excluding(&monitors, Some(external), Some(control)).unwrap();
		assert_eq!(by_id.id, external);
		// A plain name shared by both displays resolves to the non-control one.
		let by_name = find_matching_monitor_excluding(&monitors, Some("Monitor #1234"), Some(control)).unwrap();
		assert_eq!(by_name.id, external);

		// Unplug the external display: only the control display remains.
		let unplugged: Vec<MonitorSummary> = monitors.into_iter().filter(|monitor| monitor.id == control).collect();
		assert!(find_matching_monitor_excluding(&unplugged, Some(external), Some(control)).is_none());
		assert!(find_matching_monitor_excluding(&unplugged, Some("Monitor #1234"), Some(control)).is_none());
		assert!(
			find_matching_monitor_excluding(&unplugged, Some("monitor #1234|3840x2160|1920,0"), Some(control))
				.is_none()
		);
		// Without exclusion the legacy lookup would have hijacked the control display.
		assert_eq!(find_matching_monitor(&unplugged, Some(external)).unwrap().id, control);
	}

	#[test]
	fn explicitly_selected_control_display_is_honored() {
		let control = "Monitor #1234|1920x1080|0,0|cg:1";
		let single: Vec<MonitorSummary> = same_named()
			.into_iter()
			.filter(|monitor| monitor.id == control)
			.collect();
		let found = find_matching_monitor_excluding(&single, Some(control), Some(control)).unwrap();
		assert_eq!(found.id, control);
		// The same physical display after a resolution change is still the operator's choice.
		let moved = find_matching_monitor_excluding(&single, Some("Monitor #1234|3840x2160|0,0|cg:1"), Some(control));
		assert_eq!(moved.unwrap().id, control);
	}

	#[test]
	fn exclusion_without_native_ids_keeps_fallbacks_off_the_control_display() {
		let monitors = vec![
			summary("DELL|1920x1080|0,0", "DELL", true),
			summary("DELL|1920x1080|1920,0", "DELL", false),
		];
		let found = find_matching_monitor_excluding(&monitors, Some("DELL|2560x1440|0,0"), Some("DELL|1920x1080|0,0"));
		assert_eq!(found.unwrap().id, "DELL|1920x1080|1920,0");
		let only_control = vec![summary("DELL|1920x1080|0,0", "DELL", true)];
		assert!(
			find_matching_monitor_excluding(&only_control, Some("DELL|1920x1080|1920,0"), Some("DELL|1920x1080|0,0"))
				.is_none()
		);
	}

	#[test]
	fn app_nap_guard_can_be_created_and_dropped() {
		let guard = prevent_app_nap();
		assert!(format!("{guard:?}").contains("AppNapGuard"));
		drop(guard);
	}

	#[test]
	fn default_external_skips_primary() {
		let monitors = monitors();
		let picked = pick_default_external_monitor(&monitors).unwrap();
		assert_eq!(picked.id, "DELL U2720Q|3840x2160|3024,0");

		let only_primary = vec![summary("a", "Built-in", true)];
		assert!(pick_default_external_monitor(&only_primary).is_none());
		assert!(pick_default_external_monitor::<MonitorSummary>(&[]).is_none());
	}

	#[test]
	fn summaries_clone_in_order() {
		let monitors = monitors();
		let cloned = summaries(&monitors);
		assert_eq!(cloned, monitors);
	}

	#[test]
	fn scale_factor_is_sanitized() {
		assert_eq!(sanitize_scale_factor(2.0), 2.0);
		assert_eq!(sanitize_scale_factor(0.0), 1.0);
		assert_eq!(sanitize_scale_factor(-1.0), 1.0);
		assert_eq!(sanitize_scale_factor(f64::NAN), 1.0);
		assert_eq!(sanitize_scale_factor(f64::INFINITY), 1.0);
	}

	#[test]
	fn logical_rect_converts_retina_geometry() {
		let rect = LogicalRect::from_physical(PhysicalPosition::new(6048, -400), PhysicalSize::new(3840, 2160), 2.0);
		assert_eq!(
			rect,
			LogicalRect {
				x: 3024.0,
				y: -200.0,
				width: 1920.0,
				height: 1080.0
			}
		);
	}

	#[test]
	fn flipping_matches_appkit_coordinates_and_is_involutive() {
		// Primary screen 1512x982 points; external 1920x1080 to its right, top-aligned.
		let top_left = LogicalRect {
			x: 1512.0,
			y: 0.0,
			width: 1920.0,
			height: 1080.0,
		};
		let appkit = top_left.flipped(982.0);
		assert_eq!(appkit.y, -98.0);
		assert_eq!(appkit.flipped(982.0), top_left);
	}

	#[test]
	fn approx_eq_and_overlap() {
		let a = LogicalRect {
			x: 0.0,
			y: 0.0,
			width: 100.0,
			height: 100.0,
		};
		let nudged = LogicalRect { x: 0.5, ..a };
		assert!(a.approx_eq(&nudged, 1.0));
		assert!(!a.approx_eq(&LogicalRect { x: 5.0, ..a }, 1.0));

		let half = LogicalRect { x: 50.0, ..a };
		assert!((a.overlap_ratio(&half) - 0.5).abs() < 1e-9);
		assert_eq!(a.overlap_ratio(&LogicalRect { x: 200.0, ..a }), 0.0);
		let empty = LogicalRect { width: 0.0, ..a };
		assert_eq!(empty.overlap_ratio(&a), 0.0);
	}

	#[test]
	fn quit_flag_is_consumed_once_and_wakes() {
		use std::sync::Arc;
		use std::sync::atomic::AtomicUsize;

		let wakes = Arc::new(AtomicUsize::new(0));
		let counter = Arc::clone(&wakes);
		set_quit_waker(Box::new(move || {
			counter.fetch_add(1, Ordering::SeqCst);
		}));
		let _ = poll_quit_requested();
		request_quit();
		assert!(poll_quit_requested());
		assert!(!poll_quit_requested());
		assert_eq!(wakes.load(Ordering::SeqCst), 1);
	}
}
