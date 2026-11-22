//! Windows back end: DPI-aware borderless fullscreen placement of the output window.
//!
//! winit opts the process into per-monitor DPI awareness (v2) when the event loop is
//! created, so monitor rectangles and window sizes are reported in physical pixels and
//! `WM_DPICHANGED` is handled for windows moving between displays of different scale.
//! The output window uses winit's `Fullscreen::Borderless` on the target monitor, which
//! switches to a `WS_POPUP` window covering `rcMonitor` (including the taskbar area).
//! The monitor bounds are read back through `GetMonitorInfoW` both to detect a monitor
//! that disappeared and to report the exact placement.
//!
//! Exit requests need no native hook here: `Alt+F4`, the close button, and the system
//! menu all arrive as `WindowEvent::CloseRequested`.

use windows_sys::Win32::Foundation::RECT;
use windows_sys::Win32::Graphics::Gdi::{GetMonitorInfoW, HMONITOR, MONITORINFO};
use winit::monitor::MonitorHandle;
use winit::platform::windows::MonitorHandleExtWindows;
use winit::window::{Fullscreen, Window, WindowLevel};

use super::{OutputPlacement, PlatformError, sanitize_scale_factor};

/// Bounds of a monitor in physical desktop pixels, as reported by `GetMonitorInfoW`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonitorBounds {
	/// Full monitor rectangle `(left, top, right, bottom)`.
	pub monitor: (i32, i32, i32, i32),
	/// Work area excluding the taskbar `(left, top, right, bottom)`.
	pub work_area: (i32, i32, i32, i32),
	/// Whether Windows flags this monitor as primary.
	pub is_primary: bool,
}

impl MonitorBounds {
	/// Width of the full monitor rectangle in physical pixels.
	pub fn width(&self) -> u32 {
		let (left, _, right, _) = self.monitor;
		u32::try_from(right.saturating_sub(left)).unwrap_or(0)
	}

	/// Height of the full monitor rectangle in physical pixels.
	pub fn height(&self) -> u32 {
		let (_, top, _, bottom) = self.monitor;
		u32::try_from(bottom.saturating_sub(top)).unwrap_or(0)
	}
}

fn rect_tuple(rect: RECT) -> (i32, i32, i32, i32) {
	(rect.left, rect.top, rect.right, rect.bottom)
}

/// Queries the physical bounds of `monitor`; `None` when the monitor is no longer attached.
pub fn monitor_bounds(monitor: &MonitorHandle) -> Option<MonitorBounds> {
	let empty = RECT {
		left: 0,
		top: 0,
		right: 0,
		bottom: 0,
	};
	let mut info = MONITORINFO {
		cbSize: u32::try_from(std::mem::size_of::<MONITORINFO>()).unwrap_or(0),
		rcMonitor: empty,
		rcWork: empty,
		dwFlags: 0,
	};
	let hmonitor = monitor.hmonitor() as HMONITOR;
	// SAFETY: `info` is a properly sized, writable `MONITORINFO` with `cbSize` set, and
	// `GetMonitorInfoW` tolerates stale monitor handles by returning `FALSE`.
	let ok = unsafe { GetMonitorInfoW(hmonitor, &mut info) };
	if ok == 0 {
		return None;
	}
	let bounds = MonitorBounds {
		monitor: rect_tuple(info.rcMonitor),
		work_area: rect_tuple(info.rcWork),
		is_primary: info.dwFlags & windows_sys::Win32::UI::WindowsAndMessaging::MONITORINFOF_PRIMARY != 0,
	};
	(bounds.width() > 0 && bounds.height() > 0).then_some(bounds)
}

/// Places `window` borderless fullscreen over `monitor` with DPI-aware geometry.
///
/// `elevate` makes the window topmost so the taskbar of that display never covers it;
/// pass `false` when the output shares the control window's display.
pub fn configure_output_window(
	window: &Window,
	monitor: &MonitorHandle,
	elevate: bool,
) -> Result<OutputPlacement, PlatformError> {
	let bounds = monitor_bounds(monitor).ok_or_else(|| {
		PlatformError::MonitorUnavailable(monitor.name().unwrap_or_else(|| "unnamed display".to_owned()))
	})?;

	window.set_decorations(false);
	window.set_window_level(if elevate {
		WindowLevel::AlwaysOnTop
	} else {
		WindowLevel::Normal
	});
	window.set_fullscreen(Some(Fullscreen::Borderless(Some(monitor.clone()))));

	let (left, top, _, _) = bounds.monitor;
	let placement = OutputPlacement {
		x: left,
		y: top,
		width: bounds.width(),
		height: bounds.height(),
		scale_factor: sanitize_scale_factor(monitor.scale_factor()),
		elevated: elevate,
	};
	tracing::debug!(
		target: "platform",
		x = placement.x,
		y = placement.y,
		width = placement.width,
		height = placement.height,
		scale_factor = placement.scale_factor,
		elevate,
		"output window placed"
	);
	Ok(placement)
}

/// Moves `window` back to the normal z-order, e.g. while output is paused.
pub fn lower_output_window(window: &Window) -> Result<(), PlatformError> {
	window.set_window_level(WindowLevel::Normal);
	Ok(())
}
