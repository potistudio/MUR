//! macOS back end: borderless external output placement and the `Cmd+Q` guard.
//!
//! # Output window
//!
//! winit's `Fullscreen::Borderless` on macOS enters a native fullscreen Space (animated,
//! hides the other display's content when "Displays have separate Spaces" is off, and
//! unwraps the target `NSScreen` internally). MUR instead keeps the output as a normal
//! borderless window and configures the `NSWindow` directly:
//!
//! - `hidesOnDeactivate = false`, so focusing the control window never hides the output;
//! - collection behavior `CanJoinAllSpaces | Stationary | IgnoresCycle | FullScreenAuxiliary`,
//!   so Space switches, Mission Control, and `Cmd+\`` leave it in place;
//! - window level `NSMainMenuWindowLevel + 1`, above the menu bar and Dock of that display;
//! - frame equal to the target `NSScreen` frame (AppKit points, bottom-left origin), so
//!   the Retina backing scale is applied by AppKit and the drawable covers every pixel.
//!
//! # Quit guard
//!
//! winit owns the `NSApplication` delegate (`WinitApplicationDelegate`) and panics if it
//! is replaced, and it builds the default menu whose "Quit" item sends `terminate:`.
//! [`install_exit_guard`] therefore never touches the delegate object itself; it:
//!
//! 1. retargets every `terminate:` menu item (including its `Cmd+Q` key equivalent) to a
//!    small MUR-owned object that calls [`super::request_quit`];
//! 2. adds an `applicationShouldTerminate:` method to the delegate's class when the
//!    class does not implement one, so Dock "Quit" and other `terminate:` senders are
//!    cancelled and routed to the guard until [`super::allow_termination`] is called.
//!
//! winit's own `ActiveEventLoop::exit()` stops the run loop without `terminate:`, so a
//! confirmed exit is unaffected by the hook.

use std::cell::OnceCell;

use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject, Imp, NSObject, Sel};
use objc2::{MainThreadMarker, MainThreadOnly, define_class, ffi, msg_send, sel};
use objc2_app_kit::{
	NSApplication, NSApplicationTerminateReply, NSMainMenuWindowLevel, NSMenu, NSNormalWindowLevel, NSScreen, NSView,
	NSWindow, NSWindowCollectionBehavior,
};
use objc2_foundation::{NSObjectProtocol, NSRect, NSUInteger};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::monitor::MonitorHandle;
use winit::window::Window;

use super::{ExitGuardStatus, LogicalRect, OutputPlacement, PlatformError, sanitize_scale_factor};

/// Maximum distance, in points, between a winit monitor rectangle and an `NSScreen` frame.
const FRAME_MATCH_TOLERANCE: f64 = 2.0;

/// Minimum fraction of the monitor an `NSScreen` must cover to be used as a fallback match.
const FRAME_MIN_OVERLAP: f64 = 0.5;

/// Maximum menu nesting depth searched for `terminate:` items.
const MAX_MENU_DEPTH: usize = 4;

/// Window level used while elevated: just above the menu bar (`NSMainMenuWindowLevel + 1`).
pub fn elevated_window_level() -> isize {
	NSMainMenuWindowLevel.saturating_add(1)
}

/// Resolves the `NSWindow` backing a winit window.
fn ns_window(window: &Window, _mtm: MainThreadMarker) -> Result<Retained<NSWindow>, PlatformError> {
	let handle = window
		.window_handle()
		.map_err(|error| PlatformError::WindowHandle(error.to_string()))?;
	let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
		return Err(PlatformError::WindowHandle(
			"expected an AppKit window handle".to_owned(),
		));
	};
	// SAFETY: winit guarantees `ns_view` points to the live `NSView` of this window for as
	// long as the window (borrowed here) exists, and the `MainThreadMarker` proves we are
	// on the main thread where AppKit objects may be used.
	let view: &NSView = unsafe { appkit.ns_view.cast::<NSView>().as_ref() };
	view.window()
		.ok_or_else(|| PlatformError::WindowHandle("view is not attached to an NSWindow".to_owned()))
}

/// Converts an AppKit rectangle into a [`LogicalRect`] without changing its origin convention.
fn logical_from_ns(rect: NSRect) -> LogicalRect {
	LogicalRect {
		x: rect.origin.x,
		y: rect.origin.y,
		width: rect.size.width,
		height: rect.size.height,
	}
}

/// Finds the `NSScreen` frame (AppKit coordinates) that corresponds to `monitor`.
///
/// winit reports monitor geometry in top-left-origin CoreGraphics desktop space, so the
/// monitor rectangle is compared against each screen frame flipped by the primary
/// screen's height. Returns `None` when no connected screen matches, i.e. the monitor
/// was unplugged.
fn screen_frame_for(monitor: &MonitorHandle, mtm: MainThreadMarker) -> Option<NSRect> {
	let target = LogicalRect::from_physical(monitor.position(), monitor.size(), monitor.scale_factor());
	let screens = NSScreen::screens(mtm);
	let primary_height = screens.firstObject()?.frame().size.height;
	let frames: Vec<NSRect> = screens.iter().map(|screen| screen.frame()).collect();
	let desktop = |frame: &NSRect| logical_from_ns(*frame).flipped(primary_height);

	if let Some(frame) = frames
		.iter()
		.find(|frame| desktop(frame).approx_eq(&target, FRAME_MATCH_TOLERANCE))
	{
		return Some(*frame);
	}
	frames
		.iter()
		.map(|frame| (frame, target.overlap_ratio(&desktop(frame))))
		.filter(|(_, ratio)| *ratio >= FRAME_MIN_OVERLAP)
		.max_by(|(_, left), (_, right)| left.total_cmp(right))
		.map(|(frame, _)| *frame)
}

/// Places `window` borderless over `monitor`; see the module docs for the AppKit settings.
///
/// Must run on the main thread. `elevate` raises the window above the menu bar and Dock;
/// pass `false` when the output shares the control window's display.
pub fn configure_output_window(
	window: &Window,
	monitor: &MonitorHandle,
	elevate: bool,
) -> Result<OutputPlacement, PlatformError> {
	let mtm = MainThreadMarker::new().ok_or(PlatformError::NotMainThread)?;
	let frame = screen_frame_for(monitor, mtm).ok_or_else(|| {
		PlatformError::MonitorUnavailable(monitor.name().unwrap_or_else(|| "unnamed display".to_owned()))
	})?;
	let ns_window = ns_window(window, mtm)?;

	if window.fullscreen().is_some() {
		window.set_fullscreen(None);
	}
	window.set_decorations(false);
	window.set_resizable(false);

	ns_window.setHidesOnDeactivate(false);
	ns_window.setCollectionBehavior(
		NSWindowCollectionBehavior::CanJoinAllSpaces
			| NSWindowCollectionBehavior::Stationary
			| NSWindowCollectionBehavior::IgnoresCycle
			| NSWindowCollectionBehavior::FullScreenAuxiliary,
	);
	ns_window.setLevel(if elevate {
		elevated_window_level()
	} else {
		NSNormalWindowLevel
	});
	ns_window.setHasShadow(false);
	ns_window.setMovable(false);
	ns_window.setFrame_display(frame, true);
	// Show without activating MUR's output window, so keyboard focus stays on the control UI.
	ns_window.orderFrontRegardless();

	let scale_factor = sanitize_scale_factor(monitor.scale_factor());
	let placed = logical_from_ns(ns_window.frame());
	tracing::debug!(
		target: "platform",
		x = placed.x,
		y = placed.y,
		width = placed.width,
		height = placed.height,
		scale_factor,
		elevate,
		"output window placed"
	);
	Ok(OutputPlacement::from_monitor(monitor, elevate))
}

/// Moves `window` back to the normal window level, e.g. while output is paused.
pub fn lower_output_window(window: &Window) -> Result<(), PlatformError> {
	let mtm = MainThreadMarker::new().ok_or(PlatformError::NotMainThread)?;
	ns_window(window, mtm)?.setLevel(NSNormalWindowLevel);
	Ok(())
}

define_class!(
	// SAFETY:
	// - `NSObject` has no subclassing requirements.
	// - `MurQuitTarget` has no ivars and does not implement `Drop`.
	#[unsafe(super(NSObject))]
	#[thread_kind = MainThreadOnly]
	#[name = "MURQuitTarget"]
	struct MurQuitTarget;

	// SAFETY: `NSObjectProtocol` has no safety requirements.
	unsafe impl NSObjectProtocol for MurQuitTarget {}

	impl MurQuitTarget {
		// SAFETY: Standard action signature `- (void)murRequestQuit:(id)sender`.
		#[unsafe(method(murRequestQuit:))]
		fn mur_request_quit(&self, _sender: Option<&AnyObject>) {
			super::request_quit();
		}
	}
);

impl MurQuitTarget {
	fn new(mtm: MainThreadMarker) -> Retained<Self> {
		let this = Self::alloc(mtm).set_ivars(());
		// SAFETY: `NSObject`'s designated initializer `init` takes no arguments.
		unsafe { msg_send![super(this), init] }
	}
}

thread_local! {
	/// Keeps the quit target alive; `NSMenuItem.target` is only a weak reference.
	static QUIT_TARGET: OnceCell<Retained<MurQuitTarget>> = const { OnceCell::new() };
}

/// Returns the main-thread quit target, creating it on first use.
fn quit_target(mtm: MainThreadMarker) -> Retained<MurQuitTarget> {
	QUIT_TARGET.with(|cell| cell.get_or_init(|| MurQuitTarget::new(mtm)).clone())
}

/// Retargets every `terminate:` item in `menu` (recursively) to `target`.
fn retarget_quit_items(menu: &NSMenu, target: &MurQuitTarget, depth: usize) -> usize {
	if depth > MAX_MENU_DEPTH {
		return 0;
	}
	let mut count = 0;
	for item in menu.itemArray().iter() {
		match item.action() {
			Some(action) if action == sel!(terminate:) => {
				// SAFETY: `target` is a live `MurQuitTarget` kept alive by `QUIT_TARGET` for
				// the rest of the program, and it implements the `murRequestQuit:` selector
				// with the standard `(id)sender` action signature.
				unsafe {
					item.setTarget(Some(target.as_ref()));
					item.setAction(Some(sel!(murRequestQuit:)));
				}
				count += 1;
			}
			Some(action) if action == sel!(murRequestQuit:) => count += 1,
			_ => {}
		}
		if let Some(submenu) = item.submenu() {
			count += retarget_quit_items(&submenu, target, depth.saturating_add(1));
		}
	}
	count
}

/// Signature of `- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender`.
type ShouldTerminateFn = extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject) -> NSUInteger;

/// `applicationShouldTerminate:` implementation added to winit's delegate class.
extern "C-unwind" fn should_terminate(_this: *mut AnyObject, _cmd: Sel, _sender: *mut AnyObject) -> NSUInteger {
	if super::termination_allowed() {
		return NSApplicationTerminateReply::TerminateNow.0;
	}
	super::request_quit();
	NSApplicationTerminateReply::TerminateCancel.0
}

/// Objective-C type encoding of [`ShouldTerminateFn`] (`NSUInteger` is `unsigned long`).
const SHOULD_TERMINATE_TYPES: &std::ffi::CStr = c"Q@:@";

/// Adds `applicationShouldTerminate:` to the application delegate's class.
///
/// Returns `true` when MUR's implementation is active, `false` when there is no delegate
/// or the delegate already implements the method (it is then left untouched).
fn hook_should_terminate(app: &NSApplication) -> bool {
	let Some(delegate) = app.delegate() else {
		tracing::warn!(target: "platform", "no NSApplication delegate; Dock quit is not guarded");
		return false;
	};
	let object: &AnyObject = delegate.as_ref();
	let class: &'static AnyClass = object.class();
	let selector = sel!(applicationShouldTerminate:);
	let ours = should_terminate as ShouldTerminateFn;

	if let Some(method) = class.instance_method(selector) {
		// SAFETY: Only compares addresses; the IMP is never called through this type.
		let existing = unsafe { std::mem::transmute::<Imp, ShouldTerminateFn>(method.implementation()) };
		let hooked = std::ptr::fn_addr_eq(existing, ours);
		if !hooked {
			tracing::warn!(
				target: "platform",
				"application delegate already implements applicationShouldTerminate:; leaving it untouched"
			);
		}
		return hooked;
	}

	// SAFETY: `ShouldTerminateFn` matches the Objective-C signature of
	// `applicationShouldTerminate:` described by `SHOULD_TERMINATE_TYPES`; converting the
	// function pointer to the untyped `Imp` is how the runtime stores implementations.
	let imp = unsafe { std::mem::transmute::<ShouldTerminateFn, Imp>(ours) };
	// SAFETY: `class` is a registered, live class (the delegate's dynamic class). Adding a
	// method the class does not implement is a supported runtime operation, it runs on the
	// main thread before AppKit could concurrently message the delegate, and the type
	// encoding is a static NUL-terminated string.
	let added = unsafe {
		ffi::class_addMethod(
			std::ptr::from_ref(class).cast_mut(),
			selector,
			imp,
			SHOULD_TERMINATE_TYPES.as_ptr(),
		)
	};
	added.as_bool()
}

/// Routes `Cmd+Q`, the "Quit" menu item, and Dock "Quit" to the exit confirmation guard.
///
/// Must run on the main thread after winit finished launching (its default menu is built
/// in `applicationDidFinishLaunching:`). Safe to call repeatedly, e.g. after menus change.
pub fn install_exit_guard() -> Result<ExitGuardStatus, PlatformError> {
	let mtm = MainThreadMarker::new().ok_or(PlatformError::NotMainThread)?;
	let app = NSApplication::sharedApplication(mtm);
	let target = quit_target(mtm);
	let intercepted_menu_items = app.mainMenu().map_or(0, |menu| retarget_quit_items(&menu, &target, 0));
	if intercepted_menu_items == 0 {
		tracing::warn!(target: "platform", "no Quit menu item found; Cmd+Q relies on applicationShouldTerminate:");
	}
	let intercepts_terminate = hook_should_terminate(&app);
	Ok(ExitGuardStatus {
		intercepted_menu_items,
		intercepts_terminate,
	})
}
