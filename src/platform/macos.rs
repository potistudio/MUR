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
//!
//! # Display identity
//!
//! winit names macOS monitors `"Monitor #<model number>"`, which is identical for two
//! displays of the same model and unrelated to what System Settings shows. MUR resolves
//! the `NSScreen` of each monitor through its `CGDirectDisplayID`
//! (`deviceDescription[@"NSScreenNumber"]`), uses `NSScreen.localizedName` as the display
//! name, and derives the primary display from the first entry of `NSScreen.screens`
//! instead of `ActiveEventLoop::primary_monitor`.
//!
//! # App Nap
//!
//! [`AppNapActivity`] holds an `NSProcessInfo` activity (`UserInitiated | LatencyCritical`)
//! so macOS never throttles timers or the run loop while the show is running.

use std::cell::OnceCell;

use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject, Imp, NSObject, ProtocolObject, Sel};
use objc2::{MainThreadMarker, MainThreadOnly, define_class, ffi, msg_send, sel};
use objc2_app_kit::{
	NSApplication, NSApplicationTerminateReply, NSEventType, NSMainMenuWindowLevel, NSMenu, NSNormalWindowLevel,
	NSScreen, NSView, NSWindow, NSWindowCollectionBehavior,
};
use objc2_foundation::{
	NSActivityOptions, NSNumber, NSObjectProtocol, NSProcessInfo, NSRect, NSString, NSUInteger, ns_string,
};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::monitor::MonitorHandle;
use winit::platform::macos::MonitorHandleExtMacOS;
use winit::window::Window;

use super::{
	DISCONNECTED_DISPLAY_NAME, ExitGuardStatus, LogicalRect, OutputPlacement, PlatformError, sanitize_scale_factor,
};

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

/// Returns the `CGDirectDisplayID` of `screen` (`deviceDescription[@"NSScreenNumber"]`).
fn screen_display_id(screen: &NSScreen) -> Option<u32> {
	let description = screen.deviceDescription();
	let value = description.objectForKey(ns_string!("NSScreenNumber"))?;
	let number = value.downcast::<NSNumber>().ok()?;
	Some(number.unsignedIntValue())
}

/// Finds the `NSScreen` that corresponds to `monitor`.
///
/// The screen whose `NSScreenNumber` equals the monitor's `CGDirectDisplayID` wins. When
/// no screen reports that id (some docks briefly report none), the monitor rectangle —
/// top-left-origin CoreGraphics desktop space — is compared against each screen frame
/// flipped by the primary screen's height. Returns `None` when no connected screen
/// matches, i.e. the monitor was unplugged.
fn ns_screen_for(monitor: &MonitorHandle, mtm: MainThreadMarker) -> Option<Retained<NSScreen>> {
	let screens = NSScreen::screens(mtm);
	let display_id = monitor.native_id();
	if let Some(screen) = screens
		.iter()
		.find(|screen| screen_display_id(screen) == Some(display_id))
	{
		return Some(screen);
	}

	let target = LogicalRect::from_physical(monitor.position(), monitor.size(), monitor.scale_factor());
	let primary_height = screens.firstObject()?.frame().size.height;
	let desktop = |screen: &NSScreen| logical_from_ns(screen.frame()).flipped(primary_height);
	if let Some(screen) = screens
		.iter()
		.find(|screen| desktop(screen).approx_eq(&target, FRAME_MATCH_TOLERANCE))
	{
		return Some(screen);
	}
	screens
		.iter()
		.map(|screen| {
			let ratio = target.overlap_ratio(&desktop(&screen));
			(screen, ratio)
		})
		.filter(|(_, ratio)| *ratio >= FRAME_MIN_OVERLAP)
		.max_by(|(_, left), (_, right)| left.total_cmp(right))
		.map(|(screen, _)| screen)
}

/// AppKit details of a connected monitor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScreenInfo {
	/// `NSScreen.localizedName`, e.g. `"DELL U2723QE"` or `"Built-in Retina Display"`.
	pub localized_name: Option<String>,
	/// Whether this is the primary display (the first entry of `NSScreen.screens`).
	pub is_primary: bool,
}

/// Looks up the `NSScreen` details of `monitor`.
///
/// Returns `None` off the main thread or when no `NSScreen` corresponds to the monitor
/// (it was unplugged); callers then fall back to the winit name and geometry.
pub fn screen_info(monitor: &MonitorHandle) -> Option<ScreenInfo> {
	let mtm = MainThreadMarker::new()?;
	let screen = ns_screen_for(monitor, mtm)?;
	let primary_id = NSScreen::screens(mtm)
		.firstObject()
		.and_then(|primary| screen_display_id(&primary));
	let is_primary = match (primary_id, screen_display_id(&screen)) {
		(Some(primary), Some(own)) => primary == own,
		_ => monitor.position() == winit::dpi::PhysicalPosition::new(0, 0),
	};
	let localized_name = screen.localizedName().to_string();
	let localized_name = localized_name.trim();
	Some(ScreenInfo {
		localized_name: (!localized_name.is_empty()).then(|| localized_name.to_owned()),
		is_primary,
	})
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
	let frame = ns_screen_for(monitor, mtm)
		.map(|screen| screen.frame())
		.ok_or_else(|| PlatformError::MonitorUnavailable(DISCONNECTED_DISPLAY_NAME.to_owned()))?;
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

/// An `NSProcessInfo` activity that keeps App Nap and timer coalescing away from MUR.
///
/// The activity ends when the value is dropped.
pub struct AppNapActivity {
	activity: Retained<ProtocolObject<dyn NSObjectProtocol>>,
}

impl std::fmt::Debug for AppNapActivity {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter.debug_struct("AppNapActivity").finish_non_exhaustive()
	}
}

impl AppNapActivity {
	/// Begins a `UserInitiated | LatencyCritical | IdleDisplaySleepDisabled` activity
	/// explaining live playback.
	///
	/// `IdleDisplaySleepDisabled` keeps the displays awake even when the operator drives
	/// a set purely over MIDI, which macOS does not count as user activity.
	pub fn begin() -> Self {
		let reason = NSString::from_str("MUR live video output must render at a steady 60 fps");
		let activity = NSProcessInfo::processInfo().beginActivityWithOptions_reason(
			NSActivityOptions::UserInitiated
				| NSActivityOptions::LatencyCritical
				| NSActivityOptions::IdleDisplaySleepDisabled,
			&reason,
		);
		Self { activity }
	}
}

impl Drop for AppNapActivity {
	fn drop(&mut self) {
		// SAFETY: `self.activity` is the token returned by `beginActivityWithOptions:reason:`
		// on the shared process info, which is exactly the type `endActivity:` expects, and
		// it is ended once because `drop` runs once.
		unsafe { NSProcessInfo::processInfo().endActivity(&self.activity) };
	}
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
			// A held `Cmd+Q` auto-repeats the menu key equivalent; only the initial press
			// may count, or one long press would open the exit guard and confirm it.
			if let Some(mtm) = MainThreadMarker::new()
				&& let Some(event) = NSApplication::sharedApplication(mtm).currentEvent()
				// `isARepeat` raises for non-key events, so check the type first.
				&& matches!(event.r#type(), NSEventType::KeyDown | NSEventType::KeyUp)
				&& event.isARepeat()
			{
				return;
			}
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
