//! Four isolated, panic-supervised per-layer decode workers.
//!
//! Each [`LayerIndex`] owns one worker thread. The render thread publishes the
//! desired playback state with [`LayerStreamer::request_layer_state`] (a short,
//! lock-protected struct update plus a non-blocking wake-up) and fetches
//! decoded frames with [`LayerStreamer::poll_frame`], which never waits for a
//! decode: if the frame is not ready it returns `None` and the renderer keeps
//! showing the previous frame (SPEC §8).
//!
//! Workers keep their clip's file open, prefetch up to [`PREFETCH_AHEAD`]
//! frames along the playback trajectory into a bounded ring buffer, and discard
//! stale work via a per-layer `generation` counter. When the exact frame is not
//! decoded yet, [`LayerStreamer::poll_frame`] serves the nearest buffered frame
//! of the current generation behind the playhead (or at most
//! [`MAX_SERVE_AHEAD`] frame ahead of it) instead of freezing.
//!
//! I/O errors, deletion, and unsupported data stop the affected layer at once
//! and are reported as [`LayerStreamEvent::Error`]. An isolated corrupt frame is
//! skipped; only [`MAX_CONSECUTIVE_FRAME_ERRORS`] failures in a row stop the
//! layer. A panicking worker is caught with [`std::panic::catch_unwind`],
//! reported, and restarted. Dropping the [`LayerStreamer`] waits at most
//! [`SHUTDOWN_TIMEOUT`] for the workers, so a worker stuck in slow I/O can
//! never hang the app on exit.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TrySendError};

use crate::error::MediaError;
use crate::model::layer::{DEFAULT_CLIP_FPS, LAYER_COUNT};
use crate::model::{ClipId, ClipKind, DecodedFrame, LayerIndex, PlaybackMode};

use super::hap::decode_hap_frame;
use super::mov::{MovIndex, read_mov_sample_into};
use super::{ClipAsset, ClipBacking};

/// Number of frames prefetched ahead of the playhead in the playback direction.
pub const PREFETCH_AHEAD: usize = 12;
/// Maximum number of decoded frames buffered per layer.
pub const RING_CAPACITY: usize = PREFETCH_AHEAD + 4;
/// Output frame rate assumed when spacing prefetch for fast playback.
pub const RENDER_FPS: f32 = 60.0;
/// Maximum number of undrained events kept before new ones are dropped.
pub const EVENT_CAPACITY: usize = 256;
/// Frames just behind the playhead that stay buffered, absorbing tick jitter.
pub const RETAIN_BEHIND: usize = 2;
/// Furthest a fallback frame may lie *behind* the playhead in [`LayerStreamer::poll_frame`].
pub const MAX_SERVE_BEHIND: usize = PREFETCH_AHEAD;
/// Furthest a fallback frame may lie *ahead of* the playhead in [`LayerStreamer::poll_frame`].
///
/// Kept at one frame (sub-frame playhead rounding) so an overloaded layer never
/// jumps into future content; beyond that the renderer holds the previous
/// frame instead (SPEC §8: skip and hold the previous valid frame).
pub const MAX_SERVE_AHEAD: usize = 1;
/// Corrupt or undecodable frames in a row that stop a layer's stream.
pub const MAX_CONSECUTIVE_FRAME_ERRORS: u32 = 8;
/// Longest time [`LayerStreamer`]'s `Drop` waits for its workers to exit.
pub const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

/// How long an idle worker sleeps before re-checking its control block.
const IDLE_WAIT: Duration = Duration::from_millis(50);
/// Initial delay before restarting a panicked worker.
const RESTART_BACKOFF: Duration = Duration::from_millis(10);
/// Upper bound of the restart delay.
const MAX_RESTART_BACKOFF: Duration = Duration::from_secs(1);

/// Asynchronous notification from a layer worker.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerStreamEvent {
	/// The layer stopped streaming its clip because of an I/O or decode failure.
	Error {
		/// Affected layer.
		layer: LayerIndex,
		/// Clip that failed.
		clip_id: ClipId,
		/// Generation of the layer when the failure happened.
		generation: u64,
		/// Human readable failure description.
		message: String,
	},
	/// The worker panicked and was restarted by its supervisor.
	WorkerRestarted {
		/// Affected layer.
		layer: LayerIndex,
		/// Total number of restarts of this layer's worker.
		restarts: u32,
	},
}

/// Computes the frames a layer should hold, in expected playback order.
///
/// The first entry is always `desired` (clamped to the clip). While playing, up
/// to `ahead` further frames (at most [`RING_CAPACITY`]` - 1`) follow along the
/// playback trajectory: with `step` clip frames advanced per output frame at
/// [`RENDER_FPS`], output frame `k` shows one of the frames `floor(k * step)` or
/// `ceil(k * step)` after `desired` (depending on the playhead's fractional
/// phase), so both are included. For `step <= 1` — and any fractional step below
/// two, such as `1.37` — this yields every consecutive frame; integer steps such
/// as `4.0` yield exactly the visited frames. `Loop` wraps around both ends;
/// `OneShot` stops at the clip boundaries.
pub fn prefetch_window(
	desired: u32,
	frame_count: u32,
	speed: f32,
	clip_fps: f32,
	mode: PlaybackMode,
	is_playing: bool,
	ahead: usize,
) -> Vec<u32> {
	let Some(last) = frame_count.checked_sub(1) else {
		return Vec::new();
	};
	let desired = desired.min(last);
	let ahead = ahead.min(RING_CAPACITY.saturating_sub(1));
	let mut window = Vec::with_capacity(ahead.saturating_add(1));
	window.push(desired);
	if !is_playing || !speed.is_finite() || speed == 0.0 || frame_count == 1 {
		return window;
	}
	let fps = if clip_fps.is_finite() && clip_fps > 0.0 {
		clip_fps
	} else {
		DEFAULT_CLIP_FPS
	};
	let step = f64::from(speed.abs() * fps / RENDER_FPS);
	let direction: i64 = if speed < 0.0 { -1 } else { 1 };
	let count = i64::from(frame_count);
	let limit = ahead.saturating_add(1);

	// Every offset along the trajectory, in playback order; each output frame
	// contributes `floor(k * step)` and `ceil(k * step)`.
	let push_offset = |offset: i64, window: &mut Vec<u32>| -> bool {
		let raw = i64::from(desired).saturating_add(offset.saturating_mul(direction));
		let frame = match mode {
			PlaybackMode::Loop => raw.rem_euclid(count),
			PlaybackMode::OneShot => {
				if raw < 0 || raw >= count {
					return false;
				}
				raw
			}
		};
		if let Ok(frame) = u32::try_from(frame)
			&& !window.contains(&frame)
		{
			window.push(frame);
		}
		true
	};
	if step <= 1.0 {
		for offset in 1..=i64::from(frame_count) {
			if window.len() >= limit || !push_offset(offset, &mut window) {
				break;
			}
		}
		return window;
	}
	// One trajectory pass over the clip is enough: later offsets only revisit frames.
	let max_steps = (f64::from(frame_count) / step).ceil() as i64 + 1;
	'trajectory: for k in 1..=max_steps.max(1) {
		let position = k as f64 * step;
		let lower = position.floor();
		let mut offsets = [lower as i64, lower as i64];
		if position - lower > TRAJECTORY_EPSILON {
			offsets[1] = lower as i64 + 1;
		}
		for offset in offsets {
			if window.len() >= limit || !push_offset(offset, &mut window) {
				break 'trajectory;
			}
		}
	}
	window
}

/// Fractional parts below this are treated as an exact integer trajectory step.
const TRAJECTORY_EPSILON: f64 = 1e-6;

/// Frames just behind `desired` in the playback direction that stay buffered.
fn behind_frames(desired: u32, frame_count: u32, speed: f32, mode: PlaybackMode) -> Vec<u32> {
	let mut frames = Vec::with_capacity(RETAIN_BEHIND);
	if frame_count <= 1 {
		return frames;
	}
	let direction: i64 = if speed < 0.0 { -1 } else { 1 };
	let count = i64::from(frame_count);
	for offset in 1..=RETAIN_BEHIND as i64 {
		let raw = i64::from(desired).saturating_sub(offset.saturating_mul(direction));
		let frame = match mode {
			PlaybackMode::Loop => raw.rem_euclid(count),
			PlaybackMode::OneShot => {
				if raw < 0 || raw >= count {
					break;
				}
				raw
			}
		};
		if let Ok(frame) = u32::try_from(frame)
			&& frame != desired
			&& !frames.contains(&frame)
		{
			frames.push(frame);
		}
	}
	frames
}

/// Rank of a buffered frame `offset` frames from the playhead as a fallback.
///
/// Returns `None` outside the serve window ([`MAX_SERVE_BEHIND`] behind,
/// [`MAX_SERVE_AHEAD`] ahead) and for the exact frame. Lower ranks win: the
/// nearest frame first, behind before ahead at the same distance.
fn fallback_rank(offset: i64) -> Option<(i64, bool)> {
	let window = -(MAX_SERVE_BEHIND as i64)..=MAX_SERVE_AHEAD as i64;
	(offset != 0 && window.contains(&offset)).then_some((offset.abs(), offset > 0))
}

/// Signed distance of `frame` from `desired` in the playback direction.
///
/// Positive values lie ahead, negative values behind. `Loop` clips measure the
/// shorter way around.
fn playback_offset(frame: u32, desired: u32, frame_count: u32, forward: bool, mode: PlaybackMode) -> i64 {
	let mut offset = i64::from(frame) - i64::from(desired);
	if !forward {
		offset = -offset;
	}
	if mode == PlaybackMode::Loop && frame_count > 0 {
		let count = i64::from(frame_count);
		offset = offset.rem_euclid(count);
		if offset > count / 2 {
			offset -= count;
		}
	}
	offset
}

// ---------------------------------------------------------------------------
// Shared per-layer state
// ---------------------------------------------------------------------------

/// Disk source of a streamed clip.
#[derive(Debug, Clone)]
struct StreamSource {
	path: PathBuf,
	index: Arc<MovIndex>,
}

/// Latest playback request for a layer.
#[derive(Debug, Clone)]
struct StreamTarget {
	clip_id: ClipId,
	desired_frame: u32,
	speed: f32,
	mode: PlaybackMode,
	is_playing: bool,
	frame_count: u32,
	fps: f32,
	source: Option<StreamSource>,
}

#[derive(Debug, Default)]
struct LayerControl {
	/// Incremented on every request.
	seq: u64,
	/// Incremented whenever the bound clip changes or the layer is cleared.
	generation: u64,
	target: Option<StreamTarget>,
	/// Generation that hit an unrecoverable error; no decoding until it changes.
	failed_generation: Option<u64>,
	shutdown: bool,
}

#[derive(Debug, Default)]
struct FrameRing {
	generation: u64,
	entries: VecDeque<(ClipId, Arc<DecodedFrame>)>,
}

impl FrameRing {
	fn contains(&self, clip_id: ClipId, frame_index: u32) -> bool {
		self.entries
			.iter()
			.any(|(id, frame)| *id == clip_id && frame.frame_index == frame_index)
	}
}

#[derive(Debug)]
struct LayerShared {
	layer: LayerIndex,
	control: Mutex<LayerControl>,
	ring: Mutex<FrameRing>,
	wake_tx: Sender<()>,
	wake_rx: Receiver<()>,
	events: Sender<LayerStreamEvent>,
	restarts: AtomicU32,
	/// Generation the worker is currently processing (for panic attribution).
	active_generation: AtomicU64,
	/// Set by the supervisor right before the worker thread exits.
	finished: Mutex<bool>,
	/// Signalled together with `finished`.
	finished_cv: Condvar,
	#[cfg(test)]
	inject_panic: std::sync::atomic::AtomicBool,
	/// Number of wake-ups sent to the worker (to verify request deduplication).
	#[cfg(test)]
	wakes: AtomicU64,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
	mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl LayerShared {
	fn new(layer: LayerIndex, events: Sender<LayerStreamEvent>) -> Self {
		let (wake_tx, wake_rx) = crossbeam_channel::bounded(1);
		Self {
			layer,
			control: Mutex::new(LayerControl::default()),
			ring: Mutex::new(FrameRing::default()),
			wake_tx,
			wake_rx,
			events,
			restarts: AtomicU32::new(0),
			active_generation: AtomicU64::new(0),
			finished: Mutex::new(false),
			finished_cv: Condvar::new(),
			#[cfg(test)]
			inject_panic: std::sync::atomic::AtomicBool::new(false),
			#[cfg(test)]
			wakes: AtomicU64::new(0),
		}
	}

	/// Wakes the worker without ever blocking; a pending token is enough.
	fn wake(&self) {
		#[cfg(test)]
		self.wakes.fetch_add(1, Ordering::Relaxed);
		let _ = self.wake_tx.try_send(());
	}

	/// Records that the worker thread is about to exit and notifies `Drop`.
	fn mark_finished(&self) {
		*lock(&self.finished) = true;
		self.finished_cv.notify_all();
	}

	/// Waits until the worker finished or `deadline` passed; returns `true` when finished.
	fn wait_finished(&self, deadline: Instant) -> bool {
		let mut finished = lock(&self.finished);
		while !*finished {
			let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
				return false;
			};
			if remaining.is_zero() {
				return false;
			}
			finished = self
				.finished_cv
				.wait_timeout(finished, remaining)
				.map(|(guard, _)| guard)
				.unwrap_or_else(|poisoned| poisoned.into_inner().0);
		}
		true
	}

	/// Drops every buffered frame and tags the ring with `generation`.
	///
	/// Callers must not hold the `control` lock (lock order stays flat).
	fn reset_ring(&self, generation: u64) {
		let mut ring = lock(&self.ring);
		ring.generation = generation;
		ring.entries.clear();
	}

	fn emit(&self, event: LayerStreamEvent) {
		match self.events.try_send(event) {
			Ok(()) => {}
			Err(TrySendError::Full(dropped)) => {
				tracing::warn!(layer = %self.layer, ?dropped, "layer event queue full; dropping event");
			}
			Err(TrySendError::Disconnected(_)) => {}
		}
	}

	fn report_error(&self, clip_id: ClipId, generation: u64, message: String) {
		tracing::warn!(layer = %self.layer, clip = clip_id.0, generation, %message, "layer streaming stopped");
		{
			let mut control = lock(&self.control);
			if control.generation == generation {
				control.failed_generation = Some(generation);
			}
		}
		self.emit(LayerStreamEvent::Error {
			layer: self.layer,
			clip_id,
			generation,
			message,
		});
	}

	/// Clears the bound clip, bumps the generation, and drops buffered frames at once.
	///
	/// After this returns, [`LayerStreamer::poll_frame`] serves nothing from the
	/// ring for this layer until a new clip is requested.
	fn clear(&self) {
		let generation = {
			let mut control = lock(&self.control);
			if control.target.is_some() {
				control.generation = control.generation.wrapping_add(1);
			}
			control.target = None;
			control.seq = control.seq.wrapping_add(1);
			control.generation
		};
		self.reset_ring(generation);
		self.wake();
	}
}

// ---------------------------------------------------------------------------
// Public streamer
// ---------------------------------------------------------------------------

/// Registry of loaded clips plus the four per-layer decode workers.
///
/// All methods take `&self`, so the streamer can be shared between the UI and
/// render threads behind an `Arc`.
#[derive(Debug)]
pub struct LayerStreamer {
	assets: RwLock<HashMap<ClipId, ClipAsset>>,
	layers: Vec<Arc<LayerShared>>,
	events_rx: Receiver<LayerStreamEvent>,
	handles: Mutex<Vec<JoinHandle<()>>>,
}

impl LayerStreamer {
	/// Spawns one supervised decode worker per layer.
	pub fn new() -> Result<Self, MediaError> {
		let (events_tx, events_rx) = crossbeam_channel::bounded(EVENT_CAPACITY);
		let layers: Vec<Arc<LayerShared>> = LayerIndex::ALL
			.iter()
			.map(|&layer| Arc::new(LayerShared::new(layer, events_tx.clone())))
			.collect();
		let mut handles = Vec::with_capacity(LAYER_COUNT);
		for shared in &layers {
			let worker_shared = Arc::clone(shared);
			let spawned = std::thread::Builder::new()
				.name(format!("mur-decode-{}", shared.layer))
				.spawn(move || supervise(&worker_shared));
			match spawned {
				Ok(handle) => handles.push(handle),
				Err(error) => {
					for started in &layers {
						lock(&started.control).shutdown = true;
						started.wake();
					}
					for handle in handles {
						let _ = handle.join();
					}
					return Err(MediaError::WorkerStopped(format!(
						"failed to spawn decode worker for {}: {error}",
						shared.layer
					)));
				}
			}
		}
		Ok(Self {
			assets: RwLock::new(HashMap::new()),
			layers,
			events_rx,
			handles: Mutex::new(handles),
		})
	}

	fn layer(&self, layer: LayerIndex) -> Option<&Arc<LayerShared>> {
		self.layers.get(layer.index())
	}

	/// Adds or replaces a clip. Layers bound to a replaced clip are cleared.
	pub fn register_asset(&self, asset: ClipAsset) {
		let id = asset.id();
		let replaced = self
			.assets
			.write()
			.unwrap_or_else(PoisonError::into_inner)
			.insert(id, asset)
			.is_some();
		if replaced {
			self.clear_layers_bound_to(id);
		}
	}

	/// Removes a clip and clears every layer streaming it.
	pub fn remove_asset(&self, id: ClipId) -> Option<ClipAsset> {
		let removed = self.assets.write().unwrap_or_else(PoisonError::into_inner).remove(&id);
		self.clear_layers_bound_to(id);
		removed
	}

	/// Returns a clone of the registered asset `id`.
	pub fn asset(&self, id: ClipId) -> Option<ClipAsset> {
		self.assets
			.read()
			.unwrap_or_else(PoisonError::into_inner)
			.get(&id)
			.cloned()
	}

	/// Number of registered clips.
	pub fn asset_count(&self) -> usize {
		self.assets.read().unwrap_or_else(PoisonError::into_inner).len()
	}

	fn clear_layers_bound_to(&self, id: ClipId) {
		for shared in &self.layers {
			let bound = lock(&shared.control).target.as_ref().map(|target| target.clip_id) == Some(id);
			if bound {
				shared.clear();
			}
		}
	}

	/// Publishes the desired playback state of `layer` without blocking.
	///
	/// Changing `clip_id` bumps the layer generation, which drops buffered
	/// frames and makes the worker reopen the file. Returns `false` (and clears
	/// the layer) when `clip_id` is not registered.
	pub fn request_layer_state(
		&self,
		layer: LayerIndex,
		clip_id: ClipId,
		desired_frame: u32,
		speed: f32,
		mode: PlaybackMode,
		is_playing: bool,
	) -> bool {
		let Some(shared) = self.layer(layer) else {
			return false;
		};
		let speed = if speed.is_finite() { speed } else { 0.0 };
		let in_place = {
			let mut guard = lock(&shared.control);
			let control = &mut *guard;
			match control.target.as_mut() {
				Some(target) if target.clip_id == clip_id => {
					let desired_frame = desired_frame.min(target.frame_count.saturating_sub(1));
					let unchanged = target.desired_frame == desired_frame
						&& target.speed.to_bits() == speed.to_bits()
						&& target.mode == mode
						&& target.is_playing == is_playing;
					if unchanged {
						// Nothing new for the worker: no sequence bump, no wake-up.
						Some(false)
					} else {
						target.desired_frame = desired_frame;
						target.speed = speed;
						target.mode = mode;
						target.is_playing = is_playing;
						control.seq = control.seq.wrapping_add(1);
						Some(true)
					}
				}
				_ => None,
			}
		};
		match in_place {
			Some(true) => {
				shared.wake();
				return true;
			}
			Some(false) => return true,
			None => {}
		}

		let Some(asset) = self.asset(clip_id) else {
			shared.clear();
			return false;
		};
		let source = match &asset.backing {
			ClipBacking::StaticImage => None,
			ClipBacking::HapMov { path, index } => Some(StreamSource {
				path: path.clone(),
				index: Arc::clone(index),
			}),
		};
		let frame_count = asset.frame_count().max(1);
		let generation = {
			let mut control = lock(&shared.control);
			control.generation = control.generation.wrapping_add(1);
			control.failed_generation = None;
			control.target = Some(StreamTarget {
				clip_id,
				desired_frame: desired_frame.min(frame_count.saturating_sub(1)),
				speed,
				mode,
				is_playing,
				frame_count,
				fps: asset.info.fps,
				source,
			});
			control.seq = control.seq.wrapping_add(1);
			control.generation
		};
		// Frames of the previous clip must never be served for the new generation.
		shared.reset_ring(generation);
		shared.wake();
		true
	}

	/// Unbinds the clip from `layer`, dropping its buffered frames.
	pub fn clear_layer(&self, layer: LayerIndex) {
		if let Some(shared) = self.layer(layer) {
			shared.clear();
		}
	}

	/// Returns the decoded frame `desired_frame` of `clip_id`, or the nearest ready one.
	///
	/// Never waits for decoding. Frame `0` (and every frame of a still image)
	/// is served from the resident first frame. When the exact frame is not
	/// decoded yet (decode overload, a skipped corrupt frame), the buffered frame
	/// of the layer's current generation closest to `desired_frame` is returned,
	/// preferring frames just behind the playhead over frames ahead of it, within
	/// [`MAX_SERVE_BEHIND`] frames behind and [`MAX_SERVE_AHEAD`] frame ahead, so
	/// overload never skips into future content. Frames of a cleared or replaced
	/// generation are never returned. `None` means "keep showing the previous
	/// frame".
	pub fn poll_frame(&self, layer: LayerIndex, clip_id: ClipId, desired_frame: u32) -> Option<Arc<DecodedFrame>> {
		{
			let assets = self.assets.read().unwrap_or_else(PoisonError::into_inner);
			let asset = assets.get(&clip_id)?;
			if desired_frame == 0 || matches!(asset.backing, ClipBacking::StaticImage) {
				return Some(Arc::clone(&asset.first_frame));
			}
		}
		let shared = self.layer(layer)?;
		// Read the control block first and release it before locking the ring, so
		// the two locks are never nested.
		let (generation, playback) = {
			let control = lock(&shared.control);
			let playback = control
				.target
				.as_ref()
				.filter(|target| target.clip_id == clip_id)
				.map(|target| (target.frame_count, target.speed >= 0.0, target.mode));
			(control.generation, playback)
		};
		let (frame_count, forward, mode) = playback?;
		let ring = lock(&shared.ring);
		if ring.generation != generation {
			return None;
		}
		if let Some((_, frame)) = ring
			.entries
			.iter()
			.find(|(id, frame)| *id == clip_id && frame.frame_index == desired_frame)
		{
			return Some(Arc::clone(frame));
		}
		ring.entries
			.iter()
			.filter(|(id, _)| *id == clip_id)
			.filter_map(|(_, frame)| {
				let offset = playback_offset(frame.frame_index, desired_frame, frame_count, forward, mode);
				fallback_rank(offset).map(|rank| (rank, frame))
			})
			.min_by_key(|(rank, _)| *rank)
			.map(|(_, frame)| Arc::clone(frame))
	}

	/// Takes every pending worker event.
	pub fn drain_events(&self) -> Vec<LayerStreamEvent> {
		self.events_rx.try_iter().collect()
	}

	/// Current generation of `layer`.
	pub fn layer_generation(&self, layer: LayerIndex) -> Option<u64> {
		self.layer(layer).map(|shared| lock(&shared.control).generation)
	}

	/// Frame indices currently buffered for `layer`, in insertion order.
	pub fn buffered_frames(&self, layer: LayerIndex) -> Vec<u32> {
		self.layer(layer)
			.map(|shared| {
				lock(&shared.ring)
					.entries
					.iter()
					.map(|(_, frame)| frame.frame_index)
					.collect()
			})
			.unwrap_or_default()
	}

	/// Number of times the worker of `layer` was restarted after a panic.
	pub fn worker_restarts(&self, layer: LayerIndex) -> u32 {
		self.layer(layer)
			.map(|shared| shared.restarts.load(Ordering::Relaxed))
			.unwrap_or(0)
	}

	/// Makes the next decode on `layer` panic, to exercise the supervisor.
	#[cfg(test)]
	fn inject_panic(&self, layer: LayerIndex) {
		if let Some(shared) = self.layer(layer) {
			shared.inject_panic.store(true, Ordering::SeqCst);
			shared.wake();
		}
	}
}

impl Drop for LayerStreamer {
	/// Signals every worker to stop and waits up to [`SHUTDOWN_TIMEOUT`] in total.
	///
	/// A worker still blocked in slow I/O after the deadline is detached with a
	/// warning instead of hanging the caller; it exits on its own once the read
	/// returns, because the shutdown flag stays set.
	fn drop(&mut self) {
		for shared in &self.layers {
			lock(&shared.control).shutdown = true;
			shared.wake();
		}
		let deadline = Instant::now() + SHUTDOWN_TIMEOUT;
		let handles = std::mem::take(&mut *lock(&self.handles));
		for (shared, handle) in self.layers.iter().zip(handles) {
			if shared.wait_finished(deadline) {
				let _ = handle.join();
			} else {
				tracing::warn!(
					layer = %shared.layer,
					timeout_ms = SHUTDOWN_TIMEOUT.as_millis() as u64,
					"decode worker did not stop in time; detaching it"
				);
				drop(handle);
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// Runs [`supervise_loop`] and reports the thread's exit to `Drop`.
fn supervise(shared: &LayerShared) {
	supervise_loop(shared);
	shared.mark_finished();
}

/// Restarts `worker_main` after panics until shutdown.
fn supervise_loop(shared: &LayerShared) {
	let mut backoff = RESTART_BACKOFF;
	loop {
		let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker_main(shared)));
		let payload = match outcome {
			Ok(()) => return,
			Err(payload) => payload,
		};
		let message = payload
			.downcast_ref::<&str>()
			.map(|text| (*text).to_string())
			.or_else(|| payload.downcast_ref::<String>().cloned())
			.unwrap_or_else(|| "unknown panic".to_string());
		let restarts = shared.restarts.fetch_add(1, Ordering::Relaxed).saturating_add(1);
		tracing::error!(layer = %shared.layer, restarts, %message, "decode worker panicked; restarting");

		let generation = shared.active_generation.load(Ordering::Relaxed);
		let (clip_id, shutdown) = {
			let mut control = lock(&shared.control);
			let clip_id = if control.generation == generation {
				control.failed_generation = Some(generation);
				control.target.as_ref().map(|target| target.clip_id)
			} else {
				None
			};
			(clip_id, control.shutdown)
		};
		if let Some(clip_id) = clip_id {
			shared.emit(LayerStreamEvent::Error {
				layer: shared.layer,
				clip_id,
				generation,
				message: format!("decode worker panicked: {message}"),
			});
		}
		shared.emit(LayerStreamEvent::WorkerRestarted {
			layer: shared.layer,
			restarts,
		});
		if shutdown {
			return;
		}
		std::thread::sleep(backoff);
		backoff = backoff.saturating_mul(2).min(MAX_RESTART_BACKOFF);
	}
}

/// Clip opened by the worker for the current generation.
struct OpenClip {
	clip_id: ClipId,
	path: PathBuf,
	file: File,
	index: Arc<MovIndex>,
}

/// What the worker learned from one look at the control block.
struct Snapshot {
	seq: u64,
	generation: u64,
	failed: bool,
	target: Option<TargetView>,
	/// Present only when the generation differs from the worker's.
	source: Option<StreamSource>,
}

#[derive(Clone, Copy)]
struct TargetView {
	clip_id: ClipId,
	desired_frame: u32,
	speed: f32,
	mode: PlaybackMode,
	is_playing: bool,
	frame_count: u32,
	fps: f32,
}

fn take_snapshot(shared: &LayerShared, known_generation: Option<u64>) -> Option<Snapshot> {
	let control = lock(&shared.control);
	if control.shutdown {
		return None;
	}
	let generation_changed = known_generation != Some(control.generation);
	Some(Snapshot {
		seq: control.seq,
		generation: control.generation,
		failed: control.failed_generation == Some(control.generation),
		target: control.target.as_ref().map(|target| TargetView {
			clip_id: target.clip_id,
			desired_frame: target.desired_frame,
			speed: target.speed,
			mode: target.mode,
			is_playing: target.is_playing,
			frame_count: target.frame_count,
			fps: target.fps,
		}),
		source: if generation_changed {
			control.target.as_ref().and_then(|target| target.source.clone())
		} else {
			None
		},
	})
}

fn worker_main(shared: &LayerShared) {
	let mut generation: Option<u64> = None;
	let mut applied_seq: Option<u64> = None;
	let mut open: Option<OpenClip> = None;
	let mut window: Vec<u32> = Vec::new();
	let mut retained: Vec<u32> = Vec::new();
	let mut scratch: Vec<u8> = Vec::new();
	// Frames of this generation that failed to decode; skipped instead of retried.
	let mut skipped: HashSet<u32> = HashSet::new();
	let mut consecutive_decode_errors: u32 = 0;

	loop {
		let Some(snapshot) = take_snapshot(shared, generation) else {
			return;
		};
		shared.active_generation.store(snapshot.generation, Ordering::Relaxed);

		if generation != Some(snapshot.generation) {
			generation = Some(snapshot.generation);
			applied_seq = None;
			open = None;
			window.clear();
			retained.clear();
			skipped.clear();
			consecutive_decode_errors = 0;
			shared.reset_ring(snapshot.generation);
			if let (Some(source), Some(target), false) = (snapshot.source, snapshot.target, snapshot.failed) {
				match File::open(&source.path) {
					Ok(file) => {
						open = Some(OpenClip {
							clip_id: target.clip_id,
							path: source.path,
							file,
							index: source.index,
						});
					}
					Err(error) => {
						shared.report_error(
							target.clip_id,
							snapshot.generation,
							format!("cannot open `{}`: {error}", source.path.display()),
						);
					}
				}
			}
		}

		if snapshot.failed {
			open = None;
		}

		if applied_seq != Some(snapshot.seq) {
			applied_seq = Some(snapshot.seq);
			(window, retained) = match snapshot.target {
				Some(target) if open.is_some() => (
					prefetch_window(
						target.desired_frame,
						target.frame_count,
						target.speed,
						target.fps,
						target.mode,
						target.is_playing,
						PREFETCH_AHEAD,
					),
					behind_frames(target.desired_frame, target.frame_count, target.speed, target.mode),
				),
				_ => (Vec::new(), Vec::new()),
			};
			// Keep the window plus the frames just behind the playhead, so tick
			// jitter never evicts a freshly decoded frame that is still on screen.
			let mut ring = lock(&shared.ring);
			ring.entries
				.retain(|(_, frame)| window.contains(&frame.frame_index) || retained.contains(&frame.frame_index));
		}

		let next = match &open {
			Some(clip) => {
				let ring = lock(&shared.ring);
				window
					.iter()
					.copied()
					.find(|&frame| frame != 0 && !skipped.contains(&frame) && !ring.contains(clip.clip_id, frame))
			}
			None => None,
		};

		let (Some(frame_index), Some(clip)) = (next, open.as_mut()) else {
			match shared.wake_rx.recv_timeout(IDLE_WAIT) {
				Ok(()) | Err(RecvTimeoutError::Timeout) => continue,
				Err(RecvTimeoutError::Disconnected) => return,
			}
		};

		#[cfg(test)]
		if shared.inject_panic.swap(false, Ordering::SeqCst) {
			injected_panic();
		}

		let current_generation = snapshot.generation;
		let frame_count = snapshot.target.map_or(0, |target| target.frame_count);
		match decode_frame(clip, frame_index, &mut scratch) {
			Ok(frame) => {
				consecutive_decode_errors = 0;
				let mut ring = lock(&shared.ring);
				if ring.generation == current_generation {
					while ring.entries.len() >= RING_CAPACITY {
						// Evict frames outside the window first, oldest first.
						let stale = ring
							.entries
							.iter()
							.position(|(_, buffered)| !window.contains(&buffered.frame_index));
						match stale {
							Some(position) => {
								let _ = ring.entries.remove(position);
							}
							None => {
								let _ = ring.entries.pop_front();
							}
						}
					}
					ring.entries.push_back((clip.clip_id, Arc::new(frame)));
				}
			}
			Err(error) => {
				consecutive_decode_errors = consecutive_decode_errors.saturating_add(1);
				let message = format!("{}: {error}", clip.path.display());
				let clip_id = clip.clip_id;
				if is_fatal_stream_error(&error, frame_count, consecutive_decode_errors) {
					open = None;
					shared.report_error(clip_id, current_generation, message);
				} else {
					tracing::warn!(
						layer = %shared.layer,
						clip = clip_id.0,
						frame_index,
						consecutive = consecutive_decode_errors,
						%message,
						"skipping undecodable frame"
					);
					skipped.insert(frame_index);
				}
			}
		}
	}
}

/// Returns `true` when `error` must stop the layer's stream for this generation.
///
/// I/O failures (deleted or truncated file, unplugged drive), unsupported data,
/// and failures of single-frame clips stop the stream at once. Corrupt or
/// undecodable frames of a multi-frame clip are skipped until
/// [`MAX_CONSECUTIVE_FRAME_ERRORS`] of them fail in a row.
fn is_fatal_stream_error(error: &MediaError, frame_count: u32, consecutive: u32) -> bool {
	match error {
		MediaError::Corrupt { .. } | MediaError::Decode { .. } => {
			frame_count <= 1 || consecutive >= MAX_CONSECUTIVE_FRAME_ERRORS
		}
		MediaError::Io { .. } | MediaError::UnsupportedFormat(_) | MediaError::WorkerStopped(_) => true,
	}
}

#[cfg(test)]
#[allow(clippy::panic)]
fn injected_panic() -> ! {
	panic!("injected decode worker panic")
}

fn decode_frame(clip: &mut OpenClip, frame_index: u32, scratch: &mut Vec<u8>) -> Result<DecodedFrame, MediaError> {
	let position = usize::try_from(frame_index).map_err(|_| MediaError::Decode {
		frame_index,
		detail: "frame index overflow".into(),
	})?;
	let sample = clip.index.samples.get(position).ok_or_else(|| MediaError::Decode {
		frame_index,
		detail: "frame index beyond the sample table".into(),
	})?;
	read_mov_sample_into(&mut clip.file, sample, scratch)?;
	let kind: ClipKind = clip.index.kind;
	decode_hap_frame(scratch, kind, clip.index.width, clip.index.height, frame_index)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use std::io::{Seek, SeekFrom, Write};
	use std::path::Path;
	use std::time::Instant;

	use super::*;
	use crate::media::load_clip;
	use crate::media::test_fixtures::{FixtureCompression, HapMovFixture, TempDir};
	use crate::model::PixelPayload;

	const L1: LayerIndex = LayerIndex::ALL[0];
	const L2: LayerIndex = LayerIndex::ALL[1];

	fn wait_until(mut condition: impl FnMut() -> bool) -> bool {
		let start = Instant::now();
		while start.elapsed() < Duration::from_secs(10) {
			if condition() {
				return true;
			}
			std::thread::sleep(Duration::from_millis(2));
		}
		false
	}

	fn hap_clip(dir: &Path, name: &str, id: u64, frames: u32) -> (ClipAsset, HapMovFixture) {
		let path = dir.join(name);
		let mut fixture = HapMovFixture::new(ClipKind::Hap, 32, 16, frames);
		fixture.compression = FixtureCompression::Chunked {
			chunks: 2,
			mixed: false,
			offset_table: false,
		};
		fixture.write_to(&path).unwrap();
		let (_, asset) = load_clip(ClipId(id), &path).unwrap();
		(asset, fixture)
	}

	fn payload_bytes(frame: &DecodedFrame) -> &[u8] {
		match &frame.payload {
			PixelPayload::Bc1(bytes) => bytes,
			other => panic!("unexpected payload {other:?}"),
		}
	}

	#[test]
	fn window_loop_forward_and_reverse_wrap() {
		let forward = prefetch_window(8, 10, 1.0, 60.0, PlaybackMode::Loop, true, 4);
		assert_eq!(forward, [8, 9, 0, 1, 2]);
		let reverse = prefetch_window(1, 10, -1.0, 60.0, PlaybackMode::Loop, true, 4);
		assert_eq!(reverse, [1, 0, 9, 8, 7]);
	}

	#[test]
	fn window_one_shot_clamps_at_ends() {
		assert_eq!(
			prefetch_window(8, 10, 1.0, 60.0, PlaybackMode::OneShot, true, 12),
			[8, 9]
		);
		assert_eq!(
			prefetch_window(2, 10, -1.0, 60.0, PlaybackMode::OneShot, true, 12),
			[2, 1, 0]
		);
	}

	#[test]
	fn window_covers_fractional_fast_trajectories() {
		// 1.37 clip frames per output frame: every consecutive frame is needed.
		let window = prefetch_window(0, 100, 1.37, 60.0, PlaybackMode::Loop, true, 12);
		assert_eq!(window, (0..=12).collect::<Vec<u32>>());
		let reverse = prefetch_window(20, 100, -1.37, 60.0, PlaybackMode::Loop, true, 5);
		assert_eq!(reverse, [20, 19, 18, 17, 16, 15]);
		// 2.5 frames per output frame: floor and ceil of each visited position.
		let window = prefetch_window(0, 100, 2.5, 60.0, PlaybackMode::Loop, true, 6);
		assert_eq!(window, [0, 2, 3, 5, 7, 8, 10]);
		// Every frame a playhead can land on is part of the window.
		for phase in [0.0, 0.25, 0.5, 0.99] {
			let window = prefetch_window(10, 1000, 1.37, 60.0, PlaybackMode::Loop, true, 12);
			for tick in 1..=6 {
				let frame = (10.0 + phase + f64::from(tick) * 1.37).floor() as u32;
				assert!(window.contains(&frame), "phase {phase} tick {tick} frame {frame}");
			}
		}
		// The window never exceeds the ring.
		assert_eq!(
			prefetch_window(0, 1000, 1.0, 60.0, PlaybackMode::Loop, true, 100).len(),
			RING_CAPACITY
		);
		// One-shot stops at the end even on a fractional trajectory.
		assert_eq!(
			prefetch_window(95, 100, 2.5, 60.0, PlaybackMode::OneShot, true, 12),
			[95, 97, 98]
		);
	}

	#[test]
	fn behind_frames_and_offsets_follow_direction() {
		assert_eq!(behind_frames(5, 10, 1.0, PlaybackMode::Loop), [4, 3]);
		assert_eq!(behind_frames(5, 10, -1.0, PlaybackMode::Loop), [6, 7]);
		assert_eq!(behind_frames(0, 10, 1.0, PlaybackMode::Loop), [9, 8]);
		assert_eq!(behind_frames(0, 10, 1.0, PlaybackMode::OneShot), Vec::<u32>::new());
		assert_eq!(behind_frames(0, 1, 1.0, PlaybackMode::Loop), Vec::<u32>::new());

		assert_eq!(playback_offset(4, 5, 10, true, PlaybackMode::Loop), -1);
		assert_eq!(playback_offset(6, 5, 10, false, PlaybackMode::Loop), -1);
		assert_eq!(playback_offset(9, 0, 10, true, PlaybackMode::Loop), -1);
		assert_eq!(playback_offset(1, 9, 10, true, PlaybackMode::Loop), 2);
		assert_eq!(playback_offset(1, 9, 10, true, PlaybackMode::OneShot), -8);
	}

	#[test]
	fn window_paused_fast_and_degenerate() {
		assert_eq!(prefetch_window(5, 10, 1.0, 60.0, PlaybackMode::Loop, false, 12), [5]);
		assert_eq!(prefetch_window(5, 10, 0.0, 60.0, PlaybackMode::Loop, true, 12), [5]);
		assert_eq!(prefetch_window(50, 10, 1.0, 60.0, PlaybackMode::Loop, false, 12), [9]);
		assert!(prefetch_window(0, 0, 1.0, 60.0, PlaybackMode::Loop, true, 12).is_empty());
		assert_eq!(prefetch_window(0, 1, 1.0, 60.0, PlaybackMode::Loop, true, 12), [0]);
		// 4x speed at 60 fps → every 4th frame.
		assert_eq!(
			prefetch_window(0, 100, 4.0, 60.0, PlaybackMode::Loop, true, 3),
			[0, 4, 8, 12]
		);
		// Slow speed still prefetches consecutive frames.
		assert_eq!(
			prefetch_window(0, 100, 0.25, 30.0, PlaybackMode::Loop, true, 3),
			[0, 1, 2, 3]
		);
		// Tiny loop deduplicates wrapped frames.
		assert_eq!(
			prefetch_window(0, 3, 1.0, 60.0, PlaybackMode::Loop, true, 12),
			[0, 1, 2]
		);
		// Non-finite fps falls back to the default.
		assert_eq!(
			prefetch_window(0, 10, 1.0, f32::NAN, PlaybackMode::Loop, true, 2),
			[0, 1, 2]
		);
	}

	#[test]
	fn first_frame_is_served_without_waiting() {
		let dir = TempDir::new("stream_first").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		let (asset, _) = hap_clip(dir.path(), "a.mov", 1, 8);
		let first = Arc::clone(&asset.first_frame);
		streamer.register_asset(asset);
		let frame = streamer.poll_frame(L1, ClipId(1), 0).unwrap();
		assert!(Arc::ptr_eq(&frame, &first));
		assert!(streamer.poll_frame(L1, ClipId(1), 3).is_none());
		assert!(streamer.poll_frame(L1, ClipId(99), 0).is_none());
		assert_eq!(streamer.asset_count(), 1);
	}

	#[test]
	fn static_images_serve_every_frame_from_memory() {
		let dir = TempDir::new("stream_static").unwrap();
		let path = dir.path().join("still.png");
		image::RgbaImage::from_pixel(4, 4, image::Rgba([9, 9, 9, 255]))
			.save(&path)
			.unwrap();
		let (_, asset) = load_clip(ClipId(2), &path).unwrap();
		let streamer = LayerStreamer::new().unwrap();
		streamer.register_asset(asset);
		assert!(streamer.request_layer_state(L1, ClipId(2), 0, 1.0, PlaybackMode::Loop, true));
		assert!(streamer.poll_frame(L1, ClipId(2), 0).is_some());
		assert!(streamer.poll_frame(L1, ClipId(2), 7).is_some());
	}

	#[test]
	fn streams_correct_frames_with_forward_prefetch() {
		let dir = TempDir::new("stream_fwd").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		let (asset, fixture) = hap_clip(dir.path(), "a.mov", 1, 30);
		streamer.register_asset(asset);
		assert!(streamer.request_layer_state(L1, ClipId(1), 5, 1.0, PlaybackMode::Loop, true));
		assert!(wait_until(|| streamer.poll_frame(L1, ClipId(1), 5).is_some()));
		let frame = streamer.poll_frame(L1, ClipId(1), 5).unwrap();
		assert_eq!(frame.frame_index, 5);
		assert_eq!(payload_bytes(&frame), fixture.frame_payload(5).unwrap().as_slice());

		let expected: Vec<u32> = (5..=5 + PREFETCH_AHEAD as u32).collect();
		assert!(wait_until(|| {
			let mut buffered = streamer.buffered_frames(L1);
			buffered.sort_unstable();
			buffered == expected
		}));
		for index in 6..=17 {
			let frame = streamer.poll_frame(L1, ClipId(1), index).unwrap();
			assert_eq!(payload_bytes(&frame), fixture.frame_payload(index).unwrap().as_slice());
		}
		// Advancing the playhead evicts frames behind it.
		assert!(streamer.request_layer_state(L1, ClipId(1), 10, 1.0, PlaybackMode::Loop, true));
		assert!(wait_until(|| {
			let buffered = streamer.buffered_frames(L1);
			!buffered.contains(&5) && buffered.contains(&22)
		}));
		assert!(streamer.drain_events().is_empty());
	}

	#[test]
	fn reverse_loop_prefetch_wraps_past_frame_zero() {
		let dir = TempDir::new("stream_rev").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		let (asset, fixture) = hap_clip(dir.path(), "a.mov", 1, 20);
		streamer.register_asset(asset);
		streamer.request_layer_state(L1, ClipId(1), 2, -1.0, PlaybackMode::Loop, true);
		// Window: 2, 1, (0 resident), 19, 18, ... 10.
		assert!(wait_until(|| {
			let buffered = streamer.buffered_frames(L1);
			buffered.contains(&19) && buffered.contains(&10) && buffered.contains(&1)
		}));
		let buffered = streamer.buffered_frames(L1);
		assert!(!buffered.contains(&0), "frame 0 stays resident only");
		assert!(!buffered.contains(&3));
		assert!(!buffered.contains(&9));
		let frame = streamer.poll_frame(L1, ClipId(1), 19).unwrap();
		assert_eq!(payload_bytes(&frame), fixture.frame_payload(19).unwrap().as_slice());
	}

	#[test]
	fn one_shot_prefetch_stops_at_clip_end() {
		let dir = TempDir::new("stream_oneshot").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		let (asset, _) = hap_clip(dir.path(), "a.mov", 1, 20);
		streamer.register_asset(asset);
		streamer.request_layer_state(L1, ClipId(1), 17, 1.0, PlaybackMode::OneShot, true);
		assert!(wait_until(|| {
			let mut buffered = streamer.buffered_frames(L1);
			buffered.sort_unstable();
			buffered == [17, 18, 19]
		}));
		std::thread::sleep(Duration::from_millis(50));
		let mut buffered = streamer.buffered_frames(L1);
		buffered.sort_unstable();
		assert_eq!(buffered, [17, 18, 19]);

		// Paused: only the desired frame.
		streamer.request_layer_state(L1, ClipId(1), 4, 1.0, PlaybackMode::OneShot, false);
		assert!(wait_until(|| streamer.buffered_frames(L1) == [4]));
	}

	#[test]
	fn clip_switch_bumps_generation_and_drops_stale_frames() {
		let dir = TempDir::new("stream_gen").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		let (a, _) = hap_clip(dir.path(), "a.mov", 1, 20);
		let (b, fixture_b) = hap_clip(dir.path(), "b.mov", 2, 20);
		streamer.register_asset(a);
		streamer.register_asset(b);
		streamer.request_layer_state(L1, ClipId(1), 3, 1.0, PlaybackMode::Loop, true);
		let gen_a = streamer.layer_generation(L1).unwrap();
		assert!(wait_until(|| streamer.poll_frame(L1, ClipId(1), 3).is_some()));

		streamer.request_layer_state(L1, ClipId(2), 3, 1.0, PlaybackMode::Loop, true);
		let gen_b = streamer.layer_generation(L1).unwrap();
		assert!(gen_b > gen_a);
		// Same clip again: no generation bump.
		streamer.request_layer_state(L1, ClipId(2), 4, 1.0, PlaybackMode::Loop, true);
		assert_eq!(streamer.layer_generation(L1).unwrap(), gen_b);

		assert!(wait_until(|| streamer.poll_frame(L1, ClipId(2), 4).is_some()));
		assert!(streamer.poll_frame(L1, ClipId(1), 4).is_none());
		let frame = streamer.poll_frame(L1, ClipId(2), 4).unwrap();
		assert_eq!(payload_bytes(&frame), fixture_b.frame_payload(4).unwrap().as_slice());
	}

	#[test]
	fn unknown_and_removed_clips_clear_the_layer() {
		let dir = TempDir::new("stream_remove").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		assert!(!streamer.request_layer_state(L1, ClipId(42), 3, 1.0, PlaybackMode::Loop, true));
		let (asset, _) = hap_clip(dir.path(), "a.mov", 1, 20);
		streamer.register_asset(asset);
		streamer.request_layer_state(L1, ClipId(1), 3, 1.0, PlaybackMode::Loop, true);
		assert!(wait_until(|| streamer.poll_frame(L1, ClipId(1), 3).is_some()));
		let before = streamer.layer_generation(L1).unwrap();
		assert!(streamer.remove_asset(ClipId(1)).is_some());
		assert!(streamer.layer_generation(L1).unwrap() > before);
		assert!(streamer.poll_frame(L1, ClipId(1), 3).is_none());
		assert!(streamer.poll_frame(L1, ClipId(1), 0).is_none());
		assert!(wait_until(|| streamer.buffered_frames(L1).is_empty()));
	}

	#[test]
	fn deleted_file_only_affects_its_layer() {
		let dir = TempDir::new("stream_delete").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		let (a, _) = hap_clip(dir.path(), "a.mov", 1, 40);
		let (b, _) = hap_clip(dir.path(), "b.mov", 2, 40);
		let (c, _) = hap_clip(dir.path(), "c.mov", 3, 40);
		streamer.register_asset(a);
		streamer.register_asset(b);
		streamer.register_asset(c);

		streamer.request_layer_state(L1, ClipId(1), 1, 1.0, PlaybackMode::Loop, true);
		streamer.request_layer_state(L2, ClipId(2), 1, 1.0, PlaybackMode::Loop, true);
		assert!(wait_until(|| streamer.poll_frame(L1, ClipId(1), 1).is_some()));
		assert!(wait_until(|| streamer.poll_frame(L2, ClipId(2), 1).is_some()));

		// Delete the file layer 1 is playing, plus the next clip it will switch to.
		std::fs::remove_file(dir.path().join("a.mov")).unwrap();
		std::fs::remove_file(dir.path().join("c.mov")).unwrap();
		for desired in 2..30 {
			streamer.request_layer_state(L1, ClipId(1), desired, 1.0, PlaybackMode::Loop, true);
			streamer.request_layer_state(L2, ClipId(2), desired, 1.0, PlaybackMode::Loop, true);
			std::thread::sleep(Duration::from_millis(1));
		}
		// Switching layer 1 to the deleted clip fails to open → error for L1 only.
		streamer.request_layer_state(L1, ClipId(3), 5, 1.0, PlaybackMode::Loop, true);
		let mut events = Vec::new();
		assert!(wait_until(|| {
			events.extend(streamer.drain_events());
			events
				.iter()
				.any(|event| matches!(event, LayerStreamEvent::Error { clip_id: ClipId(3), .. }))
		}));
		assert!(
			events
				.iter()
				.all(|event| matches!(event, LayerStreamEvent::Error { layer, .. } if *layer == L1))
		);
		assert!(streamer.poll_frame(L1, ClipId(3), 5).is_none());

		// Layer 2 keeps streaming.
		streamer.request_layer_state(L2, ClipId(2), 33, 1.0, PlaybackMode::Loop, true);
		assert!(wait_until(|| streamer.poll_frame(L2, ClipId(2), 33).is_some()));
		assert_eq!(streamer.worker_restarts(L1), 0);
		assert_eq!(streamer.worker_restarts(L2), 0);
	}

	#[test]
	fn truncated_file_mid_playback_reports_error_on_that_layer() {
		let dir = TempDir::new("stream_truncate").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		let (a, _) = hap_clip(dir.path(), "a.mov", 1, 40);
		let (b, _) = hap_clip(dir.path(), "b.mov", 2, 40);
		streamer.register_asset(a);
		streamer.register_asset(b);
		streamer.request_layer_state(L1, ClipId(1), 1, 1.0, PlaybackMode::Loop, true);
		streamer.request_layer_state(L2, ClipId(2), 1, 1.0, PlaybackMode::Loop, true);
		assert!(wait_until(|| streamer.poll_frame(L1, ClipId(1), 1).is_some()));

		std::fs::OpenOptions::new()
			.write(true)
			.open(dir.path().join("a.mov"))
			.unwrap()
			.set_len(64)
			.unwrap();
		streamer.request_layer_state(L1, ClipId(1), 25, 1.0, PlaybackMode::Loop, true);
		let mut events = Vec::new();
		assert!(wait_until(|| {
			events.extend(streamer.drain_events());
			!events.is_empty()
		}));
		match &events[0] {
			LayerStreamEvent::Error {
				layer,
				clip_id,
				message,
				..
			} => {
				assert_eq!(*layer, L1);
				assert_eq!(*clip_id, ClipId(1));
				assert!(message.contains("a.mov"), "{message}");
			}
			other => panic!("unexpected event {other:?}"),
		}
		// Layer 1 stays stopped for this generation; layer 2 is unaffected.
		assert!(streamer.poll_frame(L1, ClipId(1), 25).is_none());
		streamer.request_layer_state(L2, ClipId(2), 25, 1.0, PlaybackMode::Loop, true);
		assert!(wait_until(|| streamer.poll_frame(L2, ClipId(2), 25).is_some()));
		std::thread::sleep(Duration::from_millis(30));
		assert!(streamer.drain_events().iter().all(|event| !matches!(
			event,
			LayerStreamEvent::Error { layer, .. } if *layer == L2
		)));
	}

	fn corrupt_samples(asset: &ClipAsset, frames: impl IntoIterator<Item = usize>) {
		let ClipBacking::HapMov { index, path } = &asset.backing else {
			panic!("expected HAP backing");
		};
		let mut file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
		for frame in frames {
			let sample = index.samples[frame];
			file.seek(SeekFrom::Start(sample.offset)).unwrap();
			file.write_all(&vec![0xEE; sample.size as usize]).unwrap();
		}
	}

	#[test]
	fn fallback_never_skips_more_than_one_frame_ahead() {
		assert_eq!(fallback_rank(0), None, "the exact frame is not a fallback");
		assert_eq!(fallback_rank(-1), Some((1, false)));
		assert_eq!(fallback_rank(1), Some((1, true)));
		assert!(fallback_rank(-1) < fallback_rank(1), "behind wins at equal distance");
		assert!(fallback_rank(1) < fallback_rank(-2), "nearest wins");
		assert_eq!(
			fallback_rank(-(MAX_SERVE_BEHIND as i64)),
			Some((MAX_SERVE_BEHIND as i64, false))
		);
		assert_eq!(fallback_rank(-(MAX_SERVE_BEHIND as i64) - 1), None);
		for ahead in 2..=PREFETCH_AHEAD as i64 {
			assert_eq!(fallback_rank(ahead), None, "{ahead} frames ahead must not be served");
		}
	}

	#[test]
	fn isolated_corrupt_frame_is_skipped_without_stopping_the_stream() {
		let dir = TempDir::new("stream_corrupt").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		let (asset, fixture) = hap_clip(dir.path(), "a.mov", 1, 10);
		corrupt_samples(&asset, [4]);
		streamer.register_asset(asset);

		streamer.request_layer_state(L1, ClipId(1), 1, 1.0, PlaybackMode::Loop, true);
		// Frames on both sides of the corrupt one decode normally.
		assert!(wait_until(|| {
			let buffered = streamer.buffered_frames(L1);
			[1, 2, 3, 5, 6, 7, 8, 9].iter().all(|frame| buffered.contains(frame))
		}));
		assert!(!streamer.buffered_frames(L1).contains(&4));
		std::thread::sleep(Duration::from_millis(30));
		assert!(
			streamer.drain_events().is_empty(),
			"one corrupt frame must not stop the layer"
		);

		// Reaching the corrupt frame serves the nearest frame instead of freezing.
		streamer.request_layer_state(L1, ClipId(1), 4, 1.0, PlaybackMode::Loop, true);
		let fallback = streamer.poll_frame(L1, ClipId(1), 4).unwrap();
		assert_eq!(fallback.frame_index, 3, "prefers the frame just behind the playhead");
		let next = streamer.poll_frame(L1, ClipId(1), 5).unwrap();
		assert_eq!(payload_bytes(&next), fixture.frame_payload(5).unwrap().as_slice());
		assert!(streamer.drain_events().is_empty());
	}

	#[test]
	fn consecutive_corrupt_frames_stop_the_layer() {
		let dir = TempDir::new("stream_corrupt_run").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		let (asset, _) = hap_clip(dir.path(), "a.mov", 1, 20);
		corrupt_samples(&asset, 1..20);
		streamer.register_asset(asset);
		streamer.request_layer_state(L1, ClipId(1), 1, 1.0, PlaybackMode::Loop, true);
		let mut events = Vec::new();
		assert!(wait_until(|| {
			events.extend(streamer.drain_events());
			!events.is_empty()
		}));
		assert!(matches!(&events[0], LayerStreamEvent::Error { layer, clip_id: ClipId(1), .. } if *layer == L1));
		assert!(streamer.buffered_frames(L1).is_empty());

		// Retrying after a clear starts a fresh generation (and fails again cleanly).
		streamer.clear_layer(L1);
		assert!(streamer.request_layer_state(L1, ClipId(1), 2, 1.0, PlaybackMode::Loop, false));
		assert!(
			streamer.poll_frame(L1, ClipId(1), 0).is_some(),
			"resident frame 0 stays available"
		);
	}

	#[test]
	fn fatal_error_classification() {
		let io = MediaError::Io {
			path: "a.mov".into(),
			source: std::io::Error::from(std::io::ErrorKind::UnexpectedEof),
		};
		let corrupt = MediaError::Corrupt {
			path: "a.mov".into(),
			detail: "bad".into(),
		};
		let decode = MediaError::Decode {
			frame_index: 3,
			detail: "bad".into(),
		};
		assert!(is_fatal_stream_error(&io, 100, 1));
		assert!(is_fatal_stream_error(
			&MediaError::UnsupportedFormat("x".into()),
			100,
			1
		));
		assert!(!is_fatal_stream_error(&corrupt, 100, 1));
		assert!(!is_fatal_stream_error(&decode, 100, MAX_CONSECUTIVE_FRAME_ERRORS - 1));
		assert!(is_fatal_stream_error(&decode, 100, MAX_CONSECUTIVE_FRAME_ERRORS));
		assert!(is_fatal_stream_error(&decode, 1, 1), "single-frame clips fail at once");
	}

	#[test]
	fn unchanged_request_does_not_wake_the_worker() {
		let dir = TempDir::new("stream_dedupe").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		let (asset, _) = hap_clip(dir.path(), "a.mov", 1, 20);
		streamer.register_asset(asset);
		assert!(streamer.request_layer_state(L1, ClipId(1), 3, 1.0, PlaybackMode::Loop, true));
		let shared = streamer.layer(L1).unwrap();
		let seq = lock(&shared.control).seq;
		let wakes = shared.wakes.load(Ordering::Relaxed);
		let generation = streamer.layer_generation(L1).unwrap();
		for _ in 0..100 {
			assert!(streamer.request_layer_state(L1, ClipId(1), 3, 1.0, PlaybackMode::Loop, true));
		}
		assert_eq!(lock(&shared.control).seq, seq);
		assert_eq!(shared.wakes.load(Ordering::Relaxed), wakes);
		assert_eq!(streamer.layer_generation(L1).unwrap(), generation);

		// Any real change still reaches the worker without a generation bump.
		assert!(streamer.request_layer_state(L1, ClipId(1), 4, 1.0, PlaybackMode::Loop, true));
		assert_eq!(lock(&shared.control).seq, seq.wrapping_add(1));
		assert_eq!(shared.wakes.load(Ordering::Relaxed), wakes + 1);
		assert!(streamer.request_layer_state(L1, ClipId(1), 4, -1.0, PlaybackMode::Loop, true));
		assert!(streamer.request_layer_state(L1, ClipId(1), 4, -1.0, PlaybackMode::OneShot, true));
		assert!(streamer.request_layer_state(L1, ClipId(1), 4, -1.0, PlaybackMode::OneShot, false));
		assert_eq!(lock(&shared.control).seq, seq.wrapping_add(4));
		assert_eq!(streamer.layer_generation(L1).unwrap(), generation);
	}

	#[test]
	fn clear_layer_hides_buffered_frames_immediately() {
		let dir = TempDir::new("stream_clear").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		let (asset, _) = hap_clip(dir.path(), "a.mov", 1, 20);
		streamer.register_asset(asset);
		streamer.request_layer_state(L1, ClipId(1), 3, 1.0, PlaybackMode::Loop, true);
		assert!(wait_until(|| streamer.poll_frame(L1, ClipId(1), 3).is_some()));
		streamer.clear_layer(L1);
		// No waiting for the worker: the ring is emptied by `clear_layer` itself.
		assert!(streamer.poll_frame(L1, ClipId(1), 3).is_none());
		assert!(streamer.poll_frame(L1, ClipId(1), 4).is_none());
		assert!(streamer.buffered_frames(L1).is_empty());
	}

	#[test]
	fn switching_clips_never_serves_the_previous_clip() {
		let dir = TempDir::new("stream_switch").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		let (a, _) = hap_clip(dir.path(), "a.mov", 1, 20);
		let (b, _) = hap_clip(dir.path(), "b.mov", 2, 20);
		streamer.register_asset(a);
		streamer.register_asset(b);
		streamer.request_layer_state(L1, ClipId(1), 3, 1.0, PlaybackMode::Loop, true);
		assert!(wait_until(|| streamer.poll_frame(L1, ClipId(1), 3).is_some()));
		streamer.request_layer_state(L1, ClipId(2), 3, 1.0, PlaybackMode::Loop, true);
		assert!(streamer.poll_frame(L1, ClipId(1), 3).is_none());
	}

	#[test]
	fn fractional_fast_speed_prefetches_every_visited_frame() {
		let dir = TempDir::new("stream_fractional").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		let (asset, fixture) = hap_clip(dir.path(), "a.mov", 1, 120);
		streamer.register_asset(asset);
		// 30 fps clip at 2.74x on a 60 fps output: 1.37 clip frames per output frame.
		let speed = 2.74_f32;
		let step = f64::from(speed) * 30.0 / f64::from(RENDER_FPS);
		let playhead = |tick: u32| 1.0 + f64::from(tick) * step;
		for tick in 0..40 {
			let desired = playhead(tick).floor() as u32;
			let next = playhead(tick + 1).floor() as u32;
			assert!(streamer.request_layer_state(L1, ClipId(1), desired, speed, PlaybackMode::Loop, true));
			assert!(wait_until(|| streamer.poll_frame(L1, ClipId(1), desired).is_some()));
			let frame = streamer.poll_frame(L1, ClipId(1), desired).unwrap();
			assert_eq!(frame.frame_index, desired);
			assert_eq!(
				payload_bytes(&frame),
				fixture.frame_payload(desired).unwrap().as_slice()
			);
			// The next visited frame was prefetched before it is requested.
			assert!(
				wait_until(|| streamer.buffered_frames(L1).contains(&next)),
				"frame {next} after {desired} was not prefetched"
			);
		}
		assert!(streamer.drain_events().is_empty());
	}

	#[test]
	fn shutdown_wait_is_bounded() {
		let (events, _rx) = crossbeam_channel::bounded(1);
		let shared = LayerShared::new(L1, events);
		let started = Instant::now();
		assert!(!shared.wait_finished(started + Duration::from_millis(30)));
		assert!(started.elapsed() >= Duration::from_millis(25));
		shared.mark_finished();
		assert!(shared.wait_finished(Instant::now()));

		let streamer = LayerStreamer::new().unwrap();
		let dropped = Instant::now();
		drop(streamer);
		assert!(dropped.elapsed() < SHUTDOWN_TIMEOUT);
	}

	#[test]
	fn worker_panics_are_caught_and_restarted() {
		let dir = TempDir::new("stream_panic").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		let (a, _) = hap_clip(dir.path(), "a.mov", 1, 20);
		let (b, _) = hap_clip(dir.path(), "b.mov", 2, 20);
		streamer.register_asset(a);
		streamer.register_asset(b);
		streamer.request_layer_state(L2, ClipId(2), 1, 1.0, PlaybackMode::Loop, false);

		streamer.inject_panic(L1);
		streamer.request_layer_state(L1, ClipId(1), 3, 1.0, PlaybackMode::Loop, true);
		let mut events = Vec::new();
		assert!(wait_until(|| {
			events.extend(streamer.drain_events());
			events
				.iter()
				.any(|event| matches!(event, LayerStreamEvent::WorkerRestarted { restarts: 1, .. }))
		}));
		assert!(
			events.iter().any(
				|event| matches!(event, LayerStreamEvent::Error { layer, clip_id: ClipId(1), .. } if *layer == L1)
			)
		);
		assert_eq!(streamer.worker_restarts(L1), 1);
		assert_eq!(streamer.worker_restarts(L2), 0);
		assert!(wait_until(|| streamer.poll_frame(L2, ClipId(2), 1).is_some()));

		// The restarted worker serves a fresh generation normally.
		streamer.clear_layer(L1);
		streamer.request_layer_state(L1, ClipId(1), 3, 1.0, PlaybackMode::Loop, true);
		assert!(wait_until(|| streamer.poll_frame(L1, ClipId(1), 3).is_some()));
	}

	#[test]
	fn four_layers_stream_independently() {
		let dir = TempDir::new("stream_four").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		for (offset, layer) in LayerIndex::ALL.iter().enumerate() {
			let id = offset as u64 + 1;
			let (asset, _) = hap_clip(dir.path(), &format!("{id}.mov"), id, 24);
			streamer.register_asset(asset);
			let speed = if offset % 2 == 0 { 1.0 } else { -1.0 };
			streamer.request_layer_state(*layer, ClipId(id), 10, speed, PlaybackMode::Loop, true);
		}
		for (offset, layer) in LayerIndex::ALL.iter().enumerate() {
			let id = ClipId(offset as u64 + 1);
			let next = if offset % 2 == 0 { 11 } else { 9 };
			assert!(wait_until(|| streamer.poll_frame(*layer, id, next).is_some()));
		}
		assert!(streamer.drain_events().is_empty());
		drop(streamer);
	}
}
