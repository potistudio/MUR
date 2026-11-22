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
//! frames in the playback direction into a bounded ring buffer, and discard
//! stale work via a per-layer `generation` counter. I/O, deletion, and decode
//! errors only stop the affected layer and are reported as
//! [`LayerStreamEvent::Error`]. A panicking worker is caught with
//! [`std::panic::catch_unwind`], reported, and restarted.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};
use std::thread::JoinHandle;
use std::time::Duration;

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

/// Computes the frames a layer should hold, nearest first.
///
/// The first entry is always `desired` (clamped to the clip). While playing,
/// up to `ahead` further frames follow in the playback direction, spaced by the
/// number of clip frames advanced per output frame at [`RENDER_FPS`]. `Loop`
/// wraps around both ends; `OneShot` stops at the clip boundaries.
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
	let step = f64::from((speed.abs() * fps / RENDER_FPS).max(1.0));
	let direction: i64 = if speed < 0.0 { -1 } else { 1 };
	let count = i64::from(frame_count);
	for k in 1..=ahead {
		let offset = ((k as f64) * step).round() as i64;
		let raw = i64::from(desired).saturating_add(offset.saturating_mul(direction));
		let frame = match mode {
			PlaybackMode::Loop => raw.rem_euclid(count),
			PlaybackMode::OneShot => {
				if raw < 0 || raw >= count {
					break;
				}
				raw
			}
		};
		let Ok(frame) = u32::try_from(frame) else {
			break;
		};
		if !window.contains(&frame) {
			window.push(frame);
		}
	}
	window
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
	#[cfg(test)]
	inject_panic: std::sync::atomic::AtomicBool,
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
			#[cfg(test)]
			inject_panic: std::sync::atomic::AtomicBool::new(false),
		}
	}

	/// Wakes the worker without ever blocking; a pending token is enough.
	fn wake(&self) {
		let _ = self.wake_tx.try_send(());
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

	/// Clears the bound clip and bumps the generation.
	fn clear(&self) {
		{
			let mut control = lock(&self.control);
			if control.target.is_some() {
				control.generation = control.generation.wrapping_add(1);
			}
			control.target = None;
			control.seq = control.seq.wrapping_add(1);
		}
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
		let updated_in_place = {
			let mut guard = lock(&shared.control);
			let control = &mut *guard;
			match control.target.as_mut() {
				Some(target) if target.clip_id == clip_id => {
					target.desired_frame = desired_frame.min(target.frame_count.saturating_sub(1));
					target.speed = if speed.is_finite() { speed } else { 0.0 };
					target.mode = mode;
					target.is_playing = is_playing;
					control.seq = control.seq.wrapping_add(1);
					true
				}
				_ => false,
			}
		};
		if updated_in_place {
			shared.wake();
			return true;
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
		{
			let mut control = lock(&shared.control);
			control.generation = control.generation.wrapping_add(1);
			control.failed_generation = None;
			control.target = Some(StreamTarget {
				clip_id,
				desired_frame: desired_frame.min(frame_count.saturating_sub(1)),
				speed: if speed.is_finite() { speed } else { 0.0 },
				mode,
				is_playing,
				frame_count,
				fps: asset.info.fps,
				source,
			});
			control.seq = control.seq.wrapping_add(1);
		}
		shared.wake();
		true
	}

	/// Unbinds the clip from `layer`, dropping its buffered frames.
	pub fn clear_layer(&self, layer: LayerIndex) {
		if let Some(shared) = self.layer(layer) {
			shared.clear();
		}
	}

	/// Returns the decoded frame `desired_frame` of `clip_id` if it is ready.
	///
	/// Never waits for decoding. Frame `0` (and every frame of a still image)
	/// is served from the resident first frame. `None` means "keep showing the
	/// previous frame".
	pub fn poll_frame(&self, layer: LayerIndex, clip_id: ClipId, desired_frame: u32) -> Option<Arc<DecodedFrame>> {
		{
			let assets = self.assets.read().unwrap_or_else(PoisonError::into_inner);
			let asset = assets.get(&clip_id)?;
			if desired_frame == 0 || matches!(asset.backing, ClipBacking::StaticImage) {
				return Some(Arc::clone(&asset.first_frame));
			}
		}
		let shared = self.layer(layer)?;
		let ring = lock(&shared.ring);
		ring.entries
			.iter()
			.find(|(id, frame)| *id == clip_id && frame.frame_index == desired_frame)
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
	fn drop(&mut self) {
		for shared in &self.layers {
			lock(&shared.control).shutdown = true;
			shared.wake();
		}
		let handles = std::mem::take(&mut *lock(&self.handles));
		for handle in handles {
			let _ = handle.join();
		}
	}
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// Restarts `worker_main` after panics until shutdown.
fn supervise(shared: &LayerShared) {
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
	let mut scratch: Vec<u8> = Vec::new();

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
			{
				let mut ring = lock(&shared.ring);
				ring.generation = snapshot.generation;
				ring.entries.clear();
			}
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
			window = match snapshot.target {
				Some(target) if open.is_some() => prefetch_window(
					target.desired_frame,
					target.frame_count,
					target.speed,
					target.fps,
					target.mode,
					target.is_playing,
					PREFETCH_AHEAD,
				),
				_ => Vec::new(),
			};
			let mut ring = lock(&shared.ring);
			ring.entries.retain(|(_, frame)| window.contains(&frame.frame_index));
		}

		let next = match &open {
			Some(clip) => {
				let ring = lock(&shared.ring);
				window
					.iter()
					.copied()
					.find(|&frame| frame != 0 && !ring.contains(clip.clip_id, frame))
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
		match decode_frame(clip, frame_index, &mut scratch) {
			Ok(frame) => {
				let mut ring = lock(&shared.ring);
				if ring.generation == current_generation {
					while ring.entries.len() >= RING_CAPACITY {
						ring.entries.pop_front();
					}
					ring.entries.push_back((clip.clip_id, Arc::new(frame)));
				}
			}
			Err(error) => {
				let message = format!("{}: {error}", clip.path.display());
				let clip_id = clip.clip_id;
				open = None;
				shared.report_error(clip_id, current_generation, message);
			}
		}
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

	#[test]
	fn corrupt_frame_reports_decode_error() {
		let dir = TempDir::new("stream_corrupt").unwrap();
		let streamer = LayerStreamer::new().unwrap();
		let (asset, _) = hap_clip(dir.path(), "a.mov", 1, 10);
		let ClipBacking::HapMov { index, path } = &asset.backing else {
			panic!("expected HAP backing");
		};
		let sample = index.samples[4];
		let mut file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
		file.seek(SeekFrom::Start(sample.offset)).unwrap();
		file.write_all(&vec![0xEE; sample.size as usize]).unwrap();
		drop(file);
		streamer.register_asset(asset);

		streamer.request_layer_state(L1, ClipId(1), 1, 1.0, PlaybackMode::Loop, true);
		let mut events = Vec::new();
		assert!(wait_until(|| {
			events.extend(streamer.drain_events());
			!events.is_empty()
		}));
		assert!(matches!(&events[0], LayerStreamEvent::Error { layer, .. } if *layer == L1));
		// Frames decoded before the corrupt one remain available.
		assert!(streamer.poll_frame(L1, ClipId(1), 1).is_some());
		assert!(streamer.poll_frame(L1, ClipId(1), 4).is_none());

		// Retrying after a clear starts a fresh generation.
		streamer.clear_layer(L1);
		streamer.request_layer_state(L1, ClipId(1), 2, 1.0, PlaybackMode::Loop, false);
		assert!(wait_until(|| streamer.poll_frame(L1, ClipId(1), 2).is_some()));
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
