//! Small helpers for background threads.

use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Polling interval of [`join_with_timeout`].
const JOIN_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Joins `handle` if the thread finishes within `timeout`.
///
/// Returns `true` when the thread finished and was joined. Otherwise the handle
/// is dropped, which detaches the thread, and `false` is returned: shutdown must
/// never hang on a thread stuck in a blocking OS call (a wedged USB MIDI
/// backend, an `fsync` on a dying disk).
pub fn join_with_timeout(handle: JoinHandle<()>, timeout: Duration) -> bool {
	let deadline = Instant::now() + timeout;
	while !handle.is_finished() {
		let remaining = deadline.saturating_duration_since(Instant::now());
		if remaining.is_zero() {
			return false;
		}
		std::thread::sleep(remaining.min(JOIN_POLL_INTERVAL));
	}
	let _ = handle.join();
	true
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;

	#[test]
	fn joins_finished_threads_and_detaches_stuck_ones() {
		let quick = std::thread::spawn(|| {});
		assert!(join_with_timeout(quick, Duration::from_secs(2)));

		let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
		let stuck = std::thread::spawn(move || {
			let _ = release_rx.recv();
		});
		let started = Instant::now();
		assert!(!join_with_timeout(stuck, Duration::from_millis(50)));
		assert!(started.elapsed() < Duration::from_secs(1));
		drop(release_tx);
	}
}
