//! Error types and non-blocking notification primitives.
//!
//! MUR never surfaces errors through modal dialogs while performing: failures are
//! returned as [`MurError`] values and reported to the operator via a bounded
//! [`ToastQueue`], so that rendering and MIDI handling keep running.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Maximum number of toast notifications retained at once.
pub const TOAST_CAPACITY: usize = 8;

/// Maximum number of log entries retained in memory.
pub const LOG_CAPACITY: usize = 200;

/// Aggregate error covering every MUR subsystem.
#[derive(Debug, thiserror::Error)]
pub enum MurError {
	/// A media file could not be loaded, validated, or decoded.
	#[error(transparent)]
	Media(#[from] MediaError),
	/// The MIDI subsystem failed to initialize or lost its port.
	#[error(transparent)]
	Midi(#[from] MidiError),
	/// Rendering, GPU device, or surface handling failed.
	#[error(transparent)]
	Render(#[from] RenderError),
	/// Configuration or persisted-state handling failed.
	#[error(transparent)]
	Config(#[from] ConfigError),
	/// A background worker terminated unexpectedly and needs a restart.
	#[error("worker thread `{0}` stopped unexpectedly")]
	WorkerStopped(String),
}

/// Errors raised while loading or decoding media files.
#[derive(Debug, thiserror::Error)]
pub enum MediaError {
	/// The container or codec is not supported by MUR.
	#[error("unsupported media format: {0}")]
	UnsupportedFormat(String),
	/// Reading a media file from disk failed.
	#[error("failed to read `{path}`: {source}")]
	Io {
		/// Path of the file that could not be read.
		path: String,
		/// Underlying I/O failure.
		#[source]
		source: std::io::Error,
	},
	/// The container or a frame is malformed or truncated.
	#[error("corrupt media data in `{path}`: {detail}")]
	Corrupt {
		/// Path of the offending file.
		path: String,
		/// Human readable description of the corruption.
		detail: String,
	},
	/// A single frame could not be decoded.
	#[error("failed to decode frame {frame_index}: {detail}")]
	Decode {
		/// Zero-based index of the failing frame.
		frame_index: u32,
		/// Decoder error detail.
		detail: String,
	},
	/// A decode worker stopped and must be restarted.
	#[error("decode worker stopped: {0}")]
	WorkerStopped(String),
}

/// Errors raised by the MIDI input subsystem.
#[derive(Debug, thiserror::Error)]
pub enum MidiError {
	/// The MIDI backend could not be initialized.
	#[error("failed to initialize MIDI input: {0}")]
	Init(String),
	/// A MIDI port could not be opened.
	#[error("failed to open MIDI port `{port}`: {detail}")]
	OpenPort {
		/// Name of the port that failed to open.
		port: String,
		/// Backend error detail.
		detail: String,
	},
	/// The active MIDI port was disconnected and must be reacquired.
	#[error("MIDI port `{0}` disconnected")]
	Disconnected(String),
	/// An incoming MIDI message could not be decoded.
	#[error("invalid MIDI message: {0}")]
	InvalidMessage(String),
}

/// Errors raised while rendering or while managing GPU resources.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
	/// The GPU adapter or device could not be created.
	#[error("failed to create GPU device: {0}")]
	DeviceCreation(String),
	/// The GPU device was lost and resources must be recreated.
	#[error("GPU device lost: {0}")]
	DeviceLost(String),
	/// The output surface was lost, for example after a monitor change.
	#[error("output surface lost: {0}")]
	SurfaceLost(String),
	/// A shader or pipeline failed to build.
	#[error("failed to build render pipeline `{name}`: {detail}")]
	PipelineCreation {
		/// Name of the pipeline being built.
		name: String,
		/// Backend error detail.
		detail: String,
	},
	/// The render target geometry is invalid.
	#[error("invalid render target: {0}")]
	InvalidTarget(String),
}

/// Errors raised while reading or writing configuration and state files.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
	/// A configuration or state file could not be read or written.
	#[error("failed to access `{path}`: {source}")]
	Io {
		/// Path involved in the failure.
		path: String,
		/// Underlying I/O failure.
		#[source]
		source: std::io::Error,
	},
	/// A configuration file could not be parsed.
	#[error("failed to parse {format} configuration `{path}`: {detail}")]
	Parse {
		/// Path of the offending file.
		path: String,
		/// Human readable format name, such as `TOML`.
		format: &'static str,
		/// Parser error detail.
		detail: String,
	},
	/// The configuration file extension is not supported.
	#[error("unsupported configuration file format: `{path}`")]
	UnsupportedFormat {
		/// Path whose extension is unsupported.
		path: String,
	},
	/// Serializing the in-memory state failed.
	#[error("failed to serialize state for `{path}`: {detail}")]
	Serialize {
		/// Destination path of the failed write.
		path: String,
		/// Serializer error detail.
		detail: String,
	},
}

impl ConfigError {
	/// Builds an [`ConfigError::Io`] from a path and its I/O failure.
	pub(crate) fn io(path: &std::path::Path, source: std::io::Error) -> Self {
		Self::Io {
			path: path.display().to_string(),
			source,
		}
	}

	/// Builds a [`ConfigError::Parse`] for the given file and format.
	pub(crate) fn parse(path: &std::path::Path, format: &'static str, detail: impl Into<String>) -> Self {
		Self::Parse {
			path: path.display().to_string(),
			format,
			detail: detail.into(),
		}
	}
}

/// Severity of a toast notification or log entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ToastLevel {
	/// Purely informational message.
	#[default]
	Info,
	/// Recoverable problem the operator should notice.
	Warning,
	/// Failure that degraded or silenced part of the output.
	Error,
}

impl ToastLevel {
	/// Short uppercase label suitable for compact UI badges.
	pub fn label(self) -> &'static str {
		match self {
			Self::Info => "INFO",
			Self::Warning => "WARN",
			Self::Error => "ERROR",
		}
	}

	/// Default time-to-live in seconds for toasts of this level.
	pub fn default_ttl_secs(self) -> f64 {
		match self {
			Self::Info => 3.5,
			Self::Warning => 6.0,
			Self::Error => 10.0,
		}
	}
}

/// A transient notification rendered by the control UI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToastNotification {
	/// Monotonic identifier, unique within a session.
	pub id: u64,
	/// Severity of the notification.
	pub level: ToastLevel,
	/// Short headline.
	pub title: String,
	/// Optional detail body.
	pub message: String,
	/// Creation time on the application clock, in seconds.
	pub created_at_secs: f64,
	/// Lifetime in seconds after which the toast expires.
	pub ttl_secs: f64,
}

impl ToastNotification {
	/// Creates a notification with explicit timing values.
	pub fn new(
		id: u64,
		level: ToastLevel,
		title: impl Into<String>,
		message: impl Into<String>,
		created_at_secs: f64,
		ttl_secs: f64,
	) -> Self {
		Self {
			id,
			level,
			title: title.into(),
			message: message.into(),
			created_at_secs,
			ttl_secs,
		}
	}

	/// Returns `true` once `now_secs` reached the expiry time of this toast.
	pub fn is_expired(&self, now_secs: f64) -> bool {
		if !now_secs.is_finite() || !self.ttl_secs.is_finite() {
			return false;
		}
		now_secs >= self.created_at_secs + self.ttl_secs
	}
}

/// A single line of the in-memory runtime log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogEntry {
	/// Wall-clock timestamp rendered by the logger.
	pub timestamp: String,
	/// Severity of the entry.
	pub level: ToastLevel,
	/// Subsystem that emitted the entry, such as `decoder`.
	pub source: String,
	/// Log message body.
	pub message: String,
}

impl LogEntry {
	/// Creates a log entry from already formatted field values.
	pub fn new(
		timestamp: impl Into<String>,
		level: ToastLevel,
		source: impl Into<String>,
		message: impl Into<String>,
	) -> Self {
		Self {
			timestamp: timestamp.into(),
			level,
			source: source.into(),
			message: message.into(),
		}
	}
}

/// Bounded, allocation-once queue of toasts plus a rolling log history.
///
/// Every operation is `O(1)` amortized for pushes and never blocks, so it is safe
/// to call from the render loop and from worker threads reporting failures.
#[derive(Debug, Clone)]
pub struct ToastQueue {
	toasts: VecDeque<ToastNotification>,
	logs: VecDeque<LogEntry>,
	next_id: u64,
}

impl ToastQueue {
	/// Creates an empty queue.
	pub fn new() -> Self {
		Self {
			toasts: VecDeque::with_capacity(TOAST_CAPACITY),
			logs: VecDeque::with_capacity(LOG_CAPACITY),
			next_id: 1,
		}
	}

	/// Pushes a prepared toast, evicting the oldest one when the cap is reached.
	pub fn push(&mut self, toast: ToastNotification) {
		if self.toasts.len() >= TOAST_CAPACITY {
			self.toasts.pop_front();
		}
		self.toasts.push_back(toast);
	}

	/// Builds, enqueues, and returns a toast using the level's default lifetime.
	pub fn notify(
		&mut self,
		level: ToastLevel,
		title: impl Into<String>,
		message: impl Into<String>,
		now_secs: f64,
	) -> ToastNotification {
		let id = self.next_id;
		self.next_id = self.next_id.saturating_add(1);
		let toast = ToastNotification::new(id, level, title, message, now_secs, level.default_ttl_secs());
		self.push(toast.clone());
		toast
	}

	/// Appends a log line, dropping the oldest entries beyond [`LOG_CAPACITY`].
	pub fn push_log(&mut self, entry: LogEntry) {
		if self.logs.len() >= LOG_CAPACITY {
			self.logs.pop_front();
		}
		self.logs.push_back(entry);
	}

	/// Removes toasts whose lifetime elapsed at `now_secs`; returns the count removed.
	pub fn prune_expired(&mut self, now_secs: f64) -> usize {
		if !now_secs.is_finite() {
			return 0;
		}
		let before = self.toasts.len();
		self.toasts.retain(|toast| !toast.is_expired(now_secs));
		before.saturating_sub(self.toasts.len())
	}

	/// Removes every active toast while keeping the log history intact.
	pub fn clear_toasts(&mut self) {
		self.toasts.clear();
	}

	/// Active toasts in insertion order, oldest first.
	pub fn toasts(&self) -> &VecDeque<ToastNotification> {
		&self.toasts
	}

	/// Retained log entries, oldest first.
	pub fn logs(&self) -> &VecDeque<LogEntry> {
		&self.logs
	}

	/// Identifier that the next [`ToastQueue::notify`] call will use.
	pub fn next_id(&self) -> u64 {
		self.next_id
	}
}

impl Default for ToastQueue {
	fn default() -> Self {
		Self::new()
	}
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;

	fn toast(id: u64, level: ToastLevel) -> ToastNotification {
		ToastNotification::new(id, level, "title", "message", 10.0, 2.0)
	}

	#[test]
	fn notify_assigns_increasing_ids_and_default_ttl() {
		let mut queue = ToastQueue::new();
		let first = queue.notify(ToastLevel::Info, "loaded", "clip.mov", 0.0);
		let second = queue.notify(ToastLevel::Error, "failed", "clip.mov", 0.5);
		assert_eq!(first.id, 1);
		assert_eq!(second.id, 2);
		assert_eq!(queue.next_id(), 3);
		assert_eq!(first.ttl_secs, ToastLevel::Info.default_ttl_secs());
		assert_eq!(queue.toasts().len(), 2);
	}

	#[test]
	fn push_is_bounded_and_evicts_oldest() {
		let mut queue = ToastQueue::new();
		for id in 0..(TOAST_CAPACITY as u64 + 4) {
			queue.push(toast(id, ToastLevel::Info));
		}
		assert_eq!(queue.toasts().len(), TOAST_CAPACITY);
		assert_eq!(queue.toasts().front().map(|t| t.id), Some(4));
		assert_eq!(queue.toasts().back().map(|t| t.id), Some(11));
	}

	#[test]
	fn log_history_is_bounded() {
		let mut queue = ToastQueue::new();
		for index in 0..(LOG_CAPACITY + 5) {
			queue.push_log(LogEntry::new(index.to_string(), ToastLevel::Warning, "test", "entry"));
		}
		assert_eq!(queue.logs().len(), LOG_CAPACITY);
		assert_eq!(queue.logs().front().map(|entry| entry.timestamp.as_str()), Some("5"));
	}

	#[test]
	fn prune_removes_only_expired_toasts() {
		let mut queue = ToastQueue::new();
		queue.push(toast(1, ToastLevel::Info));
		queue.push(ToastNotification::new(2, ToastLevel::Info, "t", "m", 10.0, 100.0));

		assert_eq!(queue.prune_expired(11.9), 0);
		assert_eq!(queue.prune_expired(12.0), 1);
		assert_eq!(queue.toasts().len(), 1);
		assert_eq!(queue.toasts().front().map(|t| t.id), Some(2));
		assert_eq!(queue.prune_expired(f64::NAN), 0);
	}

	#[test]
	fn clear_toasts_keeps_logs() {
		let mut queue = ToastQueue::new();
		queue.notify(ToastLevel::Info, "t", "m", 0.0);
		queue.push_log(LogEntry::new("now", ToastLevel::Info, "src", "msg"));
		queue.clear_toasts();
		assert!(queue.toasts().is_empty());
		assert_eq!(queue.logs().len(), 1);
	}
}
