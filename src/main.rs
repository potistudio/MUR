//! MUR entry point: CLI parsing, rolling log files, the panic hook, and the app runtime.

use std::process::ExitCode;

use clap::Parser as _;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;
use tracing_subscriber::{EnvFilter, Layer as _};

use mur::config::{AppPaths, CliArgs};

/// Log files kept on disk before the oldest daily file is deleted.
const MAX_LOG_FILES: usize = 14;

fn main() -> ExitCode {
	let args = CliArgs::parse();
	let paths = AppPaths::discover();
	let _log_guard = init_logging(&paths);
	install_panic_hook();

	tracing::info!(version = env!("CARGO_PKG_VERSION"), "starting MUR");
	match mur::app::run(args) {
		Ok(()) => {
			tracing::info!("MUR exited cleanly");
			ExitCode::SUCCESS
		}
		Err(error) => {
			tracing::error!("MUR stopped with an error: {error}");
			eprintln!("mur: {error}");
			ExitCode::FAILURE
		}
	}
}

/// Logs to stderr and, when the log directory is writable, to a daily rolling file.
///
/// The returned guard flushes the non-blocking file writer and must live until `main`
/// returns. Logging setup never aborts startup: any failure falls back to stderr only.
fn init_logging(paths: &AppPaths) -> Option<WorkerGuard> {
	let filter = || EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
	let stderr_layer = tracing_subscriber::fmt::layer()
		.with_writer(std::io::stderr)
		.with_filter(filter());

	let file_writer = match open_log_file(paths) {
		Ok(appender) => Some(tracing_appender::non_blocking(appender)),
		Err(message) => {
			eprintln!("mur: file logging disabled: {message}");
			None
		}
	};

	match file_writer {
		Some((writer, guard)) => {
			let file_layer = tracing_subscriber::fmt::layer()
				.with_ansi(false)
				.with_writer(writer)
				.with_filter(filter());
			let installed = tracing_subscriber::registry()
				.with(stderr_layer)
				.with(file_layer)
				.try_init();
			if let Err(error) = installed {
				eprintln!("mur: failed to install logger: {error}");
			}
			tracing::info!(dir = %paths.log_dir.display(), "logging to rolling file");
			Some(guard)
		}
		None => {
			if let Err(error) = tracing_subscriber::registry().with(stderr_layer).try_init() {
				eprintln!("mur: failed to install logger: {error}");
			}
			None
		}
	}
}

/// Creates the log directory and a daily-rotating `mur.log.YYYY-MM-DD` appender.
///
/// Equivalent to `tracing_appender::rolling::daily(dir, "mur.log")`, but uses the fallible
/// builder because `rolling::daily` panics when the directory is not writable.
fn open_log_file(paths: &AppPaths) -> Result<RollingFileAppender, String> {
	std::fs::create_dir_all(&paths.log_dir)
		.map_err(|error| format!("cannot create `{}`: {error}", paths.log_dir.display()))?;
	RollingFileAppender::builder()
		.rotation(Rotation::DAILY)
		.filename_prefix("mur.log")
		.max_log_files(MAX_LOG_FILES)
		.build(&paths.log_dir)
		.map_err(|error| format!("cannot open log file in `{}`: {error}", paths.log_dir.display()))
}

/// Records every panic with its location and backtrace in the log file and on stderr.
///
/// Worker panics are caught and recovered by their supervisors; the hook makes sure each
/// one still leaves a trace for the post-show review.
fn install_panic_hook() {
	std::panic::set_hook(Box::new(|info| {
		let message = info.payload_as_str().unwrap_or("non-string panic payload");
		let location = info
			.location()
			.map(|location| format!("{}:{}:{}", location.file(), location.line(), location.column()))
			.unwrap_or_else(|| "unknown location".to_owned());
		let thread = std::thread::current();
		let thread_name = thread.name().unwrap_or("unnamed");
		let backtrace = std::backtrace::Backtrace::force_capture();
		tracing::error!(
			target: "panic",
			thread = thread_name,
			%location,
			"panic: {message}\n{backtrace}"
		);
		eprintln!("mur: thread `{thread_name}` panicked at {location}: {message}\n{backtrace}");
	}));
}
