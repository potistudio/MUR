//! Command line arguments, project files, and persisted session state.
//!
//! Media paths from a project file are resolved relative to the project file's
//! parent directory, while the persisted state stores clip *paths* rather than
//! transient [`ClipId`] values so a restart restores exactly the same clips.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;
use crate::model::app_state::{AppState, OutputDisplaySelection};
use crate::model::blend::BlendMode;
use crate::model::clip::ClipId;
use crate::model::effect::MasterEffects;
use crate::model::layer::{LayerIndex, LayerState, PlaybackMode};

/// Version stamped into saved state files.
pub const STATE_VERSION: u32 = 1;

/// Default quiet period before a dirty session is written to disk.
pub const DEFAULT_AUTOSAVE_DEBOUNCE: Duration = Duration::from_millis(750);

/// Command line arguments accepted by the `mur` binary.
#[derive(Debug, Clone, clap::Parser)]
#[command(name = "mur", version, about = "MUR - live performance VJ software")]
pub struct CliArgs {
	/// Folder scanned for clips at startup.
	#[arg(long, value_name = "PATH")]
	pub folder: Option<PathBuf>,
	/// Project configuration file in TOML or JSON.
	#[arg(long, value_name = "PATH")]
	pub project: Option<PathBuf>,
	/// Session state file to use instead of the discovered default.
	#[arg(long, value_name = "PATH")]
	pub state: Option<PathBuf>,
	/// Start empty instead of restoring the previous session.
	#[arg(long)]
	pub no_restore: bool,
}

/// Well-known directories used by MUR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppPaths {
	/// Directory holding user-editable configuration.
	pub config_dir: PathBuf,
	/// Directory holding generated state, logs, and caches.
	pub data_dir: PathBuf,
	/// Default location of the auto-saved session state.
	pub state_file: PathBuf,
	/// Directory for rolling log files.
	pub log_dir: PathBuf,
}

impl AppPaths {
	/// Discovers OS-specific directories.
	///
	/// Falls back to a temporary directory when the platform reports no home
	/// directory, so startup never fails because of a missing profile.
	pub fn discover() -> Self {
		match directories::ProjectDirs::from("io", "potistudio", "MUR") {
			Some(dirs) => Self::from_dirs(dirs.config_dir().to_path_buf(), dirs.data_dir().to_path_buf()),
			None => {
				let base = std::env::temp_dir().join("mur");
				Self::from_dirs(base.clone(), base)
			}
		}
	}

	/// Builds the path set from explicit config and data directories.
	pub fn from_dirs(config_dir: PathBuf, data_dir: PathBuf) -> Self {
		let state_file = data_dir.join("state.json");
		let log_dir = data_dir.join("logs");
		Self {
			config_dir,
			data_dir,
			state_file,
			log_dir,
		}
	}

	/// Returns `override_path` when the operator supplied one, else the default.
	pub fn resolve_state_file(&self, override_path: Option<&Path>) -> PathBuf {
		override_path.map_or_else(|| self.state_file.clone(), Path::to_path_buf)
	}
}

/// Project configuration loaded from a TOML or JSON file.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProjectConfig {
	/// Folder scanned for clips when [`ProjectConfig::clips`] is empty.
	pub clip_folder: Option<PathBuf>,
	/// Explicit clip files, in grid order.
	pub clips: Vec<PathBuf>,
	/// Requested output display; `None` keeps the current selection.
	pub output: Option<OutputDisplaySelection>,
	/// Master effect values applied on load.
	pub effects: Option<MasterEffects>,
	/// Initial layer settings, at most four entries are used.
	pub layers: Vec<PersistedLayerState>,
	/// MIDI input port to open on startup.
	pub preferred_midi_port: Option<String>,
}

impl ProjectConfig {
	/// Loads a project file, resolving relative media paths against its directory.
	pub fn load(path: &Path) -> Result<Self, ConfigError> {
		let contents = std::fs::read_to_string(path).map_err(|error| ConfigError::io(path, error))?;
		let mut config = Self::parse(&contents, path)?;
		let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
		config.resolve_paths(base_dir);
		Ok(config)
	}

	/// Parses project file contents according to the path's extension.
	///
	/// Master effect values are clamped with [`MasterEffects::clamp_all`], so a
	/// project file loaded through this function always yields exactly the same
	/// values as [`ProjectConfig::load`].
	pub fn parse(contents: &str, path: &Path) -> Result<Self, ConfigError> {
		let extension = path
			.extension()
			.and_then(|ext| ext.to_str())
			.map(str::to_ascii_lowercase);
		let mut config: Self = match extension.as_deref() {
			Some("toml") => {
				toml::from_str(contents).map_err(|error| ConfigError::parse(path, "TOML", error.to_string()))?
			}
			Some("json") => {
				serde_json::from_str(contents).map_err(|error| ConfigError::parse(path, "JSON", error.to_string()))?
			}
			_ => {
				return Err(ConfigError::UnsupportedFormat {
					path: path.display().to_string(),
				});
			}
		};
		if let Some(effects) = config.effects.as_mut() {
			effects.clamp_all();
		}
		Ok(config)
	}

	/// Rewrites every relative media path to be absolute against `base_dir`.
	pub fn resolve_paths(&mut self, base_dir: &Path) {
		if let Some(folder) = self.clip_folder.as_mut() {
			*folder = resolve_relative(folder, base_dir);
		}
		self.clips = self.clips.iter().map(|clip| resolve_relative(clip, base_dir)).collect();
		for layer in &mut self.layers {
			layer.resolve_paths(base_dir);
		}
	}

	/// Explicit clip files, or `None` when the loader should scan the folder.
	pub fn explicit_clips(&self) -> Option<&[PathBuf]> {
		if self.clips.is_empty() { None } else { Some(&self.clips) }
	}
}

/// Restart-persistent settings of a single layer.
///
/// The active clip is identified by its file path because [`ClipId`] values are
/// reassigned on every launch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PersistedLayerState {
	/// Path of the clip that was active (triggered) on this layer.
	pub clip_path: Option<PathBuf>,
	/// Path of the clip that was selected on this layer.
	///
	/// Kept apart from [`Self::clip_path`] so a clip that was only selected is
	/// never restored as the active clip: an untriggered layer must stay empty
	/// after a restart, and a stray pause key must not start it. `None` falls
	/// back to the active clip.
	pub selected_clip_path: Option<PathBuf>,
	/// Loop or one-shot behaviour.
	pub playback_mode: PlaybackMode,
	/// Playback speed in `-2.0..=4.0`.
	pub speed: f32,
	/// Layer opacity in `0.0..=1.0`.
	pub opacity: f32,
	/// Blend mode of the layer.
	pub blend_mode: BlendMode,
	/// Playhead position in frames; restored for convenience only.
	pub playhead_frame: f64,
}

impl Default for PersistedLayerState {
	fn default() -> Self {
		Self {
			clip_path: None,
			selected_clip_path: None,
			playback_mode: PlaybackMode::Loop,
			speed: 1.0,
			opacity: 1.0,
			blend_mode: BlendMode::Normal,
			playhead_frame: 0.0,
		}
	}
}

impl PersistedLayerState {
	/// Captures a live layer, storing clip paths instead of runtime ids.
	///
	/// `clip_path` is the active clip and `selected_clip_path` the selected one.
	pub fn capture(layer: &LayerState, clip_path: Option<PathBuf>, selected_clip_path: Option<PathBuf>) -> Self {
		Self {
			clip_path,
			selected_clip_path,
			playback_mode: layer.playback_mode,
			speed: layer.speed,
			opacity: layer.opacity,
			blend_mode: layer.blend_mode,
			playhead_frame: layer.playhead_frame,
		}
	}

	/// Rebuilds a layer, binding `clip` (the active clip) when the persisted path
	/// could be resolved.
	///
	/// `selected` is the resolved [`Self::selected_clip_path`]; when it is `None`
	/// the selection falls back to the active clip. A selection-only clip is
	/// never made active.
	///
	/// The persisted playhead is restored so the operator returns to the frame
	/// they left, but the layer stays [`PlaybackState::Stopped`](crate::model::PlaybackState::Stopped): live output must
	/// never start moving on its own after a crash recovery. Invalid playhead
	/// values (negative or non-finite) fall back to frame `0`.
	pub fn restore(&self, clip: Option<ClipId>, selected: Option<ClipId>) -> LayerState {
		let mut layer = LayerState {
			selected_clip: selected.or(clip),
			active_clip: clip,
			playback_mode: self.playback_mode,
			blend_mode: self.blend_mode,
			..LayerState::default()
		};
		layer.set_speed(self.speed);
		layer.set_opacity(self.opacity);
		layer.playhead_frame = if self.playhead_frame.is_finite() && self.playhead_frame >= 0.0 {
			self.playhead_frame
		} else {
			0.0
		};
		layer
	}

	/// Resolves the persisted clip path against `base_dir`.
	pub fn resolve_paths(&mut self, base_dir: &Path) {
		if let Some(path) = self.clip_path.as_mut() {
			*path = resolve_relative(path, base_dir);
		}
		if let Some(path) = self.selected_clip_path.as_mut() {
			*path = resolve_relative(path, base_dir);
		}
	}
}

/// Auto-saved snapshot of the previous session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PersistedState {
	/// Schema version of this file.
	pub version: u32,
	/// Every clip that was loaded, in grid order.
	pub clip_paths: Vec<PathBuf>,
	/// Per-layer settings, always four entries.
	///
	/// Deserialization is tolerant: files written by older or newer builds with
	/// fewer or more than four entries are padded with defaults and truncated, so
	/// a stale `state.json` never blocks startup.
	#[serde(
		default = "default_persisted_layers",
		deserialize_with = "deserialize_persisted_layers"
	)]
	pub layers: [PersistedLayerState; 4],
	/// Master effect values.
	pub effects: MasterEffects,
	/// One-based number of the focused layer.
	pub focused_layer: u8,
	/// Requested output display.
	pub output: OutputDisplaySelection,
	/// MIDI port the operator selected.
	pub selected_midi_port: Option<String>,
}

impl Default for PersistedState {
	fn default() -> Self {
		Self {
			version: STATE_VERSION,
			clip_paths: Vec::new(),
			layers: default_persisted_layers(),
			effects: MasterEffects::default(),
			focused_layer: 1,
			output: OutputDisplaySelection::default(),
			selected_midi_port: None,
		}
	}
}

impl PersistedState {
	/// Captures the restart-persistent parts of `app`.
	pub fn capture(app: &AppState) -> Self {
		let layers = std::array::from_fn(|index| {
			app.layers
				.get(index)
				.map_or_else(PersistedLayerState::default, |layer| {
					let path_of =
						|id: Option<ClipId>| id.and_then(|id| app.clip_by_id(id)).map(|clip| clip.path.clone());
					PersistedLayerState::capture(layer, path_of(layer.active_clip), path_of(layer.selected_clip))
				})
		});

		Self {
			version: STATE_VERSION,
			clip_paths: app.clips.iter().map(|clip| clip.path.clone()).collect(),
			layers,
			effects: app.effects.clone(),
			focused_layer: app.focused_layer.number(),
			output: app.output.clone(),
			selected_midi_port: app.selected_midi_port.clone(),
		}
	}

	/// Applies persisted settings to `app`.
	///
	/// `resolve_clip` maps a persisted clip path onto the [`ClipId`] of the clip
	/// that this session loaded from it; returning `None` leaves the layer empty.
	pub fn restore_into<F>(&self, app: &mut AppState, mut resolve_clip: F)
	where
		F: FnMut(&Path) -> Option<ClipId>,
	{
		app.effects = self.effects.clone();
		app.effects.clamp_all();
		app.output = self.output.clone();
		app.selected_midi_port = self.selected_midi_port.clone();
		app.focused_layer = LayerIndex::from_number(self.focused_layer).unwrap_or_default();

		for (index, persisted) in self.layers.iter().enumerate() {
			let clip = persisted.clip_path.as_deref().and_then(&mut resolve_clip);
			let selected = persisted.selected_clip_path.as_deref().and_then(&mut resolve_clip);
			if let Some(layer) = app.layers.get_mut(index) {
				*layer = persisted.restore(clip, selected);
			}
		}

		app.dirty = false;
	}
}

/// Default per-layer settings used when a state file has no layer entry.
fn default_persisted_layers() -> [PersistedLayerState; 4] {
	std::array::from_fn(|_| PersistedLayerState::default())
}

/// Reads any number of persisted layers into the fixed four-layer array.
///
/// Missing entries are filled with [`PersistedLayerState::default`], and entries
/// past the fourth layer are ignored, because MUR always has exactly four
/// layers. This keeps older (0/1/3-entry) and newer (5-entry) state files
/// loadable instead of rejecting them as malformed.
fn deserialize_persisted_layers<'de, D>(deserializer: D) -> Result<[PersistedLayerState; 4], D::Error>
where
	D: serde::Deserializer<'de>,
{
	let entries = Vec::<PersistedLayerState>::deserialize(deserializer)?;
	let mut layers = default_persisted_layers();
	for (slot, entry) in layers.iter_mut().zip(entries) {
		*slot = entry;
	}
	Ok(layers)
}

/// Deletes a temporary file when dropped unless [`TempFileGuard::disarm`] was called.
struct TempFileGuard<'a> {
	path: &'a Path,
	armed: bool,
}

impl<'a> TempFileGuard<'a> {
	fn new(path: &'a Path) -> Self {
		Self { path, armed: true }
	}

	/// Keeps the file: it was renamed into place.
	fn disarm(&mut self) {
		self.armed = false;
	}
}

impl Drop for TempFileGuard<'_> {
	fn drop(&mut self) {
		if self.armed {
			let _ = std::fs::remove_file(self.path);
		}
	}
}

/// Writes `state` to `path` through a sibling `.tmp` file and an atomic rename.
///
/// The temporary file is removed whenever the save fails after creating it.
pub fn save_state_atomic(path: &Path, state: &PersistedState) -> Result<(), ConfigError> {
	let json = serde_json::to_string_pretty(state).map_err(|error| ConfigError::Serialize {
		path: path.display().to_string(),
		detail: error.to_string(),
	})?;

	if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
		std::fs::create_dir_all(parent).map_err(|error| ConfigError::io(parent, error))?;
	}

	let temp_path = sibling_temp_path(path);
	// Removes the temp file on every early return (create, write, fsync, or
	// rename failure), so repeated failed autosaves during a long show (e.g. a
	// full disk) never pile up `.tmp-{pid}-{seq}` files.
	let mut temp_guard = TempFileGuard::new(&temp_path);
	let mut file = std::fs::File::create(&temp_path).map_err(|error| ConfigError::io(&temp_path, error))?;
	file.write_all(json.as_bytes())
		.map_err(|error| ConfigError::io(&temp_path, error))?;
	file.sync_all().map_err(|error| ConfigError::io(&temp_path, error))?;
	drop(file);

	std::fs::rename(&temp_path, path).map_err(|error| ConfigError::io(path, error))?;
	temp_guard.disarm();

	// POSIX only guarantees that a rename is durable once the containing
	// directory has been flushed; this is best effort and never fails a save.
	#[cfg(unix)]
	if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty())
		&& let Ok(dir) = std::fs::File::open(parent)
	{
		let _ = dir.sync_all();
	}

	Ok(())
}

/// Loads a persisted snapshot.
///
/// Returns `Ok(None)` when the file does not exist or was written by a newer MUR
/// version, and an error when it exists but cannot be read or parsed.
pub fn load_state(path: &Path) -> Result<Option<PersistedState>, ConfigError> {
	let contents = match std::fs::read_to_string(path) {
		Ok(contents) => contents,
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
		Err(error) => return Err(ConfigError::io(path, error)),
	};

	let state: PersistedState =
		serde_json::from_str(&contents).map_err(|error| ConfigError::parse(path, "JSON", error.to_string()))?;

	if state.version > STATE_VERSION {
		return Ok(None);
	}
	Ok(Some(state))
}

/// Debounced, non-blocking helper that decides when to auto-save.
///
/// The caller marks state dirty on every change and polls [`StateSaver::save_if_due`]
/// from the render loop; disk writes happen at most once per debounce interval.
#[derive(Debug, Clone)]
pub struct StateSaver {
	debounce: Duration,
	last_save: Instant,
	dirty: bool,
}

impl StateSaver {
	/// Creates a saver that waits `debounce` after the last change before writing.
	pub fn new(debounce: Duration) -> Self {
		Self {
			debounce,
			last_save: Instant::now(),
			dirty: false,
		}
	}

	/// Creates a saver with [`DEFAULT_AUTOSAVE_DEBOUNCE`].
	pub fn with_default_debounce() -> Self {
		Self::new(DEFAULT_AUTOSAVE_DEBOUNCE)
	}

	/// Marks the session as changed and awaiting a save.
	pub fn mark_dirty(&mut self) {
		self.dirty = true;
	}

	/// Returns `true` when changes are waiting to be written.
	pub fn is_dirty(&self) -> bool {
		self.dirty
	}

	/// Records that the current state was written at `now`.
	///
	/// Callers that hand the disk write to a background task use this instead of
	/// [`StateSaver::new`] to keep their custom debounce interval while clearing
	/// the dirty flag.
	pub fn mark_saved(&mut self, now: Instant) {
		self.last_save = now;
		self.dirty = false;
	}

	/// Instant of the last save attempt.
	pub fn last_save(&self) -> Instant {
		self.last_save
	}

	/// Returns `true` when a save is pending and the debounce interval elapsed.
	pub fn is_due(&self, now: Instant) -> bool {
		self.dirty && now.saturating_duration_since(self.last_save) >= self.debounce
	}

	/// Writes `state` when [`StateSaver::is_due`] holds.
	///
	/// Returns the save result when a write was attempted, or `None` when the
	/// caller may skip work. Failures keep the dirty flag so the next poll retries.
	pub fn save_if_due(
		&mut self,
		now: Instant,
		path: &Path,
		state: &PersistedState,
	) -> Option<Result<(), ConfigError>> {
		if !self.is_due(now) {
			return None;
		}

		self.last_save = now;
		let result = save_state_atomic(path, state);
		if result.is_ok() {
			self.dirty = false;
		}
		Some(result)
	}
}

impl Default for StateSaver {
	fn default() -> Self {
		Self::with_default_debounce()
	}
}

/// Joins `path` onto `base_dir` unless it is already absolute.
fn resolve_relative(path: &Path, base_dir: &Path) -> PathBuf {
	if path.is_absolute() {
		path.to_path_buf()
	} else {
		base_dir.join(path)
	}
}

/// Process-wide counter that makes every staging file name unique.
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Builds the `path.tmp-{pid}-{seq}` sibling used for atomic writes.
///
/// Both the process id and a process-wide sequence number are part of the name,
/// so two MUR instances (or two concurrent saves) writing the same state file
/// can never truncate each other's staging file before the final rename.
fn sibling_temp_path(path: &Path) -> PathBuf {
	let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
	let mut temp = path.as_os_str().to_os_string();
	temp.push(format!(".tmp-{}-{sequence}", std::process::id()));
	PathBuf::from(temp)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
	use super::*;
	use clap::Parser as _;

	use crate::model::clip::{ClipInfo, ClipKind, ClipStatus};
	use crate::model::layer::PlaybackState;

	/// Creates a unique, not-yet-existing path under the system temp directory.
	fn temp_path(name: &str) -> PathBuf {
		let unique = std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.map(|elapsed| elapsed.as_nanos())
			.unwrap_or_default();
		std::env::temp_dir().join(format!("mur-test-{}-{}-{}", std::process::id(), unique, name))
	}

	fn clip(id: u64, path: &str) -> ClipInfo {
		ClipInfo {
			id: ClipId(id),
			path: PathBuf::from(path),
			name: "clip.mov".to_string(),
			kind: ClipKind::Hap,
			width: 1920,
			height: 1080,
			fps: 60.0,
			frame_count: 600,
			has_alpha: false,
			status: ClipStatus::Ready,
			thumbnail: None,
		}
	}

	#[test]
	fn cli_parses_all_flags() {
		let args = CliArgs::try_parse_from([
			"mur",
			"--folder",
			"/media",
			"--project",
			"show.toml",
			"--state",
			"state.json",
			"--no-restore",
		])
		.unwrap();
		assert_eq!(args.folder, Some(PathBuf::from("/media")));
		assert_eq!(args.project, Some(PathBuf::from("show.toml")));
		assert_eq!(args.state, Some(PathBuf::from("state.json")));
		assert!(args.no_restore);

		let empty = CliArgs::try_parse_from(["mur"]).unwrap();
		assert!(empty.folder.is_none());
		assert!(!empty.no_restore);
	}

	#[test]
	fn app_paths_compose_state_and_log_locations() {
		let paths = AppPaths::from_dirs(PathBuf::from("/cfg"), PathBuf::from("/data"));
		assert_eq!(paths.state_file, PathBuf::from("/data/state.json"));
		assert_eq!(paths.log_dir, PathBuf::from("/data/logs"));
		assert_eq!(paths.resolve_state_file(None), PathBuf::from("/data/state.json"));
		assert_eq!(
			paths.resolve_state_file(Some(Path::new("/custom.json"))),
			PathBuf::from("/custom.json")
		);
	}

	#[test]
	fn app_paths_discovery_is_total() {
		let paths = AppPaths::discover();
		assert!(paths.state_file.starts_with(&paths.data_dir));
		assert!(paths.log_dir.starts_with(&paths.data_dir));
	}

	#[test]
	fn project_json_resolves_relative_paths() {
		let json = r#"{
			"clip_folder": "media",
			"clips": ["intro.mov", "/absolute/loop.mov"],
			"layers": [{"clip_path": "layers/a.mov"}],
			"preferred_midi_port": "Launch Control XL 3"
		}"#;
		let mut config = ProjectConfig::parse(json, Path::new("/show/project.json")).unwrap();
		config.resolve_paths(Path::new("/show"));

		assert_eq!(config.clip_folder, Some(PathBuf::from("/show/media")));
		assert_eq!(
			config.explicit_clips().unwrap().to_vec(),
			vec![PathBuf::from("/show/intro.mov"), PathBuf::from("/absolute/loop.mov")]
		);
		assert_eq!(config.layers[0].clip_path, Some(PathBuf::from("/show/layers/a.mov")));
		assert_eq!(config.preferred_midi_port.as_deref(), Some("Launch Control XL 3"));
	}

	#[test]
	fn project_toml_parses_partial_layer_settings() {
		let toml = r#"
			clips = ["a.mov"]
			[[layers]]
			clip_path = "layers/one.mov"
			opacity = 0.5
			[effects]
			master_dimmer = 0.4
		"#;
		let config = ProjectConfig::parse(toml, Path::new("show.toml")).unwrap();
		assert_eq!(config.layers.len(), 1);
		assert_eq!(config.layers[0].opacity, 0.5);
		assert_eq!(config.layers[0].blend_mode, BlendMode::Normal);
		assert_eq!(config.layers[0].playback_mode, PlaybackMode::Loop);
		assert_eq!(config.effects.unwrap().master_dimmer, 0.4);
	}

	#[test]
	fn project_load_rejects_unknown_extension() {
		let error = ProjectConfig::parse("{}", Path::new("show.yaml")).unwrap_err();
		assert!(matches!(error, ConfigError::UnsupportedFormat { .. }));
	}

	#[test]
	fn project_load_reads_from_disk_and_resolves_media() {
		let dir = temp_path("project");
		std::fs::create_dir_all(&dir).unwrap();
		let file = dir.join("show.json");
		std::fs::write(&file, r#"{"clips":["clips/a.mov"]}"#).unwrap();

		let config = ProjectConfig::load(&file).unwrap();
		assert_eq!(config.clips, [dir.join("clips/a.mov")]);
		std::fs::remove_dir_all(&dir).ok();
	}

	#[test]
	fn persisted_state_round_trips_through_atomic_save() {
		let mut app = AppState::default();
		app.upsert_clip(clip(1, "/media/intro.mov"));
		app.trigger_clip_on_layer(LayerIndex::ALL[1], ClipId(1));
		app.set_layer_opacity(LayerIndex::ALL[1], 0.35);
		app.effects.master_dimmer = 0.6;
		app.selected_midi_port = Some("Launch Control XL 3".to_string());
		app.focused_layer = LayerIndex::ALL[2];

		let expected = PersistedState::capture(&app);
		assert_eq!(expected.version, STATE_VERSION);
		assert_eq!(expected.layers[1].clip_path, Some(PathBuf::from("/media/intro.mov")));
		assert_eq!(expected.focused_layer, 3);

		let dir = temp_path("state");
		let path = dir.join("state.json");
		save_state_atomic(&path, &expected).unwrap();
		assert!(!sibling_temp_path(&path).exists());

		let loaded = load_state(&path).unwrap().unwrap();
		assert_eq!(loaded, expected);
		std::fs::remove_dir_all(&dir).ok();
	}

	#[test]
	fn restoring_state_rebinds_clips_by_path() {
		let mut app = AppState::default();
		app.upsert_clip(clip(7, "/media/intro.mov"));
		app.effects = MasterEffects {
			master_dimmer: 0.2,
			..MasterEffects::default()
		};
		app.focused_layer = LayerIndex::ALL[3];
		app.layers[0].playback_mode = PlaybackMode::OneShot;

		let mut state = PersistedState::capture(&app);
		state.layers[0].clip_path = Some(PathBuf::from("/media/intro.mov"));
		state.layers[1].clip_path = Some(PathBuf::from("/media/missing.mov"));

		let mut restored = AppState {
			effects: MasterEffects {
				master_dimmer: 2.0,
				..MasterEffects::default()
			},
			..AppState::default()
		};
		state.restore_into(&mut restored, |path| {
			if path == Path::new("/media/intro.mov") {
				Some(ClipId(7))
			} else {
				None
			}
		});

		assert_eq!(restored.focused_layer, LayerIndex::ALL[3]);
		assert_eq!(restored.effects.master_dimmer, 0.2);
		assert_eq!(restored.layer(LayerIndex::ALL[0]).active_clip, Some(ClipId(7)));
		assert_eq!(restored.layer(LayerIndex::ALL[0]).playback_mode, PlaybackMode::OneShot);
		assert_eq!(
			restored.layer(LayerIndex::ALL[0]).playback_state,
			PlaybackState::Stopped
		);
		assert_eq!(restored.layer(LayerIndex::ALL[1]).active_clip, None);
		assert!(!restored.dirty);
	}

	#[test]
	fn selected_only_clip_is_not_restored_as_active() {
		let mut app = AppState::default();
		app.upsert_clip(clip(1, "/media/selected.mov"));
		app.upsert_clip(clip(2, "/media/active.mov"));
		// Layer 1: selected but never triggered. Layer 2: active, then another clip selected.
		app.layers[0].selected_clip = Some(ClipId(1));
		app.layers[1].active_clip = Some(ClipId(2));
		app.layers[1].selected_clip = Some(ClipId(1));

		let state = PersistedState::capture(&app);
		assert_eq!(state.layers[0].clip_path, None);
		assert_eq!(
			state.layers[0].selected_clip_path,
			Some(PathBuf::from("/media/selected.mov"))
		);
		assert_eq!(state.layers[1].clip_path, Some(PathBuf::from("/media/active.mov")));
		assert_eq!(
			state.layers[1].selected_clip_path,
			Some(PathBuf::from("/media/selected.mov"))
		);

		let json = serde_json::to_string(&state).unwrap();
		let state: PersistedState = serde_json::from_str(&json).unwrap();
		let mut restored = AppState::default();
		state.restore_into(&mut restored, |path| match path.to_str() {
			Some("/media/selected.mov") => Some(ClipId(1)),
			Some("/media/active.mov") => Some(ClipId(2)),
			_ => None,
		});
		assert_eq!(restored.layers[0].active_clip, None);
		assert_eq!(restored.layers[0].selected_clip, Some(ClipId(1)));
		// A pause key on the untriggered layer must not start the selected clip.
		assert!(!restored.toggle_pause_layer(LayerIndex::ALL[0]));
		assert_eq!(restored.layers[1].active_clip, Some(ClipId(2)));
		assert_eq!(restored.layers[1].selected_clip, Some(ClipId(1)));

		// Older files without `selected_clip_path` select the active clip.
		let legacy: PersistedLayerState = serde_json::from_str(r#"{"clip_path":"/media/active.mov"}"#).unwrap();
		assert_eq!(legacy.restore(Some(ClipId(2)), None).selected_clip, Some(ClipId(2)));
	}

	#[test]
	fn load_state_returns_none_for_missing_file() {
		let path = temp_path("missing").join("state.json");
		assert!(load_state(&path).unwrap().is_none());
	}

	#[test]
	fn load_state_rejects_malformed_and_future_files() {
		let dir = temp_path("malformed");
		std::fs::create_dir_all(&dir).unwrap();

		let malformed = dir.join("bad.json");
		std::fs::write(&malformed, "not json").unwrap();
		assert!(matches!(load_state(&malformed).unwrap_err(), ConfigError::Parse { .. }));

		let future = dir.join("future.json");
		std::fs::write(&future, format!(r#"{{"version":{}}}"#, STATE_VERSION + 1)).unwrap();
		assert!(load_state(&future).unwrap().is_none());

		std::fs::remove_dir_all(&dir).ok();
	}

	#[test]
	fn persisted_layers_tolerate_any_entry_count() {
		let layer = |opacity: f32| format!(r#"{{"opacity":{opacity}}}"#);

		// Empty list: four default layers.
		let state: PersistedState = serde_json::from_str(r#"{"version":1,"layers":[]}"#).unwrap();
		assert_eq!(state.layers, default_persisted_layers());

		// One entry: padded with defaults.
		let json = format!(r#"{{"version":1,"layers":[{}]}}"#, layer(0.25));
		let state: PersistedState = serde_json::from_str(&json).unwrap();
		assert_eq!(state.layers[0].opacity, 0.25);
		assert_eq!(state.layers[1], PersistedLayerState::default());
		assert_eq!(state.layers[3].playback_mode, PlaybackMode::Loop);

		// Three entries: the fourth layer keeps its default.
		let json = format!(
			r#"{{"version":1,"layers":[{},{},{}]}}"#,
			layer(0.1),
			layer(0.2),
			layer(0.3)
		);
		let state: PersistedState = serde_json::from_str(&json).unwrap();
		assert_eq!(state.layers[2].opacity, 0.3);
		assert_eq!(state.layers[3], PersistedLayerState::default());

		// Five entries: the fifth layer is ignored, not an error.
		let json = format!(
			r#"{{"version":1,"layers":[{},{},{},{},{}]}}"#,
			layer(0.1),
			layer(0.2),
			layer(0.3),
			layer(0.4),
			layer(0.5)
		);
		let state: PersistedState = serde_json::from_str(&json).unwrap();
		assert_eq!(state.layers.len(), 4);
		assert_eq!(state.layers[3].opacity, 0.4);

		// A missing `layers` key falls back to four default layers.
		let state: PersistedState = serde_json::from_str(r#"{"version":1}"#).unwrap();
		assert_eq!(state.layers, default_persisted_layers());

		// Serialization still emits exactly four entries.
		let json = serde_json::to_string(&PersistedState::default()).unwrap();
		let value: serde_json::Value = serde_json::from_str(&json).unwrap();
		assert_eq!(value["layers"].as_array().map(Vec::len), Some(4));
	}

	#[test]
	fn restore_keeps_a_valid_playhead_and_rejects_bad_values() {
		let mut app = AppState::default();
		app.upsert_clip(clip(1, "/media/intro.mov"));
		let mut state = PersistedState::capture(&app);
		state.layers[0].clip_path = Some(PathBuf::from("/media/intro.mov"));
		state.layers[0].playhead_frame = 37.5;

		let restored = state.layers[0].restore(Some(ClipId(1)), None);
		assert_eq!(restored.playhead_frame, 37.5);
		assert_eq!(restored.playback_state, PlaybackState::Stopped);

		for bad in [-1.0, f64::NAN, f64::INFINITY] {
			state.layers[0].playhead_frame = bad;
			assert_eq!(
				state.layers[0].restore(Some(ClipId(1)), None).playhead_frame,
				0.0,
				"{bad}"
			);
		}
	}

	#[test]
	fn state_saver_mark_saved_clears_dirty_and_keeps_debounce() {
		let mut saver = StateSaver::new(Duration::from_millis(500));
		saver.mark_dirty();
		assert!(saver.is_dirty());

		let now = Instant::now();
		saver.mark_saved(now);
		assert!(!saver.is_dirty());
		assert_eq!(saver.last_save(), now);

		// The custom 500 ms debounce survives; it is not reset to the default.
		saver.mark_dirty();
		assert!(!saver.is_due(now + Duration::from_millis(100)));
		assert!(saver.is_due(now + Duration::from_millis(600)));
	}

	#[test]
	fn failed_saves_leave_no_temp_files_behind() {
		let dir = temp_path("failed-save");
		std::fs::create_dir_all(&dir).unwrap();
		// Renaming a file onto an existing directory fails after the temp file
		// was written, which exercises the late-failure cleanup path.
		let target = dir.join("state.json");
		std::fs::create_dir_all(target.join("occupied")).unwrap();
		for _ in 0..3 {
			assert!(save_state_atomic(&target, &PersistedState::default()).is_err());
		}
		let leftovers: Vec<_> = std::fs::read_dir(&dir)
			.unwrap()
			.filter_map(Result::ok)
			.map(|entry| entry.file_name().to_string_lossy().into_owned())
			.filter(|name| name.contains(".tmp-"))
			.collect();
		assert!(leftovers.is_empty(), "temp files left behind: {leftovers:?}");

		// The guard removes the file on drop and keeps it once disarmed.
		let kept = dir.join("kept.tmp");
		std::fs::write(&kept, b"x").unwrap();
		TempFileGuard::new(&kept).disarm();
		assert!(kept.exists());
		{
			let _guard = TempFileGuard::new(&kept);
		}
		assert!(!kept.exists());

		std::fs::remove_dir_all(&dir).ok();
	}

	#[test]
	fn sibling_temp_paths_are_unique() {
		let path = Path::new("/tmp/mur-state.json");
		let first = sibling_temp_path(path);
		let second = sibling_temp_path(path);
		assert_ne!(first, second);
		assert!(first.to_string_lossy().contains(&std::process::id().to_string()));
		assert!(first.to_string_lossy().starts_with("/tmp/mur-state.json.tmp-"));
		assert_eq!(first.parent(), path.parent());
	}

	#[test]
	fn project_parse_clamps_effects() {
		let toml = r#"
			[effects]
			master_dimmer = 4.5
			strobe_speed_hz = -100.0
		"#;
		let effects = ProjectConfig::parse(toml, Path::new("show.toml"))
			.unwrap()
			.effects
			.unwrap();
		assert_eq!(effects.master_dimmer, 1.0);
		assert!(effects.strobe_speed_hz > 0.0);

		let json = r#"{"effects":{"master_dimmer":-3.0,"contrast":9.0}}"#;
		let effects = ProjectConfig::parse(json, Path::new("show.json"))
			.unwrap()
			.effects
			.unwrap();
		assert_eq!(effects.master_dimmer, 0.0);
		assert_eq!(effects.contrast, 2.0);
	}

	#[test]
	fn state_saver_debounces_until_dirty_and_elapsed() {
		let dir = temp_path("saver");
		let path = dir.join("state.json");
		let mut saver = StateSaver::new(Duration::from_millis(100));
		let state = PersistedState::default();

		let now = Instant::now();
		assert!(saver.save_if_due(now, &path, &state).is_none());
		saver.mark_dirty();
		assert!(saver.is_dirty());
		assert!(!saver.is_due(now));

		let due = now + Duration::from_millis(150);
		let result = saver.save_if_due(due, &path, &state);
		assert!(result.is_some());
		assert!(result.unwrap().is_ok());
		assert!(!saver.is_dirty());
		assert_eq!(saver.last_save(), due);
		assert!(path.exists());

		std::fs::remove_dir_all(&dir).ok();
	}
}
