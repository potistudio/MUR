//! Domain model shared by the control UI, decoders, renderer, and MIDI loop.

pub mod app_state;
pub mod blend;
pub mod clip;
pub mod effect;
pub mod layer;

pub use app_state::{
	AppState, MidiConnectionStatus, MonitorSummary, OutputDisplaySelection, OutputRuntimeStatus, PerformanceStats,
};
pub use blend::BlendMode;
pub use clip::{ClipId, ClipInfo, ClipKind, ClipStatus, DecodedFrame, PixelPayload, ThumbnailImage};
pub use effect::MasterEffects;
pub use layer::{LayerIndex, LayerState, PlaybackMode, PlaybackState};
