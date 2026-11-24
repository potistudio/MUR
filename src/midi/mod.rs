//! MIDI input: the fixed Launch Control XL 3 mapping and the hot-plug worker.
//!
//! The subsystem is split in two layers:
//!
//! - [`mapping`] turns raw Channel 1 bytes into semantic [`MidiAction`] values and
//!   applies them to [`AppState`](crate::model::AppState) without touching the
//!   renderer, so the mapping is fully unit-testable.
//! - [`worker`] owns a background thread around `midir`, polls ports once per
//!   second, survives backend panics, and reconnects automatically after a
//!   hot-unplug (SPEC §6.1).
//!
//! Both halves communicate through plain values, so no MIDI state is ever shared
//! across threads and the render loop only drains an event channel.

pub mod mapping;
pub mod worker;

pub use mapping::{MidiAction, MidiApplyResult, apply_midi_action, parse_midi_message, resolve_clip_index};
pub use worker::{
	MIDI_CLIENT_NAME, MIDI_CONNECTION_NAME, MidiCommand, MidiEvent, MidiWorkerHandle, PORT_POLL_INTERVAL,
	RECONNECT_INTERVAL, choose_preferred_port,
};
