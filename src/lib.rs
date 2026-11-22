//! MUR — live performance VJ software.
//!
//! This crate provides the building blocks shared by the control UI, the media
//! decoders, the renderer, and the MIDI loop: the domain model ([`model`]),
//! configuration and persistence ([`config`]), and the error/notification
//! primitives ([`error`]) that keep failures off the render thread.

#![warn(missing_docs)]

pub mod config;
pub mod error;
pub mod midi;
pub mod model;
