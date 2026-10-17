// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Model assembly and stateless single-step inference for StarLight.
//!
//! `sl-core` owns command ordering and memory offsets; it never compiles a
//! graph, parses a model or implements an operator. The build phase validates
//! two [`CommandBuffer`](sl_ir::CommandBuffer)s, allocates every buffer and
//! pre-resolves each command port to a [`BufferRef`](sl_device::BufferRef). The
//! step phase only updates scalars, uploads token ids and positions, dispatches
//! the pre-resolved plans in order, synchronizes and downloads logits.
//!
//!
//! The types are frozen by `docs/ARCH-0007` §3.2.

#![deny(unsafe_code)]

mod engine;
mod error;

pub use engine::{
    CompiledProgram, Engine, EngineConfig, EngineInfo, StepInput, StepMode, StepStats,
};
pub use error::{BuildError, StepError};
