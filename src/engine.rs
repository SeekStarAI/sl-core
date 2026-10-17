// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Engine assembly and the allocation-free single-step hot path.
//!
//! [`Engine::build`] validates both compiled programs, allocates the
//! scratchpad, KV state and per-program input/output buffers, and resolves
//! every command port to a [`BufferRef`] up front. [`Engine::step`] only
//! updates scalars, uploads tokens and positions, dispatches the pre-resolved
//! plans in order, synchronizes and downloads the logits. It never allocates
//! and never takes a lock.

use sl_device::{BufferRef, Device};
use sl_ir::{
    BufferId, Command, CommandBuffer, DType, Effect, Op, StateKind, Value, ValueId, ValueRole,
    IR_VERSION,
};

use crate::error::{BuildError, StepError};

const PREFILL: &str = "prefill";
const DECODE: &str = "decode";
const ENGINE: &str = "engine";

/// Maximum number of buffer inputs of one command.
///
/// The largest signature (`CausalAttention`) has three value inputs plus the
/// appended KV-state input.
const MAX_INPUTS: usize = 4;

/// Maximum number of buffer outputs of one command.
const MAX_OUTPUTS: usize = 1;

/// Maximum number of scalar operands of one command.
const MAX_SCALARS: usize = 4;

/// Alignment passed to [`sl_ir::CommandBuffer::validate`] and to allocations.
const ALIGNMENT: usize = 64;

const NO_VALUE: ValueId = ValueId(u32::MAX);
const NO_BUFFER: BufferRef = BufferRef {
    buffer: BufferId(u32::MAX),
    offset: 0,
    len: 0,
};

/// A compiled command stream together with the semantic bindings the IR lacks.
///
/// The IR only knows value roles; names such as `token_ids` or `seq_len` are
/// supplied by the assembly layer that produced the program.
#[derive(Clone, Debug)]
pub struct CompiledProgram {
    /// Physical command stream.
    pub ir: CommandBuffer,
    /// Persistent parameter bindings, uploaded by the caller before build.
    pub weights: Vec<(ValueId, BufferId)>,
    /// `Input` value carrying token ids.
    pub token_ids: ValueId,
    /// `Input` value carrying positions; `None` when the graph has no RoPE.
    pub positions: Option<ValueId>,
    /// `Scalar` value carrying the current sequence length.
    pub seq_len: ValueId,
    /// `Scalar` value carrying the KV prefix length; `None` without KV cache.
    pub past_len: Option<ValueId>,
    /// `Output` value carrying logits.
    pub logits: ValueId,
}

/// Everything needed to assemble an [`Engine`].
pub struct EngineConfig {
    /// Program used by [`StepMode::Prefill`].
    pub prefill: CompiledProgram,
    /// Program used by [`StepMode::Decode`].
    pub decode: CompiledProgram,
    /// Device that owns all buffers.
    pub device: Box<dyn Device>,
    /// Maximum batch size.
    pub max_batch: u32,
    /// Maximum total sequence length (`past_len + seq_len`).
    pub max_context: u32,
}

/// Which compiled program a step runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepMode {
    /// Prompt ingestion over one or more new tokens.
    Prefill,
    /// Single-token generation against the existing KV cache.
    Decode,
}

/// Per-step inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepInput<'a> {
    /// Token ids for this step, in row-major `[batch, seq_len]` order.
    pub token_ids: &'a [u32],
    /// RoPE positions, in row-major `[batch, seq_len]` order. Ignored when the
    /// program has no `positions` binding.
    pub positions: &'a [u32],
    /// Number of new tokens carried by this step.
    pub seq_len: u32,
    /// KV prefix length. Ignored when the program has no `past_len` binding.
    pub past_len: u32,
}

/// Statistics of one completed step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepStats {
    /// Number of tokens processed by this step.
    pub tokens: u32,
}

/// Static engine description.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineInfo {
    /// IR version of the compiled programs.
    pub ir_version: u32,
    /// Configured maximum total sequence length.
    pub max_context: u32,
    /// Configured maximum batch size.
    pub max_batch: u32,
}

/// A fully bound, single-step inference engine.
///
/// The engine owns the device and every buffer it allocated. Weights are
/// *not* uploaded by [`Engine::build`]; the caller uploads them before build
/// and passes their [`BufferId`]s in [`CompiledProgram::weights`].
pub struct Engine {
    prefill: ProgramState,
    decode: ProgramState,
    device: Box<dyn Device>,
    max_batch: u32,
    max_context: u32,
}

impl Engine {
    /// Validates, allocates and pre-resolves both compiled programs.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] for a mismatched IR version, `sl-ir` validation
    /// failure, a missing weight/state/input/output/temp binding, an
    /// inconsistent state declaration, an incompatible port, a size overflow,
    /// or a device allocation failure.
    pub fn build(config: EngineConfig) -> Result<Self, BuildError> {
        let EngineConfig {
            prefill,
            decode,
            mut device,
            max_batch,
            max_context,
        } = config;

        if max_batch == 0 {
            return Err(BuildError::InvalidConfig("max_batch must be nonzero"));
        }
        if max_context == 0 {
            return Err(BuildError::InvalidConfig("max_context must be nonzero"));
        }

        validate_program(PREFILL, &prefill.ir)?;
        validate_program(DECODE, &decode.ir)?;
        check_state_declarations(&prefill.ir, &decode.ir)?;

        let scratchpad = allocate_scratchpad(&mut *device, &prefill.ir, &decode.ir)?;
        let states = allocate_states(&mut *device, &prefill.ir, max_batch, max_context)?;

        let prefill = build_program(PREFILL, prefill, &states, scratchpad, &mut *device)?;
        let decode = build_program(DECODE, decode, &states, scratchpad, &mut *device)?;

        Ok(Self {
            prefill,
            decode,
            device,
            max_batch,
            max_context,
        })
    }

    /// Executes one prefill or decode step.
    ///
    /// `output` must contain exactly as many elements as the program's logits
    /// value. The method allocates nothing and takes no lock.
    ///
    /// # Errors
    ///
    /// Returns [`StepError`] for empty input, a length or batch mismatch, a
    /// sequence that exceeds `max_context`, a wrong logits buffer size, or a
    /// device failure during upload, dispatch, synchronization or download.
    pub fn step(
        &mut self,
        mode: StepMode,
        input: StepInput<'_>,
        output: &mut [f32],
    ) -> Result<StepStats, StepError> {
        let program = match mode {
            StepMode::Prefill => &self.prefill,
            StepMode::Decode => &self.decode,
        };

        let tokens = input.token_ids.len();
        let seq_len = input.seq_len;
        if tokens == 0 || seq_len == 0 {
            return Err(StepError::EmptyInput);
        }
        if tokens != program.token_ids.len {
            return Err(StepError::InputLength {
                expected: program.token_ids.len,
                actual: tokens,
            });
        }

        // Capacity checks come before divisibility so an obviously oversized
        // sequence is reported as such.
        if seq_len > self.max_context {
            return Err(StepError::SequenceTooLong {
                requested: seq_len,
                limit: self.max_context,
            });
        }
        let past_len = if program.past_len.is_some() {
            input.past_len
        } else {
            0
        };
        let context_end = past_len
            .checked_add(seq_len)
            .ok_or(StepError::ArithmeticOverflow("past_len + seq_len"))?;
        if context_end > self.max_context {
            return Err(StepError::ContextTooLong {
                past_len,
                seq_len,
                limit: self.max_context,
            });
        }

        let seq_len_usize = usize::try_from(seq_len)
            .map_err(|_| StepError::ArithmeticOverflow("seq_len does not fit in usize"))?;
        if tokens % seq_len_usize != 0 {
            return Err(StepError::SequenceMismatch { tokens, seq_len });
        }
        let batch = tokens / seq_len_usize;
        let batch_u32 = u32::try_from(batch)
            .map_err(|_| StepError::ArithmeticOverflow("batch does not fit in u32"))?;
        if batch_u32 > self.max_batch {
            return Err(StepError::BatchTooLarge {
                requested: batch_u32,
                limit: self.max_batch,
            });
        }

        if let Some(positions) = program.positions {
            if input.positions.len() != positions.len {
                return Err(StepError::PositionsLength {
                    expected: positions.len,
                    actual: input.positions.len(),
                });
            }
        }

        if output.len() != program.logits.len {
            return Err(StepError::OutputLength {
                expected: program.logits.len,
                actual: output.len(),
            });
        }

        self.device
            .upload_u32(program.token_ids.buffer, input.token_ids)
            .map_err(StepError::Device)?;
        if let Some(positions) = program.positions {
            self.device
                .upload_u32(positions.buffer, input.positions)
                .map_err(StepError::Device)?;
        }

        run_plans(&mut *self.device, program, seq_len, past_len)?;
        self.device.synchronize().map_err(StepError::Device)?;
        self.device
            .download_f32(program.logits.buffer, output)
            .map_err(StepError::Device)?;

        Ok(StepStats {
            tokens: u32::try_from(tokens)
                .map_err(|_| StepError::ArithmeticOverflow("token count does not fit in u32"))?,
        })
    }

    /// Returns static engine metadata.
    pub fn info(&self) -> EngineInfo {
        EngineInfo {
            ir_version: self.prefill.ir.ir_version,
            max_context: self.max_context,
            max_batch: self.max_batch,
        }
    }
}

/// One command with every port and scalar already resolved.
#[derive(Debug)]
struct DispatchPlan {
    op: Op,
    input_buffers: [BufferRef; MAX_INPUTS],
    input_values: [ValueId; MAX_INPUTS],
    input_count: u8,
    output_buffers: [BufferRef; MAX_OUTPUTS],
    output_values: [ValueId; MAX_OUTPUTS],
    output_count: u8,
    scalar_sources: [ScalarSource; MAX_SCALARS],
    scalar_count: u8,
}

impl DispatchPlan {
    fn new(op: Op) -> Self {
        Self {
            op,
            input_buffers: [NO_BUFFER; MAX_INPUTS],
            input_values: [NO_VALUE; MAX_INPUTS],
            input_count: 0,
            output_buffers: [NO_BUFFER; MAX_OUTPUTS],
            output_values: [NO_VALUE; MAX_OUTPUTS],
            output_count: 0,
            scalar_sources: [ScalarSource::SeqLen; MAX_SCALARS],
            scalar_count: 0,
        }
    }
}

/// Where a scalar operand's per-step value comes from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScalarSource {
    SeqLen,
    PastLen,
}

/// One fully bound compiled program.
struct ProgramState {
    ir: CommandBuffer,
    token_ids: BufferRef,
    positions: Option<BufferRef>,
    logits: BufferRef,
    plans: Vec<DispatchPlan>,
    /// Whether the program binds a `past_len` scalar. When `None`, the step's
    /// `past_len` input is ignored.
    past_len: Option<ValueId>,
}

fn validate_program(program: &'static str, ir: &CommandBuffer) -> Result<(), BuildError> {
    if ir.ir_version != IR_VERSION {
        return Err(BuildError::InvalidIrVersion {
            program,
            found: ir.ir_version,
            expected: IR_VERSION,
        });
    }
    ir.validate(ALIGNMENT as u64)
        .map_err(|error| BuildError::InvalidCommandBuffer { program, error })
}

fn check_state_declarations(
    prefill: &CommandBuffer,
    decode: &CommandBuffer,
) -> Result<(), BuildError> {
    if prefill.states.len() != decode.states.len() {
        return Err(BuildError::StateSetMismatch {
            prefill: prefill.states.len(),
            decode: decode.states.len(),
        });
    }
    for declaration in &prefill.states {
        let other = decode
            .states
            .iter()
            .find(|candidate| candidate.value == declaration.value)
            .ok_or(BuildError::StateMismatch {
                value: declaration.value,
            })?;
        if other.kind != declaration.kind {
            return Err(BuildError::StateMismatch {
                value: declaration.value,
            });
        }
    }
    Ok(())
}

fn allocate_scratchpad(
    device: &mut dyn Device,
    prefill: &CommandBuffer,
    decode: &CommandBuffer,
) -> Result<Option<BufferId>, BuildError> {
    let bytes = prefill
        .plan
        .scratchpad_bytes
        .max(decode.plan.scratchpad_bytes);
    let has_slots = !prefill.plan.slots.is_empty() || !decode.plan.slots.is_empty();
    if bytes == 0 && !has_slots {
        return Ok(None);
    }
    let elements_u64 = (bytes / 4).saturating_add(u64::from(bytes % 4 != 0));
    let elements = usize::try_from(elements_u64).map_err(|_| BuildError::ArithmeticOverflow {
        program: ENGINE,
        what: "scratchpad element count",
    })?;
    let buffer = device
        .allocate_f32(elements, ALIGNMENT)
        .map_err(BuildError::Device)?;
    Ok(Some(buffer))
}

fn allocate_states(
    device: &mut dyn Device,
    prefill: &CommandBuffer,
    max_batch: u32,
    max_context: u32,
) -> Result<Vec<(ValueId, BufferRef)>, BuildError> {
    let mut states = Vec::with_capacity(prefill.states.len());
    for declaration in &prefill.states {
        let value = value_at(PREFILL, prefill, declaration.value)?;
        let elements = kv_cache_elements(
            PREFILL,
            declaration.value,
            &declaration.kind,
            max_batch,
            max_context,
        )?;
        if value.dtype != DType::F32 {
            return Err(BuildError::InvalidDType {
                program: PREFILL,
                value: declaration.value,
                expected: DType::F32,
                actual: value.dtype,
            });
        }
        let declared =
            value
                .shape
                .element_count()
                .map_err(|error| BuildError::InvalidCommandBuffer {
                    program: PREFILL,
                    error,
                })?;
        let declared = usize::try_from(declared).map_err(|_| BuildError::ArithmeticOverflow {
            program: PREFILL,
            what: "state element count",
        })?;
        if declared != elements {
            return Err(BuildError::ShapeMismatch {
                program: PREFILL,
                value: declaration.value,
                expected: elements,
                actual: declared,
            });
        }
        let buffer = device
            .allocate_f32(elements, ALIGNMENT)
            .map_err(BuildError::Device)?;
        states.push((
            declaration.value,
            BufferRef {
                buffer,
                offset: 0,
                len: elements,
            },
        ));
    }
    Ok(states)
}

fn kv_cache_elements(
    program: &'static str,
    value: ValueId,
    kind: &StateKind,
    max_batch: u32,
    max_context: u32,
) -> Result<usize, BuildError> {
    let StateKind::KvCache {
        batch,
        layers,
        kv_heads,
        head_dim,
        max_seq,
        dtype,
    } = *kind;
    if dtype != DType::F32 {
        return Err(BuildError::InvalidState {
            program,
            value,
            reason: "KV cache must be F32",
        });
    }
    let max_seq = max_seq.as_const().ok_or(BuildError::InvalidState {
        program,
        value,
        reason: "KV cache max_seq is symbolic",
    })?;
    if batch == 0 || layers == 0 || kv_heads == 0 || head_dim == 0 || max_seq == 0 {
        return Err(BuildError::InvalidState {
            program,
            value,
            reason: "KV cache dimensions must be nonzero",
        });
    }
    if batch > max_batch {
        return Err(BuildError::InvalidState {
            program,
            value,
            reason: "KV cache batch exceeds max_batch",
        });
    }
    if max_seq > max_context {
        return Err(BuildError::InvalidState {
            program,
            value,
            reason: "KV cache max_seq exceeds max_context",
        });
    }
    let elements = [batch, layers, 2, max_seq, kv_heads, head_dim]
        .iter()
        .try_fold(1_u64, |product, extent| {
            product.checked_mul(u64::from(*extent))
        })
        .ok_or(BuildError::ArithmeticOverflow {
            program,
            what: "KV cache element count",
        })?;
    usize::try_from(elements).map_err(|_| BuildError::ArithmeticOverflow {
        program,
        what: "KV cache element count",
    })
}

#[allow(clippy::too_many_arguments)]
fn build_program(
    program: &'static str,
    compiled: CompiledProgram,
    states: &[(ValueId, BufferRef)],
    scratchpad: Option<BufferId>,
    device: &mut dyn Device,
) -> Result<ProgramState, BuildError> {
    let CompiledProgram {
        ir,
        weights,
        token_ids,
        positions,
        seq_len,
        past_len,
        logits,
    } = compiled;

    let token_value = value_at(program, &ir, token_ids)?;
    require_role(program, token_ids, token_value, ValueRole::Input)?;
    require_dtype(program, token_ids, token_value, DType::U32)?;
    if !ir.inputs.contains(&token_ids) {
        return Err(BuildError::MissingInput {
            program,
            value: token_ids,
        });
    }

    if let Some(id) = positions {
        if id == token_ids {
            return Err(BuildError::InvalidBinding {
                program,
                reason: "token_ids and positions must be distinct values",
            });
        }
        let value = value_at(program, &ir, id)?;
        require_role(program, id, value, ValueRole::Input)?;
        require_dtype(program, id, value, DType::U32)?;
        if !ir.inputs.contains(&id) {
            return Err(BuildError::MissingInput { program, value: id });
        }
        let token_elements = element_count(program, &ir, token_ids)?;
        let position_elements = element_count(program, &ir, id)?;
        if token_elements != position_elements {
            return Err(BuildError::ShapeMismatch {
                program,
                value: id,
                expected: token_elements,
                actual: position_elements,
            });
        }
    }

    let seq_value = value_at(program, &ir, seq_len)?;
    require_role(program, seq_len, seq_value, ValueRole::Scalar)?;
    require_dtype(program, seq_len, seq_value, DType::U32)?;
    require_scalar_shape(program, seq_len, seq_value)?;

    if let Some(id) = past_len {
        let value = value_at(program, &ir, id)?;
        require_role(program, id, value, ValueRole::Scalar)?;
        require_dtype(program, id, value, DType::U32)?;
        require_scalar_shape(program, id, value)?;
        if id == seq_len {
            return Err(BuildError::InvalidBinding {
                program,
                reason: "seq_len and past_len must be distinct scalar values",
            });
        }
    }

    let logits_value = value_at(program, &ir, logits)?;
    require_role(program, logits, logits_value, ValueRole::Output)?;
    require_dtype(program, logits, logits_value, DType::F32)?;
    if !ir.outputs.contains(&logits) {
        return Err(BuildError::MissingOutput {
            program,
            value: logits,
        });
    }

    validate_weights(program, &ir, &weights)?;

    let input_buffers = allocate_buffers(program, &ir, &ir.inputs, device, false)?;
    let output_buffers = allocate_buffers(program, &ir, &ir.outputs, device, true)?;

    let token_ids_ref = lookup(&input_buffers, token_ids).ok_or(BuildError::MissingInput {
        program,
        value: token_ids,
    })?;
    let positions_ref = match positions {
        Some(id) => Some(
            lookup(&input_buffers, id).ok_or(BuildError::MissingInput { program, value: id })?,
        ),
        None => None,
    };
    let logits_ref = lookup(&output_buffers, logits).ok_or(BuildError::MissingOutput {
        program,
        value: logits,
    })?;

    let mut plans = Vec::with_capacity(ir.commands.len());
    for (index, command) in ir.commands.iter().enumerate() {
        plans.push(build_plan(
            program,
            &ir,
            command,
            index,
            &weights,
            states,
            scratchpad,
            &input_buffers,
            &output_buffers,
            token_ids,
            positions,
            seq_len,
            past_len,
        )?);
    }

    Ok(ProgramState {
        ir,
        token_ids: token_ids_ref,
        positions: positions_ref,
        logits: logits_ref,
        plans,
        past_len,
    })
}

fn validate_weights(
    program: &'static str,
    ir: &CommandBuffer,
    weights: &[(ValueId, BufferId)],
) -> Result<(), BuildError> {
    for (index, &(id, _)) in weights.iter().enumerate() {
        let value = value_at(program, ir, id)?;
        require_role(program, id, value, ValueRole::Param)?;
        if value.dtype != DType::F32 {
            return Err(BuildError::InvalidDType {
                program,
                value: id,
                expected: DType::F32,
                actual: value.dtype,
            });
        }
        if !value.is_dense() {
            return Err(BuildError::UnsupportedEncoding { program, value: id });
        }
        if weights[..index].iter().any(|(other, _)| *other == id) {
            return Err(BuildError::DuplicateWeight { program, value: id });
        }
    }
    Ok(())
}

fn allocate_buffers(
    program: &'static str,
    ir: &CommandBuffer,
    ids: &[ValueId],
    device: &mut dyn Device,
    outputs: bool,
) -> Result<Vec<(ValueId, BufferRef)>, BuildError> {
    let mut buffers = Vec::with_capacity(ids.len());
    for &id in ids {
        let value = value_at(program, ir, id)?;
        let elements = element_count(program, ir, id)?;
        let buffer = match value.dtype {
            DType::F32 => device.allocate_f32(elements, ALIGNMENT),
            DType::U32 if !outputs => device.allocate_u32(elements, ALIGNMENT),
            dtype => {
                return Err(if dtype == DType::U32 {
                    BuildError::InvalidDType {
                        program,
                        value: id,
                        expected: DType::F32,
                        actual: dtype,
                    }
                } else {
                    BuildError::UnsupportedDType {
                        program,
                        value: id,
                        dtype,
                    }
                })
            }
        }
        .map_err(BuildError::Device)?;
        buffers.push((
            id,
            BufferRef {
                buffer,
                offset: 0,
                len: elements,
            },
        ));
    }
    Ok(buffers)
}

#[allow(clippy::too_many_arguments)]
fn build_plan(
    program: &'static str,
    ir: &CommandBuffer,
    command: &Command,
    index: usize,
    weights: &[(ValueId, BufferId)],
    states: &[(ValueId, BufferRef)],
    scratchpad: Option<BufferId>,
    inputs: &[(ValueId, BufferRef)],
    outputs: &[(ValueId, BufferRef)],
    token_ids: ValueId,
    positions: Option<ValueId>,
    seq_len: ValueId,
    past_len: Option<ValueId>,
) -> Result<DispatchPlan, BuildError> {
    let signature = command.op.code().signature();
    if command.inputs.len() != signature.inputs.len()
        || command.outputs.len() != signature.outputs.len()
    {
        return Err(BuildError::SignatureMismatch {
            program,
            command: index,
        });
    }

    let mut plan = DispatchPlan::new(command.op);

    for (port, &id) in command.inputs.iter().enumerate() {
        let value = value_at(program, ir, id)?;
        let spec = signature.inputs[port];
        if !spec.dtypes.contains(value.dtype)
            || !spec.roles.contains(value.role)
            || value.shape.rank() < spec.min_rank as usize
            || value.shape.rank() > spec.max_rank as usize
        {
            return Err(BuildError::InvalidPort {
                program,
                command: index,
                port,
            });
        }

        if value.role == ValueRole::Scalar {
            let source = if id == seq_len {
                ScalarSource::SeqLen
            } else if Some(id) == past_len {
                ScalarSource::PastLen
            } else {
                return Err(BuildError::UnboundScalar { program, value: id });
            };
            let slot = usize::from(plan.scalar_count);
            if slot >= MAX_SCALARS {
                return Err(BuildError::TooManyPorts {
                    program,
                    command: index,
                    inputs: usize::from(plan.input_count),
                    outputs: usize::from(plan.output_count),
                    scalars: slot + 1,
                });
            }
            plan.scalar_sources[slot] = source;
            plan.scalar_count += 1;
        } else {
            let reference = resolve_buffer(
                program, ir, id, weights, states, scratchpad, inputs, outputs, token_ids, positions,
            )?;
            let slot = usize::from(plan.input_count);
            if slot >= MAX_INPUTS {
                return Err(BuildError::TooManyPorts {
                    program,
                    command: index,
                    inputs: slot + 1,
                    outputs: usize::from(plan.output_count),
                    scalars: usize::from(plan.scalar_count),
                });
            }
            plan.input_buffers[slot] = reference;
            plan.input_values[slot] = id;
            plan.input_count += 1;
        }
    }

    for (port, &id) in command.outputs.iter().enumerate() {
        let value = value_at(program, ir, id)?;
        let spec = signature.outputs[port];
        if !spec.dtypes.contains(value.dtype)
            || !spec.roles.contains(value.role)
            || value.shape.rank() < spec.min_rank as usize
            || value.shape.rank() > spec.max_rank as usize
        {
            return Err(BuildError::InvalidPort {
                program,
                command: index,
                port: command.inputs.len() + port,
            });
        }
        if value.role == ValueRole::Scalar {
            return Err(BuildError::InvalidPort {
                program,
                command: index,
                port: command.inputs.len() + port,
            });
        }
        let reference = resolve_buffer(
            program, ir, id, weights, states, scratchpad, inputs, outputs, token_ids, positions,
        )?;
        let slot = usize::from(plan.output_count);
        if slot >= MAX_OUTPUTS {
            return Err(BuildError::TooManyPorts {
                program,
                command: index,
                inputs: usize::from(plan.input_count),
                outputs: slot + 1,
                scalars: usize::from(plan.scalar_count),
            });
        }
        plan.output_buffers[slot] = reference;
        plan.output_values[slot] = id;
        plan.output_count += 1;
    }

    let stateful = is_stateful(command.op);
    match command.effect {
        Effect::None => {
            if stateful {
                return Err(BuildError::StateEffectMismatch {
                    program,
                    command: index,
                    reason: "stateful operator has no state effect",
                });
            }
        }
        Effect::ReadState(id) | Effect::WriteState(id) => {
            if !stateful {
                return Err(BuildError::StateEffectMismatch {
                    program,
                    command: index,
                    reason: "operator does not access persistent state",
                });
            }
            let state_value = value_at(program, ir, id)?;
            require_role(program, id, state_value, ValueRole::State)?;
            let reference = states
                .iter()
                .find(|(candidate, _)| *candidate == id)
                .map(|(_, reference)| *reference)
                .ok_or(BuildError::MissingState { program, value: id })?;
            let state_elements = element_count(program, ir, id)?;
            if state_elements != reference.len {
                return Err(BuildError::ShapeMismatch {
                    program,
                    value: id,
                    expected: state_elements,
                    actual: reference.len,
                });
            }
            let slot = usize::from(plan.input_count);
            if slot >= MAX_INPUTS {
                return Err(BuildError::TooManyPorts {
                    program,
                    command: index,
                    inputs: slot + 1,
                    outputs: usize::from(plan.output_count),
                    scalars: usize::from(plan.scalar_count),
                });
            }
            plan.input_buffers[slot] = reference;
            plan.input_values[slot] = id;
            plan.input_count += 1;
        }
    }

    Ok(plan)
}

#[allow(clippy::too_many_arguments)]
fn resolve_buffer(
    program: &'static str,
    ir: &CommandBuffer,
    id: ValueId,
    weights: &[(ValueId, BufferId)],
    states: &[(ValueId, BufferRef)],
    scratchpad: Option<BufferId>,
    inputs: &[(ValueId, BufferRef)],
    outputs: &[(ValueId, BufferRef)],
    token_ids: ValueId,
    positions: Option<ValueId>,
) -> Result<BufferRef, BuildError> {
    let value = value_at(program, ir, id)?;
    let elements = element_count(program, ir, id)?;
    let reference = match value.role {
        ValueRole::Param => weights
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, buffer)| BufferRef {
                buffer: *buffer,
                offset: 0,
                len: elements,
            })
            .ok_or(BuildError::MissingWeight { program, value: id })?,
        ValueRole::State => states
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, reference)| *reference)
            .ok_or(BuildError::MissingState { program, value: id })?,
        ValueRole::Input => {
            if id != token_ids && Some(id) != positions {
                return Err(BuildError::UnboundInput { program, value: id });
            }
            lookup(inputs, id).ok_or(BuildError::MissingInput { program, value: id })?
        }
        ValueRole::Output => {
            lookup(outputs, id).ok_or(BuildError::MissingOutput { program, value: id })?
        }
        ValueRole::Temp => {
            if value.dtype != DType::F32 {
                return Err(BuildError::InvalidDType {
                    program,
                    value: id,
                    expected: DType::F32,
                    actual: value.dtype,
                });
            }
            let slot = ir
                .plan
                .slots
                .iter()
                .find(|slot| slot.value == id)
                .ok_or(BuildError::MissingTemp { program, value: id })?;
            if slot.offset % 4 != 0 || slot.bytes % 4 != 0 {
                return Err(BuildError::InvalidBinding {
                    program,
                    reason: "scratchpad slot is not 4-byte aligned",
                });
            }
            let offset =
                usize::try_from(slot.offset / 4).map_err(|_| BuildError::ArithmeticOverflow {
                    program,
                    what: "scratchpad offset",
                })?;
            let len =
                usize::try_from(slot.bytes / 4).map_err(|_| BuildError::ArithmeticOverflow {
                    program,
                    what: "scratchpad slot length",
                })?;
            if len != elements {
                return Err(BuildError::ShapeMismatch {
                    program,
                    value: id,
                    expected: elements,
                    actual: len,
                });
            }
            let buffer = scratchpad.ok_or(BuildError::MissingTemp { program, value: id })?;
            BufferRef {
                buffer,
                offset,
                len,
            }
        }
        ValueRole::Scalar => return Err(BuildError::UnboundScalar { program, value: id }),
    };
    if value.role != ValueRole::Temp && reference.len != elements {
        return Err(BuildError::ShapeMismatch {
            program,
            value: id,
            expected: elements,
            actual: reference.len,
        });
    }
    Ok(reference)
}

fn value_at<'a>(
    program: &'static str,
    ir: &'a CommandBuffer,
    id: ValueId,
) -> Result<&'a Value, BuildError> {
    ir.values
        .get(id.0 as usize)
        .ok_or(BuildError::UnknownValue { program, value: id })
}

fn element_count(
    program: &'static str,
    ir: &CommandBuffer,
    id: ValueId,
) -> Result<usize, BuildError> {
    let value = value_at(program, ir, id)?;
    let count = value
        .shape
        .element_count()
        .map_err(|error| BuildError::InvalidCommandBuffer { program, error })?;
    usize::try_from(count).map_err(|_| BuildError::ArithmeticOverflow {
        program,
        what: "value element count",
    })
}

fn lookup(buffers: &[(ValueId, BufferRef)], id: ValueId) -> Option<BufferRef> {
    buffers
        .iter()
        .find(|(candidate, _)| *candidate == id)
        .map(|(_, reference)| *reference)
}

fn require_role(
    program: &'static str,
    id: ValueId,
    value: &Value,
    expected: ValueRole,
) -> Result<(), BuildError> {
    if value.role == expected {
        Ok(())
    } else {
        Err(BuildError::InvalidRole {
            program,
            value: id,
            expected,
            actual: value.role,
        })
    }
}

fn require_dtype(
    program: &'static str,
    id: ValueId,
    value: &Value,
    expected: DType,
) -> Result<(), BuildError> {
    if value.dtype == expected {
        Ok(())
    } else {
        Err(BuildError::InvalidDType {
            program,
            value: id,
            expected,
            actual: value.dtype,
        })
    }
}

fn require_scalar_shape(
    program: &'static str,
    id: ValueId,
    value: &Value,
) -> Result<(), BuildError> {
    let count = value
        .shape
        .element_count()
        .map_err(|error| BuildError::InvalidCommandBuffer { program, error })?;
    if count == 1 {
        Ok(())
    } else {
        Err(BuildError::ShapeMismatch {
            program,
            value: id,
            expected: 1,
            actual: usize::try_from(count).unwrap_or(usize::MAX),
        })
    }
}

fn is_stateful(op: Op) -> bool {
    matches!(op, Op::KvAppend { .. } | Op::CausalAttention { .. })
}

fn run_plans(
    device: &mut dyn Device,
    program: &ProgramState,
    seq_len: u32,
    past_len: u32,
) -> Result<(), StepError> {
    if program.plans.is_empty() {
        return Ok(());
    }
    let values = &program.ir.values;
    let fallback = values
        .first()
        .ok_or(StepError::InvalidProgram("command plan has no values"))?;

    for plan in &program.plans {
        let input_count = usize::from(plan.input_count);
        let output_count = usize::from(plan.output_count);
        let scalar_count = usize::from(plan.scalar_count);

        let mut input_values: [&Value; MAX_INPUTS] = core::array::from_fn(|_| fallback);
        for (slot, value) in input_values.iter_mut().enumerate().take(input_count) {
            *value =
                values
                    .get(plan.input_values[slot].0 as usize)
                    .ok_or(StepError::InvalidProgram(
                        "dispatch references an unknown input value",
                    ))?;
        }
        let mut output_values: [&Value; MAX_OUTPUTS] = core::array::from_fn(|_| fallback);
        for (slot, value) in output_values.iter_mut().enumerate().take(output_count) {
            *value = values.get(plan.output_values[slot].0 as usize).ok_or(
                StepError::InvalidProgram("dispatch references an unknown output value"),
            )?;
        }
        let mut scalars = [0_u32; MAX_SCALARS];
        for (slot, scalar) in scalars.iter_mut().enumerate().take(scalar_count) {
            *scalar = match plan.scalar_sources[slot] {
                ScalarSource::SeqLen => seq_len,
                ScalarSource::PastLen => past_len,
            };
        }

        device
            .dispatch(
                &plan.op,
                &plan.input_buffers[..input_count],
                &input_values[..input_count],
                &plan.output_buffers[..output_count],
                &output_values[..output_count],
                &scalars[..scalar_count],
            )
            .map_err(StepError::Device)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use sl_device::DeviceError;
    use sl_ir::{CommandId, Dim, MemoryPlan, Shape, Slot, StateDecl};

    struct MockDevice {
        state: Arc<Mutex<MockState>>,
    }

    #[derive(Clone, Debug)]
    struct MockDispatch {
        op: Op,
        inputs: Vec<BufferRef>,
        input_values: Vec<Value>,
        outputs: Vec<BufferRef>,
        output_values: Vec<Value>,
        scalars: Vec<u32>,
    }

    #[derive(Default)]
    struct MockState {
        floats: Vec<Option<Vec<f32>>>,
        integers: Vec<Option<Vec<u32>>>,
        dispatches: Vec<MockDispatch>,
        uploads_f32: Vec<(BufferId, Vec<f32>)>,
        uploads_u32: Vec<(BufferId, Vec<u32>)>,
        downloads: Vec<BufferId>,
        fail_dispatch_at: Option<usize>,
        fail_synchronize: bool,
        fill_on_sync: bool,
    }

    impl MockDevice {
        fn new(state: Arc<Mutex<MockState>>) -> Self {
            Self { state }
        }
    }

    impl Device for MockDevice {
        fn allocate_f32(
            &mut self,
            elements: usize,
            _align: usize,
        ) -> Result<BufferId, DeviceError> {
            let mut state = self.state.lock().expect("mock state lock");
            let buffer =
                BufferId(u32::try_from(state.floats.len()).expect("test buffer id fits in u32"));
            state.floats.push(Some(vec![0.0; elements]));
            state.integers.push(None);
            Ok(buffer)
        }

        fn allocate_u32(
            &mut self,
            elements: usize,
            _align: usize,
        ) -> Result<BufferId, DeviceError> {
            let mut state = self.state.lock().expect("mock state lock");
            let buffer =
                BufferId(u32::try_from(state.floats.len()).expect("test buffer id fits in u32"));
            state.floats.push(None);
            state.integers.push(Some(vec![0_u32; elements]));
            Ok(buffer)
        }

        fn upload_f32(&mut self, buffer: BufferId, data: &[f32]) -> Result<(), DeviceError> {
            let mut state = self.state.lock().expect("mock state lock");
            state.uploads_f32.push((buffer, data.to_vec()));
            let slot = state
                .floats
                .get_mut(buffer.0 as usize)
                .and_then(Option::as_mut)
                .ok_or(DeviceError::InvalidBuffer(buffer))?;
            if slot.len() != data.len() {
                return Err(DeviceError::InvalidRange {
                    buffer,
                    offset: 0,
                    len: data.len(),
                });
            }
            slot.copy_from_slice(data);
            Ok(())
        }

        fn upload_u32(&mut self, buffer: BufferId, data: &[u32]) -> Result<(), DeviceError> {
            let mut state = self.state.lock().expect("mock state lock");
            state.uploads_u32.push((buffer, data.to_vec()));
            let slot = state
                .integers
                .get_mut(buffer.0 as usize)
                .and_then(Option::as_mut)
                .ok_or(DeviceError::InvalidBuffer(buffer))?;
            if slot.len() != data.len() {
                return Err(DeviceError::InvalidRange {
                    buffer,
                    offset: 0,
                    len: data.len(),
                });
            }
            slot.copy_from_slice(data);
            Ok(())
        }

        fn download_f32(&mut self, buffer: BufferId, out: &mut [f32]) -> Result<(), DeviceError> {
            let mut state = self.state.lock().expect("mock state lock");
            state.downloads.push(buffer);
            let slot = state
                .floats
                .get(buffer.0 as usize)
                .and_then(Option::as_deref)
                .ok_or(DeviceError::InvalidBuffer(buffer))?;
            if slot.len() != out.len() {
                return Err(DeviceError::InvalidRange {
                    buffer,
                    offset: 0,
                    len: out.len(),
                });
            }
            out.copy_from_slice(slot);
            Ok(())
        }

        fn download_u32(&mut self, buffer: BufferId, out: &mut [u32]) -> Result<(), DeviceError> {
            let state = self.state.lock().expect("mock state lock");
            let slot = state
                .integers
                .get(buffer.0 as usize)
                .and_then(Option::as_deref)
                .ok_or(DeviceError::InvalidBuffer(buffer))?;
            if slot.len() != out.len() {
                return Err(DeviceError::InvalidRange {
                    buffer,
                    offset: 0,
                    len: out.len(),
                });
            }
            out.copy_from_slice(slot);
            Ok(())
        }

        fn dispatch(
            &mut self,
            op: &Op,
            inputs: &[BufferRef],
            input_values: &[&Value],
            outputs: &[BufferRef],
            output_values: &[&Value],
            scalars: &[u32],
        ) -> Result<(), DeviceError> {
            let mut state = self.state.lock().expect("mock state lock");
            if state.fail_dispatch_at == Some(state.dispatches.len()) {
                return Err(DeviceError::Execution("mock dispatch failure"));
            }
            state.dispatches.push(MockDispatch {
                op: *op,
                inputs: inputs.to_vec(),
                input_values: input_values.iter().map(|value| (*value).clone()).collect(),
                outputs: outputs.to_vec(),
                output_values: output_values.iter().map(|value| (*value).clone()).collect(),
                scalars: scalars.to_vec(),
            });
            Ok(())
        }

        fn synchronize(&mut self) -> Result<(), DeviceError> {
            let mut state = self.state.lock().expect("mock state lock");
            if state.fail_synchronize {
                return Err(DeviceError::Execution("mock synchronize failure"));
            }
            if state.fill_on_sync {
                for (index, slot) in state.floats.iter_mut().enumerate() {
                    if let Some(data) = slot {
                        for (offset, value) in data.iter_mut().enumerate() {
                            *value = index as f32 * 1000.0 + offset as f32;
                        }
                    }
                }
            }
            Ok(())
        }
    }

    fn f32_value(shape: &[u32], role: ValueRole) -> Value {
        Value::new(DType::F32, Shape::consts(shape), role)
    }

    fn u32_value(shape: &[u32], role: ValueRole) -> Value {
        Value::new(DType::U32, Shape::consts(shape), role)
    }

    fn scalar_value() -> Value {
        Value::new(DType::U32, Shape::default(), ValueRole::Scalar)
    }

    fn command(op: Op, inputs: &[u32], outputs: &[u32], effect: Effect) -> Command {
        Command {
            op,
            inputs: inputs.iter().map(|&id| ValueId(id)).collect(),
            outputs: outputs.iter().map(|&id| ValueId(id)).collect(),
            deps: Vec::<CommandId>::new(),
            effect,
        }
    }

    fn command_buffer(
        values: Vec<Value>,
        commands: Vec<Command>,
        plan: MemoryPlan,
        inputs: &[u32],
        outputs: &[u32],
        states: Vec<StateDecl>,
    ) -> CommandBuffer {
        CommandBuffer {
            ir_version: IR_VERSION,
            values,
            commands,
            plan,
            inputs: inputs.iter().map(|&id| ValueId(id)).collect(),
            outputs: outputs.iter().map(|&id| ValueId(id)).collect(),
            states,
        }
    }

    /// `Gather -> RmsNorm -> MatMulNT -> LastToken` with three scratch slots.
    fn chain_program(seq: u32) -> CompiledProgram {
        let hidden_bytes = u64::from(seq) * 4 * 4;
        let logits_bytes = u64::from(seq) * 8 * 4;
        let values = vec![
            u32_value(&[1, seq], ValueRole::Input),
            f32_value(&[8, 4], ValueRole::Param),
            f32_value(&[1, seq, 4], ValueRole::Temp),
            f32_value(&[1, seq, 4], ValueRole::Temp),
            f32_value(&[1, seq, 8], ValueRole::Temp),
            f32_value(&[4], ValueRole::Param),
            f32_value(&[8, 4], ValueRole::Param),
            scalar_value(),
            f32_value(&[1, 8], ValueRole::Output),
        ];
        let commands = vec![
            command(Op::Gather, &[1, 0], &[2], Effect::None),
            command(Op::RmsNorm { eps: 1e-5 }, &[2, 5], &[3], Effect::None),
            command(Op::MatMulNT, &[3, 6], &[4], Effect::None),
            command(Op::LastToken, &[4, 7], &[8], Effect::None),
        ];
        let plan = MemoryPlan {
            scratchpad_bytes: 128 + logits_bytes,
            slots: vec![
                Slot {
                    value: ValueId(2),
                    offset: 0,
                    bytes: hidden_bytes,
                },
                Slot {
                    value: ValueId(3),
                    offset: 64,
                    bytes: hidden_bytes,
                },
                Slot {
                    value: ValueId(4),
                    offset: 128,
                    bytes: logits_bytes,
                },
            ],
        };
        let ir = command_buffer(values, commands, plan, &[0], &[8], Vec::new());
        CompiledProgram {
            ir,
            weights: Vec::new(),
            token_ids: ValueId(0),
            positions: None,
            seq_len: ValueId(7),
            past_len: None,
            logits: ValueId(8),
        }
    }

    fn build_chain(
        missing_weight: Option<ValueId>,
        fail_dispatch_at: Option<usize>,
    ) -> Result<(Engine, Arc<Mutex<MockState>>), BuildError> {
        let shared = Arc::new(Mutex::new(MockState {
            fail_dispatch_at,
            fill_on_sync: true,
            ..MockState::default()
        }));
        let mut device = MockDevice::new(shared.clone());
        let table = device.allocate_f32(32, 64).expect("mock weight allocation");
        let norm = device.allocate_f32(4, 64).expect("mock weight allocation");
        let w = device.allocate_f32(32, 64).expect("mock weight allocation");

        let mut weights = vec![(ValueId(1), table), (ValueId(5), norm), (ValueId(6), w)];
        if let Some(missing) = missing_weight {
            weights.retain(|(value, _)| *value != missing);
        }

        let mut prefill = chain_program(2);
        prefill.weights = weights.clone();
        let mut decode = chain_program(1);
        decode.weights = weights;
        let engine = Engine::build(EngineConfig {
            prefill,
            decode,
            device: Box::new(device),
            max_batch: 1,
            max_context: 8,
        })?;
        Ok((engine, shared))
    }

    fn chain_engine() -> (Engine, Arc<Mutex<MockState>>) {
        build_chain(None, None).expect("chain engine should build")
    }

    fn attention_program(seq: u32, state_kind: StateKind) -> CompiledProgram {
        let max_seq = match state_kind {
            StateKind::KvCache { max_seq, .. } => max_seq
                .as_const()
                .expect("test state kind uses a concrete max_seq"),
        };
        let values = vec![
            f32_value(&[1, 1, seq, 2], ValueRole::Param),
            f32_value(&[1, 1, seq, 2], ValueRole::Param),
            f32_value(&[1, 1, seq, 2], ValueRole::Param),
            scalar_value(),
            f32_value(&[1, 1, seq, 2], ValueRole::Output),
            f32_value(&[1, 1, 2, max_seq, 1, 2], ValueRole::State),
            u32_value(&[1, seq], ValueRole::Input),
            scalar_value(),
        ];
        let commands = vec![command(
            Op::CausalAttention {
                layer: 0,
                heads: 1,
                kv_heads: 1,
                head_dim: 2,
                scale: 0.5,
            },
            &[0, 1, 2, 3],
            &[4],
            Effect::WriteState(ValueId(5)),
        )];
        let ir = command_buffer(
            values,
            commands,
            MemoryPlan::default(),
            &[6],
            &[4],
            vec![StateDecl {
                value: ValueId(5),
                kind: state_kind,
            }],
        );
        CompiledProgram {
            ir,
            weights: Vec::new(),
            token_ids: ValueId(6),
            positions: None,
            seq_len: ValueId(7),
            past_len: Some(ValueId(3)),
            logits: ValueId(4),
        }
    }

    fn kv_state_kind(max_seq: u32) -> StateKind {
        StateKind::KvCache {
            batch: 1,
            layers: 1,
            kv_heads: 1,
            head_dim: 2,
            max_seq: Dim::Const(max_seq),
            dtype: DType::F32,
        }
    }

    #[test]
    fn builds_engine_and_steps_prefill_and_decode() {
        let (mut engine, shared) = chain_engine();
        assert_eq!(
            engine.info(),
            EngineInfo {
                ir_version: IR_VERSION,
                max_context: 8,
                max_batch: 1,
            }
        );

        let mut logits = [0.0_f32; 8];
        let stats = engine
            .step(
                StepMode::Prefill,
                StepInput {
                    token_ids: &[3, 5],
                    positions: &[],
                    seq_len: 2,
                    past_len: 0,
                },
                &mut logits,
            )
            .expect("prefill should succeed");
        assert_eq!(stats, StepStats { tokens: 2 });

        let state = shared.lock().expect("mock state lock");
        assert_eq!(state.dispatches.len(), 4);
        assert!(matches!(state.dispatches[0].op, Op::Gather));
        assert!(matches!(state.dispatches[1].op, Op::RmsNorm { .. }));
        assert!(matches!(state.dispatches[2].op, Op::MatMulNT));
        assert!(matches!(state.dispatches[3].op, Op::LastToken));
        assert_eq!(state.dispatches[3].scalars, [2]);
        assert_eq!(state.uploads_u32.len(), 1);
        assert_eq!(state.uploads_u32[0].1, [3, 5]);
        assert_eq!(state.downloads.len(), 1);
        let logits_buffer = state.downloads[0];
        let expected = state.floats[logits_buffer.0 as usize]
            .as_deref()
            .expect("logits buffer");
        assert_eq!(logits, expected[..]);
        drop(state);

        let stats = engine
            .step(
                StepMode::Decode,
                StepInput {
                    token_ids: &[7],
                    positions: &[],
                    seq_len: 1,
                    past_len: 2,
                },
                &mut logits,
            )
            .expect("decode should succeed");
        assert_eq!(stats, StepStats { tokens: 1 });

        let state = shared.lock().expect("mock state lock");
        assert_eq!(state.dispatches.len(), 8);
        assert_eq!(state.dispatches[7].scalars, [1]);
        assert_eq!(state.uploads_u32[1].1, [7]);
        let logits_buffer = state.downloads[1];
        let expected = state.floats[logits_buffer.0 as usize]
            .as_deref()
            .expect("decode logits buffer");
        assert_eq!(logits, expected[..]);
    }

    #[test]
    fn resolves_command_order_and_scratchpad_offsets() {
        let (mut engine, shared) = chain_engine();
        let mut logits = [0.0_f32; 8];
        engine
            .step(
                StepMode::Prefill,
                StepInput {
                    token_ids: &[1, 2],
                    positions: &[],
                    seq_len: 2,
                    past_len: 0,
                },
                &mut logits,
            )
            .expect("prefill should succeed");

        let state = shared.lock().expect("mock state lock");
        let gather = &state.dispatches[0];
        let rms = &state.dispatches[1];
        let matmul = &state.dispatches[2];
        let last = &state.dispatches[3];

        assert_eq!(gather.outputs[0].offset, 0);
        assert_eq!(gather.outputs[0].len, 8);
        assert_eq!(gather.inputs[1].len, 2);
        assert_eq!(rms.inputs[0].buffer, gather.outputs[0].buffer);
        assert_eq!(rms.inputs[0].offset, 0);
        assert_eq!(rms.outputs[0].offset, 16);
        assert_eq!(rms.outputs[0].len, 8);
        assert_eq!(matmul.inputs[0].buffer, gather.outputs[0].buffer);
        assert_eq!(matmul.inputs[0].offset, 16);
        assert_eq!(matmul.outputs[0].offset, 32);
        assert_eq!(matmul.outputs[0].len, 16);
        assert_eq!(last.inputs[0].buffer, gather.outputs[0].buffer);
        assert_eq!(last.inputs[0].offset, 32);
        assert_eq!(last.outputs[0].len, 8);
        assert_ne!(last.outputs[0].buffer, gather.outputs[0].buffer);
    }

    #[test]
    fn reports_missing_weight_before_stepping() {
        let error = match build_chain(Some(ValueId(1)), None) {
            Ok(_) => panic!("a missing weight must be rejected"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            BuildError::MissingWeight {
                program: PREFILL,
                value: ValueId(1),
            }
        );
    }

    #[test]
    fn rejects_oversized_sequences_and_wrong_outputs() {
        let (mut engine, _shared) = chain_engine();
        let mut logits = [0.0_f32; 8];
        let error = engine
            .step(
                StepMode::Prefill,
                StepInput {
                    token_ids: &[1, 2],
                    positions: &[],
                    seq_len: 9,
                    past_len: 0,
                },
                &mut logits,
            )
            .expect_err("seq_len above max_context must be rejected");
        assert_eq!(
            error,
            StepError::SequenceTooLong {
                requested: 9,
                limit: 8,
            }
        );

        let error = engine
            .step(
                StepMode::Prefill,
                StepInput {
                    token_ids: &[1, 2],
                    positions: &[],
                    seq_len: 2,
                    past_len: 0,
                },
                &mut [0.0_f32; 7],
            )
            .expect_err("a short logits buffer must be rejected");
        assert_eq!(
            error,
            StepError::OutputLength {
                expected: 8,
                actual: 7,
            }
        );
    }

    #[test]
    fn surfaces_device_dispatch_failures() {
        let (mut engine, _shared) =
            build_chain(None, Some(2)).expect("engine should build with a failing device");
        let mut logits = [0.0_f32; 8];
        let error = engine
            .step(
                StepMode::Prefill,
                StepInput {
                    token_ids: &[1, 2],
                    positions: &[],
                    seq_len: 2,
                    past_len: 0,
                },
                &mut logits,
            )
            .expect_err("the mock device must fail");
        assert_eq!(
            error,
            StepError::Device(DeviceError::Execution("mock dispatch failure"))
        );
    }

    #[test]
    fn empty_and_zero_length_inputs_return_errors_without_panicking() {
        let (mut engine, _shared) = chain_engine();
        let mut logits = [0.0_f32; 8];
        assert_eq!(
            engine
                .step(
                    StepMode::Prefill,
                    StepInput {
                        token_ids: &[],
                        positions: &[],
                        seq_len: 2,
                        past_len: 0,
                    },
                    &mut logits,
                )
                .expect_err("empty tokens must be rejected"),
            StepError::EmptyInput
        );
        assert_eq!(
            engine
                .step(
                    StepMode::Prefill,
                    StepInput {
                        token_ids: &[1, 2],
                        positions: &[],
                        seq_len: 0,
                        past_len: 0,
                    },
                    &mut logits,
                )
                .expect_err("zero seq_len must be rejected"),
            StepError::EmptyInput
        );
    }

    #[test]
    fn rejects_invalid_ir_versions() {
        let shared = Arc::new(Mutex::new(MockState::default()));
        let device = MockDevice::new(shared);
        let mut prefill = chain_program(2);
        prefill.ir.ir_version = IR_VERSION + 1;
        let decode = chain_program(1);
        let error = match Engine::build(EngineConfig {
            prefill,
            decode,
            device: Box::new(device),
            max_batch: 1,
            max_context: 8,
        }) {
            Ok(_) => panic!("a mismatched IR version must be rejected"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            BuildError::InvalidIrVersion {
                program: PREFILL,
                found: IR_VERSION + 1,
                expected: IR_VERSION,
            }
        );
    }

    #[test]
    fn shares_kv_state_and_appends_it_to_stateful_dispatches() {
        let shared = Arc::new(Mutex::new(MockState {
            fill_on_sync: false,
            ..MockState::default()
        }));
        let mut device = MockDevice::new(shared.clone());
        let weights = vec![
            (
                ValueId(0),
                device.allocate_f32(4, 64).expect("q weight allocation"),
            ),
            (
                ValueId(1),
                device.allocate_f32(4, 64).expect("k weight allocation"),
            ),
            (
                ValueId(2),
                device.allocate_f32(4, 64).expect("v weight allocation"),
            ),
        ];
        let mut prefill = attention_program(2, kv_state_kind(4));
        prefill.weights = weights.clone();
        let mut decode = attention_program(1, kv_state_kind(4));
        decode.weights = weights;
        let mut engine = Engine::build(EngineConfig {
            prefill,
            decode,
            device: Box::new(device),
            max_batch: 1,
            max_context: 8,
        })
        .expect("attention engine should build");

        let mut logits = [0.0_f32; 4];
        engine
            .step(
                StepMode::Prefill,
                StepInput {
                    token_ids: &[1, 2],
                    positions: &[],
                    seq_len: 2,
                    past_len: 0,
                },
                &mut logits,
            )
            .expect("prefill should succeed");
        engine
            .step(
                StepMode::Decode,
                StepInput {
                    token_ids: &[3],
                    positions: &[],
                    seq_len: 1,
                    past_len: 2,
                },
                &mut logits[..2],
            )
            .expect("decode should succeed");

        let state = shared.lock().expect("mock state lock");
        assert_eq!(state.dispatches.len(), 2);
        let prefill = &state.dispatches[0];
        let decode = &state.dispatches[1];
        assert_eq!(prefill.inputs.len(), 4);
        assert_eq!(prefill.scalars, [0]);
        assert_eq!(decode.inputs.len(), 4);
        assert_eq!(decode.scalars, [2]);
        assert_eq!(prefill.inputs[3].buffer, decode.inputs[3].buffer);
        assert_eq!(prefill.input_values[3].role, ValueRole::State);
        assert_eq!(decode.input_values[3].role, ValueRole::State);
        assert_eq!(prefill.output_values[0].role, ValueRole::Output);
        assert_eq!(decode.output_values[0].role, ValueRole::Output);
    }

    #[test]
    fn rejects_inconsistent_state_declarations() {
        let shared = Arc::new(Mutex::new(MockState::default()));
        let device = MockDevice::new(shared);
        let prefill = attention_program(2, kv_state_kind(4));
        let decode = attention_program(1, kv_state_kind(8));
        let error = match Engine::build(EngineConfig {
            prefill,
            decode,
            device: Box::new(device),
            max_batch: 1,
            max_context: 8,
        }) {
            Ok(_) => panic!("inconsistent state declarations must be rejected"),
            Err(error) => error,
        };
        assert_eq!(error, BuildError::StateMismatch { value: ValueId(5) });
    }
}
