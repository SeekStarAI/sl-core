// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Build-time and step-time errors for [`crate::Engine`].

use core::fmt;

use sl_device::DeviceError;
use sl_ir::{DType, IrError, ValueId, ValueRole};

/// Failure while assembling an [`crate::Engine`] from compiled programs.
///
/// Every variant is allocation-free and names the program (`"prefill"` or
/// `"decode"`) and the offending value or command where applicable so a
/// malformed binding can be diagnosed without re-running the build.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BuildError {
    /// A command buffer was compiled against a different IR version.
    InvalidIrVersion {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Version found in the buffer.
        found: u32,
        /// Version expected by the engine.
        expected: u32,
    },
    /// `sl-ir` rejected the command buffer.
    InvalidCommandBuffer {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Validation error.
        error: IrError,
    },
    /// The engine configuration is inconsistent.
    InvalidConfig(&'static str),
    /// A binding is structurally invalid.
    InvalidBinding {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Human-readable reason.
        reason: &'static str,
    },
    /// A command references a value id outside the value table.
    UnknownValue {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Offending value.
        value: ValueId,
    },
    /// A `Param` value has no entry in [`crate::CompiledProgram::weights`].
    MissingWeight {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Unbound parameter.
        value: ValueId,
    },
    /// A parameter uses a storage encoding this engine cannot execute.
    UnsupportedEncoding {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// The quantized parameter.
        value: ValueId,
    },
    /// A state value has no allocation.
    MissingState {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Unbound state.
        value: ValueId,
    },
    /// A command reads an input that the program never declared.
    MissingInput {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Unbound input.
        value: ValueId,
    },
    /// A command writes an output that the program never declared.
    MissingOutput {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Unbound output.
        value: ValueId,
    },
    /// A `Temp` value has no scratchpad slot.
    MissingTemp {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Unbound temporary.
        value: ValueId,
    },
    /// A command reads an input value that is neither token ids nor positions.
    UnboundInput {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Input value without an upload binding.
        value: ValueId,
    },
    /// A scalar operand is neither the `seq_len` nor the `past_len` binding.
    UnboundScalar {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Scalar value without a binding.
        value: ValueId,
    },
    /// A value has a role the engine does not accept at that position.
    InvalidRole {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Offending value.
        value: ValueId,
        /// Role required by the engine.
        expected: ValueRole,
        /// Role found in the IR.
        actual: ValueRole,
    },
    /// A value has a dtype the engine does not accept at that position.
    InvalidDType {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Offending value.
        value: ValueId,
        /// Dtype required by the engine.
        expected: DType,
        /// Dtype found in the IR.
        actual: DType,
    },
    /// A dtype is not supported by this engine at all.
    UnsupportedDType {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Offending value.
        value: ValueId,
        /// Dtype found in the IR.
        dtype: DType,
    },
    /// The same parameter is bound twice.
    DuplicateWeight {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Repeated parameter.
        value: ValueId,
    },
    /// A command's port is incompatible with the operator signature.
    InvalidPort {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Command index.
        command: usize,
        /// Input or output port index (inputs first).
        port: usize,
    },
    /// A command's arity does not match the operator signature.
    SignatureMismatch {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Command index.
        command: usize,
    },
    /// A command has more ports than the engine supports.
    TooManyPorts {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Command index.
        command: usize,
        /// Buffer inputs, including an appended state buffer.
        inputs: usize,
        /// Buffer outputs.
        outputs: usize,
        /// Scalar operands.
        scalars: usize,
    },
    /// A value's element count does not match its binding.
    ShapeMismatch {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Offending value.
        value: ValueId,
        /// Element count implied by the IR shape.
        expected: usize,
        /// Element count implied by the binding.
        actual: usize,
    },
    /// Prefill and decode disagree about a state declaration.
    StateMismatch {
        /// Offending state value.
        value: ValueId,
    },
    /// Prefill and decode declare a different number of states.
    StateSetMismatch {
        /// Number of prefill state declarations.
        prefill: usize,
        /// Number of decode state declarations.
        decode: usize,
    },
    /// A KV-cache declaration is unusable.
    InvalidState {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Offending state value.
        value: ValueId,
        /// Human-readable reason.
        reason: &'static str,
    },
    /// A stateful operator has no state effect, or a stateless one has one.
    StateEffectMismatch {
        /// `"prefill"` or `"decode"`.
        program: &'static str,
        /// Command index.
        command: usize,
        /// Human-readable reason.
        reason: &'static str,
    },
    /// A size computation overflowed.
    ArithmeticOverflow {
        /// `"prefill"`, `"decode"`, or `"engine"`.
        program: &'static str,
        /// Quantity that overflowed.
        what: &'static str,
    },
    /// The device rejected an allocation.
    Device(DeviceError),
}

impl fmt::Display for BuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIrVersion {
                program,
                found,
                expected,
            } => write!(
                formatter,
                "{program} program uses IR version {found}; expected {expected}"
            ),
            Self::InvalidCommandBuffer { program, error } => {
                write!(formatter, "{program} command buffer is invalid: {error}")
            }
            Self::InvalidConfig(reason) => write!(formatter, "invalid engine config: {reason}"),
            Self::InvalidBinding { program, reason } => {
                write!(formatter, "{program} binding is invalid: {reason}")
            }
            Self::UnknownValue { program, value } => {
                write!(formatter, "{program} references unknown value {}", value.0)
            }
            Self::MissingWeight { program, value } => {
                write!(formatter, "{program} has no weight for value {}", value.0)
            }
            Self::UnsupportedEncoding { program, value } => write!(
                formatter,
                "{program} parameter {} uses a storage encoding this engine cannot execute",
                value.0
            ),
            Self::MissingState { program, value } => {
                write!(formatter, "{program} has no state buffer for value {}", value.0)
            }
            Self::MissingInput { program, value } => {
                write!(formatter, "{program} has no input buffer for value {}", value.0)
            }
            Self::MissingOutput { program, value } => {
                write!(formatter, "{program} has no output buffer for value {}", value.0)
            }
            Self::MissingTemp { program, value } => {
                write!(
                    formatter,
                    "{program} has no scratchpad slot for value {}",
                    value.0
                )
            }
            Self::UnboundInput { program, value } => {
                write!(
                    formatter,
                    "{program} reads input value {} without an upload binding",
                    value.0
                )
            }
            Self::UnboundScalar { program, value } => {
                write!(
                    formatter,
                    "{program} reads scalar value {} without a binding",
                    value.0
                )
            }
            Self::InvalidRole {
                program,
                value,
                expected,
                actual,
            } => write!(
                formatter,
                "{program} value {} has role {actual}; expected {expected}",
                value.0
            ),
            Self::InvalidDType {
                program,
                value,
                expected,
                actual,
            } => write!(
                formatter,
                "{program} value {} has dtype {actual}; expected {expected}",
                value.0
            ),
            Self::UnsupportedDType {
                program,
                value,
                dtype,
            } => write!(
                formatter,
                "{program} value {} uses unsupported dtype {dtype}",
                value.0
            ),
            Self::DuplicateWeight { program, value } => {
                write!(formatter, "{program} binds weight {} twice", value.0)
            }
            Self::InvalidPort {
                program,
                command,
                port,
            } => write!(
                formatter,
                "{program} command {command} port {port} does not match the operator signature"
            ),
            Self::SignatureMismatch { program, command } => write!(
                formatter,
                "{program} command {command} has the wrong number of ports"
            ),
            Self::TooManyPorts {
                program,
                command,
                inputs,
                outputs,
                scalars,
            } => write!(
                formatter,
                "{program} command {command} needs {inputs} inputs, {outputs} outputs and {scalars} scalars; at most 4/1/4 are supported"
            ),
            Self::ShapeMismatch {
                program,
                value,
                expected,
                actual,
            } => write!(
                formatter,
                "{program} value {} has {actual} bound elements; expected {expected}",
                value.0
            ),
            Self::StateMismatch { value } => write!(
                formatter,
                "prefill and decode disagree about state {}",
                value.0
            ),
            Self::StateSetMismatch { prefill, decode } => write!(
                formatter,
                "prefill declares {prefill} states but decode declares {decode}"
            ),
            Self::InvalidState {
                program,
                value,
                reason,
            } => write!(
                formatter,
                "{program} state {} is invalid: {reason}",
                value.0
            ),
            Self::StateEffectMismatch {
                program,
                command,
                reason,
            } => write!(
                formatter,
                "{program} command {command} has an invalid state effect: {reason}"
            ),
            Self::ArithmeticOverflow { program, what } => {
                write!(formatter, "{program} {what} overflows")
            }
            Self::Device(error) => write!(formatter, "device error during build: {error}"),
        }
    }
}

impl core::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Device(error) => Some(error),
            Self::InvalidCommandBuffer { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Failure while executing one [`crate::Engine::step`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepError {
    /// No token was supplied, or `seq_len` is zero.
    EmptyInput,
    /// The token buffer does not have the size the program was compiled for.
    InputLength {
        /// Expected number of token ids.
        expected: usize,
        /// Supplied number of token ids.
        actual: usize,
    },
    /// The positions buffer does not have the size the program was compiled for.
    PositionsLength {
        /// Expected number of positions.
        expected: usize,
        /// Supplied number of positions.
        actual: usize,
    },
    /// `seq_len` does not divide the number of supplied tokens.
    SequenceMismatch {
        /// Number of supplied token ids.
        tokens: usize,
        /// Requested new-token count.
        seq_len: u32,
    },
    /// `seq_len` exceeds `max_context`.
    SequenceTooLong {
        /// Requested new-token count.
        requested: u32,
        /// Configured context capacity.
        limit: u32,
    },
    /// `past_len + seq_len` exceeds `max_context`.
    ContextTooLong {
        /// Requested prefix length.
        past_len: u32,
        /// Requested new-token count.
        seq_len: u32,
        /// Configured context capacity.
        limit: u32,
    },
    /// The derived batch size exceeds `max_batch`.
    BatchTooLarge {
        /// Derived batch size.
        requested: u32,
        /// Configured batch capacity.
        limit: u32,
    },
    /// The caller's logits buffer has the wrong length.
    OutputLength {
        /// Expected number of logits.
        expected: usize,
        /// Supplied buffer length.
        actual: usize,
    },
    /// A size computation overflowed.
    ArithmeticOverflow(&'static str),
    /// A pre-resolved plan violates an engine invariant.
    InvalidProgram(&'static str),
    /// The device rejected an upload, dispatch, synchronization or download.
    Device(DeviceError),
}

impl fmt::Display for StepError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => formatter.write_str("step requires at least one token"),
            Self::InputLength { expected, actual } => write!(
                formatter,
                "step received {actual} token ids; the program expects {expected}"
            ),
            Self::PositionsLength { expected, actual } => write!(
                formatter,
                "step received {actual} positions; the program expects {expected}"
            ),
            Self::SequenceMismatch { tokens, seq_len } => write!(
                formatter,
                "step received {tokens} token ids, which is not a multiple of seq_len {seq_len}"
            ),
            Self::SequenceTooLong { requested, limit } => write!(
                formatter,
                "seq_len {requested} exceeds the context capacity {limit}"
            ),
            Self::ContextTooLong {
                past_len,
                seq_len,
                limit,
            } => write!(
                formatter,
                "past_len {past_len} + seq_len {seq_len} exceeds the context capacity {limit}"
            ),
            Self::BatchTooLarge { requested, limit } => {
                write!(formatter, "batch {requested} exceeds max_batch {limit}")
            }
            Self::OutputLength { expected, actual } => write!(
                formatter,
                "logits buffer has {actual} elements; expected {expected}"
            ),
            Self::ArithmeticOverflow(what) => write!(formatter, "{what} overflows"),
            Self::InvalidProgram(reason) => {
                write!(formatter, "engine program invariant violated: {reason}")
            }
            Self::Device(error) => write!(formatter, "device error during step: {error}"),
        }
    }
}

impl core::error::Error for StepError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Device(error) => Some(error),
            _ => None,
        }
    }
}
