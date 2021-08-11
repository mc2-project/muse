use std::fmt::{self, Display, Formatter};

/// Errors that may occur when using the `MPC` trait.
#[derive(Debug)]
pub enum MpcError {
    /// Mismatched input vector lengths to a gate
    MismatchedInputLength {
        /// Left gate length
        left: usize,
        /// Right gate lengh
        right: usize,
    },
    /// Insufficient number of triples for requested operation
    InsufficientTriples {
        /// Number triples available
        num: usize,
        /// Number triples needed
        needed: usize,
    },
    /// Insufficient number of rands for requested operation
    InsufficientRand {
        /// Number rands available
        num: usize,
        /// Number rands needed
        needed: usize,
    },
    /// A communication error occured
    CommunicationError(String),
    /// Attempted to open share with invalid MAC
    InvalidMAC,
    /// Committed values were not bits
    NotBits,
}

impl From<crypto_primitives::additive_share::AuthError> for MpcError {
    fn from(_: crypto_primitives::additive_share::AuthError) -> Self {
        MpcError::InvalidMAC
    }
}

impl From<bincode::Error> for MpcError {
    fn from(e: bincode::Error) -> Self {
        MpcError::CommunicationError(e.to_string())
    }
}

impl From<std::io::Error> for MpcError {
    fn from(e: std::io::Error) -> Self {
        MpcError::CommunicationError(e.to_string())
    }
}

impl Display for MpcError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            MpcError::MismatchedInputLength { left, right } => {
                write!(f, "InsufficientTriples: Left {} right {}", left, right)
            }
            MpcError::InsufficientTriples { num, needed } => {
                write!(f, "InsufficientTriples: Had {} needed {}", num, needed)
            }
            MpcError::InsufficientRand { num, needed } => {
                write!(f, "InsufficientTriples: Had {} needed {}", num, needed)
            }
            MpcError::CommunicationError(s) => write!(f, "Communication error: {}", s),
            MpcError::InvalidMAC => "Attempted to open share with an invalid MAC".fmt(f),
            MpcError::NotBits => "Committed values were not bits".fmt(f),
        }
    }
}
