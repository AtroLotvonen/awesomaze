use super::dimensions::Dimensions4D;
use serde_json::Error as SerdeError;
use std::io::Error as IOError;

#[derive(thiserror::Error, Debug)]
pub enum TensorError {
    #[error("The method: {0} is not implemented.")]
    NotImplemented(String),
    #[error("Empty model.")]
    EmptyModel,
    #[error("The backend had an Error with message {0}.")]
    BackendError(String),
    #[error("Dimension mismatch in `{tensor_op_name}`. Expected {expected:?} but received {received:?}.")]
    TensorDimensionMismatch {
        tensor_op_name: String,
        expected: Dimensions4D,
        received: Dimensions4D,
    },
    #[error("Dimension mismatch when initializing `{tensor_op_name}`. Expected values with dimensions {expected:?} and len of {expected_len:?} but received values of len {received:?}.")]
    TensorInitMismatch {
        tensor_op_name: String,
        expected: Dimensions4D,
        expected_len: usize,
        received: usize,
    },
    #[error("Dimension mismatch when broadcasting a dimension. Tried to broadcast dimensions {lhs:?} with {rhs:?}.")]
    BroadcastDimensionsMismatch {
        lhs: Dimensions4D,
        rhs: Dimensions4D,
    },
    #[error(
        "Tensor dimension: {dim} out of bounds indexed with size: `{indexed_size}`, when the real size is: `{real_size}`."
    )]
    TensorIndexOutOfBounds {
        dim: usize,
        indexed_size: usize,
        real_size: usize,
    },
    #[error(
        "Dimensions tried to index with `{index}`. Index must be a value between 0 and 3 inclusive."
    )]
    DimensionIndexError { index: usize },
    #[error("Cannot transpose the dimension {index} with itself")]
    TransposeDimensionError { index: usize },
    #[error("Can't find output dimensions for empty model config.")]
    EmptyModelConfig,
    #[error("Faulty model config in layer number: {0}.")]
    FaultyModelConfig(usize),
    #[error("Error computing loss.")]
    LossComputeError,
    #[error("Tensor with id:{tensor_id}")]
    MissingGradient { tensor_id: usize },
    #[error("Couldn't serialize the model.")]
    ModelSerializatioError(#[from] SerdeError),
    #[error("IO error")]
    IOError(#[from] IOError),
    #[error("Indice buffer had insufficient amount of indices.")]
    InsufficientIndices,
}

pub type Result<T> = std::result::Result<T, TensorError>;
