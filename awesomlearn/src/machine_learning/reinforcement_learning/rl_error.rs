use crate::machine_learning::nn::tensor_error::TensorError;

#[derive(thiserror::Error, Debug)]
pub enum RLError {
    #[error("Action out of range got: {value} when max action value is {max}.")]
    ActionOutOfRange { value: u32, max: u32 },
    #[error("Tensor error")]
    TensorError(#[from] TensorError),
}

pub type RLResult<T> = std::result::Result<T, RLError>;
