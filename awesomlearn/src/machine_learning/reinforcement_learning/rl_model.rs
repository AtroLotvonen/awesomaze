use crate::machine_learning::nn::dimensions::{Dimensions3D, Dimensions4D};
use std::path::PathBuf;

use super::rl_error::RLResult;

pub trait RLModel<B: Clone, T: RLTensor<B>, O>: Clone {
    fn init(
        path: Option<PathBuf>,
        backend: B,
        feature_size: Dimensions3D,
        action_size: u32,
    ) -> RLResult<Self>;
    fn save_model(&self, path: PathBuf) -> RLResult<()>;
    fn predict(&self, observation: T) -> T;
    fn optimize(&mut self, loss: T, optimizer: &mut O, lr: f32, time_step: usize);
}

pub enum LossInfo {
    Mse(f32),
}

pub trait RLTensor<B: Clone> : Clone {
    fn from_values(values: Vec<f32>, dims: Dimensions4D, backend: &B, has_grad: bool) -> Self;
    fn to_values(&self) -> Vec<f32>;
    fn dimensions(&self) -> Dimensions4D;
    fn arg_max(self, dim: usize) -> Self; // TODO: implement this to awesome
    fn max_dim(self, dim: usize) -> Self; // TODO: implement this to awesome
    fn gather(self, dim: usize, indices: Self) -> Self; // TODO: implement this to awesome
    fn add(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn add_scalar(self, rhs: f32) -> Self;
    fn mul_scalar(self, rhs: f32) -> Self;
    fn detach(self) -> Self;
    fn loss(self, gt: Self, loss: RLLoss) -> Self;
}

pub enum RLLoss {
    Mse,
}

// impl<B: Backend> RLTensor<Rc<RefCell<B>>> for Tensor<B> {
//     fn from_values(
//         values: Vec<f32>,
//         dims: Dimensions4D,
//         backend: &Rc<RefCell<B>>,
//         has_grad: bool,
//     ) -> Self {
//         Tensor::from_values(backend.clone(), values, dims, has_grad)
//             .expect("Couldn't create awesom tensor from values.")
//     }
//
//     fn to_values(&self) -> Vec<f32> {
//         self.get_values()
//             .expect("Couldn't read values from awesom tensor.")
//     }
//
//     fn dimensions(&self) -> Dimensions4D {
//         self.dimensions()
//     }
// }

// type BurnBackend = burn::prelude::Backend;

// impl <B: BurnBackend>

// pub trait OffPolicyModel: RLModel {}
