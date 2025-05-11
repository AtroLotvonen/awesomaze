use std::{cell::RefCell, rc::Rc};

use crate::machine_learning::reinforcement_learning::rl_model::{RLLoss, RLTensor};

use super::{backend::Backend, dimensions::Dimensions4D, tensor::Tensor};

impl<B: Backend> RLTensor<Rc<RefCell<B>>> for Tensor<B> {
    fn from_values(
        values: Vec<f32>,
        dims: Dimensions4D,
        backend: &Rc<RefCell<B>>,
        has_grad: bool,
    ) -> Self {
        Tensor::from_values(backend.clone(), values, dims, has_grad)
            .expect("Couldn't execute awesom operation.")
    }

    fn to_values(&self) -> Vec<f32> {
        self.get_values().expect("Couldn't get awesom values")
    }

    fn dimensions(&self) -> Dimensions4D {
        Tensor::dimensions(self)
    }

    fn arg_max(self, dim: usize) -> Self {
        Tensor::argmax(&self, dim).expect("Couldn't execute awesom operation.")
    }

    fn max_dim(self, dim: usize) -> Self {
        Tensor::max_dim(&self, dim).expect("Couldn't execute awesom operation.")
    }

    fn gather(self, dim: usize, indices: Self) -> Self {
        Tensor::gather(&self, dim, &indices).expect("Couldn't execute awesom operation.")
    }

    fn add(self, rhs: Self) -> Self {
        Tensor::add(&self, &rhs).expect("Couldn't execute awesom operation.")
    }

    fn mul(self, rhs: Self) -> Self {
        Tensor::mul(&self, &rhs).expect("Couldn't execute awesom operation.")
    }

    fn add_scalar(self, rhs: f32) -> Self {
        let scalar_tensor = Tensor::from_values(
            self.backend(),
            vec![rhs],
            Dimensions4D::new(1, 1, 1, 1),
            false,
        )
        .expect("Couldn't execute awesom operation.");
        Tensor::add(&self, &scalar_tensor).expect("Couldn't execute awesom operation.")
    }

    fn mul_scalar(self, rhs: f32) -> Self {
        let scalar_tensor = Tensor::from_values(
            self.backend(),
            vec![rhs],
            Dimensions4D::new(1, 1, 1, 1),
            false,
        )
        .expect("Couldn't execute awesom operation.");
        Tensor::mul(&self, &scalar_tensor).expect("Couldn't execute awesom operation.")
    }

    fn detach(mut self) -> Self {
        Tensor::detach(&mut self)
    }

    fn loss(self, gt: Self, loss: RLLoss) -> Self {
        match loss {
            RLLoss::Mse => self.mse(&gt).expect("Couldn't execute awesom operation."),
        }
    }
}
