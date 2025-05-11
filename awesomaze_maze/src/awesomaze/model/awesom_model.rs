use std::{cell::RefCell, rc::Rc};

use awesomlearn::machine_learning::{
    nn::{
        backend::Backend,
        dimensions::{Dimensions2D, Dimensions3D, Dimensions4D},
        model_config::{ActivationType, NNModelConfig, PoolType},
        nn_model::{Layer, NNModel},
        nn_model_builder::NNModelBuilder,
        optimizer::AdamOptimizer,
        tensor::Tensor,
    },
    reinforcement_learning::{
        q_learning::dqnet::DQNTargetModel,
        rl_error::RLResult,
        rl_model::{RLModel, RLTensor},
    },
};

use crate::awesomaze::game_model::MazeAction;

use super::InferenceModel;

pub struct AwesomModel<B: Backend> {
    backend: Rc<RefCell<B>>,
    model: NNModel<B>,
}

// FIXME: Hack. Since the model is really cloned once in the dqnet (with soft updates), this should
// be fine
impl<B: Backend> Clone for AwesomModel<B> {
    fn clone(&self) -> Self {
        let config = self.model.as_config().unwrap();
        let model_copy = NNModel::from_config(config, self.backend.clone(), 32).unwrap();
        Self {
            backend: self.backend.clone(),
            model: model_copy,
        }
    }
}

impl<B: Backend> InferenceModel for AwesomModel<B> {
    fn predict_move(&self, state: Vec<f32>) -> MazeAction {
        let state_tensor = Tensor::from_values(
            self.backend.clone(),
            state,
            Dimensions4D::from_d3(1, self.model.input_size()),
            false,
        )
        .unwrap();
        self.predict(state_tensor).detach().arg_max(3).to_values()[0].into()
    }
}

impl<B: Backend> AwesomModel<B> {
    pub fn new(backend: Rc<RefCell<B>>, feature_size: Dimensions3D, action_size: u32) -> Self {
        // let backend = Rc::new(RefCell::new(backend));
        let config = NNModelBuilder::new(32)
            .input(feature_size.depth, feature_size.height, feature_size.width)
            .conv2d(Some(ActivationType::Relu), (3, 3), 32, true)
            .pool2d(PoolType::MaxPool2D, Dimensions2D::new(2, 2))
            .conv2d(Some(ActivationType::Relu), (3, 3), 64, true)
            .pool2d(PoolType::MaxPool2D, Dimensions2D::new(2, 2))
            .conv2d(Some(ActivationType::Relu), (3, 3), 128, true)
            .pool2d(PoolType::MaxPool2D, Dimensions2D::new(2, 2))
            .flatten()
            .dense(Some(ActivationType::Relu), 512)
            .dense(None, action_size as usize)
            .get_config();
        let model = NNModel::from_config(config, backend, 1).unwrap();
        Self {
            backend: model.backend(),
            model,
        }
    }

    pub fn from_file(backend: Rc<RefCell<B>>, path: std::path::PathBuf) -> Self {
        let config = NNModelConfig::load(&path).unwrap();
        let model = NNModel::from_config(config, backend.clone(), 1).unwrap();
        AwesomModel { backend, model }
    }
}

impl<B: Backend> RLModel<Rc<RefCell<B>>, Tensor<B>, AdamOptimizer<B>> for AwesomModel<B> {
    fn init(
        path: Option<std::path::PathBuf>,
        backend: Rc<RefCell<B>>,
        feature_size: Dimensions3D,
        action_size: u32,
    ) -> RLResult<Self> {
        let model = if let Some(path) = path {
            let config = NNModelConfig::load(&path).unwrap();
            let model = NNModel::from_config(config, backend.clone(), 1)?;
            AwesomModel { backend, model }
        } else {
            AwesomModel::new(backend, feature_size, action_size)
        };
        Ok(model)
    }

    fn save_model(&self, path: std::path::PathBuf) -> RLResult<()> {
        self.model.as_config()?.save(path)?;
        Ok(())
    }

    fn predict(&self, observation: Tensor<B>) -> Tensor<B> {
        self.model.forward(observation).unwrap()
    }

    fn optimize(
        &mut self,
        loss: Tensor<B>,
        optimizer: &mut AdamOptimizer<B>,
        lr: f32,
        time_step: usize,
    ) {
        self.model
            .optimize(&loss, lr as f64, optimizer, time_step)
            .unwrap()
    }
}

impl<B: Backend> DQNTargetModel for AwesomModel<B> {
    fn soft_update(&mut self, other: &Self, tau: f32) {
        let tau_tensor = Tensor::from_values(
            self.backend.clone(),
            vec![tau],
            Dimensions4D::new(1, 1, 1, 1),
            false,
        )
        .unwrap();
        let one_tensor =
            Tensor::ones(self.backend.clone(), Dimensions4D::new(1, 1, 1, 1), false).unwrap();
        for (layer, other_layer) in self.model.layers_mut().iter_mut().zip(other.model.layers()) {
            match (layer, other_layer) {
                (
                    Layer::Dense {
                        weights,
                        biases,
                        activation: _,
                    },
                    Layer::Dense {
                        weights: weights_other,
                        biases: biases_other,
                        activation: _,
                    },
                ) => {
                    soft_update_param(weights, weights_other, &tau_tensor, &one_tensor).unwrap();
                    soft_update_param(biases, biases_other, &tau_tensor, &one_tensor).unwrap();
                }
                (
                    Layer::Conv2d {
                        kernels,
                        biases,
                        activation: _,
                        padding: _,
                    },
                    Layer::Conv2d {
                        kernels: kernels_other,
                        biases: biases_other,
                        activation: _,
                        padding: _,
                    },
                ) => {
                    soft_update_param(kernels, kernels_other, &tau_tensor, &one_tensor).unwrap();
                    soft_update_param(biases, biases_other, &tau_tensor, &one_tensor).unwrap();
                }
                _ => {}
            }
        }
    }
}

fn soft_update_param<B: Backend>(
    param: &mut Tensor<B>,
    other: &Tensor<B>,
    tau_tensor: &Tensor<B>,
    one_tensor: &Tensor<B>,
) -> RLResult<()> {
    // let updated_tensor = param * (1.0 - tau) + other * tau;
    let updated_tensor = param
        .clone()
        .mul(one_tensor.sub(tau_tensor)?)
        .add(other.mul(tau_tensor)?);
    let mut updated_tensor = updated_tensor.detach();
    Tensor::detach(&mut updated_tensor);
    param.replace_storage(updated_tensor);
    Ok(())
}
