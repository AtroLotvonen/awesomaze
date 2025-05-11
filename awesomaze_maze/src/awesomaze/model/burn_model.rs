use std::path::PathBuf;

use awesomlearn::machine_learning::{
    nn::dimensions::{Dimensions3D, Dimensions4D},
    reinforcement_learning::{
        environment::RLAction,
        q_learning::dqnet::DQNTargetModel,
        rl_error::RLResult,
        rl_model::{RLLoss, RLModel, RLTensor},
    },
};
use burn::{
    module::{Param, ParamId},
    nn::PaddingConfig2d,
    optim::{GradientsParams, Optimizer},
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{
        backend::{AutodiffBackend, Backend},
        cast::ToElement,
    },
    LearningRate,
};
use nn::loss::{MseLoss, Reduction};

use crate::awesomaze::game_model::MazeAction;

use super::InferenceModel;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: nn::conv::Conv2d<B>,
    conv2: nn::conv::Conv2d<B>,
    conv3: nn::conv::Conv2d<B>,
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    activation: nn::Relu,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device, feature_size: Dimensions3D, action_size: u32) -> Self {
        let conv1 = nn::conv::Conv2dConfig::new([feature_size.depth, 32], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_stride([2, 2])
            .with_bias(true)
            .init(device);
        let conv2 = nn::conv::Conv2dConfig::new([32, 64], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_stride([2, 2])
            .with_bias(true)
            .init(device);
        let conv3 = nn::conv::Conv2dConfig::new([64, 128], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_stride([2, 2])
            .with_bias(true)
            .init(device);

        let hidden_width = feature_size.width / 8;
        let hidden_height = feature_size.height / 8;

        let hidden_size = 128 * hidden_height * hidden_width;

        let fc1 = nn::LinearConfig::new(hidden_size, 512)
            .with_bias(true)
            .init(device);
        let fc2 = nn::LinearConfig::new(512, action_size as usize)
            .with_bias(true)
            .init(device);

        Self {
            conv1,
            conv2,
            conv3,
            fc1,
            fc2,
            activation: nn::Relu::new(),
        }
    }

    pub fn forward(&self, input: BurnTensor<B, 4>) -> BurnTensor<B, 4> {
        let x = input.detach().0;

        let x = self.conv1.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv3.forward(x);
        let x = self.activation.forward(x);

        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);

        BurnTensor(self.fc2.forward(x).reshape([-1, 1, 1, 4]))
    }
}

pub struct InferenceBurnModel<B: Backend> {
    model: Model<B>,
    device: B::Device,
    input_size: Dimensions3D,
}

impl<B: Backend> InferenceBurnModel<B> {
    pub fn load_model(path: PathBuf, input_size: Dimensions3D, device: B::Device) -> Self {
        let model = Model::new(&device, input_size, MazeAction::SIZE);
        let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
            .load(path, &device)
            .expect("Model could not be loaded.");
        let model = model.load_record(record);
        Self {
            model,
            device,
            input_size,
        }
    }
}

impl<B: Backend> InferenceModel for InferenceBurnModel<B> {
    fn predict_move(&self, state: Vec<f32>) -> MazeAction {
        let tensor_data = TensorData::new(
            state,
            [
                1,
                self.input_size.depth,
                self.input_size.height,
                self.input_size.width,
            ],
        );
        let input = Tensor::from_data(tensor_data, &self.device);
        self.model
            .forward(BurnTensor(input))
            .0
            .argmax(3)
            .into_scalar()
            .to_u32()
            .into()
    }
}

impl<B: AutodiffBackend> Model<B> {
    pub fn optimize(
        &mut self,
        loss: BurnTensor<B, 4>,
        optimizer: &mut (impl Optimizer<Self, B> + Sized),
        lr: LearningRate,
    ) {
        let grads = loss.0.backward();
        let gradient_params = GradientsParams::from_grads(grads, self);
        *self = optimizer.step(lr, self.clone(), gradient_params);
    }
}

impl<B: AutodiffBackend, O: Optimizer<Self, B> + Sized> RLModel<B::Device, BurnTensor<B, 4>, O>
    for Model<B>
{
    fn init(
        path: Option<PathBuf>,
        backend: B::Device,
        feature_size: Dimensions3D,
        action_size: u32,
    ) -> RLResult<Self> {
        let model = Self::new(&backend, feature_size, action_size);
        match path {
            Some(path) => {
                let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                    .load(path, &backend)
                    .expect("Model could not be loaded.");
                Ok(model.load_record(record))
            }
            None => Ok(model),
        }
    }

    fn save_model(&self, path: PathBuf) -> RLResult<()> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.clone()
            .save_file(path, &recorder)
            .expect("Couldn't save the model.");
        Ok(())
    }

    fn predict(&self, observation: BurnTensor<B, 4>) -> BurnTensor<B, 4> {
        self.forward(observation)
    }

    fn optimize(&mut self, loss: BurnTensor<B, 4>, optimizer: &mut O, lr: f32, _time_step: usize) {
        Model::optimize(self, loss, optimizer, lr.into())
    }
}

impl<B: AutodiffBackend> DQNTargetModel for Model<B> {
    fn soft_update(&mut self, other: &Self, tau: f32) {
        soft_update_conv2d(&mut self.conv1, &other.conv1, tau);
        soft_update_conv2d(&mut self.conv2, &other.conv2, tau);
        soft_update_conv2d(&mut self.conv3, &other.conv3, tau);
        soft_update_fc(&mut self.fc1, &other.fc1, tau);
        soft_update_fc(&mut self.fc2, &other.fc2, tau);
    }
}

fn soft_update_conv2d<B: Backend>(
    conv: &mut nn::conv::Conv2d<B>,
    other: &nn::conv::Conv2d<B>,
    tau: f32,
) {
    let updated_weights = soft_update_param(conv.weight.val(), other.weight.val(), tau);
    let updated_bias = match (&conv.bias, &other.bias) {
        (Some(bias), Some(other_bias)) => {
            Some(soft_update_param(bias.val(), other_bias.val(), tau))
        }
        _ => None,
    };
    conv.weight = updated_weights;
    conv.bias = updated_bias;
}

fn soft_update_fc<B: Backend>(linear: &mut nn::Linear<B>, other: &nn::Linear<B>, tau: f32) {
    let updated_weights = soft_update_param(linear.weight.val(), other.weight.val(), tau);
    let updated_bias = match (&linear.bias, &other.bias) {
        (Some(bias), Some(other_bias)) => {
            Some(soft_update_param(bias.val(), other_bias.val(), tau))
        }
        _ => None,
    };
    linear.weight = updated_weights;
    linear.bias = updated_bias;
}

fn soft_update_param<const D: usize, B: Backend>(
    tensor: Tensor<B, D>,
    other: Tensor<B, D>,
    tau: f32,
) -> Param<Tensor<B, D>> {
    let updated_tensor = tensor * (1.0 - tau) + other * tau;
    let deteched_tensor = updated_tensor.detach();
    Param::initialized(ParamId::new(), deteched_tensor)
}

#[derive(Clone, Debug)]
pub struct BurnTensor<B: Backend, const D: usize>(pub Tensor<B, D>);

// Hack to use only the 4 dimensional tensors to simplify so dynamic tensor sizing sot that the
// burn tenssor can be used like with awesom_tensor
impl<B: Backend> RLTensor<B::Device> for BurnTensor<B, 4> {
    fn from_values(
        values: Vec<f32>,
        dims: Dimensions4D,
        backend: &B::Device,
        has_grad: bool,
    ) -> Self {
        let tensor_data = TensorData::new(values, dims);
        let tensor = Tensor::from_data(tensor_data, backend);
        let inner = match has_grad {
            true => tensor,
            false => tensor.no_grad(),
        };
        Self(inner)
    }

    fn to_values(&self) -> Vec<f32> {
        self.0
            .to_data()
            .to_vec()
            .expect("Couldn't read tensor data.")
    }

    fn dimensions(&self) -> Dimensions4D {
        let dims = self.0.dims();
        Dimensions4D::new(dims[0], dims[1], dims[2], dims[3])
    }

    fn arg_max(self, dim: usize) -> Self {
        Self(self.0.argmax(dim).float())
    }

    fn max_dim(self, dim: usize) -> Self {
        Self(self.0.max_dim(dim))
    }

    fn gather(self, dim: usize, indices: Self) -> Self {
        let indices_int = indices.0.int();
        Self(Tensor::gather(self.0, dim, indices_int))
    }

    fn add(self, rhs: Self) -> Self {
        Self(Tensor::add(self.0, rhs.0))
    }

    fn mul(self, rhs: Self) -> Self {
        Self(Tensor::mul(self.0, rhs.0))
    }

    fn add_scalar(self, rhs: f32) -> Self {
        Self(Tensor::add_scalar(self.0, rhs))
    }

    fn mul_scalar(self, rhs: f32) -> Self {
        Self(Tensor::mul_scalar(self.0, rhs))
    }

    fn detach(self) -> Self {
        Self(Tensor::detach(self.0))
    }

    fn loss(self, gt: Self, loss: RLLoss) -> Self {
        let loss = match loss {
            RLLoss::Mse => MseLoss::new().forward(self.0, gt.0, Reduction::Mean),
        };
        Self(loss.reshape([1, 1, 1, 1]))
    }
}
