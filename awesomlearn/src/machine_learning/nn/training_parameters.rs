use serde::{Deserialize, Serialize};

use super::{backend::Backend, dimensions::Dimensions4D, tensor::Tensor, tensor_error::Result};

pub struct TrainingParameters {
    pub epochs: usize,
    pub loss_function: LossFunction,
    pub learning_rate: f64,
    // pub dataset: Box<dyn Dataloader>,
}

#[derive(Serialize, Deserialize)]
pub enum LossFunction {
    MSE,
    MAE,
    CrossEntropy,
}

impl LossFunction {
    pub fn loss<B: Backend>(&self, pred: &Tensor<B>, gt: &Tensor<B>) -> Result<Tensor<B>> {
        match self {
            LossFunction::MSE => Ok(pred.mse(gt)?),
            LossFunction::CrossEntropy => {
                // let softmax = pred.softmax()?;
                let softmax = pred;
                let minus_one = Tensor::from_values(
                    softmax.backend(),
                    vec![-1.0; 1],
                    Dimensions4D::new(1, 1, 1, 1),
                    false,
                )?;

                let div_value = vec![pred.dimensions().batch as f32];
                let div_tensor = Tensor::from_values(
                    softmax.backend(),
                    div_value,
                    Dimensions4D::new(1, 1, 1, 1),
                    false,
                )?;
                let log_soft = softmax.ln()?;
                let class_log = gt.mul(&log_soft)?;
                let summed = class_log.sum(Some(&[1, 2, 3]))?;
                let mean = summed.sum(None)?.div(&div_tensor)?;
                let negated = minus_one.mul(&mean)?;
                Ok(negated)
            }
            LossFunction::MAE => todo!(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct TrainingHistory {
    pub training_step_stats: Vec<TrainingStepStats>,
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self {
            training_step_stats: Vec::new(),
        }
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize, Deserialize)]
pub struct TrainingStepStats {
    loss: f64,
    grad: f64,
}

impl std::fmt::Display for TrainingStepStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "loss: {}", self.loss)
    }
}

impl TrainingStepStats {
    pub fn new(loss: f64, grad: f64) -> Self {
        Self { loss, grad }
    }
}
