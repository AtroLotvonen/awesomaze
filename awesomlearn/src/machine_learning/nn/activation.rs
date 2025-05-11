use serde::{Deserialize, Serialize};

// Serde could be used to serialize this
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum Activation {
    ReLU,
    Sigmoid,
}

impl Activation {
    pub fn forward(&self, x: f32) -> f32 {
        match self {
            Activation::Sigmoid => Self::sigmoid(x),
            Activation::ReLU => f32::max(0.0, x),
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn backward(&self, x: f32) -> f32 {
        match self {
            Activation::ReLU => match x {
                x if x > 0.0 => 1.0,
                _ => 0.0,
            },
            Activation::Sigmoid => {
                let sigma = Self::sigmoid(x);
                sigma * (1.0 - sigma)
            }
        }
    }
}
