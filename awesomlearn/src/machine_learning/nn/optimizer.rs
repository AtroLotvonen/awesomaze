use std::{cell::RefCell, collections::HashMap, rc::Rc};

use super::{backend::Backend, dimensions::Dimensions4D, tensor::Tensor, tensor_error::Result};

pub trait Optimizer<B: Backend> {
    fn update_grad(&self) -> Tensor<B>;
}

pub struct AdamOptimizer<B: Backend> {
    beta_1: Tensor<B>,
    beta_2: Tensor<B>,
    epsilon: Tensor<B>,
    weight_decay: Option<Tensor<B>>,
    clip_norm: Option<Tensor<B>>,
    one_tensor: Tensor<B>,
    half_tensor: Tensor<B>,
    first_moments: HashMap<usize, Tensor<B>>,
    second_moments: HashMap<usize, Tensor<B>>,
    backend: Rc<RefCell<B>>,
}

impl<B: Backend> AdamOptimizer<B> {
    pub fn new(
        backend: Rc<RefCell<B>>,
        beta_1: f64,
        beta_2: f64,
        epsilon: f64,
        weight_decay: Option<f64>,
        clip_norm: Option<f64>,
    ) -> Result<Self> {
        let beta_1 = Tensor::from_values(
            backend.clone(),
            vec![beta_1 as f32],
            Dimensions4D::new(1, 1, 1, 1),
            false,
        )?;
        let beta_2 = Tensor::from_values(
            backend.clone(),
            vec![beta_2 as f32],
            Dimensions4D::new(1, 1, 1, 1),
            false,
        )?;
        let epsilon = Tensor::from_values(
            backend.clone(),
            vec![epsilon as f32],
            Dimensions4D::new(1, 1, 1, 1),
            false,
        )?;
        let one_tensor = Tensor::from_values(
            backend.clone(),
            vec![1.0],
            Dimensions4D::new(1, 1, 1, 1),
            false,
        )?;
        let half_tensor = Tensor::from_values(
            backend.clone(),
            vec![0.5],
            Dimensions4D::new(1, 1, 1, 1),
            false,
        )?;
        let weight_decay = if let Some(weight_decay) = weight_decay {
            let tensor = Tensor::from_values(
                backend.clone(),
                vec![weight_decay as f32],
                Dimensions4D::new(1, 1, 1, 1),
                false,
            )?;

            Some(tensor)
        } else {
            None
        };
        let clip_norm = if let Some(clip_norm) = clip_norm {
            let tensor = Tensor::from_values(
                backend.clone(),
                vec![clip_norm as f32],
                Dimensions4D::new(1, 1, 1, 1),
                false,
            )?;

            Some(tensor)
        } else {
            None
        };
        Ok(Self {
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            clip_norm,
            one_tensor,
            half_tensor,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            backend: backend.clone(),
        })
    }

    pub fn step(
        &mut self,
        parameter: &mut Tensor<B>,
        grad: &Tensor<B>,
        time_step: usize,
        learning_rate: f64,
    ) -> Result<()> {
        let learning_rate_tensor = Tensor::from_values(
            self.backend.clone(),
            vec![learning_rate as f32],
            Dimensions4D::new(1, 1, 1, 1),
            false,
        )?;
        // Check if moments exist, if not init to zero
        let first_moment = match self.first_moments.remove(&parameter.id()) {
            Some(first_moment) => first_moment,
            None => Tensor::zeros(self.backend.clone(), Dimensions4D::new(1, 1, 1, 1), false)?,
        };
        let second_moment = match self.second_moments.remove(&parameter.id()) {
            Some(second_moment) => second_moment,
            None => Tensor::zeros(self.backend.clone(), Dimensions4D::new(1, 1, 1, 1), false)?,
        };
        // gradient_clipping
        let grad = if let Some(clip_norm) = &self.clip_norm {
            let grad_norm = grad.squared()?.sum(None)?.pow(&self.half_tensor)?;
            let greater_than = grad_norm.greater_than(clip_norm)?;
            let scaled_gradient = grad.mul(&clip_norm.div(&grad_norm)?)?;
            greater_than
                .mul(&scaled_gradient)?
                .add(&self.one_tensor.sub(&greater_than)?.mul(grad)?)?
        } else {
            grad.clone()
        };

        // update moments
        let first_moment = self
            .beta_1
            .mul(&first_moment)?
            .add(&self.one_tensor.sub(&self.beta_1)?.mul(&grad)?)?
            .detach();
        let grad_squared = grad.squared()?;
        let second_moment = self
            .beta_2
            .mul(&second_moment)?
            .add(&self.one_tensor.sub(&self.beta_2)?.mul(&grad_squared)?)?
            .detach();

        // Do bias correction
        let time_step_tensor = Tensor::from_values(
            self.backend.clone(),
            vec![time_step as f32],
            Dimensions4D::new(1, 1, 1, 1),
            false,
        )?;
        let beta_1t = self.beta_1.pow(&time_step_tensor)?;
        let beta_2t = self.beta_2.pow(&time_step_tensor)?;

        let first_moment_corrected = first_moment.div(&self.one_tensor.sub(&beta_1t)?)?;
        let second_moment_corrected = second_moment.div(&self.one_tensor.sub(&beta_2t)?)?;

        // update gradient
        let updated_grad = learning_rate_tensor.mul(
            &first_moment_corrected.div(
                &second_moment_corrected
                    .pow(&self.half_tensor)?
                    .add(&self.epsilon)?,
            )?,
        )?;

        // Update param
        let new_param = parameter.sub(&updated_grad)?;
        let mut new_param = if let Some(weight_decay) = &self.weight_decay {
            // weight decay
            let weight_decay = learning_rate_tensor.mul(&parameter.mul(weight_decay)?)?;
            new_param.sub(&weight_decay)?
        } else {
            new_param
        };
        new_param.detach();
        parameter.replace_storage(new_param);
        self.first_moments.insert(parameter.id(), first_moment);
        self.second_moments.insert(parameter.id(), second_moment);
        Ok(())
    }
}
