use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use super::backend::Backend;
use super::dimensions::{Dimensions2D, Dimensions3D};
use super::model_config::{
    ActivationConfig, ActivationType, Conv2dConfig, DenseConfig, InputConfig, LayerConfig,
    NNModelConfig, PoolType, PoolingConfig,
};

use super::nn_model::NNModel;
use super::tensor_error::Result;

pub struct InputNotInitialized;
pub struct InputInitialized;

/// Each backend needs to implement their own build function for the builder
#[derive(Default)]
pub struct NNModelBuilder<HasInput = InputNotInitialized> {
    pub model_config: NNModelConfig,
    pub batch_size: usize,
    has_input: PhantomData<HasInput>,
}

impl NNModelBuilder<InputNotInitialized> {
    /// Empty builder
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            model_config: NNModelConfig::default(),
            has_input: PhantomData,
        }
    }

    pub fn input(
        mut self,
        depth: usize,
        height: usize,
        width: usize,
    ) -> NNModelBuilder<InputInitialized> {
        self.model_config
            .add_layer(LayerConfig::Input(InputConfig::new(depth, height, width)));

        NNModelBuilder {
            batch_size: self.batch_size,
            model_config: self.model_config,
            has_input: PhantomData,
        }
    }
}

impl NNModelBuilder<InputInitialized> {
    pub fn dense(mut self, activation: Option<ActivationType>, layer_size: usize) -> Self {
        // Phantomdata for the input initialized guarantees that there is last layer present
        let input_dimensions = self.model_config.output_dimensions().unwrap();
        let dense = LayerConfig::Dense(DenseConfig::new(input_dimensions, activation, layer_size));
        self.model_config.add_layer(dense);
        self
    }

    pub fn conv2d(
        mut self,
        activation: Option<ActivationType>,
        filter_size: (usize, usize),
        filter_count: usize,
        zero_padding: bool,
    ) -> Self {
        // Phantomdata for the input initialized guarantees that there is last layer present
        let input_dimensions = self.model_config.output_dimensions().unwrap();
        let conv_layer = LayerConfig::Conv2d(Conv2dConfig::new(
            input_dimensions,
            activation,
            filter_size,
            filter_count,
            zero_padding,
        ));
        self.model_config.add_layer(conv_layer);
        self
    }

    pub fn pool2d(mut self, pooling: PoolType, pool_dimensions: Dimensions2D) -> Self {
        let input_dimensions = self.model_config.output_dimensions().unwrap();
        let pool2d_layer = LayerConfig::Pool(PoolingConfig::new(
            pooling,
            input_dimensions,
            pool_dimensions,
        ));
        self.model_config.add_layer(pool2d_layer);
        self
    }

    pub fn activation(mut self, activation: ActivationType) -> Self {
        let input_dimensions = self.model_config.output_dimensions().unwrap();
        let activation_layer =
            LayerConfig::Activation(ActivationConfig::new(input_dimensions, activation));
        self.model_config.add_layer(activation_layer);
        self
    }

    pub fn flatten(mut self) -> Self {
        let input_dimensions = self.model_config.output_dimensions().unwrap();
        let flatten_layer = LayerConfig::Flatten(Dimensions3D::new(
            1,
            1,
            input_dimensions.depth * input_dimensions.height * input_dimensions.width,
        ));
        self.model_config.add_layer(flatten_layer);
        self
    }

    /// Returns the model built with the specified backend. Might return Err if unsupported layer
    /// type for the backend.
    pub fn build<B: Backend>(self, backend: B, batch_size: usize) -> Result<NNModel<B>> {
        let backend = Rc::new(RefCell::new(backend));
        let model = NNModel::from_config(self.model_config, backend, batch_size)?;
        // Phantomdata for the input initialized guarantees that the first layer is the input
        Ok(model)
    }

    pub fn get_config(&self) -> NNModelConfig {
        self.model_config.clone()
    }
}
