use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};

use super::{
    dimensions::{Dimensions2D, Dimensions3D, Dimensions4D},
    tensor_error::{Result, TensorError},
    weight_initialization::he_uniform,
};

/// The configuration to represent a neural network model. This is used to build a new model for
/// different backends so this is backend independent.
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct NNModelConfig {
    layer_configs: Vec<LayerConfig>,
}

impl NNModelConfig {
    pub fn new() -> Self {
        Self {
            layer_configs: Vec::new(),
        }
    }

    pub fn add_layer(&mut self, layer_config: LayerConfig) {
        self.layer_configs.push(layer_config);
    }

    pub fn new_from_layers(layer_configs: Vec<LayerConfig>) -> Self {
        Self { layer_configs }
    }

    pub fn layer_configs_ref(&self) -> &Vec<LayerConfig> {
        &self.layer_configs
    }

    pub fn layer_configs(self) -> Vec<LayerConfig> {
        self.layer_configs
    }

    pub fn output_dimensions(&self) -> Result<Dimensions3D> {
        match self.layer_configs_ref().len() {
            0 => Err(TensorError::EmptyModelConfig),
            _ => Ok(self.layer_configs_ref().last().unwrap().output_dimensions()),
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let model_str = serde_json::to_string_pretty(self)?;
        let mut file = File::create(path)?;
        file.write_all(model_str.as_bytes())?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut config_str = String::new();
        file.read_to_string(&mut config_str)?;
        let config: NNModelConfig = serde_json::from_str(&config_str)?;
        Ok(config)
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum LayerConfig {
    Dense(DenseConfig),
    Conv2d(Conv2dConfig),
    Activation(ActivationConfig),
    Input(InputConfig),
    Pool(PoolingConfig),
    Flatten(Dimensions3D),
}

impl LayerConfig {
    pub fn output_dimensions(&self) -> Dimensions3D {
        match self {
            LayerConfig::Dense(dense) => dense.output_dimensions(),
            LayerConfig::Conv2d(conv2d) => conv2d.output_dimensions(),
            LayerConfig::Activation(activation) => activation.output_dimensions(),
            LayerConfig::Input(input) => input.dimensions(),
            LayerConfig::Pool(pool) => pool.output_size(),
            LayerConfig::Flatten(dimensions) => Dimensions3D::new(
                1,
                1,
                dimensions.depth * dimensions.height * dimensions.width,
            ),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DenseConfig {
    pub activation: Option<ActivationType>,
    pub layer_size: usize,
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
    pub input_dimensions: Dimensions3D,
}

impl DenseConfig {
    pub fn new(
        input_dimensions: Dimensions3D,
        activation: Option<ActivationType>,
        layer_size: usize,
    ) -> Self {
        let weights = (0..layer_size)
            .map(|_| he_uniform(input_dimensions.size()))
            .collect();
        let bias: Vec<f32> = vec![0.0; layer_size];
        Self {
            activation,
            layer_size,
            weights,
            bias,
            input_dimensions,
        }
    }

    pub fn from_initialized(
        input_dimensions: Dimensions3D,
        activation: Option<ActivationType>,
        layer_size: usize,
        weights: Vec<Vec<f32>>,
        bias: Vec<f32>,
    ) -> Self {
        Self {
            activation,
            layer_size,
            weights,
            bias,
            input_dimensions,
        }
    }

    pub fn output_dimensions(&self) -> Dimensions3D {
        Dimensions3D::new(1, 1, self.layer_size)
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Conv2dConfig {
    pub input_dimensions: Dimensions3D,
    pub activation: Option<ActivationType>,
    pub filter_size: Dimensions4D,
    pub filter_count: usize,
    pub filters: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
    pub zero_padding: bool,
}

impl Conv2dConfig {
    pub fn new(
        input_dimensions: Dimensions3D,
        activation: Option<ActivationType>,
        filter_size: (usize, usize),
        filter_count: usize,
        zero_padding: bool,
    ) -> Self {
        let filter_size = Dimensions3D::new(input_dimensions.depth, filter_size.1, filter_size.0);
        let filters = (0..filter_count)
            .map(|_| he_uniform(filter_size.size()))
            .collect();
        let bias: Vec<f32> = vec![0.0; filter_count];
        let filter_size = Dimensions4D::new(
            filter_count,
            input_dimensions.depth,
            filter_size.height,
            filter_size.width,
        );
        Self {
            activation,
            input_dimensions,
            filter_size,
            filter_count,
            filters,
            bias,
            zero_padding,
        }
    }

    pub fn from_initialized(
        input_dimensions: Dimensions3D,
        activation: Option<ActivationType>,
        filter_size: Dimensions4D,
        filter_count: usize,
        filters: Vec<Vec<f32>>,
        bias: Vec<f32>,
        zero_padding: bool,
    ) -> Self {
        Self {
            activation,
            input_dimensions,
            filter_size,
            filter_count,
            filters,
            bias,
            zero_padding,
        }
    }

    /// Is always only width as the dense layer flattens the input
    pub fn output_dimensions(&self) -> Dimensions3D {
        match self.zero_padding {
            true => Dimensions3D::new(
                self.filter_count,
                self.input_dimensions.width,
                self.input_dimensions.height,
            ),
            false => Dimensions3D::new(
                self.filter_count,
                self.input_dimensions.width - (self.filter_size.width / 2) * 2,
                self.input_dimensions.height - (self.filter_size.height / 2) * 2,
            ),
        }
    }
}
/// Not a tensor - no batch size
#[derive(Serialize, Deserialize, Clone)]
pub struct InputConfig {
    pub dimensions: Dimensions3D,
}

impl InputConfig {
    pub fn new(depth: usize, height: usize, width: usize) -> Self {
        let dimensions = Dimensions3D {
            depth,
            height,
            width,
        };
        Self { dimensions }
    }

    pub fn dimensions(&self) -> Dimensions3D {
        self.dimensions
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
pub enum PoolType {
    MaxPool2D,
    AvgPool2D,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PoolingConfig {
    pub pooling: PoolType,
    pub input_dimensions: Dimensions3D,
    pub pool_dimensions: Dimensions2D,
}

impl PoolingConfig {
    pub fn new(
        pooling: PoolType,
        input_dimensions: Dimensions3D,
        pool_dimensions: Dimensions2D,
    ) -> Self {
        Self {
            pooling,
            input_dimensions,
            pool_dimensions,
        }
    }
    pub fn output_size(&self) -> Dimensions3D {
        Dimensions3D::new(
            self.input_dimensions.depth,
            self.input_dimensions.width / self.pool_dimensions.width,
            self.input_dimensions.height / self.pool_dimensions.height,
        )
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
pub enum ActivationType {
    Relu,
    LeakyRelu(f32),
    Sigmoid,
    Softmax,
}

#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
pub enum Padding {
    Zero(usize, usize),
    None,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ActivationConfig {
    pub input_dimensions: Dimensions3D,
    pub activation: ActivationType,
}

impl ActivationConfig {
    pub fn new(input_dimensions: Dimensions3D, activation: ActivationType) -> Self {
        Self {
            input_dimensions,
            activation,
        }
    }

    pub fn output_dimensions(&self) -> Dimensions3D {
        self.input_dimensions
    }
}
