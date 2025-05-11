use std::{cell::RefCell, rc::Rc};

use super::{
    model_config::{Conv2dConfig, InputConfig, PoolingConfig},
    optimizer::AdamOptimizer,
    tensor_error::{Result, TensorError},
};

use crate::machine_learning::dataloader::Dataloader;

use super::{
    backend::Backend,
    dimensions::{Dimensions2D, Dimensions3D, Dimensions4D},
    model_config::{
        ActivationConfig, ActivationType, DenseConfig, LayerConfig, NNModelConfig, Padding,
        PoolType,
    },
    tensor::Tensor,
    training_parameters::{TrainingHistory, TrainingParameters, TrainingStepStats},
};

pub enum ModelMode {
    Inference,
    Training(TrainingParameters),
}

/// Generic Model for different backends
pub struct NNModel<B: Backend> {
    batch_size: usize,
    input_size: Dimensions3D,
    output_size: Dimensions3D,
    backend: Rc<RefCell<B>>,
    input_layer: InputLayer,
    layers: Vec<Layer<B>>,
}

impl<B: Backend> NNModel<B> {
    pub fn from_config(
        config: NNModelConfig,
        backend: Rc<RefCell<B>>,
        batch_size: usize,
    ) -> Result<Self> {
        // Checking if the config is empty or the first layer is not input
        if config.layer_configs_ref().is_empty() {
            return Err(TensorError::EmptyModelConfig);
        }
        let input_layer = match config.layer_configs_ref().first().unwrap() {
            LayerConfig::Input(config) => InputLayer(config.dimensions),
            _ => return Err(TensorError::FaultyModelConfig(0)),
        };
        let input_size = config
            .layer_configs_ref()
            .first()
            .unwrap()
            .output_dimensions();
        let output_size = config
            .layer_configs_ref()
            .last()
            .unwrap()
            .output_dimensions();
        // let backend = Rc::new(RefCell::new(backend));
        let mut layers = Vec::with_capacity(config.layer_configs_ref().len());
        for layer_config in config.layer_configs().into_iter().skip(1) {
            let layer = match layer_config {
                // Handled separately
                LayerConfig::Input(_) => {
                    return Err(TensorError::FaultyModelConfig(0));
                }
                LayerConfig::Dense(config) => {
                    let weight_dimensions =
                        Dimensions4D::new(1, 1, config.layer_size, config.input_dimensions.width);
                    let weight_values = config.weights.into_iter().flatten().collect::<Vec<_>>();
                    let weights = Tensor::from_values(
                        backend.clone(),
                        weight_values,
                        weight_dimensions,
                        true,
                    )?;
                    let bias_dimensions = Dimensions4D::new(1, 1, 1, config.layer_size);
                    let biases =
                        Tensor::from_values(backend.clone(), config.bias, bias_dimensions, true)?;

                    Layer::Dense {
                        weights,
                        biases,
                        activation: config.activation,
                    }
                }
                LayerConfig::Conv2d(config) => {
                    let kernel_dimensions = config.filter_size;
                    let kernel_values = config.filters.into_iter().flatten().collect::<Vec<_>>();
                    let kernels = Tensor::from_values(
                        backend.clone(),
                        kernel_values,
                        kernel_dimensions,
                        true,
                    )?;
                    // each filter has one bias and this is saved to the depth dimension and
                    // broadcasted to batch, width and height
                    let bias_dimensions = Dimensions4D::new(1, config.filter_size.batch, 1, 1);
                    let biases =
                        Tensor::from_values(backend.clone(), config.bias, bias_dimensions, true)?;

                    let padding_y = kernel_dimensions.height / 2;
                    let padding_x = kernel_dimensions.width / 2;

                    Layer::Conv2d {
                        kernels,
                        biases,
                        activation: config.activation,
                        padding: Padding::Zero(padding_x, padding_y),
                    }
                }
                LayerConfig::Activation(config) => Layer::Activation(config.activation),
                LayerConfig::Pool(config) => Layer::Pool {
                    pool_dimensions: config.pool_dimensions,
                    pool_type: config.pooling,
                },
                LayerConfig::Flatten(_dimensions) => Layer::Flatten,
            };
            layers.push(layer);
        }
        Ok(Self {
            batch_size,
            input_size,
            output_size,
            backend,
            input_layer,
            layers,
        })
    }

    pub fn as_config(&self) -> Result<NNModelConfig> {
        let mut model_config = NNModelConfig::new();
        let mut last_output_dimensions = self.input_layer.0;
        let input_layer = LayerConfig::Input(InputConfig {
            dimensions: last_output_dimensions,
        });
        model_config.add_layer(input_layer);

        for layer in self.layers.iter() {
            let config = match layer {
                Layer::Dense {
                    weights,
                    biases,
                    activation,
                } => {
                    let layer_size = weights.dimensions().height;
                    let weights = weights.get_values()?;
                    let weights: Vec<Vec<f32>> = weights
                        .chunks(last_output_dimensions.size())
                        .map(|chunk| chunk.to_vec())
                        .collect();
                    let biases = biases.get_values()?;

                    LayerConfig::Dense(DenseConfig::from_initialized(
                        last_output_dimensions,
                        *activation,
                        layer_size,
                        weights,
                        biases,
                    ))
                }
                Layer::Conv2d {
                    kernels,
                    biases,
                    activation,
                    padding,
                } => {
                    let layer_size = kernels.dimensions().depth
                        * kernels.dimensions().height
                        * kernels.dimensions().width;
                    let weights = kernels.get_values()?;
                    let weights: Vec<Vec<f32>> = weights
                        .chunks(layer_size)
                        .map(|chunk| chunk.to_vec())
                        .collect();

                    let biases = biases.get_values()?;
                    let padding = match padding {
                        Padding::Zero(_, _) => true,
                        Padding::None => false,
                    };
                    LayerConfig::Conv2d(Conv2dConfig::from_initialized(
                        last_output_dimensions,
                        *activation,
                        kernels.dimensions(),
                        kernels.dimensions().batch,
                        weights,
                        biases,
                        padding,
                    ))
                }
                Layer::Pool {
                    pool_dimensions,
                    pool_type,
                } => LayerConfig::Pool(PoolingConfig::new(
                    *pool_type,
                    last_output_dimensions,
                    *pool_dimensions,
                )),
                Layer::Flatten => LayerConfig::Flatten(Dimensions3D::new(
                    1,
                    1,
                    last_output_dimensions.depth
                        * last_output_dimensions.height
                        * last_output_dimensions.width,
                )),
                Layer::Activation(activation_type) => LayerConfig::Activation(ActivationConfig {
                    input_dimensions: last_output_dimensions,
                    activation: *activation_type,
                }),
            };
            last_output_dimensions = config.output_dimensions();
            model_config.add_layer(config);
        }
        Ok(model_config)
    }

    pub fn train(
        &mut self,
        training_parameters: TrainingParameters,
        mut optimizer: AdamOptimizer<B>,
        train_dataset: &mut dyn Dataloader,
        mut validation_dataset: Option<&mut dyn Dataloader>,
    ) -> Result<TrainingHistory> {
        let mut history = TrainingHistory::default();
        let mut time_step = 1;
        for epoch in 1..=training_parameters.epochs {
            let sample_count = train_dataset.size();
            let iters = sample_count.div_ceil(self.batch_size);
            for i in 0..iters {
                let (batch, gt, amount) = train_dataset.get(self.batch_size);
                let input_dimensions = Dimensions4D::new(
                    amount,
                    self.input_size.depth,
                    self.input_size.height,
                    self.input_size.width,
                );
                let output_dimensions = Dimensions4D::new(
                    amount,
                    self.output_size.depth,
                    self.output_size.height,
                    self.output_size.width,
                );
                let batch_tensor =
                    Tensor::from_values(self.backend.clone(), batch, input_dimensions, false)?;
                let gt_tensor =
                    Tensor::from_values(self.backend.clone(), gt, output_dimensions, false)?;
                let pred = self.forward(batch_tensor)?;
                let training_step = self.backward(
                    pred,
                    gt_tensor,
                    &training_parameters,
                    &mut optimizer,
                    time_step,
                )?;
                time_step += 1;
                println!("epoch: {epoch}, sample: {i}/{iters}, {training_step}");
                history.training_step_stats.push(training_step);
            }
            if let Some(test_dataset) = validation_dataset.as_deref_mut() {
                let (_, acc) = self.predict(test_dataset).unwrap();
                println!("epoch: {epoch}, validation acc: {acc}");
            }
        }

        Ok(history)
    }

    pub fn predict(&self, dataset: &mut dyn Dataloader) -> Result<(Vec<Vec<f32>>, f32)> {
        let sample_count = dataset.size();
        let iters = sample_count.div_ceil(self.batch_size);
        let mut predictions = Vec::with_capacity(dataset.size());
        let mut gts = Vec::with_capacity(dataset.size());
        for _i in 0..iters {
            let (batch, gt, amount) = dataset.get(self.batch_size);
            let input_dimensions = Dimensions4D::new(
                amount,
                self.input_size.depth,
                self.input_size.height,
                self.input_size.width,
            );
            let batch_tensor =
                Tensor::from_values(self.backend.clone(), batch, input_dimensions, false)?;
            let prediction_tensor = self.forward(batch_tensor)?;
            let prediction = prediction_tensor.get_values()?;
            let chunk_size = (prediction.len() + amount - 1) / amount; // Calculate chunk size
            for single in prediction.chunks(chunk_size) {
                predictions.push(single.to_vec());
            }
            for single in gt.chunks(chunk_size) {
                gts.push(single.to_vec());
            }
        }
        let mut correct_predictions = 0.0;
        // don't evaluate the zero padded
        for (gt, prediction) in gts.iter().zip(predictions.iter()).take(dataset.size()) {
            match prediction.len() {
                1 => {
                    if prediction[0].round() == (*gt)[0] {
                        correct_predictions += 1.0;
                    }
                }
                _ => {
                    let prediction_max = prediction
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(index, _)| index);
                    let gt_max = (*gt)
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(index, _)| index);
                    if prediction_max == gt_max {
                        correct_predictions += 1.0;
                    }
                }
            }
        }
        let acc = correct_predictions / sample_count as f32;
        Ok((predictions, acc))
    }

    pub fn forward(&self, input: Tensor<B>) -> Result<Tensor<B>> {
        let mut last_output = input;

        for layer in self.layers.iter() {
            match layer {
                Layer::Dense {
                    weights,
                    biases,
                    activation,
                } => {
                    let lhs_dimensions = Dimensions4D::new(
                        1,
                        1,
                        last_output.dimensions().batch,
                        last_output.dimensions().width,
                    );
                    let lhs = last_output.reshape(lhs_dimensions)?;
                    let matmul_result = lhs.matmul(weights, true)?;
                    let dense_output_dimensions = Dimensions4D::new(
                        last_output.dimensions().batch,
                        1,
                        1,
                        matmul_result.dimensions().width,
                    );
                    let dense_reshaped = matmul_result.reshape(dense_output_dimensions)?;
                    let bias_add_result = dense_reshaped.add(biases)?;
                    last_output = bias_add_result.clone();
                    if let Some(activation) = activation {
                        let activation_result = bias_add_result.activation(*activation)?;
                        last_output = activation_result;
                    };
                }
                Layer::Conv2d {
                    kernels,
                    biases,
                    activation,
                    padding,
                } => {
                    let convolved = last_output.conv2d(kernels, padding)?;
                    let bias_add_result = convolved.add(biases)?;
                    last_output = bias_add_result.clone();
                    if let Some(activation) = activation {
                        let activation_result = bias_add_result.activation(*activation)?;
                        last_output = activation_result;
                    }
                }
                Layer::Pool {
                    pool_dimensions,
                    pool_type,
                } => {
                    let pooled = last_output.pool(*pool_type, *pool_dimensions)?;
                    last_output = pooled;
                }
                Layer::Flatten => {
                    let flattened = last_output.flatten()?;
                    last_output = flattened;
                }
                Layer::Activation(activation) => {
                    let activation_result = last_output.activation(*activation)?;
                    last_output = activation_result;
                }
            };
        }
        Ok(last_output)
    }

    pub fn optimize(
        &mut self,
        loss: &Tensor<B>,
        lr: f64,
        optimizer: &mut AdamOptimizer<B>,
        time_step: usize,
    ) -> Result<()> {
        let gradients = loss.backward()?;
        for layer in self.layers.iter_mut() {
            for param in layer.get_params() {
                let param_grad = gradients
                    .get(param.id())
                    .ok_or(TensorError::MissingGradient {
                        tensor_id: param.id(),
                    })?;
                optimizer.step(param, &param_grad, time_step, lr)?;
            }
        }
        Ok(())
    }

    pub fn backward(
        &mut self,
        train_pred: Tensor<B>,
        gt: Tensor<B>,
        training_parameters: &TrainingParameters,
        optimizer: &mut AdamOptimizer<B>,
        time_step: usize,
    ) -> Result<TrainingStepStats> {
        let loss = training_parameters.loss_function.loss(&train_pred, &gt)?;

        // optimize
        self.optimize(
            &loss,
            training_parameters.learning_rate,
            optimizer,
            time_step,
        )?;
        let loss = *loss
            .sum(None)?
            .get_values()?
            .first()
            .ok_or(TensorError::LossComputeError)?;
        let step_stats = TrainingStepStats::new((loss).into(), 0.0);
        Ok(step_stats)
    }

    pub fn backend(&self) -> Rc<RefCell<B>> {
        self.backend.clone()
    }

    pub fn layers(&self) -> &Vec<Layer<B>> {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> &mut Vec<Layer<B>> {
        &mut self.layers
    }

    pub fn input_size(&self) -> Dimensions3D {
        self.input_size
    }
}

pub struct InputLayer(Dimensions3D);

pub enum Layer<B: Backend> {
    Dense {
        weights: Tensor<B>,
        biases: Tensor<B>,
        activation: Option<ActivationType>,
    },
    Conv2d {
        kernels: Tensor<B>,
        biases: Tensor<B>,
        activation: Option<ActivationType>,
        padding: Padding,
    },
    Pool {
        pool_dimensions: Dimensions2D,
        pool_type: PoolType,
    },
    Flatten,
    Activation(ActivationType),
}

impl<B: Backend> Layer<B> {
    pub fn get_params(&mut self) -> Vec<&mut Tensor<B>> {
        match self {
            Layer::Dense {
                weights,
                biases,
                activation: _,
            } => vec![weights, biases],
            Layer::Conv2d {
                kernels,
                biases,
                activation: _,
                padding: _,
            } => vec![kernels, biases],
            Layer::Pool {
                pool_dimensions: _,
                pool_type: _,
            } => vec![],
            Layer::Flatten => vec![],
            Layer::Activation(_) => vec![],
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::machine_learning::{
        compare_outputs, linspace,
        nn::{
            backends::{cpu_backend::CpuBackend, wgpu_backend::WgpuBackend},
            nn_model_builder::NNModelBuilder,
        },
        print_matrix,
    };

    use super::*;

    #[test]
    fn test_nn_model() {
        let builder = NNModelBuilder::new(10);
        let activation = builder
            .input(3, 10, 10)
            .conv2d(Some(ActivationType::Relu), (3, 3), 10, true)
            // TODO: check pooling dimensions
            .pool2d(PoolType::MaxPool2D, Dimensions2D::new(2, 2))
            .flatten()
            .dense(Some(ActivationType::Sigmoid), 2)
            .activation(ActivationType::Sigmoid);
        let config = activation.get_config();

        let backend = CpuBackend {};
        let cpu_model =
            NNModel::from_config(config.clone(), Rc::new(RefCell::new(backend)), 10).unwrap();
        let input_dimensions = Dimensions4D::new(
            cpu_model.batch_size,
            cpu_model.input_size.depth,
            cpu_model.input_size.height,
            cpu_model.input_size.width,
        );
        let input = linspace(-1.0, 1.0, input_dimensions.size());
        let input_tensor = Tensor::from_values(
            cpu_model.backend.clone(),
            input.clone(),
            input_dimensions,
            true,
        )
        .unwrap();

        let cpu_output = cpu_model.forward(input_tensor).unwrap();
        let output_values_cpu = cpu_output.get_values().unwrap();

        let backend = WgpuBackend::default();
        let gpu_model = NNModel::from_config(config, Rc::new(RefCell::new(backend)), 10).unwrap();
        let input_tensor = Tensor::from_values(
            gpu_model.backend.clone(),
            input.clone(),
            input_dimensions,
            true,
        )
        .unwrap();
        let gpu_output = gpu_model.forward(input_tensor).unwrap();
        let output_values_gpu = gpu_output.get_values().unwrap();

        println!("cpu_output:");
        print_matrix(&output_values_cpu, cpu_output.dimensions());
        println!("gpu_output:");
        print_matrix(&output_values_gpu, gpu_output.dimensions());
        assert!(compare_outputs(&output_values_cpu, &output_values_gpu));
    }

    #[test]
    fn test_model_saving() {
        let batch_size = 128;
        let builder = NNModelBuilder::new(batch_size);
        let config = builder
            .input(1, 28, 28)
            .conv2d(Some(ActivationType::LeakyRelu(1.0)), (3, 3), 3, true)
            .pool2d(PoolType::MaxPool2D, Dimensions2D::new(2, 2))
            .conv2d(Some(ActivationType::LeakyRelu(1.0)), (3, 3), 1, true)
            .pool2d(PoolType::MaxPool2D, Dimensions2D::new(2, 2))
            .flatten()
            .dense(Some(ActivationType::LeakyRelu(1.0)), 10)
            .dense(Some(ActivationType::Sigmoid), 2)
            .get_config();
        config.save("mnist_before.json").unwrap();

        let backend = WgpuBackend::default();
        let model =
            NNModel::from_config(config.clone(), Rc::new(RefCell::new(backend)), batch_size)
                .unwrap();
        model.as_config().unwrap().save("mnist_after.json").unwrap();
    }
}
