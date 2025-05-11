use super::{
    backend::Backend,
    dimensions::Dimensions4D,
    model_config::{ActivationType, PoolType},
    tensor::{Operation, Tensor},
};
use std::collections::{HashMap, HashSet, VecDeque};

pub struct GradientStorage<B: Backend> {
    grad_storage: HashMap<usize, Tensor<B>>,
}

use super::tensor_error::Result;

impl<B: Backend> GradientStorage<B> {
    pub fn new() -> Self {
        Self {
            grad_storage: HashMap::new(),
        }
    }

    pub fn insert_or_add(&mut self, tensor: &Tensor<B>, new_grad: Tensor<B>) -> Result<()> {
        let tensor_id = tensor.id();
        self.add_grad(tensor_id, new_grad)
    }

    fn add_grad(&mut self, tensor_id: usize, mut new_grad: Tensor<B>) -> Result<()> {
        match self.grad_storage.get(&tensor_id) {
            Some(old_grad) => {
                let mut added_grad = old_grad.add(&new_grad)?;
                let detached_grad = added_grad.detach();
                self.grad_storage.insert(tensor_id, detached_grad);
            }
            None => {
                let detached_grad = new_grad.detach();
                self.grad_storage.insert(tensor_id, detached_grad);
            }
        }
        Ok(())
    }

    pub fn get(&self, tensor_id: usize) -> Option<Tensor<B>> {
        self.grad_storage.get(&tensor_id).cloned()
    }

    pub fn remove(&mut self, tensor_id: usize) -> Option<Tensor<B>> {
        self.grad_storage.remove(&tensor_id)
    }

    pub fn gradient_accumulation(&mut self, other: &Self) -> Result<()> {
        let mut acc_grads = Vec::with_capacity(self.grad_storage.len());
        for &tensor_id in self.grad_storage.keys() {
            if let Some(other_grad) = other.get(tensor_id) {
                acc_grads.push((tensor_id, other_grad));
            }
        }
        for (tensor_id, new_grad) in acc_grads {
            self.add_grad(tensor_id, new_grad)?;
        }
        Ok(())
    }
}

impl<B: Backend> Default for GradientStorage<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Tensor<B> {
    fn get_graph_tensors_sorted(&self) -> Vec<Tensor<B>> {
        let mut tensor_stack = Vec::new();
        let mut tensor_queue = VecDeque::new();
        tensor_queue.push_back(self);
        while let Some(tensor) = tensor_queue.pop_front() {
            if tensor.has_grad() {
                tensor_stack.push(tensor);
            }
            if let Some(op) = tensor.back_propagation_op() {
                match op {
                    Operation::Add(lhs, rhs) => {
                        tensor_queue.extend([&lhs, &rhs]);
                    }
                    Operation::Sub(lhs, rhs) => {
                        tensor_queue.extend([&lhs, &rhs]);
                    }
                    Operation::Mul(lhs, rhs) => {
                        tensor_queue.extend([&lhs, &rhs]);
                    }
                    Operation::Div(lhs, rhs) => {
                        tensor_queue.extend([&lhs, &rhs]);
                    }
                    Operation::Pow(lhs, rhs) => {
                        tensor_queue.extend([&lhs, &rhs]);
                    }
                    Operation::Squared(input) => {
                        tensor_queue.push_back(input);
                    }
                    Operation::Exp(input) => {
                        tensor_queue.push_back(input);
                    }
                    Operation::Ln(input) => {
                        tensor_queue.push_back(input);
                    }
                    Operation::ReduceSum(input) => {
                        tensor_queue.push_back(input);
                    }
                    Operation::Broadcast(input, _) => {
                        tensor_queue.push_back(input);
                    }
                    Operation::Matmul {
                        lhs,
                        rhs,
                        is_rhs_transposed: _,
                    } => {
                        tensor_queue.extend([&lhs, &rhs]);
                    }
                    Operation::Conv2d { input, kernel, .. } => {
                        tensor_queue.extend([&input, &kernel]);
                    }
                    Operation::Transpose(input, _) => {
                        tensor_queue.push_back(input);
                    }
                    Operation::Reshape(input) => {
                        tensor_queue.push_back(input);
                    }
                    Operation::Activation(input, _) => {
                        tensor_queue.push_back(input);
                    }
                    Operation::Pool(input, _, _) => {
                        tensor_queue.push_back(input);
                    }
                    Operation::Upsample(tensor, _upsample_dimensions) => {
                        tensor_queue.push_back(tensor)
                    }
                    Operation::ConvTranspose2d {
                        input,
                        kernel: _,
                        input_padding: _,
                    } => tensor_queue.push_back(input),
                    Operation::CustomUnary(input, _) => {
                        tensor_queue.push_back(input);
                    }
                    Operation::CustomBinary(lhs, rhs, _) => {
                        tensor_queue.extend([&lhs, &rhs]);
                    }
                    Operation::Cmp(_, _, _) => {}
                    Operation::ArgMax(input, _) => {
                        tensor_queue.extend([&input]);
                    }
                    Operation::MaxDim(input, _) => {
                        tensor_queue.extend([&input]);
                    }
                    Operation::Gather {
                        input,
                        indices: _,
                        dim: _,
                    } => {
                        tensor_queue.extend([&input]);
                    }
                    Operation::Scatter {
                        input,
                        indices: _,
                        dim: _,
                    } => {
                        tensor_queue.extend([&input]);
                    }
                }
            }
        }

        // Used to keep book if tensor is already in the graph
        let mut graph_tensors = HashSet::new();
        // Start from the tail to confirm that if the tensor is used in backprop op, its grad is
        // computed completely already
        let mut ordered_tensor_stack = Vec::new();
        for node in tensor_stack.into_iter().rev() {
            if !graph_tensors.contains(&node.id()) {
                ordered_tensor_stack.push(node.clone());
                graph_tensors.insert(node.id());
            }
        }
        ordered_tensor_stack
    }

    pub fn backward(&self) -> Result<GradientStorage<B>> {
        let mut grad_storage = GradientStorage::<B>::new();
        let first_grad_dimensions = self.dimensions();
        let first_grad = Tensor::ones(self.backend(), first_grad_dimensions, false)?;
        grad_storage.insert_or_add(self, first_grad.clone())?;
        let mut ordered_backward_tensors = self.get_graph_tensors_sorted();
        while let Some(tensor) = ordered_backward_tensors.pop() {
            // let grad = grad_storage.get(tensor.id());
            let grad = match tensor.has_grad() {
                true => grad_storage.get(tensor.id()),
                false => grad_storage.remove(tensor.id()),
            };
            // If grad and backprop_op exists - backpropagate (non grad tensors should have been
            // filtered already when getting the graph tensors)
            if let Some(grad) = grad {
                if let Some(backprop_op) = tensor.back_propagation_op() {
                    match backprop_op {
                        Operation::Add(lhs, rhs) => {
                            let lhs_grad = Tensor::zeros_like(lhs)?.add(&grad)?;
                            grad_storage.insert_or_add(lhs, lhs_grad)?;

                            let rhs_grad = Tensor::zeros_like(rhs)?.add(&grad)?;
                            grad_storage.insert_or_add(rhs, rhs_grad)?;
                        }
                        Operation::Sub(lhs, rhs) => {
                            let lhs_grad = Tensor::zeros_like(lhs)?.add(&grad)?;
                            grad_storage.insert_or_add(lhs, lhs_grad)?;

                            let rhs_grad = Tensor::zeros_like(rhs)?.sub(&grad)?;
                            grad_storage.insert_or_add(rhs, rhs_grad)?;
                        }
                        Operation::Mul(lhs, rhs) => {
                            let lhs_grad = grad.mul(rhs)?;
                            grad_storage.insert_or_add(lhs, lhs_grad)?;

                            let rhs_grad = grad.mul(lhs)?;
                            grad_storage.insert_or_add(rhs, rhs_grad)?;
                        }
                        Operation::Div(lhs, rhs) => {
                            let lhs_grad = grad.div(rhs)?;
                            grad_storage.insert_or_add(lhs, lhs_grad)?;

                            let minus_one = Tensor::from_values(
                                self.backend(),
                                vec![-1.0],
                                Dimensions4D::new(1, 1, 1, 1),
                                false,
                            )?;
                            let rhs_grad = grad.mul(&minus_one.mul(lhs)?)?.div(&rhs.squared()?)?;
                            grad_storage.insert_or_add(rhs, rhs_grad)?;
                        }
                        Operation::Pow(input, power) => {
                            // NOTE: ignore the powers grad
                            let by_one =
                                Tensor::ones(self.backend(), Dimensions4D::new(1, 1, 1, 1), false)?;
                            let new_power = power.sub(&by_one)?;
                            let input_grad = grad.mul(&input.pow(&new_power)?.mul(power)?)?;
                            grad_storage.insert_or_add(input, input_grad)?;
                        }
                        Operation::Squared(input) => {
                            let two = Tensor::from_values(
                                self.backend(),
                                vec![2.0],
                                Dimensions4D::new(1, 1, 1, 1),
                                false,
                            )?;
                            let squared_grad = &two.mul(input)?;
                            let input_grad = grad.mul(squared_grad)?;
                            grad_storage.insert_or_add(input, input_grad)?;
                        }
                        Operation::Exp(input) => {
                            // d/dx e^x = e^x
                            let input_grad = grad.mul(&tensor)?;
                            grad_storage.insert_or_add(input, input_grad)?;
                        }
                        Operation::Ln(input) => {
                            // d/dx ln x = 1/x
                            let one =
                                Tensor::ones(self.backend(), Dimensions4D::new(1, 1, 1, 1), false)?;
                            let new_grad = one.div(input)?;
                            let input_grad = grad.mul(&new_grad)?;
                            grad_storage.insert_or_add(input, input_grad)?;
                        }
                        Operation::ReduceSum(input) => {
                            // TODO: broadcast should probably be an operation
                            let old_dimensions = input.dimensions();
                            let zeros = Tensor::zeros(self.backend(), old_dimensions, false)?;
                            let new_grad = zeros.add(&grad)?;
                            grad_storage.insert_or_add(input, new_grad)?;
                        }
                        Operation::Broadcast(input, broadcasted_dimensions) => {
                            let old_dimensions = input.dimensions();
                            let broadcasted_dimensions =
                                broadcasted_dimensions.matching_dimensions(old_dimensions);
                            let reduced_as_arr = [
                                broadcasted_dimensions.0,
                                broadcasted_dimensions.1,
                                broadcasted_dimensions.2,
                                broadcasted_dimensions.3,
                            ];
                            let dims = [0, 1, 2, 3]
                                .into_iter()
                                .zip(reduced_as_arr)
                                .filter(|(_d, r)| !*r)
                                .map(|(d, _r)| d)
                                .collect::<Vec<_>>();
                            let sum_reduced_grad = grad.sum(Some(&dims))?;
                            grad_storage.insert_or_add(input, sum_reduced_grad)?;
                        }
                        Operation::Matmul {
                            lhs,
                            rhs,
                            is_rhs_transposed,
                        } => {
                            let lhs_grad = grad.matmul(rhs, !*is_rhs_transposed)?;
                            grad_storage.insert_or_add(lhs, lhs_grad)?;

                            let rhs_grad = if *is_rhs_transposed {
                                lhs.transpose(2, 3)?.matmul(&grad, false)?.transpose(2, 3)?
                            } else {
                                lhs.transpose(2, 3)?.matmul(&grad, false)?
                            };
                            grad_storage.insert_or_add(rhs, rhs_grad)?;
                        }
                        Operation::Conv2d {
                            input,
                            kernel,
                            padding,
                        } => {
                            let kernel_grad = input
                                .transpose(0, 1)?
                                .conv2d(&grad.transpose(0, 1)?, padding)?
                                .transpose(0, 1)?;
                            grad_storage.insert_or_add(kernel, kernel_grad)?;

                            let input_grad = grad.conv_transpose2d(kernel, padding)?;
                            grad_storage.insert_or_add(input, input_grad)?;
                        }
                        Operation::Transpose(input, (dim1, dim2)) => {
                            let transposed_back_grad = grad.transpose(*dim1, *dim2)?;
                            grad_storage.insert_or_add(input, transposed_back_grad)?;
                        }
                        Operation::Reshape(input) => {
                            let grad_reshaped = grad.reshape(input.dimensions())?;
                            grad_storage.insert_or_add(input, grad_reshaped)?;
                        }
                        Operation::Activation(input, activation_type) => match activation_type {
                            ActivationType::Relu => {
                                let zero_tensor = Tensor::zeros(
                                    self.backend(),
                                    Dimensions4D::new(1, 1, 1, 1),
                                    false,
                                )?;
                                let relu_grad = input.greater_or_equal(&zero_tensor)?;
                                let new_grad = relu_grad.mul(&grad)?;
                                grad_storage.insert_or_add(input, new_grad)?;
                            }
                            ActivationType::LeakyRelu(alpha) => {
                                // first get the relu grad
                                let zero_tensor = Tensor::zeros(
                                    self.backend(),
                                    Dimensions4D::new(1, 1, 1, 1),
                                    false,
                                )?;
                                let relu_grad = input.greater_or_equal(&zero_tensor)?;

                                let alpha_tensor = Tensor::from_values(
                                    self.backend(),
                                    vec![*alpha],
                                    Dimensions4D::new(1, 1, 1, 1),
                                    false,
                                )?;
                                // find the values which were zeroed in the relu and replace those
                                // with alpha
                                let complement = relu_grad.equal(&zero_tensor)?;
                                let leaky_alpha = complement.mul(&alpha_tensor)?;
                                let leaky_relu_grad = relu_grad.add(&leaky_alpha)?;
                                let new_grad = leaky_relu_grad.mul(&grad)?;
                                grad_storage.insert_or_add(input, new_grad)?;
                            }
                            ActivationType::Sigmoid => {
                                let one_tensor =
                                    Tensor::ones(self.backend(), input.dimensions(), false)?;
                                let sigma = input.activation(ActivationType::Sigmoid)?;
                                let sigma_grad = one_tensor.sub(&sigma)?.mul(&sigma)?;
                                let new_grad = sigma_grad.mul(&grad)?;
                                grad_storage.insert_or_add(input, new_grad)?;
                            }
                            ActivationType::Softmax => {
                                // Softmax is implemented with tensor ops so no separate grad
                                // operation needed
                            }
                        },
                        Operation::Pool(input, pool_type, pool_dimensions) => {
                            match pool_type {
                                PoolType::MaxPool2D => {
                                    // upsample the output, compare with the input to get the
                                    // gradient mask and the multiply the upsampled gradient with
                                    // the mask
                                    let upsampled_output = tensor.upsample2d(*pool_dimensions)?;
                                    let gradient_mask =
                                        input.greater_or_equal(&upsampled_output)?;
                                    let upsampled_grad = grad.upsample2d(*pool_dimensions)?;
                                    let masked_grad = upsampled_grad.mul(&gradient_mask)?;
                                    grad_storage.insert_or_add(input, masked_grad)?;
                                }
                                PoolType::AvgPool2D => {
                                    let upsampled_grad = grad.upsample2d(*pool_dimensions)?;
                                    let pool_area = pool_dimensions.height * pool_dimensions.width;
                                    let pool_area_tensor = Tensor::from_values(
                                        self.backend(),
                                        vec![pool_area as f32],
                                        Dimensions4D::new(1, 1, 1, 1),
                                        false,
                                    )?;
                                    let avg_grad = upsampled_grad.div(&pool_area_tensor)?;
                                    grad_storage.insert_or_add(input, avg_grad)?;
                                }
                            }
                        }
                        Operation::Upsample(_tensor, _upsample_dimensions) => todo!(),
                        Operation::ConvTranspose2d {
                            input,
                            kernel,
                            input_padding,
                        } => {
                            let kernel_grad = grad
                                .transpose(0, 1)?
                                .conv2d(&input.transpose(0, 1)?, input_padding)?
                                .transpose(0, 1)?;
                            grad_storage.insert_or_add(kernel, kernel_grad)?;

                            let input_grad = grad.conv2d(kernel, input_padding)?;
                            grad_storage.insert_or_add(input, input_grad)?;
                        }
                        Operation::CustomUnary(_, _) => todo!(),
                        Operation::CustomBinary(_, _, _) => todo!(),
                        // No backwards
                        Operation::Cmp(_, _, _) => {}
                        Operation::ArgMax(_, _) => todo!(), // no backward
                        Operation::MaxDim(input, dim) => {
                            let indices = input.argmax(*dim)?;
                            let input_grad = grad.scatter(*dim, &indices, input.dimensions())?;
                            grad_storage.insert_or_add(input, input_grad)?;
                        }
                        Operation::Gather {
                            input,
                            indices,
                            dim,
                        } => {
                            let input_grad = grad.scatter(*dim, indices, input.dimensions())?;
                            grad_storage.insert_or_add(input, input_grad)?;
                        }
                        Operation::Scatter {
                            input,
                            indices,
                            dim,
                        } => {
                            let input_grad = grad.gather(*dim, indices)?;
                            grad_storage.insert_or_add(input, input_grad)?;
                        }
                    }
                }
            }
        }
        Ok(grad_storage)
    }
}

#[cfg(test)]
mod test {
    use std::{cell::RefCell, rc::Rc};

    use rand::Rng;
    use wgpu::naga::BinaryOperator;

    use crate::machine_learning::{
        compare_outputs,
        nn::{
            backend::Backend,
            backends::{cpu_backend::CpuBackend, wgpu_backend::WgpuBackend},
            dimensions::Dimensions4D,
            linspace,
            model_config::{Padding, PoolType},
            tensor::Tensor,
        },
    };

    fn tensor_conv2d_backward<B: Backend>(backend: Rc<RefCell<B>>) {
        let dimensions = Dimensions4D::new(2, 6, 5, 4);
        let values = linspace(0.0, 1.0, dimensions.size());
        let input_tensor =
            Tensor::from_values(backend.clone(), values.clone(), dimensions, true).unwrap();
        // let values = vec![1.0; dimensions.size()];
        let kernel_dimensions = Dimensions4D::new(2, 6, 3, 3);
        let kernel_values = linspace(1.0, 3.0, kernel_dimensions.size());
        let kernel_tensor = Tensor::from_values(
            backend.clone(),
            kernel_values.clone(),
            kernel_dimensions,
            true,
        )
        .unwrap();
        let conv2d_result = input_tensor
            .conv2d(&kernel_tensor, &Padding::Zero(1, 1))
            .unwrap();
        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected_input_grad = [
            12.336448598130842, 18.61682242990654, 18.61682242990654, 12.485981308411215,
            18.841121495327105, 28.429906542056074, 28.429906542056074, 19.065420560747665,
            18.841121495327105, 28.429906542056074, 28.429906542056074, 19.065420560747665,
            18.841121495327105, 28.429906542056074, 28.429906542056074, 19.065420560747665,
            12.785046728971963, 19.289719626168225, 19.289719626168225, 12.934579439252335,

            13.682242990654206, 20.63551401869159, 20.63551401869159, 13.83177570093458,
            20.85981308411215, 31.45794392523365, 31.45794392523365, 21.084112149532707,
            20.85981308411215, 31.45794392523365, 31.45794392523365, 21.084112149532707,
            20.85981308411215, 31.45794392523365, 31.45794392523365, 21.084112149532707,
            14.130841121495326, 21.308411214953274, 21.308411214953274, 14.280373831775702,

            15.02803738317757, 22.654205607476637, 22.654205607476637, 15.177570093457945,
            22.878504672897197, 34.48598130841121, 34.48598130841121, 23.102803738317757,
            22.878504672897197, 34.48598130841121, 34.48598130841121, 23.102803738317757,
            22.878504672897197, 34.48598130841121, 34.48598130841121, 23.102803738317757,
            15.476635514018692, 23.327102803738317, 23.327102803738317, 15.626168224299066,

            16.373831775700936, 24.67289719626168, 24.67289719626168, 16.523364485981308,
            24.897196261682243, 37.51401869158879, 37.51401869158879, 25.121495327102803,
            24.897196261682243, 37.51401869158879, 37.51401869158879, 25.121495327102803,
            24.897196261682243, 37.51401869158879, 37.51401869158879, 25.121495327102803,
            16.822429906542055, 25.345794392523366, 25.345794392523366, 16.97196261682243,

            17.7196261682243, 26.691588785046726, 26.691588785046726, 17.869158878504674,
            26.91588785046729, 40.54205607476635, 40.54205607476635, 27.140186915887853,
            26.91588785046729, 40.54205607476635, 40.54205607476635, 27.140186915887853,
            26.91588785046729, 40.54205607476635, 40.54205607476635, 27.140186915887853,
            18.168224299065418, 27.364485981308412, 27.364485981308412, 18.317757009345794,

            19.06542056074766, 28.710280373831775, 28.710280373831775, 19.214953271028037,
            28.93457943925233, 43.570093457943926, 43.570093457943926, 29.1588785046729,
            28.93457943925233, 43.570093457943926, 43.570093457943926, 29.1588785046729,
            28.93457943925233, 43.570093457943926, 43.570093457943926, 29.1588785046729,
            19.514018691588785, 29.38317757009346, 29.38317757009346, 19.663551401869157,


            12.336448598130842, 18.61682242990654, 18.61682242990654, 12.485981308411215,
            18.841121495327105, 28.429906542056074, 28.429906542056074, 19.065420560747665,
            18.841121495327105, 28.429906542056074, 28.429906542056074, 19.065420560747665,
            18.841121495327105, 28.429906542056074, 28.429906542056074, 19.065420560747665,
            12.785046728971963, 19.289719626168225, 19.289719626168225, 12.934579439252335,

            13.682242990654206, 20.63551401869159, 20.63551401869159, 13.83177570093458,
            20.85981308411215, 31.45794392523365, 31.45794392523365, 21.084112149532707,
            20.85981308411215, 31.45794392523365, 31.45794392523365, 21.084112149532707,
            20.85981308411215, 31.45794392523365, 31.45794392523365, 21.084112149532707,
            14.130841121495326, 21.308411214953274, 21.308411214953274, 14.280373831775702,

            15.02803738317757, 22.654205607476637, 22.654205607476637, 15.177570093457945,
            22.878504672897197, 34.48598130841121, 34.48598130841121, 23.102803738317757,
            22.878504672897197, 34.48598130841121, 34.48598130841121, 23.102803738317757,
            22.878504672897197, 34.48598130841121, 34.48598130841121, 23.102803738317757,
            15.476635514018692, 23.327102803738317, 23.327102803738317, 15.626168224299066,

            16.373831775700936, 24.67289719626168, 24.67289719626168, 16.523364485981308,
            24.897196261682243, 37.51401869158879, 37.51401869158879, 25.121495327102803,
            24.897196261682243, 37.51401869158879, 37.51401869158879, 25.121495327102803,
            24.897196261682243, 37.51401869158879, 37.51401869158879, 25.121495327102803,
            16.822429906542055, 25.345794392523366, 25.345794392523366, 16.97196261682243,

            17.7196261682243, 26.691588785046726, 26.691588785046726, 17.869158878504674,
            26.91588785046729, 40.54205607476635, 40.54205607476635, 27.140186915887853,
            26.91588785046729, 40.54205607476635, 40.54205607476635, 27.140186915887853,
            26.91588785046729, 40.54205607476635, 40.54205607476635, 27.140186915887853,
            18.168224299065418, 27.364485981308412, 27.364485981308412, 18.317757009345794,

            19.06542056074766, 28.710280373831775, 28.710280373831775, 19.214953271028037,
            28.93457943925233, 43.570093457943926, 43.570093457943926, 29.1588785046729,
            28.93457943925233, 43.570093457943926, 43.570093457943926, 29.1588785046729,
            28.93457943925233, 43.570093457943926, 43.570093457943926, 29.1588785046729,
            19.514018691588785, 29.38317757009346, 29.38317757009346, 19.663551401869157,
        ];

        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected_kernel_grad = [
            6.728033472803346, 9.037656903765688, 6.8284518828451874,
            8.661087866108785, 11.631799163179915, 8.786610878661088,
            7.12970711297071, 9.573221757322175, 7.2301255230125525,

            8.736401673640167, 11.715481171548117, 8.836820083682007,
            11.171548117154812, 14.97907949790795, 11.297071129707112,
            9.13807531380753, 12.2510460251046, 9.238493723849373,

            10.744769874476987, 14.393305439330545, 10.84518828451883,
            13.682008368200837, 18.326359832635987, 13.807531380753138,
            11.146443514644352, 14.92887029288703, 11.246861924686193,

            12.753138075313807, 17.07112970711297, 12.853556485355648,
            16.19246861924686, 21.673640167364017, 16.317991631799163,
            13.154811715481173, 17.606694560669457, 13.255230125523013,

            14.761506276150625, 19.748953974895393, 14.861924686192467,
            18.702928870292883, 25.020920502092046, 18.828451882845187,
            15.163179916317992, 20.284518828451883, 15.263598326359833,

            16.769874476987447, 22.42677824267782, 16.870292887029287,
            21.213389121338913, 28.36820083682008, 21.338912133891213,
            17.171548117154813, 22.96234309623431, 17.271966527196653,


            6.728033472803346, 9.037656903765688, 6.8284518828451874,
            8.661087866108785, 11.631799163179915, 8.786610878661088,
            7.12970711297071, 9.573221757322175, 7.2301255230125525,

            8.736401673640167, 11.715481171548117, 8.836820083682007,
            11.171548117154812, 14.97907949790795, 11.297071129707112,
            9.13807531380753, 12.2510460251046, 9.238493723849373,

            10.744769874476987, 14.393305439330545, 10.84518828451883,
            13.682008368200837, 18.326359832635987, 13.807531380753138,
            11.146443514644352, 14.92887029288703, 11.246861924686193,

            12.753138075313807, 17.07112970711297, 12.853556485355648,
            16.19246861924686, 21.673640167364017, 16.317991631799163,
            13.154811715481173, 17.606694560669457, 13.255230125523013,

            14.761506276150625, 19.748953974895393, 14.861924686192467,
            18.702928870292883, 25.020920502092046, 18.828451882845187,
            15.163179916317992, 20.284518828451883, 15.263598326359833,

            16.769874476987447, 22.42677824267782, 16.870292887029287,
            21.213389121338913, 28.36820083682008, 21.338912133891213,
            17.171548117154813, 22.96234309623431, 17.271966527196653,
        ];

        let gradients = conv2d_result.backward().unwrap();

        let input_grad = gradients
            .get(input_tensor.id())
            .expect("Input should have grad.")
            .get_values()
            .unwrap();
        let kernel_grad = gradients
            .get(kernel_tensor.id())
            .expect("Kernel should have grad.")
            .get_values()
            .unwrap();

        assert!(compare_outputs(&input_grad, &expected_input_grad));
        assert!(compare_outputs(&kernel_grad, &expected_kernel_grad));
    }

    #[test]
    fn test_conv2d_backward() {
        tensor_conv2d_backward(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_conv2d_backward(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_transpose_conv2d_backward<B: Backend>(backend: Rc<RefCell<B>>) {
        let dimensions = Dimensions4D::new(2, 6, 5, 4);
        let values = linspace(0.0, 1.0, dimensions.size());
        let input_tensor =
            Tensor::from_values(backend.clone(), values.clone(), dimensions, true).unwrap();
        // let values = vec![1.0; dimensions.size()];
        let kernel_dimensions = Dimensions4D::new(6, 2, 3, 3);
        let kernel_values = linspace(1.0, 3.0, kernel_dimensions.size());
        let kernel_tensor = Tensor::from_values(
            backend.clone(),
            kernel_values.clone(),
            kernel_dimensions,
            true,
        )
        .unwrap();
        let conv_transpose2d_result = input_tensor
            .conv_transpose2d(&kernel_tensor, &Padding::Zero(1, 1))
            .unwrap();
        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected_input_grad = [
            9.570093457943925, 14.242990654205608, 14.242990654205608, 9.420560747663552,
            14.018691588785046, 20.85981308411215, 20.85981308411215, 13.794392523364488,
            14.018691588785046, 20.85981308411215, 20.85981308411215, 13.794392523364488,
            14.018691588785046, 20.85981308411215, 20.85981308411215, 13.794392523364488,
            9.121495327102803, 13.570093457943926, 13.570093457943926, 8.971962616822431,

            12.261682242990656, 18.280373831775698, 18.280373831775698, 12.112149532710278,
            18.056074766355138, 26.915887850467293, 26.915887850467293, 17.831775700934575,
            18.056074766355138, 26.915887850467293, 26.915887850467293, 17.831775700934575,
            18.056074766355138, 26.915887850467293, 26.915887850467293, 17.831775700934575,
            11.813084112149532, 17.60747663551402, 17.60747663551402, 11.66355140186916,

            14.953271028037383, 22.317757009345797, 22.317757009345797, 14.803738317757007,
            22.09345794392523, 32.97196261682243, 32.97196261682243, 21.86915887850467,
            22.09345794392523, 32.97196261682243, 32.97196261682243, 21.86915887850467,
            22.09345794392523, 32.97196261682243, 32.97196261682243, 21.86915887850467,
            14.504672897196262, 21.64485981308411, 21.64485981308411, 14.355140186915888,

            17.64485981308411, 26.35514018691589, 26.35514018691589, 17.495327102803735,
            26.130841121495326, 39.028037383177555, 39.028037383177555, 25.906542056074763,
            26.130841121495326, 39.028037383177555, 39.028037383177555, 25.906542056074763,
            26.130841121495326, 39.028037383177555, 39.028037383177555, 25.906542056074763,
            17.19626168224299, 25.682242990654203, 25.682242990654203, 17.046728971962615,

            20.33644859813084, 30.392523364485985, 30.392523364485985, 20.186915887850468,
            30.168224299065418, 45.08411214953271, 45.08411214953271, 29.943925233644855,
            30.168224299065418, 45.08411214953271, 45.08411214953271, 29.943925233644855,
            30.168224299065418, 45.08411214953271, 45.08411214953271, 29.943925233644855,
            19.887850467289717, 29.719626168224302, 29.719626168224302, 19.738317757009348,

            23.028037383177566, 34.429906542056074, 34.429906542056074, 22.878504672897193,
            34.205607476635514, 51.14018691588785, 51.14018691588785, 33.98130841121495,
            34.205607476635514, 51.14018691588785, 51.14018691588785, 33.98130841121495,
            34.205607476635514, 51.14018691588785, 51.14018691588785, 33.98130841121495,
            22.57943925233645, 33.75700934579439, 33.75700934579439, 22.429906542056074,


            9.570093457943925, 14.242990654205608, 14.242990654205608, 9.420560747663552,
            14.018691588785046, 20.85981308411215, 20.85981308411215, 13.794392523364488,
            14.018691588785046, 20.85981308411215, 20.85981308411215, 13.794392523364488,
            14.018691588785046, 20.85981308411215, 20.85981308411215, 13.794392523364488,
            9.121495327102803, 13.570093457943926, 13.570093457943926, 8.971962616822431,

            12.261682242990656, 18.280373831775698, 18.280373831775698, 12.112149532710278,
            18.056074766355138, 26.915887850467293, 26.915887850467293, 17.831775700934575,
            18.056074766355138, 26.915887850467293, 26.915887850467293, 17.831775700934575,
            18.056074766355138, 26.915887850467293, 26.915887850467293, 17.831775700934575,
            11.813084112149532, 17.60747663551402, 17.60747663551402, 11.66355140186916,

            14.953271028037383, 22.317757009345797, 22.317757009345797, 14.803738317757007,
            22.09345794392523, 32.97196261682243, 32.97196261682243, 21.86915887850467,
            22.09345794392523, 32.97196261682243, 32.97196261682243, 21.86915887850467,
            22.09345794392523, 32.97196261682243, 32.97196261682243, 21.86915887850467,
            14.504672897196262, 21.64485981308411, 21.64485981308411, 14.355140186915888,

            17.64485981308411, 26.35514018691589, 26.35514018691589, 17.495327102803735,
            26.130841121495326, 39.028037383177555, 39.028037383177555, 25.906542056074763,
            26.130841121495326, 39.028037383177555, 39.028037383177555, 25.906542056074763,
            26.130841121495326, 39.028037383177555, 39.028037383177555, 25.906542056074763,
            17.19626168224299, 25.682242990654203, 25.682242990654203, 17.046728971962615,

            20.33644859813084, 30.392523364485985, 30.392523364485985, 20.186915887850468,
            30.168224299065418, 45.08411214953271, 45.08411214953271, 29.943925233644855,
            30.168224299065418, 45.08411214953271, 45.08411214953271, 29.943925233644855,
            30.168224299065418, 45.08411214953271, 45.08411214953271, 29.943925233644855,
            19.887850467289717, 29.719626168224302, 29.719626168224302, 19.738317757009348,

            23.028037383177566, 34.429906542056074, 34.429906542056074, 22.878504672897193,
            34.205607476635514, 51.14018691588785, 51.14018691588785, 33.98130841121495,
            34.205607476635514, 51.14018691588785, 51.14018691588785, 33.98130841121495,
            34.205607476635514, 51.14018691588785, 51.14018691588785, 33.98130841121495,
            22.57943925233645, 33.75700934579439, 33.75700934579439, 22.429906542056074,

        ];

        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected_kernel_grad = [
            7.2301255230125525, 9.573221757322177, 7.129707112970712,
            8.786610878661088, 11.631799163179911, 8.661087866108787,
            6.828451882845188, 9.037656903765688, 6.7280334728033475,

            7.2301255230125525, 9.573221757322177, 7.129707112970712,
            8.786610878661088, 11.631799163179911, 8.661087866108787,
            6.828451882845188, 9.037656903765688, 6.728033472803346,


            9.238493723849373, 12.251046025104603, 9.138075313807532,
            11.297071129707113, 14.97907949790795, 11.171548117154812,
            8.836820083682008, 11.715481171548117, 8.736401673640167,

            9.238493723849373, 12.251046025104603, 9.138075313807532,
            11.297071129707113, 14.97907949790795, 11.171548117154812,
            8.836820083682008, 11.715481171548117, 8.736401673640167,


            11.246861924686193, 14.928870292887026, 11.146443514644352,
            13.807531380753133, 18.326359832635973, 13.682008368200831,
            10.845188284518825, 14.393305439330536, 10.744769874476985,

            11.246861924686193, 14.928870292887026, 11.146443514644352,
            13.807531380753133, 18.326359832635973, 13.682008368200831,
            10.845188284518825, 14.393305439330545, 10.744769874476987,


            13.255230125523013, 17.606694560669457, 13.154811715481172,
            16.31799163179916, 21.673640167364017, 16.19246861924686,
            12.853556485355647, 17.07112970711297, 12.753138075313807,

            13.255230125523013, 17.606694560669457, 13.154811715481172,
            16.31799163179916, 21.673640167364017, 16.19246861924686,
            12.853556485355647, 17.07112970711297, 12.753138075313807,


            15.26359832635983, 20.284518828451876, 15.163179916317988,
            18.82845188284518, 25.020920502092046, 18.70292887029288,
            14.861924686192463, 19.748953974895393, 14.761506276150623,

            15.26359832635983, 20.284518828451876, 15.163179916317988,
            18.82845188284518, 25.020920502092046, 18.70292887029288,
            14.861924686192463, 19.748953974895393, 14.761506276150625,


            17.271966527196653, 22.96234309623431, 17.171548117154813,
            21.338912133891213, 28.368200836820087, 21.213389121338913,
            16.870292887029287, 22.426778242677827, 16.769874476987447,

            17.271966527196653, 22.96234309623431, 17.171548117154813,
            21.338912133891213, 28.368200836820087, 21.213389121338913,
            16.870292887029287, 22.42677824267782, 16.769874476987447,
        ];

        let gradients = conv_transpose2d_result.backward().unwrap();

        let input_grad = gradients
            .get(input_tensor.id())
            .expect("Input should have grad.")
            .get_values()
            .unwrap();
        let kernel_grad = gradients
            .get(kernel_tensor.id())
            .expect("Kernel should have grad.")
            .get_values()
            .unwrap();

        assert!(compare_outputs(&input_grad, &expected_input_grad));
        assert!(compare_outputs(&kernel_grad, &expected_kernel_grad));
    }

    #[test]
    fn test_transpose_conv2d_backward() {
        tensor_transpose_conv2d_backward(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_transpose_conv2d_backward(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_max_pool<B: Backend>(backend: Rc<RefCell<B>>) {
        let dimensions = Dimensions4D::new(2, 6, 4, 2);
        let values = linspace(0.0, 1.0, dimensions.size());
        let input_tensor =
            Tensor::from_values(backend.clone(), values.clone(), dimensions, true).unwrap();
        // let values = vec![1.0; dimensions.size()];
        let conv2d_result = input_tensor
            .pool(PoolType::MaxPool2D, [2, 2].into())
            .unwrap();

        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected_input_grad = [
            0.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 1.0,

            0.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 1.0,

            0.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 1.0,

            0.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 1.0,

            0.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 1.0,

            0.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 1.0,


            0.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 1.0,

            0.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 1.0,

            0.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 1.0,

            0.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 1.0,

            0.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 1.0,

            0.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 1.0,
        ];

        let gradients = conv2d_result.backward().unwrap();

        let input_grad = gradients
            .get(input_tensor.id())
            .expect("Input should have grad.")
            .get_values()
            .unwrap();

        assert!(compare_outputs(&input_grad, &expected_input_grad));
    }

    #[test]
    fn test_maxpool2d() {
        tensor_max_pool(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_max_pool(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_avg_pool<B: Backend>(backend: Rc<RefCell<B>>) {
        let dimensions = Dimensions4D::new(2, 6, 4, 2);
        let values = linspace(0.0, 1.0, dimensions.size());
        let input_tensor =
            Tensor::from_values(backend.clone(), values.clone(), dimensions, true).unwrap();
        // let values = vec![1.0; dimensions.size()];
        let conv2d_result = input_tensor
            .pool(PoolType::AvgPool2D, [2, 2].into())
            .unwrap();

        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected_input_grad = [
            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,

            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,

            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,

            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,

            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,

            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,


            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,

            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,

            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,

            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,

            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,

            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,
            0.25, 0.25,
        ];

        let gradients = conv2d_result.backward().unwrap();

        let input_grad = gradients
            .get(input_tensor.id())
            .expect("Input should have grad.")
            .get_values()
            .unwrap();

        assert!(compare_outputs(&input_grad, &expected_input_grad));
    }

    #[test]
    fn test_avgpool2d() {
        tensor_avg_pool(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_avg_pool(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_mse_backward<B: Backend>(backend: Rc<RefCell<B>>) {
        let gts = vec![
            -0.51075903f32,
            -1.348_058_8,
            1.179_314_6,
            -1.537_625,
            0.47580442,
            -0.670_917_6,
            -0.49488483,
            -1.505_284_4,
            -0.79802066,
            0.03364044,
        ];

        let preds = vec![
            0.797_840_4_f32,
            -0.929_251_6,
            0.38215753,
            -1.134_334,
            -0.44187116,
            -0.03925696,
            2.377_588_5,
            -1.516_904_1,
            -0.6910318,
            -1.130_936_6,
        ];

        let gt_tensor = Tensor::from_values(
            backend.clone(),
            gts.clone(),
            Dimensions4D::new(10, 1, 1, 1),
            false,
        )
        .unwrap();
        let pred_tensor = Tensor::from_values(
            backend.clone(),
            preds.clone(),
            Dimensions4D::new(10, 1, 1, 1),
            true,
        )
        .unwrap();

        let mse_tensor = pred_tensor.mse(&gt_tensor).unwrap();
        let gradients = mse_tensor.backward().unwrap();

        let pred_gradients = gradients
            .get(pred_tensor.id())
            .expect("Predictions should have grad.")
            .get_values()
            .unwrap();

        let expected_pred_grads = [
            0.2617, 0.0838, -0.1594, 0.0807, -0.1835, 0.1263, 0.5745, -0.0023, 0.0214, -0.2329,
        ];

        assert!(compare_outputs(&pred_gradients, &expected_pred_grads));
    }

    #[test]
    fn test_mse_backward() {
        tensor_mse_backward(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_mse_backward(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_binary_operations_backwards<B: Backend>(backend: Rc<RefCell<B>>, op: BinaryOperator) {
        let dimensions = Dimensions4D::new(2, 3, 4, 5);
        let lhs = linspace(-1.0, 2.0, dimensions.size());
        let rhs = linspace(3.0, 6.0, dimensions.size());

        let lhs_tensor =
            Tensor::from_values(backend.clone(), lhs.clone(), dimensions, true).unwrap();
        let rhs_tensor =
            Tensor::from_values(backend.clone(), rhs.clone(), dimensions, true).unwrap();

        let (result_tensor, expected_grad_lhs, expected_grad_rhs) = match op {
            BinaryOperator::Add => {
                let result_tensor = lhs_tensor.add(&rhs_tensor).unwrap();
                let expected_grad_lhs = vec![1.0; lhs.len()];
                let expected_grad_rhs = vec![1.0; rhs.len()];
                (result_tensor, expected_grad_lhs, expected_grad_rhs)
            }
            BinaryOperator::Subtract => {
                let result_tensor = lhs_tensor.sub(&rhs_tensor).unwrap();
                let expected_grad_lhs = vec![1.0; lhs.len()];
                let expected_grad_rhs = vec![-1.0; rhs.len()];
                (result_tensor, expected_grad_lhs, expected_grad_rhs)
            }
            BinaryOperator::Multiply => {
                let result_tensor = lhs_tensor.mul(&rhs_tensor).unwrap();
                let expected_grad_lhs = rhs.clone();
                let expected_grad_rhs = lhs.clone();
                (result_tensor, expected_grad_lhs, expected_grad_rhs)
            }
            BinaryOperator::Divide => {
                let result_tensor = lhs_tensor.div(&rhs_tensor).unwrap();
                let expected_grad_lhs = rhs.iter().map(|&x| 1.0 / x).collect();
                let expected_grad_rhs = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(&l, &r)| -l / (r * r))
                    .collect();
                (result_tensor, expected_grad_lhs, expected_grad_rhs)
            }
            BinaryOperator::Equal => {
                let result_tensor = lhs_tensor.equal(&rhs_tensor).unwrap();
                let expected_grad_lhs = rhs.iter().map(|&x| 1.0 / x).collect();
                let expected_grad_rhs = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(&l, &r)| -l / (r * r))
                    .collect();
                (result_tensor, expected_grad_lhs, expected_grad_rhs)
            }
            _ => todo!(),
        };
        let gradients = result_tensor.backward().unwrap();

        let lhs_gradients = gradients
            .get(lhs_tensor.id())
            .expect("Predictions should have grad.")
            .get_values()
            .unwrap();

        let rhs_gradients = gradients
            .get(rhs_tensor.id())
            .expect("Predictions should have grad.")
            .get_values()
            .unwrap();

        assert!(compare_outputs(&lhs_gradients, &expected_grad_lhs));
        assert!(compare_outputs(&rhs_gradients, &expected_grad_rhs));
    }

    #[test]
    fn test_binary_operations_backward() {
        tensor_binary_operations_backwards(
            Rc::new(RefCell::new(WgpuBackend::default())),
            BinaryOperator::Add,
        );
        tensor_binary_operations_backwards(
            Rc::new(RefCell::new(CpuBackend::default())),
            BinaryOperator::Add,
        );
        tensor_binary_operations_backwards(
            Rc::new(RefCell::new(WgpuBackend::default())),
            BinaryOperator::Subtract,
        );
        tensor_binary_operations_backwards(
            Rc::new(RefCell::new(CpuBackend::default())),
            BinaryOperator::Subtract,
        );
        tensor_binary_operations_backwards(
            Rc::new(RefCell::new(WgpuBackend::default())),
            BinaryOperator::Multiply,
        );
        tensor_binary_operations_backwards(
            Rc::new(RefCell::new(CpuBackend::default())),
            BinaryOperator::Multiply,
        );
        tensor_binary_operations_backwards(
            Rc::new(RefCell::new(WgpuBackend::default())),
            BinaryOperator::Divide,
        );
        tensor_binary_operations_backwards(
            Rc::new(RefCell::new(CpuBackend::default())),
            BinaryOperator::Divide,
        );
    }

    fn tensor_binary_broadcast_backwards<B: Backend>(backend: Rc<RefCell<B>>, op: BinaryOperator) {
        let lhs_dimensions = Dimensions4D::new(2, 3, 4, 5);
        let rhs_dimensions = Dimensions4D::new(1, 3, 1, 1);
        let lhs = linspace(-1.0, 2.0, lhs_dimensions.size());
        let rhs = linspace(3.0, 6.0, rhs_dimensions.size());

        let lhs_tensor =
            Tensor::from_values(backend.clone(), lhs.clone(), lhs_dimensions, true).unwrap();
        let rhs_tensor =
            Tensor::from_values(backend.clone(), rhs.clone(), rhs_dimensions, true).unwrap();

        let (result_tensor, expected_grad_lhs, expected_grad_rhs) = match op {
            BinaryOperator::Add => {
                let result_tensor = lhs_tensor.add(&rhs_tensor).unwrap();
                let expected_grad_lhs = vec![1.0; lhs.len()];
                let expected_grad_rhs = vec![40.0; rhs.len()];
                (result_tensor, expected_grad_lhs, expected_grad_rhs)
            }
            BinaryOperator::Subtract => {
                let result_tensor = lhs_tensor.sub(&rhs_tensor).unwrap();
                let expected_grad_lhs = vec![1.0; lhs.len()];
                let expected_grad_rhs = vec![-40.0; rhs.len()];
                (result_tensor, expected_grad_lhs, expected_grad_rhs)
            }
            BinaryOperator::Multiply => {
                let result_tensor = lhs_tensor.mul(&rhs_tensor).unwrap();
                let expected_grad_lhs = vec![
                    3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                    3.0, 3.0, 3.0, 3.0, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5,
                    4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                    6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 3.0, 3.0, 3.0, 3.0,
                    3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                    4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5,
                    4.5, 4.5, 4.5, 4.5, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                    6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                ];
                let expected_grad_rhs = vec![-0.168_067_23, 20.000000000000004, 40.168_068];
                (result_tensor, expected_grad_lhs, expected_grad_rhs)
            }
            BinaryOperator::Divide => {
                let result_tensor = lhs_tensor.div(&rhs_tensor).unwrap();
                let expected_grad_lhs = vec![
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.333_333_34,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.222_222_22,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                    0.166_666_67,
                ];
                let expected_grad_rhs = vec![0.018_674_137, -0.987_654_3, -1.115_779_6];
                (result_tensor, expected_grad_lhs, expected_grad_rhs)
            }
            _ => todo!(),
        };
        let gradients = result_tensor.backward().unwrap();

        let lhs_gradients = gradients
            .get(lhs_tensor.id())
            .expect("Predictions should have grad.")
            .get_values()
            .unwrap();

        let rhs_gradients = gradients
            .get(rhs_tensor.id())
            .expect("Predictions should have grad.")
            .get_values()
            .unwrap();

        assert!(compare_outputs(&lhs_gradients, &expected_grad_lhs));
        assert!(compare_outputs(&rhs_gradients, &expected_grad_rhs));
    }

    #[test]
    fn test_binary_broadcast_backward() {
        tensor_binary_broadcast_backwards(
            Rc::new(RefCell::new(WgpuBackend::default())),
            BinaryOperator::Add,
        );
        tensor_binary_broadcast_backwards(
            Rc::new(RefCell::new(CpuBackend::default())),
            BinaryOperator::Add,
        );
        tensor_binary_broadcast_backwards(
            Rc::new(RefCell::new(WgpuBackend::default())),
            BinaryOperator::Subtract,
        );
        tensor_binary_broadcast_backwards(
            Rc::new(RefCell::new(CpuBackend::default())),
            BinaryOperator::Subtract,
        );
        tensor_binary_broadcast_backwards(
            Rc::new(RefCell::new(WgpuBackend::default())),
            BinaryOperator::Multiply,
        );
        tensor_binary_broadcast_backwards(
            Rc::new(RefCell::new(CpuBackend::default())),
            BinaryOperator::Multiply,
        );
        tensor_binary_broadcast_backwards(
            Rc::new(RefCell::new(WgpuBackend::default())),
            BinaryOperator::Divide,
        );
        tensor_binary_broadcast_backwards(
            Rc::new(RefCell::new(CpuBackend::default())),
            BinaryOperator::Divide,
        );
    }

    fn tensor_exp_backward<B: Backend>(backend: Rc<RefCell<B>>) {
        let input_dimensions = Dimensions4D::new(2, 3, 4, 5);
        let input = linspace(-1.0, 2.0, input_dimensions.size());

        let input_tensor =
            Tensor::from_values(backend.clone(), input.clone(), input_dimensions, true).unwrap();

        let exp_tensor = input_tensor.exp().unwrap();
        let gradients = exp_tensor.backward().unwrap();

        let exp_gradients = gradients
            .get(input_tensor.id())
            .expect("Predictions should have grad.")
            .get_values()
            .unwrap();

        let expected_values = exp_tensor.get_values().unwrap();

        // the input grad should be the same as the output of exp!!!
        assert!(compare_outputs(&exp_gradients, &expected_values));
    }

    #[test]
    fn test_exp_backward() {
        tensor_exp_backward(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_exp_backward(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_ln_backward<B: Backend>(backend: Rc<RefCell<B>>) {
        let input_dimensions = Dimensions4D::new(2, 3, 4, 5);
        let input = linspace(-1.0, 2.0, input_dimensions.size());

        let input_tensor =
            Tensor::from_values(backend.clone(), input.clone(), input_dimensions, true).unwrap();

        let ln_tensor = input_tensor.ln().unwrap();
        let gradients = ln_tensor.backward().unwrap();

        let ln_gradients = gradients
            .get(input_tensor.id())
            .expect("Predictions should have grad.")
            .get_values()
            .unwrap();

        let expected_values = input.iter().map(|x| 1.0 / x).collect::<Vec<_>>();

        assert!(compare_outputs(&ln_gradients, &expected_values));
    }

    #[test]
    fn test_ln_backward() {
        tensor_ln_backward(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_ln_backward(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_gather_backward<B: Backend>(backend: Rc<RefCell<B>>) {
        let dimensions = Dimensions4D::new(32, 1, 1, 4);
        let index_dimensions = Dimensions4D::new(32, 1, 1, 1);
        let values = linspace(1.0, 4.0, dimensions.size());
        let mut rng = rand::thread_rng();
        let indices = (0..32)
            .map(|_| rng.gen_range(0..4) as f32)
            .collect::<Vec<_>>();

        let input_tensor =
            Tensor::from_values(backend.clone(), values.clone(), dimensions, true).unwrap();
        let index_tensor =
            Tensor::from_values(backend, indices.clone(), index_dimensions, false).unwrap();

        let gather_tensor = input_tensor.gather(3, &index_tensor).unwrap();
        let gradients = gather_tensor.backward().unwrap();

        let gather_gradients = gradients
            .get(input_tensor.id())
            .expect("Predictions should have grad.")
            .get_values()
            .unwrap();

        let mut expected_values = vec![0.0; dimensions.size()];
        expected_values
            .chunks_mut(dimensions.width)
            .zip(indices)
            .for_each(|(chunk, gather_index)| {
                chunk[gather_index as usize] = 1.0;
            });

        assert!(compare_outputs(&gather_gradients, &expected_values));
    }

    #[test]
    fn test_gather_backward() {
        tensor_gather_backward(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_gather_backward(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_max_dim_backward<B: Backend>(backend: Rc<RefCell<B>>) {
        let dimensions = Dimensions4D::new(32, 1, 1, 4);
        let values = linspace(1.0, 4.0, dimensions.size());

        let input_tensor =
            Tensor::from_values(backend.clone(), values.clone(), dimensions, true).unwrap();

        let max_dim_tensor = input_tensor.max_dim(3).unwrap();
        let gradients = max_dim_tensor.backward().unwrap();

        let max_dim_gradients = gradients
            .get(input_tensor.id())
            .expect("Predictions should have grad.")
            .get_values()
            .unwrap();

        let mut expected_values = vec![0.0; dimensions.size()];
        expected_values
            .chunks_mut(dimensions.width)
            .zip(values.chunks(dimensions.width))
            .for_each(|(chunk, values)| {
                let gather_index = values
                    .iter()
                    .enumerate()
                    .max_by(|(_, x), (_, y)| x.total_cmp(y))
                    .map(|(i, _)| i)
                    .unwrap();
                chunk[gather_index] = 1.0;
            });

        assert!(compare_outputs(&max_dim_gradients, &expected_values));
    }

    #[test]
    fn test_max_dim_backward() {
        tensor_max_dim_backward(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_max_dim_backward(Rc::new(RefCell::new(CpuBackend::default())));
    }
}
