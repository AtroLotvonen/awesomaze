use core::fmt;
use std::{
    cell::RefCell,
    fmt::Debug,
    rc::Rc,
    sync::atomic::{self, AtomicUsize},
};

use super::{
    backend::{Backend, CmpType},
    dimensions::{Dimensions2D, Dimensions4D},
    model_config::{ActivationType, Padding, PoolType},
    tensor_error::{Result, TensorError},
};

// #[derive(Clone)]
// Raw tensor shouldn't implement clone
#[derive(Debug)]
struct RawTensor<B: Backend> {
    id: usize,
    dimensions: Dimensions4D,
    view_dimensions: Option<Dimensions4D>,
    has_grad: bool,
    backpropagation_op: Option<Operation<B>>,
    // These 2 need interior mutability: The buffer itself when updating the values when training
    // | and the backend because tensorops require ability to queue computations from the backend |
    storage: Rc<B::Storage>,
    // This might not be needed
    backend: Rc<RefCell<B>>,
}

#[derive(Debug)]
pub struct Tensor<B: Backend>(Rc<RawTensor<B>>);

impl<B: Backend> Clone for Tensor<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

static COUNTER: AtomicUsize = AtomicUsize::new(1);

impl<B: Backend> Tensor<B> {
    pub fn new(
        dimensions: Dimensions4D,
        has_grad: bool,
        back_propagation_op: Option<Operation<B>>,
        storage: Rc<B::Storage>,
        backend: Rc<RefCell<B>>,
    ) -> Self {
        let id = COUNTER.fetch_add(1, atomic::Ordering::Relaxed);
        // let back_propagation_op = BackPropOp(back_propagation_op);
        let raw = RawTensor {
            id,
            dimensions,
            view_dimensions: None,
            has_grad,
            backpropagation_op: back_propagation_op,
            storage,
            backend,
        };
        Self(Rc::new(raw))
    }

    fn new_with_id(
        id: usize,
        dimensions: Dimensions4D,
        has_grad: bool,
        back_propagation_op: Option<Operation<B>>,
        storage: Rc<B::Storage>,
        backend: Rc<RefCell<B>>,
    ) -> Self {
        let raw = RawTensor {
            id,
            dimensions,
            view_dimensions: None,
            has_grad,
            backpropagation_op: back_propagation_op,
            storage,
            backend,
        };
        Self(Rc::new(raw))
    }

    fn new_view(
        dimensions: Dimensions4D,
        view_dimensions: Option<Dimensions4D>,
        has_grad: bool,
        back_propagation_op: Option<Operation<B>>,
        storage: Rc<B::Storage>,
        backend: Rc<RefCell<B>>,
    ) -> Self {
        let id = COUNTER.fetch_add(1, atomic::Ordering::Relaxed);
        // let back_propagation_op = BackPropOp(back_propagation_op);
        let raw = RawTensor {
            id,
            dimensions,
            view_dimensions,
            has_grad,
            backpropagation_op: back_propagation_op,
            storage,
            backend,
        };
        Self(Rc::new(raw))
    }

    pub fn id(&self) -> usize {
        self.0.id
    }

    pub fn dimensions(&self) -> Dimensions4D {
        self.0.dimensions
    }

    pub fn view_dimensions(&self) -> Option<Dimensions4D> {
        self.0.view_dimensions
    }

    pub fn backend(&self) -> Rc<RefCell<B>> {
        self.0.backend.clone()
    }

    pub(self) fn storage(&self) -> &Rc<B::Storage> {
        &self.0.storage
    }

    pub fn has_grad(&self) -> bool {
        self.0.has_grad
    }

    pub fn back_propagation_op(&self) -> &Option<Operation<B>> {
        &self.0.backpropagation_op
    }

    pub fn replace_storage(&mut self, other: Self) {
        let new_tensor = Self::new_with_id(
            self.id(),
            other.dimensions(),
            self.has_grad(),
            None,
            other.0.storage.clone(),
            self.backend().clone(),
        );
        self.0 = new_tensor.0;
    }

    pub fn detach(&mut self) -> Self {
        Self::new_with_id(
            self.id(),
            self.dimensions(),
            self.has_grad(),
            None,
            self.0.storage.clone(),
            self.backend().clone(),
        )
    }

    pub fn zeros_like(other: &Self) -> Result<Self> {
        let buffer = other.backend().borrow_mut().zeros(other.dimensions())?;
        Ok(Self::new(
            other.dimensions(),
            other.has_grad(),
            None,
            Rc::new(buffer),
            other.backend(),
        ))
    }

    pub fn zeros(
        backend: Rc<RefCell<B>>,
        dimensions: Dimensions4D,
        has_grad: bool,
    ) -> Result<Self> {
        let buffer = backend.borrow_mut().zeros(dimensions)?;
        Ok(Self::new(
            dimensions,
            has_grad,
            None,
            Rc::new(buffer),
            backend,
        ))
    }

    pub fn ones(backend: Rc<RefCell<B>>, dimensions: Dimensions4D, has_grad: bool) -> Result<Self> {
        let buffer = backend.borrow_mut().ones(dimensions)?;
        Ok(Self::new(
            dimensions,
            has_grad,
            None,
            Rc::new(buffer),
            backend,
        ))
    }

    pub fn from_values(
        backend: Rc<RefCell<B>>,
        values: Vec<f32>,
        dimensions: Dimensions4D,
        has_grad: bool,
    ) -> Result<Self> {
        let buffer = backend
            .borrow_mut()
            .buffer_from_values(dimensions, values)?;
        Ok(Self::new(
            dimensions,
            has_grad,
            None,
            Rc::new(buffer),
            backend,
        ))
    }

    pub fn exp(&self) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();
        let result = backend.exp(self.storage(), self.dimensions())?;
        let backprop_op = Operation::Exp(self.clone());
        let has_grad = self.has_grad();
        Ok(Self::new(
            self.dimensions(),
            has_grad,
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }

    pub fn ln(&self) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();
        let result = backend.ln(self.storage(), self.dimensions())?;
        let backprop_op = Operation::Ln(self.clone());
        let has_grad = self.has_grad();
        Ok(Self::new(
            self.dimensions(),
            has_grad,
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }

    pub fn squared(&self) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();
        let pow_value = vec![2.0];
        let pow_buffer = backend.buffer_from_values(Dimensions4D::new(1, 1, 1, 1), pow_value)?;
        let result = backend.pow(
            self.storage(),
            &pow_buffer,
            self.dimensions(),
            Dimensions4D::new(1, 1, 1, 1),
        )?;
        let backprop_op = Operation::Squared(self.clone());
        let has_grad = self.has_grad();
        Ok(Self::new(
            self.dimensions(),
            has_grad,
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }

    pub fn pow(&self, rhs: &Self) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();

        let new_dimensions = self.dimensions().broadcast_dimensions(rhs.dimensions())?;
        let lhs_broadcasted;
        let rhs_broadcasted;
        let (lhs, rhs) = {
            let lhs = match self.dimensions() == new_dimensions {
                true => self,
                false => {
                    lhs_broadcasted = self.broadcast(new_dimensions)?;
                    &lhs_broadcasted
                }
            };
            let rhs = match rhs.dimensions() == new_dimensions {
                true => rhs,
                false => {
                    rhs_broadcasted = rhs.broadcast(new_dimensions)?;
                    &rhs_broadcasted
                }
            };
            (lhs, rhs)
        };
        let result = backend.pow(
            lhs.storage(),
            rhs.storage(),
            lhs.dimensions(),
            rhs.dimensions(),
        )?;
        let backprop_op = Operation::Pow(lhs.clone(), rhs.clone());
        let has_grad = self.has_grad() || rhs.has_grad();
        Ok(Self::new(
            new_dimensions,
            has_grad,
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();

        let new_dimensions = self.dimensions().broadcast_dimensions(rhs.dimensions())?;
        let lhs_broadcasted;
        let rhs_broadcasted;
        let (lhs, rhs) = {
            let lhs = match self.dimensions() == new_dimensions {
                true => self,
                false => {
                    lhs_broadcasted = self.broadcast(new_dimensions)?;
                    &lhs_broadcasted
                }
            };
            let rhs = match rhs.dimensions() == new_dimensions {
                true => rhs,
                false => {
                    rhs_broadcasted = rhs.broadcast(new_dimensions)?;
                    &rhs_broadcasted
                }
            };
            (lhs, rhs)
        };
        let result = backend.mul(
            lhs.storage(),
            rhs.storage(),
            lhs.dimensions(),
            rhs.dimensions(),
        )?;
        let backprop_op = Operation::Mul(lhs.clone(), rhs.clone());
        let has_grad = self.has_grad() || rhs.has_grad();
        Ok(Self::new(
            new_dimensions,
            has_grad,
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }

    pub fn div(&self, rhs: &Self) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();

        let new_dimensions = self.dimensions().broadcast_dimensions(rhs.dimensions())?;
        let lhs_broadcasted;
        let rhs_broadcasted;
        let (lhs, rhs) = {
            let lhs = match self.dimensions() == new_dimensions {
                true => self,
                false => {
                    lhs_broadcasted = self.broadcast(new_dimensions)?;
                    &lhs_broadcasted
                }
            };
            let rhs = match rhs.dimensions() == new_dimensions {
                true => rhs,
                false => {
                    rhs_broadcasted = rhs.broadcast(new_dimensions)?;
                    &rhs_broadcasted
                }
            };
            (lhs, rhs)
        };
        let result = backend.div(
            lhs.storage(),
            rhs.storage(),
            lhs.dimensions(),
            rhs.dimensions(),
        )?;
        let backprop_op = Operation::Div(lhs.clone(), rhs.clone());
        let has_grad = self.has_grad() || rhs.has_grad();
        Ok(Self::new(
            new_dimensions,
            has_grad,
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();

        let new_dimensions = self.dimensions().broadcast_dimensions(rhs.dimensions())?;
        let lhs_broadcasted;
        let rhs_broadcasted;
        let (lhs, rhs) = {
            let lhs = match self.dimensions() == new_dimensions {
                true => self,
                false => {
                    lhs_broadcasted = self.broadcast(new_dimensions)?;
                    &lhs_broadcasted
                }
            };
            let rhs = match rhs.dimensions() == new_dimensions {
                true => rhs,
                false => {
                    rhs_broadcasted = rhs.broadcast(new_dimensions)?;
                    &rhs_broadcasted
                }
            };
            (lhs, rhs)
        };
        let result = backend.add(
            lhs.storage(),
            rhs.storage(),
            lhs.dimensions(),
            rhs.dimensions(),
        )?;
        let backprop_op = Operation::Add(lhs.clone(), rhs.clone());
        let has_grad = self.has_grad() || rhs.has_grad();
        Ok(Self::new(
            new_dimensions,
            has_grad,
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();

        let new_dimensions = self.dimensions().broadcast_dimensions(rhs.dimensions())?;
        let lhs_broadcasted;
        let rhs_broadcasted;
        let (lhs, rhs) = {
            let lhs = match self.dimensions() == new_dimensions {
                true => self,
                false => {
                    lhs_broadcasted = self.broadcast(new_dimensions)?;
                    &lhs_broadcasted
                }
            };
            let rhs = match rhs.dimensions() == new_dimensions {
                true => rhs,
                false => {
                    rhs_broadcasted = rhs.broadcast(new_dimensions)?;
                    &rhs_broadcasted
                }
            };
            (lhs, rhs)
        };
        let result = backend.sub(
            lhs.storage(),
            rhs.storage(),
            lhs.dimensions(),
            rhs.dimensions(),
        )?;
        let backprop_op = Operation::Sub(lhs.clone(), rhs.clone());
        let has_grad = self.has_grad() || rhs.has_grad();
        Ok(Self::new(
            new_dimensions,
            has_grad,
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }

    pub fn activation(&self, activation: ActivationType) -> Result<Self> {
        let result = match activation {
            ActivationType::Relu => {
                let mut backend = self.0.backend.borrow_mut();
                backend.relu(self.storage(), self.dimensions())?
            }
            ActivationType::LeakyRelu(alpha) => {
                let mut backend = self.0.backend.borrow_mut();
                backend.leaky_relu(self.storage(), self.dimensions(), alpha)?
            }
            ActivationType::Sigmoid => {
                let mut backend = self.0.backend.borrow_mut();
                backend.sigmoid(self.storage(), self.dimensions())?
            }
            ActivationType::Softmax => return self.softmax(),
        };
        let backprop_op = Operation::Activation(self.clone(), activation);
        Ok(Self::new(
            self.dimensions(),
            self.has_grad(),
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }

    /// Matmul operation for tensor. Both tensors shapes in batch and depth have to be explicitely
    /// 1 for now. Also the k must match. If rhs is transposed the k is expected to be the width
    /// for both tensors.
    pub fn matmul(&self, rhs: &Self, is_rhs_transposed: bool) -> Result<Self> {
        // NOTE: don't allow implicit batch size or depth in matmul for now, but an explicit
        // reshape, since
        if self.dimensions().batch != 1 || self.dimensions().depth != 1 {
            return Err(TensorError::TensorDimensionMismatch {
                tensor_op_name: "matmul lhs".into(),
                expected: Dimensions4D::new(1, 1, 0, 0),
                received: Dimensions4D::new(self.dimensions().batch, self.dimensions().depth, 0, 0),
            });
        }
        if rhs.dimensions().batch != 1 || rhs.dimensions().depth != 1 {
            return Err(TensorError::TensorDimensionMismatch {
                tensor_op_name: "matmul rhs".into(),
                expected: Dimensions4D::new(1, 1, 0, 0),
                received: Dimensions4D::new(rhs.dimensions().batch, rhs.dimensions().depth, 0, 0),
            });
        }

        let k1 = self.dimensions().width;
        let (k2, n) = match is_rhs_transposed {
            true => (rhs.dimensions().width, rhs.dimensions().height),
            false => (rhs.dimensions().height, rhs.dimensions().width),
        };
        if k1 != k2 {
            return match is_rhs_transposed {
                true => Err(TensorError::TensorDimensionMismatch {
                    tensor_op_name: "matmul (rhs transposed)".into(),
                    expected: Dimensions4D::new(0, 0, 0, k1),
                    received: Dimensions4D::new(0, 0, 0, k2),
                }),
                false => Err(TensorError::TensorDimensionMismatch {
                    tensor_op_name: "matmul".into(),
                    expected: Dimensions4D::new(0, 0, k1, 0),
                    received: Dimensions4D::new(0, 0, k2, 0),
                }),
            };
        }
        let m = self.dimensions().height;
        let mut backend = self.0.backend.borrow_mut();
        let result =
            backend.matmul(self.storage(), rhs.storage(), (m, n, k1), is_rhs_transposed)?;
        let out_dimensions = Dimensions4D::new(1, 1, m, n);
        let backprop_op = Operation::Matmul {
            lhs: self.clone(),
            rhs: rhs.clone(),
            is_rhs_transposed,
        };
        let has_grad = self.has_grad() || rhs.has_grad();
        Ok(Self::new(
            out_dimensions,
            has_grad,
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }

    pub fn reshape(&self, new_dimensions: Dimensions4D) -> Result<Self> {
        if new_dimensions.size() != self.dimensions().size() {
            return Err(TensorError::TensorDimensionMismatch {
                tensor_op_name: "reshape shape".into(),
                expected: self.dimensions(),
                received: new_dimensions,
            });
        }
        let backprop_op = Operation::Reshape(self.clone());
        Ok(Self::new(
            new_dimensions,
            self.has_grad(),
            Some(backprop_op),
            self.storage().clone(), // NOTE: Same buffer used
            self.0.backend.clone(),
        ))
    }

    /// Flattens the C, H, W
    ///
    /// # Arguments
    ///
    /// * `argument_name` - type and description.
    ///
    /// # Returns
    /// type and description of the returned object.
    ///
    /// # Examples
    pub fn flatten(&self) -> Result<Self> {
        let new_width =
            self.dimensions().depth * self.dimensions().height * self.dimensions().width;
        let new_dimension = Dimensions4D::new(self.dimensions().batch, 1, 1, new_width);
        self.reshape(new_dimension)
    }

    pub fn broadcast(&self, broadcast_dimensions: Dimensions4D) -> Result<Self> {
        let view_dimension = self
            .dimensions()
            .broadcast_dimensions(broadcast_dimensions)?;
        let backprop_op = Operation::Broadcast(self.clone(), view_dimension);
        Ok(Self::new_view(
            self.dimensions(),
            Some(view_dimension),
            self.has_grad(),
            Some(backprop_op),
            self.storage().clone(),
            self.0.backend.clone(),
        ))
    }

    /// Transposes the width and height of the `Tensor`
    ///
    /// # Returns
    /// A new `Tensor` where the width and height are transposed. Error if depth > 1.
    /// WARN: Doesn't support depth in tensor dimensions yet. Supports Batching though. If Tensor
    /// has depth, reshape it first to either batches, height or width.
    ///
    /// # Examples
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        if !(0..=3).contains(&dim1) {
            return Err(TensorError::DimensionIndexError { index: dim1 });
        }
        if !(0..=3).contains(&dim2) {
            return Err(TensorError::DimensionIndexError { index: dim2 });
        }
        // dimension reorder
        let (dim1, dim2) = match dim1.cmp(&dim2) {
            std::cmp::Ordering::Less => (dim1, dim2),
            std::cmp::Ordering::Greater => (dim2, dim1),
            std::cmp::Ordering::Equal => {
                return Err(TensorError::TransposeDimensionError { index: dim1 })
            }
        };

        let input_dimensions = self.dimensions();
        let output_dimensions = input_dimensions.swap(dim1, dim2)?;
        let mut backend = self.0.backend.borrow_mut();
        let result = backend.transpose(self.storage(), input_dimensions, dim1, dim2)?;
        let backprop_op = Operation::Transpose(self.clone(), (dim1, dim2));
        Ok(Self::new(
            output_dimensions,
            self.has_grad(),
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }

    pub fn conv2d(&self, kernels: &Self, padding: &Padding) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();
        let (padded, input_dimensions) = match padding {
            Padding::Zero(x, y) => {
                let offset = Dimensions4D::new(0, 0, *y, *x);
                let new_size = Dimensions4D::new(
                    self.dimensions().batch,
                    self.dimensions().depth,
                    self.dimensions().height + 2 * y,
                    self.dimensions().width + 2 * x,
                );
                (
                    Rc::new(backend.zero_pad(
                        self.storage(),
                        self.dimensions(),
                        offset,
                        new_size,
                    )?),
                    new_size,
                )
            }
            Padding::None => (self.storage().clone(), self.dimensions()),
        };
        let result = backend.conv2d(
            // self.storage(),
            &padded,
            kernels.storage(),
            input_dimensions,
            // self.dimensions(),
            kernels.dimensions(),
        )?;
        let backprop_op = Operation::Conv2d {
            input: self.clone(),
            kernel: kernels.clone(),
            padding: *padding,
        };
        let has_grad = self.has_grad() || kernels.has_grad();
        let output_height = input_dimensions.height - kernels.dimensions().height + 1;
        let output_width = input_dimensions.width - kernels.dimensions().width + 1;
        let output_dimensions = Dimensions4D::new(
            self.dimensions().batch,
            kernels.dimensions().batch,
            output_height,
            output_width,
        );
        Ok(Self::new(
            output_dimensions,
            has_grad,
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }

    pub fn conv_transpose2d(&self, kernels: &Self, input_padding: &Padding) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();
        // TODO: zero padding is necessary only when upsampling and is controlled with stride
        let (padded, input_dimensions) = (self.storage().clone(), self.dimensions());
        let result = backend.conv_transpose2d(
            // self.storage(),
            &padded,
            kernels.storage(),
            input_dimensions,
            // self.dimensions(),
            kernels.dimensions(),
        )?;
        let backprop_op = Operation::ConvTranspose2d {
            input: self.clone(),
            kernel: kernels.clone(),
            input_padding: *input_padding,
        };
        let has_grad = self.has_grad() || kernels.has_grad();
        let output_height = input_dimensions.height + kernels.dimensions().height - 1;
        let output_width = input_dimensions.width + kernels.dimensions().width - 1;
        let output_dimensions = Dimensions4D::new(
            self.dimensions().batch,
            kernels.dimensions().depth,
            output_height,
            output_width,
        );
        let (result, output_dimensions) = match input_padding {
            Padding::Zero(x, y) => {
                let offset = Dimensions4D::new(0, 0, *y, *x);
                let new_size = Dimensions4D::new(
                    output_dimensions.batch,
                    output_dimensions.depth,
                    output_dimensions.height - 2 * y,
                    output_dimensions.width - 2 * x,
                );
                (
                    backend.slice(&result, output_dimensions, offset, new_size)?,
                    new_size,
                )
            }
            Padding::None => (result, output_dimensions),
        };
        Ok(Self::new(
            output_dimensions,
            has_grad,
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }

    pub fn pool(&self, pool_type: PoolType, pool_dimensions: Dimensions2D) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();
        let result = match pool_type {
            PoolType::MaxPool2D => {
                backend.max_pool2d(self.storage(), self.dimensions(), pool_dimensions)?
            }
            PoolType::AvgPool2D => {
                backend.avg_pool2d(self.storage(), self.dimensions(), pool_dimensions)?
            }
        };
        let backprop_op = Operation::Pool(self.clone(), pool_type, pool_dimensions);
        let has_grad = self.has_grad();
        let new_dimensions = Dimensions4D::new(
            self.dimensions().batch,
            self.dimensions().depth,
            self.dimensions().height / pool_dimensions.height,
            self.dimensions().width / pool_dimensions.width,
        );
        Ok(Self::new(
            new_dimensions,
            has_grad,
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }

    pub fn argmax(&self, dim: usize) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();
        let result_sum = backend.arg_max(self.storage(), self.dimensions(), dim)?;
        let has_grad = self.has_grad();
        let backprop_op = Operation::ArgMax(self.clone(), dim);
        let reduced_dimensions = self.dimensions().reduce_dims(&[dim])?;
        Ok(Self::new(
            reduced_dimensions,
            has_grad,
            Some(backprop_op),
            result_sum.into(),
            self.0.backend.clone(),
        ))
    }

    pub fn max_dim(&self, dim: usize) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();
        let result_sum = backend.max_dim(self.storage(), self.dimensions(), dim)?;
        let has_grad = self.has_grad();
        let backprop_op = Operation::MaxDim(self.clone(), dim);
        let reduced_dimensions = self.dimensions().reduce_dims(&[dim])?;
        Ok(Self::new(
            reduced_dimensions,
            has_grad,
            Some(backprop_op),
            result_sum.into(),
            self.0.backend.clone(),
        ))
    }

    pub fn gather(&self, dim: usize, indices: &Self) -> Result<Self> {
        let reduced_dimensions = self.dimensions().reduce_dims(&[dim])?;
        if indices.dimensions() != reduced_dimensions {
            return Err(TensorError::TensorDimensionMismatch {
                tensor_op_name: "gather".into(),
                expected: reduced_dimensions,
                received: indices.dimensions(),
            });
        }
        let mut backend = self.0.backend.borrow_mut();
        let result_sum =
            backend.gather(self.storage(), indices.storage(), self.dimensions(), dim)?;
        let has_grad = self.has_grad();
        let backprop_op = Operation::Gather {
            input: self.clone(),
            indices: indices.clone(),
            dim,
        };
        Ok(Self::new(
            reduced_dimensions,
            has_grad,
            Some(backprop_op),
            result_sum.into(),
            self.0.backend.clone(),
        ))
    }

    pub fn scatter(
        &self,
        dim: usize,
        indices: &Self,
        out_dimensions: Dimensions4D,
    ) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();
        let result_sum = backend.scatter(
            self.storage(),
            indices.storage(),
            self.dimensions(),
            out_dimensions,
            dim,
        )?;
        let has_grad = self.has_grad();
        let backprop_op = Operation::Scatter {
            input: self.clone(),
            indices: indices.clone(),
            dim,
        };
        Ok(Self::new(
            out_dimensions,
            has_grad,
            Some(backprop_op),
            result_sum.into(),
            self.0.backend.clone(),
        ))
    }

    pub fn get_values(&self) -> Result<Vec<f32>> {
        let mut backend = self.0.backend.borrow_mut();
        backend.get_values(self.storage(), self.dimensions())
    }

    pub fn mse(&self, gt: &Self) -> Result<Self> {
        if self.dimensions() != gt.dimensions() {
            return Err(TensorError::TensorDimensionMismatch {
                tensor_op_name: "mse".into(),
                expected: gt.dimensions(),
                received: self.dimensions(),
            });
        }

        let div_value = vec![self.dimensions().size() as f32];
        // TODO: these should really be just done with scalars
        let div_buffer = {
            let mut backend = self.0.backend.borrow_mut();
            backend.buffer_from_values(Dimensions4D::new(1, 1, 1, 1), div_value)?
        };

        let div_tensor = Self::new(
            Dimensions4D::new(1, 1, 1, 1),
            false,
            None,
            Rc::new(div_buffer),
            self.0.backend.clone(),
        );
        let diff_tensor = self.sub(gt)?;

        let squared = diff_tensor.squared()?;

        let averaged = squared.div(&div_tensor)?;
        // let mut backend = self.0.backend.borrow_mut();
        // let batch_reduced = backend.batch_reduce_sum(averaged.storage(), gt.dimensions())?;
        // let batch_reduced = averaged.sum(Some(&[0]))?;
        let batch_reduced = averaged.sum(None)?;
        // let sum = batch_reduced.sum(None)?;
        Ok(batch_reduced)
    }

    pub fn softmax(&self) -> Result<Self> {
        // TODO: implement max and do shift for logits

        let exp_tensor = self.exp()?;
        let sum_exp = exp_tensor.sum(Some(&[1, 2, 3]))?;

        let normalized = exp_tensor.div(&sum_exp)?;

        Ok(normalized)
    }

    pub fn sum(&self, dims: Option<&[usize]>) -> Result<Self> {
        let all_dims = [0, 1, 2, 3];
        let (reduced_dimensions, dims) = match dims {
            Some(dims) => (self.dimensions().reduce_dims(dims)?, dims),
            None => (Dimensions4D::new(1, 1, 1, 1), &all_dims[..]),
        };
        let mut backend = self.0.backend.borrow_mut();
        let result_sum = backend.sum_reduce(self.storage(), self.dimensions(), dims)?;
        let has_grad = self.has_grad();
        let backprop_op = Operation::ReduceSum(self.clone());
        Ok(Self::new(
            reduced_dimensions,
            has_grad,
            Some(backprop_op),
            result_sum.into(),
            self.0.backend.clone(),
        ))
    }

    // TODO: implement max
    // pub fn max(&self, dims: Option<&[usize]>) -> Result<Self> {
    //     let all_dims = [0, 1, 2, 3];
    //     let (reduced_dimensions, dims) = match dims {
    //         Some(dims) => (self.dimensions().reduce_dims(dims)?, dims),
    //         None => (Dimensions4D::new(1, 1, 1, 1), &all_dims[..]),
    //     };
    //     let mut backend = self.0.backend.borrow_mut();
    //     let result_sum = backend.sum_reduce(self.storage(), self.dimensions(), dims)?;
    //     let has_grad = self.has_grad();
    //     let backprop_op = Operation::ReduceSum(self.clone());
    //     Ok(Self::new(
    //         reduced_dimensions,
    //         has_grad,
    //         Some(backprop_op),
    //         result_sum.into(),
    //         self.0.backend.clone(),
    //     ))
    // }

    pub fn greater_than(&self, rhs: &Self) -> Result<Self> {
        self.compare(rhs, CmpType::GreaterThan)
    }

    pub fn lesser_than(&self, rhs: &Self) -> Result<Self> {
        self.compare(rhs, CmpType::LesserThan)
    }

    pub fn equal(&self, rhs: &Self) -> Result<Self> {
        self.compare(rhs, CmpType::Equal)
    }

    pub fn greater_or_equal(&self, rhs: &Self) -> Result<Self> {
        self.compare(rhs, CmpType::GreaterOrEqualThan)
    }

    pub fn lesser_or_equal(&self, rhs: &Self) -> Result<Self> {
        self.compare(rhs, CmpType::LesserOrEqualThan)
    }

    pub fn not_equal(&self, rhs: &Self) -> Result<Self> {
        self.compare(rhs, CmpType::NotEqual)
    }

    pub fn upsample2d(&self, upsample_dimensions: Dimensions2D) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();
        let result = backend.upsample2d(self.storage(), self.dimensions(), upsample_dimensions)?;
        let backprop_op = Operation::Upsample(self.clone(), upsample_dimensions);
        let has_grad = self.has_grad();
        let upsampled_dimensions = Dimensions4D::new(
            self.dimensions().batch,
            self.dimensions().depth,
            self.dimensions().height * upsample_dimensions.height,
            self.dimensions().width * upsample_dimensions.width,
        );
        Ok(Self::new(
            upsampled_dimensions,
            has_grad,
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }

    fn compare(&self, rhs: &Self, comparison_type: CmpType) -> Result<Self> {
        let mut backend = self.0.backend.borrow_mut();
        let result = backend.comparison(
            self.storage(),
            rhs.storage(),
            self.dimensions(),
            rhs.dimensions(),
            comparison_type.clone(),
        )?;
        let backprop_op = Operation::Cmp(self.clone(), rhs.clone(), comparison_type);
        let has_grad = self.has_grad() || rhs.has_grad();
        Ok(Self::new(
            self.dimensions(),
            has_grad,
            Some(backprop_op),
            Rc::new(result),
            self.0.backend.clone(),
        ))
    }
}

impl<B: Backend> fmt::Display for Tensor<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dimensions = self.dimensions();
        let id = self.id();
        let buffer = self.get_values().ok();
        if let Some(buffer) = buffer {
            writeln!(f, "Tensor with id {id}:")?;
            for n in 0..dimensions.batch {
                writeln!(f, "Batch {n}:")?;
                for c in 0..dimensions.depth {
                    writeln!(f, "  Channel {c}:")?;
                    for h in 0..dimensions.height {
                        write!(f, "    [")?;
                        for w in 0..dimensions.width {
                            if w != 0 {
                                write!(f, ", ")?
                            }
                            let i = n * dimensions.depth * dimensions.height * dimensions.width
                                + c * dimensions.height * dimensions.width
                                + h * dimensions.width
                                + w;
                            let x = buffer[i];
                            write!(f, "{x}")?;
                        }
                        writeln!(f, "]")?;
                    }
                }
            }
        } else {
            writeln!(f, "No storage for buffer with id: {id}")?;
        }
        Ok(())
    }
}

// #[derive(Clone)]
// pub struct BackPropOp<B: Backend>(Option<Operation<B>>);

#[derive(Clone, Debug)]
pub enum Operation<B: Backend> {
    // Binary operations
    Add(Tensor<B>, Tensor<B>),
    Sub(Tensor<B>, Tensor<B>),
    Mul(Tensor<B>, Tensor<B>),
    Div(Tensor<B>, Tensor<B>),
    Pow(Tensor<B>, Tensor<B>),
    Cmp(Tensor<B>, Tensor<B>, CmpType),
    Squared(Tensor<B>),
    Exp(Tensor<B>),
    Ln(Tensor<B>),
    ReduceSum(Tensor<B>),
    ArgMax(Tensor<B>, usize),
    MaxDim(Tensor<B>, usize),
    Gather {
        input: Tensor<B>,
        indices: Tensor<B>,
        dim: usize,
    },
    Scatter {
        input: Tensor<B>,
        indices: Tensor<B>,
        dim: usize,
    },
    Broadcast(Tensor<B>, Dimensions4D),
    Matmul {
        lhs: Tensor<B>,
        rhs: Tensor<B>,
        is_rhs_transposed: bool,
    },
    Conv2d {
        input: Tensor<B>,
        kernel: Tensor<B>,
        padding: Padding,
    },
    ConvTranspose2d {
        input: Tensor<B>,
        kernel: Tensor<B>,
        input_padding: Padding,
    },
    Transpose(Tensor<B>, (usize, usize)),
    Reshape(Tensor<B>),
    // Unary operations
    Activation(Tensor<B>, ActivationType),
    Pool(Tensor<B>, PoolType, Dimensions2D),
    Upsample(Tensor<B>, Dimensions2D),
    // A way to add custom operations, for for example to fused ops
    CustomUnary(Tensor<B>, Rc<dyn CustomUnaryOperation<B>>),
    CustomBinary(Tensor<B>, Tensor<B>, Rc<dyn CustomBinaryOperation<B>>),
}

pub trait CustomUnaryOperation<B: Backend>: Debug {
    fn forward(&mut self, tensor: Tensor<B>) -> Result<B>;
    fn backward(&mut self, tensor: Tensor<B>) -> Result<B>;
}

pub trait CustomBinaryOperation<B: Backend>: Debug {
    fn forward(&mut self, tensor1: Tensor<B>, tensor2: Tensor<B>) -> Result<B>;
    fn backward(&mut self, tensor1: Tensor<B>, tensor2: Tensor<B>) -> Result<B>;
}

#[cfg(test)]
mod test {
    use std::{cell::RefCell, rc::Rc};

    use itertools::Itertools;
    use rand::Rng;

    use crate::machine_learning::{
        compare_outputs,
        nn::{
            backend::Backend,
            backends::{cpu_backend::CpuBackend, wgpu_backend::WgpuBackend},
            dimensions::Dimensions4D,
            linspace,
            model_config::Padding,
        },
    };

    use super::Tensor;

    fn tensor_mse<B: Backend>(backend: Rc<RefCell<B>>) {
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

        // Cpu mse
        let gt_tensor = Tensor::from_values(
            backend.clone(),
            gts.clone(),
            Dimensions4D::new(10, 1, 1, 1),
            false,
        )
        .unwrap();
        let pred_tensor =
            Tensor::from_values(backend, preds.clone(), Dimensions4D::new(10, 1, 1, 1), true)
                .unwrap();

        let mse_tensor = pred_tensor.mse(&gt_tensor).unwrap();
        let mse_result = mse_tensor.get_values().unwrap();

        let expected_mse = vec![1.354_598_3];

        assert!(compare_outputs(&mse_result, &expected_mse));
    }

    #[test]
    fn test_mse() {
        tensor_mse(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_mse(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_sum_reduce<B: Backend>(backend: Rc<RefCell<B>>) {
        let dimensions = Dimensions4D::new(6, 5, 7, 4);
        let values = linspace(1.0, 4.0, dimensions.size());

        let tensor = Tensor::from_values(backend, values, dimensions, false).unwrap();

        let result_tensor = tensor.sum(Some(&[0, 3])).unwrap();
        let result = result_tensor.get_values().unwrap();

        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected_values = [
            54.16448152562574,
            54.507747318236,
            54.851013110846246,
            55.194278903456485,
            55.537544696066746,
            55.880810488677,
            56.224076281287246,

            56.56734207389749,
            56.910607866507746,
            57.253873659118,
            57.597139451728246,
            57.94040524433849,
            58.283671036948746,
            58.626936829559,

            58.970202622169246,
            59.31346841477949,
            59.656734207389746,
            60.0,
            60.34326579261025,
            60.68653158522049,
            61.02979737783073,

            61.373063170441,
            61.71632896305125,
            62.05959475566149,
            62.40286054827176,
            62.746126340882,
            63.08939213349225,
            63.43265792610249,

            63.77592371871275,
            64.119189511323,
            64.46245530393325,
            64.8057210965435,
            65.14898688915376,
            65.492252681764,
            65.83551847437425
        ];

        assert!(compare_outputs(&result, &expected_values))
    }

    #[test]
    fn test_sum_reduce() {
        tensor_sum_reduce(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_sum_reduce(Rc::new(RefCell::new(CpuBackend::default())));
    }

    #[test]
    fn test_pow() {
        let twos = vec![2.0; 10];
        let gpu_backend = Rc::new(RefCell::new(WgpuBackend::default()));
        let two_tensor = Tensor::from_values(
            gpu_backend.clone(),
            twos.clone(),
            Dimensions4D::new(10, 1, 1, 1),
            false,
        )
        .unwrap();
        let pow_tensor =
            Tensor::from_values(gpu_backend, vec![2.0], Dimensions4D::new(1, 1, 1, 1), true)
                .unwrap();

        let squared_tensor = two_tensor.pow(&pow_tensor).unwrap();
        let squared_result = squared_tensor.get_values().unwrap();

        println!("{:?}", squared_result);
    }

    #[test]
    fn test_zeros_like() {
        let tensor_dimensions = Dimensions4D::new(6, 5, 7, 8);
        let values = linspace(1.0, 2.0, tensor_dimensions.size());
        let gpu_backend = Rc::new(RefCell::new(WgpuBackend::default()));
        let tensor = Tensor::from_values(gpu_backend, values, tensor_dimensions, false).unwrap();
        let zero_tensor = Tensor::zeros_like(&tensor).unwrap();
        let zero_values = zero_tensor.get_values().unwrap();

        assert!(compare_outputs(
            &vec![0.0; tensor_dimensions.size()],
            &zero_values
        ))
    }

    fn tensor_conv<B: Backend>(backend: Rc<RefCell<B>>) {
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
        let expected_values = [
             9.07191178,  13.64986509,  13.87932585,   9.27744105,
            13.85351738,  20.83588159,  21.17373793,  14.14773394,
            14.76009854,  22.18730693,  22.52516326,  15.04305322,
            15.6666797,   23.53873226,  23.87658859,  15.9383725,
            10.55566418,  15.85437766,  16.07539202,  10.72740781,
                                                              
            14.39314902,  21.70773863,  22.08923474,  14.80139209,
            22.13944394,  33.37879795,  33.94470731,  22.7377312,
            23.6541665,   35.64243538,  36.20834474,  24.24119188,
            25.16888906,  37.90607281,  38.47198217,  25.74465256,
            17.09318422,  25.7366754,   26.1097251,   17.46764165,
                                                              
                                                              
            27.54139131,  41.18515622,  41.41461698,  27.52168303,
            41.05095218,  61.37864154,  61.71649787,  41.0073124,
            41.95753334,  62.73006687,  63.0679232,   41.90263168,
            42.8641145,   64.0814922,   64.41934853,  42.79795096,
            28.34943104,  42.37609979,  42.59711414,  28.29593712,
                                                              
            45.02545654,  67.48727173,  67.86876784,  45.20846205,
            67.58112071, 101.28792085, 101.85383021,  67.84155164,
            69.09584327, 103.55155828, 104.11746764,  69.34501232,
            70.61056583, 105.81519571, 106.38110507,  70.848473,
            47.04977906,  70.5026395,   70.8756892,   47.19899894,
        ];
        let result = conv2d_result.get_values().unwrap();

        assert!(compare_outputs(&result, &expected_values));
    }

    #[test]
    fn test_conv2d() {
        tensor_conv(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_conv(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_conv2<B: Backend>(backend: Rc<RefCell<B>>) {
        let dimensions = Dimensions4D::new(4, 8, 28, 28);
        let values = linspace(0.0, 1.0, dimensions.size());
        let input_tensor =
            Tensor::from_values(backend.clone(), values.clone(), dimensions, true).unwrap();
        // let values = vec![1.0; dimensions.size()];
        let kernel_dimensions = Dimensions4D::new(2, 8, 28, 28);
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
        let expected_values = [
            1211.34271557272, 1256.3285250800066, 1211.576224473346,
            1259.7285980354025, 1306.510410438341, 1259.96967907339,
            1217.8529414980999, 1263.0787941436652, 1218.0843745993075,

            1936.9425538089422, 2008.9229815282283, 1937.4085522444082,
            2015.5779079713861, 2090.479154209619, 2016.0600892677269,
            1949.9624867098476, 2022.4240578257645, 1950.426409345895,


            3401.7428680695366, 3527.7340590216027, 3401.743887435323,
            3527.8792784892357, 3658.5416567009497, 3527.8792592688715,
            3401.743387019392, 3527.733520851383, 3401.7423305857583,

            5585.517068823408, 5792.509335858498, 5585.750577724033,
            5795.909408813893, 6010.697917912331, 5796.150489851881,
            5592.027294748787, 5799.2596049221565, 5592.258727849996,


            5592.143020566353, 5799.139592963201, 5591.9115503973,
            5796.0299589430715, 6010.572902963555, 5795.788839464351,
            5585.633832540683, 5792.388247559103, 5585.40028657221,

            9234.091583837875, 9576.095690188768, 9234.092603203659,
            9576.240909656402, 9930.916681615043, 9576.240890436035,
            9234.092102787727, 9576.095152018548, 9234.091046354095,


            7782.54317306317, 8070.545126904797, 7782.079213359276,
            8064.180639396905, 8362.604149226161, 8063.698419659833,
            7769.524278061974, 8057.042974266821, 7769.058242558664,

            12882.666098852342, 13359.682044519035, 12882.434628683284,
            13356.572410498906, 13851.135445317761, 13356.33129102019,
            12876.15691082667, 13352.930699114939, 12875.923364858196,
        ];
        let result = conv2d_result.get_values().unwrap();

        assert!(compare_outputs(&result, &expected_values));
    }

    #[test]
    fn test_conv2d_2() {
        tensor_conv2(Rc::new(RefCell::new(CpuBackend::default())));
        tensor_conv2(Rc::new(RefCell::new(WgpuBackend::default())));
    }

    #[test]
    fn compare_conv2d() {
        let image_height = 100;
        let image_width = 120;
        let image_batches = 16;
        let kernel_count = 32;
        let image_depth = 8;
        let kernel_height = 3;
        let kernel_width = 3;

        let dimensions = Dimensions4D::new(image_batches, image_depth, image_height, image_width);
        let _out_dimensions = Dimensions4D::new(
            image_batches,
            kernel_count,
            image_height - (kernel_height / 2) * 2,
            image_width - (kernel_width / 2) * 2,
        );

        let values = linspace(0.0, 1.0, dimensions.size());
        // let values = vec![1.0; dimensions.size()];
        let kernel_dimensions =
            Dimensions4D::new(kernel_count, image_depth, kernel_height, kernel_width);
        // let kernel_values = linspace(1.0, 3.0, kernel_dimensions.size());
        // let kernel_values = vec![1.0; kernel_dimensions.size()];
        let mut kernel_values = vec![0.0; kernel_dimensions.size() / 2];
        let mut ones = vec![1.0; kernel_dimensions.size() / 2];
        kernel_values.append(&mut ones);

        // Cpu
        let backend = Rc::new(RefCell::new(CpuBackend::default()));

        let input_tensor =
            Tensor::from_values(backend.clone(), values.clone(), dimensions, true).unwrap();
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
        let cpu_result = conv2d_result.get_values().unwrap();

        // Wgpu
        let backend = Rc::new(RefCell::new(WgpuBackend::default()));

        let input_tensor =
            Tensor::from_values(backend.clone(), values.clone(), dimensions, true).unwrap();
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
        let gpu_result = conv2d_result.get_values().unwrap();

        assert!(compare_outputs(&cpu_result, &gpu_result));
    }

    fn tensor_conv_transpose2d<B: Backend>(backend: Rc<RefCell<B>>) {
        // input – input tensor of shape (minibatch,in_channels,iH,iW)
        // weight – filters of shape (in_channels,out_channel/groups,kH,kW)
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
        let conv2d_result = input_tensor
            .conv_transpose2d(&kernel_tensor, &Padding::Zero(1, 1))
            .unwrap();
        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected_values = [
            11.868142181206741, 18.01712743909592, 18.30148985257889, 12.347710475892542,
            18.584209908888276, 28.206076721542246, 28.63895514800766, 19.31763969811911,
            19.732921440581862, 29.937590427403897, 30.370468853869315, 20.477613107574392,
            20.881632972275444, 31.669104133265552, 32.10198255973096, 21.63758651702968,
            14.49497516912368, 21.978492941774526, 22.271301763578773, 15.00832909709459,

            12.755015054940756, 19.360106362178858, 19.66980800062566, 13.268368982911664,
            19.965197669416963, 30.296562781058142, 30.767450044969298, 20.749305908575444,
            21.21526610096586, 32.180111836702764, 32.65099910061392, 22.010636217886052,
            22.46533453251476, 34.06366089234739, 34.53454815625855, 23.27196652719665,
            15.58456184256833, 23.625542564423412, 23.943690611191485, 16.131701403824344,


            34.504516482227345, 52.14061705705235, 52.42497947053533, 35.20932233214718,
            53.045555859695774, 80.15148789739177, 80.58436632385718, 54.11684198177765,
            54.194267391389346, 81.88300160325343, 82.31588002971884, 55.27681539123294,
            55.34297892308294, 83.61451530911506, 84.0473937355805, 56.43678880068822,
            37.807062135846394, 57.11555155828412, 57.40836038008837, 38.545653619051336,

            37.41852735306769, 56.524302975794775, 56.83400461424158, 38.15711883627263,
            57.46725061588393, 86.8030344503969, 87.27392171430805, 58.58921518789347,
            58.71731904743283, 88.68658350604153, 89.15747076995268, 59.85054549720408,
            59.967387478981735, 90.57013256168615, 91.0410198255973, 61.111875806514675,
            40.92378680639737, 61.80330817659249, 62.12145622336057, 41.696163922887415,
        ];
        // shape(2, 2, 5, 4)

        let result = conv2d_result.get_values().unwrap();

        assert!(compare_outputs(&result, &expected_values));
    }

    #[test]
    fn test_conv_transpose2d() {
        tensor_conv_transpose2d(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_conv_transpose2d(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_add_broadcast<B: Backend>(backend: Rc<RefCell<B>>) {
        let lhs_dimensions = Dimensions4D::new(2, 3, 4, 5);
        let rhs_dimensions = Dimensions4D::new(1, 3, 1, 1);
        let lhs = linspace(-1.0, 2.0, lhs_dimensions.size());
        let rhs = linspace(3.0, 6.0, rhs_dimensions.size());

        let lhs_tensor =
            Tensor::from_values(backend.clone(), lhs.clone(), lhs_dimensions, true).unwrap();
        let rhs_tensor =
            Tensor::from_values(backend.clone(), rhs.clone(), rhs_dimensions, true).unwrap();

        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected_result = [
            2.0, 2.0252100840336134, 2.0504201680672267, 2.07563025210084, 2.100840336134454,
            2.1260504201680672, 2.1512605042016806, 2.1764705882352944, 2.2016806722689077, 2.226890756302521,
            2.2521008403361344, 2.277310924369748, 2.302521008403361, 2.3277310924369745, 2.3529411764705883,
            2.3781512605042017, 2.403361344537815, 2.428571428571429, 2.453781512605042, 2.4789915966386555,

            4.004201680672269, 4.029411764705882, 4.054621848739496, 4.079831932773109, 4.105042016806722,
            4.130252100840336, 4.15546218487395, 4.180672268907563, 4.205882352941177, 4.23109243697479,
            4.256302521008403, 4.281512605042017, 4.30672268907563, 4.331932773109243, 4.357142857142858,
            4.382352941176471, 4.407563025210084, 4.432773109243698, 4.457983193277311, 4.483193277310924,

            6.008403361344538, 6.033613445378151, 6.0588235294117645, 6.084033613445378, 6.109243697478991,
            6.134453781512605, 6.159663865546219, 6.184873949579832, 6.2100840336134455, 6.235294117647059,
            6.260504201680672, 6.285714285714286, 6.310924369747899, 6.336134453781513, 6.3613445378151265,
            6.38655462184874, 6.411764705882353, 6.436974789915967, 6.46218487394958, 6.487394957983193,


            3.5126050420168067, 3.53781512605042, 3.5630252100840334, 3.588235294117647, 3.6134453781512605,
            3.638655462184874, 3.6638655462184877, 3.689075630252101, 3.7142857142857144, 3.7394957983193278,
            3.764705882352941, 3.7899159663865545, 3.815126050420168, 3.8403361344537816, 3.865546218487395,
            3.8907563025210083, 3.915966386554622, 3.9411764705882355, 3.966386554621849, 3.991596638655462,

            5.516806722689076, 5.542016806722689, 5.567226890756302, 5.592436974789916, 5.617647058823529,
            5.642857142857142, 5.668067226890756, 5.69327731092437, 5.718487394957983, 5.743697478991597,
            5.76890756302521, 5.794117647058823, 5.819327731092438, 5.844537815126051, 5.869747899159664,
            5.894957983193278, 5.920168067226891, 5.945378151260504, 5.970588235294118, 5.995798319327731,

            7.5210084033613445, 7.546218487394958, 7.571428571428571, 7.5966386554621845, 7.621848739495798,
            7.647058823529411, 7.6722689075630255, 7.697478991596639, 7.722689075630252, 7.7478991596638656,
            7.773109243697479, 7.798319327731093, 7.8235294117647065, 7.84873949579832, 7.873949579831933,
            7.899159663865547, 7.92436974789916, 7.949579831932773, 7.974789915966387, 8.0,
        ];

        let result_tensor = lhs_tensor.add(&rhs_tensor).unwrap();
        let result = result_tensor.get_values().unwrap();

        assert!(compare_outputs(&result, &expected_result))
    }

    #[test]
    fn test_tensor_add_broadcast() {
        tensor_add_broadcast(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_add_broadcast(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_pow<B: Backend>(backend: Rc<RefCell<B>>) {
        let lhs_dimensions = Dimensions4D::new(2, 3, 4, 5);
        let rhs_dimensions = Dimensions4D::new(1, 1, 1, 1);
        let lhs = linspace(1.0, 2.0, lhs_dimensions.size());
        let rhs = vec![0.5];

        let lhs_tensor =
            Tensor::from_values(backend.clone(), lhs.clone(), lhs_dimensions, true).unwrap();
        let rhs_tensor =
            Tensor::from_values(backend.clone(), rhs.clone(), rhs_dimensions, true).unwrap();

        let result_tensor = lhs_tensor.pow(&rhs_tensor).unwrap();
        let result = result_tensor.get_values().unwrap();

        let expected_result = lhs.iter().map(|x| x.sqrt()).collect::<Vec<_>>();

        assert!(compare_outputs(&result, &expected_result))
    }

    #[test]
    fn test_tensor_pow() {
        tensor_pow(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_pow(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_exp<B: Backend>(backend: Rc<RefCell<B>>) {
        let lhs_dimensions = Dimensions4D::new(2, 3, 4, 5);
        let input = linspace(1.0, 2.0, lhs_dimensions.size());

        let input_tensor =
            Tensor::from_values(backend.clone(), input.clone(), lhs_dimensions, true).unwrap();

        let result_tensor = input_tensor.exp().unwrap();
        let result = result_tensor.get_values().unwrap();

        let expected_result = input.iter().map(|x| x.exp()).collect::<Vec<_>>();

        assert!(compare_outputs(&result, &expected_result))
    }

    #[test]
    fn test_tensor_exp() {
        tensor_exp(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_exp(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_ln<B: Backend>(backend: Rc<RefCell<B>>) {
        let lhs_dimensions = Dimensions4D::new(2, 3, 4, 5);
        let input = linspace(1.0, 2.0, lhs_dimensions.size());

        let input_tensor =
            Tensor::from_values(backend.clone(), input.clone(), lhs_dimensions, true).unwrap();

        let result_tensor = input_tensor.ln().unwrap();
        let result = result_tensor.get_values().unwrap();

        let expected_result = input.iter().map(|x| x.ln()).collect::<Vec<_>>();

        assert!(compare_outputs(&result, &expected_result))
    }

    #[test]
    fn test_tensor_ln() {
        tensor_ln(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_ln(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_argmax<B: Backend>(backend: Rc<RefCell<B>>) {
        let dimensions = Dimensions4D::new(6, 5, 7, 4);
        let values = linspace(1.0, 4.0, dimensions.size());

        let tensor = Tensor::from_values(backend, values.clone(), dimensions, false).unwrap();

        let result_tensor = tensor.argmax(3).unwrap();
        let result = result_tensor.get_values().unwrap();

        let expected_values = values
            .into_iter()
            .chunks(dimensions.width)
            .into_iter()
            .map(|chunk| {
                chunk
                    .enumerate()
                    .max_by(|(_, x), (_, y)| x.total_cmp(y))
                    .map(|(i, _)| i as f32)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        assert!(compare_outputs(&result, &expected_values))
    }

    #[test]
    fn test_tensor_argmax() {
        tensor_argmax(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_argmax(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_max_dim<B: Backend>(backend: Rc<RefCell<B>>) {
        let dimensions = Dimensions4D::new(6, 5, 7, 4);
        let values = linspace(1.0, 4.0, dimensions.size());

        let tensor = Tensor::from_values(backend, values.clone(), dimensions, false).unwrap();

        let result_tensor = tensor.max_dim(3).unwrap();
        let result = result_tensor.get_values().unwrap();

        let expected_values = values
            .into_iter()
            .chunks(dimensions.width)
            .into_iter()
            .map(|chunk| chunk.max_by(|x, y| x.total_cmp(y)).unwrap())
            .collect::<Vec<_>>();

        assert!(compare_outputs(&result, &expected_values))
    }

    #[test]
    fn test_tensor_max_dim() {
        tensor_max_dim(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_max_dim(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_gather<B: Backend>(backend: Rc<RefCell<B>>) {
        let dimensions = Dimensions4D::new(32, 1, 1, 4);
        let index_dimensions = Dimensions4D::new(32, 1, 1, 1);
        let values = linspace(1.0, 4.0, dimensions.size());
        let mut rng = rand::thread_rng();
        let indices = (0..32)
            .map(|_| rng.gen_range(0..4) as f32)
            .collect::<Vec<_>>();

        let tensor =
            Tensor::from_values(backend.clone(), values.clone(), dimensions, false).unwrap();
        let index_tensor =
            Tensor::from_values(backend, indices.clone(), index_dimensions, false).unwrap();

        let result_tensor = tensor.gather(3, &index_tensor).unwrap();
        let result = result_tensor.get_values().unwrap();

        let expected_values = values
            .into_iter()
            .chunks(dimensions.width)
            .into_iter()
            .zip(indices)
            .map(|(chunk, gather_index)| chunk.collect::<Vec<_>>()[gather_index as usize])
            .collect::<Vec<_>>();

        assert!(compare_outputs(&result, &expected_values))
    }

    #[test]
    fn test_tensor_gather() {
        tensor_gather(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_gather(Rc::new(RefCell::new(CpuBackend::default())));
    }

    fn tensor_scatter<B: Backend>(backend: Rc<RefCell<B>>) {
        let dimensions = Dimensions4D::new(32, 1, 1, 1);
        let out_dimensions = Dimensions4D::new(32, 1, 1, 4);
        let index_dimensions = Dimensions4D::new(32, 1, 1, 1);
        let values = linspace(1.0, 4.0, dimensions.size());
        let mut rng = rand::thread_rng();
        let indices = (0..32)
            .map(|_| rng.gen_range(0..4) as f32)
            .collect::<Vec<_>>();

        let tensor =
            Tensor::from_values(backend.clone(), values.clone(), dimensions, false).unwrap();
        let index_tensor =
            Tensor::from_values(backend, indices.clone(), index_dimensions, false).unwrap();

        let result_tensor = tensor.scatter(3, &index_tensor, out_dimensions).unwrap();
        let result = result_tensor.get_values().unwrap();

        let mut expected_values = vec![0.0; out_dimensions.size()];

        expected_values
            .chunks_mut(out_dimensions.width)
            .zip(values.into_iter().zip(indices))
            .for_each(|(chunk, (input, gather_index))| {
                chunk[gather_index as usize] = input;
            });

        assert!(compare_outputs(&result, &expected_values))
    }

    #[test]
    fn test_tensor_scatter() {
        tensor_scatter(Rc::new(RefCell::new(WgpuBackend::default())));
        tensor_scatter(Rc::new(RefCell::new(CpuBackend::default())));
    }
}
