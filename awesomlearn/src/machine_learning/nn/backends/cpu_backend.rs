use crate::machine_learning::nn::backend::CmpType;
use crate::machine_learning::nn::dimensions::{Dimensions2D, Dimensions4D};
use crate::machine_learning::nn::{
    backend::Backend,
    tensor_error::{Result, TensorError},
};

#[derive(Clone, Debug, Default)]
pub struct CpuBackend {
    // layers: Vec<Box<dyn CpuLayer>>,
    // ping_pong: (Vec<f32>, Vec<f32>),
}

type BinaryOperatorIterator<'a> = Box<dyn Iterator<Item = f32> + 'a>;

impl CpuBackend {
    pub fn new() -> Self {
        Self {}
    }

    fn binary_operator_broadcast<'a, 'b>(
        lhs: &'a Vec<f32>,
        rhs: &'b Vec<f32>,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<(BinaryOperatorIterator<'a>, BinaryOperatorIterator<'b>)> {
        let (a, b) = {
            if lhs_dimensions != rhs_dimensions {
                let broadcast_dimensions = lhs_dimensions.broadcast_dimensions(rhs_dimensions)?;
                let lhs_broadcast = lhs_dimensions != broadcast_dimensions;
                let rhs_broadcast = rhs_dimensions != broadcast_dimensions;
                match (lhs_broadcast, rhs_broadcast) {
                    (true, true) => (
                        Box::new(BroadcastedStorage::new(
                            lhs,
                            lhs_dimensions,
                            broadcast_dimensions,
                        )) as Box<dyn Iterator<Item = f32> + 'a>,
                        Box::new(BroadcastedStorage::new(
                            rhs,
                            rhs_dimensions,
                            broadcast_dimensions,
                        )) as Box<dyn Iterator<Item = f32> + 'b>,
                    ),
                    (true, false) => (
                        Box::new(BroadcastedStorage::new(
                            lhs,
                            lhs_dimensions,
                            broadcast_dimensions,
                        )) as Box<dyn Iterator<Item = f32> + 'a>,
                        Box::new(rhs.iter().copied()) as Box<dyn Iterator<Item = f32> + 'b>,
                    ),
                    (false, true) => (
                        Box::new(lhs.iter().copied()) as Box<dyn Iterator<Item = f32> + 'a>,
                        Box::new(BroadcastedStorage::new(
                            rhs,
                            rhs_dimensions,
                            broadcast_dimensions,
                        )) as Box<dyn Iterator<Item = f32> + 'b>,
                    ),
                    (false, false) => {
                        return Err(TensorError::BroadcastDimensionsMismatch {
                            lhs: lhs_dimensions,
                            rhs: rhs_dimensions,
                        });
                    }
                }
            } else {
                (
                    Box::new(lhs.iter().copied()) as Box<dyn Iterator<Item = f32> + 'a>,
                    Box::new(rhs.iter().copied()) as Box<dyn Iterator<Item = f32> + 'b>,
                )
            }
        };
        Ok((a, b))
    }
}

struct BroadcastedStorage<'a> {
    storage: &'a Vec<f32>,
    original_dimensions: Dimensions4D,
    broadcast_dimensions: Dimensions4D,
    broadcasted_dimensions: (usize, usize, usize, usize),
    index: usize,
}

impl<'a> BroadcastedStorage<'a> {
    fn new(
        storage: &'a Vec<f32>,
        original_dimensions: Dimensions4D,
        broadcast_dimensions: Dimensions4D,
    ) -> Self {
        let broadcasted_dimensions = (
            (original_dimensions.batch == broadcast_dimensions.batch) as usize,
            (original_dimensions.depth == broadcast_dimensions.depth) as usize,
            (original_dimensions.height == broadcast_dimensions.height) as usize,
            (original_dimensions.width == broadcast_dimensions.width) as usize,
        );
        Self {
            storage,
            original_dimensions,
            broadcast_dimensions,
            broadcasted_dimensions,
            index: 0,
        }
    }
}

impl<'a> Iterator for BroadcastedStorage<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.broadcast_dimensions.size() {
            // NOTE: unwrap ok since the out-of-bounds is tested before
            let broadcast_nchw_index = self.broadcast_dimensions.nchw_index(self.index).unwrap();
            let original_nchw_index = (
                broadcast_nchw_index.0 * self.broadcasted_dimensions.0,
                broadcast_nchw_index.1 * self.broadcasted_dimensions.1,
                broadcast_nchw_index.2 * self.broadcasted_dimensions.2,
                broadcast_nchw_index.3 * self.broadcasted_dimensions.3,
            );
            let original_flat_index = self
                .original_dimensions
                .flat_index(original_nchw_index)
                .unwrap();
            self.index += 1;
            Some(self.storage[original_flat_index])
        } else {
            None
        }
    }
}

impl Backend for CpuBackend {
    type Storage = Vec<f32>;

    fn zeros(&mut self, dimensions: Dimensions4D) -> Result<Self::Storage> {
        let capacity = dimensions.size();
        let buffer = vec![0.0; capacity];
        Ok(buffer)
    }

    fn ones(&mut self, dimensions: Dimensions4D) -> Result<Self::Storage> {
        let capacity = dimensions.size();
        let buffer = vec![1.0; capacity];
        Ok(buffer)
    }

    fn buffer_from_values(
        &mut self,
        dimensions: Dimensions4D,
        values: Vec<f32>,
    ) -> Result<Self::Storage> {
        let capacity = dimensions.size();
        if capacity != values.len() {
            return Err(TensorError::TensorInitMismatch {
                tensor_op_name: "tensor_from_values".into(),
                expected: dimensions,
                expected_len: capacity,
                received: values.len(),
            });
        }
        let buffer = values;
        Ok(buffer)
    }

    fn copy(&mut self, buffer: &Self::Storage) -> Result<Self::Storage> {
        let buffer = buffer.clone();
        Ok(buffer)
    }

    fn matmul(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        mnk: (usize, usize, usize),
        is_rhs_transposed: bool,
    ) -> Result<Self::Storage> {
        let (m, n, k) = mnk;
        // The batch might also be just for example channels when doing convolution
        let c_dimensions = Dimensions4D::new(1, 1, m, n);
        let mut c = self.zeros(c_dimensions)?;

        // Derive the a and b dimensions from bmnk, these could be taken also as parameters but
        // might be redundant on other backends where bmnk are used directly
        if !is_rhs_transposed {
            let lhs_dimensions = Dimensions4D::new(1, 1, m, k);
            let rhs_dimensions = Dimensions4D::new(1, 1, k, n);

            for j in 0..m {
                for i in 0..n {
                    // We aren't indexing c efficiently now, but should the a and b
                    let c_index = c_dimensions.flat_index((0, 0, j, i))?;
                    for kk in 0..k {
                        let a_index = lhs_dimensions.flat_index((0, 0, j, kk))?;
                        let b_index = rhs_dimensions.flat_index((0, 0, kk, i))?;
                        c[c_index] += lhs[a_index] * rhs[b_index];
                    }
                }
            }
        } else {
            let lhs_dimensions = Dimensions4D::new(1, 1, m, k);
            let rhs_dimensions = Dimensions4D::new(1, 1, n, k);

            for j in 0..m {
                for i in 0..n {
                    // We aren't indexing c efficiently now, but should the a and b
                    let c_index = c_dimensions.flat_index((0, 0, j, i))?;
                    for kk in 0..k {
                        let a_index = lhs_dimensions.flat_index((0, 0, j, kk))?;
                        let b_index = rhs_dimensions.flat_index((0, 0, i, kk))?;
                        c[c_index] += lhs[a_index] * rhs[b_index];
                    }
                }
            }
        }

        Ok(c)
    }

    fn pow(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        let (a, b) = Self::binary_operator_broadcast(lhs, rhs, lhs_dimensions, rhs_dimensions)?;
        let result = a.zip(b).map(|(l, r)| l.powf(r)).collect::<Self::Storage>();
        Ok(result)
    }

    fn mul(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        let (a, b) = Self::binary_operator_broadcast(lhs, rhs, lhs_dimensions, rhs_dimensions)?;
        let result = a.zip(b).map(|(l, r)| l * r).collect::<Self::Storage>();
        Ok(result)
    }

    fn add(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        let (a, b) = Self::binary_operator_broadcast(lhs, rhs, lhs_dimensions, rhs_dimensions)?;
        let result = a.zip(b).map(|(l, r)| l + r).collect::<Self::Storage>();
        Ok(result)
    }

    fn sub(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        let (a, b) = Self::binary_operator_broadcast(lhs, rhs, lhs_dimensions, rhs_dimensions)?;
        let result = a.zip(b).map(|(l, r)| l - r).collect::<Self::Storage>();
        Ok(result)
    }

    fn div(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        let (a, b) = Self::binary_operator_broadcast(lhs, rhs, lhs_dimensions, rhs_dimensions)?;
        let result = a.zip(b).map(|(l, r)| l / r).collect::<Self::Storage>();
        Ok(result)
    }

    fn transpose(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        dim1: usize,
        dim2: usize,
    ) -> Result<Self::Storage> {
        let mut transposed_buffer = self.zeros(dimensions)?;
        let transposed_dimensions = dimensions.swap(dim1, dim2)?;
        let strides = transposed_dimensions.strides();
        let strides_as_d: Dimensions4D = strides.into();
        let new_strides_as_d = strides_as_d.swap(dim1, dim2)?;
        let new_strides: [usize; 4] = new_strides_as_d.into();

        for n in 0..dimensions.batch {
            for c in 0..dimensions.depth {
                for h in 0..dimensions.height {
                    for w in 0..dimensions.width {
                        let old_index = dimensions.flat_index((n, c, h, w))?;
                        let new_index = n * new_strides[0]
                            + c * new_strides[1]
                            + h * new_strides[2]
                            + w * new_strides[3];
                        transposed_buffer[new_index] = buffer[old_index];
                    }
                }
            }
        }
        Ok(transposed_buffer)
    }

    fn zero_pad(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        offset: Dimensions4D,
        new_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        let mut padded_buffer = self.zeros(new_dimensions)?;

        for n in 0..dimensions.batch {
            for c in 0..dimensions.depth {
                for h in 0..dimensions.height {
                    for w in 0..dimensions.width {
                        let old_index = dimensions.flat_index((n, c, h, w))?;
                        // Add the offest to the new index before getting the index
                        let new_index = new_dimensions.flat_index((
                            n + offset.batch,
                            c + offset.depth,
                            h + offset.height,
                            w + offset.width,
                        ))?;
                        padded_buffer[new_index] = buffer[old_index];
                    }
                }
            }
        }
        Ok(padded_buffer)
    }

    fn slice(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        offset: Dimensions4D,
        new_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        let mut sliced_buffer = self.zeros(new_dimensions)?;
        if offset.batch > dimensions.batch
            || offset.depth > dimensions.depth
            || offset.height > dimensions.height
            || offset.width > dimensions.width
        {
            return Err(TensorError::TensorDimensionMismatch {
                tensor_op_name: "slice".into(),
                expected: dimensions,
                received: offset,
            });
        }

        for n in 0..new_dimensions.batch {
            for c in 0..new_dimensions.depth {
                for h in 0..new_dimensions.height {
                    for w in 0..new_dimensions.width {
                        let old_index = dimensions.flat_index((
                            n + offset.batch,
                            c + offset.depth,
                            h + offset.height,
                            w + offset.width,
                        ))?;
                        let new_index = new_dimensions.flat_index((n, c, h, w))?;
                        sliced_buffer[new_index] = buffer[old_index];
                    }
                }
            }
        }
        Ok(sliced_buffer)
    }

    fn max_pool2d(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        pool_size: Dimensions2D,
    ) -> Result<Self::Storage> {
        let mut pool = vec![0.0; pool_size.size()];
        let new_height = dimensions.height / pool_size.height;
        let new_width = dimensions.width / pool_size.width;
        let new_dimensions =
            Dimensions4D::new(dimensions.batch, dimensions.depth, new_height, new_width);
        let mut result = self.zeros(new_dimensions)?;
        for n in 0..dimensions.batch {
            for c in 0..dimensions.depth {
                for h in 0..new_height {
                    for w in 0..new_width {
                        for j in 0..pool_size.height {
                            for i in 0..pool_size.width {
                                let index = dimensions.flat_index((
                                    n,
                                    c,
                                    h * pool_size.height + j,
                                    w * pool_size.width + i,
                                ))?;
                                pool[j * pool_size.width + i] = buffer[index];
                            }
                        }
                        let max = pool.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let pooled_index = new_dimensions.flat_index((n, c, h, w))?;
                        result[pooled_index] = max;
                    }
                }
            }
        }
        Ok(result)
    }

    fn avg_pool2d(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        pool_size: Dimensions2D,
    ) -> Result<Self::Storage> {
        let mut pool = vec![0.0; pool_size.size()];
        let new_height = dimensions.height / pool_size.height;
        let new_width = dimensions.width / pool_size.width;
        let new_dimensions =
            Dimensions4D::new(dimensions.batch, dimensions.depth, new_height, new_width);
        let mut result = self.zeros(new_dimensions)?;
        for n in 0..dimensions.batch {
            for c in 0..dimensions.depth {
                for h in 0..new_height {
                    for w in 0..new_width {
                        for j in 0..pool_size.height {
                            for i in 0..pool_size.width {
                                let index = dimensions.flat_index((
                                    n,
                                    c,
                                    h * pool_size.height + j,
                                    w * pool_size.width + i,
                                ))?;
                                pool[j * pool_size.width + i] = buffer[index];
                            }
                        }
                        let avg = pool.iter().sum::<f32>() / (pool.len() as f32);
                        let pooled_index = new_dimensions.flat_index((n, c, h, w))?;
                        result[pooled_index] = avg;
                    }
                }
            }
        }
        Ok(result)
    }

    fn conv2d(
        &mut self,
        buffer: &Self::Storage,
        kernels: &Self::Storage,
        dimensions: Dimensions4D,
        kernel_size: Dimensions4D,
    ) -> Result<Self::Storage> {
        let start_offset_y = kernel_size.height / 2;
        let start_offset_x = kernel_size.width / 2;
        let stop_offset_y = (kernel_size.height - 1) / 2;
        let stop_offset_x = (kernel_size.width - 1) / 2;

        let output_height = dimensions.height - kernel_size.height + 1;
        let output_width = dimensions.width - kernel_size.width + 1;
        let new_dimensions = Dimensions4D::new(
            dimensions.batch,
            kernel_size.batch,
            output_height,
            output_width,
        );
        let mut result = self.zeros(new_dimensions)?;
        for n in 0..dimensions.batch {
            for c in 0..dimensions.depth {
                for h in start_offset_y..(dimensions.height - stop_offset_y) {
                    for w in start_offset_x..(dimensions.width - stop_offset_x) {
                        // Kernel indexing
                        for k in 0..kernel_size.batch {
                            let output_index = new_dimensions.flat_index((
                                n,
                                k,
                                h - start_offset_y,
                                w - start_offset_x,
                            ))?;
                            for j in 0..kernel_size.height {
                                for i in 0..kernel_size.width {
                                    let input_offset_w = i as i32 - start_offset_x as i32;
                                    let input_offset_h = j as i32 - start_offset_y as i32;
                                    let kernel_w = w as i32 + input_offset_w;
                                    let kernel_h = h as i32 + input_offset_h;
                                    let input_index = dimensions.flat_index((
                                        n,
                                        c,
                                        kernel_h as usize,
                                        kernel_w as usize,
                                    ))?;
                                    let value = buffer[input_index];
                                    let kernel_index = kernel_size.flat_index((k, c, j, i))?;
                                    result[output_index] += kernels[kernel_index] * value;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(result)
    }
    fn conv_transpose2d(
        &mut self,
        buffer: &Self::Storage,
        kernels: &Self::Storage,
        dimensions: Dimensions4D,
        kernel_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        let output_height = dimensions.height + kernel_dimensions.height - 1;
        let output_width = dimensions.width + kernel_dimensions.width - 1;
        let new_dimensions = Dimensions4D::new(
            dimensions.batch,
            kernel_dimensions.depth,
            output_height,
            output_width,
        );
        // input – input tensor of shape (minibatch,in_channels,iH,iW)
        // weight – filters of shape (in_channels,out_channel/groups,kH,kW)
        let mut result = self.zeros(new_dimensions)?;
        for n in 0..dimensions.batch {
            for c in 0..dimensions.depth {
                for h in 0..dimensions.height {
                    for w in 0..dimensions.width {
                        let input_index = dimensions.flat_index((n, c, h, w))?;
                        // kernels batch or OUTPUT_DEPTH
                        for k in 0..kernel_dimensions.depth {
                            for j in 0..kernel_dimensions.height {
                                for i in 0..kernel_dimensions.width {
                                    let output_index =
                                        new_dimensions.flat_index((n, k, h + j, w + i))?;
                                    let value = buffer[input_index];
                                    let kernel_index =
                                        kernel_dimensions.flat_index((c, k, j, i))?;
                                    result[output_index] += kernels[kernel_index] * value;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(result)
    }

    fn relu(&mut self, buffer: &Self::Storage, _dimensions: Dimensions4D) -> Result<Self::Storage> {
        Ok(buffer.iter().cloned().map(|x| f32::max(0.0, x)).collect())
    }

    fn leaky_relu(
        &mut self,
        buffer: &Self::Storage,
        _dimensions: Dimensions4D,
        alpha: f32,
    ) -> Result<Self::Storage> {
        Ok(buffer
            .iter()
            .cloned()
            .map(|x| f32::max(alpha * x, x))
            .collect())
    }

    fn sigmoid(
        &mut self,
        buffer: &Self::Storage,
        _dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        Ok(buffer
            .iter()
            .cloned()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect())
    }

    fn exp(&mut self, buffer: &Self::Storage, _dimensions: Dimensions4D) -> Result<Self::Storage> {
        Ok(buffer.iter().cloned().map(|x| x.exp()).collect())
    }

    fn ln(&mut self, buffer: &Self::Storage, _dimensions: Dimensions4D) -> Result<Self::Storage> {
        Ok(buffer.iter().cloned().map(|x| x.ln()).collect())
    }

    fn get_values(
        &mut self,
        buffer: &Self::Storage,
        _dimensions: Dimensions4D,
    ) -> Result<Vec<f32>> {
        Ok(buffer.clone())
    }

    fn sum_reduce(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        dims: &[usize],
    ) -> Result<Self::Storage> {
        let reduced_dimensions = dimensions.reduce_dims(dims)?;
        let mut result = self.zeros(reduced_dimensions)?;

        for n in 0..reduced_dimensions.batch {
            for c in 0..reduced_dimensions.depth {
                for h in 0..reduced_dimensions.height {
                    for w in 0..reduced_dimensions.width {
                        let mut sum = 0.0;
                        // Sum over the dimensions specified in `dims`
                        for nn in if dims.contains(&0) {
                            0..dimensions.batch
                        } else {
                            n..n + 1
                        } {
                            for nc in if dims.contains(&1) {
                                0..dimensions.depth
                            } else {
                                c..c + 1
                            } {
                                for nh in if dims.contains(&2) {
                                    0..dimensions.height
                                } else {
                                    h..h + 1
                                } {
                                    for nw in if dims.contains(&3) {
                                        0..dimensions.width
                                    } else {
                                        w..w + 1
                                    } {
                                        let input_index =
                                            dimensions.flat_index((nn, nc, nh, nw))?;
                                        sum += buffer[input_index];
                                    }
                                }
                            }
                        }
                        let output_index = reduced_dimensions.flat_index((n, c, h, w))?;
                        result[output_index] = sum;
                    }
                }
            }
        }
        Ok(result)
    }

    fn comparison(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
        comparison_type: CmpType,
    ) -> Result<Self::Storage> {
        let (a, b) = Self::binary_operator_broadcast(lhs, rhs, lhs_dimensions, rhs_dimensions)?;
        let cmp_fn = match comparison_type {
            CmpType::Equal => |l: &f32, r: &f32| l == r,
            CmpType::NotEqual => |l: &f32, r: &f32| l != r,
            CmpType::GreaterThan => |l: &f32, r: &f32| l > r,
            CmpType::GreaterOrEqualThan => |l: &f32, r: &f32| l >= r,
            CmpType::LesserThan => |l: &f32, r: &f32| l < r,
            CmpType::LesserOrEqualThan => |l: &f32, r: &f32| l <= r,
        };
        let result = a
            .zip(b)
            .map(|(l, r)| cmp_fn(&l, &r))
            .map(|v| match v {
                true => 1.0,
                false => 0.0,
            })
            .collect::<Self::Storage>();
        Ok(result)
    }

    fn upsample2d(
        &mut self,
        buffer: &Self::Storage,
        input_dimensions: Dimensions4D,
        upsample_size: Dimensions2D,
    ) -> Result<Self::Storage> {
        let new_height = input_dimensions.height * upsample_size.height;
        let new_width = input_dimensions.width * upsample_size.width;
        let new_dimensions = Dimensions4D::new(
            input_dimensions.batch,
            input_dimensions.depth,
            new_height,
            new_width,
        );
        let mut result = self.zeros(new_dimensions)?;
        for n in 0..input_dimensions.batch {
            for c in 0..input_dimensions.depth {
                for h in 0..new_height {
                    for w in 0..new_width {
                        let input_h = h / upsample_size.height;
                        let input_w = w / upsample_size.width;
                        let input_index = input_dimensions.flat_index((n, c, input_h, input_w))?;
                        let output_index = new_dimensions.flat_index((n, c, h, w))?;
                        result[output_index] = buffer[input_index];
                    }
                }
            }
        }
        Ok(result)
    }

    fn max_dim(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        dim: usize,
    ) -> Result<Self::Storage> {
        let reduced_dimensions = dimensions.reduce_dims(&[dim])?;
        let dim_size = dimensions.from_index(dim)?;
        let mut result = self.zeros(reduced_dimensions)?;

        for n in 0..reduced_dimensions.batch {
            for c in 0..reduced_dimensions.depth {
                for h in 0..reduced_dimensions.height {
                    for w in 0..reduced_dimensions.width {
                        let output_index = reduced_dimensions.flat_index((n, c, h, w))?;

                        let mut max_value = f32::NEG_INFINITY; // Assuming f32 for storage

                        for d in 0..dim_size {
                            let mut input_index = (n, c, h, w);
                            match dim {
                                0 => input_index.0 = d,
                                1 => input_index.1 = d,
                                2 => input_index.2 = d,
                                3 => input_index.3 = d,
                                _ => return Err(TensorError::DimensionIndexError { index: dim }),
                            }
                            let flat_index = dimensions.flat_index(input_index)?;

                            if buffer[flat_index] > max_value {
                                max_value = buffer[flat_index];
                            }
                        }

                        result[output_index] = max_value;
                    }
                }
            }
        }
        Ok(result)
    }

    fn arg_max(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        dim: usize,
    ) -> Result<Self::Storage> {
        let reduced_dimensions = dimensions.reduce_dims(&[dim])?;
        let dim_size = dimensions.from_index(dim)?;
        let mut result = self.zeros(reduced_dimensions)?;

        for n in 0..reduced_dimensions.batch {
            for c in 0..reduced_dimensions.depth {
                for h in 0..reduced_dimensions.height {
                    for w in 0..reduced_dimensions.width {
                        let output_index = reduced_dimensions.flat_index((n, c, h, w))?;

                        let mut max_value = f32::NEG_INFINITY; // Assuming f32 for storage
                        let mut max_dim = 0;

                        for d in 0..dim_size {
                            let mut input_index = (n, c, h, w);
                            match dim {
                                0 => input_index.0 = d,
                                1 => input_index.1 = d,
                                2 => input_index.2 = d,
                                3 => input_index.3 = d,
                                _ => return Err(TensorError::DimensionIndexError { index: dim }),
                            }
                            let flat_index = dimensions.flat_index(input_index)?;

                            if buffer[flat_index] > max_value {
                                max_value = buffer[flat_index];
                                max_dim = d;
                            }
                        }

                        result[output_index] = max_dim as f32;
                    }
                }
            }
        }
        Ok(result)
    }

    fn gather(
        &mut self,
        buffer: &Self::Storage,
        indices: &Self::Storage,
        dimensions: Dimensions4D,
        dim: usize,
    ) -> Result<Self::Storage> {
        let reduced_dimensions = dimensions.reduce_dims(&[dim])?;
        let mut result = self.zeros(reduced_dimensions)?;
        let mut index_iter = indices.iter().map(|&f| f as usize);

        for n in 0..reduced_dimensions.batch {
            for c in 0..reduced_dimensions.depth {
                for h in 0..reduced_dimensions.height {
                    for w in 0..reduced_dimensions.width {
                        let output_index = reduced_dimensions.flat_index((n, c, h, w))?;

                        let d = index_iter.next().ok_or(TensorError::InsufficientIndices)?;
                        let mut input_index = (n, c, h, w);
                        match dim {
                            0 => input_index.0 = d,
                            1 => input_index.1 = d,
                            2 => input_index.2 = d,
                            3 => input_index.3 = d,
                            _ => unreachable!(),
                        }
                        let input_index = dimensions.flat_index(input_index)?;

                        result[output_index] = buffer[input_index];
                    }
                }
            }
        }
        Ok(result)
    }

    fn scatter(
        &mut self,
        buffer: &Self::Storage,
        indices: &Self::Storage,
        dimensions: Dimensions4D,
        out_dimensions: Dimensions4D,
        dim: usize,
    ) -> Result<Self::Storage> {
        let mut result = self.zeros(out_dimensions)?;
        let mut index_iter = indices.iter().map(|&f| f as usize);

        for n in 0..dimensions.batch {
            for c in 0..dimensions.depth {
                for h in 0..dimensions.height {
                    for w in 0..dimensions.width {
                        let input_index = dimensions.flat_index((n, c, h, w))?;

                        let d = index_iter.next().ok_or(TensorError::InsufficientIndices)?;
                        let mut output_index = (n, c, h, w);
                        match dim {
                            0 => output_index.0 = d,
                            1 => output_index.1 = d,
                            2 => output_index.2 = d,
                            3 => output_index.3 = d,
                            _ => unreachable!(),
                        }
                        let output_index = out_dimensions.flat_index(output_index)?;

                        result[output_index] = buffer[input_index];
                    }
                }
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {

    use crate::machine_learning::{compare_outputs, linspace};

    use super::*;

    #[test]
    fn transposing_width_and_height() {
        let mut backend = CpuBackend {};
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dimensions = Dimensions4D::new(1, 1, 2, 3);

        let buffer = backend.buffer_from_values(dimensions, values).unwrap();
        let transpose = backend.transpose(&buffer, dimensions, 2, 3).unwrap();

        let expected_values = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];

        compare_outputs(&transpose, &expected_values);
    }

    #[test]
    fn transposing_channel_and_width() {
        let mut backend = CpuBackend {};

        #[rustfmt::skip]
        let values = [
            // channel 1
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            // channel 2
            13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0,
        ];

        let dimensions = Dimensions4D::new(1, 2, 3, 4);

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let transpose = backend.transpose(&buffer, dimensions, 1, 3).unwrap();

        #[rustfmt::skip]
        let expected_values = [
            // row 1
            1.0, 13.0,
            5.0, 17.0,
            9.0, 21.0,
            // row 2
            2.0, 14.0,
            6.0, 18.0,
            10.0, 22.0,
            // row 3
            3.0, 15.0,
            7.0, 19.0,
            11.0, 23.0,
            // row 4
            4.0, 16.0,
            8.0, 20.0,
            12.0, 24.0,
        ];

        compare_outputs(&transpose, &expected_values);
    }

    #[test]
    fn transposing_batch_and_channel() {
        let mut backend = CpuBackend::default();

        let dimensions = Dimensions4D::new(2, 3, 4, 5);
        let values = linspace(0.0, dimensions.size() as f32, dimensions.size());

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let transpose = backend.transpose(&buffer, dimensions, 0, 1).unwrap();
        let transpose = backend.get_values(&transpose, dimensions).unwrap();

        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected_values = [
            0.0, 1.0084033613445378, 2.0168067226890756, 3.0252100840336134, 4.033613445378151,
            5.042016806722689, 6.050420168067227, 7.0588235294117645, 8.067226890756302, 9.07563025210084,
            10.084033613445378, 11.092436974789916, 12.100840336134453, 13.109243697478991, 14.117647058823529,
            15.126050420168067, 16.134453781512605, 17.142857142857142, 18.15126050420168, 19.159663865546218,

            60.50420168067227, 61.5126050420168, 62.52100840336134, 63.529411764705884, 64.53781512605042,
            65.54621848739495, 66.55462184873949, 67.56302521008404, 68.57142857142857, 69.5798319327731,
            70.58823529411765, 71.59663865546219, 72.60504201680672, 73.61344537815125, 74.62184873949579,
            75.63025210084034, 76.63865546218487, 77.6470588235294, 78.65546218487395, 79.66386554621849,


            20.168067226890756, 21.176470588235293, 22.18487394957983, 23.19327731092437, 24.201680672268907,
            25.210084033613445, 26.218487394957982, 27.22689075630252, 28.235294117647058, 29.243697478991596,
            30.252100840336134, 31.26050420168067, 32.26890756302521, 33.27731092436974, 34.285714285714285,
            35.294117647058826, 36.30252100840336, 37.310924369747895, 38.319327731092436, 39.32773109243698,

            80.67226890756302, 81.68067226890756, 82.68907563025209, 83.69747899159664, 84.70588235294117,
            85.71428571428571, 86.72268907563026, 87.73109243697479, 88.73949579831933, 89.74789915966386,
            90.7563025210084, 91.76470588235294, 92.77310924369748, 93.78151260504201, 94.78991596638656,
            95.7983193277311, 96.80672268907563, 97.81512605042016, 98.8235294117647, 99.83193277310924,


            40.33613445378151, 41.344537815126046, 42.35294117647059, 43.36134453781513, 44.36974789915966,
            45.3781512605042, 46.38655462184874, 47.39495798319328, 48.403361344537814, 49.41176470588235,
            50.42016806722689, 51.42857142857143, 52.436974789915965, 53.4453781512605, 54.45378151260504,
            55.46218487394958, 56.470588235294116, 57.47899159663865, 58.48739495798319, 59.49579831932773,

            100.84033613445378, 101.84873949579831, 102.85714285714286, 103.8655462184874, 104.87394957983193,
            105.88235294117646, 106.890756302521, 107.89915966386555, 108.90756302521008, 109.91596638655462,
            110.92436974789916, 111.9327731092437, 112.94117647058823, 113.94957983193277, 114.9579831932773,
            115.96638655462185, 116.97478991596638, 117.98319327731092, 118.99159663865547, 120.0,
        ];

        assert!(compare_outputs(&transpose, &expected_values));
    }

    #[test]
    fn test_zero_pad() {
        let mut backend = CpuBackend {};

        #[rustfmt::skip]
        let values = [
            // batch 1
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            // batch 2
            13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0,
        ];

        let dimensions = Dimensions4D::new(1, 2, 3, 4);

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();

        let padded_dimensions = Dimensions4D::new(2, 6, 7, 11);
        let offset = Dimensions4D::new(0, 1, 2, 3);
        let zero_padded = backend
            .zero_pad(&buffer, dimensions, offset, padded_dimensions)
            .unwrap();

        #[rustfmt::skip]
        let expected_values = [
            // batch 1
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // batch 2
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 9.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // batch 3
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 13.0, 14.0, 15.0, 16.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 17.0, 18.0, 19.0, 20.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 21.0, 22.0, 23.0, 24.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

            // batch 4
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

            // batch 5
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

            // batch 6
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        compare_outputs(&zero_padded, &expected_values);
    }

    #[test]
    fn test_slicing() {
        let mut backend = CpuBackend {};

        #[rustfmt::skip]
        let values = [
            // batch 1
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            // batch 2
            13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0,
        ];

        let dimensions = Dimensions4D::new(2, 1, 3, 4);

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();

        let sliced_dimensions = Dimensions4D::new(1, 1, 2, 2);
        let offset = Dimensions4D::new(1, 0, 1, 1);
        let sliced = backend
            .slice(&buffer, dimensions, offset, sliced_dimensions)
            .unwrap();

        #[rustfmt::skip]
        let expected_values = [
            18.0, 19.0,
            22.0, 23.0,
        ];

        compare_outputs(&sliced, &expected_values);
    }

    #[test]
    fn test_slicing2() {
        let mut backend = CpuBackend {};

        let dimensions = Dimensions4D::new(3, 6, 5, 4);
        let values = linspace(0.0, 1.0, dimensions.size());

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();

        let sliced_dimensions = Dimensions4D::new(1, 4, 3, 2);
        let offset = Dimensions4D::new(1, 2, 2, 1);
        let sliced = backend
            .slice(&buffer, dimensions, offset, sliced_dimensions)
            .unwrap();
        let sliced = backend.get_values(&sliced, sliced_dimensions).unwrap();

        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected_values = [
            0.47075208913649025, 0.4735376044568245,
            0.4818941504178273, 0.48467966573816157,
            0.49303621169916434, 0.4958217270194986,

            0.5264623955431755, 0.5292479108635098,
            0.5376044568245125, 0.5403899721448469,
            0.5487465181058496, 0.5515320334261838,

            0.5821727019498607, 0.584958217270195,
            0.5933147632311978, 0.596100278551532,
            0.6044568245125348, 0.6072423398328691,

            0.637883008356546, 0.6406685236768802,
            0.649025069637883, 0.6518105849582173,
            0.6601671309192201, 0.6629526462395543,
        ];

        compare_outputs(&sliced, &expected_values);
    }

    fn setup_matrices(bmnk: (usize, usize, usize, usize)) -> (Vec<f32>, Vec<f32>) {
        let (b, m, n, k) = bmnk;
        // the size should be the same as the dimensions size
        let lhs = linspace(0.0, 1.0, k * m);
        let rhs = linspace(0.0, 1.0, b * k * n);
        // let mut rhs = linspace(0.0, 1.0, b * n * k / 2);
        // rhs.append(&mut vec![0.0_f32; b * n * k / 2]);
        // let b_values = linspace(0.0, 1.0, n * k);
        (lhs, rhs)
    }

    #[test]
    fn cpu_backend_matmul() {
        // TODO: gpu doesn't currently support batching
        let mut cpu_backend = CpuBackend {};

        let (m, n, k) = (3, 4, 2);
        let a_dimensions = Dimensions4D::new(1, 1, m, k);
        let b_dimensions = Dimensions4D::new(1, 1, k, n);

        let (a_values, b_values) = setup_matrices((1, m, n, k));
        let lhs = cpu_backend
            .buffer_from_values(a_dimensions, a_values)
            .unwrap();
        let rhs = cpu_backend
            .buffer_from_values(b_dimensions, b_values)
            .unwrap();

        // Shape should be (1, 1, 3, 4)
        #[rustfmt::skip]
        let expected_values = [
            0.11428571, 0.14285714, 0.17142857, 0.2,
            0.34285714, 0.485_714_3, 0.62857143, 0.771_428_6,
            0.571_428_6, 0.82857143, 1.085_714_3, 1.342_857_1,
        ];

        let cpu_result = cpu_backend.matmul(&lhs, &rhs, (m, n, k), false).unwrap();

        assert!(compare_outputs(&cpu_result, &expected_values));
    }

    #[test]
    fn test_max_pool2d() {
        let mut backend = CpuBackend {};

        let dimensions = Dimensions4D::new(2, 6, 4, 2);
        let values = linspace(0.0, 1.0, dimensions.size());

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let pool_size = Dimensions2D::new(2, 2);
        let max_pooled = backend.max_pool2d(&buffer, dimensions, pool_size).unwrap();

        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected_values = [
            0.03157895,
            0.07368421,
            0.11578947,
            0.15789474,
            0.2,
            0.24210526,
            0.28421053,
            0.32631579,
            0.36842105,
            0.41052632,
            0.45263158,
            0.49473684,
            0.53684211,
            0.57894737,
            0.62105263,
            0.66315789,
            0.70526316,
            0.74736842,
            0.78947368,
            0.83157895,
            0.87368421,
            0.91578947,
            0.95789474,
            1.,
        ];
        // shape (2, 6, 2, 1)

        assert!(compare_outputs(&max_pooled, &expected_values));
    }

    #[test]
    fn test_conv2d() {
        let mut backend = CpuBackend {};

        let dimensions = Dimensions4D::new(2, 6, 5, 4);
        let values = linspace(0.0, 1.0, dimensions.size());
        let kernel_dimensions = Dimensions4D::new(2, 6, 3, 3);
        let kernel_values = linspace(1.0, 3.0, kernel_dimensions.size());

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();

        let kernels = backend
            .buffer_from_values(kernel_dimensions, kernel_values.to_vec())
            .unwrap();

        let result = backend
            .conv2d(&buffer, &kernels, dimensions, kernel_dimensions)
            .unwrap();

        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected_values = [
            20.835881593868535, 21.173737926719582,
            22.187306925272743, 22.5251632581238,
            23.538732256676962, 23.876588589528012,

            33.37879795096391, 33.94470730848942,
            35.642435381065965, 36.20834473859148,
            37.90607281116803, 38.47198216869353,


            61.37864153599499, 61.71649786884604,
            62.730066867399195, 63.06792320025026,
            64.08149219880343, 64.41934853165446,

            101.2879208540257, 101.85383021155124,
            103.5515582841278, 104.11746764165329,
            105.81519571422984, 106.38110507175534,
        ];
        // shape (2, 2, 5, 4)

        assert!(compare_outputs(&result, &expected_values));
    }

    #[test]
    fn test_addition_with_identical_dimensions() {
        let mut backend = CpuBackend {};

        let lhs_dimensions = Dimensions4D::new(2, 6, 7, 8);
        let rhs_dimensions = lhs_dimensions;
        let lhs = linspace(0.0, 1.0, lhs_dimensions.size());
        let rhs = linspace(1.0, 3.0, rhs_dimensions.size());

        let lhs_buffer = backend
            .buffer_from_values(lhs_dimensions, lhs.clone())
            .unwrap();
        let rhs_buffer = backend
            .buffer_from_values(rhs_dimensions, rhs.clone())
            .unwrap();

        let result = backend
            .add(&lhs_buffer, &rhs_buffer, lhs_dimensions, rhs_dimensions)
            .unwrap();
        let result = backend.get_values(&result, lhs_dimensions).unwrap();

        let expected = lhs.iter().zip(rhs).map(|(l, r)| l + r).collect::<Vec<_>>();

        assert!(compare_outputs(&result, &expected));
    }

    #[test]
    fn test_addition_with_broadcasting_dimensions() {
        let mut backend = CpuBackend {};

        let lhs_dimensions = Dimensions4D::new(1, 4, 3, 1);
        let rhs_dimensions = Dimensions4D::new(2, 4, 3, 6);
        let lhs = linspace(0.0, 1.0, lhs_dimensions.size());
        let rhs = linspace(1.0, 3.0, rhs_dimensions.size());

        let lhs_buffer = backend
            .buffer_from_values(lhs_dimensions, lhs.clone())
            .unwrap();
        let rhs_buffer = backend
            .buffer_from_values(rhs_dimensions, rhs.clone())
            .unwrap();

        let result = backend
            .add(&lhs_buffer, &rhs_buffer, lhs_dimensions, rhs_dimensions)
            .unwrap();
        let result = backend
            .get_values(&result, Dimensions4D::new(2, 4, 3, 6))
            .unwrap();

        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected = [
            1.0, 1.013986013986014, 1.027972027972028, 1.0419580419580419, 1.055944055944056, 1.06993006993007,
            1.1748251748251748, 1.1888111888111887, 1.2027972027972027, 1.2167832167832167, 1.2307692307692306, 1.2447552447552446,
            1.3496503496503498, 1.3636363636363638, 1.3776223776223777, 1.3916083916083917, 1.4055944055944056, 1.4195804195804196,

            1.5244755244755244, 1.5384615384615385, 1.5524475524475525, 1.5664335664335665, 1.5804195804195804, 1.5944055944055944,
            1.6993006993006992, 1.7132867132867133, 1.7272727272727275, 1.7412587412587412, 1.7552447552447554, 1.7692307692307692,
            1.8741258741258742, 1.8881118881118881, 1.902097902097902, 1.9160839160839163, 1.93006993006993, 1.9440559440559442,

            2.048951048951049, 2.062937062937063, 2.0769230769230766, 2.090909090909091, 2.104895104895105, 2.1188811188811187,
            2.2237762237762237, 2.2377622377622375, 2.2517482517482517, 2.265734265734266, 2.2797202797202796, 2.2937062937062938,
            2.3986013986013983, 2.4125874125874125, 2.4265734265734267, 2.440559440559441, 2.4545454545454546, 2.4685314685314683,

            2.5734265734265733, 2.5874125874125875, 2.6013986013986017, 2.6153846153846154, 2.6293706293706296, 2.6433566433566433,
            2.7482517482517483, 2.7622377622377625, 2.7762237762237763, 2.7902097902097904, 2.804195804195804, 2.8181818181818183,
            2.9230769230769234, 2.937062937062937, 2.951048951048951, 2.965034965034965, 2.979020979020979, 2.993006993006993,


            2.006993006993007, 2.020979020979021, 2.034965034965035, 2.048951048951049, 2.062937062937063, 2.0769230769230766,
            2.1818181818181817, 2.195804195804196, 2.20979020979021, 2.2237762237762237, 2.2377622377622375, 2.2517482517482517,
            2.3566433566433567, 2.3706293706293704, 2.384615384615384, 2.3986013986013983, 2.4125874125874125, 2.4265734265734267,

            2.5314685314685317, 2.545454545454545, 2.559440559440559, 2.5734265734265733, 2.5874125874125875, 2.6013986013986017,
            2.7062937062937062, 2.7202797202797204, 2.7342657342657346, 2.7482517482517483, 2.762237762237762, 2.7762237762237763,
            2.8811188811188813, 2.8951048951048954, 2.909090909090909, 2.923076923076923, 2.937062937062937, 2.9510489510489513,

            3.055944055944056, 3.0699300699300696, 3.0839160839160837, 3.097902097902098, 3.111888111888112, 3.125874125874126,
            3.2307692307692304, 3.2447552447552446, 3.2587412587412588, 3.272727272727273, 3.2867132867132867, 3.3006993006993004,
            3.405594405594406, 3.41958041958042, 3.4335664335664333, 3.4475524475524475, 3.4615384615384617, 3.475524475524476,

            3.580419580419581, 3.5944055944055946, 3.6083916083916083, 3.6223776223776225, 3.6363636363636367, 3.6503496503496504,
            3.7552447552447554, 3.769230769230769, 3.7832167832167833, 3.7972027972027975, 3.8111888111888113, 3.825174825174825,
            3.93006993006993, 3.944055944055944, 3.9580419580419584, 3.972027972027972, 3.986013986013986, 4.0,
        ];
        // shape (2, 2, 5, 4)

        assert!(compare_outputs(&result, &expected));
    }

    #[test]
    fn test_comparison() {
        let lhs_dimensions = Dimensions4D::new(2, 3, 5, 4);
        let rhs_dimensions = Dimensions4D::new(1, 1, 1, 1);

        let mut backend = CpuBackend {};

        let lhs = linspace(-1.0, 1.0, lhs_dimensions.size());
        let rhs = vec![0.0];
        let expected = [
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];
        let lhs = backend.buffer_from_values(lhs_dimensions, lhs).unwrap();
        let rhs = backend.buffer_from_values(rhs_dimensions, rhs).unwrap();

        let result = backend
            .comparison(
                &lhs,
                &rhs,
                lhs_dimensions,
                rhs_dimensions,
                CmpType::GreaterThan,
            )
            .unwrap();
        let result = backend.get_values(&result, lhs_dimensions).unwrap();

        assert!(compare_outputs(&result, &expected));
    }

    #[test]
    fn test_upsample() {
        let mut backend = CpuBackend {};

        let input_dimensions = Dimensions4D::new(2, 4, 3, 2);
        let values = linspace(0.0, 1.0, input_dimensions.size());

        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected = [
            0.0, 0.0, 0.02127659574468085, 0.02127659574468085,
            0.0, 0.0, 0.02127659574468085, 0.02127659574468085,
            0.0425531914893617, 0.0425531914893617, 0.06382978723404255, 0.06382978723404255,
            0.0425531914893617, 0.0425531914893617, 0.06382978723404255, 0.06382978723404255,
            0.0851063829787234, 0.0851063829787234, 0.10638297872340426, 0.10638297872340426,
            0.0851063829787234, 0.0851063829787234, 0.10638297872340426, 0.10638297872340426,

            0.1276595744680851, 0.1276595744680851, 0.14893617021276595, 0.14893617021276595,
            0.1276595744680851, 0.1276595744680851, 0.14893617021276595, 0.14893617021276595,
            0.1702127659574468, 0.1702127659574468, 0.19148936170212766, 0.19148936170212766,
            0.1702127659574468, 0.1702127659574468, 0.19148936170212766, 0.19148936170212766,
            0.2127659574468085, 0.2127659574468085, 0.23404255319148937, 0.23404255319148937,
            0.2127659574468085, 0.2127659574468085, 0.23404255319148937, 0.23404255319148937,

            0.2553191489361702, 0.2553191489361702, 0.2765957446808511, 0.2765957446808511,
            0.2553191489361702, 0.2553191489361702, 0.2765957446808511, 0.2765957446808511,
            0.2978723404255319, 0.2978723404255319, 0.3191489361702128, 0.3191489361702128,
            0.2978723404255319, 0.2978723404255319, 0.3191489361702128, 0.3191489361702128,
            0.3404255319148936, 0.3404255319148936, 0.36170212765957444, 0.36170212765957444,
            0.3404255319148936, 0.3404255319148936, 0.36170212765957444, 0.36170212765957444,

            0.3829787234042553, 0.3829787234042553, 0.40425531914893614, 0.40425531914893614,
            0.3829787234042553, 0.3829787234042553, 0.40425531914893614, 0.40425531914893614,
            0.425531914893617, 0.425531914893617, 0.44680851063829785, 0.44680851063829785,
            0.425531914893617, 0.425531914893617, 0.44680851063829785, 0.44680851063829785,
            0.46808510638297873, 0.46808510638297873, 0.48936170212765956, 0.48936170212765956,
            0.46808510638297873, 0.46808510638297873, 0.48936170212765956, 0.48936170212765956,


            0.5106382978723404, 0.5106382978723404, 0.5319148936170213, 0.5319148936170213,
            0.5106382978723404, 0.5106382978723404, 0.5319148936170213, 0.5319148936170213,
            0.5531914893617021, 0.5531914893617021, 0.5744680851063829, 0.5744680851063829,
            0.5531914893617021, 0.5531914893617021, 0.5744680851063829, 0.5744680851063829,
            0.5957446808510638, 0.5957446808510638, 0.6170212765957447, 0.6170212765957447,
            0.5957446808510638, 0.5957446808510638, 0.6170212765957447, 0.6170212765957447,

            0.6382978723404256, 0.6382978723404256, 0.6595744680851063, 0.6595744680851063,
            0.6382978723404256, 0.6382978723404256, 0.6595744680851063, 0.6595744680851063,
            0.6808510638297872, 0.6808510638297872, 0.7021276595744681, 0.7021276595744681,
            0.6808510638297872, 0.6808510638297872, 0.7021276595744681, 0.7021276595744681,
            0.7234042553191489, 0.7234042553191489, 0.7446808510638298, 0.7446808510638298,
            0.7234042553191489, 0.7234042553191489, 0.7446808510638298, 0.7446808510638298,

            0.7659574468085106, 0.7659574468085106, 0.7872340425531915, 0.7872340425531915,
            0.7659574468085106, 0.7659574468085106, 0.7872340425531915, 0.7872340425531915,
            0.8085106382978723, 0.8085106382978723, 0.8297872340425532, 0.8297872340425532,
            0.8085106382978723, 0.8085106382978723, 0.8297872340425532, 0.8297872340425532,
            0.851063829787234, 0.851063829787234, 0.8723404255319148, 0.8723404255319148,
            0.851063829787234, 0.851063829787234, 0.8723404255319148, 0.8723404255319148,

            0.8936170212765957, 0.8936170212765957, 0.9148936170212766, 0.9148936170212766,
            0.8936170212765957, 0.8936170212765957, 0.9148936170212766, 0.9148936170212766,
            0.9361702127659575, 0.9361702127659575, 0.9574468085106382, 0.9574468085106382,
            0.9361702127659575, 0.9361702127659575, 0.9574468085106382, 0.9574468085106382,
            0.9787234042553191, 0.9787234042553191, 1.0, 1.0,
            0.9787234042553191, 0.9787234042553191, 1.0, 1.0,
        ];

        let input = backend
            .buffer_from_values(input_dimensions, values)
            .unwrap();

        let upsample_size = Dimensions2D::new(2, 2);
        let output_dimensions = Dimensions4D::new(
            input_dimensions.batch,
            input_dimensions.depth,
            input_dimensions.height * upsample_size.height,
            input_dimensions.width * upsample_size.height,
        );
        let upsampled = backend
            .upsample2d(&input, input_dimensions, upsample_size)
            .unwrap();
        let result = backend.get_values(&upsampled, output_dimensions).unwrap();

        assert!(compare_outputs(&result, &expected));
    }
}
