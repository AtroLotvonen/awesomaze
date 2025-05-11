use serde::{Deserialize, Serialize};

use super::tensor_error::{Result, TensorError};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dimensions4D {
    pub batch: usize,
    pub depth: usize,
    pub height: usize,
    pub width: usize,
}

impl From<Dimensions4D> for Vec<usize> {
    fn from(val: Dimensions4D) -> Self {
        vec![val.batch, val.depth, val.height, val.width]
    }
}

impl Dimensions4D {
    pub fn new(batch: usize, depth: usize, height: usize, width: usize) -> Self {
        Self {
            batch,
            depth,
            height,
            width,
        }
    }

    pub fn from_d3(batch: usize, dimensions3: Dimensions3D) -> Self {
        Self {
            batch,
            depth: dimensions3.depth,
            height: dimensions3.height,
            width: dimensions3.width,
        }
    }

    pub fn size(&self) -> usize {
        self.batch * self.depth * self.height * self.width
    }

    pub fn from_index(&self, index: usize) -> Result<usize> {
        match index {
            0 => Ok(self.batch),
            1 => Ok(self.depth),
            2 => Ok(self.height),
            3 => Ok(self.width),
            _ => Err(TensorError::DimensionIndexError { index }),
        }
    }

    pub fn stride(&self, index: usize) -> Result<usize> {
        match index {
            0 => Ok(self.depth * self.height * self.width),
            1 => Ok(self.height * self.width),
            2 => Ok(self.width),
            3 => Ok(1),
            _ => Err(TensorError::DimensionIndexError { index }),
        }
    }

    pub fn reduce_dims(&self, dims: &[usize]) -> Result<Self> {
        let mut reduced_dimensions =
            Dimensions4D::new(self.batch, self.depth, self.height, self.width);
        for dim in dims {
            match dim {
                0 => reduced_dimensions.batch = 1,
                1 => reduced_dimensions.depth = 1,
                2 => reduced_dimensions.height = 1,
                3 => reduced_dimensions.width = 1,
                _ => {
                    return Err(TensorError::DimensionIndexError { index: *dim });
                }
            }
        }
        Ok(reduced_dimensions)
    }

    pub fn keep_dims(&self, keep_dims: &[usize]) -> Result<Self> {
        let mut reduced_dimensions = Dimensions4D::new(1, 1, 1, 1);
        for dim in keep_dims {
            match dim {
                0 => reduced_dimensions.batch = self.batch,
                1 => reduced_dimensions.depth = self.depth,
                2 => reduced_dimensions.height = self.height,
                3 => reduced_dimensions.width = self.width,
                _ => {
                    return Err(TensorError::DimensionIndexError { index: *dim });
                }
            }
        }
        Ok(reduced_dimensions)
    }

    pub fn strides(&self) -> [usize; 4] {
        let n_stride = self.stride(0).unwrap();
        let c_stride = self.stride(1).unwrap();
        let h_stride = self.stride(2).unwrap();
        let w_stride = self.stride(3).unwrap();

        [n_stride, c_stride, h_stride, w_stride]
    }

    pub fn swap(&self, index1: usize, index2: usize) -> Result<Self> {
        let shape = Self::swap_shape(index1, index2)?;
        Ok(Self::new(
            self.from_index(shape[0])?,
            self.from_index(shape[1])?,
            self.from_index(shape[2])?,
            self.from_index(shape[3])?,
        ))
    }

    pub fn flat_index(&self, nchw: (usize, usize, usize, usize)) -> Result<usize> {
        let strides = self.strides();
        if nchw.0 >= self.batch {
            return Err(TensorError::TensorIndexOutOfBounds {
                dim: 0,
                indexed_size: nchw.0,
                real_size: self.batch,
            });
        }
        if nchw.1 >= self.depth {
            return Err(TensorError::TensorIndexOutOfBounds {
                dim: 1,
                indexed_size: nchw.1,
                real_size: self.depth,
            });
        }
        if nchw.2 >= self.height {
            return Err(TensorError::TensorIndexOutOfBounds {
                dim: 2,
                indexed_size: nchw.2,
                real_size: self.height,
            });
        }
        if nchw.3 >= self.width {
            return Err(TensorError::TensorIndexOutOfBounds {
                dim: 3,
                indexed_size: nchw.3,
                real_size: self.width,
            });
        }
        let index =
            nchw.0 * strides[0] + nchw.1 * strides[1] + nchw.2 * strides[2] + nchw.3 * strides[3];
        Ok(index)
    }

    pub fn nchw_index(&self, flat_index: usize) -> Result<(usize, usize, usize, usize)> {
        if flat_index >= self.size() {
            return Err(TensorError::DimensionIndexError { index: flat_index });
        }
        let (n, remainder) = div_mod(flat_index, self.depth * self.height * self.width);
        let (c, remainder) = div_mod(remainder, self.height * self.width);
        let (h, w) = div_mod(remainder, self.width);
        Ok((n, c, h, w))
    }

    fn swap_shape(index1: usize, index2: usize) -> Result<[usize; 4]> {
        let mut shape = [0, 1, 2, 3];
        if index1 == index2 {
            return Ok(shape);
        } else if index1 > 3 {
            return Err(TensorError::DimensionIndexError { index: index1 });
        } else if index2 > 3 {
            return Err(TensorError::DimensionIndexError { index: index2 });
        }
        shape.swap(index1, index2);
        Ok(shape)
    }

    /// Checks if the current dimension is broadcastable to the other dimensions and gives the
    /// multiples as dimensions which represent how many times each dimension is broadcastable to
    /// the other [`TensorError`]
    ///
    /// # Arguments
    ///
    /// * `other` - [`Dimensions4D`] the other dimension which self is tried to be broadcasted as
    ///
    /// # Returns
    /// the steps each dimension `batch`, `depth`, `height` and `width` are repeated
    ///
    /// # Examples
    /// ```rust
    /// use awesomlearn::machine_learning::nn::dimensions::Dimensions4D;
    /// let x = Dimensions4D::new(4, 8, 3, 10);
    /// let y = Dimensions4D::new(4, 1, 3, 1);
    /// let b = y.broadcast_dimensions(x).unwrap();
    /// assert_eq!(b, Dimensions4D::new(4, 8, 3, 10));
    /// ```
    /// ```rust
    /// use awesomlearn::machine_learning::nn::dimensions::Dimensions4D;
    /// let x = Dimensions4D::new(4, 8, 3, 10);
    /// let y = Dimensions4D::new(4, 8, 3, 10);
    /// let b = y.broadcast_dimensions(x).unwrap();
    /// assert_eq!(b, Dimensions4D::new(4, 8, 3, 10));
    /// ```
    pub fn broadcast_dimensions(&self, other: Dimensions4D) -> Result<Self> {
        if self.batch != other.batch && (self.batch != 1 && other.batch != 1) {
            return Err(TensorError::BroadcastDimensionsMismatch {
                lhs: *self,
                rhs: other,
            });
        };
        if self.depth != other.depth && (self.depth != 1 && other.depth != 1) {
            return Err(TensorError::BroadcastDimensionsMismatch {
                lhs: *self,
                rhs: other,
            });
        };
        if self.height != other.height && (self.height != 1 && other.height != 1) {
            return Err(TensorError::BroadcastDimensionsMismatch {
                lhs: *self,
                rhs: other,
            });
        };
        if self.width != other.width && (self.width != 1 && other.width != 1) {
            return Err(TensorError::BroadcastDimensionsMismatch {
                lhs: *self,
                rhs: other,
            });
        };
        let b = usize::max(self.batch, other.batch);
        let d = usize::max(self.depth, other.depth);
        let h = usize::max(self.height, other.height);
        let w = usize::max(self.width, other.width);
        let broadcasted_dimensions = Dimensions4D::new(b, d, h, w);
        Ok(broadcasted_dimensions)
    }

    pub fn matching_dimensions(&self, other: Dimensions4D) -> (bool, bool, bool, bool) {
        (
            self.batch == other.batch,
            self.depth == other.depth,
            self.height == other.height,
            self.width == other.width,
        )
    }
}

fn div_mod(a: usize, b: usize) -> (usize, usize) {
    let quotient = a / b;
    let remainder = a % b;
    (quotient, remainder)
}

impl From<[usize; 4]> for Dimensions4D {
    fn from(value: [usize; 4]) -> Self {
        Self::new(value[0], value[1], value[2], value[3])
    }
}

impl From<Dimensions4D> for [usize; 4] {
    fn from(value: Dimensions4D) -> Self {
        [value.batch, value.depth, value.height, value.width]
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Dimensions3D {
    pub depth: usize,
    pub height: usize,
    pub width: usize,
}

impl Dimensions3D {
    pub fn new(depth: usize, height: usize, width: usize) -> Self {
        Self {
            depth,
            height,
            width,
        }
    }

    pub fn size(&self) -> usize {
        self.depth * self.height * self.width
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct Dimensions2D {
    pub height: usize,
    pub width: usize,
}

impl Dimensions2D {
    pub fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }

    pub fn size(&self) -> usize {
        self.height * self.width
    }
}

impl From<[usize; 2]> for Dimensions2D {
    fn from(value: [usize; 2]) -> Self {
        Self::new(value[0], value[1])
    }
}
