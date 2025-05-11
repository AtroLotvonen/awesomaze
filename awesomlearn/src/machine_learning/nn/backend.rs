use std::fmt::Debug;

use super::dimensions::{Dimensions2D, Dimensions4D};
use super::tensor_error::Result;

#[derive(Clone, Debug)]
pub enum CmpType {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterOrEqualThan,
    LesserThan,
    LesserOrEqualThan,
}

pub trait Backend: Sized + Debug {
    type Storage: Debug;

    fn zeros(&mut self, dimensions: Dimensions4D) -> Result<Self::Storage>;
    fn ones(&mut self, dimensions: Dimensions4D) -> Result<Self::Storage>;
    fn buffer_from_values(
        &mut self,
        dimensions: Dimensions4D,
        values: Vec<f32>,
    ) -> Result<Self::Storage>;
    // Copy the content of the tensor
    fn copy(&mut self, buffer: &Self::Storage) -> Result<Self::Storage>;
    // Binary operations
    fn matmul(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        mnk: (usize, usize, usize),
        is_rhs_transposed: bool,
    ) -> Result<Self::Storage>;
    fn mul(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<Self::Storage>;
    fn add(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<Self::Storage>;
    fn sub(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<Self::Storage>;
    fn div(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<Self::Storage>;
    fn pow(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<Self::Storage>;
    // Unary operations
    fn transpose(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        dim1: usize,
        dim2: usize,
    ) -> Result<Self::Storage>;
    // Unary with parameters
    fn zero_pad(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        offset: Dimensions4D,
        new_dimensions: Dimensions4D,
    ) -> Result<Self::Storage>;
    fn slice(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        offset: Dimensions4D,
        new_dimensions: Dimensions4D,
    ) -> Result<Self::Storage>;
    fn max_pool2d(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        pool_size: Dimensions2D,
    ) -> Result<Self::Storage>;
    fn avg_pool2d(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        pool_size: Dimensions2D,
    ) -> Result<Self::Storage>;
    fn upsample2d(
        &mut self,
        buffer: &Self::Storage,
        input_dimensions: Dimensions4D,
        upsample_size: Dimensions2D,
    ) -> Result<Self::Storage>;
    fn conv2d(
        &mut self,
        buffer: &Self::Storage,
        kernels: &Self::Storage,
        dimensions: Dimensions4D,
        kernel_size: Dimensions4D,
    ) -> Result<Self::Storage>;
    fn conv_transpose2d(
        &mut self,
        buffer: &Self::Storage,
        kernels: &Self::Storage,
        dimensions: Dimensions4D,
        kernel_size: Dimensions4D,
    ) -> Result<Self::Storage>;
    fn relu(&mut self, buffer: &Self::Storage, dimensions: Dimensions4D) -> Result<Self::Storage>;
    fn leaky_relu(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        alpha: f32,
    ) -> Result<Self::Storage>;
    fn sigmoid(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
    ) -> Result<Self::Storage>;
    fn exp(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
    ) -> Result<Self::Storage>;
    fn ln(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
    ) -> Result<Self::Storage>;
    fn sum_reduce(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        dims: &[usize],
    ) -> Result<Self::Storage>;
    fn max_dim(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        dim: usize,
    ) -> Result<Self::Storage>;
    fn arg_max(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        dim: usize,
    ) -> Result<Self::Storage>;
    fn gather(
        &mut self,
        buffer: &Self::Storage,
        indices: &Self::Storage,
        dimensions: Dimensions4D,
        dim: usize,
    ) -> Result<Self::Storage>;
    fn scatter(
        &mut self,
        buffer: &Self::Storage,
        indices: &Self::Storage,
        dimensions: Dimensions4D,
        out_dimensions: Dimensions4D,
        dim: usize,
    ) -> Result<Self::Storage>;
    fn comparison(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
        comparison_type: CmpType,
    ) -> Result<Self::Storage>;
    fn get_values(&mut self, buffer: &Self::Storage, dimensions: Dimensions4D) -> Result<Vec<f32>>;
}
