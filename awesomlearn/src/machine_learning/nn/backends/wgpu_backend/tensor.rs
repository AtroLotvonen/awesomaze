use std::{borrow::Borrow, collections::HashMap, ops::Deref, rc::Rc};

use crate::machine_learning::nn::nn_model::Backend;
use bytemuck::{Pod, Zeroable};
use serde::Serialize;
use uuid::Uuid;
use wgpu::Buffer;

use crate::machine_learning::nn::nn_model_config::Dimensions4D;

use super::{
    kernel::{Kernel, Workgroup},
    WgpuBackend,
};

pub enum TensorInit {
    Zeros,
    Ones,
    FromValues(Vec<f32>),
}

pub struct Tensor {
    dimensions: Dimensions4D,
    buffer: Rc<Buffer>,
}

impl Tensor {
    pub fn new(init_values: TensorInit, dimensions: Dimensions4D, backend: &mut WgpuBackend) -> Self {
        let values = match init_values {
            TensorInit::Zeros => vec![0.0; dimensions.size()],
            TensorInit::Ones => vec![1.0; dimensions.size()],
            TensorInit::FromValues(values) => {
                assert_eq!(dimensions.size(), values.len());
                values.to_vec()
            }
        };
        let buffer = backend.buffer_from_values(dimensions, values).unwrap();
        Self { buffer, dimensions }
    }

    pub fn uniform_from_bytes(
        values: &[u8],
        dimensions: Dimensions4D,
        backend: &WgpuBackend,
    ) -> Self {
        let buffer = Rc::new(backend.create_uniform_buffer_from_values(values));
        Self { buffer, dimensions }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn dimensions(&self) -> Dimensions4D {
        self.dimensions
    }

    // pub fn matmul(&self, other: &Tensor) -> Tensor {
    //     // let matmul_kernel = self.backend.create_kernel(kernel_source, workgroup_size, kernel_name)
    //
    // }
}

pub struct UnaryNode<'a> {
    input: &'a Tensor,
    output: Tensor,
    constants: Tensor,
    kernel: Rc<Kernel>,
}

pub struct BinaryNode<'a> {
    a: &'a Tensor,
    b: &'a Tensor,
    c: Tensor,
    constants: Tensor,
    kernel: Rc<Kernel>,
}

impl<'a> UnaryNode<'a> {
    pub fn forward(&self, backend: &mut WgpuBackend) {
        // The order of the tensors must be the same as in the kernel!
        let buffers = &[
            self.constants.buffer.borrow(),
            self.input.buffer.borrow(),
            self.output.buffer.borrow(),
        ];
        backend.schedule_kernel(&self.kernel, buffers);
    }
}

impl<'a> BinaryNode<'a> {
    pub fn prepare(&self, _backend: &mut WgpuBackend) {
        todo!()
    }

    pub fn forward(&self, backend: &mut WgpuBackend) {
        // The order of the tensors must be the same as in the kernel!
        let buffers = vec![
            self.constants.buffer.borrow(),
            self.a.buffer.borrow(),
            self.b.buffer.borrow(),
            self.c.buffer.borrow(),
        ];
        backend.schedule_kernel(&self.kernel, &buffers);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct MatMulConstants {
    m: u32, // layer_size in forward
    n: u32, // batch_size in forward
    k: u32, // input_size in forward
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct TransposeConstants {
    p: u32,
    q: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct PaddingConstants {
    p: u32,
    q: u32,
    p_xl: u32,
    q_xl: u32,
}

pub enum Op {
    Matmul(),
    Transpose(),
}

pub fn gemm_naive<'a>(
    a: &'a Tensor,
    b: &'a Tensor,
    // FIXME: interior mutability for backend?
    backend: &mut WgpuBackend,
) -> BinaryNode<'a> {
    let m = a.dimensions().height;
    let n = b.dimensions().width;
    let k = a.dimensions().width;
    // dimensions must match!
    assert_eq!(k, b.dimensions().height);

    let ts = 16;
    let local = [ts, ts];

    let m_global: u32 = ((m as u32 - 1) / ts + 1) * ts;
    let n_global: u32 = ((n as u32 - 1) / ts + 1) * ts;

    let global = [m_global, n_global];
    // Get the kernel or create if doesn't exist
    let kernel_name = "gemm_naive";
    let kernel = match backend.get_kernel(kernel_name) {
        Some(kernel) => kernel,
        None => {
            let kernel_source = include_str!("./gemm_naive_rm.wgsl");
            backend
                .create_kernel(
                    kernel_source,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    kernel_name,
                )
                .unwrap()
        }
    };

    // create constant and c tensor
    let constant_dimensions = Dimensions4D::new(1, 1, 1, 1);
    let c_dimensions = Dimensions4D::new(1, 1, m, n);
    let constants = MatMulConstants {
        m: m.try_into().unwrap(),
        n: n.try_into().unwrap(),
        k: k.try_into().unwrap(),
    };
    let constant_tensor = Tensor::uniform_from_bytes(
        bytemuck::cast_slice(&[constants]),
        constant_dimensions,
        backend,
    );
    let c = Tensor::new(TensorInit::Zeros, c_dimensions, backend);
    BinaryNode {
        a,
        b,
        c,
        constants: constant_tensor,
        kernel,
    }
}

pub fn gemm_tiling<'a>(a: &'a Tensor, b: &'a Tensor, backend: &mut WgpuBackend) -> BinaryNode<'a> {
    let m = a.dimensions().height;
    let n = b.dimensions().width;
    let k = a.dimensions().width;
    // dimensions must match!
    assert_eq!(k, b.dimensions().height);

    let ts = 16;
    let local = [ts, ts];
    let global = [m.try_into().unwrap(), n.try_into().unwrap()];
    // Get the kernel or create if doesn't exist
    let kernel_name = "gemm_tiling";
    let kernel = match backend.get_kernel(kernel_name) {
        Some(kernel) => kernel,
        None => {
            let kernel_source = include_str!("./gemm_tiling.wgsl");
            backend
                .create_kernel(
                    kernel_source,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    kernel_name,
                )
                .unwrap()
        }
    };

    // create constant and c tensor
    let constant_dimensions = Dimensions4D::new(1, 1, 1, 1);
    let c_dimensions = Dimensions4D::new(1, 1, m, n);
    let constants = MatMulConstants {
        m: m.try_into().unwrap(),
        n: n.try_into().unwrap(),
        k: k.try_into().unwrap(),
    };
    let constant_tensor = Tensor::uniform_from_bytes(
        bytemuck::cast_slice(&[constants]),
        constant_dimensions,
        backend,
    );
    let c = Tensor::new(TensorInit::Zeros, c_dimensions, backend);
    BinaryNode {
        a,
        b,
        c,
        constants: constant_tensor,
        kernel,
    }
}

pub fn gemm_tiling_wpt<'a>(
    a: &'a Tensor,
    b: &'a Tensor,
    backend: &mut WgpuBackend,
) -> BinaryNode<'a> {
    let m = a.dimensions().height;
    let n = b.dimensions().width;
    let k = a.dimensions().width;
    // dimensions must match!
    assert_eq!(k, b.dimensions().height);

    let ts = 32;
    let wpt = 8;
    let local = [ts, ts / wpt];
    let mut n_wpt = n.try_into().unwrap();
    n_wpt /= wpt;
    let global = [m.try_into().unwrap(), n_wpt];
    // Get the kernel or create if doesn't exist
    let kernel_name = "gemm_tiling_more_work_per_thread";
    let kernel = match backend.get_kernel(kernel_name) {
        Some(kernel) => kernel,
        None => {
            let kernel_source = include_str!("./gemm_more_work_per_tile.wgsl");
            backend
                .create_kernel(
                    kernel_source,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    kernel_name,
                )
                .unwrap()
        }
    };

    // create constant and c tensor
    let constant_dimensions = Dimensions4D::new(1, 1, 1, 1);
    let c_dimensions = Dimensions4D::new(1, 1, m, n);
    let constants = MatMulConstants {
        m: m.try_into().unwrap(),
        n: n.try_into().unwrap(),
        k: k.try_into().unwrap(),
    };
    let constant_tensor = Tensor::uniform_from_bytes(
        bytemuck::cast_slice(&[constants]),
        constant_dimensions,
        backend,
    );
    let c = Tensor::new(TensorInit::Zeros, c_dimensions, backend);
    BinaryNode {
        a,
        b,
        c,
        constants: constant_tensor,
        kernel,
    }
}

pub fn gemm_tiling_wide_float<'a>(
    a: &'a Tensor,
    b: &'a Tensor,
    backend: &mut WgpuBackend,
) -> BinaryNode<'a> {
    let m = a.dimensions().height;
    let n = b.dimensions().width;
    let k = a.dimensions().width;
    // dimensions must match!
    assert_eq!(k, b.dimensions().height);

    let ts = 16;
    let width = 4;
    let local = [ts / width, ts];
    let mut m_width = m.try_into().unwrap();
    m_width /= width;
    let global = [m_width, n.try_into().unwrap()];
    // Get the kernel or create if doesn't exist
    let kernel_name = "gemm_wide_float";
    let kernel = match backend.get_kernel(kernel_name) {
        Some(kernel) => kernel,
        None => {
            let kernel_source = include_str!("./gemm_wide_float.wgsl");
            backend
                .create_kernel(
                    kernel_source,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    kernel_name,
                )
                .unwrap()
        }
    };

    // create constant and c tensor
    let constant_dimensions = Dimensions4D::new(1, 1, 1, 1);
    let c_dimensions = Dimensions4D::new(1, 1, m, n);
    let constants = MatMulConstants {
        m: m.try_into().unwrap(),
        n: n.try_into().unwrap(),
        k: k.try_into().unwrap(),
    };
    let constant_tensor = Tensor::uniform_from_bytes(
        bytemuck::cast_slice(&[constants]),
        constant_dimensions,
        backend,
    );
    let c = Tensor::new(TensorInit::Zeros, c_dimensions, backend);
    BinaryNode {
        a,
        b,
        c,
        constants: constant_tensor,
        kernel,
    }
}

pub fn gemm_tiling_wpt_b_transpose<'a>(
    a: &'a Tensor,
    b: &'a Tensor,
    backend: &mut WgpuBackend,
) -> BinaryNode<'a> {
    let m = a.dimensions().height;
    // B is transposed so n is the height (normally it's the width)
    let n = b.dimensions().height;
    let k = a.dimensions().width;
    // the b is transposed so the m and n should match
    assert_eq!(k, b.dimensions().width);

    let ts = 32;
    let wpt = 8;
    let rts = ts / wpt; // reduced tile size in y dimension
    let local = [ts, rts];
    let mut n_wpt = n.try_into().unwrap();
    n_wpt /= wpt;
    let global = [m.try_into().unwrap(), n_wpt];
    // Get the kernel or create if doesn't exist
    let kernel_name = "gemm_tiling_more_work_per_thread_tranposed_b";
    let kernel_name = format!("{}-{}", kernel_name, Uuid::new_v4());
    let kernel = match backend.get_kernel(&kernel_name) {
        Some(kernel) => kernel,
        None => {
            let kernel_source = include_str!("./gemm_more_work_per_tile_b_transpose.wgsl");
            backend
                .create_kernel(
                    kernel_source,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    &kernel_name,
                )
                .unwrap()
        }
    };

    // create constant and c tensor
    let constant_dimensions = Dimensions4D::new(1, 1, 1, 1);
    let c_dimensions = Dimensions4D::new(1, 1, m, n);
    let constants = MatMulConstants {
        m: m.try_into().unwrap(),
        n: n.try_into().unwrap(),
        k: k.try_into().unwrap(),
    };
    let constant_tensor = Tensor::uniform_from_bytes(
        bytemuck::cast_slice(&[constants]),
        constant_dimensions,
        backend,
    );
    let c = Tensor::new(TensorInit::Zeros, c_dimensions, backend);
    BinaryNode {
        a,
        b,
        c,
        constants: constant_tensor,
        kernel,
    }
}

pub fn transpose_cpu<T>(mat: &[T], dimensions: Dimensions4D) -> Vec<T>
where
    T: Copy,
{
    let mut transposed_mat = Vec::new();

    for x in 0..dimensions.width {
        for y in 0..dimensions.height {
            let index = y * dimensions.width + x;
            transposed_mat.push(mat[index])
        }
    }
    transposed_mat
}

pub fn zeropad_cpu<T>(mat: &[T], dimensions: Dimensions4D, new_dimensions: Dimensions4D) -> Vec<T>
where
    T: Copy + Default,
{
    let mut padded_mat = Vec::with_capacity(new_dimensions.width * new_dimensions.height);

    for x in 0..new_dimensions.width {
        for y in 0..new_dimensions.height {
            let mut value = T::default();
            if x < dimensions.width && y < dimensions.height {
                let index = x * dimensions.height + y;
                value = mat[index];
            }
            padded_mat.push(value)
        }
    }
    padded_mat
}

pub fn transpose<'a>(input: &'a Tensor, backend: &mut WgpuBackend) -> UnaryNode<'a> {
    let p = input.dimensions().height;
    let q = input.dimensions().width;
    let ts = 16;
    let local = [ts, ts];
    let global = [p.try_into().unwrap(), q.try_into().unwrap()];
    // Transpose kernel
    let kernel_name = format!("{}-{}", "transpose", Uuid::new_v4());
    let kernel = match backend.get_kernel(&kernel_name) {
        Some(kernel) => kernel,
        None => {
            let kernel_source = include_str!("./transpose.wgsl");
            backend
                .create_kernel(
                    kernel_source,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    &kernel_name,
                )
                .unwrap()
        }
    };

    // create constant and c tensor
    let constant_dimensions = Dimensions4D::new(1, 1, 1, 1);
    // Swap the p and q for the output dimensions
    let output_dimensions = Dimensions4D::new(1, 1, q, p);
    let constants = TransposeConstants {
        p: p.try_into().unwrap(),
        q: q.try_into().unwrap(),
    };
    let constant_tensor = Tensor::uniform_from_bytes(
        bytemuck::cast_slice(&[constants]),
        constant_dimensions,
        backend,
    );
    let output = Tensor::new(TensorInit::Zeros, output_dimensions, backend);
    UnaryNode {
        input,
        output,
        constants: constant_tensor,
        kernel,
    }
}

pub fn add_zero_padding<'a>(
    input: &'a Tensor,
    backend: &mut WgpuBackend,
    new_dimensions: Dimensions4D,
) -> UnaryNode<'a> {
    let p = input.dimensions().height;
    let q = input.dimensions().width;
    let p_xl = new_dimensions.height;
    let q_xl = new_dimensions.width;
    let ts = 16;
    let local = [ts, ts];
    let global = [p_xl.try_into().unwrap(), q_xl.try_into().unwrap()];
    // Transpose kernel
    let kernel_name = format!("{}-{}", "zero-padding", Uuid::new_v4());
    let kernel = match backend.get_kernel(&kernel_name) {
        Some(kernel) => kernel,
        None => {
            let kernel_source = include_str!("./add_zero_padding.wgsl");
            backend
                .create_kernel(
                    kernel_source,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    &kernel_name,
                )
                .unwrap()
        }
    };

    // create constant and c tensor
    let constant_dimensions = Dimensions4D::new(1, 1, 1, 1);
    // Swap the p and q for the output dimensions
    let output_dimensions = Dimensions4D::new(1, 1, p_xl, q_xl);
    let constants = PaddingConstants {
        p: p.try_into().unwrap(),
        q: q.try_into().unwrap(),
        p_xl: p_xl.try_into().unwrap(),
        q_xl: q_xl.try_into().unwrap(),
    };
    let constant_tensor = Tensor::uniform_from_bytes(
        bytemuck::cast_slice(&[constants]),
        constant_dimensions,
        backend,
    );
    let output = Tensor::new(TensorInit::Zeros, output_dimensions, backend);
    UnaryNode {
        input,
        output,
        constants: constant_tensor,
        kernel,
    }
}

pub fn remove_zero_padding<'a>(
    input: &'a Tensor,
    backend: &mut WgpuBackend,
    old_dimensions: Dimensions4D,
) -> UnaryNode<'a> {
    let p_xl = input.dimensions().height;
    let q_xl = input.dimensions().width;
    let p = old_dimensions.height;
    let q = old_dimensions.width;
    let ts = 16;
    let local = [ts, ts];
    let p_global: u32 = ((p as u32 - 1) / ts + 1) * ts;
    let q_global: u32 = ((q as u32 - 1) / ts + 1) * ts;
    let global = [p_global, q_global];
    // Transpose kernel
    let kernel_name = format!("{}-{}", "remove-zero-padding", Uuid::new_v4());
    let kernel = match backend.get_kernel(&kernel_name) {
        Some(kernel) => kernel,
        None => {
            let kernel_source = include_str!("./remove_zero_padding.wgsl");
            backend
                .create_kernel(
                    kernel_source,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    &kernel_name,
                )
                .unwrap()
        }
    };

    // create constant and c tensor
    let constant_dimensions = Dimensions4D::new(1, 1, 1, 1);
    // Swap the p and q for the output dimensions
    let output_dimensions = Dimensions4D::new(1, 1, p, q);
    let constants = PaddingConstants {
        p: p.try_into().unwrap(),
        q: q.try_into().unwrap(),
        p_xl: p_xl.try_into().unwrap(),
        q_xl: q_xl.try_into().unwrap(),
    };
    let constant_tensor = Tensor::uniform_from_bytes(
        bytemuck::cast_slice(&[constants]),
        constant_dimensions,
        backend,
    );
    let output = Tensor::new(TensorInit::Zeros, output_dimensions, backend);
    UnaryNode {
        input,
        output,
        constants: constant_tensor,
        kernel,
    }
}

pub fn gemm_register_blocking<'a>(
    a: &'a Tensor,
    b: &'a Tensor,
    backend: &mut WgpuBackend,
) -> BinaryNode<'a> {
    let m = a.dimensions().height;
    // B is transposed so n is the height (normally it's the width)
    let n = b.dimensions().height;
    let k = a.dimensions().width;
    // the b is transposed so the m and n should match
    assert_eq!(k, b.dimensions().width);

    let wptm = 8;
    let wptn = 8;

    let tsm = 128;
    let tsn = 128;
    let local = [tsm / wptm, tsn / wptn];

    let mut m_wpt = m.try_into().unwrap();
    m_wpt /= wptm;
    let mut n_wpt = n.try_into().unwrap();
    n_wpt /= wptn;
    let global = [m_wpt, n_wpt];

    // Get the kernel or create if doesn't exist
    let kernel_name = "gemm_tiling_more_work_per_thread";
    let kernel = match backend.get_kernel(kernel_name) {
        Some(kernel) => kernel,
        None => {
            let kernel_source = include_str!("./gemm_register_blocking.wgsl");
            backend
                .create_kernel(
                    kernel_source,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    kernel_name,
                )
                .unwrap()
        }
    };

    // create constant and c tensor
    let constant_dimensions = Dimensions4D::new(1, 1, 1, 1);
    let c_dimensions = Dimensions4D::new(1, 1, m, n);
    let constants = MatMulConstants {
        m: m.try_into().unwrap(),
        n: n.try_into().unwrap(),
        k: k.try_into().unwrap(),
    };
    let constant_tensor = Tensor::uniform_from_bytes(
        bytemuck::cast_slice(&[constants]),
        constant_dimensions,
        backend,
    );
    let c = Tensor::new(TensorInit::Zeros, c_dimensions, backend);
    BinaryNode {
        a,
        b,
        c,
        constants: constant_tensor,
        kernel,
    }
}

pub fn gemm_register_blocking_general<'a>(
    a: &'a Tensor,
    b: &'a Tensor,
    backend: &mut WgpuBackend,
) -> BinaryNode<'a> {
    let m = a.dimensions().height;
    // B is transposed so n is the height (normally it's the width)
    let n = b.dimensions().height;
    let k = a.dimensions().width;
    // the b is transposed so the m and n should match
    assert_eq!(k, b.dimensions().width);

    let wptm = 8;
    let wptn = 8;

    // These can be varied in this kernel
    let tsm = 128;
    let tsn = 128;
    let local = [tsm / wptm, tsn / wptn];

    let mut m_wpt = m.try_into().unwrap();
    m_wpt /= wptm;
    let mut n_wpt = n.try_into().unwrap();
    n_wpt /= wptn;
    let global = [m_wpt, n_wpt];

    // Get the kernel or create if doesn't exist
    let kernel_name = "gemm_register_blocking_with_arbitary_size";
    let kernel = match backend.get_kernel(kernel_name) {
        Some(kernel) => kernel,
        None => {
            let kernel_source = include_str!("./gemm_register_blocking_with_arbitary_size.wgsl");
            backend
                .create_kernel(
                    kernel_source,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    kernel_name,
                )
                .unwrap()
        }
    };

    // create constant and c tensor
    let constant_dimensions = Dimensions4D::new(1, 1, 1, 1);
    let c_dimensions = Dimensions4D::new(1, 1, m, n);
    let constants = MatMulConstants {
        m: m.try_into().unwrap(),
        n: n.try_into().unwrap(),
        k: k.try_into().unwrap(),
    };
    let constant_tensor = Tensor::uniform_from_bytes(
        bytemuck::cast_slice(&[constants]),
        constant_dimensions,
        backend,
    );
    let c = Tensor::new(TensorInit::Zeros, c_dimensions, backend);
    BinaryNode {
        a,
        b,
        c,
        constants: constant_tensor,
        kernel,
    }
}

pub fn execute_naive_gemm(
    lhs: Vec<f32>,
    rhs: Vec<f32>,
    bmnk: (usize, usize, usize, usize),
) -> Vec<f32> {
    // TODO: doesn't currently support batching
    let (_b, m, n, k) = bmnk;
    let mut backend = WgpuBackend::default();

    let a_dimensions = Dimensions4D::new(1, 1, m, k);
    let b_dimensions = Dimensions4D::new(1, 1, k, n);
    // the size should be the same as the dimensions size
    let a = Tensor::new(TensorInit::FromValues(lhs), a_dimensions, &mut backend);
    let b = Tensor::new(TensorInit::FromValues(rhs), b_dimensions, &mut backend);
    let gemm = gemm_naive(&a, &b, &mut backend);
    gemm.forward(&mut backend);
    // Wait for completion
    backend.buffer_to_values(gemm.c.buffer())
}

pub fn execute_gemm8(lhs: Vec<f32>, rhs: Vec<f32>, bmnk: (usize, usize, usize, usize)) -> Vec<f32> {
    // TODO: doesn't currently support batching
    let (_b, m, n, k) = bmnk;
    let mut backend = WgpuBackend::default();

    let a_dimensions = Dimensions4D::new(1, 1, m, k);
    let b_dimensions = Dimensions4D::new(1, 1, k, n);
    // the size should be the same as the dimensions size
    let a = Tensor::new(TensorInit::FromValues(lhs), a_dimensions, &mut backend);
    let b = Tensor::new(TensorInit::FromValues(rhs), b_dimensions, &mut backend);
    // padding kernels
    // These consts need to be synced with the kernel
    const TSN: usize = 128;
    const TSM: usize = 128;
    const TSK: usize = 16;
    // if B
    let b_p_xl = ((b.dimensions().height - 1) / TSK + 1) * TSK;
    let b_q_xl = ((b.dimensions().width - 1) / TSN + 1) * TSN;
    let b_padded_dimensions = Dimensions4D::new(1, 1, b_p_xl, b_q_xl);
    // if A
    let a_p_xl = ((a.dimensions().height - 1) / TSM + 1) * TSM;
    let a_q_xl = ((a.dimensions().width - 1) / TSK + 1) * TSK;
    let a_padded_dimensions = Dimensions4D::new(1, 1, a_p_xl, a_q_xl);

    let c_unpadded_dimensions =
        Dimensions4D::new(1, 1, a.dimensions().height, b.dimensions().width);

    // Kernels
    let a_pad = add_zero_padding(&a, &mut backend, a_padded_dimensions);
    let b_pad = add_zero_padding(&b, &mut backend, b_padded_dimensions);

    let t = transpose(&b_pad.output, &mut backend);
    let gemm = gemm_register_blocking_general(&a_pad.output, &t.output, &mut backend);

    let remove_gemm_padding = remove_zero_padding(&gemm.c, &mut backend, c_unpadded_dimensions);

    // Padding, transpose, gemm and remove padding
    a_pad.forward(&mut backend);
    b_pad.forward(&mut backend);
    t.forward(&mut backend);
    gemm.forward(&mut backend);
    remove_gemm_padding.forward(&mut backend);
    // Wait for completion
    backend.buffer_to_values(&remove_gemm_padding.output.buffer())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use float_cmp::approx_eq;

    use super::*;
    use crate::machine_learning::{linspace, nn::nn_model_config::Dimensions4D};

    fn compare_outputs(a: &[f32], b: &[f32]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for (&a, &b) in a.iter().zip(b) {
            if !approx_eq!(f32, a, b) {
                return false;
            }
        }
        true
    }

    fn setup_gemm(n: usize, m: usize, k: usize) -> (Tensor, Tensor, WgpuBackend) {
        let mut backend = WgpuBackend::default();
        let a_dimensions = Dimensions4D::new(1, 1, m, k);
        let b_dimensions = Dimensions4D::new(1, 1, k, n);
        // the size should be the same as the dimensions size
        let a_values = linspace(0.0, 1.0, k * m);
        // let mut b_values = vec![0.0_f32; n * k/ 2];
        let mut b_values = linspace(0.0, 1.0, n * k / 2);
        b_values.append(&mut vec![0.0_f32; n * k / 2]);
        // let b_values = linspace(0.0, 1.0, n * k);
        let a = Tensor::new(
            TensorInit::FromValues(a_values.clone()),
            a_dimensions,
            &mut backend,
        );
        let b = Tensor::new(
            TensorInit::FromValues(b_values.clone()),
            b_dimensions,
            &mut backend,
        );
        (a, b, backend)
    }

    fn print_tensor_to_file(tensor: &Tensor, backend: &mut WgpuBackend) {
        let values = backend.buffer_to_values(tensor.buffer());
        let rows = tensor.dimensions().width;
        let mut output_str = String::new();
        let values = values
            .iter()
            .map(|&x| format!("{:.2}", x))
            .collect::<Vec<_>>();
        for row in values.chunks(rows) {
            output_str.push_str(&format!("{:?}", row));
            output_str.push('\n');
        }
        let mut file = std::fs::File::create("gemm_result.txt").unwrap();
        file.write_all(output_str.as_bytes()).unwrap();
    }

    fn test_naive_gemm(n: usize, m: usize, k: usize, iters: usize) -> Vec<f32> {
        tracy_client::Client::start();
        let _span = tracy_client::span!("naive gemm");

        // tracy_client::Client::start();
        let (a, b, mut backend) = setup_gemm(n, m, k);
        let gemm = gemm_naive(&a, &b, &mut backend);
        for _ in 0..iters {
            gemm.forward(&mut backend);
        }
        // Wait for completion
        backend.buffer_to_values(&gemm.c.buffer())
        // print_tensor_to_file(&gemm.c, &mut backend)
    }

    fn test_gemm2(n: usize, m: usize, k: usize, iters: usize) -> Vec<f32> {
        tracy_client::Client::start();
        let _span = tracy_client::span!("tiled gemm");

        let (a, b, mut backend) = setup_gemm(n, m, k);
        let gemm = gemm_tiling(&a, &b, &mut backend);
        for _ in 0..iters {
            gemm.forward(&mut backend);
        }
        // Wait for completion
        backend.buffer_to_values(&gemm.c.buffer())
        // print_tensor_to_file(&gemm.c, &mut backend)
    }

    fn test_gemm3(n: usize, m: usize, k: usize, iters: usize) -> Vec<f32> {
        tracy_client::Client::start();
        let _span = tracy_client::span!("tiled gemm with MWPT");

        let (a, b, mut backend) = setup_gemm(n, m, k);
        let gemm = gemm_tiling_wpt(&a, &b, &mut backend);
        for _ in 0..iters {
            gemm.forward(&mut backend);
        }
        // Wait for completion
        backend.buffer_to_values(&gemm.c.buffer())
        // print_tensor_to_file(&gemm.c, &mut backend)
    }

    fn test_gemm4(n: usize, m: usize, k: usize, iters: usize) -> Vec<f32> {
        tracy_client::Client::start();
        let _span = tracy_client::span!("tiled gemm with wider float");

        let (a, b, mut backend) = setup_gemm(n, m, k);
        let gemm = gemm_tiling_wide_float(&a, &b, &mut backend);
        for _ in 0..iters {
            gemm.forward(&mut backend);
        }
        // Wait for completion
        backend.buffer_to_values(&gemm.c.buffer())
        // print_tensor_to_file(&gemm.c, &mut backend)
    }

    fn test_gemm5(n: usize, m: usize, k: usize, iters: usize) -> Vec<f32> {
        tracy_client::Client::start();
        let _span = tracy_client::span!("gemm with transposed B");

        let (a, b, mut backend) = setup_gemm(n, m, k);
        let t = transpose(&b, &mut backend);
        let gemm = gemm_tiling_wpt_b_transpose(&a, &t.output, &mut backend);
        // let out_t = transpose(&gemm.c, &mut backend);

        // Execute
        t.forward(&mut backend);
        for _ in 0..iters {
            gemm.forward(&mut backend);
        }
        // out_t.forward(&mut backend);
        // Wait for completion
        backend.buffer_to_values(&gemm.c.buffer())
    }

    fn test_gemm6(n: usize, m: usize, k: usize, iters: usize) -> Vec<f32> {
        tracy_client::Client::start();
        let _span = tracy_client::span!("gemm with register blocking");

        let (a, b, mut backend) = setup_gemm(n, m, k);
        let t = transpose(&b, &mut backend);
        let b_transpose = &t.output;
        let gemm = gemm_register_blocking(&a, b_transpose, &mut backend);

        // Transpose and then gemm
        t.forward(&mut backend);
        for _ in 0..iters {
            gemm.forward(&mut backend);
        }
        // Wait for completion
        backend.buffer_to_values(&gemm.c.buffer())
        // print_tensor_to_file(&gemm.c, &mut backend)
    }

    fn test_gemm7(n: usize, m: usize, k: usize, iters: usize) -> Vec<f32> {
        tracy_client::Client::start();
        let _span = tracy_client::span!("gemm with register blocking and arbitary size");

        let (a, b, mut backend) = setup_gemm(n, m, k);
        // padding kernels
        // These consts need to be synced with the kernel
        const TSN: usize = 128;
        const TSM: usize = 128;
        const TSK: usize = 16;
        // if B
        let b_p_xl = ((b.dimensions().height - 1) / TSK + 1) * TSK;
        let b_q_xl = ((b.dimensions().width - 1) / TSN + 1) * TSN;
        let b_padded_dimensions = Dimensions4D::new(1, 1, b_p_xl, b_q_xl);
        // if A
        let a_p_xl = ((a.dimensions().height - 1) / TSM + 1) * TSM;
        let a_q_xl = ((a.dimensions().width - 1) / TSK + 1) * TSK;
        let a_padded_dimensions = Dimensions4D::new(1, 1, a_p_xl, a_q_xl);

        let c_unpadded_dimensions =
            Dimensions4D::new(1, 1, a.dimensions().height, b.dimensions().width);

        // Kernels
        let a_pad = add_zero_padding(&a, &mut backend, a_padded_dimensions);
        let b_pad = add_zero_padding(&b, &mut backend, b_padded_dimensions);

        let t = transpose(&b_pad.output, &mut backend);
        let gemm = gemm_register_blocking(&a_pad.output, &t.output, &mut backend);

        let remove_gemm_padding = remove_zero_padding(&gemm.c, &mut backend, c_unpadded_dimensions);

        // Padding, transpose, gemm and remove padding
        a_pad.forward(&mut backend);
        b_pad.forward(&mut backend);
        t.forward(&mut backend);
        for _ in 0..iters {
            gemm.forward(&mut backend);
        }
        remove_gemm_padding.forward(&mut backend);
        // Wait for completion
        backend.buffer_to_values(&remove_gemm_padding.output.buffer())
    }

    fn test_gemm8(n: usize, m: usize, k: usize, iters: usize) -> Vec<f32> {
        tracy_client::Client::start();
        let _span = tracy_client::span!("gemm with register blocking general");

        let (a, b, mut backend) = setup_gemm(n, m, k);
        // padding kernels
        // These consts need to be synced with the kernel
        const TSN: usize = 128;
        const TSM: usize = 128;
        const TSK: usize = 16;
        // if B
        let b_p_xl = ((b.dimensions().height - 1) / TSK + 1) * TSK;
        let b_q_xl = ((b.dimensions().width - 1) / TSN + 1) * TSN;
        let b_padded_dimensions = Dimensions4D::new(1, 1, b_p_xl, b_q_xl);
        // if A
        let a_p_xl = ((a.dimensions().height - 1) / TSM + 1) * TSM;
        let a_q_xl = ((a.dimensions().width - 1) / TSK + 1) * TSK;
        let a_padded_dimensions = Dimensions4D::new(1, 1, a_p_xl, a_q_xl);

        let c_unpadded_dimensions =
            Dimensions4D::new(1, 1, a.dimensions().height, b.dimensions().width);

        // Kernels
        let a_pad = add_zero_padding(&a, &mut backend, a_padded_dimensions);
        let b_pad = add_zero_padding(&b, &mut backend, b_padded_dimensions);

        let t = transpose(&b_pad.output, &mut backend);
        let gemm = gemm_register_blocking_general(&a_pad.output, &t.output, &mut backend);

        let remove_gemm_padding = remove_zero_padding(&gemm.c, &mut backend, c_unpadded_dimensions);

        // Padding, transpose, gemm and remove padding
        a_pad.forward(&mut backend);
        b_pad.forward(&mut backend);
        t.forward(&mut backend);
        for _ in 0..iters {
            gemm.forward(&mut backend);
        }
        remove_gemm_padding.forward(&mut backend);
        // Wait for completion
        backend.buffer_to_values(&remove_gemm_padding.output.buffer())
    }

    fn test_gemm9(n: usize, m: usize, k: usize, iters: usize) -> Vec<f32> {
        tracy_client::Client::start();
        let _span = tracy_client::span!("gemm with register blocking general");
        let mut res = Vec::new();
        let (a, b, mut backend) = setup_gemm(n, m, k);
        for i in 0..iters {
            // padding kernels
            // These consts need to be synced with the kernel
            const TSN: usize = 128;
            const TSM: usize = 128;
            const TSK: usize = 16;
            // if B
            let b_p_xl = ((b.dimensions().height - 1) / TSK + 1) * TSK;
            let b_q_xl = ((b.dimensions().width - 1) / TSN + 1) * TSN;
            let b_padded_dimensions = Dimensions4D::new(1, 1, b_p_xl, b_q_xl);
            // if A
            let a_p_xl = ((a.dimensions().height - 1) / TSM + 1) * TSM;
            let a_q_xl = ((a.dimensions().width - 1) / TSK + 1) * TSK;
            let a_padded_dimensions = Dimensions4D::new(1, 1, a_p_xl, a_q_xl);

            let c_unpadded_dimensions =
                Dimensions4D::new(1, 1, a.dimensions().height, b.dimensions().width);

            // Kernels
            let a_pad = add_zero_padding(&a, &mut backend, a_padded_dimensions);
            let b_pad = add_zero_padding(&b, &mut backend, b_padded_dimensions);

            let t = transpose(&b_pad.output, &mut backend);
            let gemm = gemm_register_blocking_general(&a_pad.output, &t.output, &mut backend);

            let remove_gemm_padding =
                remove_zero_padding(&gemm.c, &mut backend, c_unpadded_dimensions);

            // Padding, transpose, gemm and remove padding
            a_pad.forward(&mut backend);
            b_pad.forward(&mut backend);
            t.forward(&mut backend);
            gemm.forward(&mut backend);
            remove_gemm_padding.forward(&mut backend);
            // Wait for completion
            if i == (iters - 1) {
                res = backend.buffer_to_values(&remove_gemm_padding.output.buffer());
            }
        }
        res
    }

    fn test_backend_gemm_impl(n: usize, m: usize, k: usize, iters: usize) -> Vec<f32> {
        tracy_client::Client::start();
        let _span = tracy_client::span!("backend impl");
        let (a, b, mut backend) = setup_gemm(n, m, k);
        let bmnk = (1, m, n, k);
        let (batch, m, n, _k) = bmnk;
        let c = backend.zeros(Dimensions4D::new(batch, 1, m, n)).unwrap();
        for _ in 0..iters {
            backend.gemm_compute(&a.buffer, &b.buffer, &c, bmnk, false)
        }
        backend.buffer_to_values(&c)
    }

    fn test_backend_gemm_rm_impl(n: usize, m: usize, k: usize, iters: usize) -> Vec<f32> {
        tracy_client::Client::start();
        let _span = tracy_client::span!("backend impl rm");
        let (a, b, mut backend) = setup_gemm(n, m, k);
        let mnk = (n, m, k);
        let (m, n, _k) = mnk;
        let c = backend.zeros(Dimensions4D::new(1, 1, m, n)).unwrap();
        for _ in 0..iters {
            backend.gemm_compute_rm(&b.buffer, &a.buffer, &c, mnk, false)
        }
        backend.buffer_to_values(&c)
    }

    #[test]
    fn test_transpose() {
        // Setup
        let mut backend = WgpuBackend::default();
        let p = 32;
        let q = 16;
        tracy_client::Client::start();
        let b_dimensions = Dimensions4D::new(1, p, q, 1);
        // the size should be the same as the dimensions size
        let mut b_values = linspace(0.0, 1.0, p * q / 2);
        b_values.append(&mut vec![1.0; p * q / 2]);
        let b_transposed_dims = Dimensions4D::new(1, b_dimensions.height, b_dimensions.width, 1);
        let b_transposed = transpose_cpu(&b_values, b_transposed_dims);
        let b = Tensor::new(
            TensorInit::FromValues(b_values.clone()),
            b_dimensions,
            &mut backend,
        );

        let _span = tracy_client::span!("transpose");
        let t = transpose(&b, &mut backend);
        // let t_t = transpose(&t.output, &mut backend);

        // Execute
        t.forward(&mut backend);
        // t_t.forward(&mut backend);
        // Wait for completion
        let result = backend.buffer_to_values(&t.output.buffer());
        // Assert
        assert!(compare_outputs(&result, &b_transposed));
    }

    #[test]
    fn test_padding() {
        // Setup
        let mut backend = WgpuBackend::default();
        let p = 15; // height
        let q = 11; // width
        tracy_client::Client::start();
        let b_dimensions = Dimensions4D::new(1, q, p, 1);
        // the size should be the same as the dimensions size
        let b_values = linspace(0.0, 1.0, p * q);
        let b = Tensor::new(
            TensorInit::FromValues(b_values.clone()),
            b_dimensions,
            &mut backend,
        );
        const TSN: usize = 20;
        const TSM: usize = 22;
        const TSK: usize = 16;
        // if B
        let q_xl = ((q - 1) / TSK + 1) * TSK;
        let p_xl = ((p - 1) / TSN + 1) * TSN;
        // if A
        // let p_xl = ((p - 1) / TSK + 1) * TSK;
        // let q_xl = ((q - 1) / TSM + 1) * TSM;
        let new_dimensions = Dimensions4D::new(1, q_xl, p_xl, 1);

        let b_padded = zeropad_cpu(&b_values, b_dimensions, new_dimensions);

        let _span = tracy_client::span!("add padding");
        let zero_pad = add_zero_padding(&b, &mut backend, new_dimensions);
        // let unpadded = remove_zero_padding(&b, &mut backend, b_dimensions);

        // Execute
        zero_pad.forward(&mut backend);
        // unpadded.forward(&mut backend);
        // Wait for completion
        let padded_out = backend.buffer_to_values(&zero_pad.output.buffer());
        // Assert
        assert!(compare_outputs(&b_padded, &padded_out));
        print_tensor_to_file(&zero_pad.output, &mut backend)
    }

    #[test]
    fn add_and_remove_zero_padding() {
        let mut backend = WgpuBackend::default();
        let p = 15; // height
        let q = 11; // width
        tracy_client::Client::start();
        let b_dimensions = Dimensions4D::new(1, q, p, 1);
        // the size should be the same as the dimensions size
        let b_values = linspace(0.0, 1.0, p * q);
        let b = Tensor::new(
            TensorInit::FromValues(b_values.clone()),
            b_dimensions,
            &mut backend,
        );
        const TSN: usize = 32;
        const TSK: usize = 32;
        // if B
        let p_xl = ((p - 1) / TSN + 1) * TSN;
        let q_xl = ((q - 1) / TSK + 1) * TSK;
        let new_dimensions = Dimensions4D::new(1, q_xl, p_xl, 1);

        let b_padded = zeropad_cpu(&b_values, b_dimensions, new_dimensions);

        let _span = tracy_client::span!("add padding");
        let zero_pad = add_zero_padding(&b, &mut backend, new_dimensions);
        let unpadded = remove_zero_padding(&zero_pad.output, &mut backend, b_dimensions);

        // Execute
        zero_pad.forward(&mut backend);
        unpadded.forward(&mut backend);
        // Wait for completion
        let padded_out = backend.buffer_to_values(&zero_pad.output.buffer());
        let unpadded_out = backend.buffer_to_values(&unpadded.output.buffer());
        let b_values = backend.buffer_to_values(&b.buffer());
        // Assert
        assert!(compare_outputs(&b_padded, &padded_out));
        assert!(compare_outputs(&b_values, &unpadded_out));
    }

    #[test]
    fn benchmark_gemms() {
        let n = 16 * 128;
        let m = 16 * 128;
        let k = 32 * 64;
        let iters = 100;

        // test_naive_gemm(n, m, k, iters);
        test_gemm2(n, m, k, iters);
        test_gemm3(n, m, k, iters);
        test_gemm4(n, m, k, iters);
        test_gemm5(n, m, k, iters);
        test_gemm6(n, m, k, iters);
        test_gemm7(n, m, k, iters);
        test_gemm8(n, m, k, iters);
        // test_gemm9(n, m, k, iters);
        test_backend_gemm_rm_impl(n, m, k, iters);
        test_backend_gemm_impl(n, m, k, iters);
    }

    #[test]
    fn compare_output_with_naive() {
        // 16 * 8 is min for gemm6
        // if n == k the kernels which use transpose work
        let n = 1560;
        let m = 134;
        let k = 126;
        let iters = 1;

        let x1 = test_backend_gemm_impl(n, m, k, iters);
        let x2 = test_backend_gemm_rm_impl(n, m, k, iters);
        assert!(compare_outputs(&x1, &x2))
    }
}
