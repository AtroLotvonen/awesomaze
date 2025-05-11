use std::rc::Rc;

use wgpu::Buffer;

use crate::machine_learning::nn::{
    backend::CmpType,
    backends::wgpu_backend::WgpuBackend,
    dimensions::{Dimensions2D, Dimensions4D},
    tensor_error::Result,
};

use super::{
    kernel::{Kernel, KernelParameter, KernelParameters},
    BinaryOperation,
};

impl WgpuBackend {
    pub(super) fn set_to_zeros_compute(&mut self, buffer: &Buffer, dimensions: Dimensions4D) {
        let q = dimensions.width;
        let p = dimensions.height;
        // Since q and p are rounded up they might handle other channels and batches also
        let ts = 16;
        let q_real = q.div_ceil(ts as usize) * ts as usize;
        let p_real = p.div_ceil(ts as usize) * ts as usize;
        let n_real = dimensions.size().div_ceil(p_real * q_real);
        let local = [ts, ts];
        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::N(n_real.try_into().unwrap()),
            KernelParameter::Length(dimensions.size().try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "zeros", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./zeros.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[buffer];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn set_to_ones_compute(&mut self, buffer: &Buffer, dimensions: Dimensions4D) {
        let q = dimensions.width;
        let p = dimensions.height;
        // Since q and p are rounded up they might handle other channels and batches also
        let ts = 16;
        let q_real = q.div_ceil(ts as usize) * ts as usize;
        let p_real = p.div_ceil(ts as usize) * ts as usize;
        let n_real = dimensions.size().div_ceil(p_real * q_real);
        let local = [ts, ts];
        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::N(n_real.try_into().unwrap()),
            KernelParameter::Length(dimensions.size().try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./ones.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[buffer];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn binary_operation_compute(
        &mut self,
        lhs: &Buffer,
        rhs: &Buffer,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
        operation: BinaryOperation,
    ) -> Result<Rc<wgpu::Buffer>> {
        if lhs_dimensions == rhs_dimensions {
            let result_storage = self.get_buffer_with_dimensions(lhs_dimensions);
            self.binary_operation_matching_rm_compute(
                lhs,
                rhs,
                &result_storage,
                lhs_dimensions,
                operation,
            );
            Ok(result_storage)
        } else {
            // try broadcasting
            let broadcast_dimensions = lhs_dimensions.broadcast_dimensions(rhs_dimensions)?;
            let result_storage = self.get_buffer_with_dimensions(broadcast_dimensions);
            self.binary_operation_broadcast_rm_compute(
                lhs,
                rhs,
                &result_storage,
                (lhs_dimensions, rhs_dimensions, broadcast_dimensions),
                operation,
            );
            Ok(result_storage)
        }
    }

    pub(super) fn binary_operation_matching_rm_compute(
        &mut self,
        lhs: &Buffer,
        rhs: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
        operation: BinaryOperation,
    ) {
        let q = input_dimensions.width;
        let p = input_dimensions.height;
        // Since q and p are rounded up they might handle other channels and batches also
        let ts = 16;
        let q_real = q.div_ceil(ts as usize) * ts as usize;
        let p_real = p.div_ceil(ts as usize) * ts as usize;
        let n_real = input_dimensions.size().div_ceil(p_real * q_real);
        let local = [ts, ts];
        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::N(n_real.try_into().unwrap()),
            KernelParameter::Length(input_dimensions.size().try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);
        let kernel_operation_name = match operation {
            BinaryOperation::Add => "add-rm",
            BinaryOperation::Sub => "sub-rm",
            BinaryOperation::Mul => "mul-rm",
            BinaryOperation::Div => "div-rm",
            BinaryOperation::Pow => "pow-rm",
        };

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", kernel_operation_name, parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = match operation {
                    BinaryOperation::Add => include_str!("./add_rm.wgsl"),
                    BinaryOperation::Sub => include_str!("./sub_rm.wgsl"),
                    BinaryOperation::Mul => include_str!("./mul_rm.wgsl"),
                    BinaryOperation::Div => include_str!("./div_rm.wgsl"),
                    BinaryOperation::Pow => include_str!("./pow_rm.wgsl"),
                };
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[lhs, rhs, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn binary_operation_broadcast_rm_compute(
        &mut self,
        lhs: &Buffer,
        rhs: &Buffer,
        output: &Buffer,
        dimensions: (Dimensions4D, Dimensions4D, Dimensions4D),
        operation: BinaryOperation,
    ) {
        let (lhs_dimensions, rhs_dimensions, broadcast_dimensions) = dimensions;
        let lhs_broadcast = lhs_dimensions.matching_dimensions(broadcast_dimensions);
        let lhs_broadcast = (
            lhs_broadcast.0 as u32,
            lhs_broadcast.1 as u32,
            lhs_broadcast.2 as u32,
            lhs_broadcast.3 as u32,
        );
        let rhs_broadcast = rhs_dimensions.matching_dimensions(broadcast_dimensions);
        let rhs_broadcast = (
            rhs_broadcast.0 as u32,
            rhs_broadcast.1 as u32,
            rhs_broadcast.2 as u32,
            rhs_broadcast.3 as u32,
        );
        let nc = broadcast_dimensions.batch * broadcast_dimensions.depth;
        let h = broadcast_dimensions.height;
        let w = broadcast_dimensions.width;
        let ts = 16;
        let local = [ts, ts];
        let global = [
            w.try_into().unwrap(),
            h.try_into().unwrap(),
            nc.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::N(broadcast_dimensions.batch.try_into().unwrap()),
            KernelParameter::C(broadcast_dimensions.depth.try_into().unwrap()),
            KernelParameter::H(broadcast_dimensions.height.try_into().unwrap()),
            KernelParameter::W(broadcast_dimensions.width.try_into().unwrap()),
            KernelParameter::AN(lhs_dimensions.batch.try_into().unwrap()),
            KernelParameter::AC(lhs_dimensions.depth.try_into().unwrap()),
            KernelParameter::AH(lhs_dimensions.height.try_into().unwrap()),
            KernelParameter::AW(lhs_dimensions.width.try_into().unwrap()),
            KernelParameter::BN(rhs_dimensions.batch.try_into().unwrap()),
            KernelParameter::BC(rhs_dimensions.depth.try_into().unwrap()),
            KernelParameter::BH(rhs_dimensions.height.try_into().unwrap()),
            KernelParameter::BW(rhs_dimensions.width.try_into().unwrap()),
            KernelParameter::BroadAN(lhs_broadcast.0),
            KernelParameter::BroadAC(lhs_broadcast.1),
            KernelParameter::BroadAH(lhs_broadcast.2),
            KernelParameter::BroadAW(lhs_broadcast.3),
            KernelParameter::BroadBN(rhs_broadcast.0),
            KernelParameter::BroadBC(rhs_broadcast.1),
            KernelParameter::BroadBH(rhs_broadcast.2),
            KernelParameter::BroadBW(rhs_broadcast.3),
        ];
        let parameters = KernelParameters(&parameter_list);

        let kernel_operation_name = match operation {
            BinaryOperation::Add => "add",
            BinaryOperation::Sub => "sub",
            BinaryOperation::Mul => "mul",
            BinaryOperation::Div => "div",
            BinaryOperation::Pow => "pow",
        };

        // reuse kernels with the same name and parameters
        let kernel_name = format!(
            "{}-{}-{}",
            kernel_operation_name, "-broadcast-rm", parameters
        );
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = match operation {
                    BinaryOperation::Add => include_str!("./add_broadcast_rm.wgsl"),
                    BinaryOperation::Sub => include_str!("./sub_broadcast_rm.wgsl"),
                    BinaryOperation::Mul => include_str!("./mul_broadcast_rm.wgsl"),
                    BinaryOperation::Div => include_str!("./div_broadcast_rm.wgsl"),
                    BinaryOperation::Pow => include_str!("./pow_broadcast_rm.wgsl"),
                };
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[lhs, rhs, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn binary_comparison_compute(
        &mut self,
        lhs: &Buffer,
        rhs: &Buffer,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
        comparison_type: CmpType,
    ) -> Result<Rc<wgpu::Buffer>> {
        let broadcast_dimensions = lhs_dimensions.broadcast_dimensions(rhs_dimensions)?;
        let result_storage = self.get_buffer_with_dimensions(broadcast_dimensions);

        let lhs_broadcast = lhs_dimensions.matching_dimensions(broadcast_dimensions);
        let lhs_broadcast = (
            lhs_broadcast.0 as u32,
            lhs_broadcast.1 as u32,
            lhs_broadcast.2 as u32,
            lhs_broadcast.3 as u32,
        );
        let rhs_broadcast = rhs_dimensions.matching_dimensions(broadcast_dimensions);
        let rhs_broadcast = (
            rhs_broadcast.0 as u32,
            rhs_broadcast.1 as u32,
            rhs_broadcast.2 as u32,
            rhs_broadcast.3 as u32,
        );
        let nc = broadcast_dimensions.batch * broadcast_dimensions.depth;
        let h = broadcast_dimensions.height;
        let w = broadcast_dimensions.width;
        let ts = 16;
        let local = [ts, ts];
        let global = [
            w.try_into().unwrap(),
            h.try_into().unwrap(),
            nc.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::N(broadcast_dimensions.batch.try_into().unwrap()),
            KernelParameter::C(broadcast_dimensions.depth.try_into().unwrap()),
            KernelParameter::H(broadcast_dimensions.height.try_into().unwrap()),
            KernelParameter::W(broadcast_dimensions.width.try_into().unwrap()),
            KernelParameter::AN(lhs_dimensions.batch.try_into().unwrap()),
            KernelParameter::AC(lhs_dimensions.depth.try_into().unwrap()),
            KernelParameter::AH(lhs_dimensions.height.try_into().unwrap()),
            KernelParameter::AW(lhs_dimensions.width.try_into().unwrap()),
            KernelParameter::BN(rhs_dimensions.batch.try_into().unwrap()),
            KernelParameter::BC(rhs_dimensions.depth.try_into().unwrap()),
            KernelParameter::BH(rhs_dimensions.height.try_into().unwrap()),
            KernelParameter::BW(rhs_dimensions.width.try_into().unwrap()),
            KernelParameter::BroadAN(lhs_broadcast.0),
            KernelParameter::BroadAC(lhs_broadcast.1),
            KernelParameter::BroadAH(lhs_broadcast.2),
            KernelParameter::BroadAW(lhs_broadcast.3),
            KernelParameter::BroadBN(rhs_broadcast.0),
            KernelParameter::BroadBC(rhs_broadcast.1),
            KernelParameter::BroadBH(rhs_broadcast.2),
            KernelParameter::BroadBW(rhs_broadcast.3),
        ];
        let parameters = KernelParameters(&parameter_list);

        let kernel_operation_name = match comparison_type {
            CmpType::Equal => "equal",
            CmpType::NotEqual => "not-equal",
            CmpType::GreaterThan => "greater-than",
            CmpType::GreaterOrEqualThan => "greater-or-equal",
            CmpType::LesserThan => "lesser-than",
            CmpType::LesserOrEqualThan => "lesser-or-equal",
        };

        // reuse kernels with the same name and parameters
        let kernel_name = format!(
            "{}-{}-{}",
            kernel_operation_name, "-broadcast-rm", parameters
        );
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = match comparison_type {
                    CmpType::Equal => include_str!("./broadcast_equal.wgsl"),
                    CmpType::NotEqual => include_str!("./broadcast_not_equal.wgsl"),
                    CmpType::GreaterThan => include_str!("./broadcast_greater_than.wgsl"),
                    CmpType::GreaterOrEqualThan => {
                        include_str!("./broadcast_greater_or_equal.wgsl")
                    }
                    CmpType::LesserThan => include_str!("./broadcast_lesser_than.wgsl"),
                    CmpType::LesserOrEqualThan => include_str!("./broadcast_lesser_or_equal.wgsl"),
                };
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[lhs, rhs, &result_storage];
        self.schedule_kernel(&kernel, buffers);
        Ok(result_storage)
    }

    pub(super) fn reduce_one_dim_compute_rm(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        dimensions: Dimensions4D,
        reduced_dimensions: Dimensions4D,
        reduce_dim: usize,
    ) {
        let ts = 16;
        let local = [ts, ts];
        let mut dims: [usize; 4] = reduced_dimensions.into();
        dims.sort();
        let length: usize = dims.iter().product();
        let q_real = dims[3].div_ceil(ts as usize) * ts as usize;
        let p_real = dims[2].div_ceil(ts as usize) * ts as usize;
        let n_real = length.div_ceil(dims[2] * dims[3]);

        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];
        let (step, count) = match reduce_dim {
            0 => {
                let step = dimensions.stride(0).unwrap();
                let count = dimensions.batch;
                (step, count)
            }
            1 => {
                let step = dimensions.stride(1).unwrap();
                let count = dimensions.depth;
                (step, count)
            }
            2 => {
                let step = dimensions.stride(2).unwrap();
                let count = dimensions.height;
                (step, count)
            }
            3 => {
                let step = dimensions.stride(3).unwrap();
                let count = dimensions.width;
                (step, count)
            }
            _ => return,
        };

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::AN(reduced_dimensions.batch.try_into().unwrap()),
            KernelParameter::AC(reduced_dimensions.depth.try_into().unwrap()),
            KernelParameter::AH(reduced_dimensions.height.try_into().unwrap()),
            KernelParameter::AW(reduced_dimensions.width.try_into().unwrap()),
            KernelParameter::BN(dimensions.batch.try_into().unwrap()),
            KernelParameter::BC(dimensions.depth.try_into().unwrap()),
            KernelParameter::BH(dimensions.height.try_into().unwrap()),
            KernelParameter::BW(dimensions.width.try_into().unwrap()),
            KernelParameter::Length(length.try_into().unwrap()),
            KernelParameter::Step(step.try_into().unwrap()),
            KernelParameter::Count(count.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "reduce-sum-rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./reduce_sum_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn max_dim_compute_rm(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        dimensions: Dimensions4D,
        reduced_dimensions: Dimensions4D,
        reduce_dim: usize,
    ) {
        let ts = 16;
        let local = [ts, ts];
        let mut dims: [usize; 4] = reduced_dimensions.into();
        dims.sort();
        let length: usize = dims.iter().product();
        let q_real = dims[3].div_ceil(ts as usize) * ts as usize;
        let p_real = dims[2].div_ceil(ts as usize) * ts as usize;
        let n_real = length.div_ceil(dims[2] * dims[3]);

        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];
        let (step, count) = match reduce_dim {
            0 => {
                let step = dimensions.stride(0).unwrap();
                let count = dimensions.batch;
                (step, count)
            }
            1 => {
                let step = dimensions.stride(1).unwrap();
                let count = dimensions.depth;
                (step, count)
            }
            2 => {
                let step = dimensions.stride(2).unwrap();
                let count = dimensions.height;
                (step, count)
            }
            3 => {
                let step = dimensions.stride(3).unwrap();
                let count = dimensions.width;
                (step, count)
            }
            _ => return,
        };

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::AN(reduced_dimensions.batch.try_into().unwrap()),
            KernelParameter::AC(reduced_dimensions.depth.try_into().unwrap()),
            KernelParameter::AH(reduced_dimensions.height.try_into().unwrap()),
            KernelParameter::AW(reduced_dimensions.width.try_into().unwrap()),
            KernelParameter::BN(dimensions.batch.try_into().unwrap()),
            KernelParameter::BC(dimensions.depth.try_into().unwrap()),
            KernelParameter::BH(dimensions.height.try_into().unwrap()),
            KernelParameter::BW(dimensions.width.try_into().unwrap()),
            KernelParameter::Length(length.try_into().unwrap()),
            KernelParameter::Step(step.try_into().unwrap()),
            KernelParameter::Count(count.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "max-dim-rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./max_dim_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn argmax_compute_rm(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        dimensions: Dimensions4D,
        reduced_dimensions: Dimensions4D,
        reduce_dim: usize,
    ) {
        let ts = 16;
        let local = [ts, ts];
        let mut dims: [usize; 4] = reduced_dimensions.into();
        dims.sort();
        let length: usize = dims.iter().product();
        let q_real = dims[3].div_ceil(ts as usize) * ts as usize;
        let p_real = dims[2].div_ceil(ts as usize) * ts as usize;
        let n_real = length.div_ceil(dims[2] * dims[3]);

        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];
        let (step, count) = match reduce_dim {
            0 => {
                let step = dimensions.stride(0).unwrap();
                let count = dimensions.batch;
                (step, count)
            }
            1 => {
                let step = dimensions.stride(1).unwrap();
                let count = dimensions.depth;
                (step, count)
            }
            2 => {
                let step = dimensions.stride(2).unwrap();
                let count = dimensions.height;
                (step, count)
            }
            3 => {
                let step = dimensions.stride(3).unwrap();
                let count = dimensions.width;
                (step, count)
            }
            _ => return,
        };

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::AN(reduced_dimensions.batch.try_into().unwrap()),
            KernelParameter::AC(reduced_dimensions.depth.try_into().unwrap()),
            KernelParameter::AH(reduced_dimensions.height.try_into().unwrap()),
            KernelParameter::AW(reduced_dimensions.width.try_into().unwrap()),
            KernelParameter::BN(dimensions.batch.try_into().unwrap()),
            KernelParameter::BC(dimensions.depth.try_into().unwrap()),
            KernelParameter::BH(dimensions.height.try_into().unwrap()),
            KernelParameter::BW(dimensions.width.try_into().unwrap()),
            KernelParameter::Length(length.try_into().unwrap()),
            KernelParameter::Step(step.try_into().unwrap()),
            KernelParameter::Count(count.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "arg-max-rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./arg_max_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn gather_compute_rm(
        &mut self,
        input: &Buffer,
        indices: &Buffer,
        output: &Buffer,
        dimensions: Dimensions4D,
        reduced_dimensions: Dimensions4D,
        reduce_dim: usize,
    ) {
        let ts = 16;
        let local = [ts, ts];
        let mut dims: [usize; 4] = reduced_dimensions.into();
        dims.sort();
        let length: usize = dims.iter().product();
        let q_real = dims[3].div_ceil(ts as usize) * ts as usize;
        let p_real = dims[2].div_ceil(ts as usize) * ts as usize;
        let n_real = length.div_ceil(dims[2] * dims[3]);

        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];
        let (step, count) = match reduce_dim {
            0 => {
                let step = dimensions.stride(0).unwrap();
                let count = dimensions.batch;
                (step, count)
            }
            1 => {
                let step = dimensions.stride(1).unwrap();
                let count = dimensions.depth;
                (step, count)
            }
            2 => {
                let step = dimensions.stride(2).unwrap();
                let count = dimensions.height;
                (step, count)
            }
            3 => {
                let step = dimensions.stride(3).unwrap();
                let count = dimensions.width;
                (step, count)
            }
            _ => return,
        };

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::AN(reduced_dimensions.batch.try_into().unwrap()),
            KernelParameter::AC(reduced_dimensions.depth.try_into().unwrap()),
            KernelParameter::AH(reduced_dimensions.height.try_into().unwrap()),
            KernelParameter::AW(reduced_dimensions.width.try_into().unwrap()),
            KernelParameter::BN(dimensions.batch.try_into().unwrap()),
            KernelParameter::BC(dimensions.depth.try_into().unwrap()),
            KernelParameter::BH(dimensions.height.try_into().unwrap()),
            KernelParameter::BW(dimensions.width.try_into().unwrap()),
            KernelParameter::Length(length.try_into().unwrap()),
            KernelParameter::Step(step.try_into().unwrap()),
            KernelParameter::Count(count.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "gather-rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./gather_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, indices, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn scatter_compute_rm(
        &mut self,
        input: &Buffer,
        indices: &Buffer,
        output: &Buffer,
        reduced_dimensions: Dimensions4D,
        dimensions: Dimensions4D,
        reduce_dim: usize,
    ) {
        let ts = 16;
        let local = [ts, ts];
        let mut dims: [usize; 4] = reduced_dimensions.into();
        dims.sort();
        let length: usize = dims.iter().product();
        let q_real = dims[3].div_ceil(ts as usize) * ts as usize;
        let p_real = dims[2].div_ceil(ts as usize) * ts as usize;
        let n_real = length.div_ceil(dims[2] * dims[3]);

        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];
        let (step, count) = match reduce_dim {
            0 => {
                let step = dimensions.stride(0).unwrap();
                let count = dimensions.batch;
                (step, count)
            }
            1 => {
                let step = dimensions.stride(1).unwrap();
                let count = dimensions.depth;
                (step, count)
            }
            2 => {
                let step = dimensions.stride(2).unwrap();
                let count = dimensions.height;
                (step, count)
            }
            3 => {
                let step = dimensions.stride(3).unwrap();
                let count = dimensions.width;
                (step, count)
            }
            _ => return,
        };

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::AN(reduced_dimensions.batch.try_into().unwrap()),
            KernelParameter::AC(reduced_dimensions.depth.try_into().unwrap()),
            KernelParameter::AH(reduced_dimensions.height.try_into().unwrap()),
            KernelParameter::AW(reduced_dimensions.width.try_into().unwrap()),
            KernelParameter::BN(dimensions.batch.try_into().unwrap()),
            KernelParameter::BC(dimensions.depth.try_into().unwrap()),
            KernelParameter::BH(dimensions.height.try_into().unwrap()),
            KernelParameter::BW(dimensions.width.try_into().unwrap()),
            KernelParameter::Length(length.try_into().unwrap()),
            KernelParameter::Step(step.try_into().unwrap()),
            KernelParameter::Count(count.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "scatter-rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./scatter_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, indices, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn add_zero_padding_compute(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
        offset: Dimensions4D,
        new_dimensions: Dimensions4D,
    ) {
        let p = input_dimensions.height;
        let q = input_dimensions.width;
        let c = input_dimensions.depth;
        let n = input_dimensions.batch;
        let p_xl = new_dimensions.height;
        let q_xl = new_dimensions.width;
        let c_xl = new_dimensions.depth;
        let n_xl = new_dimensions.batch;
        let ts = 16;
        let global_x = q_xl.div_ceil(ts as usize) * ts as usize;
        let global_y = p_xl.div_ceil(ts as usize) * ts as usize;
        let global_z = new_dimensions.size().div_ceil(global_x * global_y);
        let local = [ts, ts];
        let global = [
            global_x.try_into().unwrap(),
            global_y.try_into().unwrap(),
            global_z.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::GlobalW(global_x.try_into().unwrap()),
            KernelParameter::GlobalH(global_y.try_into().unwrap()),
            KernelParameter::P(p.try_into().unwrap()),
            KernelParameter::Q(q.try_into().unwrap()),
            KernelParameter::N(n.try_into().unwrap()),
            KernelParameter::C(c.try_into().unwrap()),
            KernelParameter::PXL(p_xl.try_into().unwrap()),
            KernelParameter::QXL(q_xl.try_into().unwrap()),
            KernelParameter::NXL(n_xl.try_into().unwrap()),
            KernelParameter::CXL(c_xl.try_into().unwrap()),
            KernelParameter::StartOffsetX(offset.width.try_into().unwrap()),
            KernelParameter::StartOffsetY(offset.height.try_into().unwrap()),
            KernelParameter::StartOffsetZ(offset.depth.try_into().unwrap()),
            KernelParameter::StartOffsetN(offset.batch.try_into().unwrap()),
            KernelParameter::Length(new_dimensions.size().try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "zero-padding", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./add_zero_padding2.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    #[allow(dead_code)]
    pub(super) fn remove_zero_padding_rm_compute(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
        offset: Dimensions4D,
        old_dimension: Dimensions4D,
    ) {
        let p_xl = input_dimensions.height;
        let q_xl = input_dimensions.width;
        let p = old_dimension.height;
        let q = old_dimension.width;
        let ts = 16;
        let local = [ts, ts];
        let global = [q as u32, p as u32];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p.try_into().unwrap()),
            KernelParameter::Q(q.try_into().unwrap()),
            KernelParameter::PXL(p_xl.try_into().unwrap()),
            KernelParameter::QXL(q_xl.try_into().unwrap()),
            KernelParameter::StartOffsetX(offset.width.try_into().unwrap()),
            KernelParameter::StartOffsetY(offset.height.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "remove-zero-padding-rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./remove_zero_padding_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    #[allow(dead_code)]
    pub(super) fn remove_zero_padding_compute(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
        offset: Dimensions4D,
        old_dimension: Dimensions4D,
    ) {
        let p_xl = input_dimensions.height;
        let q_xl = input_dimensions.width;
        let p = old_dimension.height;
        let q = old_dimension.width;
        let ts = 16;
        let local = [ts, ts];
        let global = [p as u32, q as u32];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p.try_into().unwrap()),
            KernelParameter::Q(q.try_into().unwrap()),
            KernelParameter::PXL(p_xl.try_into().unwrap()),
            KernelParameter::QXL(q_xl.try_into().unwrap()),
            KernelParameter::StartOffsetX(offset.height.try_into().unwrap()),
            KernelParameter::StartOffsetY(offset.width.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "remove-zero-padding", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./remove_zero_padding2.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn slice_rm_compute(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
        offset: Dimensions4D,
        sliced_dimensions: Dimensions4D,
    ) {
        let p_xl = input_dimensions.height;
        let q_xl = input_dimensions.width;
        let c_xl = input_dimensions.depth;
        let n_xl = input_dimensions.batch;
        let p = sliced_dimensions.height;
        let q = sliced_dimensions.width;
        let c = sliced_dimensions.depth;
        let n = sliced_dimensions.batch;
        let ts = 16;
        let global_x = q.div_ceil(ts as usize) * ts as usize;
        let global_y = p.div_ceil(ts as usize) * ts as usize;
        let global_z = sliced_dimensions.size().div_ceil(global_x * global_y);
        let local = [ts, ts];
        let global = [
            global_x.try_into().unwrap(),
            global_y.try_into().unwrap(),
            global_z.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::GlobalW(global_x.try_into().unwrap()),
            KernelParameter::GlobalH(global_y.try_into().unwrap()),
            KernelParameter::P(p.try_into().unwrap()),
            KernelParameter::Q(q.try_into().unwrap()),
            KernelParameter::N(n.try_into().unwrap()),
            KernelParameter::C(c.try_into().unwrap()),
            KernelParameter::PXL(p_xl.try_into().unwrap()),
            KernelParameter::QXL(q_xl.try_into().unwrap()),
            KernelParameter::NXL(n_xl.try_into().unwrap()),
            KernelParameter::CXL(c_xl.try_into().unwrap()),
            KernelParameter::StartOffsetX(offset.width.try_into().unwrap()),
            KernelParameter::StartOffsetY(offset.height.try_into().unwrap()),
            KernelParameter::StartOffsetZ(offset.depth.try_into().unwrap()),
            KernelParameter::StartOffsetN(offset.batch.try_into().unwrap()),
            KernelParameter::Length(sliced_dimensions.size().try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "slice-rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./slice_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn transpose2d_compute(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
    ) {
        let p = input_dimensions.height;
        let q = input_dimensions.width;
        let n = input_dimensions.batch * input_dimensions.depth;
        let ts = 16;
        let local = [ts, ts];
        let global = [
            p.try_into().unwrap(),
            q.try_into().unwrap(),
            n.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p.try_into().unwrap()),
            KernelParameter::Q(q.try_into().unwrap()),
            KernelParameter::N(n.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);
        // Transpose kernel
        let kernel_name = format!("{}-{}", "transpose", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./transpose2d.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn transpose_general_rm_compute(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
        transposed_dimensions: Dimensions4D,
        swap_dimensions: (usize, usize),
    ) {
        let w = input_dimensions.width;
        let h = input_dimensions.height;
        let c = input_dimensions.depth;
        let n = input_dimensions.batch;
        let tw = transposed_dimensions.width;
        let th = transposed_dimensions.height;
        let tc = transposed_dimensions.depth;
        let tn = transposed_dimensions.batch;
        let ts = 16;
        let global_x = w.div_ceil(ts as usize) * ts as usize;
        let global_y = h.div_ceil(ts as usize) * ts as usize;
        let global_z = input_dimensions.size().div_ceil(global_x * global_y);
        let local = [ts, ts];
        let global = [
            global_x.try_into().unwrap(),
            global_y.try_into().unwrap(),
            global_z.try_into().unwrap(),
        ];
        let mut t_dims = [0, 1, 2, 3];
        t_dims.swap(swap_dimensions.0, swap_dimensions.1);

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::GlobalW(global_x.try_into().unwrap()),
            KernelParameter::GlobalH(global_y.try_into().unwrap()),
            KernelParameter::AW(w.try_into().unwrap()),
            KernelParameter::AH(h.try_into().unwrap()),
            KernelParameter::AC(c.try_into().unwrap()),
            KernelParameter::AN(n.try_into().unwrap()),
            KernelParameter::BW(tw.try_into().unwrap()),
            KernelParameter::BH(th.try_into().unwrap()),
            KernelParameter::BC(tc.try_into().unwrap()),
            KernelParameter::BN(tn.try_into().unwrap()),
            KernelParameter::Dim0(t_dims[0].try_into().unwrap()),
            KernelParameter::Dim1(t_dims[1].try_into().unwrap()),
            KernelParameter::Dim2(t_dims[2].try_into().unwrap()),
            KernelParameter::Dim3(t_dims[3].try_into().unwrap()),
            KernelParameter::Length(input_dimensions.size().try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);
        // Transpose kernel
        let kernel_name = format!("{}-{}", "transpose_naive", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./transpose_general_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    #[allow(dead_code)]
    pub(super) fn gemm_naive_rm_compute(
        &mut self,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        mnk: (usize, usize, usize),
    ) {
        let (m, n, k) = mnk;
        let global = [n as u32, m as u32];
        let ts = 16;
        let local = [ts, ts];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(local[0]),
            KernelParameter::WorkgroupSizeY(local[1]),
            KernelParameter::M(m.try_into().unwrap()),
            KernelParameter::N(n.try_into().unwrap()),
            KernelParameter::K(k.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // Get the kernel or create if doesn't exist
        let kernel_name = format!("{}-{}", "gemm_naive_rm", parameters);

        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./gemm_naive_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[a, b, c];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn gemm_register_blocking_general_rm_compute(
        &mut self,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        a_dimensions: Dimensions4D,
        b_dimensions: Dimensions4D,
    ) {
        let m = a_dimensions.height;
        // B is transposed so n is the height (normally it's the width)
        let n = b_dimensions.height;
        let k = a_dimensions.width;
        // the b is transposed so the m and n should match
        assert_eq!(k, b_dimensions.width);

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

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(local[0]),
            KernelParameter::WorkgroupSizeY(local[1]),
            KernelParameter::M(m.try_into().unwrap()),
            KernelParameter::N(n.try_into().unwrap()),
            KernelParameter::K(k.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // Get the kernel or create if doesn't exist
        let kernel_name = format!(
            "{}-{}",
            "gemm_register_blocking_with_arbitary_size_rm", parameters
        );

        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source =
                    include_str!("./gemm_register_blocking_with_arbitary_size_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[a, b, c];
        self.schedule_kernel(&kernel, buffers);
    }

    #[allow(dead_code)]
    pub(super) fn gemm_register_blocking_general_compute(
        &mut self,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        a_dimensions: Dimensions4D,
        b_dimensions: Dimensions4D,
    ) {
        let m = a_dimensions.height;
        // B is transposed so n is the height (normally it's the width)
        let n = b_dimensions.height;
        let k = a_dimensions.width;
        // the b is transposed so the m and n should match
        assert_eq!(k, b_dimensions.width);

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

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(local[0]),
            KernelParameter::WorkgroupSizeY(local[1]),
            KernelParameter::M(m.try_into().unwrap()),
            KernelParameter::N(n.try_into().unwrap()),
            KernelParameter::K(k.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // Get the kernel or create if doesn't exist
        let kernel_name = format!(
            "{}-{}",
            "gemm_register_blocking_with_arbitary_size", parameters
        );

        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source =
                    include_str!("./gemm_register_blocking_with_arbitary_size2.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], 1],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[a, b, c];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn gemm_compute_rm(
        &mut self,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        mnk: (usize, usize, usize),
        is_b_transposed: bool,
    ) {
        let (m, n, k) = mnk;
        const TSN: usize = 128;
        const TSM: usize = 128;
        const TSK: usize = 16;
        let a_dimensions = Dimensions4D::new(1, 1, m, k);
        let b_dimensions = Dimensions4D::new(1, 1, k, n);
        // if B
        let b_p_xl = ((b_dimensions.height - 1) / TSK + 1) * TSK;
        let b_q_xl = ((b_dimensions.width - 1) / TSN + 1) * TSN;
        let b_transposed_dimensions = Dimensions4D::new(1, 1, n, k);
        let b_padded_transposed_dimensions = Dimensions4D::new(1, 1, b_q_xl, b_p_xl);
        // if A
        let a_p_xl = ((a_dimensions.height - 1) / TSM + 1) * TSM;
        let a_q_xl = ((a_dimensions.width - 1) / TSK + 1) * TSK;
        let a_padded_dimensions = Dimensions4D::new(1, 1, a_p_xl, a_q_xl);
        let c_unpadded_dimensions = Dimensions4D::new(1, 1, m, n);

        let padding_start_offset = Dimensions4D::new(0, 0, 0, 0);
        let b_padded_transposed = self.get_buffer_with_dimensions(b_padded_transposed_dimensions);
        {
            if !is_b_transposed {
                let b_transposed_batched_dimensions = Dimensions4D::new(1, 1, n, k);
                let b_transposed = self.get_buffer_with_dimensions(b_transposed_batched_dimensions);
                self.transpose2d_compute(b, &b_transposed, b_transposed_batched_dimensions);
                self.add_zero_padding_compute(
                    &b_transposed,
                    &b_padded_transposed,
                    b_transposed_dimensions,
                    padding_start_offset,
                    b_padded_transposed_dimensions,
                );
            } else {
                self.add_zero_padding_compute(
                    b,
                    &b_padded_transposed,
                    b_transposed_dimensions,
                    padding_start_offset,
                    b_padded_transposed_dimensions,
                );
            }
        }

        let a_padded = self.get_buffer_with_dimensions(a_padded_dimensions);
        self.add_zero_padding_compute(
            a,
            &a_padded,
            a_dimensions,
            padding_start_offset,
            a_padded_dimensions,
        );

        let c_padded_dimensions = Dimensions4D::new(
            1,
            1,
            a_padded_dimensions.height,
            b_padded_transposed_dimensions.height,
        );
        let c_padded = self.get_buffer_with_dimensions(c_padded_dimensions);
        self.gemm_register_blocking_general_rm_compute(
            &a_padded,
            &b_padded_transposed,
            &c_padded,
            a_padded_dimensions,
            b_padded_transposed_dimensions,
        );
        self.slice_rm_compute(
            &c_padded,
            c,
            c_padded_dimensions,
            padding_start_offset,
            c_unpadded_dimensions,
        );
    }

    /// Add zero pads to inputs, transpose b, gemm_with_register_blocking, remove zeropad. Uses
    /// scratch memory for the intermediate results. Uses column-major order. To "use row-major
    /// matrices" switch the a and b and m and n when calling the this compute
    #[allow(dead_code)]
    pub(super) fn gemm_compute(
        &mut self,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        bmnk: (usize, usize, usize, usize),
        is_b_transposed: bool,
    ) {
        // TODO: add support for batched gemm for `b`
        let (_batch_size, m, n, k) = bmnk;
        const TSN: usize = 128;
        const TSM: usize = 128;
        const TSK: usize = 16;
        let a_dimensions = Dimensions4D::new(1, 1, m, k);
        let b_dimensions = Dimensions4D::new(1, 1, k, n);
        // if B
        let b_p_xl = ((b_dimensions.height - 1) / TSK + 1) * TSK;
        let b_q_xl = ((b_dimensions.width - 1) / TSN + 1) * TSN;
        // let b_padded_dimensions = Dimensions4D::new(1, 1, b_p_xl, b_q_xl);
        let b_transposed_dimensions = Dimensions4D::new(1, 1, n, k);
        let b_padded_transposed_dimensions = Dimensions4D::new(1, 1, b_q_xl, b_p_xl);
        // if A
        let a_p_xl = ((a_dimensions.height - 1) / TSM + 1) * TSM;
        let a_q_xl = ((a_dimensions.width - 1) / TSK + 1) * TSK;
        let a_padded_dimensions = Dimensions4D::new(1, 1, a_p_xl, a_q_xl);
        let c_unpadded_dimensions =
            Dimensions4D::new(1, 1, a_dimensions.height, b_dimensions.width);

        let padding_start_offset = Dimensions4D::new(0, 0, 0, 0);
        // Dropping the b_padded scratch back to pool after transposing
        // let b_transposed = self.get_scratch_buffer_with_dimensions(b_padded_dimensions);
        let b_padded_transposed = self.get_buffer_with_dimensions(b_padded_transposed_dimensions);
        {
            let b_transposed_dimensions_rm = Dimensions4D::new(1, 1, k, n);
            let b_padded_transposed_dimensions_rm = Dimensions4D::new(1, 1, b_p_xl, b_q_xl);
            if !is_b_transposed {
                let b_transposed = self.get_buffer_with_dimensions(b_transposed_dimensions);
                self.transpose2d_compute(b, &b_transposed, b_dimensions);
                self.add_zero_padding_compute(
                    &b_transposed,
                    &b_padded_transposed,
                    b_transposed_dimensions_rm,
                    padding_start_offset,
                    b_padded_transposed_dimensions_rm,
                );
                // print_matrix(&self.get_buffer(b), b_dimensions);
                // print_matrix(&self.get_buffer(&b_transposed), b_transposed_dimensions);
            } else {
                self.add_zero_padding_compute(
                    b,
                    &b_padded_transposed,
                    b_transposed_dimensions_rm,
                    padding_start_offset,
                    b_padded_transposed_dimensions_rm,
                );
            }
        }

        let a_padded = self.get_buffer_with_dimensions(a_padded_dimensions);
        let a_dimensions_rm = Dimensions4D::new(1, 1, k, m);
        let a_padded_dimensions_rm = Dimensions4D::new(1, 1, a_q_xl, a_p_xl);
        self.add_zero_padding_compute(
            a,
            &a_padded,
            a_dimensions_rm,
            padding_start_offset,
            a_padded_dimensions_rm,
        );

        let c_padded_dimensions = Dimensions4D::new(
            1,
            1,
            a_padded_dimensions.height,
            b_padded_transposed_dimensions.height,
        );
        let c_padded = self.get_buffer_with_dimensions(c_padded_dimensions);
        self.gemm_register_blocking_general_compute(
            &a_padded,
            &b_padded_transposed,
            &c_padded,
            a_padded_dimensions,
            b_padded_transposed_dimensions,
        );

        self.remove_zero_padding_compute(
            &c_padded,
            c,
            c_padded_dimensions,
            padding_start_offset,
            c_unpadded_dimensions,
        );
    }

    #[allow(dead_code)]
    pub(super) fn add_broadcast_rm_compute(
        &mut self,
        lhs: &Buffer,
        rhs: &Buffer,
        output: &Buffer,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
        broadcast_dimensions: Dimensions4D,
    ) {
        let lhs_broadcast = lhs_dimensions.matching_dimensions(broadcast_dimensions);
        let lhs_broadcast = (
            lhs_broadcast.0 as u32,
            lhs_broadcast.1 as u32,
            lhs_broadcast.2 as u32,
            lhs_broadcast.3 as u32,
        );
        let rhs_broadcast = rhs_dimensions.matching_dimensions(broadcast_dimensions);
        let rhs_broadcast = (
            rhs_broadcast.0 as u32,
            rhs_broadcast.1 as u32,
            rhs_broadcast.2 as u32,
            rhs_broadcast.3 as u32,
        );
        let nc = broadcast_dimensions.batch * broadcast_dimensions.depth;
        let h = broadcast_dimensions.height;
        let w = broadcast_dimensions.width;
        let ts = 16;
        let local = [ts, ts];
        let global = [
            w.try_into().unwrap(),
            h.try_into().unwrap(),
            nc.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::N(broadcast_dimensions.batch.try_into().unwrap()),
            KernelParameter::C(broadcast_dimensions.depth.try_into().unwrap()),
            KernelParameter::H(broadcast_dimensions.height.try_into().unwrap()),
            KernelParameter::W(broadcast_dimensions.width.try_into().unwrap()),
            KernelParameter::AN(lhs_dimensions.batch.try_into().unwrap()),
            KernelParameter::AC(lhs_dimensions.depth.try_into().unwrap()),
            KernelParameter::AH(lhs_dimensions.height.try_into().unwrap()),
            KernelParameter::AW(lhs_dimensions.width.try_into().unwrap()),
            KernelParameter::BN(rhs_dimensions.batch.try_into().unwrap()),
            KernelParameter::BC(rhs_dimensions.depth.try_into().unwrap()),
            KernelParameter::BH(rhs_dimensions.height.try_into().unwrap()),
            KernelParameter::BW(rhs_dimensions.width.try_into().unwrap()),
            KernelParameter::BroadAN(lhs_broadcast.0),
            KernelParameter::BroadAC(lhs_broadcast.1),
            KernelParameter::BroadAH(lhs_broadcast.2),
            KernelParameter::BroadAW(lhs_broadcast.3),
            KernelParameter::BroadBN(rhs_broadcast.0),
            KernelParameter::BroadBC(rhs_broadcast.1),
            KernelParameter::BroadBH(rhs_broadcast.2),
            KernelParameter::BroadBW(rhs_broadcast.3),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "add-broadcast-rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./add_broadcast_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[lhs, rhs, output];
        self.schedule_kernel(&kernel, buffers);
    }

    #[allow(dead_code)]
    pub(super) fn add_rm_compute(
        &mut self,
        lhs: &Buffer,
        rhs: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
    ) {
        let q = input_dimensions.width;
        let p = input_dimensions.height;
        // Since q and p are rounded up they might handle other channels and batches also
        let ts = 16;
        let q_real = q.div_ceil(ts as usize) * ts as usize;
        let p_real = p.div_ceil(ts as usize) * ts as usize;
        let n_real = input_dimensions.size().div_ceil(p_real * q_real);
        let local = [ts, ts];
        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::N(n_real.try_into().unwrap()),
            KernelParameter::Length(input_dimensions.size().try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "add-rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./add_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[lhs, rhs, output];
        self.schedule_kernel(&kernel, buffers);
    }

    #[allow(dead_code)]
    pub(super) fn sub_rm_compute(
        &mut self,
        lhs: &Buffer,
        rhs: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
    ) {
        let q = input_dimensions.width;
        let p = input_dimensions.height;
        // Since q and p are rounded up they might handle other channels and batches also
        let ts = 16;
        let q_real = q.div_ceil(ts as usize) * ts as usize;
        let p_real = p.div_ceil(ts as usize) * ts as usize;
        let n_real = input_dimensions.size().div_ceil(p_real * q_real);
        let local = [ts, ts];
        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::N(n_real.try_into().unwrap()),
            KernelParameter::Length(input_dimensions.size().try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "sub-rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./sub_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[lhs, rhs, output];
        self.schedule_kernel(&kernel, buffers);
    }

    #[allow(dead_code)]
    pub(super) fn mul_rm_compute(
        &mut self,
        lhs: &Buffer,
        rhs: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
    ) {
        let q = input_dimensions.width;
        let p = input_dimensions.height;
        // Since q and p are rounded up they might handle other channels and batches also
        let ts = 16;
        let q_real = q.div_ceil(ts as usize) * ts as usize;
        let p_real = p.div_ceil(ts as usize) * ts as usize;
        let n_real = input_dimensions.size().div_ceil(p_real * q_real);
        let local = [ts, ts];
        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::N(n_real.try_into().unwrap()),
            KernelParameter::Length(input_dimensions.size().try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "mul-rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./mul_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[lhs, rhs, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn max_pool2d_rm_compute(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
        pool_size: Dimensions2D,
    ) {
        let q = input_dimensions.width;
        let p = input_dimensions.height;
        let n = input_dimensions.depth * input_dimensions.batch;
        let ts = 16;
        let local = [ts, ts];
        let global = [
            q.try_into().unwrap(),
            p.try_into().unwrap(),
            n.try_into().unwrap(),
        ];
        let pool_size_x = pool_size.width;
        let pool_size_y = pool_size.height;

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p.try_into().unwrap()),
            KernelParameter::Q(q.try_into().unwrap()),
            KernelParameter::N(n.try_into().unwrap()),
            KernelParameter::PoolSizeX(pool_size_x.try_into().unwrap()),
            KernelParameter::PoolSizeY(pool_size_y.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "max-pool2d-rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./max_pool2d_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn avg_pool2d_rm_compute(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
        pool_size: Dimensions2D,
    ) {
        let q = input_dimensions.width;
        let p = input_dimensions.height;
        let n = input_dimensions.depth * input_dimensions.batch;
        let ts = 16;
        let local = [ts, ts];
        let global = [
            q.try_into().unwrap(),
            p.try_into().unwrap(),
            n.try_into().unwrap(),
        ];
        let pool_size_x = pool_size.width;
        let pool_size_y = pool_size.height;

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p.try_into().unwrap()),
            KernelParameter::Q(q.try_into().unwrap()),
            KernelParameter::N(n.try_into().unwrap()),
            KernelParameter::PoolSizeX(pool_size_x.try_into().unwrap()),
            KernelParameter::PoolSizeY(pool_size_y.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "avg-pool2d-rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./avg_pool2d_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn upsample2d_rm_compute(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        _input_dimensions: Dimensions4D,
        output_dimensions: Dimensions4D,
        upsample_size: Dimensions2D,
    ) {
        let q = output_dimensions.width;
        let p = output_dimensions.height;
        let n = output_dimensions.depth * output_dimensions.batch;
        let ts = 16;
        let local = [ts, ts];
        let global = [
            q.try_into().unwrap(),
            p.try_into().unwrap(),
            n.try_into().unwrap(),
        ];
        let upsample_size_x = upsample_size.width;
        let upsample_size_y = upsample_size.height;

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p.try_into().unwrap()),
            KernelParameter::Q(q.try_into().unwrap()),
            KernelParameter::N(n.try_into().unwrap()),
            KernelParameter::UpsampleSizeX(upsample_size_x.try_into().unwrap()),
            KernelParameter::UpsampleSizeY(upsample_size_y.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "upsample2d-rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./upsample2d_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn relu_compute(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
    ) {
        let q = input_dimensions.width;
        let p = input_dimensions.height;
        // Since q and p are rounded up they might handle other channels and batches also
        let ts = 16;
        let q_real = q.div_ceil(ts as usize) * ts as usize;
        let p_real = p.div_ceil(ts as usize) * ts as usize;
        let n_real = input_dimensions.size().div_ceil(p_real * q_real);
        let local = [ts, ts];
        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::N(n_real.try_into().unwrap()),
            KernelParameter::Length(input_dimensions.size().try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "relu", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./relu.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn leaky_relu_compute(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
        alpha: f32,
    ) {
        let q = input_dimensions.width;
        let p = input_dimensions.height;
        // Since q and p are rounded up they might handle other channels and batches also
        let ts = 16;
        let q_real = q.div_ceil(ts as usize) * ts as usize;
        let p_real = p.div_ceil(ts as usize) * ts as usize;
        let n_real = input_dimensions.size().div_ceil(p_real * q_real);
        let local = [ts, ts];
        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::N(n_real.try_into().unwrap()),
            KernelParameter::Length(input_dimensions.size().try_into().unwrap()),
            KernelParameter::Alpha(alpha),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "leaky_relu", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./leaky_relu.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn sigmoid_compute(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
    ) {
        let q = input_dimensions.width;
        let p = input_dimensions.height;
        // Since q and p are rounded up they might handle other channels and batches also
        let ts = 16;
        let q_real = q.div_ceil(ts as usize) * ts as usize;
        let p_real = p.div_ceil(ts as usize) * ts as usize;
        let n_real = input_dimensions.size().div_ceil(p_real * q_real);
        let local = [ts, ts];
        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::N(n_real.try_into().unwrap()),
            KernelParameter::Length(input_dimensions.size().try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "sigmoid", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./sigmoid.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn exp_compute(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
    ) {
        let q = input_dimensions.width;
        let p = input_dimensions.height;
        // Since q and p are rounded up they might handle other channels and batches also
        let ts = 16;
        let q_real = q.div_ceil(ts as usize) * ts as usize;
        let p_real = p.div_ceil(ts as usize) * ts as usize;
        let n_real = input_dimensions.size().div_ceil(p_real * q_real);
        let local = [ts, ts];
        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::N(n_real.try_into().unwrap()),
            KernelParameter::Length(input_dimensions.size().try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "exp", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./exp.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn ln_compute(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
    ) {
        let q = input_dimensions.width;
        let p = input_dimensions.height;
        // Since q and p are rounded up they might handle other channels and batches also
        let ts = 16;
        let q_real = q.div_ceil(ts as usize) * ts as usize;
        let p_real = p.div_ceil(ts as usize) * ts as usize;
        let n_real = input_dimensions.size().div_ceil(p_real * q_real);
        let local = [ts, ts];
        let global = [
            q_real.try_into().unwrap(),
            p_real.try_into().unwrap(),
            n_real.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::P(p_real.try_into().unwrap()),
            KernelParameter::Q(q_real.try_into().unwrap()),
            KernelParameter::N(n_real.try_into().unwrap()),
            KernelParameter::Length(input_dimensions.size().try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "exp", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./ln.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn im2col_nchw_rm(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
        kernel_dimensions: Dimensions4D,
    ) {
        let w = input_dimensions.width;
        let h = input_dimensions.height;
        let c = input_dimensions.depth;
        let n = input_dimensions.batch;

        let r = kernel_dimensions.height;
        let s = kernel_dimensions.width;
        // let col_height be the global_x/TS, col_count the global_y/TS and batch_count the global_z
        let col_height = c * r * s;
        let cols_w = input_dimensions.width - kernel_dimensions.width + 1;
        let cols_y = input_dimensions.height - kernel_dimensions.height + 1;
        let col_count = cols_w * cols_y;
        let ts = 16;

        let local = [ts, ts];
        let im2col_length = col_height * col_count * n;

        let global_x = col_height.div_ceil(ts as usize) * ts as usize;
        let global_y = col_count.div_ceil(ts as usize) * ts as usize;
        let global_z = (im2col_length).div_ceil(global_x * global_y);

        let global = [
            global_x.try_into().unwrap(),
            global_y.try_into().unwrap(),
            global_z.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::GlobalW(global_x.try_into().unwrap()),
            KernelParameter::GlobalH(global_y.try_into().unwrap()),
            KernelParameter::W(w.try_into().unwrap()),
            KernelParameter::H(h.try_into().unwrap()),
            KernelParameter::C(c.try_into().unwrap()),
            KernelParameter::N(n.try_into().unwrap()),
            KernelParameter::S(s.try_into().unwrap()),
            KernelParameter::R(r.try_into().unwrap()),
            KernelParameter::Length(im2col_length.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "im2col_nchw_rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./im2col_nchw_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    pub(super) fn im2col_transpose_nchw_rm(
        &mut self,
        input: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
        kernel_dimensions: Dimensions4D,
    ) {
        let w = input_dimensions.width;
        let h = input_dimensions.height;
        let c = input_dimensions.depth;
        let n = input_dimensions.batch;

        let r = kernel_dimensions.height;
        let s = kernel_dimensions.width;
        // let col_height be the global_x/TS, col_count the global_y/TS and batch_count the global_z
        let col_height = c * r * s;
        let cols_w = input_dimensions.width - kernel_dimensions.width + 1;
        let cols_y = input_dimensions.height - kernel_dimensions.height + 1;
        let col_count = cols_w * cols_y;
        let ts = 16;

        let local = [ts, ts];
        let im2col_length = col_height * col_count * n;

        let global_x = col_height.div_ceil(ts as usize) * ts as usize;
        let global_y = col_count.div_ceil(ts as usize) * ts as usize;
        let global_z = (im2col_length).div_ceil(global_x * global_y);

        let global = [
            global_x.try_into().unwrap(),
            global_y.try_into().unwrap(),
            global_z.try_into().unwrap(),
        ];

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(ts),
            KernelParameter::WorkgroupSizeY(ts),
            KernelParameter::GlobalW(global_x.try_into().unwrap()),
            KernelParameter::GlobalH(global_y.try_into().unwrap()),
            KernelParameter::W(w.try_into().unwrap()),
            KernelParameter::H(h.try_into().unwrap()),
            KernelParameter::C(c.try_into().unwrap()),
            KernelParameter::N(n.try_into().unwrap()),
            KernelParameter::S(s.try_into().unwrap()),
            KernelParameter::R(r.try_into().unwrap()),
            KernelParameter::Length(im2col_length.try_into().unwrap()),
        ];
        let parameters = KernelParameters(&parameter_list);

        // reuse kernels with the same name and parameters
        let kernel_name = format!("{}-{}", "im2col_transpose_nchw_rm", parameters);
        let kernel = match self.get_kernel(&kernel_name) {
            Some(kernel) => kernel,
            None => {
                let kernel_source = include_str!("./im2col_transpose_nchw_rm.wgsl");
                let kernel_source_with_params = Kernel::set_parameters(kernel_source, &parameters);
                self.create_kernel(
                    &kernel_source_with_params,
                    [local[0], local[1], 1, global[0], global[1], global[2]],
                    &kernel_name,
                )
                .unwrap()
            }
        };

        let buffers = &[input, output];
        self.schedule_kernel(&kernel, buffers);
    }

    #[allow(dead_code)]
    pub(super) fn im2col_conv_zero_padded(
        &mut self,
        input: &Buffer,
        kernels: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
        kernel_dimensions: Dimensions4D,
    ) {
        // First add zero padding to the input
        let padding_y = kernel_dimensions.height / 2;
        let padding_x = kernel_dimensions.width / 2;
        let padded_dimensions = Dimensions4D::new(
            input_dimensions.batch,
            input_dimensions.depth,
            input_dimensions.height + 2 * padding_y,
            input_dimensions.width + 2 * padding_x,
        );
        let col_height =
            kernel_dimensions.depth * kernel_dimensions.height * kernel_dimensions.width;
        let num_cols = input_dimensions.height * input_dimensions.width;
        let im2col_dimensions = Dimensions4D::new(input_dimensions.batch, 1, num_cols, col_height);
        let im2col_output = self.get_buffer_with_dimensions(im2col_dimensions);
        {
            let zero_padded = self.get_buffer_with_dimensions(padded_dimensions);
            let offset = Dimensions4D::new(0, 0, padding_y, padding_x);
            self.add_zero_padding_compute(
                input,
                &zero_padded,
                input_dimensions,
                offset,
                padded_dimensions,
            );
            // then im2col
            self.im2col_nchw_rm(
                &zero_padded,
                &im2col_output,
                padded_dimensions,
                kernel_dimensions,
            );
        }
        // after transforming, gemm
        let n = kernel_dimensions.batch;
        let m = im2col_dimensions.batch * im2col_dimensions.height;
        let k = col_height;
        let mnk = (m, n, k);
        let output_dimensions = Dimensions4D::new(
            input_dimensions.batch,
            kernel_dimensions.batch,
            input_dimensions.height,
            input_dimensions.width,
        );
        let gemm_nhwc_result = self.get_buffer_with_dimensions(output_dimensions);
        // let kernels_cm = self.get_scratch_buffer_with_dimensions(kernel_dimensions);
        // self.transpose_compute(kernels, &kernels_cm, Dimensions4D::new(1, 1, k, m));
        // self.gemm_compute(&kernels_cm, &im2col_output, &gemm_nhwc_result, bmnk, false);
        self.gemm_compute_rm(&im2col_output, kernels, &gemm_nhwc_result, mnk, true);
        // self.gemm_naive_rm_compute(&im2col_output, kernels, &gemm_nhwc_result, bmnk);
        // transpose NHWC back to NCHW
        let transpose_dimensions = Dimensions4D::new(
            input_dimensions.batch,
            1,
            kernel_dimensions.batch,
            input_dimensions.height * input_dimensions.width,
        );
        self.transpose2d_compute(&gemm_nhwc_result, output, transpose_dimensions);
    }

    pub(super) fn conv_transpose2d_compute_rm(
        &mut self,
        input: &Buffer,
        kernels: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
        kernel_dimensions: Dimensions4D,
        output_dimensions: Dimensions4D,
    ) {
        // nchw -> cnhw
        // (in_channels, out_channels, kH, kW) -> (out_channels, in_channels, kH, kW )
        let kernels_chwn = self.get_buffer_with_dimensions(kernel_dimensions);
        let kernel_transpose_dimensions = Dimensions4D::new(
            kernel_dimensions.depth,
            kernel_dimensions.batch,
            kernel_dimensions.height,
            kernel_dimensions.width,
        );
        self.transpose_general_rm_compute(
            kernels,
            &kernels_chwn,
            kernel_dimensions,
            kernel_transpose_dimensions,
            (0, 1),
        );
        // print_matrix(
        //     &self
        //         .get_values(&kernels_chwn, kernel_transpose_dimensions)
        //         .unwrap(),
        //     kernel_transpose_dimensions,
        // );

        // TODO: Zero pad the input here for the im2col transpose because the kernel is transposed,
        // this might include padding inside the input also if there is strides
        let offset = Dimensions4D::new(
            0,
            0,
            kernel_dimensions.height - 1,
            kernel_dimensions.width - 1,
        );
        let input_padded_height = input_dimensions.height + 2 * (kernel_dimensions.height - 1);
        let input_padded_width = input_dimensions.width + 2 * (kernel_dimensions.width - 1);
        let input_padded_dimensions = Dimensions4D::new(
            input_dimensions.batch,
            input_dimensions.depth,
            input_padded_height,
            input_padded_width,
        );
        let input_padded = self.get_buffer_with_dimensions(input_padded_dimensions);
        self.add_zero_padding_compute(
            input,
            &input_padded,
            input_dimensions,
            offset,
            input_padded_dimensions,
        );
        // print_matrix(
        //     &self
        //         .get_values(&input_padded, input_padded_dimensions)
        //         .unwrap(),
        //     input_padded_dimensions,
        // );
        let col_height =
            kernel_dimensions.batch * kernel_dimensions.height * kernel_dimensions.width;
        // These should match with the padded dimensions just before
        let num_cols = output_dimensions.height * output_dimensions.width;
        let im2col_dimensions = Dimensions4D::new(input_dimensions.batch, 1, num_cols, col_height);
        let im2col_output = self.get_buffer_with_dimensions(im2col_dimensions);
        self.im2col_transpose_nchw_rm(
            &input_padded,
            &im2col_output,
            input_padded_dimensions,
            kernel_dimensions,
        );
        // print_matrix(
        //     &self.get_values(&im2col_output, im2col_dimensions).unwrap(),
        //     im2col_dimensions,
        // );
        // after transforming, gemm
        let n = kernel_transpose_dimensions.batch; // out_channels or the n after transpose
        let m = im2col_dimensions.batch * im2col_dimensions.height;
        let k = kernel_transpose_dimensions.depth
            * kernel_transpose_dimensions.height
            * kernel_transpose_dimensions.width;
        let mnk = (m, n, k);
        let gemm_nhwc_result = self.get_buffer_with_dimensions(output_dimensions);
        self.gemm_compute_rm(&im2col_output, &kernels_chwn, &gemm_nhwc_result, mnk, true);
        // print_matrix(
        //     &self
        //         .get_values(&gemm_nhwc_result, output_dimensions)
        //         .unwrap(),
        //     output_dimensions,
        // );
        let transpose_dimensions = Dimensions4D::new(
            input_dimensions.batch,
            1,
            kernel_dimensions.depth,
            output_dimensions.height * output_dimensions.width,
        );
        self.transpose2d_compute(&gemm_nhwc_result, output, transpose_dimensions);
    }

    pub(super) fn im2col_conv(
        &mut self,
        input: &Buffer,
        kernels: &Buffer,
        output: &Buffer,
        input_dimensions: Dimensions4D,
        kernel_dimensions: Dimensions4D,
    ) {
        // First add zero padding to the input
        // let padding_y = kernel_dimensions.height / 2;
        // let padding_x = kernel_dimensions.width / 2;
        // let padded_dimensions = Dimensions4D::new(
        //     input_dimensions.batch,
        //     input_dimensions.depth,
        //     input_dimensions.height + 2 * padding_y,
        //     input_dimensions.width + 2 * padding_x,
        // );
        let output_height = input_dimensions.height - kernel_dimensions.height + 1;
        let output_width = input_dimensions.width - kernel_dimensions.width + 1;
        let unpadded_dimensions = Dimensions4D::new(
            input_dimensions.batch,
            input_dimensions.depth,
            output_height,
            output_width,
        );
        let col_height =
            kernel_dimensions.depth * kernel_dimensions.height * kernel_dimensions.width;
        let num_cols = unpadded_dimensions.height * unpadded_dimensions.width;
        let im2col_dimensions = Dimensions4D::new(input_dimensions.batch, 1, num_cols, col_height);
        let im2col_output = self.get_buffer_with_dimensions(im2col_dimensions);
        {
            // let zero_padded = self.get_buffer_with_dimensions(padded_dimensions);
            // let offset = Dimensions4D::new(0, 0, padding_y, padding_x);
            // self.add_zero_padding_compute(
            //     input,
            //     &zero_padded,
            //     input_dimensions,
            //     offset,
            //     padded_dimensions,
            // );
            // then im2col
            self.im2col_nchw_rm(input, &im2col_output, input_dimensions, kernel_dimensions);
            // print_matrix(
            //     &self.get_values(&im2col_output, im2col_dimensions).unwrap(),
            //     im2col_dimensions,
            // );
        }
        // after transforming, gemm
        let n = kernel_dimensions.batch;
        let m = im2col_dimensions.batch * im2col_dimensions.height;
        let k = col_height;
        let mnk = (m, n, k);
        let output_dimensions = Dimensions4D::new(
            input_dimensions.batch,
            kernel_dimensions.batch,
            unpadded_dimensions.height,
            unpadded_dimensions.width,
        );
        let gemm_nhwc_result = self.get_buffer_with_dimensions(output_dimensions);
        // let kernels_cm = self.get_scratch_buffer_with_dimensions(kernel_dimensions);
        // self.transpose_compute(kernels, &kernels_cm, Dimensions4D::new(1, 1, k, m));
        // self.gemm_compute(&kernels_cm, &im2col_output, &gemm_nhwc_result, bmnk, false);
        self.gemm_compute_rm(&im2col_output, kernels, &gemm_nhwc_result, mnk, true);
        // self.gemm_naive_rm_compute(&im2col_output, kernels, &gemm_nhwc_result, bmnk);
        // transpose NHWC back to NCHW
        let transpose_dimensions = Dimensions4D::new(
            input_dimensions.batch,
            1,
            kernel_dimensions.batch,
            unpadded_dimensions.height * unpadded_dimensions.width,
        );
        self.transpose2d_compute(&gemm_nhwc_result, output, transpose_dimensions);
    }
}

#[cfg(test)]
mod tests {

    use crate::machine_learning::{compare_outputs, linspace, nn::backend::Backend};

    use super::*;

    #[test]
    fn test_im2col() {
        let dimensions = Dimensions4D::new(3, 2, 5, 5);
        // let dimensions = Dimensions4D::new(5, 6, 7, 8);
        let values = linspace(1.0, dimensions.size() as f32, dimensions.size());

        // Cpu
        // Wgpu
        let mut backend = WgpuBackend::default();

        let kernel_dimensions = Dimensions4D::new(100, 2, 3, 3);
        let col_height = kernel_dimensions.height * kernel_dimensions.width;
        let padding = kernel_dimensions.height / 2;
        let num_cols = (dimensions.height - 2 * padding) * (dimensions.width - 2 * padding);
        let output_dimensions =
            Dimensions4D::new(dimensions.batch, dimensions.depth * num_cols, 1, col_height);

        let result_storage = backend.zeros(output_dimensions).unwrap();

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        backend.im2col_nchw_rm(&buffer, &result_storage, dimensions, kernel_dimensions);
        let result_gpu = backend
            .get_values(&result_storage, output_dimensions)
            .unwrap();
        // NOTE: this isnt actually the im2col but fold operation result. The im2col should have
        // the channels in the same column?
        #[rustfmt::skip]
        #[allow(clippy::all)]
        let expected = [
            1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 26.0, 27.0, 28.0, 31.0, 32.0, 33.0, 36.0, 37.0, 38.0,
            2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 12.0, 13.0, 14.0, 27.0, 28.0, 29.0, 32.0, 33.0, 34.0, 37.0, 38.0, 39.0,
            3.0, 4.0, 5.0, 8.0, 9.0, 10.0, 13.0, 14.0, 15.0, 28.0, 29.0, 30.0, 33.0, 34.0, 35.0, 38.0, 39.0, 40.0,
            6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 16.0, 17.0, 18.0, 31.0, 32.0, 33.0, 36.0, 37.0, 38.0, 41.0, 42.0, 43.0,
            7.0, 8.0, 9.0, 12.0, 13.0, 14.0, 17.0, 18.0, 19.0, 32.0, 33.0, 34.0, 37.0, 38.0, 39.0, 42.0, 43.0, 44.0,
            8.0, 9.0, 10.0, 13.0, 14.0, 15.0, 18.0, 19.0, 20.0, 33.0, 34.0, 35.0, 38.0, 39.0, 40.0, 43.0, 44.0, 45.0,
            11.0, 12.0, 13.0, 16.0, 17.0, 18.0, 21.0, 22.0, 23.0, 36.0, 37.0, 38.0, 41.0, 42.0, 43.0, 46.0, 47.0, 48.0,
            12.0, 13.0, 14.0, 17.0, 18.0, 19.0, 22.0, 23.0, 24.0, 37.0, 38.0, 39.0, 42.0, 43.0, 44.0, 47.0, 48.0, 49.0,
            13.0, 14.0, 15.0, 18.0, 19.0, 20.0, 23.0, 24.0, 25.0, 38.0, 39.0, 40.0, 43.0, 44.0, 45.0, 48.0, 49.0, 50.0,


            51.0, 52.0, 53.0, 56.0, 57.0, 58.0, 61.0, 62.0, 63.0, 76.0, 77.0, 78.0, 81.0, 82.0, 83.0, 86.0, 87.0, 88.0,
            52.0, 53.0, 54.0, 57.0, 58.0, 59.0, 62.0, 63.0, 64.0, 77.0, 78.0, 79.0, 82.0, 83.0, 84.0, 87.0, 88.0, 89.0,
            53.0, 54.0, 55.0, 58.0, 59.0, 60.0, 63.0, 64.0, 65.0, 78.0, 79.0, 80.0, 83.0, 84.0, 85.0, 88.0, 89.0, 90.0,
            56.0, 57.0, 58.0, 61.0, 62.0, 63.0, 66.0, 67.0, 68.0, 81.0, 82.0, 83.0, 86.0, 87.0, 88.0, 91.0, 92.0, 93.0,
            57.0, 58.0, 59.0, 62.0, 63.0, 64.0, 67.0, 68.0, 69.0, 82.0, 83.0, 84.0, 87.0, 88.0, 89.0, 92.0, 93.0, 94.0,
            58.0, 59.0, 60.0, 63.0, 64.0, 65.0, 68.0, 69.0, 70.0, 83.0, 84.0, 85.0, 88.0, 89.0, 90.0, 93.0, 94.0, 95.0,
            61.0, 62.0, 63.0, 66.0, 67.0, 68.0, 71.0, 72.0, 73.0, 86.0, 87.0, 88.0, 91.0, 92.0, 93.0, 96.0, 97.0, 98.0,
            62.0, 63.0, 64.0, 67.0, 68.0, 69.0, 72.0, 73.0, 74.0, 87.0, 88.0, 89.0, 92.0, 93.0, 94.0, 97.0, 98.0, 99.0,
            63.0, 64.0, 65.0, 68.0, 69.0, 70.0, 73.0, 74.0, 75.0, 88.0, 89.0, 90.0, 93.0, 94.0, 95.0, 98.0, 99.0, 100.0,


            101.0, 102.0, 103.0, 106.0, 107.0, 108.0, 111.0, 112.0, 113.0, 126.0, 127.0, 128.0, 131.0, 132.0, 133.0, 136.0, 137.0, 138.0,
            102.0, 103.0, 104.0, 107.0, 108.0, 109.0, 112.0, 113.0, 114.0, 127.0, 128.0, 129.0, 132.0, 133.0, 134.0, 137.0, 138.0, 139.0,
            103.0, 104.0, 105.0, 108.0, 109.0, 110.0, 113.0, 114.0, 115.0, 128.0, 129.0, 130.0, 133.0, 134.0, 135.0, 138.0, 139.0, 140.0,
            106.0, 107.0, 108.0, 111.0, 112.0, 113.0, 116.0, 117.0, 118.0, 131.0, 132.0, 133.0, 136.0, 137.0, 138.0, 141.0, 142.0, 143.0,
            107.0, 108.0, 109.0, 112.0, 113.0, 114.0, 117.0, 118.0, 119.0, 132.0, 133.0, 134.0, 137.0, 138.0, 139.0, 142.0, 143.0, 144.0,
            108.0, 109.0, 110.0, 113.0, 114.0, 115.0, 118.0, 119.0, 120.0, 133.0, 134.0, 135.0, 138.0, 139.0, 140.0, 143.0, 144.0, 145.0,
            111.0, 112.0, 113.0, 116.0, 117.0, 118.0, 121.0, 122.0, 123.0, 136.0, 137.0, 138.0, 141.0, 142.0, 143.0, 146.0, 147.0, 148.0,
            112.0, 113.0, 114.0, 117.0, 118.0, 119.0, 122.0, 123.0, 124.0, 137.0, 138.0, 139.0, 142.0, 143.0, 144.0, 147.0, 148.0, 149.0,
            113.0, 114.0, 115.0, 118.0, 119.0, 120.0, 123.0, 124.0, 125.0, 138.0, 139.0, 140.0, 143.0, 144.0, 145.0, 148.0, 149.0, 150.0,
        ];
        // TODO: the expected value is from fold operation so replace with im2col and enable assert
        assert!(compare_outputs(&result_gpu, &expected))
    }
}
