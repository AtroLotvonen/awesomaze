pub mod compute;
pub mod kernel;
pub mod scratch_memory;

use std::borrow::Cow;
use std::collections::HashMap;
use std::rc::Rc;

use futures::executor::block_on;
use itertools::Itertools;
use wgpu::{util::DeviceExt, Buffer, CommandEncoder, Device, Queue};
use wgpu::{BindGroup, ComputePipeline};

use crate::machine_learning::nn::backend::CmpType;
use crate::machine_learning::nn::dimensions::{Dimensions2D, Dimensions4D};
use crate::machine_learning::nn::tensor_error::TensorError;
use crate::machine_learning::nn::{backend::Backend, tensor_error::Result};

use self::kernel::{Kernel, Workgroup};
use self::scratch_memory::WgpuScratchStorage;

#[derive(Debug)]
pub struct WgpuBackend {
    device: Device,
    queue: Queue,
    command_encoder: CommandEncoder,
    current_tasks: usize,
    max_tasks: usize,
    kernel_library: HashMap<String, Rc<Kernel>>,
    scratch_buffer_storage: WgpuScratchStorage,
}

enum BinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl WgpuBackend {
    pub fn new() -> Self {
        // get instance and adapter for connection to gpu
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Failed to find an appropriate adapter");

        // Use the actual limits of the device
        let limits = adapter.limits();
        // get logical device
        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: limits, //wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                // memory_hints: MemoryHints::default(),
            },
            None,
        ))
        .unwrap();

        // generate command encoder ready, this might be unnececssary actually...
        let command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });
        Self {
            // layers: Vec::new(),
            device,
            queue,
            command_encoder,
            current_tasks: 0,
            max_tasks: 12,
            kernel_library: HashMap::new(),
            scratch_buffer_storage: WgpuScratchStorage::default(),
        }
    }

    pub fn buffer_to_values(&mut self, buffer: &Buffer, data_bytes: u64) -> Vec<f32> {
        // let data_bytes = buffer.size() as usize;
        let read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReadBuffer"),
            size: data_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.command_encoder
            .copy_buffer_to_buffer(buffer, 0, &read_buffer, 0, data_bytes);
        self.submit();
        // self.queue.submit(Some(self.command_encoder.finish()));
        // Wait for all operations to complete
        self.device.poll(wgpu::Maintain::Wait);

        block_on(Self::read_async(&self.device, &read_buffer))
    }

    async fn read_async(device: &wgpu::Device, read_buffer: &wgpu::Buffer) -> Vec<f32> {
        let buffer_slice = read_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender
                .send(v)
                .expect("Unable to send buffer slice result to async channel.")
        });

        device.poll(wgpu::Maintain::Wait);

        let result = receiver.receive().await;

        if let Some(Ok(())) = result {
            let data = buffer_slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();

            drop(data);
            read_buffer.unmap();
            result
        } else {
            panic!("Unable to read buffer {:?}", result)
        }
    }

    fn create_staging_buffer(&self, data_bytes: u64) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: data_bytes,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        })
    }

    fn create_buffer(&self, data_bytes: u64) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Empty Buffer"),
            size: data_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn get_buffer_with_dimensions(&mut self, dimensions: Dimensions4D) -> Rc<Buffer> {
        let size_bytes = (dimensions.size() * std::mem::size_of::<f32>()) as u64;
        match self.scratch_buffer_storage.get_scratch_buffer(size_bytes) {
            Some(buffer) => buffer,
            None => {
                let buffer = self.create_buffer(size_bytes);
                self.scratch_buffer_storage.add_buffer(buffer)
            }
        }
    }

    fn get_buffer(&mut self, size_bytes: u64) -> Rc<Buffer> {
        match self.scratch_buffer_storage.get_scratch_buffer(size_bytes) {
            Some(buffer) => buffer,
            None => {
                let buffer = self.create_buffer(size_bytes);
                self.scratch_buffer_storage.add_buffer(buffer)
            }
        }
    }

    pub fn create_uniform_buffer_from_values(&self, values: &[u8]) -> Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: values, // bytemuck::cast_slice(values),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    pub fn get_kernel(&self, kernel_name: &str) -> Option<Rc<Kernel>> {
        self.kernel_library.get(kernel_name).cloned()
    }

    pub fn create_kernel(
        &mut self,
        kernel_source: &str,
        workgroup_size: [u32; 6],
        kernel_name: &str,
    ) -> Result<Rc<Kernel>> {
        let workgroup = kernel::Workgroup {
            local_x: workgroup_size[0],
            local_y: workgroup_size[1],
            local_z: workgroup_size[2],
            global_x: workgroup_size[3],
            global_y: workgroup_size[4],
            global_z: workgroup_size[5],
        };
        let modified_source = Kernel::modify_workgroup_size(kernel_source, workgroup)?;
        // let modified_source = kernel_source;
        // let mut file = std::fs::File::create("modified_gemm.wgsl").unwrap();
        // file.write_all(modified_source.as_bytes()).unwrap();
        let cs_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&modified_source)),
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(kernel_name),
                    layout: None,
                    module: &cs_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
        let kernel = Rc::new(Kernel {
            workgroup,
            pipeline: compute_pipeline,
        });

        self.kernel_library
            .insert(kernel_name.to_string(), kernel.clone());
        Ok(kernel)
    }

    fn register_compute(
        &mut self,
        pipeline: &ComputePipeline,
        bind_group: BindGroup,
        work_group: Workgroup,
    ) {
        let mut compute = self
            .command_encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                // TODO: write timestamps which can be queried for tracy
                timestamp_writes: None,
            });

        compute.set_pipeline(pipeline);
        compute.set_bind_group(0, &bind_group, &[]);
        let x = work_group.global_x.div_ceil(work_group.local_x);
        let y = work_group.global_y.div_ceil(work_group.local_y);
        let z = work_group.global_z.div_ceil(work_group.local_z);
        compute.dispatch_workgroups(x, y, z);

        self.current_tasks += 1;
    }

    pub fn schedule_kernel(&mut self, kernel: &Kernel, buffers: &[&Buffer]) {
        let workgroup = kernel.workgroup;
        let group_layout = kernel.pipeline.get_bind_group_layout(0);

        let entries = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &group_layout,
            entries: &entries,
        });

        self.register_compute(&kernel.pipeline, bind_group, workgroup);

        if self.current_tasks >= self.max_tasks {
            self.submit();
        }
    }

    fn submit(&mut self) {
        let mut new_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        core::mem::swap(&mut new_encoder, &mut self.command_encoder);

        self.queue.submit(Some(new_encoder.finish()));
        self.current_tasks = 0;
    }
}

impl Default for WgpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for WgpuBackend {
    type Storage = Rc<wgpu::Buffer>;

    fn zeros(&mut self, dimensions: Dimensions4D) -> Result<Self::Storage> {
        let buffer = self.get_buffer_with_dimensions(dimensions);
        self.set_to_zeros_compute(&buffer, dimensions);
        Ok(buffer)
    }

    fn ones(&mut self, dimensions: Dimensions4D) -> Result<Self::Storage> {
        let buffer = self.get_buffer_with_dimensions(dimensions);
        self.set_to_ones_compute(&buffer, dimensions);
        Ok(buffer)
    }

    fn buffer_from_values(
        &mut self,
        dimensions: Dimensions4D,
        values: Vec<f32>,
    ) -> Result<Self::Storage> {
        let data_bytes = (dimensions.size() * std::mem::size_of::<f32>()) as u64;
        // Write first to a staging buffer
        let staging_buffer = self.create_staging_buffer(data_bytes);
        {
            let mut staging_data = staging_buffer.slice(..).get_mapped_range_mut();
            staging_data.copy_from_slice(bytemuck::cast_slice(&values));
        }
        staging_buffer.unmap();
        self.queue.submit([]);
        // After writing copy to encoded buffer from scracth memory bank so no race conditions
        // happen between compute and copying host memory
        let buffer = self.get_buffer_with_dimensions(dimensions);
        self.command_encoder
            .copy_buffer_to_buffer(&staging_buffer, 0, &buffer, 0, data_bytes);
        Ok(buffer)
    }

    fn copy(&mut self, buffer: &Self::Storage) -> Result<Self::Storage> {
        // FIXME: all of the buffer might not be in use so specfiy bytes to copy also
        let data_bytes = buffer.size() as usize;
        let copy = self.get_buffer(buffer.size());
        self.command_encoder
            .copy_buffer_to_buffer(buffer, 0, &copy, 0, data_bytes as u64);
        Ok(copy)
    }

    fn matmul(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        mnk: (usize, usize, usize),
        is_rhs_transposed: bool,
    ) -> Result<Self::Storage> {
        // NOTE: swithching lhs with rhs and m with n computes the row-major matrices correctly
        // with column-major kernels
        // because the kernels are implemented column-major order switch m with n and lhs with rhs
        // let (b, n, m, k) = bmnk;
        // let result_dimensions = Dimensions4D::new(b, 1, m, n);
        // let result_storage = self.get_buffer_with_dimensions(result_dimensions)?;
        // // let bmnk = (b, m, n, k);
        // let bmnk = (1, m * b, n, k);
        // self.gemm_compute(rhs, lhs, &result_storage, bmnk, false);
        // Ok(result_storage)
        let (m, n, _k) = mnk;
        // let bmnk = (1, m, b * n, k);
        let result_dimensions = Dimensions4D::new(1, 1, m, n);
        let result_storage = self.get_buffer_with_dimensions(result_dimensions);
        self.gemm_compute_rm(lhs, rhs, &result_storage, mnk, is_rhs_transposed);
        Ok(result_storage)
    }

    fn pow(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        self.binary_operation_compute(
            lhs,
            rhs,
            lhs_dimensions,
            rhs_dimensions,
            BinaryOperation::Pow,
        )
    }

    fn mul(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        self.binary_operation_compute(
            lhs,
            rhs,
            lhs_dimensions,
            rhs_dimensions,
            BinaryOperation::Mul,
        )
    }

    fn div(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        self.binary_operation_compute(
            lhs,
            rhs,
            lhs_dimensions,
            rhs_dimensions,
            BinaryOperation::Div,
        )
    }

    fn add(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        self.binary_operation_compute(
            lhs,
            rhs,
            lhs_dimensions,
            rhs_dimensions,
            BinaryOperation::Add,
        )
    }

    fn sub(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        self.binary_operation_compute(
            lhs,
            rhs,
            lhs_dimensions,
            rhs_dimensions,
            BinaryOperation::Sub,
        )
    }

    fn transpose(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        dim1: usize,
        dim2: usize,
    ) -> Result<Self::Storage> {
        let result_storage = self.get_buffer_with_dimensions(dimensions);
        match (dim1, dim2) {
            (2, 3) | (3, 2) => {
                let cm_dimensions_2d = Dimensions4D::new(
                    dimensions.batch * dimensions.depth,
                    1,
                    dimensions.width,
                    dimensions.height,
                );
                self.transpose2d_compute(buffer, &result_storage, cm_dimensions_2d);
            }
            (_, _) => {
                let transposed_dimensions = dimensions.swap(dim1, dim2)?;
                self.transpose_general_rm_compute(
                    buffer,
                    &result_storage,
                    dimensions,
                    transposed_dimensions,
                    (dim1, dim2),
                );
            }
        };
        Ok(result_storage)
    }

    fn zero_pad(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        offset: Dimensions4D,
        new_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        let result_storage = self.get_buffer_with_dimensions(new_dimensions);
        self.add_zero_padding_compute(buffer, &result_storage, dimensions, offset, new_dimensions);
        Ok(result_storage)
    }

    fn slice(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        offset: Dimensions4D,
        new_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        let result_storage = self.get_buffer_with_dimensions(new_dimensions);
        let dimensions = Dimensions4D::new(
            dimensions.batch,
            dimensions.depth,
            dimensions.height,
            dimensions.width,
        );
        let new_dimensions = Dimensions4D::new(
            new_dimensions.batch,
            new_dimensions.depth,
            new_dimensions.height,
            new_dimensions.width,
        );
        self.slice_rm_compute(buffer, &result_storage, dimensions, offset, new_dimensions);
        Ok(result_storage)
    }

    fn max_pool2d(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        pool_size: Dimensions2D,
    ) -> Result<Self::Storage> {
        let new_dimensions = Dimensions4D::new(
            dimensions.batch,
            dimensions.depth,
            dimensions.height / pool_size.height,
            dimensions.width / pool_size.width,
        );
        let result_storage = self.get_buffer_with_dimensions(new_dimensions);
        self.max_pool2d_rm_compute(buffer, &result_storage, dimensions, pool_size);
        Ok(result_storage)
    }

    fn avg_pool2d(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        pool_size: Dimensions2D,
    ) -> Result<Self::Storage> {
        let new_dimensions = Dimensions4D::new(
            dimensions.batch,
            dimensions.depth,
            dimensions.height / pool_size.height,
            dimensions.width / pool_size.width,
        );
        let result_storage = self.get_buffer_with_dimensions(new_dimensions);
        self.avg_pool2d_rm_compute(buffer, &result_storage, dimensions, pool_size);
        Ok(result_storage)
    }

    fn conv2d(
        &mut self,
        buffer: &Self::Storage,
        kernels: &Self::Storage,
        dimensions: Dimensions4D,
        kernel_dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        let output_height = dimensions.height - kernel_dimensions.height + 1;
        let output_width = dimensions.width - kernel_dimensions.width + 1;
        let output_dimensions = Dimensions4D::new(
            dimensions.batch,
            kernel_dimensions.batch,
            output_height,
            output_width,
        );
        let result_storage = self.get_buffer_with_dimensions(output_dimensions);
        self.im2col_conv(
            buffer,
            kernels,
            &result_storage,
            dimensions,
            kernel_dimensions,
        );
        Ok(result_storage)
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
        let output_dimensions = Dimensions4D::new(
            dimensions.batch,
            kernel_dimensions.depth,
            output_height,
            output_width,
        );
        let result_storage = self.get_buffer_with_dimensions(output_dimensions);
        self.conv_transpose2d_compute_rm(
            buffer,
            kernels,
            &result_storage,
            dimensions,
            kernel_dimensions,
            output_dimensions,
        );
        Ok(result_storage)
    }

    fn relu(&mut self, buffer: &Self::Storage, dimensions: Dimensions4D) -> Result<Self::Storage> {
        let result_storage = self.get_buffer_with_dimensions(dimensions);
        self.relu_compute(buffer, &result_storage, dimensions);
        Ok(result_storage)
    }

    fn leaky_relu(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        alpha: f32,
    ) -> Result<Self::Storage> {
        let result_storage = self.get_buffer_with_dimensions(dimensions);
        self.leaky_relu_compute(buffer, &result_storage, dimensions, alpha);
        Ok(result_storage)
    }

    fn sigmoid(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
    ) -> Result<Self::Storage> {
        let result_storage = self.get_buffer_with_dimensions(dimensions);
        self.sigmoid_compute(buffer, &result_storage, dimensions);
        Ok(result_storage)
    }

    fn exp(&mut self, buffer: &Self::Storage, dimensions: Dimensions4D) -> Result<Self::Storage> {
        let result_storage = self.get_buffer_with_dimensions(dimensions);
        self.exp_compute(buffer, &result_storage, dimensions);
        Ok(result_storage)
    }

    fn ln(&mut self, buffer: &Self::Storage, dimensions: Dimensions4D) -> Result<Self::Storage> {
        let result_storage = self.get_buffer_with_dimensions(dimensions);
        self.ln_compute(buffer, &result_storage, dimensions);
        Ok(result_storage)
    }

    fn get_values(&mut self, buffer: &Self::Storage, dimensions: Dimensions4D) -> Result<Vec<f32>> {
        let size_bytes = (dimensions.size() * std::mem::size_of::<f32>()) as u64;
        Ok(self.buffer_to_values(buffer, size_bytes))
    }

    fn sum_reduce(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        dims: &[usize],
    ) -> Result<Self::Storage> {
        let reduced_dimensions = dimensions.reduce_dims(dims)?;
        let dims_sorted = dims.iter().sorted_by(|l, r| r.cmp(l)).collect::<Vec<_>>();
        let mut sub_reduced = vec![];

        for (i, &dim) in dims_sorted.iter().enumerate() {
            let (sub_input, input_dimensions) = if i == 0 {
                (buffer, dimensions)
            } else {
                let last_outputs: &(Rc<Buffer>, Dimensions4D) = sub_reduced.last().unwrap();
                (&last_outputs.0, last_outputs.1)
            };
            let sub_dimensions = if i == (dims.len() - 1) {
                reduced_dimensions
            } else {
                match dim {
                    3 => Dimensions4D::new(
                        dimensions.batch,
                        dimensions.depth,
                        dimensions.height,
                        reduced_dimensions.width,
                    ),
                    2 => Dimensions4D::new(
                        dimensions.batch,
                        dimensions.depth,
                        reduced_dimensions.height,
                        reduced_dimensions.width,
                    ),
                    1 => Dimensions4D::new(
                        dimensions.batch,
                        reduced_dimensions.depth,
                        reduced_dimensions.height,
                        reduced_dimensions.width,
                    ),
                    0 => Dimensions4D::new(
                        reduced_dimensions.batch,
                        reduced_dimensions.depth,
                        reduced_dimensions.height,
                        reduced_dimensions.width,
                    ),
                    _ => return Err(TensorError::DimensionIndexError { index: *dim }),
                }
            };
            let sub_output = self.get_buffer_with_dimensions(sub_dimensions);
            self.reduce_one_dim_compute_rm(
                sub_input,
                &sub_output,
                input_dimensions,
                sub_dimensions,
                *dim,
            );
            sub_reduced.push((sub_output, sub_dimensions));
        }
        sub_reduced
            .pop()
            .map(|(o, _d)| o)
            .ok_or(TensorError::DimensionIndexError { index: 0 })
    }

    fn comparison(
        &mut self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_dimensions: Dimensions4D,
        rhs_dimensions: Dimensions4D,
        comparison_type: CmpType,
    ) -> Result<Self::Storage> {
        self.binary_comparison_compute(lhs, rhs, lhs_dimensions, rhs_dimensions, comparison_type)
    }

    fn upsample2d(
        &mut self,
        buffer: &Self::Storage,
        input_dimensions: Dimensions4D,
        upsample_size: Dimensions2D,
    ) -> Result<Self::Storage> {
        let new_dimensions = Dimensions4D::new(
            input_dimensions.batch,
            input_dimensions.depth,
            input_dimensions.height * upsample_size.height,
            input_dimensions.width * upsample_size.width,
        );
        let result_storage = self.get_buffer_with_dimensions(new_dimensions);
        self.upsample2d_rm_compute(
            buffer,
            &result_storage,
            input_dimensions,
            new_dimensions,
            upsample_size,
        );
        Ok(result_storage)
    }

    fn max_dim(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        dim: usize,
    ) -> Result<Self::Storage> {
        let reduced_dimensions = dimensions.reduce_dims(&[dim])?;
        let result_storage = self.get_buffer_with_dimensions(reduced_dimensions);
        self.max_dim_compute_rm(buffer, &result_storage, dimensions, reduced_dimensions, dim);
        Ok(result_storage)
    }

    fn arg_max(
        &mut self,
        buffer: &Self::Storage,
        dimensions: Dimensions4D,
        dim: usize,
    ) -> Result<Self::Storage> {
        let reduced_dimensions = dimensions.reduce_dims(&[dim])?;
        let result_storage = self.get_buffer_with_dimensions(reduced_dimensions);
        self.argmax_compute_rm(buffer, &result_storage, dimensions, reduced_dimensions, dim);
        Ok(result_storage)
    }

    fn gather(
        &mut self,
        buffer: &Self::Storage,
        indices: &Self::Storage,
        dimensions: Dimensions4D,
        dim: usize,
    ) -> Result<Self::Storage> {
        let reduced_dimensions = dimensions.reduce_dims(&[dim])?;
        let result_storage = self.get_buffer_with_dimensions(reduced_dimensions);
        self.gather_compute_rm(
            buffer,
            indices,
            &result_storage,
            dimensions,
            reduced_dimensions,
            dim,
        );
        Ok(result_storage)
    }

    fn scatter(
        &mut self,
        buffer: &Self::Storage,
        indices: &Self::Storage,
        dimensions: Dimensions4D,
        out_dimensions: Dimensions4D,
        dim: usize,
    ) -> Result<Self::Storage> {
        let result_storage = self.get_buffer_with_dimensions(out_dimensions);
        self.scatter_compute_rm(
            buffer,
            indices,
            &result_storage,
            dimensions,
            out_dimensions,
            dim,
        );
        Ok(result_storage)
    }
}

#[cfg(test)]
pub mod tests {

    use crate::machine_learning::{compare_outputs, nn::linspace};

    use super::*;

    fn setup_matrices(bmnk: (usize, usize, usize, usize)) -> (Vec<f32>, Vec<f32>) {
        let (b, m, n, k) = bmnk;
        let lhs = linspace(0.0, 1.0, k * m);
        let rhs = linspace(0.0, 1.0, b * k * n);
        (lhs, rhs)
    }

    #[test]
    fn test_gpu_backend_matmul() {
        let mut backend = WgpuBackend::default();

        let (b, m, n, k) = (1, 3, 4, 2);
        let a_dimensions = Dimensions4D::new(1, 1, m, k);
        let b_dimensions = Dimensions4D::new(b, 1, k, n);

        let (a_values, b_values) = setup_matrices((b, m, n, k));
        let lhs = backend.buffer_from_values(a_dimensions, a_values).unwrap();
        let rhs = backend.buffer_from_values(b_dimensions, b_values).unwrap();

        // Shape should be (1, 1, 3, 4)
        #[rustfmt::skip]
        let expected_values = [
            0.11428571, 0.14285714, 0.17142857, 0.2,
            0.34285714, 0.485_714_3, 0.62857143, 0.771_428_6,
            0.571_428_6, 0.82857143, 1.085_714_3, 1.342_857_1,
        ];

        let gpu_result = backend.matmul(&lhs, &rhs, (m, n, k), false).unwrap();
        let gpu_result = backend
            .get_values(&gpu_result, Dimensions4D::new(1, 1, m, n))
            .unwrap();

        assert!(compare_outputs(&gpu_result, &expected_values));
    }

    #[test]
    fn transposing_width_and_height() {
        let mut backend = WgpuBackend::default();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dimensions = Dimensions4D::new(1, 1, 2, 3);
        let out_dimensions = Dimensions4D::new(1, 1, 3, 2);

        let buffer = backend.buffer_from_values(dimensions, values).unwrap();
        let transpose = backend.transpose(&buffer, dimensions, 2, 3).unwrap();
        let transpose = backend.get_values(&transpose, out_dimensions).unwrap();

        let expected_values = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];

        assert!(compare_outputs(&transpose, &expected_values));
    }

    #[test]
    fn transposing_channel_and_width() {
        let mut backend = WgpuBackend::default();

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
        let transpose = backend.get_values(&transpose, dimensions).unwrap();

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

        assert!(compare_outputs(&transpose, &expected_values));
    }

    #[test]
    fn transposing_batch_and_channel() {
        let mut backend = WgpuBackend::default();

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
        let mut backend = WgpuBackend::default();

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

        let padded_dimensions = Dimensions4D::new(3, 1, 5, 6);
        let offset = Dimensions4D::new(0, 0, 1, 1);
        let zero_padded = backend
            .zero_pad(&buffer, dimensions, offset, padded_dimensions)
            .unwrap();
        let zero_padded = backend.get_values(&zero_padded, padded_dimensions).unwrap();

        #[rustfmt::skip]
        let expected_values = [
            // batch 2
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 2.0, 3.0, 4.0, 0.0,
            0.0, 5.0, 6.0, 7.0, 8.0, 0.0,
            0.0, 9.0, 10.0, 11.0, 12.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // batch 3
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 13.0, 14.0, 15.0, 16.0, 0.0,
            0.0, 17.0, 18.0, 19.0, 20.0, 0.0,
            0.0, 21.0, 22.0, 23.0, 24.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        assert!(compare_outputs(&zero_padded, &expected_values));
    }

    #[test]
    fn test_zero_pad2() {
        let mut backend = WgpuBackend::default();

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
        let zero_padded = backend.get_values(&zero_padded, padded_dimensions).unwrap();

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
        let mut backend = WgpuBackend::default();

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

        let dimensions = Dimensions4D::new(1, 1, 6, 4);

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();

        let sliced_dimensions = Dimensions4D::new(1, 1, 2, 2);
        let offset = Dimensions4D::new(0, 0, 4, 1);
        let sliced = backend
            .slice(&buffer, dimensions, offset, sliced_dimensions)
            .unwrap();
        let sliced = backend.get_values(&sliced, sliced_dimensions).unwrap();

        #[rustfmt::skip]
        let expected_values = [
            18.0, 19.0,
            22.0, 23.0,
        ];

        assert!(compare_outputs(&sliced, &expected_values));
    }

    #[test]
    fn test_slicing2() {
        let mut backend = WgpuBackend::default();

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

    #[test]
    fn test_max_pool2d() {
        let mut backend = WgpuBackend::default();

        let dimensions = Dimensions4D::new(2, 6, 4, 2);
        let out_dimensions = Dimensions4D::new(2, 6, 2, 1);
        let values = linspace(0.0, 1.0, dimensions.size());

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let pool_size = Dimensions2D::new(2, 2);
        let max_pooled = backend.max_pool2d(&buffer, dimensions, pool_size).unwrap();
        let max_pooled = backend.get_values(&max_pooled, out_dimensions).unwrap();

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
        let mut backend = WgpuBackend::default();

        let dimensions = Dimensions4D::new(2, 6, 5, 4);
        let values = linspace(0.0, 1.0, dimensions.size());
        // let values = vec![1.0; dimensions.size()];
        let kernel_dimensions = Dimensions4D::new(2, 6, 3, 3);
        let kernel_values = linspace(1.0, 3.0, kernel_dimensions.size());
        // let kernel_values = vec![1.0; kernel_dimensions.size()];
        let out_dimensions = Dimensions4D::new(2, 2, 3, 2);

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();

        let kernels = backend
            .buffer_from_values(kernel_dimensions, kernel_values.to_vec())
            .unwrap();

        let result = backend
            .conv2d(&buffer, &kernels, dimensions, kernel_dimensions)
            .unwrap();
        let result = backend.get_values(&result, out_dimensions).unwrap();

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
        // // output is NHWC, so need to switch the dimensions still
        // print_matrix(&result, out_dimensions);
        assert!(compare_outputs(&result, &expected_values));
    }

    #[test]
    fn test_addition_with_broadcasting_dimensions() {
        let mut backend = WgpuBackend::default();

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
        let result = backend.get_values(&result, rhs_dimensions).unwrap();

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
        // shape (2, 2, 5, )

        // print_matrix(&result, rhs_dimensions);
        assert!(compare_outputs(&result, &expected));
    }

    #[test]
    fn test_comparison() {
        let lhs_dimensions = Dimensions4D::new(2, 3, 5, 4);
        let rhs_dimensions = Dimensions4D::new(1, 1, 1, 1);

        let mut backend = WgpuBackend::default();

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
}
