#[cfg(test)]
mod tests {
    use crate::machine_learning::{
        linspace,
        nn::{
            backend::Backend,
            backends::{cpu_backend::CpuBackend, wgpu_backend::WgpuBackend},
            dimensions::{Dimensions2D, Dimensions4D},
        },
    };

    use float_cmp::approx_eq;

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

    fn setup_matrices(bmnk: (usize, usize, usize, usize)) -> (Vec<f32>, Vec<f32>) {
        let (b, m, n, k) = bmnk;
        // the size should be the same as the dimensions size
        let lhs = linspace(1.0, 2.0, k * m);
        let rhs = linspace(1.0, 2.0, b * k * n);
        (lhs, rhs)
    }

    fn compute_matmul<B: Backend>(backend: &mut B) -> B::Storage {
        let (b, m, n, k) = (1, 3, 4, 2);
        let a_dimensions = Dimensions4D::new(1, 1, m, k);
        let b_dimensions = Dimensions4D::new(b, 1, k, n);

        let (a_values, b_values) = setup_matrices((b, m, n, k));
        let lhs = backend.buffer_from_values(a_dimensions, a_values).unwrap();
        let rhs = backend.buffer_from_values(b_dimensions, b_values).unwrap();
        // (5, 1, 3, 4)
        backend.matmul(&lhs, &rhs, (m, n, k), false).unwrap()
    }

    #[test]
    fn compare_matmul() {
        let mut cpu_backend = CpuBackend {};
        let mut gpu_backend = WgpuBackend::default();

        let cpu_result = compute_matmul(&mut cpu_backend);
        let gpu_result = compute_matmul(&mut gpu_backend);
        let gpu_result = gpu_backend
            .get_values(&gpu_result, Dimensions4D::new(1, 1, 3, 4))
            .unwrap();

        assert!(compare_outputs(&cpu_result, &gpu_result));
    }

    #[test]
    fn compare_add() {
        let mut cpu_backend = CpuBackend {};
        let mut gpu_backend = WgpuBackend::default();

        let dimensions = Dimensions4D::new(1, 10, 21, 32);
        let a_values = linspace(0.0, 1.0, dimensions.size());
        let b_values = linspace(1.0, 2.0, dimensions.size());

        let lhs = cpu_backend
            .buffer_from_values(dimensions, a_values.clone())
            .unwrap();
        let rhs = cpu_backend
            .buffer_from_values(dimensions, b_values.clone())
            .unwrap();

        let cpu_result = cpu_backend.add(&lhs, &rhs, dimensions, dimensions).unwrap();

        let lhs = gpu_backend
            .buffer_from_values(dimensions, a_values.clone())
            .unwrap();
        let rhs = gpu_backend
            .buffer_from_values(dimensions, b_values.clone())
            .unwrap();

        let gpu_result = gpu_backend.add(&lhs, &rhs, dimensions, dimensions).unwrap();

        let gpu_result = gpu_backend.get_values(&gpu_result, dimensions).unwrap();

        assert!(compare_outputs(&cpu_result, &gpu_result));
    }

    #[test]
    fn compare_mul() {
        let mut cpu_backend = CpuBackend {};
        let mut gpu_backend = WgpuBackend::default();

        let dimensions = Dimensions4D::new(1, 10, 21, 32);
        let a_values = linspace(0.0, 1.0, dimensions.size());
        let b_values = linspace(1.0, 2.0, dimensions.size());

        let lhs = cpu_backend
            .buffer_from_values(dimensions, a_values.clone())
            .unwrap();
        let rhs = cpu_backend
            .buffer_from_values(dimensions, b_values.clone())
            .unwrap();

        let cpu_result = cpu_backend.mul(&lhs, &rhs, dimensions, dimensions).unwrap();

        let lhs = gpu_backend
            .buffer_from_values(dimensions, a_values.clone())
            .unwrap();
        let rhs = gpu_backend
            .buffer_from_values(dimensions, b_values.clone())
            .unwrap();

        let gpu_result = gpu_backend.mul(&lhs, &rhs, dimensions, dimensions).unwrap();

        let gpu_result = gpu_backend.get_values(&gpu_result, dimensions).unwrap();

        assert!(compare_outputs(&cpu_result, &gpu_result));
    }

    #[test]
    fn compare_sub() {
        let mut cpu_backend = CpuBackend {};
        let mut gpu_backend = WgpuBackend::default();

        let dimensions = Dimensions4D::new(1, 10, 21, 32);
        let a_values = linspace(0.0, 1.0, dimensions.size());
        let b_values = linspace(1.0, 2.0, dimensions.size());

        let lhs = cpu_backend
            .buffer_from_values(dimensions, a_values.clone())
            .unwrap();
        let rhs = cpu_backend
            .buffer_from_values(dimensions, b_values.clone())
            .unwrap();

        let cpu_result = cpu_backend.sub(&lhs, &rhs, dimensions, dimensions).unwrap();

        let lhs = gpu_backend
            .buffer_from_values(dimensions, a_values.clone())
            .unwrap();
        let rhs = gpu_backend
            .buffer_from_values(dimensions, b_values.clone())
            .unwrap();

        let gpu_result = gpu_backend.sub(&lhs, &rhs, dimensions, dimensions).unwrap();

        let gpu_result = gpu_backend.get_values(&gpu_result, dimensions).unwrap();

        assert!(compare_outputs(&cpu_result, &gpu_result));
    }

    #[test]
    fn compare_max_pool2d() {
        let dimensions = Dimensions4D::new(32, 30, 100, 100);
        let values = linspace(0.0, 1.0, dimensions.size());
        let pool_size = Dimensions2D::new(2, 2);
        let out_values = Dimensions4D::new(32, 30, 50, 50);

        // Cpu
        let mut backend = CpuBackend {};

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let result_cpu = backend.max_pool2d(&buffer, dimensions, pool_size).unwrap();

        // Wgpu
        let mut backend = WgpuBackend::default();

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let result = backend.max_pool2d(&buffer, dimensions, pool_size).unwrap();
        let result_gpu = backend.get_values(&result, out_values).unwrap();

        assert!(compare_outputs(&result_cpu, &result_gpu));
    }

    #[test]
    fn compare_relu() {
        let dimensions = Dimensions4D::new(30, 32, 200, 100);
        let values = linspace(-2.0, 2.0, dimensions.size());

        // Cpu
        let mut backend = CpuBackend {};

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let result_cpu = backend.relu(&buffer, dimensions).unwrap();

        // Wgpu
        let mut backend = WgpuBackend::default();

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let result = backend.relu(&buffer, dimensions).unwrap();
        let result_gpu = backend.get_values(&result, dimensions).unwrap();

        assert!(compare_outputs(&result_cpu, &result_gpu));
    }

    #[test]
    fn compare_sigmoid() {
        let dimensions = Dimensions4D::new(30, 32, 200, 100);
        let values = linspace(-2.0, 2.0, dimensions.size());

        // Cpu
        let mut backend = CpuBackend {};

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let result_cpu = backend.sigmoid(&buffer, dimensions).unwrap();

        // Wgpu
        let mut backend = WgpuBackend::default();

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let result = backend.sigmoid(&buffer, dimensions).unwrap();
        let result_gpu = backend.get_values(&result, dimensions).unwrap();

        assert!(compare_outputs(&result_cpu, &result_gpu));
    }

    #[test]
    fn compare_leaky_relu() {
        let dimensions = Dimensions4D::new(30, 32, 200, 100);
        let values = linspace(-2.0, 2.0, dimensions.size());

        // Cpu
        let mut backend = CpuBackend {};
        let alpha = 1.2;

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let result_cpu = backend.leaky_relu(&buffer, dimensions, alpha).unwrap();

        // Wgpu
        let mut backend = WgpuBackend::default();

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let result = backend.leaky_relu(&buffer, dimensions, alpha).unwrap();
        let result_gpu = backend.get_values(&result, dimensions).unwrap();

        assert!(compare_outputs(&result_cpu, &result_gpu));
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
        let out_dimensions = Dimensions4D::new(
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
        let mut backend = CpuBackend {};

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let kernels = backend
            .buffer_from_values(kernel_dimensions, kernel_values.to_vec())
            .unwrap();

        let result_cpu = backend
            .conv2d(&buffer, &kernels, dimensions, kernel_dimensions)
            .unwrap();
        let result_cpu = backend.get_values(&result_cpu, out_dimensions).unwrap();

        // Wgpu
        let mut backend = WgpuBackend::default();

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let kernels = backend
            .buffer_from_values(kernel_dimensions, kernel_values.to_vec())
            .unwrap();

        let result_gpu = backend
            .conv2d(&buffer, &kernels, dimensions, kernel_dimensions)
            .unwrap();
        let result_gpu = backend.get_values(&result_gpu, out_dimensions).unwrap();

        assert!(compare_outputs(&result_cpu, &result_gpu));
    }

    #[test]
    fn compare_upsample2d() {
        let dimensions = Dimensions4D::new(32, 16, 100, 100);
        let values = linspace(0.0, 1.0, dimensions.size());
        let upsample_size = Dimensions2D::new(2, 2);
        let out_values = Dimensions4D::new(32, 16, 200, 200);

        // Cpu
        let mut backend = CpuBackend {};

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let result = backend
            .upsample2d(&buffer, dimensions, upsample_size)
            .unwrap();
        let result_cpu = backend.get_values(&result, out_values).unwrap();

        // Wgpu
        let mut backend = WgpuBackend::default();

        let buffer = backend
            .buffer_from_values(dimensions, values.to_vec())
            .unwrap();
        let result = backend
            .upsample2d(&buffer, dimensions, upsample_size)
            .unwrap();
        let result_gpu = backend.get_values(&result, out_values).unwrap();

        assert!(compare_outputs(&result_cpu, &result_gpu));
    }
}
