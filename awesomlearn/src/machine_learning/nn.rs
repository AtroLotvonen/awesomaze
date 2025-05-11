// nn.rs
pub mod activation;
pub mod backend;
pub mod backends;
pub mod dimensions;
pub mod model_config;
pub mod neuron;
pub mod nn_model;
pub mod nn_model_builder;
pub mod tensor;
pub mod tensor_autodiff;
pub mod tensor_error;
pub mod training_parameters;
pub mod weight_initialization;
pub mod optimizer;
pub mod awesom_rl_tensor;

use rand::seq::SliceRandom;
use rand::{thread_rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rand_xorshift::XorShiftRng;

pub fn generate_data(mean: (f32, f32), std: (f32, f32), points: usize) -> Vec<(f32, f32)> {
    let norm_dist_x = Normal::new(mean.0, std.0).unwrap();
    let norm_dist_y = Normal::new(mean.1, std.1).unwrap();
    let mut x_rand = XorShiftRng::from_seed(*b"MyFragileSeed123");
    let mut y_rand = XorShiftRng::from_seed(*b"MyFragileSeed321");
    let x_iter = norm_dist_x.sample_iter(&mut x_rand);
    let y_iter = norm_dist_y.sample_iter(&mut y_rand);
    x_iter.zip(y_iter).take(points).collect::<Vec<_>>()
}

pub fn generate_dataset(raw_features: &[Vec<Vec<f32>>]) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut dataset: Vec<(Vec<f32>, f32)> = Vec::new();

    for (i, raw_feature) in raw_features.iter().enumerate() {
        for feature_vector in raw_feature.iter() {
            dataset.push((feature_vector.clone(), i as f32)); // Clone each feature vector and pair with its class index
        }
    }
    let mut rng = thread_rng();
    dataset.shuffle(&mut rng); // Shuffle dataset

    let (features, classes): (Vec<_>, Vec<_>) = dataset.into_iter().unzip(); // Unzip into two separate vectors
    (features, classes)
}

pub fn linspace(start: f32, end: f32, num: usize) -> Vec<f32> {
    if num == 0 {
        return Vec::new();
    }
    let step = if num == 1 {
        0.0
    } else {
        (end - start) / (num - 1) as f32
    };
    (0..num).map(|i| start + step * i as f32).collect()
}

pub fn vec_compare(va: &[f32], vb: &[f32]) -> bool {
    (va.len() == vb.len()) &&  // zip stops at the shortest
     va.iter()
       .zip(vb)
       .all(|(a,b)| eq_with_nan_eq(*a,*b))
}

fn eq_with_nan_eq(a: f32, b: f32) -> bool {
    const TOLERANCE: f32 = 0.0001;
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= TOLERANCE
}
