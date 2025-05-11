// machine_learning.rs
pub mod dataloader;
pub mod linear_classifier;
pub mod nn;
pub mod reinforcement_learning;

use dataloader::Dataloader;
use rand::seq::SliceRandom;
use rand::{thread_rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rand_xorshift::XorShiftRng;
use std::collections::HashSet;

#[cfg(test)]
fn compare_outputs(a: &[f32], b: &[f32]) -> bool {
    use float_cmp::{ApproxEq, F32Margin};

    if a.len() != b.len() {
        return false;
    }
    let lax_margin = F32Margin {
        epsilon: 1e-3, // Allows for small numerical errors in very small numbers
        ulps: 50,      // Tolerates rounding errors for larger numbers
    };
    for (&a, &b) in a.iter().zip(b) {
        if !a.approx_eq(b, lax_margin) {
            return false;
        }
    }
    true
}

#[cfg(test)]
fn print_matrix(buffer: &[f32], dimensions: nn::dimensions::Dimensions4D) {
    for n in 0..dimensions.batch {
        println!("Batch {n}:");
        for c in 0..dimensions.depth {
            println!("  Channel {c}:");
            for h in 0..dimensions.height {
                print!("    [");
                for w in 0..dimensions.width {
                    if w != 0 {
                        print!(", ")
                    }
                    let i = n * dimensions.depth * dimensions.height * dimensions.width
                        + c * dimensions.height * dimensions.width
                        + h * dimensions.width
                        + w;
                    let x = buffer[i];
                    print!("{x}");
                }
                println!("]");
            }
        }
    }
}

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

pub struct DummyDataset {
    pub curr_index: usize,
    pub features: Vec<Vec<f32>>,
    pub classes: Vec<Vec<f32>>,
}

impl DummyDataset {
    pub fn new(point_count: usize) -> Self {
        let class1_dist = ((-0.1, 0.2), (0.5, 0.23));
        let class2_dist = ((0.45, 0.13), (0.2, 0.4));

        let class1: Vec<(f32, f32)> = generate_data(class1_dist.0, class1_dist.1, point_count);
        let class2: Vec<(f32, f32)> = generate_data(class2_dist.0, class2_dist.1, point_count);
        let raw_data = vec![
            class1
                .iter()
                .map(|(x, y)| vec![*x, *y])
                .collect::<Vec<Vec<f32>>>(),
            class2
                .iter()
                .map(|(x, y)| vec![*x, *y])
                .collect::<Vec<Vec<f32>>>(),
        ];
        let (features, classes) = generate_dataset(&raw_data);
        let classes_as_one_hot = classes_to_one_hot(&classes);
        Self {
            curr_index: 0,
            features,
            classes: classes_as_one_hot,
        }
    }
}

pub fn dummy_data(point_count: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let class1_dist = ((-0.1, 0.2), (0.5, 0.23));
    let class2_dist = ((0.45, 0.13), (0.2, 0.4));

    let class1: Vec<(f32, f32)> = generate_data(class1_dist.0, class1_dist.1, point_count);
    let class2: Vec<(f32, f32)> = generate_data(class2_dist.0, class2_dist.1, point_count);
    let raw_data = vec![
        class1
            .iter()
            .map(|(x, y)| vec![*x, *y])
            .collect::<Vec<Vec<f32>>>(),
        class2
            .iter()
            .map(|(x, y)| vec![*x, *y])
            .collect::<Vec<Vec<f32>>>(),
    ];
    let (features, classes) = generate_dataset(&raw_data);
    let classes_as_one_hot = classes_to_one_hot(&classes);
    (features, classes_as_one_hot)
}

impl Dataloader for DummyDataset {
    fn get(&mut self, amount: usize) -> (Vec<f32>, Vec<f32>, usize) {
        let end_index = (self.curr_index + amount).min(self.features.len());
        let batch_features = self.features[self.curr_index..end_index].to_vec();
        let batch_size = end_index - self.curr_index;
        let batch_classes = self.classes[self.curr_index..end_index].to_vec();

        let batch_features = batch_features.into_iter().flatten().collect();
        let batch_classes = batch_classes.into_iter().flatten().collect();
        self.curr_index = if end_index < self.features.len() {
            end_index
        } else {
            0
        };
        (batch_features, batch_classes, batch_size)
    }

    fn size(&self) -> usize {
        self.classes.len()
    }
}

pub fn classes_to_one_hot(labels: &[f32]) -> Vec<Vec<f32>> {
    let mut uniques = HashSet::new();
    labels.iter().for_each(|x| {
        uniques.insert((*x) as i64);
    });
    let mut one_hot = Vec::with_capacity(labels.len());
    match uniques.len() {
        1 | 2 => {
            for label in labels {
                one_hot.push(vec![*label]);
            }
            one_hot
        }
        x => {
            for label in labels {
                let mut zeros = vec![0.0; x];
                zeros[*label as usize] = 1.0;
                one_hot.push(zeros);
            }
            one_hot
        }
    }
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
