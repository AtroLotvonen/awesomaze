use rand_distr::{Distribution, Uniform};

pub fn he_uniform(input_size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();

    // this is used as a default in burn and seems to work for the awesomaze
    let limit = (3.0 / input_size as f32).sqrt();
    let between = Uniform::from(-limit..limit);

    (0..input_size).map(|_| between.sample(&mut rng)).collect()
}

pub fn lecun_uniform(input_size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let limit = (1.0 / input_size as f32).sqrt(); // LeCun uniform limit
    let between = Uniform::from(-limit..limit);

    // Initialize weights
    (0..input_size)
        .map(|_| between.sample(&mut rng))
        .collect::<Vec<f32>>()
}

// pub fn lecun_conv_uniform(
//     in_channels: usize,
//     out_channels: usize,
//     filter_height: usize,
//     filter_width: usize,
// ) -> Vec<f32> {
//     let mut rng = rand::thread_rng();
//     let n_in = in_channels * filter_height * filter_width;
//     let limit = (1.0 / n_in as f32).sqrt();
//     let between = Uniform::from(-limit..limit);
//
//     // Initialize weights
//     (0..out_channels)
//         .flat_map(|_| {
//             (0..in_channels)
//                 .flat_map(|_| {
//                     (0..filter_height)
//                         .flat_map(|_| (0..filter_width).map(|_| between.sample(&mut rng)))
//                         .collect::<Vec<f32>>()
//                 })
//                 .collect::<Vec<f32>>()
//         })
//         .collect()
// }
