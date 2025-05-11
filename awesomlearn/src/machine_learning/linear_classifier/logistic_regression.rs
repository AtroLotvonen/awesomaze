// logistic_regression.rs

pub struct LogisticRegressionClassifier {
    weights: Vec<f32>,
    learning_rate: f32,
    max_iter: usize,
}

impl LogisticRegressionClassifier {
    pub fn new(input_size: usize, learning_rate: f32, max_iter: usize) -> Self {
        LogisticRegressionClassifier {
            weights: vec![0.0; input_size],
            learning_rate,
            max_iter,
        }
    }

    pub fn weights(&self) -> &Vec<f32> {
        &self.weights
    }

    fn sigmoid(z: f32) -> f32 {
        1.0 / (1.0 + (-z).exp())
    }

    pub fn fit(&mut self, x: &[Vec<f32>], y: &[f32]) {
        for _ in 0..self.max_iter {
            // calculate the predictions
            let mut predictions: Vec<f32> = Vec::with_capacity(y.len());
            for features in x {
                let mut z = 0.0;
                for (feature, weight) in features.iter().zip(self.weights.iter()) {
                    z += weight * feature;
                }
                predictions.push(Self::sigmoid(z));
            }

            // calculate the gradient
            for j in 0..self.weights.len() {
                let mut gradient = 0.0;
                for ((prediction, gt), features) in predictions.iter().zip(y.iter()).zip(x) {
                    gradient += (prediction - gt) * features[j];
                }
                self.weights[j] -= gradient * self.learning_rate;
            }
        }
    }

    pub fn predict(&self, x: &[Vec<f32>], y: Option<&[f32]>) -> (Vec<f32>, Option<f32>) {
        let mut predictions = Vec::with_capacity(x.len());
        for features in x {
            let mut z = 0.0;
            for (feature, weight) in features.iter().zip(self.weights.iter()) {
                z += weight * feature;
            }
            predictions.push(Self::sigmoid(z));
        }
        if let Some(y) = y {
            let mut correct_predictions = 0.0;
            for (gt, prediction) in y.iter().zip(predictions.iter()) {
                if prediction.round() == *gt {
                    correct_predictions += 1.0;
                }
            }
            let acc = correct_predictions / y.len() as f32;
            (predictions, Some(acc))
        } else {
            (predictions, None)
        }
    }
}
