// neuron.rs

use rand::Rng;

#[derive(Clone, Debug)]
pub struct Neuron {
    weights: Vec<f32>, // this could be f32 also when using wgpu
    bias: f32,
    grads: Vec<f32>,
    bias_grad: f32,
}

impl Neuron {
    pub fn new(input_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let epsilon = 0.1; //f32::EPSILON; // The smallest representable positive f32 value greater than 0
        let mut weights = Vec::with_capacity(input_size); // Vector to store the random numbers

        for _ in 0..input_size {
            let random_value: f32 = rng.gen::<f32>(); // Generates a value in [0, 1)
            let random_weight = epsilon + (1.0 - 2.0 * epsilon) * random_value; // Shift to (0, 1)
            weights.push(random_weight); // Add to the vector
        }
        let random_value: f32 = rng.gen::<f32>(); // Generates a value in [0, 1)
        let bias = epsilon + (1.0 - 2.0 * epsilon) * random_value; // Shift to (0, 1)

        Neuron {
            weights,
            bias,
            grads: vec![0.0; input_size],
            bias_grad: 0.0,
        }
    }

    pub fn weights(&self) -> &Vec<f32> {
        &self.weights
    }

    pub fn bias(&self) -> f32 {
        self.bias
    }

    pub fn new_with_weights(weights: Vec<f32>, bias: f32) -> Self {
        let weights_size = weights.len();
        Neuron {
            weights,
            bias,
            grads: vec![0.0; weights_size],
            bias_grad: 0.0,
        }
    }

    pub fn grads(&self) -> &Vec<f32> {
        &self.grads
    }

    pub fn bias_grad(&self) -> f32 {
        self.bias_grad
    }

    pub fn forward(&self, input: &[f32]) -> f32 {
        self.weights
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum::<f32>()
            + self.bias
    }

    // Returns the input grad and saves the weight and bias grads
    pub fn backward(&mut self, input: &[f32], grad: f32) -> Vec<f32> {
        // compute the activation grad, this could be done also layerwise
        debug_assert_eq!(input.len(), self.weights.len());
        let mut input_grads = vec![0.0; input.len()];
        // Compute the gradients using chain rule. These could be implemented using autodiff also!
        for ((weight_grad, next_grad), (weight, input)) in self
            .grads
            .iter_mut()
            .zip(input_grads.iter_mut())
            .zip(self.weights.iter().zip(input.iter()))
        {
            *weight_grad += grad * input;
            *next_grad += grad * weight;
        }
        // Also compute the grad for the bias
        self.bias_grad += grad * 1.0;
        input_grads
    }

    pub fn update_weights(&mut self, batch_size: usize, learning_rate: f32) {
        self.weights
            .iter_mut()
            .zip(self.grads.iter())
            .for_each(|(weight, grad)| *weight -= learning_rate * grad / batch_size as f32);
        self.bias -= learning_rate * self.bias_grad / batch_size as f32;
        self.flush();
    }

    fn flush(&mut self) {
        self.grads.iter_mut().for_each(|grad| *grad = 0.0);
        self.bias_grad = 0.0;
    }
}

#[cfg(test)]
mod tests {

    use crate::machine_learning::nn::vec_compare;

    use super::*;

    #[test]
    fn test_neuron_forward() {
        let neuron = Neuron::new_with_weights(vec![0.5, 0.2, 0.3], 0.6);
        let input = vec![1.0, 2.0, 3.0];
        let forward_result = neuron.forward(&input);
        assert_eq!(forward_result, 2.4);
    }

    #[test]
    fn test_neuron_backward() {
        let weights = vec![0.5, 0.2, 0.3];
        let bias = 0.6;
        let mut neuron = Neuron::new_with_weights(weights.clone(), bias);
        let backward_input = 0.7;
        let input = vec![1.0, 2.0, 3.0];
        let backward_result = neuron.backward(&input, backward_input);
        // The gradients for the weights should be equal to the input gradient * input
        // and the output gradients should be input gradient * weights. Bias grad should just be
        // the input gradient * the bias
        let output_grads = vec![0.35, 0.14, 0.21];
        let neuron_grads = vec![0.7, 1.4, 2.1];
        assert!(vec_compare(&neuron.grads, &neuron_grads));
        assert!(vec_compare(&backward_result, &output_grads));
        assert_eq!(neuron.bias_grad, backward_input);
        // test update weights
        neuron.update_weights(2, 0.1);
        let new_neuron_weights = vec![
            0.5 - (0.7 / 2.0 * 0.1),
            0.2 - (1.4 / 2.0 * 0.1),
            0.3 - (2.1 / 2.0 * 0.1),
        ];
        let zeroed_neuron_grads = vec![0.0, 0.0, 0.0];
        assert!(vec_compare(&neuron.weights, &new_neuron_weights));
        // grads should be left as 0 after the update
        // assert!(vec_compare(&backward_result, &output_grads));
        assert_eq!(neuron.bias_grad, 0.0);
        assert!(vec_compare(&neuron.grads, &zeroed_neuron_grads));
    }

    #[test]
    fn test_layer_forward() {
        let weights = vec![0.5, 0.2, 0.3];
        let bias = 0.6;
        let mut neuron = Neuron::new_with_weights(weights.clone(), bias);
        let backward_input = 0.7;
        let input = vec![1.0, 2.0, 3.0];
        let backward_result = neuron.backward(&input, backward_input);
        // The gradients for the weights should be equal to the input gradient * input
        // and the output gradients should be input gradient * weights. Bias grad should just be
        // the input gradient * the bias
        let output_grads = vec![0.35, 0.14, 0.21];
        let neuron_grads = vec![0.7, 1.4, 2.1];
        assert!(vec_compare(&neuron.grads, &neuron_grads));
        assert!(vec_compare(&backward_result, &output_grads));
        assert_eq!(neuron.bias_grad, backward_input);
    }
}
