use rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use rand::distributions::normal::StandardNormal;
use rand::{random, thread_rng, Rng};

type Fmatrix = Matrix<f64>;

struct Layer {
    weights: Fmatrix,
    biases: Fmatrix,
}

impl Layer {
    fn new(neurons_prev: usize, neurons: usize) -> Layer {
        Layer {
            weights: Fmatrix::from_fn(neurons, neurons_prev, |_, _| rand_stdnorm()),
            biases: Fmatrix::from_fn(neurons, 1, |_, _| rand_stdnorm()),
        }
    }

    fn compute_activation(&self, previous_activation: &Fmatrix) -> Fmatrix {
        sigmoid(
            self.weights.clone() * previous_activation + self.biases.clone(),
        )
    }

    // return (activation, weighted_input)
    fn compute_act_and_weight_input(&self, previous_activation: &Fmatrix) -> (Fmatrix, Fmatrix) {
        let weighted_input = self.weights.clone() * previous_activation + self.biases.clone();
        (sigmoid(weighted_input.clone()), weighted_input)
    }

    fn update_wb(&mut self, w_update: Fmatrix, b_update: Fmatrix) {
        self.weights += w_update;
        self.biases += b_update;
    }
}

pub struct NeuralNetwork {
    // layer_neuron_counts: Vec<usize>,
    // all besides input layer - that is just an activation
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    // layer_counts: input_count, hidden layers, output_count
    pub fn new(layer_counts: Vec<usize>) -> NeuralNetwork {
        NeuralNetwork {
            layers: layer_counts
                .iter()
                .enumerate()
                .skip(1)
                .map(|(i, &c)| Layer::new(layer_counts[i - 1], c))
                .collect(),
        }
    }

    // both 1 by N matrices
    pub fn run(&self, input_activation: Fmatrix) -> Fmatrix {
        let mut previous_activation = input_activation;
        for layer in self.layers.iter() {
            previous_activation = layer.compute_activation(&previous_activation);
        }
        previous_activation
    }

    pub fn stochastic_gd_learn(
        &mut self,
        training_data: &[(Fmatrix, Fmatrix)],
        epochs: usize,
        batch_size: usize,
        learn_rate: f64,
    ) {
        // shuffle the training data
        let mut training_data = training_data.to_owned();
        let mut rng = thread_rng();
        rng.shuffle(&mut training_data);

        for batch in training_data.chunks(batch_size).take_while(
            |c| c.len() == batch_size,
        )
        {
            for _ in 0..epochs {
                self.update_batch(batch, learn_rate)
            }
        }
    }

    // returns the partial derivative of the weight and bias vectors, respectively,
    // for each layer
    fn backprop(&self, x: Fmatrix, y: Fmatrix) -> Vec<(Fmatrix, Fmatrix)> {
        // setup activation and weighted input stores

        // for all layers
        let mut activations = vec![x];
        // for all layers from index 1 onwards
        let mut weighted_inputs = Vec::new();

        // feedforward step: compute activations and weighted inputs of each layer
        for (i, layer) in self.layers.iter().enumerate() {
            let (activation, weighted_input) = layer.compute_act_and_weight_input(&activations[i]);
            activations.push(activation);
            weighted_inputs.push(weighted_input);
        }

        // calculate the final output error
        let final_error = (activations.last().unwrap() - y).elemul(
            &sigmoid_derivative(
                weighted_inputs
                    .last()
                    .unwrap()
                    .clone(),
            ),
        );

        // list of errors starting from delta_L to delta_2
        let mut errors_backwards = vec![final_error];
        // backpropogate error
        // δ_l=( (w_(l+1))^T δ_(l+1) ) ⊙ σ′(z_l)
        for (layer_i, _) in self.layers.iter().enumerate().rev().skip(1) {
            let next_error_index = layer_i - (self.layers.len() - 1);
            let error = (self.layers[layer_i + 1].weights.transpose() *
                             errors_backwards[next_error_index].clone())
                .elemul(&sigmoid_derivative(weighted_inputs[layer_i - 1].clone()));
            errors_backwards.push(error);
        }

        errors_backwards
            .into_iter()
            .rev()
            .enumerate()
            .map(|(i, error)| {
                (error.clone() * activations[i].transpose(), error)
            })
            .collect()
    }

    fn update_batch(&mut self, batch: &[(Fmatrix, Fmatrix)], learn_rate: f64) {
        let derivatives = batch
            .iter()
            .cloned()
            .map(|(x, y)| self.backprop(x, y))
            .fold(
                // sum all the vectors of partial derivatives
                vec![
                    (
                    Fmatrix::zeros(self.layers.len(), 1),
                    Fmatrix::zeros(self.layers.len(), 1)
                );
                    self.layers.len()
                ],
                |acc, x| {
                    acc.iter()
                        .cloned()
                        .zip(x)
                        .map(|((a, b), (c, d))| (a + c, b + d))
                        .collect::<Vec<_>>()
                },
            )
            // average them
            .into_iter()
            .map(|(cw, cb)| (cw/(batch.len() as f64), cb/(batch.len() as f64)))
            .collect::<Vec<_>>();

        for (layer, (cw, cb)) in self.layers.iter_mut().zip(derivatives.into_iter()) {
            layer.update_wb(cw * learn_rate, cb * learn_rate);
        }
    }
}

// random number from standard normal distribution
fn rand_stdnorm() -> f64 {
    let StandardNormal(x) = random();
    x
}

// sigmoid function
fn sigmoid(z: Fmatrix) -> Fmatrix {
    z.apply(&|x| 1. / (1. + x.exp()))
}

// derivative of sigmoid function
fn sigmoid_derivative(z: Fmatrix) -> Fmatrix {
    sigmoid(z.clone()) * (-sigmoid(z) + 1.)
}
