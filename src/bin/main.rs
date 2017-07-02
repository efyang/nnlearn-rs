extern crate nnlearn;
extern crate mnist;
extern crate rulinalg;

use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};
use mnist::{Mnist, MnistBuilder};

use nnlearn::NeuralNetwork;

fn main() {
    let mnist = MnistBuilder::new().finalize()
}
