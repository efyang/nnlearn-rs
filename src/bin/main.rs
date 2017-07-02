extern crate nnlearn;
extern crate mnist;
extern crate rulinalg;

use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};
use mnist::{Mnist, MnistBuilder};

use nnlearn::NeuralNetwork;

fn main() {
    let (trn_size, tst_size, rows, cols) = (50_000usize, 10_000usize, 784, 1);
    #[allow(unused_variables)]
    let Mnist {
        trn_img,
        trn_lbl,
        val_img,
        val_lbl,
        tst_img,
        tst_lbl,
    } = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(trn_size as u32)
        .finalize();

    // Convert the flattened training images vector to a matrix.
    let trn_imgs = (0..trn_size)
        .map(|i| {
            Matrix::new(
                rows as usize,
                cols as usize,
                trn_img[784 * i..784 * (i + 1)]
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let trn_lbls = (0..trn_size)
        .map(|i| {
            Matrix::new(
                10,
                1,
                trn_lbl[10 * i..10 * (i + 1)]
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let tst_imgs = (0..tst_size)
        .map(|i| {
            Matrix::new(
                rows as usize,
                cols as usize,
                tst_img[784 * i..784 * (i + 1)]
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let tst_lbls = (0..tst_size)
        .map(|i| {
            Matrix::new(
                10,
                1,
                tst_lbl[10 * i..10 * (i + 1)]
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let training_data = trn_imgs
        .iter()
        .cloned()
        .zip(trn_lbls.iter().cloned())
        .collect::<Vec<_>>();

    let mut nn = NeuralNetwork::new(&[784, 30, 10]);
    nn.stochastic_gd_learn(training_data.as_slice(), 10, 10, 3.);

    let mut correct = 0;
    for (image, label) in tst_imgs.iter().zip(tst_lbls.iter()) {
        let output = nn.run(image.clone());
        println!("output: {}", output);
        if output == *label {
            correct += 1;
        }
    }
    println!("{}/10000", correct);

    nn.stochastic_gd_learn(training_data.as_slice(), 10, 10, 3.);

    let mut correct = 0;
    for (image, label) in tst_imgs.iter().zip(tst_lbls.iter()) {
        let output = nn.run(image.clone());
        println!("output: {}", output);
        if output == *label {
            correct += 1;
        }
    }
    println!("{}/10000", correct);

}
