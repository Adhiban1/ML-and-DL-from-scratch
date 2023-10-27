mod nn;
use nn::{rand_matrix, NN};

fn input() -> Vec<usize> {
    let i: Vec<String> = std::env::args().collect();
    i[1..i.len()]
        .iter()
        .map(|i| i.parse::<usize>().unwrap())
        .collect()
}

fn main() {
    let layers = input();
    let x = rand_matrix(100, layers[0]);

    let true_nn = NN::new(layers.clone());
    let y = true_nn.predict(&x);

    let mut nn = NN::new(layers);
    println!("Initial Loss: {}", nn.overall_loss(&y, &nn.predict(&x)));
    nn.fit(&x, &y, 100, 0.00001, 0.001);
    println!("Final Loss: {}", nn.overall_loss(&y, &nn.predict(&x)));
}
