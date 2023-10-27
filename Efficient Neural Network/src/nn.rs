use rand::Rng;
use std::io::Write;

pub struct NN {
    layers: Vec<usize>,
    wts: Vec<(usize, usize)>,
    w_size: usize,
    w: Vec<f64>,
    b: Vec<f64>,
    losses: Vec<f64>,
}

impl NN {
    pub fn new(layers: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        let mut wts = Vec::with_capacity(layers.len() - 1);
        let mut w_size = 0;
        for i in 0..layers.len() - 1 {
            let temp = layers[i..2 + i].to_vec();
            w_size += temp[0] * temp[1];
            wts.push((temp[1], temp[0]));
        }
        let w: Vec<f64> = vec![0.0; w_size]
            .iter()
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();
        let b: Vec<f64> = vec![0.0; layers.len() - 1]
            .iter()
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        Self {
            layers,
            wts,
            w_size,
            w,
            b,
            losses: vec![],
        }
    }

    pub fn forward(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut temp_x = x.clone();
        let mut shift = 0;
        for (index, (r, c)) in self.wts.iter().enumerate() {
            let mut z = Vec::with_capacity(*r);
            for i in 0..*r {
                let mut s = 0.0;
                for j in 0..*c {
                    s += temp_x[j] * self.w[shift + i * c + j];
                }
                s += self.b[index];
                z.push(s);
            }
            // print(z)
            shift += r * c;
            // std::mem::swap(&mut temp_x, &mut z);
            temp_x = z;
        }
        temp_x
    }

    fn forward_w(&self, x: &Vec<f64>, w: &Vec<f64>, b: &Vec<f64>, wi: usize, h: f64) -> Vec<f64> {
        let mut temp_x = x.clone();
        let mut shift = 0;
        for (index, (r, c)) in self.wts.iter().enumerate() {
            let mut z = Vec::with_capacity(*r);
            for i in 0..*r {
                let mut s = 0.0;
                for j in 0..*c {
                    if wi == shift + i * c + j {
                        s += temp_x[j] * (w[shift + i * c + j] + h);
                    } else {
                        s += temp_x[j] * w[shift + i * c + j];
                    }
                }
                s += b[index];
                z.push(s);
            }
            // print(z)
            shift += r * c;
            // std::mem::swap(&mut temp_x, &mut z);
            temp_x = z;
        }
        temp_x
    }

    fn forward_b(&self, x: &Vec<f64>, w: &Vec<f64>, b: &Vec<f64>, bi: usize, h: f64) -> Vec<f64> {
        let mut temp_x = x.clone();
        let mut shift = 0;
        for (index, (r, c)) in self.wts.iter().enumerate() {
            let mut z = Vec::with_capacity(*r);
            for i in 0..*r {
                let mut s = 0.0;
                for j in 0..*c {
                    s += temp_x[j] * w[shift + i * c + j];
                }
                if index == bi {
                    s += b[index] + h;
                } else {
                    s += b[index];
                }
                z.push(s);
            }
            // print(z)
            shift += r * c;
            // std::mem::swap(&mut temp_x, &mut z);
            temp_x = z;
        }
        temp_x
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        x.iter().map(|i| self.forward(i)).collect()
    }

    fn loss(&self, yt: &Vec<f64>, y: &Vec<f64>) -> f64 {
        let mut l = 0.0;
        for i in 0..yt.len() {
            l += (yt[i] - y[i]).powi(2);
        }
        l /= yt.len() as f64;
        l
    }

    pub fn overall_loss(&self, yt: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> f64 {
        let mut l = 0.0;
        for i in 0..yt.len() {
            for j in 0..yt[0].len() {
                l += (yt[i][j] - y[i][j]).powi(2);
            }
        }
        l /= (yt.len() * yt[0].len()) as f64;
        l
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>, epochs: usize, h: f64, lr: f64) {
        for epoch in 0..epochs {
            // println!("epochs:{}%", (epoch + 1) * 100 / epochs);
            // std::io::stdout().flush().unwrap();
            for xi in 0..x.len() {
                let mut dw = Vec::with_capacity(self.w_size);
                for wi in 0..self.w_size {
                    // print!("\rwi:{}%", (wi + 1) * 100 /self.w_size);
                    // std::io::stdout().flush().unwrap();
                    let dwi = (self.loss(&y[xi], &self.forward_w(&x[xi], &self.w, &self.b, wi, h))
                        - self.loss(&y[xi], &self.forward_w(&x[xi], &self.w, &self.b, wi, -h)))
                        / (2.0 * h);
                    dw.push(dwi);
                }
                // println!();

                let mut db = Vec::with_capacity(self.b.len());
                for bi in 0..self.b.len() {
                    let dbi = (self.loss(&y[xi], &self.forward_b(&x[xi], &self.w, &self.b, bi, h))
                        - self.loss(&y[xi], &self.forward_b(&x[xi], &self.w, &self.b, bi, -h)))
                        / (2.0 * h);
                    db.push(dbi);
                }

                for i in 0..dw.len() {
                    self.w[i] -= lr * dw[i];
                }

                for i in 0..db.len() {
                    self.b[i] -= lr * db[i];
                }
            }
            let l = self.overall_loss(&y, &self.predict(x));
            self.losses.push(l);
            print!("\r{}/{}: loss: {}", epoch + 1, epochs, l);
            std::io::stdout().flush().unwrap();
        }
        println!();
    }
}

pub fn rand_matrix(i: usize, j: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let mut v = vec![vec![0.0; j]; i];
    for a in 0..i {
        for b in 0..j {
            v[a][b] = rng.gen_range(-1.0..=1.0);
        }
    }
    v
}
