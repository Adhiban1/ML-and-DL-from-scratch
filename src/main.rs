use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();

    fn vec_dot_mat(a:&Vec<f64>, b:&Vec<Vec<f64>>) -> Vec<f64> {
        let mut c = vec![];
        for k in 0..b[0].len() {
            let mut s = 0.0;
            for j in 0..a.len() {
                s += a[j] * b[j][k];
            }
            c.push(s);
        }
        c
    }
    
    fn rand_matrix(i:usize, j:usize) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        let mut m = vec![vec![0.0;j];i];
        for a in 0..i {
            for b in 0..j {
                m[a][b] = rng.gen_range(-1.0..=1.0);
            }
        }
        m
    }

    let x = rand_matrix(10, 10);
    let wt1 = rand_matrix(10, 5);
    let bt1 = rng.gen_range(-1.0..=1.0);
    let wt2 = rand_matrix(5, 1);
    let bt2 = rng.gen_range(-1.0..=1.0);
    
    let mut w1 = rand_matrix(10, 5);
    let mut b1 = rng.gen_range(-1.0..=1.0);
    let mut w2 = rand_matrix(5, 1);
    let mut b2 = rng.gen_range(-1.0..=1.0);
    
    fn forward(x:&Vec<f64>, w1:&Vec<Vec<f64>>, w2:&Vec<Vec<f64>>, b1:&f64, b2:&f64) -> Vec<f64> {
        let z1:Vec<f64> = vec_dot_mat(x, w1).iter().map(|i|i+b1).collect();
        let z2 = vec_dot_mat(&z1, w2).iter().map(|i|i+b2).collect();
        z2
    }
    
    let y:Vec<Vec<f64>> = x.iter().map(|i| forward(i, &wt1, &wt2, &bt1, &bt2)).collect();

    fn mod_w(w:&Vec<Vec<f64>>, index:(usize, usize), h:f64) -> Vec<Vec<f64>>{
        let mut w1 = w.clone();
        w1[index.0][index.1] += h;
        w1
    }

    fn loss(x:&Vec<f64>, y:&Vec<f64>, w1:&Vec<Vec<f64>>, w2:&Vec<Vec<f64>>, b1:&f64, b2:&f64) -> f64 {
        let mut l = 0.0;
        let yp = forward(x, w1, w2, b1, b2);
        for i in 0..y.len() {
            l += (y[i] - yp[i]).powi(2);
        }
        l /= y.len() as f64;
        l
    }
    
    fn overall_loss(x:&Vec<Vec<f64>>, y:&Vec<Vec<f64>>, w1:&Vec<Vec<f64>>, w2:&Vec<Vec<f64>>, b1:&f64, b2:&f64) -> f64 {
        let mut yp = vec![];
        for i in x {
            let temp = forward(i, w1, w2, b1, b2);
            yp.push(temp);
        }
        let mut l = 0.0;
        for i in 0..y.len() {
            for j in 0..y[0].len() {
                l += (y[i][j] - yp[i][j]).powi(2);
            }
        }
        l /= (y.len() * y[0].len()) as f64;
        l
    }

    fn grad(x:&Vec<f64>, y:&Vec<f64>, w1:&Vec<Vec<f64>>, w2:&Vec<Vec<f64>>, b1:&f64, b2:&f64, h:f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, f64, f64) {
        let mut dw1 = vec![];
        for i in 0..w1.len() {
            let mut temp = vec![];
            for j in 0..w1[0].len() {
                temp.push((loss(x, y, &mod_w(w1, (i, j), h), w2, b1, b2) - loss(x, y, &mod_w(w1, (i, j), -h), w2, b1, b2)) / (2.0 * h));
            }
            dw1.push(temp);
        }

        let mut dw2 = vec![];
        for i in 0..w2.len() {
            let mut temp = vec![];
            for j in 0..w2[0].len() {
                temp.push((loss(x, y, w1, &mod_w(w2, (i, j), h), b1, b2) - loss(x, y, w1, &mod_w(w2, (i, j), -h), b1, b2)) / (2.0 * h));
            }
            dw2.push(temp);
        }

        let db1 = (loss(x, y, w1, w2, &(b1+h), b2) - loss(x, y, w1, w2, &(b1-h), b2)) / (2.0 * h);
        let db2 = (loss(x, y, w1, w2, b1, &(b2+h)) - loss(x, y, w1, w2, b1, &(b2-h))) / (2.0 * h);

        (dw1, dw2, db1, db2)
    }

    let epochs = 1000;
    let lr = 0.01;
    let h = 0.001;
    let mut opt_w_b = (w1.clone(), w2.clone(), b1.clone(), b2.clone());
    let mut lowest_loss = overall_loss(&x, &y, &w1, &w2, &b1, &b2);
    println!("Inital loss: {}", lowest_loss);
    for _ in 0..epochs {
        for index in 0..x.len() {
            let (dw1, dw2, db1, db2) = grad(&x[index], &y[index], &w1, &w2, &b1, &b2, h);
            for i in 0..w1.len() {
                for j in 0..w1[0].len() {
                    w1[i][j] -= lr * dw1[i][j];
                }
            }
    
            for i in 0..w2.len() {
                for j in 0..w2[0].len() {
                    w2[i][j] -= lr * dw2[i][j];
                }
            }
    
            b1 -= lr * db1;
            b2 -= lr * db2;
        }
        let l = overall_loss(&x, &y, &w1, &w2, &b1, &b2);
        if l < lowest_loss {
            lowest_loss = l;
            opt_w_b = (w1.clone(), w2.clone(), b1.clone(), b2.clone());
        }
    }
    println!("Lowest loss: {}", lowest_loss);
    println!("Final loss: {}", overall_loss(&x, &y, &w1, &w2, &b1, &b2));
    // println!("{:?}", opt_w_b);
}