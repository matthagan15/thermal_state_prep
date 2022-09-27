extern crate ndarray;
extern crate num_complex;
extern crate blas_src;

use ndarray::prelude::*;
use num_complex::Complex32 as c32;
use num_complex::Complex64 as c64;

fn expm(A: Array2<c64>) -> Array2<c64> {
    arr2(&[[c64::new(1., 0.)]])
}

fn main() {
    let a: Array2<f32> = array![[1.,2.], [3.,4.]];
    let b: Array2<f32> = array![[0., 1.], [1., 0.]];
    println!("matmul?");
    println!("{}", a.dot(&b));
    println!("Hello, world!");
}

mod tests {
    #[test]
    fn expm_random_eigenvalues() {

    }
}