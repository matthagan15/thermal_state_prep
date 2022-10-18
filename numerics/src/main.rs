extern crate ndarray;
extern crate num_complex;
extern crate blas_src;

use ndarray::prelude::*;
use num_complex::Complex32 as c32;
use num_complex::Complex64 as c64;
use expm::Expm;
use num_complex::ComplexFloat;

const i: c64 = c64::new(0.0, 1.0);

fn main() {
    let a: Array2<f32> = array![[1.,2.], [3.,4.]];
    let b: Array2<f32> = array![[0., 1.], [1., 0.]];
    println!("matmul?");
    println!("{}", a.dot(&b));
    println!("Hello, world!");
}

fn conjugate(M: &mut Array2<c64>) {
    for elem in M.iter_mut() {
        elem.im *= -1.;
    }
}

// Cannot transpose in place? so return a copy.
fn adjoint(M: &Array2<c64>) -> Array2<c64> {
    let mut adjointed = M.t().clone().to_owned();
    conjugate(&mut adjointed);
    adjointed
}

mod tests {
    use ndarray::prelude::*;
    use num_complex::{Complex64 as c64, ComplexFloat};
    use crate::{i, adjoint};

    use super::conjugate;
    #[test]
    fn expm_random_eigenvalues() {
        let mut e1 = c64::new(0., 1. * std::f64::consts::PI);
        let mut e2 = c64::new(0., 1. * std::f64::consts::PI);
        let mut eigens: Array2<c64> = array![[e1.clone(), c64::new(0., 0.)], [c64::new(0., 0.), e2.clone()]];
        let mut exp_eigens: Array2<c64> = array![[e1.clone().exp(), c64::new(0., 0.)], [c64::new(0., 0.), e2.clone().exp()]];
        let vecs: Array2<c64> = array![[c64::new(0., 0.), c64::new(1., 0.)], [c64::new(1., 0.), c64::new(0., 0.)]];
        let unexp = vecs.dot(&eigens).dot(&adjoint(&vecs));
        let mut expm_tester: Array2<c64> = Array2::zeros((2,2));
        expm::expm(&unexp, b);
        let exp_exact = vecs.dot(&exp_eigens).dot(&adjoint(&vecs));
        println!("exp_exact: {:}", exp_exact);
    }
}