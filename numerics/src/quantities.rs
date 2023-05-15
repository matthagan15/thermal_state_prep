use ndarray::Array2;
use ndarray_linalg::Scalar;
use num_complex::Complex64 as c64;

use crate::zero;

pub fn analytic_taylors_series(h_tot: &Array2<c64>, rho: &Array2<c64>, time: f64, alpha: f64) -> Array2<c64> {
    let mut out = rho.clone();
    for i in 0..h_tot.nrows() {
        let mut acc = zero();
        for j in 0..h_tot.nrows() {
            if i == j {
                continue;
            }
            let diff = (h_tot[[i, i]] - h_tot[[j, j]]).re;
            let mut numerator = rho[[i, i]] * (1. + f64::cos(diff * time)) - rho[[j, j]] * (1. - f64::cos(diff * time));
            numerator *= 2.0 * alpha * alpha / (h_tot.nrows().pow(2) as f64 - 1.);
            let denominator = diff.pow(2.);
            acc += numerator / denominator;
        }
        out[[i, i]] += acc;
    }
    out
}

fn numeric_second_derivative(h_sys: Array2<c64>, h_env: Array2<c64>, time: f64, alpha: f64) -> Array2<c64> {
    
    Array2::<c64>::eye(1)
}

mod test {
    use num_complex::Complex64 as c64;
    #[test]
    fn test_complex_mul() {
        let c = c64::new(1.2, 2.3);
        println!("{:}", c * 2.0)
    }
}