use ndarray::Array2;
use ndarray_linalg::Scalar;
use num_complex::Complex64 as c64;

use crate::{zero, harmonic_oscillator_hamiltonian, RandomInteractionGen, channel::Channel, thermal_state};

pub fn analytic_taylors_series(
    h_tot: &Array2<c64>,
    rho: &Array2<c64>,
    time: f64,
    alpha: f64,
) -> Array2<c64> {
    let mut out = rho.clone();
    for i in 0..h_tot.nrows() {
        let mut acc = zero();
        for j in 0..h_tot.nrows() {
            if i == j {
                continue;
            }
            let diff = (h_tot[[i, i]] - h_tot[[j, j]]).re;
            let mut numerator = rho[[i, i]] * (1. + f64::cos(diff * time))
                - rho[[j, j]] * (1. - f64::cos(diff * time));
            numerator *= 2.0 * alpha * alpha / (h_tot.nrows().pow(2) as f64 - 1.);
            let denominator = diff.pow(2.);
            acc += numerator / denominator;
        }
        out[[i, i]] += acc;
    }
    out
}

fn rho_func(alpha: f64) -> Array2<c64> {
    let h_sys = harmonic_oscillator_hamiltonian(10);
    let h_env = harmonic_oscillator_hamiltonian(2);
    let rho_sys = thermal_state(&h_sys, 0.0);
    let b_env = 1.0;
    let rng = RandomInteractionGen::new(1, 10 * 2);
    let phi = Channel::new(h_sys, h_env, b_env, rng);
    phi.map_monte_carlo_avg(&rho_sys, alpha, 100., 2000, 1)
}

fn numeric_first_derivative(

) {
    todo!()
}

fn numeric_second_derivative(
    h_sys: Array2<c64>,
    h_env: Array2<c64>,
    time: f64,
    alpha: f64,
) -> Array2<c64> {
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
