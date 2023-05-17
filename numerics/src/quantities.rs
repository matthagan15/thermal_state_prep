use ndarray::Array2;
use ndarray_linalg::{Scalar, OperationNorm};
use num_complex::Complex64 as c64;

use crate::{zero, harmonic_oscillator_hamiltonian, RandomInteractionGen, channel::Channel, thermal_state};

pub fn analytic_second_order(
    h_tot: &Array2<c64>,
    rho: &Array2<c64>,
    time: f64,
    alpha: f64,
) -> Array2<c64> {
    let mut out = rho.clone();
    for i in 0..h_tot.nrows() {
        let mut acc = zero();
        for j in 0..h_tot.nrows() {
            if h_tot[[i,i]] == h_tot[[j, j]] {
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

fn numeric_first_derivative(
    phi: &Channel,
    rho_sys: &Array2<c64>,
    alpha: f64,
    time: f64,
) -> Array2<c64> {
    let delta = 1e-6;
    let center = phi.map_monte_carlo_avg(rho_sys, alpha + delta, time, 1000, 1);
    let right = phi.map_monte_carlo_avg(rho_sys, alpha, time, 1000, 1);
    let mut x = (&center - right);
    x.mapv_inplace(|x| x / c64::from_real(delta));
    x
}

fn numeric_second_derivative(
    phi: &Channel,
    rho_sys: &Array2<c64>,
    time: f64,
    alpha: f64,
) -> Array2<c64> {
    let delta = 1e-6;
    let center = phi.map_monte_carlo_avg(rho_sys, alpha, time, 2000, 1);
    let right = phi.map_monte_carlo_avg(rho_sys, alpha + delta, time, 2000, 1);
    let left = phi.map_monte_carlo_avg(rho_sys, alpha - delta, time, 2000, 1);
    (right + left - c64::from_real(2.) * center) / (c64::from_real(delta) * c64::from_real(delta))
}

fn first_order_taylors() {
    let h_sys = harmonic_oscillator_hamiltonian(15);
    let h_env = harmonic_oscillator_hamiltonian(2);
    let rho_sys = thermal_state(&h_sys, 0.75);
    let rng = RandomInteractionGen::new(1, 15 * 2);
    let phi = Channel::new(h_sys, h_env, 1., rng);
    let x = numeric_first_derivative(&phi, &rho_sys, 0.0, 100.);
    let y = numeric_second_derivative(&phi, &rho_sys, 100., 0.0);
    println!("norm of first order derivative: {:}", x.opnorm_one().unwrap());
    println!("norm of second order derivative: {:}", y.opnorm_one().unwrap());
}

mod test {
    use ndarray::{Array2, array, linalg::kron};
    use ndarray_linalg::OperationNorm;
    use num_complex::Complex64 as c64;

    use crate::{thermal_state, harmonic_oscillator_hamiltonian};

    use super::{first_order_taylors, analytic_second_order};
    #[test]
    fn test_complex_mul() {
        let c = c64::new(1.2, 2.3);
        println!("{:}", c * 2.0)
    }

    #[test]
    fn test_taylors() {
        first_order_taylors();
    }

    #[test]
    fn test_analytic_second_order() {
        let h_sys = harmonic_oscillator_hamiltonian(15);
        let h_env = harmonic_oscillator_hamiltonian(2);
        let h_tot = kron(&h_sys, &Array2::<c64>::eye(2)) + kron(&Array2::<c64>::eye(15), &h_env);
        let rho_sys = thermal_state(&h_sys, 0.0);
        let rho_env = array![[1.0.into(), 0.0.into()],[0.0.into(), 0.0.into()]];
        let rho = kron(&rho_sys, &rho_env);
        let x = analytic_second_order(&h_tot, &rho, 1000., 0.001);
        println!("x norm: {:}", x.opnorm_one().unwrap());
    }
}
