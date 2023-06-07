
use core::num::{self};

use ndarray::Array2;
use ndarray_linalg::Scalar;
use num_complex::Complex64 as c64;

use crate::{marked_state_hamiltonian, thermal_state, schatten_2_distance, harmonic_oscillator_hamiltonian, RandomInteractionGen, channel::Channel};

/// Format of output is [small_mean, small_std, med_mean, med_std, large_mean, large_std]
pub fn output_distance_for_varied_gaps(num_interactions: usize) -> Vec<f64> {
    let mut results = Vec::new();
    let h_sys = marked_state_hamiltonian(10);
    let ideal_sys = thermal_state(&h_sys, 0.75);
    let dist_metric = |matrix: &Array2<c64>| {
        schatten_2_distance(matrix, &ideal_sys)
    };

    let small_env = harmonic_oscillator_hamiltonian(2) * c64::from_real(0.1);
    let rng = RandomInteractionGen::new(1, 20);
    let mut phi_small = Channel::new(h_sys.clone(), small_env, 0.01, 100., rng);
    phi_small.set_env_to_thermal_state(0.75);
    let out_small = phi_small.estimator_sys(dist_metric, 1000, num_interactions);
    results.push(out_small.0);
    results.push(out_small.1);

    let med_env = harmonic_oscillator_hamiltonian(2) * c64::from_real(1.);
    let rng = RandomInteractionGen::new(1, 20);
    let mut phi_med = Channel::new(h_sys.clone(), med_env, 0.01, 100., rng);
    phi_med.set_env_to_thermal_state(0.75);
    let out_med = phi_med.estimator_sys(dist_metric, 1000, num_interactions);
    results.push(out_med.0);
    results.push(out_med.1);

    let large_env = harmonic_oscillator_hamiltonian(2) * c64::from_real(10.);
    let rng = RandomInteractionGen::new(1, 20);
    let mut phi_large = Channel::new(h_sys, large_env, 0.01, 100., rng);
    phi_large.set_env_to_thermal_state(0.75);
    let out_large = phi_large.estimator_sys(dist_metric, 1000, num_interactions);
    results.push(out_large.0);
    results.push(out_large.1);

    results
}

pub fn qubit_thermalization(env_gap: f64) {
    let h_sys = harmonic_oscillator_hamiltonian(2);
    let h_env = harmonic_oscillator_hamiltonian(2) * c64::from_real(env_gap);
    let rng = RandomInteractionGen::new(1, 4);
    let alpha = 0.001;
    let t = 100.;
    let beta = 0.5;
    let rho_ideal = thermal_state(&h_sys, beta);
    let mut phi = Channel::new(h_sys, h_env, alpha, t, rng);
    phi.set_env_to_thermal_state(beta);
    let dist_est = |state: &Array2<c64>| {
        schatten_2_distance(state, &rho_ideal)
    };
    let num_samples = 1000;
    let num_interactions = 1000;
    let out = phi.estimator_sys(dist_est, num_samples, num_interactions);
    println!("gap: {:} yields: {:} +- {:}", env_gap, out.0, out.1);
}