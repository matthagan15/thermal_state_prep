use std::{collections::HashMap, sync::Mutex};

use ndarray_linalg::Scalar;
use num_complex::Complex64 as c64;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{HamiltonianType, harmonic_oscillator_hamiltonian, marked_state_hamiltonian, thermal_state, RandomInteractionGen, get_rng_seed, perform_fixed_interaction_channel};
use ndarray::{Array, Array2, linalg::kron};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FixedPointDistanceConfig {
    num_samples: usize,
    time: f64,
    alpha_start: f64,
    alpha_stop: f64,
    alpha_steps: usize,
    alpha_use_linear_step: bool,
    sys_beta_start: f64,
    sys_beta_stop: f64,
    sys_beta_steps: usize,
    sys_beta_use_linear_step: bool,
    env_beta: f64,
    // Currently only support harmonic oscillator and marked state
    sys_hamiltonian: HamiltonianType,
    sys_dim: usize,
    // currently only support harmonic oscillator env
    env_dim: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FixedPointDistanceResults {
    num_samples: usize,
    alphas_and_betas: Vec<(f64, f64)>,
    means: Vec<f64>,
    stds: Vec<f64>,
}

fn gen_alphas(config: &FixedPointDistanceConfig) -> Vec<f64> {
    if config.alpha_use_linear_step {
        Array::linspace(config.alpha_start, config.alpha_stop, config.alpha_steps).to_vec()
    } else {
        Array::logspace(10., config.alpha_start, config.alpha_stop, config.alpha_steps).to_vec()
    }
}

fn gen_betas(config: &FixedPointDistanceConfig) -> Vec<f64> {
    if config.sys_beta_use_linear_step {
        Array::linspace(config.sys_beta_start, config.sys_beta_stop, config.sys_beta_steps).to_vec()
    } else {
        Array::logspace(10., config.sys_beta_start, config.sys_beta_stop, config.sys_beta_steps).to_vec()
    }
}

pub fn fixed_point_distances(config: FixedPointDistanceConfig) {
    let alphas = gen_alphas(&config);
    let betas = gen_betas(&config);
    let h_env = harmonic_oscillator_hamiltonian(config.env_dim);
    let h_sys = match config.sys_hamiltonian {
        HamiltonianType::HarmonicOscillator => {
            harmonic_oscillator_hamiltonian(config.sys_dim)
        },
        HamiltonianType::MarkedState => {
            marked_state_hamiltonian(config.sys_dim)
        }
    };
    let rho_env = thermal_state(&h_env, config.env_beta);
    let h_tot = kron(&h_sys, &Array2::<c64>::eye(config.env_dim)) + kron(&Array2::<c64>::eye(config.sys_dim), &h_env);
    let rand_interaction_gen = RandomInteractionGen::new(get_rng_seed(), config.sys_dim * config.env_dim);
    let locker = Mutex::new(Vec::<(f64, f64)>::new());
    for alpha in alphas {
        for beta in betas.clone() {
            let rho_sys = thermal_state(&h_sys, beta);
            let rho = kron(&rho_sys, &rho_env);
            (0..config.num_samples).into_par_iter().for_each(|_| {
                let g = rand_interaction_gen.sample_interaction() * c64::from_real(alpha);
                perform_fixed_interaction_channel(&h_tot, &g, &rho, config.time, config.sys_dim);
            });
        }
    }
}

mod tests {
    use std::collections::HashMap;

    use super::FixedPointDistanceResults;

    #[test]
    fn test_serialize_results() {
        let res = FixedPointDistanceResults {
            num_samples: 1,
            alphas_and_betas: vec![(1., 2.)],
            means: vec![0.1],
            stds: vec![0.01],
        };
        let s = serde_json::to_string(&res).expect("couldn't serialize");
        std::fs::write("/Users/matt/scratch/fixed_point_result.json", s).expect("error writing.");
    }
}