use std::{
    collections::HashMap,
    sync::{Arc, Mutex, RwLock},
};

use ndarray_linalg::Scalar;
use num_complex::Complex64 as c64;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    get_rng_seed, harmonic_oscillator_hamiltonian, marked_state_hamiltonian,
    perform_fixed_interaction_channel, process_error_data, schatten_2_distance, thermal_state,
    HamiltonianType, RandomInteractionGen,
};
use ndarray::{linalg::kron, Array, Array2};

enum FixedPointTestType {
    SingleInteractionAverageAfter,
    SingleInteractionAverageBefore,
    MultiInteractionAverageAfter(usize),
    MultiInteractionAverageBefore(usize),
}

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

impl FixedPointDistanceResults {
    pub fn new(num_samples: usize) -> Self {
        FixedPointDistanceResults {
            num_samples,
            alphas_and_betas: Vec::new(),
            means: Vec::new(),
            stds: Vec::new(),
        }
    }

    pub fn add_result(&mut self, alpha: f64, beta: f64, mean: f64, std: f64) {
        self.alphas_and_betas.push((alpha, beta));
        self.means.push(mean);
        self.stds.push(std);
    }

    pub fn write_self_to_file(self, filename: String) {
        if let Ok(s) = serde_json::to_string(&self) {
            std::fs::write(&filename, s).expect(&format!(
                "Could not write FixedPointResults to file: {:}",
                filename
            ));
        }
    }
}

fn gen_alphas(config: &FixedPointDistanceConfig) -> Vec<f64> {
    if config.alpha_use_linear_step {
        Array::linspace(config.alpha_start, config.alpha_stop, config.alpha_steps).to_vec()
    } else {
        Array::logspace(
            10.,
            config.alpha_start.log10(),
            config.alpha_stop.log10(),
            config.alpha_steps,
        )
        .to_vec()
    }
}

fn gen_betas(config: &FixedPointDistanceConfig) -> Vec<f64> {
    if config.sys_beta_use_linear_step {
        Array::linspace(
            config.sys_beta_start,
            config.sys_beta_stop,
            config.sys_beta_steps,
        )
        .to_vec()
    } else {
        Array::logspace(
            10.,
            config.sys_beta_start.log10(),
            config.sys_beta_stop.log10(),
            config.sys_beta_steps,
        )
        .to_vec()
    }
}

pub fn fixed_point_distances(config: FixedPointDistanceConfig) {
    let alphas = gen_alphas(&config);
    let betas = gen_betas(&config);
    let h_env = harmonic_oscillator_hamiltonian(config.env_dim);
    let h_sys = match config.sys_hamiltonian {
        HamiltonianType::HarmonicOscillator => harmonic_oscillator_hamiltonian(config.sys_dim),
        HamiltonianType::MarkedState => marked_state_hamiltonian(config.sys_dim),
    };
    let rho_env = thermal_state(&h_env, config.env_beta);
    let h_tot = kron(&h_sys, &Array2::<c64>::eye(config.env_dim))
        + kron(&Array2::<c64>::eye(config.sys_dim), &h_env);
    // let rand_interaction_gen =
    //     RandomInteractionGen::new(get_rng_seed(), config.sys_dim * config.env_dim);
    let rand_interaction_gen = RandomInteractionGen::new(1, config.sys_dim * config.env_dim);
    let mut results = FixedPointDistanceResults::new(config.num_samples);
    for alpha in alphas {
        for beta in betas.clone() {
            let out = Array2::<c64>::zeros((h_tot.nrows(), h_tot.ncols()));
            let locker = Arc::new(RwLock::new(out));
            println!("Executing alpha = {:}, beta = {:}", alpha, beta);
            let rho_sys = thermal_state(&h_sys, beta);
            let rho = kron(&rho_sys, &rho_env);
            (0..config.num_samples).into_par_iter().for_each(|_| {
                let g = rand_interaction_gen.sample_gue() * c64::from_real(0.001);
                let chan_out =
                    perform_fixed_interaction_channel(&h_tot, &g, &rho, 1. / alpha, config.sys_dim);
                let mut rho_out = locker.write().expect("Could not obtain lock.");
                rho_out.scaled_add(1. / c64::from_real(config.num_samples as f64), &chan_out);
            });

            let final_state = locker.read().expect("Lock poisoned :(").clone();
            results.add_result(
                alpha,
                beta,
                schatten_2_distance(&final_state, &rho_sys),
                0.0,
            );
        }
    }
    results.write_self_to_file("/Users/matt/scratch/fixed_point_results.json".to_string());
}

mod tests {
    use std::collections::HashMap;

    use super::{fixed_point_distances, FixedPointDistanceConfig, FixedPointDistanceResults};

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

    #[test]
    fn test_fixed_point_distance() {
        let conf = FixedPointDistanceConfig {
            num_samples: 1000,
            time: 100.,
            alpha_start: 0.01,
            alpha_stop: 0.00001,
            alpha_steps: 20,
            alpha_use_linear_step: false,
            sys_beta_start: 0.0,
            sys_beta_stop: 2.0,
            sys_beta_steps: 20,
            sys_beta_use_linear_step: true,
            env_beta: 1.0,
            sys_hamiltonian: crate::HamiltonianType::HarmonicOscillator,
            sys_dim: 20,
            env_dim: 2,
        };
        fixed_point_distances(conf);
    }
}
