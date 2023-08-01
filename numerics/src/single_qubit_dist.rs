use core::num::{self};

use ndarray::{Array, Array2};
use ndarray_linalg::Scalar;
use num_complex::Complex64 as c64;
use serde::{Deserialize, Serialize};

use crate::{
    channel::Channel, harmonic_oscillator_hamiltonian, marked_state_hamiltonian,
    schatten_2_distance, thermal_state, HamiltonianType, RandomInteractionGen,
};

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub struct TraceNormReductionResult {
    alpha: f64,
    beta_sys: f64,
    env_gap: f64,
    mean: f64,
    std: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TraceNormReductionConfig {
    alpha_start: f64,
    alpha_stop: f64,
    alpha_steps: usize,
    alpha_logspace: bool,
    beta_sys_start: f64,
    beta_sys_stop: f64,
    beta_sys_steps: usize,
    beta_sys_logspace: bool,
    env_gap_start: f64,
    env_gap_stop: f64,
    env_gap_steps: usize,
    env_gap_logspace: bool,
    h_sys: HamiltonianType,
    dim_sys: usize,
    pub num_samples: usize,
    pub time: f64,
    pub beta_env: f64,
}

impl TraceNormReductionConfig {
    pub fn from_json(input_path: String) -> Self {
        if let Ok(conf_string) = std::fs::read_to_string(&input_path) {
            serde_json::from_str(&conf_string).expect("Config file could not be deserialized.")
        } else {
            panic!("No config file found at: {:}", input_path)
        }
    }

    pub fn run(&self) -> Vec<TraceNormReductionResult> {
        let alphas = if self.alpha_logspace {
            Array::logspace(
                10.0,
                self.alpha_start.log10(),
                self.alpha_stop.log10(),
                self.alpha_steps,
            )
            .to_vec()
        } else {
            Array::linspace(self.alpha_start, self.alpha_stop, self.alpha_steps).to_vec()
        };
        let betas = if self.beta_sys_logspace {
            Array::logspace(
                10.0,
                self.beta_sys_start.log10(),
                self.beta_sys_stop.log10(),
                self.beta_sys_steps,
            )
            .to_vec()
        } else {
            Array::linspace(self.beta_sys_start, self.beta_sys_stop, self.beta_sys_steps).to_vec()
        };
        let env_gaps = if self.env_gap_logspace {
            Array::logspace(
                10.0,
                self.env_gap_start.log10(),
                self.env_gap_stop.log10(),
                self.env_gap_steps,
            )
            .to_vec()
        } else {
            Array::linspace(self.env_gap_start, self.env_gap_stop, self.env_gap_steps).to_vec()
        };

        let h_sys = match self.h_sys {
            HamiltonianType::HarmonicOscillator => harmonic_oscillator_hamiltonian(self.dim_sys),
            HamiltonianType::MarkedState => marked_state_hamiltonian(self.dim_sys),
        };

        let mut results = Vec::new();
        for alpha in alphas {
            for beta in betas.clone() {
                for env_gap in env_gaps.clone() {
                    let (mean, std) = trace_norm_reduction(
                        alpha,
                        beta,
                        env_gap,
                        self.beta_env,
                        &h_sys,
                        self.num_samples,
                        self.time,
                    );
                    let out_formatted = TraceNormReductionResult {
                        alpha,
                        beta_sys: beta,
                        env_gap,
                        mean,
                        std,
                    };
                    // println!("{:}", "*".repeat(75));
                    // println!("alpha = {:}, beta = {:}, gap = {:}", alpha, beta, env_gap);
                    // println!("{:} +- {:}", mean, std);
                    results.push(out_formatted);
                }
            }
        }
        results
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TraceNormReductionOutput {
    pub label: String,
    pub experiment_type: String,
    pub num_samples: usize,
    pub beta_env: f64,
    pub time: f64,
    pub data: Vec<TraceNormReductionResult>,
}

impl TraceNormReductionOutput {
    pub fn write_json_to_file(&self, filename: String) {
        let data_string =
            serde_json::to_string(&self).expect("Could not serialize trace reduction sweep");
        std::fs::write(&filename, data_string)
            .expect(&format!("Could not write output to file: {:}", filename));
    }
}

/// Computes the reduction in trace norm of the output of the channel after
/// a single interaction normalized by the trace distance of the input to the
/// ideal output
pub fn trace_norm_reduction(
    alpha: f64,
    beta: f64,
    env_gap: f64,
    beta_env: f64,
    h_sys: &Array2<c64>,
    num_samples: usize,
    time: f64,
) -> (f64, f64) {
    let beta_sys = beta;
    let h_env = harmonic_oscillator_hamiltonian(2) * c64::from_real(env_gap);
    let rng = RandomInteractionGen::new(1, 4);
    let rho_ideal = thermal_state(h_sys, beta_env);
    let rho_input = thermal_state(h_sys, beta_sys);

    let mut phi = Channel::new(h_sys.clone(), h_env, alpha, time, rng);
    phi.set_env_to_thermal_state(beta_env);
    phi.set_sys_to_thermal_state(beta_sys);
    let input_distance = schatten_2_distance(&rho_ideal, &rho_input);
    if input_distance <= 1e-5 {
        println!("warning: input distance is too low, you're gonna have a bad time.");
        println!("input distance: {:}", input_distance);
    }
    let statistic = |state: &Array2<c64>| {
        let numerator = schatten_2_distance(&rho_ideal, state);
        numerator / input_distance
    };
    let num_interactions = 1;
    phi.estimator_sys(statistic, num_samples, num_interactions)
}
