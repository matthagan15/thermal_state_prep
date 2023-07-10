
use core::num::{self};

use ndarray::{Array2, Array};
use ndarray_linalg::Scalar;
use num_complex::Complex64 as c64;
use serde::{Deserialize, Serialize};

use crate::{marked_state_hamiltonian, thermal_state, schatten_2_distance, harmonic_oscillator_hamiltonian, RandomInteractionGen, channel::Channel, HamiltonianType};

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

pub fn trace_norm_reduction_sweep() {
    let alphas = Array::logspace(10., -2., -4., 20).to_vec();
    let gaps = Array::linspace(0.1, 2., 20).to_vec();
    let betas = Array::logspace(10., -1., 1., 20).to_vec();
    for alpha in alphas {
        for gap in gaps.clone() {
            for beta in betas.clone() {
                println!("gap= {:}, alpha= {:}", gap, alpha);
                // let out = trace_norm_reduction(alpha, beta, gap);
                // println!("reduction: {:} +- {:}", out.0, out.1);
            }
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
struct TraceNormReductionResult {
    alpha: f64,
    beta_sys: f64,
    beta_env: f64,
    env_gap: f64,
    mean_distance: f64,
    std_distance: f64
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct TraceNormReductionConfig {
    output_file_path: String,
    alpha_start: f64,
    alpha_stop: f64,
    alpha_steps: usize,
    alpha_logspace: bool,
    beta_env: f64,
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
}

impl TraceNormReductionConfig {
    fn from_json(input_path: String) -> Self {
        if let Ok(conf_string) = std::fs::read_to_string(&input_path) {
            serde_json::from_str(&conf_string).expect("Config file could not be deserialized.")
        } else {
            panic!("No config file found at: {:}", input_path)
        }
    }

    fn run(&self) {
        let alphas = if self.alpha_logspace {
            Array::logspace(10.0, self.alpha_start.log10(), self.alpha_stop.log10(), self.alpha_steps).to_vec()
        } else {
            Array::linspace(self.alpha_start, self.alpha_stop, self.alpha_steps).to_vec()
        };
        let betas = if self.beta_sys_logspace {
            Array::logspace(10.0, self.beta_sys_start.log10(), self.beta_sys_stop.log10(), self.beta_sys_steps).to_vec()
        } else {
            Array::linspace(self.beta_sys_start, self.beta_sys_stop, self.beta_sys_steps).to_vec()
        };
        let env_gaps = if self.env_gap_logspace {
            Array::logspace(10.0, self.env_gap_start.log10(), self.env_gap_stop.log10(), self.env_gap_steps).to_vec()
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
                    let (mean, std) = trace_norm_reduction(alpha, beta, env_gap, self.beta_env, &h_sys);
                    let out_formatted = TraceNormReductionResult {
                        alpha,
                        beta_sys: beta,
                        beta_env: self.beta_env,
                        env_gap,
                        mean_distance: mean,
                        std_distance: std,
                    };
                    println!("{:}", "*".repeat(75));
                    println!("alpha = {:}, beta = {:}, gap = {:}", alpha, beta, env_gap);
                    println!("{:} +- {:}", mean, std);
                    results.push(out_formatted);
                }
            }
        }
        let data_string = serde_json::to_string(&results).expect("Could not serialize trace reduction sweep");
        std::fs::write(&self.output_file_path, data_string).expect(&format!("Could not write output to file: {:}", self.output_file_path));
    }
}

/// Computes the reduction in trace norm of the output of the channel after 
/// a single interaction normalized by the trace distance of the input to the
/// ideal output
pub fn trace_norm_reduction(alpha: f64, beta: f64, env_gap: f64, beta_env: f64, h_sys: &Array2<c64>) -> (f64, f64) {
    let beta_sys = beta;
    let h_env = harmonic_oscillator_hamiltonian(2) * c64::from_real(env_gap);
    let rng = RandomInteractionGen::new(1, 4);
    let t = 1e2_f64;
    let rho_ideal = thermal_state(h_sys, beta_env);
    let rho_input = thermal_state(h_sys, beta_sys);

    let mut phi = Channel::new(h_sys.clone(), h_env, alpha, t, rng);
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
    let num_samples = 3000;
    let num_interactions = 1;
    phi.estimator_sys(statistic, num_samples, num_interactions)
}

mod tests {
    use super::{trace_norm_reduction, trace_norm_reduction_sweep, TraceNormReductionConfig};

    #[test]
    fn test_trace_norm_reduction() {
        trace_norm_reduction_sweep();
    }

    #[test]
    fn test_config_sweep() {
        let filename = "/Users/matt/repos/thermal_state_prep/numerics/tmp/trace_dist_reduction_sweep.json".to_string();
        let conf = TraceNormReductionConfig::from_json(filename);
        conf.run();
    }
}
