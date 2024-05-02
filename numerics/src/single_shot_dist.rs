use ndarray::Array;
use serde::{Deserialize, Serialize};

use crate::{harmonic_oscillator_hamiltonian, marked_state_hamiltonian, HamiltonianType};

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
        todo!()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TraceNormReductionOutput {
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
