use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

use ndarray::Array;
use serde::{Deserialize, Serialize};

use crate::{
    channel::GammaStrategy, harmonic_oscillator_hamiltonian, marked_state_hamiltonian,
    HamiltonianType,
};

#[derive(Debug, Serialize, Deserialize, Clone)]
struct SingleShotParameters {
    alpha_start: f64,
    alpha_stop: f64,
    alpha_steps: usize,
    alpha_logspace: bool,
    beta_sys_start: f64,
    beta_sys_stop: f64,
    beta_sys_steps: usize,
    beta_sys_logspace: bool,
    beta_env_start: f64,
    beta_env_stop: f64,
    beta_env_steps: usize,
    beta_env_logspace: bool,
    hamiltonian_type: HamiltonianType,
    dim_sys: usize,
    num_samples: usize,
    time: f64,
    gamma_strategy: GammaStrategy,
}

impl SingleShotParameters {
    fn to_file(&self, path: &Path) {
        let s = serde_json::to_string_pretty(self).expect("Could not serialize parameters.");
        let mut file = File::create(path).expect("Could not open file for write.");
        file.write_all(s.as_bytes())
            .expect("Could not write serialized data to file.");
    }

    fn from_file(path: &Path) -> Self {
        if path.is_file() == false {
            panic!("Provided parameter path is not a file.")
        }
        if let Ok(file) = File::open(path) {
            let reader = BufReader::new(file);
            if let Ok(out) = serde_json::from_reader(reader) {
                out
            } else {
                panic!("Could not deserialize file")
            }
        } else {
            panic!("Could not open file")
        }
    }
}

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
