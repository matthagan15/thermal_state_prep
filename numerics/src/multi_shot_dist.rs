use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::{channel::GammaStrategy, HamiltonianType};

#[derive(Debug, Serialize, Deserialize, Clone)]
struct MultiShotParameters {
    alpha_start: f64,
    alpha_stop: f64,
    alpha_steps: usize,
    alpha_logspace: bool,
    time_start: f64,
    time_stop: f64,
    time_steps: usize,
    time_logspace: bool,
    beta_sys_start: f64,
    beta_sys_stop: f64,
    beta_sys_steps: usize,
    beta_sys_logspace: bool,
    beta_env_start: f64,
    beta_env_stop: f64,
    beta_env_steps: usize,
    beta_env_logspace: bool,
    epsilon_start: f64,
    epsilon_stop: f64,
    epsilon_steps: usize,
    epsilon_logspace: bool,
    hamiltonian_type: HamiltonianType,
    dim_sys: usize,
    num_samples: usize,
    time: f64,
    gamma_strategy: GammaStrategy,
}

impl MultiShotParameters {
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