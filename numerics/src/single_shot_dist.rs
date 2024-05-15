use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

use ndarray::Array;
use serde::{Deserialize, Serialize};

use crate::{
    channel::{GammaStrategy, IteratedChannel},
    generate_floats, harmonic_oscillator_hamiltonian, marked_state_hamiltonian, HamiltonianType,
};

#[derive(Debug, Serialize, Deserialize, Clone)]
struct SingleShotConfig {
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
    time_start: f64,
    time_stop: f64,
    time_steps: usize,
    time_logspace: bool,
    hamiltonian_type: HamiltonianType,
    dim_sys: usize,
    num_samples: usize,
    gamma_strategy: GammaStrategy,
}

impl SingleShotConfig {
    fn to_file(&self, path: &Path) {
        let s = serde_json::to_string_pretty(self).expect("Could not serialize parameters.");
        let mut file = File::create(path).expect("Could not open file for write.");
        file.write_all(s.as_bytes())
            .expect("Could not write serialized data to file.");
    }

    /// Panics if cannot read properly.
    fn from_file(path: &Path) -> Self {
        if path.is_file() == false {
            panic!("Provided parameter path is not a file.")
        }
        if let Ok(file) = File::open(path) {
            let reader = BufReader::new(file);
            let out = serde_json::from_reader(reader);
            if out.is_ok() {
                out.unwrap()
            } else {
                dbg!(out.unwrap_err());
                panic!("Could not deserialize file")
            }
        } else {
            panic!("Could not open file")
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct SingleShotResults {
    /// format is (alpha, beta_env, beta_sys, time) for inputs
    /// outputs are (original_dist, mean_dist, std_dist, dist_of_mean)
    inputs: Vec<(f64, f64, f64, f64)>,
    outputs: Vec<(f64, f64, f64, f64)>,
    num_samples: usize,
    dim_sys: usize,
    label: String,
}

impl SingleShotResults {
    fn to_file(&self, path: &Path) {
        let s = serde_json::to_string(self).expect("Could not serialize parameters.");
        let mut file = File::create(path).expect("Could not open file for write.");
        file.write_all(s.as_bytes())
            .expect("Could not write serialized data to file.");
    }
}

pub fn run(config_file: &Path, results_file: &Path, label: String) {
    let conf = SingleShotConfig::from_file(config_file);
    let alphas = generate_floats(
        conf.alpha_start,
        conf.alpha_stop,
        conf.alpha_steps,
        conf.alpha_logspace,
    );
    let beta_envs = generate_floats(
        conf.beta_env_start,
        conf.beta_env_stop,
        conf.beta_env_steps,
        conf.beta_env_logspace,
    );
    let beta_syss = generate_floats(
        conf.beta_sys_start,
        conf.beta_sys_stop,
        conf.beta_sys_steps,
        conf.beta_sys_logspace,
    );
    let times = generate_floats(
        conf.time_start,
        conf.time_stop,
        conf.time_steps,
        conf.time_logspace,
    );
    let h_sys = conf.hamiltonian_type.as_ndarray(conf.dim_sys);
    let mut phi = IteratedChannel::new(h_sys, vec![0.], vec![0.], vec![0.], conf.gamma_strategy);
    let num_params = alphas.len() * beta_envs.len() * beta_syss.len() * times.len();
    let mut inputs = Vec::with_capacity(num_params);
    let mut outputs = Vec::with_capacity(num_params);
    for alpha in alphas.iter() {
        for beta_env in beta_envs.iter() {
            for beta_sys in beta_syss.iter() {
                for time in times.iter() {
                    phi.set_parameters(vec![*alpha], vec![*beta_env], vec![*time]);
                    inputs.push((*alpha, *beta_env, *beta_sys, *time));
                    let out = phi.simulate(*beta_sys, *beta_env, conf.num_samples);
                    let initial_dist = phi.state_distance(*beta_sys, *beta_env);
                    let output = (initial_dist, out[0].0, out[0].1, out[0].2);
                    outputs.push(output);
                }
            }
        }
    }
    let results = SingleShotResults {
        inputs,
        outputs,
        num_samples: conf.num_samples,
        dim_sys: conf.dim_sys,
        label,
    };
    results.to_file(results_file);
}
