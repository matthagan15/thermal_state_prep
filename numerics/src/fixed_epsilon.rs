use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::{
    channel::{GammaStrategy, IteratedChannel},
    generate_floats, HamiltonianType,
};

const BINARY_SEARCH_UPPER_LIMIT: usize = 2_usize.pow(18);
/// Returns:
/// - `None` if the binary search upper limit is exceeded
/// - `Some((L, s))` where `L` is the minimum number of interactions
/// needed to reach distance `epsilon` and `s` is the std deviation.
/// -
fn binary_search<F>(f: F, epsilon: f64) -> Option<(usize, f64, f64, f64)>
where
    F: Fn(usize) -> (f64, f64, f64),
{
    let mut lower;
    let mut upper = 1;
    let (mut cur_dist, mut cur_dist_std, mut dist_of_avg) = f(upper);
    while cur_dist > epsilon {
        upper *= 2;
        if upper >= BINARY_SEARCH_UPPER_LIMIT {
            println!("[BINARY_SEARCH] upper limit reached.");
            return None;
        }
        (cur_dist, cur_dist_std, dist_of_avg) = f(upper);
    }
    lower = upper / 2;
    println!("[BINARY_SEARCH] searching in range [{lower}, {upper}].");
    while (upper - lower) > 1 {
        let mid = (lower + upper) / 2;
        (cur_dist, cur_dist_std, dist_of_avg) = f(mid);
        if cur_dist >= epsilon {
            lower = mid;
        } else {
            upper = mid;
        }
    }
    println!(
        "[BINARY_SEARCH] min_number_interactions = {:}, distances: {:} +- {:}; dist_of_avg = {:}",
        upper, cur_dist, cur_dist_std, dist_of_avg
    );
    Some((upper, cur_dist, cur_dist_std, dist_of_avg))
}

fn linear_search<F>(f: F, epsilon: f64) -> usize
where
    F: Fn(usize) -> f64,
{
    let mut ret = 1;
    while f(ret) > epsilon && ret <= BINARY_SEARCH_UPPER_LIMIT {
        ret += 1;
    }
    ret
}

pub fn run(config_file: &Path, results_file: &Path, label: String) -> MultiShotResults {
    let conf = FixedEpsilonConfig::from_file(config_file);
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
    let times = generate_floats(
        conf.time_start,
        conf.time_stop,
        conf.time_steps,
        conf.time_logspace,
    );
    let epsilons = generate_floats(
        conf.epsilon_start,
        conf.epsilon_stop,
        conf.epsilon_steps,
        conf.epsilon_logspace,
    );
    let h_sys = conf.hamiltonian_type.as_ndarray(conf.dim_sys);
    // let oscillator_gaps = harmonic_oscillator_gaps(conf.dim_sys);
    // let gamma_strat = GammaStrategy::known(oscillator_gaps, 100);
    let gamma_strat = conf.gamma_strategy;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for alpha in alphas.iter() {
        for epsilon in epsilons.iter() {
            for time in times.iter() {
                for beta_env in beta_envs.iter() {
                    println!("{:}", "*".repeat(50));
                    println!("inputs: {:}, {:}, {:}, {:}", alpha, beta_env, epsilon, time);
                    let interactions_to_epsilon = |num_interactions: usize| {
                        let phi_alphas = vec![*alpha; num_interactions];
                        // let phi_beta_envs = vec![*beta_env; num_interactions];
                        let phi_beta_envs = ndarray::Array::linspace(
                            beta_env / num_interactions as f64,
                            *beta_env,
                            num_interactions,
                        )
                        .to_vec();
                        let phi_times = vec![*time; num_interactions];
                        let phi = IteratedChannel::new(
                            h_sys.clone(),
                            phi_alphas,
                            phi_beta_envs,
                            phi_times,
                            gamma_strat.clone(),
                        );
                        let phi_outputs = phi.simulate(0.0, *beta_env, conf.num_samples);

                        if phi_outputs.len() == 0 {
                            panic!("Simulator should have returned at least one data point.");
                        }
                        let (mean_dist, std_dist, dist_of_avg) = phi_outputs[phi_outputs.len() - 1];
                        (mean_dist, std_dist, dist_of_avg)
                    };
                    let interactions_needed = binary_search(interactions_to_epsilon, *epsilon);
                    if interactions_needed.is_none() {
                        // don't save data any data and bail on the remaining beta_e
                        break;
                    }
                    inputs.push((*alpha, *beta_env, *epsilon, *time));
                    outputs.push((
                        interactions_needed.unwrap().0,
                        interactions_needed.unwrap().1,
                        interactions_needed.unwrap().2,
                        interactions_needed.unwrap().3,
                    ))
                }
            }
        }
    }
    MultiShotResults {
        inputs,
        outputs,
        num_samples: conf.num_samples,
        dim_sys: conf.dim_sys,
        label,
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct FixedEpsilonConfig {
    alpha_start: f64,
    alpha_stop: f64,
    alpha_steps: usize,
    alpha_logspace: bool,
    time_start: f64,
    time_stop: f64,
    time_steps: usize,
    time_logspace: bool,
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
    gamma_strategy: GammaStrategy,
}

impl FixedEpsilonConfig {
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

/// format of inputs is: (alpha, beta_env, epsilon, time)
/// format of outputs is: (num_steps, mean_dist, dist_of_mean, std)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MultiShotResults {
    inputs: Vec<(f64, f64, f64, f64)>,
    outputs: Vec<(usize, f64, f64, f64)>,
    num_samples: usize,
    dim_sys: usize,
    label: String,
}

impl MultiShotResults {
    pub fn to_file(&self, path: &Path) {
        let s = serde_json::to_string_pretty(self).expect("Could not serialize parameters.");
        let mut file = File::create(path).expect("Could not open file for write.");
        file.write_all(s.as_bytes())
            .expect("Could not write serialized data to file.");
    }
}
