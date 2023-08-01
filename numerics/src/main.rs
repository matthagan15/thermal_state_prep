extern crate blas_src;
extern crate ndarray;
extern crate num_complex;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use clap::{Parser, Subcommand};
use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray_linalg::expm::expm;
use ndarray_linalg::random_hermite;
use ndarray_linalg::{OperationNorm, Scalar};
use num_complex::Complex64 as c64;
use num_complex::ComplexFloat;
use numerics::channel::Channel;
use numerics::single_qubit_dist::{TraceNormReductionConfig, TraceNormReductionOutput};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use numerics::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeConfig {
    num_interactions: u32,
    num_samples: usize,
    time: f64,
    alpha: f64,
    sys_start_beta: f64,
    env_beta: f64,
    // Currently only support harmonic oscillator and marked state
    sys_hamiltonian: HamiltonianType,
    sys_dim: usize,
    // currently only support harmonic oscillator env
    env_dim: usize,
}

impl NodeConfig {
    pub fn from_path(config_path: String) -> Self {
        let mut p = config_path.clone();
        if p.ends_with("tsp.conf") == false {
            p.push_str("tsp.conf");
        }
        if let Ok(conf_string) = std::fs::read_to_string(&p) {
            serde_json::from_str(&conf_string).expect("Config file could not be deserialized.")
        } else {
            panic!("No config file found at: {:}", p)
        }
    }
}

fn multi_interaction(
    tot_hamiltonian: &Array2<c64>,
    env_state: &Array2<c64>,
    sys_state: &Array2<c64>,
    alpha: f64,
    time: f64,
    num_interactions: usize,
    rng: RandomInteractionGen,
) -> Array2<c64> {
    let mut sys_out = sys_state.clone();
    for _ in 0..num_interactions {
        let interaction_sample: Array2<c64> = (c64::from_real(alpha)) * rng.sample_gue();
        let h_with_interaction = c64::new(0., time) * (tot_hamiltonian + &interaction_sample);
        let time_evolution_op = expm(&h_with_interaction).expect("we ball");
        let time_evolution_op_adjoint = adjoint(&time_evolution_op);
        let mut out = kron(&sys_out, env_state);
        out = time_evolution_op.dot(&out);
        out = out.dot(&time_evolution_op_adjoint);
        sys_out = partial_trace(&out, sys_state.nrows(), env_state.nrows());
    }
    sys_out
}

fn multi_interaction_error_mc(
    num_samples: usize,
    num_interactions: usize,
    sys_hamiltonian: &Array2<c64>,
    env_hamiltonian: &Array2<c64>,
    sys_initial_beta: f64,
    env_beta: f64,
    alpha: f64,
    time: f64,
    rng: RandomInteractionGen,
) -> Vec<f64> {
    let h = kron(
        sys_hamiltonian,
        &Array2::<c64>::eye(env_hamiltonian.nrows()),
    ) + kron(
        &Array2::<c64>::eye(sys_hamiltonian.nrows()),
        env_hamiltonian,
    );
    let rho_env = thermal_state(env_hamiltonian, env_beta);
    let rho_sys = thermal_state(sys_hamiltonian, sys_initial_beta);
    let sys_ideal = thermal_state(sys_hamiltonian, env_beta);
    let mut errors: Vec<f64> = Vec::with_capacity(num_samples);
    let mut locker = RwLock::new(errors);
    (0..num_samples).into_par_iter().for_each(|_| {
        let rho_evolved_sample = multi_interaction(
            &h,
            &rho_env,
            &rho_sys,
            alpha,
            time,
            num_interactions,
            rng.clone(),
        );
        let error = schatten_2_distance(&rho_evolved_sample, &sys_ideal);
        let mut v = locker.write().expect("no locker");
        v.push(error);
    });
    locker.into_inner().expect("poisoned lock")
}

fn error_vs_interaction_number(config: NodeConfig) -> HashMap<usize, (usize, f64, f64)> {
    let mut interaction_to_errors: HashMap<usize, Vec<f64>> = HashMap::new();
    let h_sys = match config.sys_hamiltonian {
        HamiltonianType::HarmonicOscillator => harmonic_oscillator_hamiltonian(config.sys_dim),
        HamiltonianType::MarkedState => marked_state_hamiltonian(config.sys_dim),
    };
    let h_env = harmonic_oscillator_hamiltonian(config.env_dim);
    let rng_seed = get_rng_seed();
    let chacha = RandomInteractionGen::new(rng_seed, h_env.nrows() * h_sys.nrows());
    for interaction in 1..=config.num_interactions as usize {
        println!("interaction: {:}", interaction);
        let errors = multi_interaction_error_mc(
            config.num_samples,
            interaction,
            &h_sys,
            &h_env,
            config.sys_start_beta,
            config.env_beta,
            config.alpha,
            config.time,
            chacha.clone(),
        );
        interaction_to_errors.insert(interaction, errors);
    }
    interaction_to_errors
        .drain()
        .map(|(k, v)| {
            let processed = process_error_data(v);
            (k, processed)
        })
        .collect()
}

fn write_results(results: HashMap<usize, (usize, f64, f64)>, base_dir: String) {
    let s = serde_json::to_string(&results).expect("no serialization.");
    let mut filepath = base_dir.clone();
    let seed = get_rng_seed();
    filepath.push_str(&seed.to_string());
    std::fs::write(filepath, s).expect(&format!("Could not write output to file: {:}", base_dir));
}

fn get_base_dir() -> String {
    let args: Vec<String> = std::env::args().collect();
    let mut s = args[1].to_string();
    if s.ends_with('/') == false {
        s.push('/');
    }
    s
}
fn run_node() {
    let base_dir = get_base_dir();
    let config = NodeConfig::from_path(base_dir.clone());
    let outputs = error_vs_interaction_number(config);
    write_results(outputs, base_dir);
}

#[derive(Subcommand)]
pub enum Experiments {
    SingleShotTraceDistSweep,
    FixedPointSweep,
}

#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    experiment_type: Experiments,
    config: PathBuf,
    output: PathBuf,
    #[arg(short = 'l', long, value_name = "LABEL")]
    experiment_label: Option<String>,
}

fn main() {
    let start = Instant::now();
    let cli = Cli::parse();
    match cli.experiment_type {
        Experiments::SingleShotTraceDistSweep => {
            println!("single shot trace distance sweep");
            if cli.config.is_file() == false {
                println!("Could not find config file at: {:}", cli.config.display());
                return;
            }
            let conf_path = cli
                .config
                .to_str()
                .expect("Could not convert input path to string.")
                .to_string();
            let out_path = cli
                .output
                .to_str()
                .expect("Could not convert output path to string.")
                .to_string();
            let conf = TraceNormReductionConfig::from_json(conf_path);
            let results = conf.run();
            let out = TraceNormReductionOutput {
                label: cli.experiment_label.unwrap_or(String::new()),
                experiment_type: String::from("SingleShotTraceDistSweep"),
                num_samples: conf.num_samples,
                data: results,
                beta_env: conf.beta_env,
                time: conf.time,
            };
            out.write_json_to_file(out_path);
        }
        Experiments::FixedPointSweep => {
            println!("fixed point sweep!");
        }
    }
    
    let duration = start.elapsed();
    println!("took this many millis: {:}", duration.as_millis());
}

mod tests {
    use crate::{adjoint, i, partial_trace, zero, NodeConfig, RandomInteractionGen};
    use ndarray::{linalg::kron, prelude::*};
    use ndarray_linalg::{expm::expm, random_hermite, OperationNorm, Trace};
    use num_complex::{Complex64 as c64, ComplexFloat};

    #[test]
    fn test_random_interaction_gen() {
        let mut gen1 = RandomInteractionGen::new(1, 2);
        let mut gen2 = RandomInteractionGen::new(1, 2);
        assert_eq!(gen1.sample_gue(), gen2.sample_gue());
    }

    #[test]
    fn serialize_a_config() {
        let nc = NodeConfig {
            num_interactions: 100,
            num_samples: 1000,
            time: 100.,
            alpha: 0.01,
            sys_start_beta: 0.,
            env_beta: 1.,
            sys_hamiltonian: numerics::HamiltonianType::HarmonicOscillator,
            sys_dim: 10,
            env_dim: 2,
        };
        let nc_string = serde_json::to_string(&nc).expect("no serialization?");
        std::fs::write("/Users/matt/scratch/tsp/test_config/tsp.conf", nc_string)
            .expect("couldn't write");
    }

    #[test]
    fn deserialize_config() {
        let nc_path = "/Users/matt/scratch/tsp/test_config/tsp.conf".to_string();
        let nc = NodeConfig::from_path(nc_path);
        println!("retrieved nc: {:#?}", nc);
    }

    #[test]
    fn test_partial_trace() {
        let dim1 = 20;
        let dim2 = 20;

        let mut a: Array2<c64> = random_hermite(dim1);
        let a_adjoint = adjoint(&a);
        a = a.dot(&a_adjoint);
        let a_trace = a.trace().unwrap();
        a.mapv_inplace(|x| x / a_trace);

        let mut b: Array2<c64> = random_hermite(dim2);
        let b_adjoint = adjoint(&b);
        b = b.dot(&b_adjoint);
        let b_trace = b.trace().unwrap();
        b.mapv_inplace(|x| x / b_trace);

        let kron_prod = kron(&a, &b);
        let partraced = partial_trace(&kron_prod, a.nrows(), b.nrows());
        let diff_norm_per_epsilon: f64 = (&a - &partraced).opnorm_one().unwrap() / f64::EPSILON;
        assert!(diff_norm_per_epsilon < (dim1 as f64));
    }
}
