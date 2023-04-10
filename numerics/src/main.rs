extern crate blas_src;
extern crate ndarray;
extern crate num_complex;

use std::collections::HashMap;
use std::fs;
use std::sync::mpsc::channel;
use std::sync::{RwLock, Mutex, Arc};
use std::time::{Duration, Instant};
use std::env;

use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray_linalg::expm::expm;
use ndarray_linalg::{random_hermite, random_hermite_using};
use ndarray_linalg::QRSquare;
use ndarray_linalg::Trace;
use ndarray_linalg::{OperationNorm, Scalar};
use num_complex::Complex64 as c64;
use num_complex::ComplexFloat;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Normal, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Serializer;

use numerics::*;

const MAX_INTERACTIONS: usize = 100000;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeConfig {
    num_interactions: u32,
    num_samples: usize,
    time: f64,
    alpha: f64,
    sys_start_beta: f64,
    env_beta: f64,
    // Path to stored hamiltonian
    sys_hamiltonian: HamiltonianType,
    sys_dim: usize,
    // currently only support harmonic oscillator env
    env_dim: usize,
    base_dir: String,
}

struct RandomInteractionGen {
    rng: Arc<Mutex<ChaCha8Rng>>,
    dim: usize,
}


impl RandomInteractionGen {
    fn new(seed: u64, dim: usize) -> Self {
        RandomInteractionGen { rng: Arc::new(Mutex::new(ChaCha8Rng::seed_from_u64(seed))) , dim: dim}
    }

    fn sample_interaction(&self) -> Array2<c64> {
        let mut chacha = self.rng.lock().expect("couldn't get cha cha");
        let mut g = Array2::<c64>::zeros((self.dim, self.dim).f());
        for i in 0..self.dim {
            for j in 0..i {
                let x: c64 = chacha.gen();
                g[[i,j]] = x;
                g[[j,i]] = x.conj();
            }
            let y: c64 = chacha.gen();
            g[[i,i]] = y + y.conj();
        }
        g
    }
}

impl Clone for RandomInteractionGen {
    fn clone(&self) -> Self {
        Self { rng: self.rng.clone(), dim: self.dim.clone() }
    }
}

fn read_config(config_path: String) -> NodeConfig {
    if let Ok(conf_string) = std::fs::read_to_string(config_path.clone()) {
        serde_json::from_str(&conf_string).expect("Config file could not be deserialized.")
    } else {
        panic!("No config file found at: {:}", config_path)
    }
}

fn one_shot_interaction(
    system_hamiltonian: &Array2<c64>,
    env_hamiltonian: &Array2<c64>,
    system_state: &Array2<c64>,
    env_state: &Array2<c64>,
    alpha: f64,
    time: f64,
) -> Array2<c64> {
    let sys_env_interaction_sample: Array2<c64> =
        (c64::from_real(alpha)) * random_hermite(system_state.nrows() * env_state.nrows());
    let sys_identity = Array2::<c64>::eye(system_hamiltonian.nrows());
    let env_identity = Array2::<c64>::eye(env_hamiltonian.nrows());

    let hamiltonian =
        kron(system_hamiltonian, &env_identity) + kron(&sys_identity, env_hamiltonian);
    let mut tot_hamiltonian = &sys_env_interaction_sample + &hamiltonian;
    tot_hamiltonian *= i() * time;

    let time_evolution_op = expm(&tot_hamiltonian).expect("we ballin");
    let time_evolution_op_adjoint = adjoint(&time_evolution_op);
    let mut out = kron(system_state, env_state);
    out = time_evolution_op.dot(&out);
    out = out.dot(&time_evolution_op_adjoint);
    partial_trace(&out, system_state.nrows(), env_state.nrows())
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
        let interaction_sample: Array2<c64> = (c64::from_real(alpha)) * rng.sample_interaction();
        let h_with_interaction =  c64::new(0., time)* (tot_hamiltonian + &interaction_sample);
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
    let h = kron(sys_hamiltonian, &Array2::<c64>::eye(env_hamiltonian.nrows())) + kron(&Array2::<c64>::eye(sys_hamiltonian.nrows()), env_hamiltonian);
    let rho_env = thermal_state(env_hamiltonian, env_beta);
    let rho_sys = thermal_state(sys_hamiltonian, sys_initial_beta);
    let sys_ideal = thermal_state(sys_hamiltonian, env_beta);
    let mut errors: Vec<f64> = Vec::with_capacity(num_samples);
    let mut locker = RwLock::new(errors);
    for interaction in 1..=num_interactions {
        (0..num_samples).into_par_iter().for_each(|_| {
            let rho_evolved_sample = multi_interaction(&h, &rho_env, &rho_sys, alpha, time, num_interactions, rng.clone());
            let error = schatten_2_distance(&rho_evolved_sample
                , &sys_ideal);
            let mut v = locker.write().expect("no locker");
            v.push(error);
        });
    }
    locker.into_inner().expect("poisoned lock")
}

/// Averages over one_shot_interaction
fn one_shot_mc(
    num_samples: usize,
    system_hamiltonian: &Array2<c64>,
    env_hamiltonian: &Array2<c64>,
    system_state: &Array2<c64>,
    env_state: &Array2<c64>,
    alpha: f64,
    time: f64,
) -> Array2<c64> {
    let out = Array2::<c64>::zeros((system_state.nrows(), system_state.ncols()).f());
    let locker = RwLock::new(out);
    (0..num_samples).into_par_iter().for_each(|_| {
        let sample = one_shot_interaction(
            system_hamiltonian,
            env_hamiltonian,
            system_state,
            env_state,
            alpha,
            time,
        );
        let mut x = locker.write().unwrap();
        for ix in 0..x.nrows() {
            for jx in 0..x.ncols() {
                x[[ix, jx]] += sample[[ix, jx]];
            }
        }
    });
    let mut out = locker.into_inner().unwrap();
    out.mapv_inplace(|x| x / (num_samples as f64));
    out
}

fn multi_interaction_mc(
    num_interactions: usize,
    num_samples: usize,
    system_hamiltonian: &Array2<c64>,
    env_hamiltonian: &Array2<c64>,
    system_start_beta: f64,
    env_beta: f64,
    alpha: f64,
    time: f64,
) -> Array2<c64> {
    let sys_start_state = thermal_state(system_hamiltonian, system_start_beta);
    let mut out = sys_start_state.clone();
    // Note this does not have to be recomputed each time as it is immutable
    let env_state = thermal_state(env_hamiltonian, env_beta);
    for i in 0..num_interactions {
        let interacted = one_shot_mc(
            num_samples,
            system_hamiltonian,
            env_hamiltonian,
            &out,
            &env_state,
            alpha,
            time,
        );
        out.assign(&interacted);
        // if i % (num_samples / 10) == 0 {
        println!("{:}% done.", (i as f64) / (num_interactions as f64));
        // }
    }
    out
}

fn find_interactions_needed_for_error(
    num_samples: usize,
    system_hamiltonian: &Array2<c64>,
    env_hamiltonian: &Array2<c64>,
    system_start_beta: f64,
    env_beta: f64,
    error: f64,
    alpha: f64,
    time: f64,
) -> usize {
    let mut sys_state = thermal_state(system_hamiltonian, system_start_beta);
    let system_target_state = thermal_state(system_hamiltonian, env_beta);
    let env_state = thermal_state(env_hamiltonian, env_beta);
    for ix in 0..MAX_INTERACTIONS {
        if ix % 50 == 0 {
            println!(
                "[find_interactions_needed_for_error] performing interaction: {:}",
                ix
            );
        }
        let out = one_shot_mc(
            num_samples,
            system_hamiltonian,
            env_hamiltonian,
            &sys_state,
            &env_state,
            alpha,
            time,
        );
        let distance_to_target = schatten_2_distance(&out, &system_target_state);
        // println!(
        //     "[find_interactions_needed_for_error] distance to target: {:}",
        //     distance_to_target
        // );
        if distance_to_target <= error {
            return ix;
        } else {
            sys_state.assign(&out);
        }
    }
    println!("[find_interactions_needed_for_error] did not converge, returning MAX_INTERACTIONS.");
    MAX_INTERACTIONS
}

fn test_minimum_interactions() {
    let alpha = 0.01;
    let t = 100.;
    let num_mc_samples = 500;
    let sys_dim = 20;
    let env_dim = 5;
    let h_sys = harmonic_oscillator_hamiltonian(sys_dim);
    let h_env = harmonic_oscillator_hamiltonian(env_dim);
    let env_beta = 1.;
    let betas = vec![1., 0.8, 0.6, 0.4, 0.2, 0.0];
    for beta in betas {
        let min_interactions = find_interactions_needed_for_error(
            num_mc_samples,
            &h_sys,
            &h_env,
            beta,
            env_beta,
            0.1,
            alpha,
            t,
        );
        println!("{:}", "*".repeat(75));
        println!("system start beta: {:}", beta);
        println!("interactions needed: {:}", min_interactions);
    }
}



/// Returns a hamiltonian with a highly degenerate spectrum. Has a single
/// ground state at energy = 0 and the remaining energies all at the
/// provided gap.
fn marked_state_hamiltonian(dim: usize, gap: f64) -> Array2<c64> {
    let mut out = Array2::<c64>::zeros((dim, dim).f());
    for ix in 1..dim {
        out[[ix, ix]] = gap.into();
    }
    out
}

fn harmonic_oscillator_alpha_and_time_grid() {
    let alphas = vec![5e-4, 1e-3, 5e-3, 1e-2];
    let times = vec![1e2];
    let mut results = Vec::new();
    for alpha in alphas.clone() {
        for time in times.clone() {
            println!("{:}", "*".repeat(100));
            println!( 
                "finding minimum interactions needed for alpha = {:}, time = {:}",
                alpha, time
            );
            results.push(find_interactions_needed_for_error(500, &&harmonic_oscillator_hamiltonian(10), &harmonic_oscillator_hamiltonian(5), 0.8, 1., 0.05, alpha, time));
        }
    }
    println!("alphas: {:?}", alphas);
    println!("results: {:?}", results)
}

fn marked_state_grid_gap_and_dim() {
    let gaps = vec![2.];
    let dims = vec![4, 8, 16];
    for gap in gaps {
        for dim in dims.clone() {
            println!("{:}", "*".repeat(100));
            println!(
                "finding minimum interactions needed for dim = {:}, gap = {:}",
                dim, gap
            );
            find_interactions_needed_for_error(
                500,
                &marked_state_hamiltonian(16, gap),
                &harmonic_oscillator_hamiltonian(dim),
                0.,
                1.,
                0.1,
                0.01,
                100.,
            );
        }
    }
}




/// Return the schatten-2 norm of the difference between the output of the channel
/// and the input state (going to use thermal state for input.) to be used in a finite
/// difference approximation scheme for the second derivative.
fn interaction_at_alpha(alpha: f64) -> f64 {
    let h_sys = harmonic_oscillator_hamiltonian(5);
    let h_env = harmonic_oscillator_hamiltonian(5);
    let beta = 1.;
    let t = 100.;
    let rho_sys = thermal_state(&h_sys, beta);
    let rho_env = thermal_state(&h_env, beta);
    let rho_tot = kron(&rho_sys, &rho_env);
    let channel_output = multi_interaction_mc(10, 20000, &h_sys, &h_env, beta, beta, alpha, t);
    // println!("channel_output shape: {:?}", channel_output.shape());
    // schatten_2_norm(&channel_output, &rho_tot)
    let diff = &channel_output - &rho_sys;
    diff.opnorm_fro().unwrap()
}

fn two_harmonic_oscillators(sys_dim: usize, env_dim: usize, sys_initial_beta: f64, env_beta: f64) {
    let alpha = 0.01;
    let t = 100.;
    let num_mc_samples = 500;
    let h_sys = harmonic_oscillator_hamiltonian(sys_dim);
    let h_env = harmonic_oscillator_hamiltonian(env_dim);
    let state_sys = thermal_state(&h_sys, sys_initial_beta);
    let state_env = thermal_state(&h_env, env_beta);
    let target = thermal_state(&h_sys, env_beta);
    let channel_output = one_shot_mc(
        num_mc_samples,
        &h_sys,
        &h_env,
        &state_sys,
        &state_env,
        alpha,
        t,
    );

    println!(
        "output frobenius distance to target: {:}",
        schatten_2_distance(&channel_output, &target)
    );
}

fn test_dimension_fix_beta() {
    let sys_dims = vec![5, 10, 15, 20, 25, 30];
    for dim in sys_dims {
        println!("dimension: {:}", dim);
        two_harmonic_oscillators(dim, 10, 0.5, 1.);
    }
}

fn test_beta_fix_dimension() {
    let sys_dim = 20;
    let betas = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    for beta in betas {
        println!("beta: {:}", beta);
        two_harmonic_oscillators(sys_dim, 10, beta, 1.);
    }
}

fn one_shot_mc_with_errors() {

}
fn error_vs_interaction_number(config: NodeConfig) {
    let mut interaction_to_errors: HashMap<usize, Vec<f64>> = HashMap::new();
    let h_sys = match config.sys_hamiltonian {
        HamiltonianType::HarmonicOscillator => {
            harmonic_oscillator_hamiltonian(config.sys_dim)
        },
        HamiltonianType::MarkedState => {
            marked_state_hamiltonian(config.sys_dim, 1.)
        }
    };
    let h_env = harmonic_oscillator_hamiltonian(config.env_dim);
    let rng_seed = get_rng_seed();
    let chacha = RandomInteractionGen::new(rng_seed, h_env.nrows() * h_sys.nrows());
    for interaction in 1..=config.num_interactions as usize {
        println!("interaction: {:}", interaction);
        let errors = multi_interaction_error_mc(config.num_samples, interaction, &h_sys, &h_env, config.sys_start_beta, config.env_beta, config.alpha, config.time, chacha.clone());
        interaction_to_errors.insert(interaction, errors);
    }
    let mut filepath = config.base_dir;
    filepath.push_str(&rng_seed.to_string());
    let out: HashMap<usize, (usize, f64, f64)> = interaction_to_errors.drain().map(|(k,v)| {
        let processed = process_error_data(v);
        (k, processed)
    }).collect();
    let s = serde_json::to_string(&out).expect("no serialization.");
    std::fs::write(filepath, s).expect("no writing");
}

fn get_conf_path() -> String {
    let args: Vec<String> = std::env::args().collect();
    let mut s = args[1].to_string();
    s.push_str("tsp.conf");
    s
}

fn run_node() {
    let conf_path = get_conf_path();
    let config = read_config(conf_path);
    error_vs_interaction_number(config)
}

fn main() {
    let start = Instant::now();
    run_node();
    let duration = start.elapsed();
    println!("took this many millis: {:}", duration.as_millis());
}

mod tests {
    use crate::{adjoint, i, partial_trace, sample_haar_unitary, zero, RandomInteractionGen, NodeConfig, read_config};
    use ndarray::{linalg::kron, prelude::*};
    use ndarray_linalg::{expm::expm, random_hermite, OperationNorm, Trace};
    use num_complex::{Complex64 as c64, ComplexFloat};

    #[test]
    fn test_random_interaction_gen() {
        let mut gen1 = RandomInteractionGen::new(1, 2);
        let mut gen2 = RandomInteractionGen::new(1, 2);
        assert_eq!(gen1.sample_interaction(), gen2.sample_interaction());
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
            base_dir: "/Users/matt/scratch/tsp/test_config/".to_string(),
        };
        let nc_string = serde_json::to_string(&nc).expect("no serialization?");
        std::fs::write("/Users/matt/scratch/tsp/test_config/tsp.conf", nc_string).expect("couldn't write");
    }

    #[test]
    fn deserialize_config() {
        let nc_path = "/Users/matt/scratch/tsp/test_config/tsp.conf".to_string();
        let nc = read_config(nc_path);
        println!("retrieved nc: {:#?}", nc);
    }

    #[test]
    fn test_haar_one_design() {
        let dim = 100;
        let rand_mat = random_hermite(dim);
        let rand_mat_conj = adjoint(&rand_mat);
        let psd = rand_mat.dot(&rand_mat_conj);
        let rho = psd.clone() / psd.trace().unwrap();
        let num_samps = 10000;
        let mut out = Array2::<c64>::zeros((dim, dim).f());
        for _ in 0..num_samps {
            let u = sample_haar_unitary(dim);
            let u_adj = adjoint(&u);
            let intermediat = rho.dot(&u_adj);
            out = out + u.dot(&intermediat);
        }
        out.mapv_inplace(|x| x / (num_samps as f64));
        for ix in 0..dim {
            out[[ix, ix]] -= 1. / (dim as f64);
        }
        println!("{:}", out);
        println!("diff norm: {:}", out.opnorm_one().unwrap());
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
