extern crate blas_src;
extern crate ndarray;
extern crate num_complex;

use std::collections::HashMap;
use std::fs;
use std::sync::mpsc::channel;
use std::sync::RwLock;
use std::time::{Duration, Instant};
use std::env;

use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray_linalg::expm::expm;
use ndarray_linalg::random_hermite;
use ndarray_linalg::QRSquare;
use ndarray_linalg::Trace;
use ndarray_linalg::{OperationNorm, Scalar};
use num_complex::Complex64 as c64;
use num_complex::ComplexFloat;
use rand::prelude::*;
use rand_distr::{Normal, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use numerics::*;

const MAX_INTERACTIONS: usize = 100000;

struct NodeConfig {
    rng_seed: usize,
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
        let distance_to_target = schatten_2_norm(&out, &system_target_state);
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

/// Computes Schatten-2 Norm, AKA frobenius error between two
/// operators
fn schatten_2_norm(a: &Array2<c64>, b: &Array2<c64>) -> f64 {
    let diff = a - b;
    let diff_adjoint = adjoint(&diff);
    let psd = diff.dot(&diff_adjoint);
    let trace = psd.trace().unwrap();
    // imaginary part should be very small
    assert!(trace.im() < f64::EPSILON * a.nrows() as f64);
    // real part should be positive
    assert!(trace.re() > 0.);
    // If above assertions pass then this is fine.
    ComplexFloat::abs(trace.sqrt())
}

fn thermal_state(hamiltonian: &Array2<c64>, beta: f64) -> Array2<c64> {
    let scaled_h = hamiltonian * (c64::from_real(-1. * beta));
    let mut out = expm(&scaled_h).expect("we ballin");
    let partition_function = out.trace().unwrap();
    if ComplexFloat::abs(partition_function) < 1e-12 {
        println!("[thermal_state] encountered near zero partition function. You're gonna have a bad time.");
        panic!("see printed msg")
    }
    out.mapv_inplace(|x| x / partition_function);
    out
}

fn harmonic_oscillator_hamiltonian(dim: usize) -> Array2<c64> {
    let mut out = Array2::<c64>::zeros((dim, dim).f());
    for ix in 0..dim {
        out[[ix, ix]] = c64::new(0.5 + ix as f64, 0.);
    }
    out
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

/// Performs the partial trace over dim2, yeilding a dim1 x dim1 Array2 object. For example,
/// if matrix = kron(A, B), then this will trace out over the B dimension. there is probably a
/// more efficient way of doing this but  I'm not sure how to at the moment.
fn partial_trace(matrix: &Array2<c64>, dim1: usize, dim2: usize) -> Array2<c64> {
    assert!(matrix.is_square());
    assert_eq!(matrix.nrows(), dim1 * dim2);
    let mut out = Array2::<c64>::zeros((dim1, dim1));
    for row_ix in 0..dim1 {
        for col_ix in 0..dim1 {
            let mut tot = zero();
            for row_jx in 0..dim2 {
                tot += matrix[[dim2 * row_ix + row_jx, dim2 * col_ix + row_jx]];
            }
            out[[row_ix, col_ix]] = tot;
        }
    }
    out
}

fn zero() -> c64 {
    c64::new(0., 0.)
}
fn i() -> c64 {
    c64::new(0., 1.)
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
        schatten_2_norm(&channel_output, &target)
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
    let mut interaction_number_to_error = HashMap::new();
    let h_sys = match config.sys_hamiltonian {
        HamiltonianType::HarmonicOscillator => {
            harmonic_oscillator_hamiltonian(config.sys_dim)
        },
        HamiltonianType::MarkedState => {
            marked_state_hamiltonian(config.sys_dim, 1.)
        }
    };
    let h_env = harmonic_oscillator_hamiltonian(config.env_dim);
    let mut rho_sys = thermal_state(&h_sys, config.sys_start_beta);
    let rho_env = thermal_state(&h_env, config.env_beta);
    for ix in 1..=1000 {
        let channel_output = one_shot_mc(
            config.num_samples, 
            &h_sys,
            &h_env,
            &rho_sys,
            &rho_env,
            config.alpha,
            config.time
        );
    }
}

fn main() {
    println!("environment: {:?}", env::var("RNG_SEED_NUMBER"));
    let start = Instant::now();
    println!("beta test");
    // println!("{:}", "^".repeat(100));
    // test_beta_fix_dimension();
    // println!("dimension test");
    // println!("{:}", "^".repeat(100));
    // test_dimension_fix_beta();
    // println!("min interactions test");
    // println!("{:}", "^".repeat(100));
    // test_minimum_interactions();
    // println!("grid test");
    // println!("{:}", "^".repeat(100));
    // marked_state_grid_gap_and_dim();

    println!("{:}", "^".repeat(100));
    harmonic_oscillator_alpha_and_time_grid();
    let duration = start.elapsed();
    println!("took this many millis: {:}", duration.as_millis());
}

// millis unoptimized : 48760
// millis release     : 2714
// reduction 18x

mod tests {
    use crate::{adjoint, i, partial_trace, sample_haar_unitary, zero};
    use ndarray::{linalg::kron, prelude::*};
    use ndarray_linalg::{expm::expm, random_hermite, OperationNorm, Trace};
    use num_complex::{Complex64 as c64, ComplexFloat};

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
