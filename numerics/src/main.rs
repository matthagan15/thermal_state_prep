extern crate blas_src;
extern crate ndarray;
extern crate num_complex;

use std::sync::RwLock;
use std::time::{Duration, Instant};

use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray_linalg::Scalar;
use ndarray_linalg::expm::expm;
use ndarray_linalg::random_hermite;
use ndarray_linalg::Trace;
use num_complex::Complex64 as c64;
use num_complex::ComplexFloat;
use rayon::prelude::*;

fn one_shot_interaction(
    system_hamiltonian: &Array2<c64>,
    env_hamiltonian: &Array2<c64>,
    system_state: &Array2<c64>,
    env_state: &Array2<c64>,
    alpha: f64,
    beta: f64,
    time: f64,
) -> Array2<c64> {
    let g: Array2<c64> = (c64::from_real(alpha)) * random_hermite(system_state.nrows() * env_state.nrows());
    let mut h = kron(system_hamiltonian, env_hamiltonian);
    let mut tot_h = &g + &h;
    tot_h *= i() * time;
    let (time_evolution_op, _) = expm(&tot_h);
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
    beta: f64,
    time: f64,
) -> Array2<c64> {
    let mut out = Array2::<c64>::zeros((system_state.nrows(), system_state.ncols()).f());
    let mut locker = RwLock::new(out);
    (0..num_samples).into_par_iter().for_each(|_|{
        let mut x = locker.write().unwrap();
        let sample =  one_shot_interaction(system_hamiltonian, env_hamiltonian, system_state, env_state, alpha, beta, time);
        for ix in 0..x.nrows() {
            for jx in 0..x.ncols() {
                x[[ix, jx]] += sample[[ix, jx]];
            }
        }
    });
    // for _ in 0..num_samples {
    //     out += &one_shot_interaction(system_hamiltonian, env_hamiltonian, system_state, env_state, alpha, beta, time);
    // }
    let mut out = locker.into_inner().unwrap();
    out.mapv_inplace(|x| x / (num_samples as f64));
    out
}

fn multi_interaction_mc(
    num_interactions: usize,
    num_samples: usize,
    system_hamiltonian: &Array2<c64>,
    env_hamiltonian: &Array2<c64>,
    system_state: &Array2<c64>,
    env_state: &Array2<c64>,
    alpha: f64,
    beta: f64,
    time: f64,
) -> Array2<c64> {
    let mut out = system_state.clone();
    for i in 0..num_interactions {
        let interacted = one_shot_mc(num_samples, system_hamiltonian, env_hamiltonian, &out, env_state, alpha, beta, time);
        out.assign(&interacted);
        // if i % (num_samples / 10) == 0 {
        println!("{:.2}% done.", (i as f64) / (num_interactions as f64));
        // }
    }
    out
}

/// Computes Schatten-2 Norm, AKA frobenius error between two
/// operators
fn schatten_2_norm(a: &Array2<c64>, b: &Array2<c64>) -> f64 {
    let diff = a - b;
    let diff_adjoint = adjoint(&diff);
    let psd = diff.dot(&diff_adjoint);
    let trace = psd.trace().unwrap();
    // imaginary part should be very small
    assert!( trace.im() < f64::EPSILON * a.nrows() as f64);
    // real part should be positive
    assert!( trace.re() > 0.);
    // If above assertions pass then this is fine.
    ComplexFloat::abs(trace.sqrt())
}

fn thermal_state(hamiltonian: &Array2<c64>, beta: f64) -> Array2<c64> {
    let scaled_h = hamiltonian * (beta + zero());
    let (mut out, _) = expm(&scaled_h);
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

// Cannot transpose in place? so return a copy.
fn adjoint(matrix: &Array2<c64>) -> Array2<c64> {
    let mut out = matrix.t().into_owned();
    out.mapv_inplace(|x| x.conj());
    out
}

fn main() {
    let start = Instant::now();
    let dim_sys = 8;
    let dim_env = 2;
    let beta = 1.;
    let alpha = 0.01;
    let t = 100.;
    let num_interactions = 1000;
    let num_mc_samples = 100;
    let h_sys = harmonic_oscillator_hamiltonian(dim_sys);
    let h_env = harmonic_oscillator_hamiltonian(dim_env);
    let state_sys = Array2::<c64>::eye(dim_sys) / (c64::from_real(dim_sys as f64));
    let state_env = thermal_state(&h_env, beta);
    let target = thermal_state(&h_sys, beta);
    let guess = thermal_state(&h_sys, beta * 2.);
    let rho_one = one_shot_mc(num_mc_samples, &h_sys, &h_env, &state_sys, &state_env, alpha, beta, t);
    let rho_final = multi_interaction_mc(num_interactions, num_mc_samples, &h_sys, &h_env, &state_sys, &state_env, alpha, beta, t);

    // println!("rho_one:\n{:}", rho_one);
    // println!("rho_ten:\n{:}", rho_final);
    println!("initial fro error w/target: {:}", schatten_2_norm(&state_sys, &target));
    println!("rho_one fro error w/ target: {:}", schatten_2_norm(&rho_one, &target));
    println!("rho_ten fro error w/ target: {:}", schatten_2_norm(&rho_final, &target));
    println!("rho_ten fro error w/ guess: {:}", schatten_2_norm(&rho_final, &guess));
    let duration = start.elapsed();
    println!("took this many millis: {:}", duration.as_millis());
}

// fn main() {
//     let start = Instant::now();
//     let num_samples = 100;
//     let dim = 1000;
//     let mut results = Vec::new();
//     for _ in 0..num_samples {
//         let m:Array2<c64> = random_hermite(dim);
//         let (_,d) = expm(&m);
//         results.push(d);
//     }
//     println!("ms per expm: {:}", start.elapsed().as_millis() / num_samples);
//     println!("results:{:}", results.iter().sum::<usize>() as u128 / num_samples);
// }

// millis unoptimized : 48760
// millis release     : 2714
// reduction 18x 

mod tests {
    use crate::{adjoint, i, partial_trace, zero};
    use ndarray::{linalg::kron, prelude::*};
    use ndarray_linalg::{expm::expm, random_hermite, OperationNorm, Trace};
    use num_complex::{Complex64 as c64, ComplexFloat};

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
