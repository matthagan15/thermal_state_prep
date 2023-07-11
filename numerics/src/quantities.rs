use std::cmp::min;

use ndarray::{linalg::kron, Array, Array2, ShapeBuilder};
use ndarray_linalg::{OperationNorm, Scalar, Trace};
use num_complex::Complex64 as c64;

use crate::{
    channel::Channel, harmonic_oscillator_hamiltonian, thermal_state, zero, RandomInteractionGen,
};

/// Returns transition probabilities along the diagonal
fn get_transition_probability(
    phi: &mut Channel,
    sys_start: usize,
    sys_end: usize,
    env_start: usize,
    env_end: usize,
    alpha: f64,
    time: f64,
) -> f64 {
    phi.set_env_state_to_energy_projector(env_start);
    phi.set_sys_to_energy_projector(sys_start);
    let output = phi.total_map(1000, 1);

    let mut tmp_sys = Array2::<c64>::zeros((phi.dim_sys, phi.dim_sys).f());
    let tgt_ix = min(sys_end, phi.dim_sys - 1);
    tmp_sys[[tgt_ix, tgt_ix]] = c64::from_real(1.);

    let mut tmp_env = Array2::<c64>::zeros((phi.dim_env, phi.dim_env).f());
    let tgt_ix = min(env_end, phi.dim_env - 1);
    tmp_env[[tgt_ix, tgt_ix]] = c64::from_real(1.);

    let tgt_out = kron(&tmp_sys, &tmp_env);
    tgt_out
        .dot(&output)
        .trace()
        .expect("couldn't take trace?")
        .re
}

fn analytic_transition_second_order(
    phi: &Channel,
    sys_start: usize,
    sys_end: usize,
    env_start: usize,
    env_end: usize,
    time: f64,
    alpha: f64,
) -> f64 {
    let h = phi.get_copy_of_h_tot();
    let ix_start = phi.convert_pair_indices_to_system(sys_start, env_start);
    let ix_end = phi.convert_pair_indices_to_system(sys_end, env_end);
    let lambda_start = h[[ix_start, ix_start]].re;
    let lambda_end = h[[ix_end, ix_end]].re;
    let mut tot = 0.0;
    if ix_start == ix_end {
        for ix in 0..h.nrows() {
            if ix == ix_start {
                continue;
            }
            if h[[ix, ix]].re == lambda_start {
                tot += time.pow(2.);
            } else {
                let delta = h[[ix, ix]].re - lambda_start;
                let num = 2. * (1. - f64::cos(delta * time));
                let den = delta.pow(2.);
                tot += num / den;
            }
        }
        tot *= -1. * alpha.pow(2.) / (h.nrows() as f64 + 1.);
        tot += 1.;
    } else if lambda_start == lambda_end {
        tot = (alpha * time).pow(2.) / (h.nrows() as f64 + 1.);
    } else {
        let delta = lambda_end - lambda_start;
        let num = 2. * alpha.pow(2.) * (1. - f64::cos(delta * time));
        let den = (h.nrows() as f64 + 1.) * delta.pow(2.);
        tot = num / den;
    }
    tot
}

fn compare_transitions(phi: &mut Channel, alpha: f64, time: f64) {
    let sys_start = 0;
    let env_start = 0;
    for sys_end in 0..phi.dim_sys {
        for env_end in 0..phi.dim_env {
            let analytic = analytic_transition_second_order(
                &phi, sys_start, sys_end, env_start, env_end, time, alpha,
            );
            let computed = get_transition_probability(
                phi, sys_start, sys_end, env_start, env_end, alpha, time,
            );
            println!(
                "({:}, {:}) -> ({:}, {:})",
                sys_start, env_start, sys_end, env_end
            );
            println!("computed = {:}", computed);
            println!("analytic = {:}", analytic);
            let diff_abs = (computed - analytic).abs();
            let diff_over_alpha_cubed = diff_abs / alpha.pow(3.);
            let diff_over_analytic = diff_abs.pow(2.) / analytic;
            println!("diff over analytic: {:}", diff_over_analytic);
            println!("diff over alpha cubed: {:}", diff_over_alpha_cubed);
            break;
        }
    }
}

pub fn analytic_second_order(
    h_tot: &Array2<c64>,
    rho: &Array2<c64>,
    time: f64,
    alpha: f64,
) -> Array2<c64> {
    let mut out = rho.clone();
    for i in 0..h_tot.nrows() {
        let mut acc = zero();
        for j in 0..h_tot.nrows() {
            if h_tot[[i, i]] == h_tot[[j, j]] {
                continue;
            }
            let diff = (h_tot[[i, i]] - h_tot[[j, j]]).re;
            let mut numerator = rho[[i, i]] * (1. + f64::cos(diff * time))
                - rho[[j, j]] * (1. - f64::cos(diff * time));
            numerator *= 2.0 * alpha * alpha / (h_tot.nrows().pow(2) as f64 - 1.);
            let denominator = diff.pow(2.);
            acc += numerator / denominator;
        }
        out[[i, i]] += acc;
    }
    out
}

mod test {
    use ndarray::{array, linalg::kron, Array2};
    use ndarray_linalg::OperationNorm;
    use num_complex::Complex64 as c64;

    use crate::{
        channel::Channel, harmonic_oscillator_hamiltonian, thermal_state, RandomInteractionGen,
    };

    use super::{analytic_second_order, compare_transitions};

    #[test]
    fn test_transitions() {
        let h_sys = harmonic_oscillator_hamiltonian(35);
        let h_env = harmonic_oscillator_hamiltonian(2);
        let rng = RandomInteractionGen::new(1, h_sys.nrows() * h_env.nrows());
        let mut phi = Channel::new(h_sys, h_env, 0.01, 100., rng);
        compare_transitions(&mut phi, 0.01, 100.);
    }

    #[test]
    fn test_analytic_second_order() {
        let h_sys = harmonic_oscillator_hamiltonian(15);
        let h_env = harmonic_oscillator_hamiltonian(2);
        let h_tot = kron(&h_sys, &Array2::<c64>::eye(2)) + kron(&Array2::<c64>::eye(15), &h_env);
        let rho_sys = thermal_state(&h_sys, 0.0);
        let rho_env = array![[1.0.into(), 0.0.into()], [0.0.into(), 0.0.into()]];
        let rho = kron(&rho_sys, &rho_env);
        let x = analytic_second_order(&h_tot, &rho, 1000., 0.001);
        println!("x norm: {:}", x.opnorm_one().unwrap());
    }

    #[test]
    fn test_kron_indexing() {
        let a = array![[1, 0, 0], [0, 2, 0], [0, 0, 3]];
        let b = array![[5, 0, 0], [0, 7, 0], [0, 0, 11]];
        let t = kron(&a, &b);
        for ix in 0..a.nrows() {
            for jx in 0..b.nrows() {
                println!("ix: {:}, jx: {:}", ix, jx);
                let first_guess = ix * b.nrows() + jx;
                let second_guess = jx * a.nrows() + ix;
                println!("first guess: {:}", t[[first_guess, first_guess]]);
                println!("second guess: {:}", t[[second_guess, second_guess]]);
                println!("reality: {:}", a[[ix, ix]] * b[[jx, jx]]);
            }
        }
    }
}
