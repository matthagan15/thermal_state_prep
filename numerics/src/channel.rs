use core::num;
use std::{
    env,
    ops::AddAssign,
    sync::{Arc, Mutex}, cmp::min,
};

use ndarray::{linalg::kron, Array2, ShapeBuilder};
use ndarray_linalg::{expm, krylov::R, Scalar};
use num_complex::Complex64 as c64;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::{adjoint, partial_trace, thermal_state, RandomInteractionGen};

pub struct Channel {
    h_sys: Array2<c64>,
    h_env: Array2<c64>,
    h_tot: Array2<c64>,
    pub dim_sys: usize,
    pub dim_env: usize,
    rho_env: Array2<c64>,
    interaction_generator: RandomInteractionGen,
}

impl Channel {
    /// Construct a new channel to be used. If environment_beta given
    /// is `f64::MAX` then the ground state will be used.
    pub fn new(
        system_hamiltonain: Array2<c64>,
        environment_hamiltonian: Array2<c64>,
        environment_beta: f64,
        rng: RandomInteractionGen,
    ) -> Self {
        let ds = system_hamiltonain.nrows();
        let de = environment_hamiltonian.nrows();
        let h_tot = kron(&system_hamiltonain, &Array2::<c64>::eye(de))
            + kron(&Array2::<c64>::eye(ds), &environment_hamiltonian);
        let r = if environment_beta == f64::MAX {
            let mut x = Array2::<c64>::zeros((de, de).f());
            x[[0, 0]] = c64::from_real(1.);
            x
        } else {
            thermal_state(&environment_hamiltonian, environment_beta)
        };
        Self {
            h_sys: system_hamiltonain,
            h_env: environment_hamiltonian,
            h_tot,
            dim_sys: ds,
            dim_env: de,
            rho_env: r,
            interaction_generator: rng,
        }
    }

    /// Set the state of the environment to an energy eigenstate projector, aka
    /// |i><i| . The index is 0 being the ground state and dim_env - 1. Sets 
    /// the state to the highest energy eigenstate if it is out of bounds.
    pub fn set_env_state_to_energy_projector(&mut self, index: usize) {
        self.rho_env.mapv_inplace(|x| x * 0.);
        let ix = min(index, self.dim_env - 1);
        self.rho_env[[ix, ix]] = c64::from_real(1.);
    }

    pub fn set_env_to_thermal_state(&mut self, beta: f64) {
        self.rho_env = thermal_state(&self.h_env, beta);
    }

    pub fn get_copy_of_h_tot(&self) -> Array2<c64> {
        self.h_tot.clone()
    }

    /// operates under the assumption that the kronecker product is 
    /// system \otimes environment.
    pub fn convert_pair_indices_to_system(&self, sys_ix: usize, env_ix: usize) -> usize {
        sys_ix * self.dim_env + env_ix
    }

    /// Performs a single application of the channel returning the reduced
    /// density matrix of the system.
    pub fn map(&self, rho_sys: &Array2<c64>, alpha: f64, time: f64) -> Array2<c64> {
        partial_trace(
            &self.total_map(rho_sys, alpha, time),
            self.dim_sys,
            self.dim_env,
        )
    }

    /// Performs a single application of the channel, returning the density
    /// matrix of the total system + environment.
    pub fn total_map(&self, rho_sys: &Array2<c64>, alpha: f64, time: f64) -> Array2<c64> {
        let g = self.interaction_generator.sample_gue();
        self.total_map_fixed_interaction(rho_sys, alpha, time, &g)
    }

    pub fn total_map_monte_carlo_avg(
        &self,
        rho_sys: &Array2<c64>,
        alpha: f64,
        time: f64,
        num_samples: usize,
        num_interactions: usize,
    ) -> Array2<c64> {
        let a =
            Array2::<c64>::zeros((self.dim_env * self.dim_sys, self.dim_env * self.dim_sys).f());
        let locker = Arc::new(Mutex::new(a));
        (0..num_samples).into_par_iter().for_each(|_| {
            let sample = self.total_map_k_times(rho_sys, alpha, time, num_interactions);
            let mut final_out = locker.lock().expect("could not lock output holder.");
            final_out.scaled_add(c64::from_real(1. / num_samples as f64), &sample);
        });
        let guard = locker
            .lock()
            .expect("POISONED LOCK in total_map_monte_carlo_avg");
        guard.clone()
    }

    pub fn map_monte_carlo_avg(
        &self,
        rho_sys: &Array2<c64>,
        alpha: f64,
        time: f64,
        num_samples: usize,
        num_interactions: usize,
    ) -> Array2<c64> {
        partial_trace(
            &self.total_map_monte_carlo_avg(rho_sys, alpha, time, num_samples, num_interactions),
            self.dim_sys,
            self.dim_env,
        )
    }

    /// Perform k applications of the channel. More efficient than repeated calls.
    pub fn total_map_k_times(
        &self,
        rho_sys: &Array2<c64>,
        alpha: f64,
        time: f64,
        num_interactions: usize,
    ) -> Array2<c64> {
        let mut sampled_interactions = Vec::with_capacity(num_interactions);
        for _ in 0..num_interactions {
            sampled_interactions.push(self.interaction_generator.sample_gue());
        }
        let mut rho_tot = kron(rho_sys, &self.rho_env);
        let mut rho_s = rho_sys.clone();
        let t = c64::new(0., -time);
        let a = c64::new(alpha, 0.);
        for k in 0..num_interactions {
            if k > 0 {
                rho_tot = kron(&rho_s, &self.rho_env);
            }
            let tot = t * (a * sampled_interactions.pop().unwrap() + &self.h_tot);
            let u = expm(&tot).expect("Could not exponentiate.");
            let u_dagger = adjoint(&u);
            rho_tot = rho_tot.dot(&u_dagger);
            rho_tot = u.dot(&rho_tot);
            if k < num_interactions - 1 {
                rho_s = partial_trace(&rho_tot, self.dim_sys, self.dim_env);
            }
        }
        rho_tot
    }

    fn total_map_fixed_interaction(
        &self,
        rho_sys: &Array2<c64>,
        alpha: f64,
        time: f64,
        interaction: &Array2<c64>,
    ) -> Array2<c64> {
        let rho = kron(rho_sys, &self.rho_env);
        let tot = c64::new(0.0, time) * (c64::from_real(alpha) * interaction + &self.h_tot);
        let u = expm(&tot).expect("Could not exponentiate");
        let u_dagger = adjoint(&u);
        let mut out = rho.dot(&u_dagger);
        out = u.dot(&out);
        out
    }

    pub fn map_k_times(
        &self,
        rho_sys: &Array2<c64>,
        alpha: f64,
        time: f64,
        num_interactions: usize,
    ) -> Array2<c64> {
        partial_trace(
            &self.total_map_k_times(rho_sys, alpha, time, num_interactions),
            self.dim_sys,
            self.dim_env,
        )
    }
}

mod test {
    use ndarray::{array, linalg::kron, Array2};
    use ndarray_linalg::{OperationNorm, Scalar};
    use num_complex::Complex64 as c64;

    use crate::{
        harmonic_oscillator_hamiltonian, perform_fixed_interaction_channel, thermal_state,
        HamiltonianType, RandomInteractionGen,
    };

    use super::Channel;

    #[test]
    fn compare_channel_with_known() {
        let r1 = RandomInteractionGen::new(1, 40);
        let r2 = RandomInteractionGen::new(1, 40);

        let h1 = r1.sample_gue();
        let h2 = r2.sample_gue();
        let d = h1 - h2;
        println!(
            "difference in first sampled interaction: {:}",
            d.opnorm_one().unwrap()
        );

        let h_sys = harmonic_oscillator_hamiltonian(20);
        let h_env = harmonic_oscillator_hamiltonian(2);
        let h_tot = kron(&h_sys, &Array2::<c64>::eye(h_env.nrows()))
            + kron(&Array2::<c64>::eye(h_sys.nrows()), &h_env);
        let b_env = 1.;
        let alpha = 0.001;
        let time = 1000.;
        let phi = Channel::new(h_sys.clone(), h_env.clone(), b_env, r1);
        let mut diffs = Vec::new();
        let rho_sys = thermal_state(&h_sys, 0.5);
        let rho_env = thermal_state(&h_env, b_env);
        let rho_tot = kron(&rho_sys, &rho_env);
        for _ in 0..100 {
            let out_phi = phi.map(&rho_sys, alpha, time);
            let g = c64::from_real(alpha) * r2.sample_gue();
            let fn_out =
                perform_fixed_interaction_channel(&h_tot, &g, &rho_tot, time, h_sys.nrows());
            let diff = (out_phi - fn_out).opnorm_one().unwrap();
            diffs.push(diff);
        }
        let avg = diffs.into_iter().sum::<f64>() / 100.;
        println!("average diff norm: {:}", avg);
    }

    #[test]
    fn test_monte_carlo_avg() {
        let rn = RandomInteractionGen::new(1, 4);
        let phi = Channel::new(
            harmonic_oscillator_hamiltonian(2),
            harmonic_oscillator_hamiltonian(2),
            100.,
            rn,
        );
        println!(
            "output state: {:}",
            phi.map_monte_carlo_avg(
                &array![[c64::from_real(0.5), 0.0.into()], [0.0.into(), 0.5.into()]],
                0.01,
                100.,
                1000,
                500
            )
        );
    }
}
