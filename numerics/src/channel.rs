use core::num;
use std::{
    cmp::min,
    env,
    ops::AddAssign,
    sync::{Arc, Mutex},
};

use ndarray::{linalg::kron, Array2, ShapeBuilder};
use ndarray_linalg::{expm, krylov::R, Scalar};
use num_complex::Complex64 as c64;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::{adjoint, mean_and_std, partial_trace, thermal_state, RandomInteractionGen};

#[derive(Debug)]
pub struct Channel {
    h_sys: Array2<c64>,
    h_env: Array2<c64>,
    h_tot: Array2<c64>,
    pub dim_sys: usize,
    pub dim_env: usize,
    rho_sys: Array2<c64>,
    rho_env: Array2<c64>,
    alpha: f64,
    time: f64,
    interaction_generator: RandomInteractionGen,
}

impl Channel {
    /// Construct a new channel to be used. Defaults the system
    /// to the maximally mixed state and the environment to the
    /// ground state.
    pub fn new(
        system_hamiltonian: Array2<c64>,
        environment_hamiltonian: Array2<c64>,
        alpha: f64,
        time: f64,
        rng: RandomInteractionGen,
    ) -> Self {
        let ds = system_hamiltonian.nrows();
        let de = environment_hamiltonian.nrows();
        let h_tot = kron(&system_hamiltonian, &Array2::<c64>::eye(de))
            + kron(&Array2::<c64>::eye(ds), &environment_hamiltonian);
        let rho_sys = Array2::<c64>::eye(ds) / (c64::from_real(ds as f64));
        let mut rho_env = Array2::<c64>::zeros((de, de).f());
        rho_env[[0, 0]] = c64::from_real(1.0);
        Self {
            h_sys: system_hamiltonian,
            h_env: environment_hamiltonian,
            h_tot,
            dim_sys: ds,
            dim_env: de,
            rho_sys: rho_sys,
            rho_env: rho_env,
            alpha,
            time,
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

    pub fn set_sys_to_energy_projector(&mut self, index: usize) {
        self.rho_sys.mapv_inplace(|x| x * 0.);
        let ix = min(index, self.dim_sys - 1);
        self.rho_sys[[ix, ix]] = c64::from_real(1.);
    }

    pub fn set_sys_to_thermal_state(&mut self, beta: f64) {
        self.rho_sys = thermal_state(&self.h_sys, beta);
    }

    pub fn get_copy_of_h_tot(&self) -> Array2<c64> {
        self.h_tot.clone()
    }

    /// operates under the assumption that the kronecker product is
    /// system \otimes environment.
    pub fn convert_pair_indices_to_system(&self, sys_ix: usize, env_ix: usize) -> usize {
        sys_ix * self.dim_env + env_ix
    }

    /// Map the stored system input state to the system output state using
    /// the specified number of samples and interactions.
    pub fn map(&self, num_samples: usize, num_interactions: usize) -> Array2<c64> {
        let tot = self.total_map(num_samples, num_interactions);
        partial_trace(&tot, self.dim_sys, self.dim_env)
    }

    pub fn total_map(&self, num_samples: usize, num_interactions: usize) -> Array2<c64> {
        let a =
            Array2::<c64>::zeros((self.dim_env * self.dim_sys, self.dim_env * self.dim_sys).f());
        let locker = Arc::new(Mutex::new(a));
        (0..num_samples).into_par_iter().for_each(|_| {
            let sample = self.sample_of_k_interactions_tot(num_interactions);
            let mut final_out = locker.lock().expect("could not lock output holder.");
            final_out.scaled_add(c64::from_real(1. / num_samples as f64), &sample);
        });
        let guard = locker
            .lock()
            .expect("POISONED LOCK in total_map_monte_carlo_avg");
        guard.clone()
    }
    /// Perform k applications of the channel. More efficient than repeated calls.
    fn sample_of_k_interactions_tot(&self, num_interactions: usize) -> Array2<c64> {
        let mut sampled_interactions = self
            .interaction_generator
            .sample_multiple_gue(num_interactions);
        let mut rho_tot = kron(&self.rho_sys, &self.rho_env);
        let mut rho_s = self.rho_sys.clone();
        let t = c64::new(0., -self.time);
        let a = c64::new(self.alpha, 0.);
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

    fn sample_of_k_interactions_system(&self, num_interactions: usize) -> Array2<c64> {
        let out = self.sample_of_k_interactions_tot(num_interactions);
        partial_trace(&out, self.dim_sys, self.dim_env)
    }

    pub fn estimator_sys_env<F>(
        &self,
        metric: F,
        num_samples: usize,
        num_interactions: usize,
    ) -> (f64, f64)
    where
        F: Fn(&Array2<c64>) -> f64 + Sync + Send,
    {
        let locker = Arc::new(Mutex::new(Vec::<f64>::new()));
        (0..num_samples).into_par_iter().for_each(|_| {
            let sample = self.sample_of_k_interactions_tot(num_interactions);
            let mut results = locker.lock().expect("could not lock output holder.");
            results.push(metric(&sample));
        });
        let lock = Arc::try_unwrap(locker).expect("Poisoned lock in total_estimate_mean_and_var");
        mean_and_std(lock.into_inner().expect("mutex machine broke."))
    }

    pub fn estimator_sys<F>(
        &self,
        metric: F,
        num_samples: usize,
        num_interactions: usize,
    ) -> (f64, f64)
    where
        F: Fn(&Array2<c64>) -> f64 + Sync + Send,
    {
        let locker = Arc::new(Mutex::new(Vec::<f64>::new()));
        (0..num_samples).into_par_iter().for_each(|_| {
            let sample = self.sample_of_k_interactions_system(num_interactions);
            let mut results = locker.lock().expect("could not lock output holder.");
            results.push(metric(&sample));
        });
        let lock = Arc::try_unwrap(locker).expect("Poisoned lock in total_estimate_mean_and_var");
        mean_and_std(lock.into_inner().expect("mutex machine broke."))
    }
}

mod test {
    use ndarray::{array, linalg::kron, Array2};
    use ndarray_linalg::{OperationNorm, Scalar};
    use num_complex::Complex64 as c64;

    use crate::{
        harmonic_oscillator_hamiltonian, perform_fixed_interaction_channel, schatten_2_distance,
        thermal_state, HamiltonianType, RandomInteractionGen,
    };

    use super::Channel;

    #[test]
    fn test_estimators() {
        let h_sys = harmonic_oscillator_hamiltonian(10);
        let h_env = harmonic_oscillator_hamiltonian(2);
        let rho_sys = thermal_state(&h_sys, 0.75);
        let rng = RandomInteractionGen::new(1, 20);
        let mut phi = Channel::new(h_sys, h_env, 0.001, 100., rng);
        phi.set_env_to_thermal_state(0.75);
        let distance_estimator = |matrix: &Array2<c64>| schatten_2_distance(matrix, &rho_sys);
        let out = phi.estimator_sys(distance_estimator, 1000, 100);
        println!("observed metrics: {:} +- {:}", out.0, out.1);
    }
}
