use core::num;
use std::{
    cmp::min,
    env,
    ops::AddAssign,
    sync::{Arc, Mutex, RwLock}, collections::HashMap,
};

use ndarray::{array, linalg::kron, Array2, ShapeBuilder};
use ndarray_linalg::{expm, krylov::R, Scalar, Trace};
use num_complex::Complex64 as c64;
use rand::{thread_rng, Rng};
use rand_distr::{uniform::UniformFloat, Gamma, Uniform};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    adjoint, mean_and_std, partial_trace, schatten_2_distance, thermal_state, RandomInteractionGen,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterHandler {
    beta_start: f64,
    beta_stop: f64,
    beta_steps: usize,
    alpha_start: f64,
    alpha_diff: f64,
    // linear scale implies we just subtract the diff each time,
    // if false then the diff is multiplied.
    alpha_linear_scale: bool,
    alpha_steps: usize,
    gammas: GammaSampler,
    num_gamma_samples: usize,
}

impl ParameterHandler {
    /// Returns a sampled schedule for simulation of a channel. "Sampled" is
    /// used because gamma could be drawn randomly if it is not fixed.
    /// Returned values are (alpha, beta, gamma).
    pub fn sample_schedule(&self) -> Vec<(f64, f64, f64)> {
        let mut ret = Vec::new();
        let delta_beta = (self.beta_stop - self.beta_start).abs() / (self.beta_steps as f64);

        for ix in 0..self.beta_steps {
            let beta = self.beta_start + delta_beta * (ix as f64);
            for jx in 0..self.alpha_steps {
                let gammas = self.gammas.gen_samples(self.num_gamma_samples);
                let alpha = if self.alpha_linear_scale {
                    self.alpha_start - self.alpha_diff * (jx as f64)
                } else {
                    self.alpha_start * self.alpha_diff.powi(jx as i32)
                };
                for g in gammas {
                    ret.push((alpha, beta, g));
                }
            }
        }
        ret
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GammaSampler {
    Gaps {
        // Vec containing the eigenvalue differences
        // of the hamiltonian.
        spectrum_gaps: Vec<f64>,
        // Parameter controlling the tradeoff between uniform sampling vs.
        // sampling the known distribution. noise of 0 implies
        // that the distribution is known, anything else controls the width
        // of a uniformly sampled float that is added on to the sampled
        // known spectrum difference.
        noise_added: f64,
    },
    Grid {
        max: f64,
        num_pts: usize,
    },
    Fixed(f64),
}

impl GammaSampler {
    pub fn gen_samples(&self, num_samples: usize) -> Vec<f64> {
        let mut rng = thread_rng();
        let mut ret = Vec::with_capacity(num_samples);
        match self {
            GammaSampler::Gaps {
                spectrum_gaps,
                noise_added,
            } => {
                for _ in 0..num_samples {
                    let ix = rng.sample(Uniform::from(0..spectrum_gaps.len()));
                    let sampled_gap = spectrum_gaps[ix];
                    let noise = noise_added * (rng.gen::<f64>() - f64::min(0.5, sampled_gap));
                    ret.push(sampled_gap + noise);
                }
            }
            GammaSampler::Grid { max, num_pts } => {
                let delta = max / ((num_pts - 1) as f64);
                for _ in 0..num_samples {
                    let ix = rng.sample(Uniform::from(0..*num_pts));
                    ret.push((ix as f64) * delta);
                }
            }
            GammaSampler::Fixed(g) => {
                for _ in 0..num_samples {
                    ret.push(*g);
                }
            }
        }
        ret
    }
}

fn simulate_paramaters(phi: &mut Channel, param_handler: ParameterHandler) {
    let params = param_handler.sample_schedule();
    let mut beta_dist_pairs = Vec::with_capacity(params.len());
    phi.set_sys_to_thermal_state(0.0);
    let h_sys = phi.get_copy_of_h_sys();
    let mut prev_beta = 0.0;
    let mut target_sys = thermal_state(&h_sys, prev_beta);
    for (a, b, g) in params {
        if b != prev_beta {
            target_sys = thermal_state(&h_sys, b);
            prev_beta = b;
        }
        phi.set_env_to_gap(g);
        phi.set_env_to_thermal_state(b);
        phi.alpha = a;
        let statistic = |state: &Array2<c64>| schatten_2_distance(&target_sys, state);
        let out = phi.map(100, 1);
        let dist = schatten_2_distance(&out, &target_sys);
        beta_dist_pairs.push((b, dist));
        phi.set_sys_state(out);
    }
    println!("beta/dist pairs:");
    for ix in 0..beta_dist_pairs.len() {
        println!(
            "beta: {:}, dist: {:}",
            beta_dist_pairs[ix].0, beta_dist_pairs[ix].1
        );
    }
}

// TODO: Allow for changing the environment gap. Also allow for cooling schedules.
#[derive(Debug)]
pub struct Channel {
    h_sys: Array2<c64>,
    h_env: Array2<c64>,
    h_tot: Array2<c64>,
    gamma_sampler: GammaSampler,
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
        gamma_strategy: GammaSampler,
        alpha: f64,
        time: f64,
        rng: RandomInteractionGen,
    ) -> Self {
        let ds = system_hamiltonian.nrows();
        let de = 2;
        let gamma = gamma_strategy.gen_samples(1);
        let h_env = array![
            [0.0.into(), 0.0.into()],
            [0.0.into(), c64::from_real(*gamma.first().unwrap())]
        ];
        let h_tot = kron(&system_hamiltonian, &Array2::<c64>::eye(de))
            + kron(&Array2::<c64>::eye(ds), &h_env);
        let rho_sys = Array2::<c64>::eye(ds) / (c64::from_real(ds as f64));
        let mut rho_env = Array2::<c64>::zeros((de, de).f());
        rho_env[[0, 0]] = c64::from_real(1.0);
        Self {
            h_sys: system_hamiltonian,
            h_env,
            h_tot,
            gamma_sampler: gamma_strategy,
            dim_sys: ds,
            dim_env: de,
            rho_sys,
            rho_env,
            alpha,
            time,
            interaction_generator: rng,
        }
    }

    pub fn set_env_to_gap(&mut self, gap: f64) {
        let mut h_gap = Array2::<c64>::zeros((2, 2));
        h_gap[[1, 1]] = c64::from_real(gap);
        let h_tot = kron(&self.h_sys, &Array2::<c64>::eye(2))
            + kron(&Array2::<c64>::eye(self.dim_sys), &h_gap);
        self.dim_env = 2;
        self.h_env = h_gap;
        self.h_tot = h_tot;
        self.rho_env = Array2::<c64>::eye(self.dim_env) * 0.5;
        self.interaction_generator.dim = self.dim_env * self.dim_sys;
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

    pub fn set_sys_state(&mut self, state: Array2<c64>) {
        if state.shape() != self.h_sys.shape() {
            panic!("Expected same shape for state as system hamiltonian.")
        }
        let trace = state.trace().expect("just taking trace");
        self.rho_sys = state / trace;
    }

    pub fn get_copy_of_h_tot(&self) -> Array2<c64> {
        self.h_tot.clone()
    }
    pub fn get_copy_of_h_sys(&self) -> Array2<c64> {
        self.h_sys.clone()
    }

    /// Returns a Vec with the average statistic, the standard deviation of the statistic, and then the statistic of the average state.
    pub fn parameter_schedule_with_statistic<F>(
        &mut self,
        params: ParameterHandler,
        num_samples: usize,
        statistic: F,
    ) -> Vec<(f64, f64, f64)>
    where
        F: Fn(&Array2<c64>) -> f64 + Sync + Send,
    {
        let schedule = params.sample_schedule();
        let avg_states: HashMap<usize, Array2<c64>> = HashMap::with_capacity(schedule.len());
        let avg_states_locker = Arc::new(Mutex::new(avg_states));
        (0..num_samples).into_par_iter().for_each(|_| {
            let mut rho_sys = thermal_state(&self.h_sys, 0.0);
            for ix in 0..schedule.len() {
                let (a, b, g) = schedule[ix];
                let env_partition_function = 1. + (-1. * b * g).exp();
                let rho_env = array![
                    [c64::from_real(1. / env_partition_function), 0.0.into()],
                    [0.0.into(), c64::from_real((-1. * b * g).exp() / env_partition_function)]
                ];
                let rho_tot = kron(&rho_sys, &rho_env);
                let h_tot = kron(&self.h_sys, &Array2::<c64>::eye(2)) + kron(&Array2::<c64>::eye(self.dim_sys), &array![
                    [0.0.into(), 0.0.into()], [0.0.into(), c64::from_real(g)]
                ]);
                let interaction = self.interaction_generator.sample_gue() * a;
                let tot = (h_tot + interaction) * (-1. * self.time);
                let u = expm(&tot).expect("Could not compute time propagator.");
                let total_out = u.dot(&rho_tot.dot(&adjoint(&u)));
                rho_sys = partial_trace(&total_out, self.dim_sys, 2);
                let mut avgs = avg_states_locker.lock().expect("Could not get average states");
                let state = avgs.get_mut(&ix).expect("could not get average state");
                *state += &rho_sys;
            }
        });
        Vec::new()
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

    /// Computes the state of the environment after a single interaction
    pub fn env_map(&self, num_samples: usize) -> Array2<c64> {
        let tot = self.total_map(num_samples, 1);
        partial_trace(&tot, self.dim_env, self.dim_sys)
    }

    pub fn print_env(&self) {
        println!("{:?}", self.rho_env);
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
    use ndarray::{arr1, array, linalg::kron, Array2};
    use ndarray_linalg::{OperationNorm, Scalar};
    use num_complex::Complex64 as c64;

    use crate::{
        harmonic_oscillator_hamiltonian, perform_fixed_interaction_channel, schatten_2_distance,
        thermal_state, HamiltonianType, RandomInteractionGen,
    };

    use super::{simulate_paramaters, Channel, GammaSampler, ParameterHandler};

    #[test]
    fn test_cooling_schedule() {
        let ph = ParameterHandler {
            beta_start: 0.0,
            beta_stop: 1.0,
            beta_steps: 10,
            alpha_start: 0.01,
            alpha_diff: 0.9,
            alpha_linear_scale: false,
            alpha_steps: 10,
            gammas: GammaSampler::Grid {
                max: 3.0,
                num_pts: 20,
            },
            num_gamma_samples: 10,
        };
        let h_sys = Array2::<c64>::from_diag(&arr1(&[
            c64::from_real(0.0),
            c64::from_real(0.5),
            c64::from_real(0.75),
            c64::from_real(1.1),
            c64::from_real(1.5),
            c64::from_real(1.8),
            c64::from_real(2.0),
            c64::from_real(2.2),
            c64::from_real(2.7),
            c64::from_real(3.0),
        ]));
        let mut channel = Channel::new(
            h_sys,
            GammaSampler::Grid {
                max: 3.0,
                num_pts: 20,
            },
            0.01,
            100.,
            RandomInteractionGen::new(1, 20),
        );
        simulate_paramaters(&mut channel, ph);
    }
    #[test]
    fn test_estimators() {
        let h_sys = harmonic_oscillator_hamiltonian(10);
        let h_env = harmonic_oscillator_hamiltonian(2);
        let rho_sys = thermal_state(&h_sys, 0.75);
        let rng = RandomInteractionGen::new(1, 20);
        let gs = GammaSampler::Fixed(1.2);
        let mut phi = Channel::new(h_sys, gs, 0.001, 100., rng);
        phi.set_env_to_thermal_state(0.75);
        let distance_estimator = |matrix: &Array2<c64>| schatten_2_distance(matrix, &rho_sys);
        let out = phi.estimator_sys(distance_estimator, 1000, 100);
        println!("observed metrics: {:} +- {:}", out.0, out.1);
    }

    #[test]
    fn test_gamma_sampler() {
        let gs = GammaSampler::Gaps {
            spectrum_gaps: vec![0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.4],
            noise_added: 0.025,
        };
        dbg!(gs.gen_samples(100));
        let gs2 = GammaSampler::Grid {
            max: 1.,
            num_pts: 10,
        };
        dbg!(gs2.gen_samples(100));
    }
}
