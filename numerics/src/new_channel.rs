use core::num;
use std::{
    cmp::min,
    collections::HashMap,
    env,
    ops::AddAssign,
    sync::{Arc, Mutex, RwLock}, path::{Path, PathBuf},
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

enum GammaStrategy {
    Fixed (f64),
    Probabilistic {
        distribution: HashMap<f64, f64>,
        num_samples: usize,
    },
    Grid {
        min: f64,
        max: f64,
        num_points: usize,
    },
    Known {
        differences: Vec<f64>,
    },
    /// Samples from a known difference with Gaussian noise
    /// with provided standard deviation.
    KnownWithNoise {
        differences: Vec<f64>,
        gaussian_noise_std: f64,
    }
}

struct SingleShotParameters {
    alpha: f64,
    beta: f64,
    time: f64,
}

struct IterationOutput {
    alpha: f64,
    beta: f64,
    time: f64,
    distance_mean: f64,
    distance_std: f64,
    distance_of_avg_state: f64,
    /// Average over samples of the variance
    /// due to sampling gammas as opposed to 
    /// deterministic gammas.
    gamma_variance: Option<f64>,
}

#[derive(Debug, Default)]
struct IteratedChannelOutput {
    distance_means: Vec<f64>,
    distance_stds: Vec<f64>,
    distance_of_avg_state: Vec<f64>,
}

impl IteratedChannelOutput {
    pub fn add(&mut self, mean: f64, std: f64, distance_of_avg_state: f64) {
        self.distance_means.push(mean);
        self.distance_stds.push(std);
        self.distance_of_avg_state.push(distance_of_avg_state);
    }
}

/// Assumes properly normalized, panics if it isn't and cannot generate a sample. assumes HashMap< value, probability > 
fn sample_from_distribution(prob_dist: &HashMap<f64, f64>) -> f64 {
    let mut acc = 0.0;
    let mut rng = thread_rng();
    let sampled_float: f64 = rng.gen();
    for (k, v) in prob_dist.into_iter() {
        if acc <= sampled_float && acc + v >= sampled_float {
            return *k;
        } else {
            acc += v;
        }
    }
    acc
}

pub struct IteratedChannel {
    h_sys: Array2<c64>,
    parameter_schedule: Vec<SingleShotParameters>,
    target_beta: f64,
    gamma_strategy: GammaStrategy,
    rng_seed: Option<u64>,
    interaction_gen: RandomInteractionGen,
}

impl IteratedChannel {
    pub fn simulate(&self, num_samples: usize) {
        let mut avg_states: HashMap<usize, Array2<c64>> = HashMap::with_capacity(self.parameter_schedule.len());
        let mut stat_map: HashMap<usize, Vec<f64>> = HashMap::with_capacity(self.parameter_schedule.len());
        for ix in 0..self.parameter_schedule.len() {
            avg_states.insert(ix, Array2::<c64>::zeros((self.h_sys.nrows(), self.h_sys.ncols()).f()));
            stat_map.insert(ix, Vec::new());
        }
        let avg_states_locker = Arc::new(Mutex::new(avg_states));
        let statistics_locker = Arc::new(Mutex::new(stat_map));

        let rho_ideal = thermal_state(&self.h_sys, self.target_beta);

        let gamma_std_map: HashMap<usize, f64> = HashMap::with_capacity(self.parameter_schedule.len());
        let gamma_std_locker = Arc::new(Mutex::new(gamma_std_map));

        (0..num_samples).into_par_iter().for_each(|_| {
            let mut rho_sys = Array2::<c64>::eye(self.h_sys.nrows()) / c64::from_real(self.h_sys.nrows() as f64);

            match &self.gamma_strategy {
                GammaStrategy::Fixed(g) => {
                    for ix in 0..self.parameter_schedule.len() {
                        let param = &self.parameter_schedule[ix];
                        let a = param.alpha;
                        let b = param.beta;
                        let t = param.time;

                        let env_partition_function = 1. + (-1. * b * g).exp();
                        let rho_env = array![
                            [c64::from_real(1. / env_partition_function), 0.0.into()],
                            [
                                0.0.into(),
                                c64::from_real((-1. * b * g).exp() / env_partition_function)
                            ]
                        ];
                        let rho_tot = kron(&rho_sys, &rho_env);
                        let h_env = kron(
                            &Array2::<c64>::eye(self.h_sys.nrows()),
                            &array![[0.0.into(), 0.0.into()], [0.0.into(), c64::from_real(*g)]],
                        );
                        let h_tot = kron(&self.h_sys, &Array2::<c64>::eye(2)) + &h_env;

                        let interaction = self.interaction_gen.sample_gue() * a;
                        let tot = (h_tot + interaction) * c64::new(0.0, -1. * t);
                        let u = expm(&tot).expect("Could not compute time propagator");
                        let total_out = u.dot(&rho_tot.dot(&adjoint(&u)));
                        rho_sys = partial_trace(&total_out, self.h_sys.nrows(), 2);

                        let mut avgs = avg_states_locker.lock().expect("Could not get average state map");
                        let state = avgs.get_mut(&ix).expect("could not get avg state");
                        *state += &rho_sys;
                        drop(avgs);

                        let sample_dist = schatten_2_distance(&rho_sys, &rho_ideal);
                        let mut stat_lock = statistics_locker.lock().expect("could not get stat lock");
                        let v = stat_lock.get_mut(&ix).expect("could not get stats records");
                        v.push(sample_dist);
                    }
                },
                GammaStrategy::Probabilistic { distribution, num_samples } => todo!(),
                GammaStrategy::Grid { min, max, num_points } => todo!(),
                GammaStrategy::Known { differences } => todo!(),
                GammaStrategy::KnownWithNoise { differences, gaussian_noise_std } => todo!(),
            }
        });
        let mut ret = Vec::new();
        let mut avg_states_lock = avg_states_locker
            .lock()
            .expect("could not get avg_states lock");
        let statistics_lock = statistics_locker
            .lock()
            .expect("coud not lock statistics_locker");
        for ix in 0..self.parameter_schedule.len() {
            let avg_state = avg_states_lock.get_mut(&ix).expect("no index");
            *avg_state /= c64::from_real(num_samples as f64);
            let statistic_of_avg = schatten_2_distance(avg_state, &rho_ideal);
            let stats = statistics_lock.get(&ix).unwrap();
            let (mean, std) = mean_and_std(stats);
            ret.push((mean, std, statistic_of_avg));
        }
        dbg!(ret);
        
    }
}

mod tests {
    use crate::{harmonic_oscillator_hamiltonian, RandomInteractionGen};

    use super::{SingleShotParameters, IteratedChannel, GammaStrategy};

    #[test]
    fn test_new_channel_simulate() {
        let h_sys = harmonic_oscillator_hamiltonian(5);
        let num_interactions = 150;
        let beta_e = 1.;
        let mut params = Vec::with_capacity(num_interactions);
        let mut alpha_t_reduction = 0.75_f64;
        let mut running_alpha = 0.01;
        let mut running_t = 100.0;
        for ix in 0..num_interactions {
            params.push(SingleShotParameters {
                alpha: running_alpha,
                beta: beta_e,
                time: running_t
            });
            running_alpha *= alpha_t_reduction.sqrt();
            running_t /= alpha_t_reduction.sqrt();
        }
        let gamma_strategy = GammaStrategy::Fixed(1.0);
        let int_gen = RandomInteractionGen::new(1, h_sys.nrows() * 2);
        let phi = IteratedChannel {
            h_sys,
            parameter_schedule: params,
            target_beta: beta_e,
            gamma_strategy,
            rng_seed: None,
            interaction_gen: int_gen,
        };
        phi.simulate(100);
    }
}