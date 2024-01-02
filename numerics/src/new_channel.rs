use core::num;
use std::{
    cmp::min,
    collections::HashMap,
    env,
    ops::AddAssign,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
};

use ndarray::{array, linalg::kron, Array, Array2, ShapeBuilder};
use ndarray_linalg::{expm, krylov::R, Scalar, Trace};
use num_complex::Complex64 as c64;
use rand::{thread_rng, Rng};
use rand_distr::{uniform::UniformFloat, Gamma, Uniform};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    adjoint, mean_and_std, partial_trace, schatten_2_distance, thermal_state, RandomInteractionGen,
};

#[derive(Debug, Clone)]
enum GammaStrategy {
    Fixed(f64),
    Iterative(Vec<f64>),
    Probabilistic {
        gamma_prob_pairs: Vec<(f64, f64)>,
        num_samples: usize,
    },
}

impl GammaStrategy {
    /// Samples from a known difference with a uniform noise over the specified
    /// width. Also the number of grid points to sample from the noise is needed.
    /// Samples from the differences uniformly.
    pub fn known_with_noise(differences: Vec<f64>, noise_width: f64, num_points_noise: usize) -> Self {
        let mut pairs = Vec::new();
        let prob = 1. / ((differences.len() * num_points_noise) as f64);
        for diff in differences {
            let points = Array::linspace(diff - (noise_width / 2.0), diff + (noise_width / 2.0), num_points_noise).to_vec();
            for point in points {
                pairs.push((point, prob));
            }
        }
        GammaStrategy::Probabilistic { gamma_prob_pairs: pairs, num_samples: 100}
    }

    /// Constructs an iterative gamma strategy that loops over the specified
    /// grid.
    pub fn grid_iterative(min: f64, max: f64, num_points: usize) -> Self {
        let points = Array::linspace(min, max, num_points).to_vec();
        GammaStrategy::Iterative(points)
    }

    /// Constructs a probabilistic gamma strategy that samples from the
    /// specified grid. Essentially a uniform distribution.
    pub fn grid_probabilistic(min: f64, max: f64, num_points: usize) -> Self {
        let prob = 1. / (num_points as f64);
        let points = Array::linspace(min, max, num_points).to_vec();
        GammaStrategy::Probabilistic { gamma_prob_pairs: points.into_iter().map(|x| (x, prob)).collect(), num_samples: 100 }
    }
}
struct SingleShotParameters {
    alpha: f64,
    beta: f64,
    time: f64,
}


/// Assumes properly normalized, panics if it isn't and cannot generate a sample. assumes `prob_dist` is formatted as `Vec<(value, prob)>`
fn sample_from_distribution(prob_dist: &Vec<(f64, f64)>) -> f64 {
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

pub fn build_rho_env(beta: f64, gamma: f64) -> Array2<c64> {
    let env_partition_function = 1. + (-1. * beta * gamma).exp();
    array![
        [c64::from_real(1. / env_partition_function), 0.0.into()],
        [
            0.0.into(),
            c64::from_real((-1. * beta * gamma).exp() / env_partition_function)
        ]
    ]
}
struct TimeVsEpsilonOut {
    sim_parameters: SingleShotParameters,
    schatten_2_norm_to_ideal: f64,
    total_sim_time: f64,
}

fn epsilon_vs_time_scaling(
    h_sys: &Array2<c64>,
    gamma_strategy: GammaStrategy,
) -> Vec<TimeVsEpsilonOut> {
    let mut ret = Vec::new();
    let mut beta_e = 1.0_f64;
    let mut alpha_t_reduction_factor = f64::square(0.75);
    let min_ep: f64 = -2.0;
    let num_points = 100;
    let eps = Array::logspace(10.0, 0.0, min_ep, num_points).to_vec();
    for ep in eps {
        let mut phi = IteratedChannel {
            h_sys: h_sys.clone(),
            parameter_schedule: vec![SingleShotParameters {
                alpha: 0.1,
                beta: 0.0,
                time: 10.,
            }],
            target_beta: 1.0,
            gamma_strategy: gamma_strategy.clone(),
            rng_seed: None,
            interaction_gen: RandomInteractionGen::new(1, h_sys.nrows() * 2),
        };
    }
    ret
}

pub struct ExperimentConstants {
    target_beta: f64,
    alpha_t_reduction: f64,
}

/// This is supposed to capture the variables we change
pub enum ExperimentType {
    EpsilonVsTime (Vec<(f64, f64)>),
    BetaVsTime (Vec<(f64, f64)>),
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
    /// Creates a fresh bath and evolves the provided rho_sys
    /// through a single application of the channel.
    pub fn simulate_sample_with_params(
        &self,
        alpha: f64,
        beta: f64,
        gamma: f64,
        time: f64,
        rho_sys: &mut Array2<c64>,
    ) {
        let rho_env = build_rho_env(beta, gamma);
        let rho_tot = kron(&rho_sys, &rho_env);
        let h_env = kron(
            &Array2::<c64>::eye(self.h_sys.nrows()),
            &array![
                [0.0.into(), 0.0.into()],
                [0.0.into(), c64::from_real(gamma)]
            ],
        );
        let h_tot = kron(&self.h_sys, &Array2::<c64>::eye(2)) + &h_env;

        let interaction = self.interaction_gen.sample_gue() * alpha;
        let tot = (h_tot + interaction) * c64::new(0.0, -1. * time);
        let u = expm(&tot).expect("Could not compute time propagator");
        let total_out = u.dot(&rho_tot.dot(&adjoint(&u)));
        *rho_sys = partial_trace(&total_out, self.h_sys.nrows(), 2);
    }

    pub fn simulate(&self, num_samples: usize) {
        let mut avg_states: HashMap<usize, Array2<c64>> =
            HashMap::with_capacity(self.parameter_schedule.len());
        let mut stat_map: HashMap<usize, Vec<f64>> =
            HashMap::with_capacity(self.parameter_schedule.len());
        let mut gamma_std_map: HashMap<usize, Vec<f64>> =
            HashMap::with_capacity(self.parameter_schedule.len());
        for ix in 0..self.parameter_schedule.len() {
            let h_sys_shape = (self.h_sys.nrows(), self.h_sys.ncols()).f();
            avg_states.insert(ix, Array2::<c64>::zeros(h_sys_shape));
            stat_map.insert(ix, Vec::new());
            gamma_std_map.insert(ix, Vec::new());
        }
        let avg_states_locker = Arc::new(Mutex::new(avg_states));
        let statistics_locker = Arc::new(Mutex::new(stat_map));
        let gamma_std_locker = Arc::new(Mutex::new(gamma_std_map));

        let rho_ideal = thermal_state(&self.h_sys, self.target_beta);

        (0..num_samples).into_par_iter().for_each(|_| {
            let mut rho_sys =
                Array2::<c64>::eye(self.h_sys.nrows()) / c64::from_real(self.h_sys.nrows() as f64);

            for ix in 0..self.parameter_schedule.len() {
                let param = &self.parameter_schedule[ix];
                let a = param.alpha;
                let b = param.beta;
                let t = param.time;
                match &self.gamma_strategy {
                    GammaStrategy::Fixed(g) => {
                        self.simulate_sample_with_params(a, b, *g, t, &mut rho_sys);
                    },
                    GammaStrategy::Iterative(gammas) => {
                        for g in gammas {
                            self.simulate_sample_with_params(a, b, *g, t, &mut rho_sys);
                        }
                    },
                    GammaStrategy::Probabilistic {
                        gamma_prob_pairs,
                        num_samples,
                    } => todo!(),
                }
                let mut avgs = avg_states_locker
                    .lock()
                    .expect("Could not get average state map");
                let state = avgs.get_mut(&ix).expect("could not get avg state");
                *state += &rho_sys;
                drop(avgs);

                let sample_dist = schatten_2_distance(&rho_sys, &rho_ideal);
                let mut stat_lock = statistics_locker.lock().expect("could not get stat lock");
                let v = stat_lock.get_mut(&ix).expect("could not get stats records");
                v.push(sample_dist);
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
    use ndarray::Array;

    use crate::{harmonic_oscillator_hamiltonian, RandomInteractionGen, mean_and_std};

    use super::{GammaStrategy, IteratedChannel, SingleShotParameters, sample_from_distribution};


    #[test]
    fn test_distribution_sampler() {
        let num_points = 10;
        let num_samples = 1000;
        let grid = Array::linspace(0., 1.0_f64, num_points).to_vec();
        let easy_dist = grid.clone().into_iter().map(|x| (x, 1.0_f64 / num_points as f64)).collect();
        let samps = (0..num_samples).map(|_| sample_from_distribution(&easy_dist)).collect();
        println!("mean + std: {:?}", mean_and_std(&samps));
    }

    #[test]
    fn test_new_channel_simulate() {
        let h_sys = harmonic_oscillator_hamiltonian(5);
        let num_interactions = 150;
        let beta_e = 1.;
        let mut params = Vec::with_capacity(num_interactions);
        let alpha_t_reduction = 0.75_f64;
        let mut running_alpha = 0.01;
        let mut running_t = 100.0;
        for _ in 0..num_interactions {
            params.push(SingleShotParameters {
                alpha: running_alpha,
                beta: beta_e,
                time: running_t,
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
