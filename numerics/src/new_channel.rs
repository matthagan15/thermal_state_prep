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
}

impl IteratedChannel {
    pub fn simulate(&self) {
        match &self.gamma_strategy {
            GammaStrategy::Fixed(_) => todo!(),
            GammaStrategy::Probabilistic { distribution, num_samples } => todo!(),
            GammaStrategy::Grid { min, max, num_points } => todo!(),
        }
    }
}

