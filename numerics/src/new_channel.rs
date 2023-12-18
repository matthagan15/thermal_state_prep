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

struct SingleShotParameters {
    alpha: f64,
    beta: f64,
    gamma: f64,
    time: f64,
}

struct IterationOutput {
    alpha: f64,
    beta: f64,
    gamma: f64,
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

pub struct IteratedChannel {
    h_sys: Array2<c64>,
    parameter_schedule: Vec<SingleShotParameters>,
    log_state: bool,
    output_file_path: PathBuf,
    target_beta: f64,
    rng_seed: Option<u64>,
}

fn simulate_iterated_channel(phi: IteratedChannel) -> IteratedChannelOutput {
    let mut ret = IteratedChannelOutput::default();
    
    ret
}

/// Designed to answer questions "how many iterations until..."
pub struct StoppedProcessChannel {
    h_sys: Array2<c64>,
}