extern crate blas_src;
extern crate ndarray;
extern crate num_complex;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use clap::{Parser, Subcommand};
use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray_linalg::expm::expm;
use ndarray_linalg::random_hermite;
use ndarray_linalg::{OperationNorm, Scalar};
use num_complex::Complex64 as c64;
use num_complex::ComplexFloat;

use numerics::single_shot_dist::{TraceNormReductionConfig, TraceNormReductionOutput};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use numerics::channel::*;
use numerics::*;

#[derive(Subcommand)]
pub enum Experiments {
    SingleShotTraceDistSweep {
        #[arg(short, long, value_name = "CONFIG")]
        config: PathBuf,

        #[arg(short, long, value_name = "OUTPUT")]
        output: PathBuf,
    },
    FixedPointSweep,
}

#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    experiment_type: Experiments,
}

fn harmonic_oscillator_gaps(dim: usize) -> Vec<f64> {
    let mut ret = Vec::new();
    let h = harmonic_oscillator_hamiltonian(dim);
    for ix in 0..dim - 1 {
        for jx in ix + 1..dim {
            ret.push(Scalar::abs(h[[jx, jx]] - h[[ix, ix]]));
        }
    }
    ret
}

fn main() {
    let dim = 15;
    let h_sys = harmonic_oscillator_hamiltonian(dim);
    let gaps = harmonic_oscillator_gaps(dim);
    let gs = GammaStrategy::known(gaps, 30);
    let beta_e = 5.0;
    let alpha = 1e-2;
    let mut conf = CaptureRadiusConfig {
        label: String::from("Harmonic Oscillator (15 dim) high temp"),
        beta_e,
        // deltas: Array::logspace(10., 0., f64::log10(beta_e), 150).to_vec(),
        deltas: Array::linspace(0.0, beta_e, 150).to_vec(),
        alpha,
        time: 0.5 / alpha,
        gamma: gs,
        h_sys,
        num_samples: 250,
    };
    let output_path = Path::new("/Users/matt/repos/thermal_state_prep/numerics/tmp/capture_radius/harmonic_oscillator_high_temp.json");
    let outputs = config_to_results(conf);
    let serialized = serde_json::to_string(&outputs).expect("Could not serialize output.");
    std::fs::write(output_path, serialized).expect("Could not write results");
}

fn cli_main() {
    let start = Instant::now();
    let cli = Cli::parse();
    match cli.experiment_type {
        Experiments::SingleShotTraceDistSweep { config, output } => {
            println!("single shot trace distance sweep");
            if config.is_file() == false {
                println!("Could not find config file at: {:}", config.display());
                return;
            }
            let conf_path = config
                .to_str()
                .expect("Could not convert input path to string.")
                .to_string();
            let out_path = output
                .to_str()
                .expect("Could not convert output path to string.")
                .to_string();
            let conf = TraceNormReductionConfig::from_json(conf_path);
            let results = conf.run();
            let out = TraceNormReductionOutput {
                experiment_type: String::from("SingleShotTraceDistSweep"),
                num_samples: conf.num_samples,
                data: results,
                beta_env: conf.beta_env,
                time: conf.time,
            };
            out.write_json_to_file(out_path);
        }
        Experiments::FixedPointSweep => {
            println!("fixed point sweep!");
        }
    }

    let duration = start.elapsed();
    println!("took this many millis: {:}", duration.as_millis());
}

mod tests {
    use crate::{adjoint, i, interaction_generator::RandomInteractionGen, partial_trace, zero};
    use ndarray::{linalg::kron, prelude::*};
    use ndarray_linalg::{expm::expm, random_hermite, OperationNorm, Trace};
    use num_complex::{Complex64 as c64, ComplexFloat};

    #[test]
    fn test_random_interaction_gen() {
        let mut gen1 = RandomInteractionGen::new(1, 2);
        let mut gen2 = RandomInteractionGen::new(1, 2);
        assert_eq!(gen1.sample_gue(), gen2.sample_gue());
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
