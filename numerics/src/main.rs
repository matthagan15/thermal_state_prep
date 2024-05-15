extern crate blas_src;
extern crate ndarray;
extern crate num_complex;

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, Subcommand};

use ndarray::prelude::*;

use ndarray_linalg::Scalar;

use numerics::channel::*;
use numerics::*;

#[derive(Subcommand)]
pub enum Experiments {
    SingleShot,
    FixedEpsilon,
    FixedNumSteps,
}

#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    experiment_type: Experiments,

    #[arg(short, long, value_name = "CONFIG_PATH")]
    config_path: PathBuf,

    #[arg(short, long, value_name = "RESULTS_PATH")]
    results_path: PathBuf,

    #[arg(short, long, value_name = "LABEL")]
    label: String,
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
    let gs = GammaStrategy::Fixed(1.0);
    println!("pretty gamma strategy below.");
    println!("{:}", serde_json::to_string_pretty(&gs).unwrap());
    let start = Instant::now();
    let cli = Cli::parse();
    match cli.experiment_type {
        Experiments::SingleShot => {
            single_shot_dist::run(&cli.config_path, &cli.results_path, cli.label.clone());
        }
        Experiments::FixedEpsilon => todo!(),
        Experiments::FixedNumSteps => todo!(),
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
