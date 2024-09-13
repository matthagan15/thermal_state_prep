use core::num;
use std::sync::{Arc, Mutex};

use ndarray::{Array2, ShapeBuilder};
use ndarray_linalg::{c64, expm, EigVals, QRSquare, Scalar, Trace};
use num_complex::ComplexFloat;
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, StandardNormal};
use serde::{Deserialize, Serialize};

pub mod channel;
pub mod fixed_epsilon;
pub mod fixed_num_steps;
pub mod interaction_generator;
pub mod single_shot_dist;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HamiltonianType {
    HarmonicOscillator,
    MarkedState,
    Log,
    Sqrt,
    Square,
}

impl HamiltonianType {
    pub fn as_ndarray(&self, dim: usize) -> Array2<c64> {
        let mut out = Array2::<c64>::zeros((dim, dim).f());
        for ix in 0..dim {
            out[[ix, ix]] = match self {
                HamiltonianType::HarmonicOscillator => c64::new(0.5 + ix as f64, 0.),
                HamiltonianType::MarkedState => {
                    if ix == 0 {
                        c64::new(1., 0.)
                    } else {
                        c64::new(0., 0.)
                    }
                }
                HamiltonianType::Log => {
                    let x = c64::new(ix as f64, 0.);
                    x.log(std::f64::consts::E)
                }
                HamiltonianType::Sqrt => c64::new((ix as f64).sqrt(), 0.),
                HamiltonianType::Square => c64::new((ix * ix) as f64, 0.),
            };
        }
        out
    }
}

impl std::str::FromStr for HamiltonianType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match &s.to_lowercase()[..] {
            "harmonic" => Ok(HamiltonianType::HarmonicOscillator),
            "marked_state" => Ok(HamiltonianType::MarkedState),
            "log" => Ok(HamiltonianType::Log),
            "sqrt" => Ok(HamiltonianType::Sqrt),
            "square" => Ok(HamiltonianType::Square),
            _ => Err(()),
        }
    }
}

pub fn harmonic_oscillator_gaps(dim: usize) -> Vec<f64> {
    let mut ret = Vec::new();
    let h = harmonic_oscillator_hamiltonian(dim);
    for ix in 0..dim - 1 {
        for jx in ix + 1..dim {
            ret.push(Scalar::abs(h[[jx, jx]] - h[[ix, ix]]));
        }
    }
    ret
}

pub fn generate_floats(start: f64, stop: f64, num_steps: usize, logspace: bool) -> Vec<f64> {
    if logspace {
        ndarray::Array::logspace(10.0, start.log10(), stop.log10(), num_steps).to_vec()
    } else {
        ndarray::Array::linspace(start, stop, num_steps).to_vec()
    }
}

/// Returns a hamiltonian with a highly degenerate spectrum. Has a single
/// ground state at energy = 0 and the remaining energies all at the
/// provided gap.
pub fn marked_state_hamiltonian(dim: usize) -> Array2<c64> {
    let mut out = Array2::<c64>::zeros((dim, dim).f());
    for ix in 1..dim {
        out[[ix, ix]] = c64::from_real(1.);
    }
    out
}

/// Computes Schatten-2 Norm, AKA frobenius error between two
/// operators
pub fn schatten_2_distance(a: &Array2<c64>, b: &Array2<c64>) -> f64 {
    let diff = a - b;
    let diff_adjoint = adjoint(&diff);
    let psd = diff.dot(&diff_adjoint);
    let trace = psd.trace().unwrap();
    // imaginary part should be very small

    if trace.im() > f64::EPSILON * a.nrows() as f64 {
        println!("WARNING: you are going to have a bad time. imaginary trace");
    }
    // real part should be positive
    if trace.re() < 0. {
        println!("WARNING: negative trace?");
    }
    // If above assertions pass then this is fine.
    ComplexFloat::abs(trace.sqrt())
}

pub fn trace_distance(a: &Array2<c64>, b: &Array2<c64>) -> f64 {
    let diff = a - b;
    let eigs = diff
        .eigvals()
        .expect("Could not compute eigenvalues for trace distance.");
    eigs.into_iter().map(|x| ComplexFloat::abs(x)).sum()
}

pub fn process_error_data(vals: Vec<f64>) -> (usize, f64, f64) {
    let samples = vals.len();
    let mean = vals.iter().sum::<f64>() / (samples as f64);
    let std =
        f64::sqrt(vals.iter().map(|x| f64::powi(x - mean, 2)).sum::<f64>() / (samples as f64));
    (samples, mean, std)
}

/// Computes the mean and standard deviation, returning
/// (mean, std).
pub fn mean_and_std(vals: &Vec<f64>) -> (f64, f64) {
    let samples = vals.len();
    let mean = vals.iter().sum::<f64>() / (samples as f64);
    let std =
        f64::sqrt(vals.iter().map(|x| f64::powi(x - mean, 2)).sum::<f64>() / (samples as f64));
    (mean, std)
}

pub fn thermal_state(hamiltonian: &Array2<c64>, beta: f64) -> Array2<c64> {
    let scaled_h = hamiltonian * (c64::from_real(-1. * beta));
    let x = scaled_h.dot(&scaled_h);
    let mut out = expm(&scaled_h).expect("we ballin");
    let partition_function = out.trace().unwrap();
    // if ComplexFloat::abs(partition_function) < 1e-12 {
    //     println!("[thermal_state] encountered near zero partition function. You're gonna have a bad time.");
    //     panic!("see printed msg")
    // }
    out.mapv_inplace(|x| x / partition_function);
    out
}

pub fn harmonic_oscillator_hamiltonian(dim: usize) -> Array2<c64> {
    let mut out = Array2::<c64>::zeros((dim, dim).f());
    for ix in 0..dim {
        out[[ix, ix]] = c64::new(0.5 + ix as f64, 0.);
    }
    out
}

/// Performs the partial trace over dim2, yeilding a dim1 x dim1 Array2 object. For example,
/// if matrix = kron(A, B), then this will trace out over the B dimension. there is probably a
/// more efficient way of doing this but  I'm not sure how to at the moment.
pub fn partial_trace(matrix: &Array2<c64>, dim1: usize, dim2: usize) -> Array2<c64> {
    assert!(matrix.is_square());
    assert_eq!(matrix.nrows(), dim1 * dim2);
    let mut out = Array2::<c64>::zeros((dim1, dim1));
    for row_ix in 0..dim1 {
        for col_ix in 0..dim1 {
            let mut tot = zero();
            for row_jx in 0..dim2 {
                tot += matrix[[dim2 * row_ix + row_jx, dim2 * col_ix + row_jx]];
            }
            out[[row_ix, col_ix]] = tot;
        }
    }
    out
}

pub fn zero() -> c64 {
    c64::new(0., 0.)
}
pub fn i() -> c64 {
    c64::new(0., 1.)
}

// Cannot transpose in place? so return a copy.
pub fn adjoint(matrix: &Array2<c64>) -> Array2<c64> {
    let mut out = matrix.t().into_owned();
    out.mapv_inplace(|x| x.conj());
    out
}

pub fn get_rng_seed() -> u64 {
    let args: Vec<String> = std::env::args().collect();
    args[2]
        .parse()
        .expect("could not parse input parameter for rng_seed as u64")
}

#[cfg(test)]
mod test {
    use ndarray::{array, Array2};
    use ndarray_linalg::OperationNorm;
    use num_complex::Complex64 as c64;

    use crate::{adjoint, interaction_generator::RandomInteractionGen, trace_distance};

    #[test]
    fn test_haar_unitary() {
        let chacha = RandomInteractionGen::new(1, 10);
        let u = chacha.sample_haar_unitary();
        let u_dagger = adjoint(&u);
        let diff = u.dot(&u_dagger) - Array2::<c64>::eye(10);
        println!(
            "diff magnitude per epsilon: {:}",
            diff.opnorm_one().unwrap() / f64::EPSILON
        );
    }

    #[test]
    fn small_trace_dist() {
        let rho = array![
            [c64::new(1.0, 0.), c64::new(0.0, 0.0)],
            [c64::new(0.0, 0.0), c64::new(1.0, 0.0)]
        ];
        let sigma = array![
            [c64::new(0., 0.), c64::new(0.0, 0.0)],
            [c64::new(0.0, 0.0), c64::new(0.9, 0.0)]
        ];
        dbg!(trace_distance(&rho, &sigma));
    }
}
