use std::sync::{Arc, Mutex};

use ndarray::{Array2, ShapeBuilder};
use ndarray_linalg::{c64, expm, QRSquare, Scalar, Trace};
use num_complex::ComplexFloat;
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, StandardNormal};
use serde::{Deserialize, Serialize};

pub mod fixed_points;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HamiltonianType {
    HarmonicOscillator,
    MarkedState,
}

pub struct RandomInteractionGen {
    rng: Arc<Mutex<ChaCha8Rng>>,
    dim: usize,
}

impl RandomInteractionGen {
    pub fn new(seed: u64, dim: usize) -> Self {
        RandomInteractionGen {
            rng: Arc::new(Mutex::new(ChaCha8Rng::seed_from_u64(seed))),
            dim: dim,
        }
    }

    pub fn sample_interaction(&self) -> Array2<c64> {
        let mut chacha = self.rng.lock().expect("couldn't get cha cha");
        let mut g = Array2::<c64>::zeros((self.dim, self.dim).f());
        for i in 0..self.dim {
            for j in 0..i {
                let x: c64 = chacha.gen();
                g[[i, j]] = x;
                g[[j, i]] = x.conj();
            }
            let y: c64 = chacha.gen();
            g[[i, i]] = y + y.conj();
        }
        g
    }
}

impl Clone for RandomInteractionGen {
    fn clone(&self) -> Self {
        Self {
            rng: self.rng.clone(),
            dim: self.dim.clone(),
        }
    }
}

pub fn perform_fixed_interaction_channel(
    h_tot: &Array2<c64>,
    interaction: &Array2<c64>,
    rho: &Array2<c64>,
    time: f64,
    dim_sys: usize,
) -> Array2<c64> {
    let x = c64::new(0., time) * (h_tot + interaction);
    let u = expm(&x).expect("Could not exponentiate");
    let u_dagger = adjoint(&u);
    let mut out = rho.dot(&u_dagger);
    out = u.dot(&out);
    partial_trace(&out, dim_sys, h_tot.nrows() / dim_sys)
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
    assert!(trace.im() < f64::EPSILON * a.nrows() as f64);
    // real part should be positive
    assert!(trace.re() > 0.);
    // If above assertions pass then this is fine.
    ComplexFloat::abs(trace.sqrt())
}

pub fn process_error_data(vals: Vec<f64>) -> (usize, f64, f64) {
    let samples = vals.len();
    let mean = vals.iter().sum::<f64>() / (samples as f64);
    let std =
        f64::sqrt(vals.iter().map(|x| f64::powi(x - mean, 2)).sum::<f64>() / (samples as f64));
    (samples, mean, std)
}

pub fn thermal_state(hamiltonian: &Array2<c64>, beta: f64) -> Array2<c64> {
    let scaled_h = hamiltonian * (c64::from_real(-1. * beta));
    let mut out = expm(&scaled_h).expect("we ballin");
    let partition_function = out.trace().unwrap();
    if ComplexFloat::abs(partition_function) < 1e-12 {
        println!("[thermal_state] encountered near zero partition function. You're gonna have a bad time.");
        panic!("see printed msg")
    }
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

pub fn sample_haar_unitary(dim: usize) -> Array2<c64> {
    let mut rng = thread_rng();
    let mut real_gauss: Vec<c64> = Vec::with_capacity(dim * dim);
    for _ in 0..(dim * dim) {
        real_gauss.push(c64::new(
            rng.sample::<f64, _>(StandardNormal) / f64::sqrt(2.),
            rng.sample::<f64, _>(StandardNormal) / f64::sqrt(2.),
        ));
    }
    let gauss_array = Array2::<c64>::from_shape_vec((dim, dim), real_gauss).unwrap();
    let (q, r) = gauss_array.qr_square().unwrap();
    let mut lambda = Array2::<c64>::zeros((dim, dim).f());
    for ix in 0..dim {
        lambda[[ix, ix]] = r[[ix, ix]] / r[[ix, ix]].norm();
    }
    q.dot(&lambda)
}

pub fn sample_perturbation_eigenvalues(dim: usize, variance: f64) -> Array2<c64> {
    let normal = Normal::new(0., variance.sqrt()).unwrap();
    let mut out = Array2::<c64>::zeros((dim, dim).f());
    for ix in 0..dim {
        out[[ix, ix]] = c64::from_real(normal.sample(&mut rand::thread_rng()));
    }
    out
}

pub fn sample_perturbation(dim: usize, variance: f64) -> Array2<c64> {
    let u = sample_haar_unitary(dim);
    let v = sample_perturbation_eigenvalues(dim, variance);
    let u_dagger = adjoint(&u);
    let out = v.dot(&u_dagger);
    u.dot(&out)
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
