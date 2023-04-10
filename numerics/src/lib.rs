use ndarray::{Array2, ShapeBuilder};
use ndarray_linalg::{c64, QRSquare, Scalar, Trace, expm};
use num_complex::ComplexFloat;
use rand::{thread_rng, Rng};
use rand_distr::{StandardNormal, Normal, Distribution};
use serde::{Deserialize, Serialize};


#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HamiltonianType {
    HarmonicOscillator,
    MarkedState
}

pub fn get_rng_seed() -> u64 {
    let var_data = std::env::var("NODE_RNG_SEED").expect("couldn't get rng seed from environment.");
    var_data.parse().expect("couldn't parse environment rng seed to u64.")
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
    let std = f64::sqrt(
        vals.iter().map(|x| f64::powi(x - mean, 2)).sum::<f64>() / (samples as f64)
    );
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
