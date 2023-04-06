use ndarray::{Array2, ShapeBuilder};
use ndarray_linalg::{c64, QRSquare, Scalar};
use rand::{thread_rng, Rng};
use rand_distr::{StandardNormal, Normal, Distribution};


pub enum HamiltonianType {
    HarmonicOscillator,
    MarkedState
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
