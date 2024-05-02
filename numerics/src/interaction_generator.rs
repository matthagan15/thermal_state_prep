use std::sync::{Arc, Mutex};

use ndarray::{Array2, ShapeBuilder};
use ndarray_linalg::{c64, QRSquare, Scalar};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Normal, StandardNormal};

use crate::adjoint;

#[derive(Debug)]
pub struct RandomInteractionGen {
    rng: Arc<Mutex<ChaCha8Rng>>,
    pub dim: usize,
}

impl RandomInteractionGen {
    pub fn new(seed: u64, dim: usize) -> Self {
        RandomInteractionGen {
            rng: Arc::new(Mutex::new(ChaCha8Rng::seed_from_u64(seed))),
            dim,
        }
    }

    pub fn sample_gue(&self) -> Array2<c64> {
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

    pub fn sample_multiple_gue(&self, num_samples: usize) -> Vec<Array2<c64>> {
        let mut ret = Vec::with_capacity(num_samples);
        let mut chacha = self.rng.lock().expect("couldn't get cha cha");
        for _ in 0..num_samples {
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
            ret.push(g);
        }
        ret
    }

    /// Samples a haar random unitary.
    pub fn sample_haar_unitary(&self) -> Array2<c64> {
        let mut rng = self.rng.lock().expect("Couldn't lock rng.");
        let dim = self.dim;
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

    /// Samples a matrix with Haar random eigenvectors and
    /// i.i.d gaussian eigenvalues.
    pub fn sample_iid_interaction(&self) -> Array2<c64> {
        let mut rng = self.rng.lock().expect("Couldn't lock rng.");
        let dim = self.dim;
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
        let u = q.dot(&lambda);
        let u_dagger = adjoint(&u);
        let normal = Normal::new(0., 1.).unwrap();
        let mut out = Array2::<c64>::zeros((self.dim, self.dim).f());
        for ix in 0..dim {
            out[[ix, ix]] = c64::from_real(rng.sample(normal));
        }
        out = out.dot(&u_dagger);
        u.dot(&out)
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
