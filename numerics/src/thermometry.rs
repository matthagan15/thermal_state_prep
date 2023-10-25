use ndarray::{Array2, ShapeBuilder};
use num_complex::Complex64 as c64;

use crate::{channel::Channel, harmonic_oscillator_hamiltonian, RandomInteractionGen};

fn quadratic_hamiltonian(dim: usize) -> Array2<c64> {
    let mut mat = Array2::zeros((dim, dim).f());
    for ix in 0..dim {
        mat[[ix, ix]] = c64::new((ix * ix) as f64, 0.);
    }
    mat
}

pub fn thermometry() {
    println!("test thermometry");
    let h_sys = quadratic_hamiltonian(10);
    let h_env = harmonic_oscillator_hamiltonian(2);
    let t = 100.0_f64;
    let alpha = 1e-3_f64;
    let rand = RandomInteractionGen::new(1, h_sys.nrows() * h_env.nrows());
    let mut phi = Channel::new(h_sys, crate::channel::GammaSampler::Fixed(1.0), alpha, t, rand);

    let beta_sys = 1.0_f64;
    phi.set_sys_to_thermal_state(beta_sys);
    phi.set_env_state_to_energy_projector(1);
    phi.print_env();
    // Now need to collect transition statistics
    let probe_out = phi.env_map(100);
    let prob_flip: f64 = probe_out[[0, 0]].norm();
    println!("prob_flip: {:}", prob_flip);
}

mod tests {
    #[test]
    fn test_thermo() {
        super::thermometry();
    }
}
