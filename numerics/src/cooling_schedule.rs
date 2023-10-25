use crate::channel;

/// The goal for this test is to study the effects of the cooling schedule
/// on distance convergence.
struct CoolingScheduleConf {
    beta_start: f64,
    beta_end: f64,
    beta_steps: f64,
    beta_linear_scale: bool,
    interactions_per_beta: u32,
    alpha: f64,
    time: f64,
    num_samples: usize,
}
