//! AURORA-X High-Performance Compute Core
//!
//! Rust-backed Python extension providing 10-100x speedups for:
//! - Physics engine (RK4 ODE solver)
//! - Kalman filters (EKF/UKF)
//! - Spectral analysis (FFT)
//! - Stream processing (windowed aggregation)
//! - Safety controller (CBF evaluation)
//! - Degradation models (Weibull/Bayesian RUL)

use pyo3::prelude::*;

mod degradation;
mod kalman;
mod physics;
mod safety;
mod spectral;
mod stream;

/// AURORA-X native compute core.
#[pymodule]
fn aurora_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<physics::PhysicsEngine>()?;
    m.add_class::<physics::RotatingMachineryDynamics>()?;
    m.add_class::<kalman::ExtendedKalmanFilter>()?;
    m.add_class::<kalman::UnscentedKalmanFilter>()?;
    m.add_class::<spectral::SpectralAnalyzer>()?;
    m.add_class::<stream::StreamProcessor>()?;
    m.add_class::<safety::SafetyController>()?;
    m.add_class::<degradation::WeibullModel>()?;
    m.add_class::<degradation::BayesianRUL>()?;
    Ok(())
}
