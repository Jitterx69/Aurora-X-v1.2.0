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
use std::sync::Once;

mod degradation;
mod kalman;
mod physics;
mod safety;
mod spectral;
mod stream;

static INIT: Once = Once::new();

/// Initialize OpenTelemetry tracing for the Rust core.
#[pyfunction]
fn init_tracing(endpoint: Option<String>) -> PyResult<()> {
    INIT.call_once(|| {
        use opentelemetry::global;
        use opentelemetry_otlp::WithExportConfig;
        use opentelemetry_sdk::propagation::TraceContextPropagator;
        use tracing_subscriber::prelude::*;

        let endpoint = endpoint.unwrap_or_else(|| "http://jaeger:4317".to_string());

        // Configure OTLP exporter
        let exporter = opentelemetry_otlp::new_exporter()
            .tonic()
            .with_endpoint(endpoint);

        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(exporter)
            .with_trace_config(opentelemetry_sdk::trace::config().with_resource(
                opentelemetry_sdk::Resource::new(vec![opentelemetry::KeyValue::new(
                    "service.name",
                    "aurora-x-rust-core",
                )]),
            ))
            .install_batch(opentelemetry_sdk::runtime::Tokio)
            .expect("Failed to initialize tracer");

        global::set_text_map_propagator(TraceContextPropagator::new());

        tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(tracer))
            .init();
    });
    Ok(())
}

/// AURORA-X native compute core.
#[pymodule]
fn aurora_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_tracing, m)?)?;
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
