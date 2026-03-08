//! AURORA-X Degradation Models — Weibull & Bayesian RUL

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
pub struct WeibullModel {
    shape: f64,
    scale: f64,
}

#[pymethods]
impl WeibullModel {
    #[new]
    #[pyo3(signature = (shape=2.5, scale=10000.0))]
    fn new(shape: f64, scale: f64) -> Self {
        Self { shape, scale }
    }

    fn pdf(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let ts = t / self.scale;
        (self.shape / self.scale) * ts.powf(self.shape - 1.0) * (-ts.powf(self.shape)).exp()
    }

    fn cdf(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        1.0 - (-(t / self.scale).powf(self.shape)).exp()
    }

    fn survival_probability(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        (-(t / self.scale).powf(self.shape)).exp()
    }

    fn hazard_rate(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        (self.shape / self.scale) * (t / self.scale).powf(self.shape - 1.0)
    }

    fn cumulative_hazard(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        (t / self.scale).powf(self.shape)
    }

    fn mttf(&self) -> f64 {
        self.scale * gamma_fn(1.0 + 1.0 / self.shape)
    }

    fn hazard_rates<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray1<'_, f64>,
    ) -> Py<PyArray1<f64>> {
        let rates: Vec<f64> = t
            .as_array()
            .iter()
            .map(|&tv| {
                let s = self.shape;
                let sc = self.scale;
                (s / sc) * (tv / sc).powf(s - 1.0)
            })
            .collect();
        let pyarray: Bound<'py, PyArray1<f64>> = Array1::from_vec(rates).into_pyarray(py);
        pyarray.unbind()
    }
}

#[pyclass]
pub struct BayesianRUL {
    n_particles: usize,
    particles: Vec<f64>,
    weights: Vec<f64>,
    #[allow(dead_code)]
    prior_mean: f64,
    #[allow(dead_code)]
    prior_std: f64,
    observation_noise: f64,
    weibull: WeibullModel,
    rng_state: u64,
}

#[pymethods]
impl BayesianRUL {
    #[new]
    #[pyo3(signature = (n_particles=1000, prior_mean=5000.0, prior_std=2000.0,
                        observation_noise=0.05, weibull_shape=2.5, weibull_scale=10000.0))]
    fn new(
        n_particles: usize,
        prior_mean: f64,
        prior_std: f64,
        observation_noise: f64,
        weibull_shape: f64,
        weibull_scale: f64,
    ) -> Self {
        let mut rng_state: u64 = 42;
        let particles: Vec<f64> = (0..n_particles)
            .map(|_| {
                let u = xorshift64(&mut rng_state);
                let z = box_muller_z(u, xorshift64(&mut rng_state));
                (prior_mean + prior_std * z).max(1.0)
            })
            .collect();
        let weights = vec![1.0 / n_particles as f64; n_particles];
        Self {
            n_particles,
            particles,
            weights,
            prior_mean,
            prior_std,
            observation_noise,
            weibull: WeibullModel::new(weibull_shape, weibull_scale),
            rng_state,
        }
    }

    fn observe(&mut self, operating_time: f64, degradation: f64) {
        for i in 0..self.n_particles {
            let predicted_rul = self.particles[i];
            let total_life = operating_time + predicted_rul;
            let expected_deg = (operating_time / total_life.max(1.0)).powf(self.weibull.shape);
            let residual = (degradation - expected_deg).abs();
            let likelihood = (-0.5 * (residual / self.observation_noise).powi(2)).exp();
            self.weights[i] *= likelihood.max(1e-300);
        }
        let w_sum: f64 = self.weights.iter().sum();
        if w_sum > 1e-300 {
            for w in &mut self.weights {
                *w /= w_sum;
            }
        } else {
            for w in &mut self.weights {
                *w = 1.0 / self.n_particles as f64;
            }
        }
        let ess: f64 = 1.0 / self.weights.iter().map(|w| w * w).sum::<f64>();
        if ess < self.n_particles as f64 / 2.0 {
            self.resample();
        }
    }

    fn estimate_rul<'py>(&self, py: Python<'py>, _operating_time: f64) -> PyResult<Py<PyDict>> {
        let mut sorted: Vec<(f64, f64)> = self
            .particles
            .iter()
            .zip(self.weights.iter())
            .map(|(&p, &w)| (p, w))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mean: f64 = self
            .particles
            .iter()
            .zip(self.weights.iter())
            .map(|(p, w)| p * w)
            .sum();
        let var: f64 = self
            .particles
            .iter()
            .zip(self.weights.iter())
            .map(|(p, w)| (p - mean).powi(2) * w)
            .sum();

        let median = weighted_percentile(&sorted, 0.5);
        let p10 = weighted_percentile(&sorted, 0.1);
        let p90 = weighted_percentile(&sorted, 0.9);

        let dict = PyDict::new(py);
        dict.set_item("mean", mean)?;
        dict.set_item("std", var.sqrt())?;
        dict.set_item("median", median)?;
        dict.set_item("p10", p10)?;
        dict.set_item("p90", p90)?;
        dict.set_item("confidence_interval", vec![p10, p90])?;
        Ok(dict.unbind())
    }
}

impl BayesianRUL {
    fn resample(&mut self) {
        let n = self.n_particles;
        let mut new_particles = Vec::with_capacity(n);
        let u0 = xorshift64(&mut self.rng_state) / n as f64;
        let mut cumsum = self.weights[0];
        let mut j = 0;
        for i in 0..n {
            let target = u0 + i as f64 / n as f64;
            while cumsum < target && j < n - 1 {
                j += 1;
                cumsum += self.weights[j];
            }
            let noise = (xorshift64(&mut self.rng_state) - 0.5) * 100.0;
            new_particles.push((self.particles[j] + noise).max(0.0));
        }
        self.particles = new_particles;
        self.weights = vec![1.0 / n as f64; n];
    }
}

fn weighted_percentile(sorted: &[(f64, f64)], p: f64) -> f64 {
    let mut cumsum = 0.0;
    for &(val, weight) in sorted {
        cumsum += weight;
        if cumsum >= p {
            return val;
        }
    }
    sorted.last().map(|&(v, _)| v).unwrap_or(0.0)
}

fn xorshift64(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x as f64) / (u64::MAX as f64)
}

fn box_muller_z(u1: f64, u2: f64) -> f64 {
    (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn gamma_fn(x: f64) -> f64 {
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_fn(1.0 - x))
    } else {
        let x = x - 1.0;
        let c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        let mut sum = c[0];
        for (i, &coef) in c.iter().enumerate().skip(1) {
            sum += coef / (x + i as f64);
        }
        let t = x + 7.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
    }
}
