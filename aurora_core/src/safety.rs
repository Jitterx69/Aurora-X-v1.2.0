//! AURORA-X Safety Controller — Control Barrier Functions

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Clone)]
struct BarrierFunction {
    name: String,
    limit: f64,
    state_index: usize,
    #[allow(dead_code)]
    alpha: f64,
}

impl BarrierFunction {
    fn evaluate(&self, state: &[f64]) -> f64 {
        if self.state_index < state.len() {
            self.limit - state[self.state_index]
        } else {
            self.limit
        }
    }
    fn is_safe(&self, state: &[f64]) -> bool {
        self.evaluate(state) >= 0.0
    }
    fn margin(&self, state: &[f64]) -> f64 {
        (self.evaluate(state) / self.limit.max(1e-10)).clamp(0.0, 1.0)
    }
}

#[pyclass]
pub struct SafetyController {
    barriers: Vec<BarrierFunction>,
    enable_fallback: bool,
    violation_count: u64,
    intervention_count: u64,
    fallback_count: u64,
}

#[pymethods]
impl SafetyController {
    #[new]
    #[pyo3(signature = (max_temperature=450.0, max_vibration=15.0, max_pressure=120.0,
                        barrier_alpha=0.1, enable_fallback=true))]
    fn new(
        max_temperature: f64,
        max_vibration: f64,
        max_pressure: f64,
        barrier_alpha: f64,
        enable_fallback: bool,
    ) -> Self {
        Self {
            barriers: vec![
                BarrierFunction {
                    name: "temperature".into(),
                    limit: max_temperature,
                    state_index: 0,
                    alpha: barrier_alpha,
                },
                BarrierFunction {
                    name: "vibration".into(),
                    limit: max_vibration,
                    state_index: 1,
                    alpha: barrier_alpha,
                },
                BarrierFunction {
                    name: "pressure".into(),
                    limit: max_pressure,
                    state_index: 2,
                    alpha: barrier_alpha,
                },
            ],
            enable_fallback,
            violation_count: 0,
            intervention_count: 0,
            fallback_count: 0,
        }
    }

    fn validate(&mut self, action: Vec<f64>, state_values: Vec<f64>) -> Vec<f64> {
        let mut safe_action = action.clone();
        if safe_action.len() < 4 {
            safe_action.resize(4, 0.0);
        }

        let mut violations: Vec<String> = Vec::new();
        let mut margins: Vec<(String, f64)> = Vec::new();

        for barrier in &self.barriers {
            let margin = barrier.margin(&state_values);
            margins.push((barrier.name.clone(), margin));
            if !barrier.is_safe(&state_values) {
                violations.push(barrier.name.clone());
                self.violation_count += 1;
            }
        }

        if !violations.is_empty() && self.enable_fallback {
            self.fallback_count += 1;
            return vec![0.8, -0.15, 1.0, -0.08];
        }

        let min_margin = margins.iter().map(|(_, m)| *m).fold(1.0f64, f64::min);
        if min_margin < 0.2 {
            self.intervention_count += 1;
            Self::apply_safe_modification(&mut safe_action, &margins);
        }

        // Predictive safety
        let temp_margin = margins
            .iter()
            .find(|(n, _)| n == "temperature")
            .map(|(_, m)| *m)
            .unwrap_or(1.0);
        let vib_margin = margins
            .iter()
            .find(|(n, _)| n == "vibration")
            .map(|(_, m)| *m)
            .unwrap_or(1.0);
        if vib_margin < 0.3 && safe_action[1] > 0.0 {
            safe_action[1] = safe_action[1].min(0.0);
            self.intervention_count += 1;
        }
        if temp_margin < 0.3 && safe_action[2] < 0.3 {
            safe_action[2] = safe_action[2].max(0.7);
            self.intervention_count += 1;
        }

        safe_action
    }

    fn get_stats<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("violations", self.violation_count)?;
        dict.set_item("interventions", self.intervention_count)?;
        dict.set_item("fallbacks", self.fallback_count)?;
        Ok(dict.unbind())
    }

    fn get_barrier_status<'py>(
        &self,
        py: Python<'py>,
        state_values: Vec<f64>,
    ) -> PyResult<Py<PyDict>> {
        let result = PyDict::new_bound(py);
        for barrier in &self.barriers {
            let status = PyDict::new_bound(py);
            status.set_item("value", barrier.evaluate(&state_values))?;
            status.set_item("margin", barrier.margin(&state_values))?;
            status.set_item("safe", barrier.is_safe(&state_values))?;
            status.set_item("limit", barrier.limit)?;
            if barrier.state_index < state_values.len() {
                status.set_item("current", state_values[barrier.state_index])?;
            }
            result.set_item(barrier.name.as_str(), &status)?;
        }
        Ok(result.unbind())
    }

    fn reset_stats(&mut self) {
        self.violation_count = 0;
        self.intervention_count = 0;
        self.fallback_count = 0;
    }
}

impl SafetyController {
    fn apply_safe_modification(action: &mut Vec<f64>, margins: &[(String, f64)]) {
        for (name, margin) in margins {
            match name.as_str() {
                "temperature" if *margin < 0.3 => {
                    action[2] = action[2].max(0.7);
                    action[1] = action[1].min(0.0);
                }
                "vibration" if *margin < 0.3 => {
                    action[1] = action[1].min(-0.1);
                }
                "pressure" if *margin < 0.3 => {
                    action[3] = -0.05;
                }
                _ => {}
            }
        }
    }
}
