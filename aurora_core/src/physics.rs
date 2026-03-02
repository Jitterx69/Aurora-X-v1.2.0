//! AURORA-X Physics Engine — RK4 ODE Solver

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

#[derive(Clone)]
struct MachineParams {
    inertia: f64,
    damping: f64,
    thermal_mass: f64,
    cooling_coeff: f64,
    ambient_temp: f64,
    friction_coeff: f64,
    degradation_rate: f64,
    seal_wear_rate: f64,
    pressure_coeff: f64,
}

impl Default for MachineParams {
    fn default() -> Self {
        Self {
            inertia: 0.5,
            damping: 0.01,
            thermal_mass: 50.0,
            cooling_coeff: 0.02,
            ambient_temp: 25.0,
            friction_coeff: 0.001,
            degradation_rate: 1e-7,
            seal_wear_rate: 5e-8,
            pressure_coeff: 0.1,
        }
    }
}

#[derive(Clone)]
struct AssetState {
    state: Array1<f64>,
    time: f64,
    params: MachineParams,
}

fn dynamics(state: &Array1<f64>, _t: f64, u: &[f64; 2], params: &MachineParams) -> Array1<f64> {
    let omega = state[1];
    let t_bearing = state[2];
    let t_housing = state[3];
    let d_bearing = state[4];
    let d_seal = state[5];
    let p_inlet = state[6];

    let torque_cmd = u[0];
    let cooling_cmd = u[1];

    let friction = params.friction_coeff * (1.0 + 5.0 * d_bearing);
    let d_omega = (torque_cmd - params.damping * omega - friction * omega) / params.inertia;

    let heat_gen = friction * omega * omega;
    let cooling = params.cooling_coeff * (1.0 + cooling_cmd) * (t_bearing - params.ambient_temp);
    let d_t_bearing = (heat_gen - cooling) / params.thermal_mass;

    let housing_coupling = 0.005 * (t_bearing - t_housing);
    let housing_cooling = 0.001 * (t_housing - params.ambient_temp);
    let d_t_housing = housing_coupling - housing_cooling;

    let temp_factor = ((t_bearing - 80.0) / 100.0).max(0.0);
    let speed_factor = (omega / 1800.0).powi(2);
    let dd_bearing = params.degradation_rate * (1.0 + temp_factor) * (1.0 + speed_factor);

    let pressure_diff = (p_inlet - 50.0).abs() / 50.0;
    let dd_seal = params.seal_wear_rate * (1.0 + pressure_diff);

    let flow_effect = params.pressure_coeff * omega / 1500.0;
    let seal_leak = d_seal * 5.0;
    let d_p_inlet = flow_effect - seal_leak - 0.01 * (p_inlet - 50.0);
    let d_p_outlet = 0.8 * d_p_inlet - 0.005 * (state[7] - 45.0);

    Array1::from_vec(vec![
        omega,
        d_omega,
        d_t_bearing,
        d_t_housing,
        dd_bearing,
        dd_seal,
        d_p_inlet,
        d_p_outlet,
    ])
}

fn rk4_step(
    state: &Array1<f64>,
    t: f64,
    dt: f64,
    u: &[f64; 2],
    params: &MachineParams,
) -> Array1<f64> {
    let k1 = dynamics(state, t, u, params);
    let k2 = dynamics(&(state + &(&k1 * (dt / 2.0))), t + dt / 2.0, u, params);
    let k3 = dynamics(&(state + &(&k2 * (dt / 2.0))), t + dt / 2.0, u, params);
    let k4 = dynamics(&(state + &(&k3 * dt)), t + dt, u, params);
    state + &((&k1 + &(&k2 * 2.0) + &(&k3 * 2.0) + &k4) * (dt / 6.0))
}

#[pyclass]
#[derive(Clone)]
pub struct RotatingMachineryDynamics {
    params: MachineParams,
}

#[pymethods]
impl RotatingMachineryDynamics {
    #[new]
    #[pyo3(signature = (inertia=0.5, damping=0.01, thermal_mass=50.0, cooling_coeff=0.02,
                        ambient_temp=25.0, friction_coeff=0.001, degradation_rate=1e-7,
                        seal_wear_rate=5e-8, pressure_coeff=0.1))]
    fn new(
        inertia: f64,
        damping: f64,
        thermal_mass: f64,
        cooling_coeff: f64,
        ambient_temp: f64,
        friction_coeff: f64,
        degradation_rate: f64,
        seal_wear_rate: f64,
        pressure_coeff: f64,
    ) -> Self {
        Self {
            params: MachineParams {
                inertia,
                damping,
                thermal_mass,
                cooling_coeff,
                ambient_temp,
                friction_coeff,
                degradation_rate,
                seal_wear_rate,
                pressure_coeff,
            },
        }
    }

    fn derivatives<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, f64>,
        t: f64,
        u: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let s = state.as_array().to_owned();
        let ctrl: [f64; 2] = [
            *u.as_array().get(0).unwrap_or(&0.0),
            *u.as_array().get(1).unwrap_or(&0.0),
        ];
        let result = dynamics(&s, t, &ctrl, &self.params);
        Ok(result.into_pyarray_bound(py).unbind())
    }
}

#[pyclass]
pub struct PhysicsEngine {
    dt: f64,
    assets: HashMap<String, AssetState>,
}

#[pymethods]
impl PhysicsEngine {
    #[new]
    #[pyo3(signature = (dt=0.01))]
    fn new(dt: f64) -> Self {
        Self {
            dt,
            assets: HashMap::new(),
        }
    }

    fn create_asset(&mut self, asset_id: &str) {
        let initial_state = Array1::from_vec(vec![0.0, 1500.0, 80.0, 45.0, 0.0, 0.0, 50.0, 45.0]);
        self.assets.insert(
            asset_id.to_string(),
            AssetState {
                state: initial_state,
                time: 0.0,
                params: MachineParams::default(),
            },
        );
    }

    #[pyo3(signature = (asset_id, u=None))]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        asset_id: &str,
        u: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<Py<PyDict>> {
        let asset = self.assets.get_mut(asset_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Asset '{}' not found", asset_id))
        })?;

        let ctrl: [f64; 2] = match &u {
            Some(arr) => {
                let a = arr.as_array();
                [*a.get(0).unwrap_or(&0.0), *a.get(1).unwrap_or(&0.0)]
            }
            None => [50.0, 5.0],
        };

        let mut new_state = rk4_step(&asset.state, asset.time, self.dt, &ctrl, &asset.params);
        new_state[4] = new_state[4].clamp(0.0, 1.0);
        new_state[5] = new_state[5].clamp(0.0, 1.0);
        asset.state = new_state;
        asset.time += self.dt;

        let dict = PyDict::new_bound(py);
        dict.set_item("theta", asset.state[0])?;
        dict.set_item("shaft_speed", asset.state[1])?;
        dict.set_item("bearing_temp", asset.state[2])?;
        dict.set_item("housing_temp", asset.state[3])?;
        dict.set_item("bearing_degradation", asset.state[4])?;
        dict.set_item("seal_degradation", asset.state[5])?;
        dict.set_item("inlet_pressure", asset.state[6])?;
        dict.set_item("outlet_pressure", asset.state[7])?;
        dict.set_item("time", asset.time)?;
        Ok(dict.unbind())
    }

    #[pyo3(signature = (asset_id, duration, u=None))]
    fn simulate<'py>(
        &mut self,
        py: Python<'py>,
        asset_id: &str,
        duration: f64,
        u: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<Py<PyDict>> {
        let _asset = self.assets.get(asset_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Asset '{}' not found", asset_id))
        })?;

        let ctrl: [f64; 2] = match &u {
            Some(arr) => {
                let a = arr.as_array();
                [*a.get(0).unwrap_or(&0.0), *a.get(1).unwrap_or(&0.0)]
            }
            None => [50.0, 5.0],
        };

        let n_steps = (duration / self.dt).ceil() as usize;
        let asset = self.assets.get_mut(asset_id).unwrap();
        for _ in 0..n_steps {
            let mut new_state = rk4_step(&asset.state, asset.time, self.dt, &ctrl, &asset.params);
            new_state[4] = new_state[4].clamp(0.0, 1.0);
            new_state[5] = new_state[5].clamp(0.0, 1.0);
            asset.state = new_state;
            asset.time += self.dt;
        }

        let dict = PyDict::new_bound(py);
        dict.set_item("theta", asset.state[0])?;
        dict.set_item("shaft_speed", asset.state[1])?;
        dict.set_item("bearing_temp", asset.state[2])?;
        dict.set_item("housing_temp", asset.state[3])?;
        dict.set_item("bearing_degradation", asset.state[4])?;
        dict.set_item("seal_degradation", asset.state[5])?;
        dict.set_item("inlet_pressure", asset.state[6])?;
        dict.set_item("outlet_pressure", asset.state[7])?;
        dict.set_item("time", asset.time)?;
        Ok(dict.unbind())
    }

    fn get_state<'py>(&self, py: Python<'py>, asset_id: &str) -> PyResult<Py<PyArray1<f64>>> {
        let asset = self.assets.get(asset_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Asset '{}' not found", asset_id))
        })?;
        Ok(asset.state.clone().into_pyarray_bound(py).unbind())
    }

    fn get_time(&self, asset_id: &str) -> PyResult<f64> {
        let asset = self.assets.get(asset_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Asset '{}' not found", asset_id))
        })?;
        Ok(asset.time)
    }

    fn asset_ids(&self) -> Vec<String> {
        self.assets.keys().cloned().collect()
    }
}
