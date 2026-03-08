//! AURORA-X Kalman Filters — EKF & UKF

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass]
pub struct ExtendedKalmanFilter {
    state_dim: usize,
    #[allow(dead_code)]
    meas_dim: usize,
    x: Array1<f64>,
    p: Array2<f64>,
    q: Array2<f64>,
    r: Array2<f64>,
    f_mat: Array2<f64>,
    h_mat: Array2<f64>,
    #[allow(dead_code)]
    initialized: bool,
}

#[pymethods]
impl ExtendedKalmanFilter {
    #[new]
    #[pyo3(signature = (state_dim, meas_dim, process_noise=0.01, measurement_noise=0.05))]
    fn new(state_dim: usize, meas_dim: usize, process_noise: f64, measurement_noise: f64) -> Self {
        let mut q = Array2::zeros((state_dim, state_dim));
        let mut r = Array2::zeros((meas_dim, meas_dim));
        for i in 0..state_dim {
            q[[i, i]] = process_noise;
        }
        for i in 0..meas_dim {
            r[[i, i]] = measurement_noise;
        }
        Self {
            state_dim,
            meas_dim,
            x: Array1::zeros(state_dim),
            p: Array2::eye(state_dim),
            q,
            r,
            f_mat: Array2::eye(state_dim),
            h_mat: {
                let mut h = Array2::zeros((meas_dim, state_dim));
                for i in 0..meas_dim.min(state_dim) {
                    h[[i, i]] = 1.0;
                }
                h
            },
            initialized: false,
        }
    }

    #[pyo3(signature = (x0, p0=None))]
    fn initialize(&mut self, x0: PyReadonlyArray1<'_, f64>, p0: Option<PyReadonlyArray2<'_, f64>>) {
        self.x = x0.as_array().to_owned();
        if let Some(p) = p0 {
            self.p = p.as_array().to_owned();
        }
        self.initialized = true;
    }

    #[pyo3(signature = (f_jacobian=None, dt=None))]
    fn predict(
        &mut self,
        f_jacobian: Option<PyReadonlyArray2<'_, f64>>,
        #[allow(unused)] dt: Option<f64>,
    ) -> PyResult<()> {
        let f = match &f_jacobian {
            Some(j) => j.as_array().to_owned(),
            None => self.f_mat.clone(),
        };
        self.x = mat_vec_mul(&f, &self.x);
        let fp = mat_mat_mul(&f, &self.p);
        self.p = mat_mat_mul_t(&fp, &f) + &self.q;
        Ok(())
    }

    #[pyo3(signature = (z, h_jacobian=None))]
    fn update(
        &mut self,
        z: PyReadonlyArray1<'_, f64>,
        h_jacobian: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<()> {
        let z_arr = z.as_array().to_owned();
        let h = match &h_jacobian {
            Some(j) => j.as_array().to_owned(),
            None => self.h_mat.clone(),
        };
        let z_pred = mat_vec_mul(&h, &self.x);
        let y = &z_arr - &z_pred;
        let hp = mat_mat_mul(&h, &self.p);
        let s = mat_mat_mul_t(&hp, &h) + &self.r;
        let s_inv = invert_matrix(&s);
        let ht = transpose(&h);
        let pht = mat_mat_mul(&self.p, &ht);
        let k = mat_mat_mul(&pht, &s_inv);
        self.x = &self.x + &mat_vec_mul(&k, &y);
        let kh = mat_mat_mul(&k, &h);
        let i_kh = &Array2::eye(self.state_dim) - &kh;
        self.p = mat_mat_mul(&i_kh, &self.p);
        Ok(())
    }

    fn state<'py>(&self, py: Python<'py>) -> Py<PyArray1<f64>> {
        self.x.clone().into_pyarray(py).unbind()
    }

    fn covariance<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        self.p.clone().into_pyarray(py).unbind()
    }

    fn uncertainty<'py>(&self, py: Python<'py>) -> Py<PyArray1<f64>> {
        let diag: Array1<f64> =
            Array1::from_iter((0..self.state_dim).map(|i| self.p[[i, i]].sqrt()));
        diag.into_pyarray(py).unbind()
    }
}

#[pyclass]
pub struct UnscentedKalmanFilter {
    state_dim: usize,
    meas_dim: usize,
    x: Array1<f64>,
    p: Array2<f64>,
    q: Array2<f64>,
    r: Array2<f64>,
    #[allow(dead_code)]
    alpha: f64,
    #[allow(dead_code)]
    beta: f64,
    #[allow(dead_code)]
    kappa: f64,
    lambda: f64,
    wm: Vec<f64>,
    wc: Vec<f64>,
}

#[pymethods]
impl UnscentedKalmanFilter {
    #[new]
    #[pyo3(signature = (state_dim, meas_dim, process_noise=0.01, measurement_noise=0.05,
                        alpha=1e-3, beta=2.0, kappa=0.0))]
    fn new(
        state_dim: usize,
        meas_dim: usize,
        process_noise: f64,
        measurement_noise: f64,
        alpha: f64,
        beta: f64,
        kappa: f64,
    ) -> Self {
        let n = state_dim as f64;
        let lambda = alpha * alpha * (n + kappa) - n;
        let n_sigma = 2 * state_dim + 1;
        let mut wm = vec![0.0; n_sigma];
        let mut wc = vec![0.0; n_sigma];
        wm[0] = lambda / (n + lambda);
        wc[0] = wm[0] + (1.0 - alpha * alpha + beta);
        for i in 1..n_sigma {
            wm[i] = 1.0 / (2.0 * (n + lambda));
            wc[i] = wm[i];
        }
        let mut q = Array2::zeros((state_dim, state_dim));
        let mut r = Array2::zeros((meas_dim, meas_dim));
        for i in 0..state_dim {
            q[[i, i]] = process_noise;
        }
        for i in 0..meas_dim {
            r[[i, i]] = measurement_noise;
        }
        Self {
            state_dim,
            meas_dim,
            x: Array1::zeros(state_dim),
            p: Array2::eye(state_dim),
            q,
            r,
            alpha,
            beta,
            kappa,
            lambda,
            wm,
            wc,
        }
    }

    #[pyo3(signature = (x0, p0=None))]
    fn initialize(&mut self, x0: PyReadonlyArray1<'_, f64>, p0: Option<PyReadonlyArray2<'_, f64>>) {
        self.x = x0.as_array().to_owned();
        if let Some(p) = p0 {
            self.p = p.as_array().to_owned();
        }
    }

    #[pyo3(signature = (dt=None))]
    fn predict(&mut self, #[allow(unused)] dt: Option<f64>) -> PyResult<()> {
        let sigma = self.gen_sigma_points();
        let mut x_mean: Array1<f64> = Array1::zeros(self.state_dim);
        for (i, pt) in sigma.iter().enumerate() {
            x_mean = &x_mean + &(pt * self.wm[i]);
        }
        let mut p_new = self.q.clone();
        for (i, pt) in sigma.iter().enumerate() {
            let diff = pt - &x_mean;
            for r in 0..self.state_dim {
                for c in 0..self.state_dim {
                    p_new[[r, c]] += self.wc[i] * diff[r] * diff[c];
                }
            }
        }
        self.x = x_mean;
        self.p = p_new;
        Ok(())
    }

    fn update(&mut self, z: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let z_arr = z.as_array().to_owned();
        let sigma = self.gen_sigma_points();
        let z_sigma: Vec<Array1<f64>> = sigma
            .iter()
            .map(|s| {
                let m = s.len().min(self.meas_dim);
                Array1::from_iter((0..self.meas_dim).map(|i| if i < m { s[i] } else { 0.0 }))
            })
            .collect();
        let mut z_mean: Array1<f64> = Array1::zeros(self.meas_dim);
        for (i, zs) in z_sigma.iter().enumerate() {
            z_mean = &z_mean + &(zs * self.wm[i]);
        }
        let mut pzz = self.r.clone();
        let mut pxz = Array2::zeros((self.state_dim, self.meas_dim));
        for i in 0..sigma.len() {
            let dz = &z_sigma[i] - &z_mean;
            let dx = &sigma[i] - &self.x;
            for r in 0..self.meas_dim {
                for c in 0..self.meas_dim {
                    pzz[[r, c]] += self.wc[i] * dz[r] * dz[c];
                }
            }
            for r in 0..self.state_dim {
                for c in 0..self.meas_dim {
                    pxz[[r, c]] += self.wc[i] * dx[r] * dz[c];
                }
            }
        }
        let pzz_inv = invert_matrix(&pzz);
        let k = mat_mat_mul(&pxz, &pzz_inv);
        let innovation = &z_arr - &z_mean;
        self.x = &self.x + &mat_vec_mul(&k, &innovation);
        let k_pzz = mat_mat_mul(&k, &pzz);
        let k_pzz_kt = mat_mat_mul_t(&k_pzz, &k);
        self.p = &self.p - &k_pzz_kt;
        Ok(())
    }

    fn state<'py>(&self, py: Python<'py>) -> Py<PyArray1<f64>> {
        self.x.clone().into_pyarray(py).unbind()
    }

    fn covariance<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        self.p.clone().into_pyarray(py).unbind()
    }

    fn uncertainty<'py>(&self, py: Python<'py>) -> Py<PyArray1<f64>> {
        let diag: Array1<f64> =
            Array1::from_iter((0..self.state_dim).map(|i| self.p[[i, i]].sqrt()));
        diag.into_pyarray(py).unbind()
    }
}

impl UnscentedKalmanFilter {
    fn gen_sigma_points(&self) -> Vec<Array1<f64>> {
        let n = self.state_dim;
        let scale = ((n as f64 + self.lambda).max(0.0)).sqrt();
        let l = cholesky(&self.p);
        let mut points = Vec::with_capacity(2 * n + 1);
        points.push(self.x.clone());
        for i in 0..n {
            let col: Array1<f64> = Array1::from_iter((0..n).map(|j| l[[j, i]] * scale));
            points.push(&self.x + &col);
            points.push(&self.x - &col);
        }
        points
    }
}

// ─── Linear algebra helpers ───

fn mat_vec_mul(a: &Array2<f64>, x: &Array1<f64>) -> Array1<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut result = Array1::zeros(m);
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a[[i, j]] * x[j];
        }
        result[i] = sum;
    }
    result
}

fn mat_mat_mul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    let mut result = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[[i, l]] * b[[l, j]];
            }
            result[[i, j]] = sum;
        }
    }
    result
}

fn mat_mat_mul_t(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.nrows();
    let mut result = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[[i, l]] * b[[j, l]];
            }
            result[[i, j]] = sum;
        }
    }
    result
}

fn transpose(a: &Array2<f64>) -> Array2<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut result = Array2::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            result[[j, i]] = a[[i, j]];
        }
    }
    result
}

fn invert_matrix(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut aug = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }
    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > aug[[max_row, col]].abs() {
                max_row = row;
            }
        }
        for j in 0..(2 * n) {
            let tmp = aug[[col, j]];
            aug[[col, j]] = aug[[max_row, j]];
            aug[[max_row, j]] = tmp;
        }
        let pivot = aug[[col, col]];
        if pivot.abs() < 1e-12 {
            return Array2::eye(n);
        }
        for j in 0..(2 * n) {
            aug[[col, j]] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[[row, col]];
                for j in 0..(2 * n) {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }
    }
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = aug[[i, n + j]];
        }
    }
    result
}

fn cholesky(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut l = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let val = a[[i, i]] - sum;
                l[[i, j]] = if val > 0.0 { val.sqrt() } else { 1e-10 };
            } else {
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]].max(1e-10);
            }
        }
    }
    l
}
