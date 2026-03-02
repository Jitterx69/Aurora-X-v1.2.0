//! AURORA-X Stream Processor — Windowed Aggregation

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::HashMap;

struct RingBuffer {
    data: Vec<f64>,
    head: usize,
    count: usize,
    capacity: usize,
    samples_since_emit: usize,
    step_size: usize,
}

impl RingBuffer {
    fn new(capacity: usize, overlap: f64) -> Self {
        let step_size = ((capacity as f64) * (1.0 - overlap)).max(1.0) as usize;
        Self {
            data: vec![0.0; capacity],
            head: 0,
            count: 0,
            capacity,
            samples_since_emit: 0,
            step_size,
        }
    }

    fn push(&mut self, value: f64) -> bool {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
        self.samples_since_emit += 1;
        if self.count >= self.capacity && self.samples_since_emit >= self.step_size {
            self.samples_since_emit = 0;
            true
        } else {
            false
        }
    }

    fn window(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.capacity);
        let start = if self.count < self.capacity {
            0
        } else {
            self.head
        };
        for i in 0..self.count {
            result.push(self.data[(start + i) % self.capacity]);
        }
        result
    }
}

struct RollingStats {
    buffer: Vec<f64>,
    capacity: usize,
    head: usize,
    count: usize,
}

impl RollingStats {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity],
            capacity,
            head: 0,
            count: 0,
        }
    }

    fn push(&mut self, value: f64) {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    fn compute(&self) -> (f64, f64, f64, f64) {
        if self.count == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let start = if self.count < self.capacity {
            0
        } else {
            self.head
        };
        let mut sum = 0.0f64;
        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;
        for i in 0..self.count {
            let v = self.buffer[(start + i) % self.capacity];
            sum += v;
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
        let mean = sum / self.count as f64;
        let mut var_sum = 0.0;
        for i in 0..self.count {
            let v = self.buffer[(start + i) % self.capacity];
            var_sum += (v - mean) * (v - mean);
        }
        (mean, (var_sum / self.count as f64).sqrt(), min_val, max_val)
    }
}

fn compute_window_features(window: &[f64]) -> Vec<(String, f64)> {
    let n = window.len() as f64;
    let mut feats = Vec::new();
    let (mut sum, mut sq_sum, mut peak) = (0.0, 0.0, 0.0f64);
    for &v in window {
        sum += v;
        sq_sum += v * v;
        let a = v.abs();
        if a > peak {
            peak = a;
        }
    }
    let mean = sum / n;
    let rms = (sq_sum / n).sqrt();
    let variance = (sq_sum / n - mean * mean).max(0.0);
    let std = variance.sqrt();
    feats.push(("rms".into(), rms));
    feats.push(("peak".into(), peak));
    feats.push(("crest_factor".into(), peak / rms.max(1e-10)));
    feats.push(("mean".into(), mean));
    feats.push(("variance".into(), variance));
    if std > 1e-10 {
        let (mut m3, mut m4) = (0.0, 0.0);
        for &v in window {
            let z = (v - mean) / std;
            m3 += z * z * z;
            m4 += z * z * z * z;
        }
        feats.push(("skewness".into(), m3 / n));
        feats.push(("kurtosis".into(), m4 / n));
    } else {
        feats.push(("skewness".into(), 0.0));
        feats.push(("kurtosis".into(), 0.0));
    }
    feats
}

fn compute_fft_features(window: &[f64], n_bins: usize) -> Vec<(String, f64)> {
    let n = window.len();
    let mut buffer: Vec<Complex<f64>> = (0..n)
        .map(|i| {
            let win =
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n.max(2) - 1) as f64).cos());
            Complex::new(window[i] * win, 0.0)
        })
        .collect();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);
    let bins = (n / 2 + 1).min(n_bins);
    let n_norm = n.max(1) as f64;
    let mag: Vec<f64> = (0..bins).map(|i| 2.0 * buffer[i].norm() / n_norm).collect();
    let power: Vec<f64> = mag.iter().map(|m| m * m).collect();
    let total_power: f64 = power.iter().sum::<f64>() + 1e-10;
    let mut feats = Vec::new();
    let (dom_idx, dom_mag) = mag
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0));
    feats.push(("fft_dominant_bin".into(), dom_idx as f64));
    feats.push(("fft_dominant_magnitude".into(), dom_mag));
    let mag_sum: f64 = mag.iter().sum::<f64>() + 1e-10;
    let centroid: f64 = (0..bins).map(|i| i as f64 * mag[i]).sum::<f64>() / mag_sum;
    feats.push(("spectral_centroid".into(), centroid));
    let entropy: f64 = power
        .iter()
        .map(|p| p / total_power)
        .filter(|p| *p > 0.0)
        .map(|p| -p * p.log2())
        .sum();
    feats.push(("spectral_entropy".into(), entropy));
    let third = bins / 3;
    if third > 0 {
        feats.push(("band_low".into(), power[..third].iter().sum()));
        feats.push(("band_mid".into(), power[third..2 * third].iter().sum()));
        feats.push(("band_high".into(), power[2 * third..].iter().sum()));
    }
    feats
}

#[pyclass]
pub struct StreamProcessor {
    window_size: usize,
    overlap: f64,
    fft_bins: usize,
    rolling_window: usize,
    buffers: HashMap<String, HashMap<String, RingBuffer>>,
    rolling: HashMap<String, RollingStats>,
}

#[pymethods]
impl StreamProcessor {
    #[new]
    #[pyo3(signature = (window_size=256, overlap=0.5, fft_bins=128, rolling_window=64))]
    fn new(window_size: usize, overlap: f64, fft_bins: usize, rolling_window: usize) -> Self {
        Self {
            window_size,
            overlap,
            fft_bins,
            rolling_window,
            buffers: HashMap::new(),
            rolling: HashMap::new(),
        }
    }

    fn process<'py>(
        &mut self,
        py: Python<'py>,
        asset_id: &str,
        sensors: &Bound<'py, PyDict>,
        timestamp: f64,
    ) -> PyResult<Option<Py<PyDict>>> {
        let mut features: Vec<(String, f64)> = Vec::new();
        let mut window_ready = false;

        for (key, value) in sensors.iter() {
            let sensor_name: String = key.extract()?;
            let val: f64 = value.extract()?;

            let asset_buffers = self.buffers.entry(asset_id.to_string()).or_default();
            let ws = self.window_size;
            let ov = self.overlap;
            let buf = asset_buffers
                .entry(sensor_name.clone())
                .or_insert_with(|| RingBuffer::new(ws, ov));

            let rw = self.rolling_window;
            let roll_key = format!("{}:{}", asset_id, sensor_name);
            let roll = self
                .rolling
                .entry(roll_key)
                .or_insert_with(|| RollingStats::new(rw));
            roll.push(val);
            let (r_mean, r_std, r_min, r_max) = roll.compute();
            features.push((format!("{}_rolling_mean", sensor_name), r_mean));
            features.push((format!("{}_rolling_std", sensor_name), r_std));
            features.push((format!("{}_rolling_min", sensor_name), r_min));
            features.push((format!("{}_rolling_max", sensor_name), r_max));

            if buf.push(val) {
                window_ready = true;
                let window = buf.window();
                for (k, v) in compute_window_features(&window) {
                    features.push((format!("{}_{}", sensor_name, k), v));
                }
                let fb = self.fft_bins;
                for (k, v) in compute_fft_features(&window, fb) {
                    features.push((format!("{}_{}", sensor_name, k), v));
                }
            }
        }

        if features.is_empty() {
            return Ok(None);
        }

        let result = PyDict::new_bound(py);
        result.set_item("asset_id", asset_id)?;
        result.set_item("timestamp", timestamp)?;
        result.set_item("window_complete", window_ready)?;

        let feat_dict = PyDict::new_bound(py);
        for (k, v) in &features {
            feat_dict.set_item(k.as_str(), *v)?;
        }
        result.set_item("features", &feat_dict)?;

        Ok(Some(result.unbind()))
    }
}
