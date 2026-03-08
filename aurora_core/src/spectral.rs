//! AURORA-X Spectral Analysis — FFT via rustfft

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustfft::{num_complex::Complex, FftPlanner};

#[pyclass]
pub struct SpectralAnalyzer {
    sample_rate: f64,
    n_fft: usize,
    freq_resolution: f64,
}

impl SpectralAnalyzer {
    /// Internal full_spectrum returning Rust vectors.
    fn compute_spectrum(&self, signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = signal.len();
        let nfft = self.n_fft;

        let mut buffer: Vec<Complex<f64>> = (0..nfft)
            .map(|i| {
                let val = if i < n { signal[i] } else { 0.0 };
                let win = 0.5
                    * (1.0
                        - (2.0 * std::f64::consts::PI * i as f64 / (nfft.max(2) - 1) as f64).cos());
                Complex::new(val * win, 0.0)
            })
            .collect();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nfft);
        fft.process(&mut buffer);

        let n_bins = nfft / 2 + 1;
        let n_norm = n.max(1) as f64;
        let mut frequencies = Vec::with_capacity(n_bins);
        let mut magnitudes = Vec::with_capacity(n_bins);

        for i in 0..n_bins {
            frequencies.push(i as f64 * self.sample_rate / nfft as f64);
            magnitudes.push(2.0 * buffer[i].norm() / n_norm);
        }
        (frequencies, magnitudes)
    }
}

#[pymethods]
impl SpectralAnalyzer {
    #[new]
    #[pyo3(signature = (sample_rate=100.0, n_fft=256))]
    fn new(sample_rate: f64, n_fft: usize) -> Self {
        Self {
            sample_rate,
            n_fft,
            freq_resolution: sample_rate / n_fft as f64,
        }
    }

    fn full_spectrum<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        let sig = signal.as_array();
        let s: Vec<f64> = sig.iter().copied().collect();
        let (freqs, mags) = self.compute_spectrum(&s);
        Ok((
            Array1::from_vec(freqs).into_pyarray(py).unbind(),
            Array1::from_vec(mags).into_pyarray(py).unbind(),
        ))
    }

    fn power_spectral_density<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        let sig: Vec<f64> = signal.as_array().iter().copied().collect();
        let (freqs, mags) = self.compute_spectrum(&sig);
        let fr = self.freq_resolution.max(1e-10);
        let psd: Vec<f64> = mags.iter().map(|m| m * m / fr).collect();
        Ok((
            Array1::from_vec(freqs).into_pyarray(py).unbind(),
            Array1::from_vec(psd).into_pyarray(py).unbind(),
        ))
    }

    fn envelope_spectrum<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        let sig: Vec<f64> = signal.as_array().iter().copied().collect();
        let n = sig.len();
        if n == 0 {
            return Ok((
                Array1::from_vec(vec![]).into_pyarray(py).unbind(),
                Array1::from_vec(vec![]).into_pyarray(py).unbind(),
            ));
        }

        let mut planner = FftPlanner::new();
        let fft_fwd = planner.plan_fft_forward(n);
        let fft_inv = planner.plan_fft_inverse(n);

        let mut buffer: Vec<Complex<f64>> = sig.iter().map(|&v| Complex::new(v, 0.0)).collect();
        fft_fwd.process(&mut buffer);

        // Hilbert filter
        if n % 2 == 0 {
            for i in 1..n / 2 {
                buffer[i] *= 2.0;
            }
            for i in (n / 2 + 1)..n {
                buffer[i] = Complex::new(0.0, 0.0);
            }
        } else {
            for i in 1..(n + 1) / 2 {
                buffer[i] *= 2.0;
            }
            for i in ((n + 1) / 2)..n {
                buffer[i] = Complex::new(0.0, 0.0);
            }
        }

        fft_inv.process(&mut buffer);
        let scale = 1.0 / n as f64;
        let envelope: Vec<f64> = buffer.iter().map(|c| (c * scale).norm()).collect();
        let env_mean: f64 = envelope.iter().sum::<f64>() / envelope.len() as f64;
        let centered: Vec<f64> = envelope.iter().map(|e| e - env_mean).collect();

        let (freqs, mags) = self.compute_spectrum(&centered);
        Ok((
            Array1::from_vec(freqs).into_pyarray(py).unbind(),
            Array1::from_vec(mags).into_pyarray(py).unbind(),
        ))
    }

    fn order_analysis<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<'py, f64>,
        shaft_freq: f64,
    ) -> PyResult<Py<PyDict>> {
        let sig: Vec<f64> = signal.as_array().iter().copied().collect();
        let (freqs, mags) = self.compute_spectrum(&sig);

        let dict = PyDict::new(py);
        for order in 1..=6u32 {
            let target = order as f64 * shaft_freq;
            let idx = freqs
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    ((*a - target).abs())
                        .partial_cmp(&((*b - target).abs()))
                        .unwrap()
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            dict.set_item(format!("order_{}x", order), mags[idx])?;
        }
        Ok(dict.unbind())
    }

    fn extract_features<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyDict>> {
        let sig: Vec<f64> = signal.as_array().iter().copied().collect();
        let (freqs, mags) = self.compute_spectrum(&sig);
        let power: Vec<f64> = mags.iter().map(|m| m * m).collect();
        let total_power: f64 = power.iter().sum::<f64>() + 1e-10;

        let dict = PyDict::new(py);

        // Peak
        let (peak_idx, peak_mag) = mags
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, &v)| (i, v))
            .unwrap_or((0, 0.0));
        dict.set_item("peak_frequency", freqs[peak_idx])?;
        dict.set_item("peak_magnitude", peak_mag)?;

        // Spectral centroid
        let mag_sum: f64 = mags.iter().sum::<f64>() + 1e-10;
        let centroid: f64 = freqs
            .iter()
            .zip(mags.iter())
            .map(|(f, m)| f * m)
            .sum::<f64>()
            / mag_sum;
        dict.set_item("spectral_centroid", centroid)?;

        // Spread
        let spread: f64 = (freqs
            .iter()
            .zip(mags.iter())
            .map(|(f, m)| (f - centroid).powi(2) * m)
            .sum::<f64>()
            / mag_sum)
            .sqrt();
        dict.set_item("spectral_spread", spread)?;

        // Entropy
        let entropy: f64 = power
            .iter()
            .map(|p| p / total_power)
            .filter(|p| *p > 0.0)
            .map(|p| -p * p.log2())
            .sum();
        dict.set_item("spectral_entropy", entropy)?;

        // Flatness
        let log_sum: f64 = mags.iter().map(|m| (*m + 1e-10).ln()).sum::<f64>();
        let log_mean = log_sum / mags.len().max(1) as f64;
        let geo_mean = log_mean.exp();
        let arith_mean: f64 = mags.iter().sum::<f64>() / mags.len().max(1) as f64;
        dict.set_item("spectral_flatness", geo_mean / (arith_mean + 1e-10))?;

        // Rolloff 95%
        let mut cumsum = 0.0;
        let threshold = 0.95 * total_power;
        let mut rolloff_freq = 0.0;
        for (i, &p) in power.iter().enumerate() {
            cumsum += p;
            if cumsum >= threshold {
                rolloff_freq = freqs[i];
                break;
            }
        }
        dict.set_item("spectral_rolloff", rolloff_freq)?;

        Ok(dict.unbind())
    }

    fn bearing_frequencies<'py>(
        &self,
        py: Python<'py>,
        shaft_freq: f64,
        n_balls: usize,
        ball_dia: f64,
        pitch_dia: f64,
        contact_angle: f64,
    ) -> PyResult<Py<PyDict>> {
        let cos_angle = contact_angle.to_radians().cos();
        let ratio = ball_dia / pitch_dia;
        let bpfo = (n_balls as f64 / 2.0) * shaft_freq * (1.0 - ratio * cos_angle);
        let bpfi = (n_balls as f64 / 2.0) * shaft_freq * (1.0 + ratio * cos_angle);
        let bsf = (pitch_dia / (2.0 * ball_dia)) * shaft_freq * (1.0 - (ratio * cos_angle).powi(2));
        let ftf = (shaft_freq / 2.0) * (1.0 - ratio * cos_angle);
        let dict = PyDict::new(py);
        dict.set_item("BPFO", bpfo)?;
        dict.set_item("BPFI", bpfi)?;
        dict.set_item("BSF", bsf)?;
        dict.set_item("FTF", ftf)?;
        Ok(dict.unbind())
    }
}
