use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn dense_program_accuracy(
    program: &[i16],
    gram: &[f64],
    cross: &[f64],
    validation: &[f64],
    groups: &[i32],
    train_ids: &[i32],
    feature_count: usize,
    class_count: usize,
    group_count: usize,
    features_per_site: usize,
    alpha: f64,
) -> Result<f64, String> {
    let k = program.len() * features_per_site + 1;
    let mut columns = Vec::with_capacity(k);
    for &channel in program {
        for feature in 0..features_per_site {
            columns.push(channel as usize * features_per_site + feature);
        }
    }
    columns.push(feature_count - 1);
    let mut lower = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut value = gram[columns[i] * feature_count + columns[j]];
            if i == j && i + 1 < k {
                value += alpha;
            }
            for t in 0..j {
                value -= lower[i * k + t] * lower[j * k + t];
            }
            if i == j {
                if !value.is_finite() || value <= 0.0 {
                    return Err("ridge gram is not positive definite".into());
                }
                lower[i * k + j] = value.sqrt();
            } else {
                lower[i * k + j] = value / lower[j * k + j];
            }
        }
    }
    let mut coefficients = vec![0.0; k * class_count];
    let mut work = vec![0.0; k];
    for class in 0..class_count {
        for i in 0..k {
            let mut value = cross[columns[i] * class_count + class];
            for j in 0..i {
                value -= lower[i * k + j] * work[j];
            }
            work[i] = value / lower[i * k + i];
        }
        for i in (0..k).rev() {
            let mut value = work[i];
            for j in (i + 1)..k {
                value -= lower[j * k + i] * coefficients[j * class_count + class];
            }
            coefficients[i * class_count + class] = value / lower[i * k + i];
        }
    }
    let mut hits = vec![0usize; group_count];
    let mut totals = vec![0usize; group_count];
    for (row, (&group, &target)) in groups.iter().zip(train_ids).enumerate() {
        let mut best_class = 0usize;
        let mut best_value = f64::NEG_INFINITY;
        for class in 0..class_count {
            let mut value = 0.0;
            for (local, &column) in columns.iter().enumerate() {
                value += validation[row * feature_count + column]
                    * coefficients[local * class_count + class];
            }
            if value > best_value {
                best_value = value;
                best_class = class;
            }
        }
        let group = group as usize;
        totals[group] += 1;
        if best_class as i32 == target {
            hits[group] += 1;
        }
    }
    Ok(hits
        .iter()
        .zip(totals.iter())
        .map(|(&hit, &total)| hit as f64 / total as f64)
        .sum::<f64>()
        / group_count as f64)
}

/// Balanced accuracy for many complete ridge programs on one prepared fold.
#[pyfunction]
fn score_dense_programs(
    py: Python<'_>,
    gram: PyReadonlyArray2<f64>,
    cross: PyReadonlyArray2<f64>,
    validation: PyReadonlyArray2<f64>,
    groups: PyReadonlyArray1<i32>,
    train_ids: PyReadonlyArray1<i32>,
    designs: PyReadonlyArray2<i16>,
    features_per_site: usize,
    alpha: f64,
    group_count: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let g_shape = gram.shape();
    let c_shape = cross.shape();
    let v_shape = validation.shape();
    let d_shape = designs.shape();
    if g_shape[0] < 2
        || g_shape[0] != g_shape[1]
        || c_shape[0] != g_shape[0]
        || v_shape[1] != g_shape[0]
        || v_shape[0] < 1
        || c_shape[1] < 1
        || d_shape[0] < 1
        || d_shape[1] < 1
        || features_per_site < 1
        || group_count < 1
        || !alpha.is_finite()
        || alpha <= 0.0
        || (g_shape[0] - 1) % features_per_site != 0
        || groups.len() != v_shape[0]
        || train_ids.len() != v_shape[0]
    {
        return Err(PyValueError::new_err("invalid dense program fold"));
    }
    let gram = gram.as_slice()?;
    let cross = cross.as_slice()?;
    let validation = validation.as_slice()?;
    let groups = groups.as_slice()?;
    let train_ids = train_ids.as_slice()?;
    let designs = designs.as_slice()?;
    let channel_count = (g_shape[0] - 1) / features_per_site;
    let mut group_counts = vec![0usize; group_count];
    for &group in groups {
        if group >= 0 && (group as usize) < group_count {
            group_counts[group as usize] += 1;
        }
    }
    if d_shape[1] > channel_count
        || group_counts.contains(&0)
        || groups
            .iter()
            .any(|&group| group < 0 || group as usize >= group_count)
        || train_ids
            .iter()
            .any(|&id| id < -1 || id >= c_shape[1] as i32)
        || designs.chunks_exact(d_shape[1]).any(|row| {
            row.iter()
                .any(|&channel| channel < 0 || channel as usize >= channel_count)
                || row
                    .iter()
                    .enumerate()
                    .any(|(i, channel)| row[..i].contains(channel))
        })
    {
        return Err(PyValueError::new_err(
            "invalid program or validation labels",
        ));
    }
    let scores: Vec<Result<f64, String>> = py.detach(|| {
        designs
            .par_chunks_exact(d_shape[1])
            .map(|program| {
                dense_program_accuracy(
                    program,
                    gram,
                    cross,
                    validation,
                    groups,
                    train_ids,
                    g_shape[0],
                    c_shape[1],
                    group_count,
                    features_per_site,
                    alpha,
                )
            })
            .collect()
    });
    let scores: Vec<f64> = scores
        .into_iter()
        .collect::<Result<_, _>>()
        .map_err(PyValueError::new_err)?;
    Ok(scores.into_pyarray(py).unbind())
}

#[pyfunction]
fn decode_gaddy_flac(py: Python<'_>, path: &str) -> PyResult<Py<PyArray1<f64>>> {
    let mut reader =
        claxon::FlacReader::open(path).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let info = reader.streaminfo();
    if info.channels != 1 || info.sample_rate != 16000 || info.bits_per_sample != 16 {
        return Err(PyValueError::new_err(
            "Gaddy energy path requires mono 16-kHz 16-bit FLAC",
        ));
    }
    let mut audio = Vec::with_capacity(info.samples.unwrap_or(0) as usize);
    for sample in reader.samples() {
        let value = sample.map_err(|error| PyValueError::new_err(error.to_string()))?;
        audio.push(((value as f32) / 32768.0) as f64);
    }
    Ok(audio.into_pyarray(py).unbind())
}

/// Match the energy and EMG feature columns of gaddy_prosody_probe.trial_frames.
/// The acoustic pitch diagnostic is intentionally separate; contour experiments
/// do not need to spend most of their framing time estimating pitch.
#[pyfunction]
fn extract_energy_features(
    py: Python<'_>,
    emg: PyReadonlyArray2<f64>,
    audio: PyReadonlyArray1<f64>,
    emg_sample_rate: f64,
    centers: PyReadonlyArray1<f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    let shape = emg.shape();
    if shape.len() != 2 || shape[1] != 8 || !emg_sample_rate.is_finite() || emg_sample_rate <= 0.0 {
        return Err(PyValueError::new_err(
            "expected eight EMG channels and positive sample rate",
        ));
    }
    let emg = emg.as_slice()?;
    let audio = audio.as_slice()?;
    let centers = centers.as_slice()?;
    let window = (0.1 * emg_sample_rate).round() as usize;
    if window < 2 {
        return Err(PyValueError::new_err(
            "EMG window must contain at least two samples",
        ));
    }
    let mut features = Vec::new();
    let mut energy = Vec::new();
    for &center in centers {
        if !center.is_finite() || center < 0.0 {
            return Err(PyValueError::new_err(
                "frame centers must be finite and nonnegative",
            ));
        }
        let emg_start = (center * emg_sample_rate) as isize - (window / 2) as isize;
        let audio_start = (center * 16000.0) as isize - 320;
        if emg_start < 0 || audio_start < 0 {
            continue;
        }
        let emg_start = emg_start as usize;
        let audio_start = audio_start as usize;
        if emg_start + window > emg.len() / 8 || audio_start + 640 > audio.len() {
            continue;
        }
        let mut sum_square = [0.0f64; 8];
        let mut waveform_length = [0.0f64; 8];
        for sample in 0..window {
            let current = (emg_start + sample) * 8;
            for channel in 0..8 {
                let value = emg[current + channel];
                sum_square[channel] += value * value;
                if sample > 0 {
                    let previous = emg[(emg_start + sample - 1) * 8 + channel];
                    waveform_length[channel] += (value - previous).abs();
                }
            }
        }
        for &value in &sum_square {
            features.push((value / window as f64).sqrt().ln_1p());
        }
        for &value in &waveform_length {
            features.push((value / (window - 1) as f64).ln_1p());
        }
        let mut audio_square = 0.0;
        for &value in &audio[audio_start..audio_start + 640] {
            audio_square += value * value;
        }
        energy.push((audio_square / 640.0).sqrt().max(1e-6).log10());
    }
    let feature_array = Array2::from_shape_vec((energy.len(), 16), features)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        feature_array.into_pyarray(py).unbind(),
        energy.into_pyarray(py).unbind(),
    ))
}

#[pyfunction]
fn scatter_unit_multichannel(
    mut x: PyReadwriteArray2<f32>,
    spike_idx: PyReadonlyArray1<i32>,
    amp_base: f32,
    env: PyReadonlyArray1<f32>,
    m: PyReadonlyArray1<f32>,
    wi: PyReadonlyArray1<f32>,
    n: usize,
    c: usize,
    li: usize,
) -> PyResult<()> {
    let mut x_arr = x.as_array_mut();
    let env = env.as_slice()?;
    let m = m.as_slice()?;
    let wi = wi.as_slice()?;
    for &t0_raw in spike_idx.as_slice()? {
        let t0 = t0_raw as usize;
        if t0 >= n {
            continue;
        }
        let a = amp_base * env[t0];
        let end = (t0 + li).min(n);
        let sl = end.saturating_sub(t0);
        if sl == 0 {
            continue;
        }
        for ch in 0..c {
            let w = wi[ch];
            for j in 0..sl {
                x_arr[[t0 + j, ch]] += a * w * m[j];
            }
        }
    }
    Ok(())
}

#[pyfunction]
fn scatter_unit_delayed(
    mut x: PyReadwriteArray2<f32>,
    spike_idx: PyReadonlyArray1<i32>,
    amp_base: f32,
    env: PyReadonlyArray1<f32>,
    m: PyReadonlyArray1<f32>,
    wi: PyReadonlyArray1<f32>,
    di: PyReadonlyArray1<i32>,
    n: usize,
    c: usize,
    li: usize,
) -> PyResult<()> {
    let mut x_arr = x.as_array_mut();
    let env = env.as_slice()?;
    let m = m.as_slice()?;
    let wi = wi.as_slice()?;
    let di = di.as_slice()?;
    for &t0_raw in spike_idx.as_slice()? {
        let t0 = t0_raw as usize;
        if t0 >= n {
            continue;
        }
        let a = amp_base * env[t0];
        for ch in 0..c {
            let start = t0 as i32 + di[ch];
            if start < 0 {
                continue;
            }
            let start = start as usize;
            if start >= n {
                continue;
            }
            let end = (start + li).min(n);
            let sl = end.saturating_sub(start);
            if sl == 0 {
                continue;
            }
            let w = wi[ch];
            for j in 0..sl {
                x_arr[[start + j, ch]] += a * w * m[j];
            }
        }
    }
    Ok(())
}

#[pyfunction]
fn scatter_pool_batched(
    mut x: PyReadwriteArray2<f32>,
    mu_idx: PyReadonlyArray1<i32>,
    spike_t0: PyReadonlyArray1<i32>,
    spike_amp: PyReadonlyArray1<f32>,
    tpl: PyReadonlyArray2<f32>,
    lengths: PyReadonlyArray1<i32>,
    w: PyReadonlyArray2<f32>,
    dly: PyReadonlyArray2<i32>,
    n: usize,
    c: usize,
    has_delays: bool,
) -> PyResult<()> {
    let mut x_arr = x.as_array_mut();
    let tpl_arr = tpl.as_array();
    let w_arr = w.as_array();
    let dly_arr = dly.as_array();
    let mu = mu_idx.as_slice()?;
    let t0s = spike_t0.as_slice()?;
    let amps = spike_amp.as_slice()?;
    let lens = lengths.as_slice()?;
    for e in 0..mu.len() {
        let i = mu[e] as usize;
        let t0 = t0s[e] as usize;
        if t0 >= n {
            continue;
        }
        let a = amps[e];
        let li = lens[i] as usize;
        if li == 0 {
            continue;
        }
        if has_delays {
            for ch in 0..c {
                let start = t0s[e] + dly_arr[[i, ch]];
                if start < 0 {
                    continue;
                }
                let start = start as usize;
                if start >= n {
                    continue;
                }
                let end = (start + li).min(n);
                for j in 0..(end - start) {
                    x_arr[[start + j, ch]] += a * w_arr[[i, ch]] * tpl_arr[[i, j]];
                }
            }
        } else {
            let end = (t0 + li).min(n);
            let sl = end - t0;
            if sl == 0 {
                continue;
            }
            for ch in 0..c {
                let wf = w_arr[[i, ch]];
                for j in 0..sl {
                    x_arr[[t0 + j, ch]] += a * wf * tpl_arr[[i, j]];
                }
            }
        }
    }
    Ok(())
}

#[pymodule]
fn openalterego_accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(score_dense_programs, m)?)?;
    m.add_function(wrap_pyfunction!(decode_gaddy_flac, m)?)?;
    m.add_function(wrap_pyfunction!(extract_energy_features, m)?)?;
    m.add_function(wrap_pyfunction!(scatter_unit_multichannel, m)?)?;
    m.add_function(wrap_pyfunction!(scatter_unit_delayed, m)?)?;
    m.add_function(wrap_pyfunction!(scatter_pool_batched, m)?)?;
    Ok(())
}
