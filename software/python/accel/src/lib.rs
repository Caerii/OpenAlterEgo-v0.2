use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::prelude::*;

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
    m.add_function(wrap_pyfunction!(scatter_unit_multichannel, m)?)?;
    m.add_function(wrap_pyfunction!(scatter_unit_delayed, m)?)?;
    m.add_function(wrap_pyfunction!(scatter_pool_batched, m)?)?;
    Ok(())
}
