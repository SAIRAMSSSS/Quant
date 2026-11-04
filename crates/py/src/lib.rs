use pyo3::prelude::*;

#[pymodule]
fn quant_engine(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
