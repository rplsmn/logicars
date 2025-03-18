use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::difflogicca::{DiffLogicCA, create_game_of_life, create_glider};

/// Python wrapper for DiffLogicCA
#[pyclass]
pub struct PyDiffLogicCA {
    model: DiffLogicCA,
}

#[pymethods]
impl PyDiffLogicCA {
    #[new]
    fn new(width: usize, height: usize, state_size: usize, n_perception_circuits: usize) -> Self {
        PyDiffLogicCA {
            model: DiffLogicCA::new(width, height, state_size, n_perception_circuits),
        }
    }
    
    fn step(&mut self) {
        self.model.step();
    }
    
    fn get_grid<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray3<bool>>> {
        let grid = self.model.get_grid().to_owned();
        Ok(grid.into_pyarray(py).into())
    }
    
    fn set_grid(&mut self, grid: PyReadonlyArray3<bool>) {
        let array = grid.as_array().to_owned();
        self.model.set_grid(array);
    }
    
    fn create_glider(&mut self, row: usize, col: usize) {
        let mut grid = self.model.get_grid().to_owned();
        create_glider(&mut grid, row, col);
        self.model.set_grid(grid);
    }

    fn train(&mut self, initial_states: PyReadonlyArray4<bool>, target_states: PyReadonlyArray4<bool>, 
        learning_rate: f32, epochs: usize) {
        let initial_array = initial_states.as_array().to_owned();
        let target_array = target_states.as_array().to_owned();
        self.model.train(&initial_array, &target_array, learning_rate, epochs);
    }
    
    fn train_epoch(&mut self, initial_states: PyReadonlyArray4<bool>, target_states: PyReadonlyArray4<bool>,
        learning_rate: f32, epoch: usize) -> PyResult<(f32, f32)> {
        let initial_array = initial_states.as_array().to_owned();
        let target_array = target_states.as_array().to_owned();
        let (soft_loss, hard_loss) = self.model.train_epoch_internal(&initial_array, &target_array, learning_rate, epoch);
        Ok((soft_loss, hard_loss))
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.model.set_batch_size(batch_size);
    }
    
    fn set_l2_strength(&mut self, l2_strength: f32) {
        self.model.set_l2_strength(l2_strength);
    }
    
    fn set_temperature(&mut self, temperature: f32) {
        self.model.set_temperature(temperature);
    }
}

#[pyfunction]
pub fn create_gol(width: usize, height: usize) -> PyResult<PyDiffLogicCA> {
    Ok(PyDiffLogicCA {
        model: create_game_of_life(width, height),
    })
}

/// Register the module with Python
pub fn register(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDiffLogicCA>()?;
    m.add_function(wrap_pyfunction!(create_gol, py)?)?;
    
    Ok(())
}