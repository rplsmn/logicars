//! N-bit Grid and Neighborhood for multi-channel cellular automata
//!
//! This module implements channel-aware grids that support 1-128 bits per cell,
//! as required by the paper's experiments:
//! - Game of Life: C=1
//! - Checkerboard: C=8
//! - Colored G: C=64
//! - Growing Lizard: C=128
//!
//! Design: Dynamic channels (runtime) for flexibility across experiments.

/// Boundary condition for grid edges
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// Toroidal: edges wrap around (Game of Life, Growing Lizard)
    Periodic,
    /// Edges are clamped to edge values (Checkerboard)
    NonPeriodic,
}

/// N-bit grid with C channels per cell
///
/// Cells are stored in soft (f64) representation for training.
/// Use `to_hard()` to convert to discrete values for inference.
#[derive(Debug, Clone)]
pub struct NGrid {
    /// Grid width
    pub width: usize,
    /// Grid height
    pub height: usize,
    /// Number of channels per cell (1-128)
    pub channels: usize,
    /// Boundary condition
    pub boundary: BoundaryCondition,
    /// Flat storage: [cell0_ch0, cell0_ch1, ..., cell1_ch0, ...]
    /// Length = width * height * channels
    cells: Vec<f64>,
}

impl NGrid {
    /// Create a new grid with all cells initialized to 0.0
    pub fn new(width: usize, height: usize, channels: usize, boundary: BoundaryCondition) -> Self {
        assert!(channels >= 1 && channels <= 128, "Channels must be 1-128");
        Self {
            width,
            height,
            channels,
            boundary,
            cells: vec![0.0; width * height * channels],
        }
    }

    /// Create a periodic grid (common case for GoL, Lizard)
    pub fn periodic(width: usize, height: usize, channels: usize) -> Self {
        Self::new(width, height, channels, BoundaryCondition::Periodic)
    }

    /// Create a non-periodic grid (Checkerboard)
    pub fn non_periodic(width: usize, height: usize, channels: usize) -> Self {
        Self::new(width, height, channels, BoundaryCondition::NonPeriodic)
    }

    /// Create from raw cell data
    pub fn from_cells(
        width: usize,
        height: usize,
        channels: usize,
        boundary: BoundaryCondition,
        cells: Vec<f64>,
    ) -> Self {
        assert_eq!(
            cells.len(),
            width * height * channels,
            "Cell data length mismatch: expected {}, got {}",
            width * height * channels,
            cells.len()
        );
        Self {
            width,
            height,
            channels,
            boundary,
            cells,
        }
    }

    /// Total number of cells
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.width * self.height
    }

    /// Total number of values (cells × channels)
    #[inline]
    pub fn num_values(&self) -> usize {
        self.cells.len()
    }

    /// Index into flat storage for cell (x, y) and channel c
    #[inline]
    fn cell_index(&self, x: usize, y: usize, c: usize) -> usize {
        (y * self.width + x) * self.channels + c
    }

    /// Get a single channel value at (x, y)
    ///
    /// For periodic grids: coordinates wrap around
    /// For non-periodic grids: coordinates are clamped to edges
    #[inline]
    pub fn get(&self, x: isize, y: isize, channel: usize) -> f64 {
        debug_assert!(channel < self.channels);
        let (x, y) = self.resolve_coords(x, y);
        self.cells[self.cell_index(x, y, channel)]
    }

    /// Set a single channel value at (x, y)
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, channel: usize, value: f64) {
        debug_assert!(channel < self.channels);
        let idx = self.cell_index(x, y, channel);
        self.cells[idx] = value;
    }

    /// Get all channels for a cell at (x, y)
    pub fn get_cell(&self, x: isize, y: isize) -> Vec<f64> {
        let (x, y) = self.resolve_coords(x, y);
        let start = self.cell_index(x, y, 0);
        self.cells[start..start + self.channels].to_vec()
    }

    /// Set all channels for a cell at (x, y)
    pub fn set_cell(&mut self, x: usize, y: usize, values: &[f64]) {
        assert_eq!(values.len(), self.channels);
        let start = self.cell_index(x, y, 0);
        self.cells[start..start + self.channels].copy_from_slice(values);
    }

    /// Get all channels for a cell as a fixed-size array (for common sizes)
    pub fn get_cell_array<const C: usize>(&self, x: isize, y: isize) -> [f64; C] {
        assert_eq!(C, self.channels, "Array size must match channel count");
        let (x, y) = self.resolve_coords(x, y);
        let start = self.cell_index(x, y, 0);
        let mut arr = [0.0; C];
        arr.copy_from_slice(&self.cells[start..start + C]);
        arr
    }

    /// Resolve coordinates based on boundary condition
    #[inline]
    fn resolve_coords(&self, x: isize, y: isize) -> (usize, usize) {
        match self.boundary {
            BoundaryCondition::Periodic => {
                let x = x.rem_euclid(self.width as isize) as usize;
                let y = y.rem_euclid(self.height as isize) as usize;
                (x, y)
            }
            BoundaryCondition::NonPeriodic => {
                let x = x.clamp(0, self.width as isize - 1) as usize;
                let y = y.clamp(0, self.height as isize - 1) as usize;
                (x, y)
            }
        }
    }

    /// Extract 3×3 neighborhood around (cx, cy)
    ///
    /// Returns NNeighborhood containing 9 cells × C channels = 9C values
    /// Order: [NW, N, NE, W, C, E, SW, S, SE] (reading order)
    pub fn neighborhood(&self, cx: usize, cy: usize) -> NNeighborhood {
        let cx = cx as isize;
        let cy = cy as isize;

        // Offsets for 3×3 in reading order
        let offsets: [(isize, isize); 9] = [
            (-1, -1), // NW
            (0, -1),  // N
            (1, -1),  // NE
            (-1, 0),  // W
            (0, 0),   // C (center)
            (1, 0),   // E
            (-1, 1),  // SW
            (0, 1),   // S
            (1, 1),   // SE
        ];

        let mut cells = Vec::with_capacity(9 * self.channels);
        for (dx, dy) in offsets {
            let cell = self.get_cell(cx + dx, cy + dy);
            cells.extend(cell);
        }

        NNeighborhood {
            channels: self.channels,
            cells,
        }
    }

    /// Raw access to underlying data
    pub fn raw_data(&self) -> &[f64] {
        &self.cells
    }

    /// Mutable raw access
    pub fn raw_data_mut(&mut self) -> &mut [f64] {
        &mut self.cells
    }

    /// Convert to hard (discrete) grid
    /// Values > 0.5 become 1.0, others become 0.0
    pub fn to_hard(&self) -> NGrid {
        let hard_cells: Vec<f64> = self.cells.iter().map(|&v| if v > 0.5 { 1.0 } else { 0.0 }).collect();
        NGrid {
            width: self.width,
            height: self.height,
            channels: self.channels,
            boundary: self.boundary,
            cells: hard_cells,
        }
    }

    /// Convert to boolean grid (for C=1 case)
    pub fn to_bool_grid(&self) -> Vec<bool> {
        assert_eq!(self.channels, 1, "to_bool_grid requires C=1");
        self.cells.iter().map(|&v| v > 0.5).collect()
    }

    /// Create from boolean grid (for C=1 case)
    pub fn from_bool_grid(
        width: usize,
        height: usize,
        boundary: BoundaryCondition,
        bools: &[bool],
    ) -> Self {
        assert_eq!(bools.len(), width * height);
        let cells: Vec<f64> = bools.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        Self {
            width,
            height,
            channels: 1,
            boundary,
            cells,
        }
    }
}

/// 3×3 neighborhood with C channels per cell
///
/// Contains 9 cells × C channels = 9C values in soft representation.
/// Order: [NW_ch0, NW_ch1, ..., N_ch0, ..., SE_chC-1]
#[derive(Debug, Clone)]
pub struct NNeighborhood {
    /// Number of channels per cell
    pub channels: usize,
    /// Flat storage: 9 cells × C channels
    /// Order: [NW_ch0..NW_chC, N_ch0..N_chC, ..., SE_ch0..SE_chC]
    pub cells: Vec<f64>,
}

impl NNeighborhood {
    /// Create from raw cell data
    pub fn new(channels: usize, cells: Vec<f64>) -> Self {
        assert_eq!(cells.len(), 9 * channels);
        Self { channels, cells }
    }

    /// Create from a 9-element array of C-channel cells
    pub fn from_cells<const C: usize>(cells: [[f64; C]; 9]) -> Self {
        let mut flat = Vec::with_capacity(9 * C);
        for cell in cells {
            flat.extend(cell);
        }
        Self {
            channels: C,
            cells: flat,
        }
    }

    /// Get all channels for a specific cell position (0-8)
    /// 0=NW, 1=N, 2=NE, 3=W, 4=C, 5=E, 6=SW, 7=S, 8=SE
    pub fn get_cell(&self, position: usize) -> &[f64] {
        let start = position * self.channels;
        &self.cells[start..start + self.channels]
    }

    /// Get a single channel from a specific cell position
    pub fn get(&self, position: usize, channel: usize) -> f64 {
        self.cells[position * self.channels + channel]
    }

    /// Get center cell (position 4)
    pub fn center(&self) -> &[f64] {
        self.get_cell(4)
    }

    /// Get center cell as array
    pub fn center_array<const C: usize>(&self) -> [f64; C] {
        assert_eq!(C, self.channels);
        let slice = self.center();
        let mut arr = [0.0; C];
        arr.copy_from_slice(slice);
        arr
    }

    /// Convert to hard (discrete) values
    pub fn to_hard(&self) -> NNeighborhood {
        let hard_cells: Vec<f64> = self.cells.iter().map(|&v| if v > 0.5 { 1.0 } else { 0.0 }).collect();
        NNeighborhood {
            channels: self.channels,
            cells: hard_cells,
        }
    }

    /// For C=1: convert to simple [bool; 9]
    pub fn to_bool_array(&self) -> [bool; 9] {
        assert_eq!(self.channels, 1);
        let mut arr = [false; 9];
        for i in 0..9 {
            arr[i] = self.cells[i] > 0.5;
        }
        arr
    }

    /// For C=1: convert to simple [f64; 9]
    pub fn to_f64_array(&self) -> [f64; 9] {
        assert_eq!(self.channels, 1);
        let mut arr = [0.0; 9];
        arr.copy_from_slice(&self.cells);
        arr
    }

    /// For C=1: create from index (0-511) like the old Neighborhood
    pub fn from_gol_index(idx: usize) -> Self {
        let mut cells = vec![0.0; 9];
        for i in 0..9 {
            cells[i] = if (idx >> i) & 1 == 1 { 1.0 } else { 0.0 };
        }
        Self { channels: 1, cells }
    }

    /// For C=1: convert to index (0-511)
    pub fn to_gol_index(&self) -> usize {
        assert_eq!(self.channels, 1);
        let mut idx = 0;
        for i in 0..9 {
            if self.cells[i] > 0.5 {
                idx |= 1 << i;
            }
        }
        idx
    }

    /// Count alive neighbors (excluding center) - for C=1 GoL
    pub fn gol_neighbor_count(&self) -> u8 {
        assert_eq!(self.channels, 1);
        let mut count = 0u8;
        for i in 0..9 {
            if i != 4 && self.cells[i] > 0.5 {
                count += 1;
            }
        }
        count
    }

    /// Apply Game of Life rule - for C=1
    pub fn gol_next_state(&self) -> bool {
        assert_eq!(self.channels, 1);
        let count = self.gol_neighbor_count();
        let center_alive = self.cells[4] > 0.5;

        if center_alive {
            count == 2 || count == 3
        } else {
            count == 3
        }
    }

    /// Total number of values (9 × channels)
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Raw data access
    pub fn raw_data(&self) -> &[f64] {
        &self.cells
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== C=1 Tests (GoL validation) ====================

    #[test]
    fn test_ngrid_creation_c1() {
        let grid = NGrid::periodic(10, 10, 1);
        assert_eq!(grid.width, 10);
        assert_eq!(grid.height, 10);
        assert_eq!(grid.channels, 1);
        assert_eq!(grid.num_cells(), 100);
        assert_eq!(grid.num_values(), 100);
    }

    #[test]
    fn test_ngrid_periodic_wrapping_c1() {
        let mut grid = NGrid::periodic(3, 3, 1);
        grid.set(0, 0, 0, 1.0);

        // Test wrapping
        assert_relative_eq!(grid.get(0, 0, 0), 1.0);
        assert_relative_eq!(grid.get(-3, 0, 0), 1.0); // Wraps to (0, 0)
        assert_relative_eq!(grid.get(3, 0, 0), 1.0);  // Wraps to (0, 0)
        assert_relative_eq!(grid.get(0, -3, 0), 1.0); // Wraps to (0, 0)
        assert_relative_eq!(grid.get(0, 3, 0), 1.0);  // Wraps to (0, 0)
    }

    #[test]
    fn test_ngrid_non_periodic_clamping_c1() {
        let mut grid = NGrid::non_periodic(3, 3, 1);
        grid.set(0, 0, 0, 1.0);
        grid.set(2, 2, 0, 0.5);

        // Test clamping
        assert_relative_eq!(grid.get(-5, 0, 0), 1.0);  // Clamps to (0, 0)
        assert_relative_eq!(grid.get(10, 10, 0), 0.5); // Clamps to (2, 2)
    }

    #[test]
    fn test_neighborhood_extraction_c1() {
        let mut grid = NGrid::periodic(5, 5, 1);
        grid.set(2, 2, 0, 1.0); // Center
        grid.set(1, 1, 0, 1.0); // NW
        grid.set(3, 3, 0, 1.0); // SE

        let n = grid.neighborhood(2, 2);
        assert_eq!(n.channels, 1);
        assert_eq!(n.len(), 9);

        // Check positions: 0=NW, 4=C, 8=SE
        assert_relative_eq!(n.get(0, 0), 1.0); // NW
        assert_relative_eq!(n.get(4, 0), 1.0); // C
        assert_relative_eq!(n.get(8, 0), 1.0); // SE
        assert_relative_eq!(n.get(1, 0), 0.0); // N
    }

    #[test]
    fn test_gol_index_roundtrip() {
        for idx in 0..512 {
            let n = NNeighborhood::from_gol_index(idx);
            assert_eq!(n.to_gol_index(), idx);
        }
    }

    #[test]
    fn test_gol_rules_via_nneighborhood() {
        // Dead cell with 3 neighbors -> alive
        let n = NNeighborhood::from_gol_index(0b000000111); // NW, N, NE alive
        assert_eq!(n.gol_neighbor_count(), 3);
        assert!(n.gol_next_state());

        // Alive cell with 2 neighbors -> survives
        let n = NNeighborhood::from_gol_index(0b000010011); // NW, N alive, center alive
        assert_eq!(n.gol_neighbor_count(), 2);
        assert!(n.gol_next_state());

        // Alive cell with 1 neighbor -> dies
        let n = NNeighborhood::from_gol_index(0b000010001); // NW alive, center alive
        assert_eq!(n.gol_neighbor_count(), 1);
        assert!(!n.gol_next_state());

        // Alive cell with 4 neighbors -> dies (overcrowding)
        let n = NNeighborhood::from_gol_index(0b000110111); // NW,N,NE,W alive, center alive
        assert_eq!(n.gol_neighbor_count(), 4);
        assert!(!n.gol_next_state());
    }

    #[test]
    fn test_to_hard_c1() {
        let mut grid = NGrid::periodic(2, 2, 1);
        grid.set(0, 0, 0, 0.7);
        grid.set(1, 0, 0, 0.3);
        grid.set(0, 1, 0, 0.5);
        grid.set(1, 1, 0, 0.51);

        let hard = grid.to_hard();
        assert_relative_eq!(hard.get(0, 0, 0), 1.0);
        assert_relative_eq!(hard.get(1, 0, 0), 0.0);
        assert_relative_eq!(hard.get(0, 1, 0), 0.0); // 0.5 -> 0.0
        assert_relative_eq!(hard.get(1, 1, 0), 1.0);
    }

    // ==================== C=8 Tests (Checkerboard) ====================

    #[test]
    fn test_ngrid_creation_c8() {
        let grid = NGrid::non_periodic(16, 16, 8);
        assert_eq!(grid.width, 16);
        assert_eq!(grid.height, 16);
        assert_eq!(grid.channels, 8);
        assert_eq!(grid.num_cells(), 256);
        assert_eq!(grid.num_values(), 256 * 8);
    }

    #[test]
    fn test_ngrid_multichannel_access_c8() {
        let mut grid = NGrid::non_periodic(3, 3, 8);

        // Set different values for each channel
        let cell_values: [f64; 8] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        grid.set_cell(1, 1, &cell_values);

        // Verify
        let retrieved = grid.get_cell(1, 1);
        for i in 0..8 {
            assert_relative_eq!(retrieved[i], cell_values[i]);
        }

        // Test individual channel access
        for i in 0..8 {
            assert_relative_eq!(grid.get(1, 1, i), cell_values[i]);
        }
    }

    #[test]
    fn test_neighborhood_extraction_c8() {
        let mut grid = NGrid::non_periodic(5, 5, 8);

        // Set center with distinct channels
        let center_values: Vec<f64> = (0..8).map(|i| (i + 1) as f64 * 0.1).collect();
        grid.set_cell(2, 2, &center_values);

        // Set NW neighbor
        let nw_values: Vec<f64> = (0..8).map(|i| (i + 1) as f64 * 0.01).collect();
        grid.set_cell(1, 1, &nw_values);

        let n = grid.neighborhood(2, 2);
        assert_eq!(n.channels, 8);
        assert_eq!(n.len(), 9 * 8);

        // Check center (position 4)
        for i in 0..8 {
            assert_relative_eq!(n.get(4, i), center_values[i]);
        }

        // Check NW (position 0)
        for i in 0..8 {
            assert_relative_eq!(n.get(0, i), nw_values[i]);
        }
    }

    #[test]
    fn test_non_periodic_edge_behavior_c8() {
        let mut grid = NGrid::non_periodic(3, 3, 8);

        // Set corner with values
        let corner_values: Vec<f64> = vec![1.0; 8];
        grid.set_cell(0, 0, &corner_values);

        // Neighborhood at corner should clamp
        let n = grid.neighborhood(0, 0);

        // NW, N, NE, W should all clamp to edge values
        // For corner (0,0): NW(-1,-1) clamps to (0,0)
        let center = n.get_cell(4);
        let nw = n.get_cell(0);
        // Both should be the corner value since they clamp
        for i in 0..8 {
            assert_relative_eq!(center[i], nw[i]);
        }
    }

    // ==================== C=64 Tests (Colored G) ====================

    #[test]
    fn test_ngrid_creation_c64() {
        let grid = NGrid::periodic(20, 20, 64);
        assert_eq!(grid.width, 20);
        assert_eq!(grid.height, 20);
        assert_eq!(grid.channels, 64);
        assert_eq!(grid.num_cells(), 400);
        assert_eq!(grid.num_values(), 400 * 64);
    }

    #[test]
    fn test_ngrid_multichannel_c64() {
        let mut grid = NGrid::periodic(3, 3, 64);

        // Set cell with 64 channels
        let cell_values: Vec<f64> = (0..64).map(|i| i as f64 / 64.0).collect();
        grid.set_cell(1, 1, &cell_values);

        let retrieved = grid.get_cell(1, 1);
        for i in 0..64 {
            assert_relative_eq!(retrieved[i], cell_values[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_neighborhood_extraction_c64() {
        let grid = NGrid::periodic(5, 5, 64);
        let n = grid.neighborhood(2, 2);

        assert_eq!(n.channels, 64);
        assert_eq!(n.len(), 9 * 64);
    }

    // ==================== C=128 Tests (Growing Lizard) ====================

    #[test]
    fn test_ngrid_creation_c128() {
        let grid = NGrid::periodic(20, 20, 128);
        assert_eq!(grid.width, 20);
        assert_eq!(grid.height, 20);
        assert_eq!(grid.channels, 128);
        assert_eq!(grid.num_cells(), 400);
        assert_eq!(grid.num_values(), 400 * 128);
    }

    #[test]
    fn test_ngrid_multichannel_c128() {
        let mut grid = NGrid::periodic(3, 3, 128);

        // Set cell with 128 channels
        let cell_values: Vec<f64> = (0..128).map(|i| i as f64 / 128.0).collect();
        grid.set_cell(1, 1, &cell_values);

        let retrieved = grid.get_cell(1, 1);
        for i in 0..128 {
            assert_relative_eq!(retrieved[i], cell_values[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_neighborhood_extraction_c128() {
        let grid = NGrid::periodic(5, 5, 128);
        let n = grid.neighborhood(2, 2);

        assert_eq!(n.channels, 128);
        assert_eq!(n.len(), 9 * 128);
        assert_eq!(n.len(), 1152); // 9 * 128
    }

    #[test]
    fn test_large_grid_c128() {
        // Test 40×40 grid (size used for Lizard generalization)
        let grid = NGrid::periodic(40, 40, 128);
        assert_eq!(grid.num_cells(), 1600);
        assert_eq!(grid.num_values(), 1600 * 128);

        // Verify neighborhood extraction works at edges
        let n = grid.neighborhood(0, 0);
        assert_eq!(n.channels, 128);

        let n = grid.neighborhood(39, 39);
        assert_eq!(n.channels, 128);
    }

    // ==================== Compatibility Tests ====================

    #[test]
    fn test_bool_grid_conversion() {
        let bools = vec![true, false, true, false, true, false, true, false, true];
        let grid = NGrid::from_bool_grid(3, 3, BoundaryCondition::Periodic, &bools);

        assert_eq!(grid.channels, 1);
        let back = grid.to_bool_grid();
        assert_eq!(back, bools);
    }

    #[test]
    fn test_nneighborhood_to_arrays_c1() {
        let n = NNeighborhood::from_gol_index(0b101010101);

        let f64_arr = n.to_f64_array();
        let bool_arr = n.to_bool_array();

        for i in 0..9 {
            let expected = (i % 2) == 0;
            assert_eq!(bool_arr[i], expected);
            assert_relative_eq!(f64_arr[i], if expected { 1.0 } else { 0.0 });
        }
    }

    #[test]
    fn test_center_extraction() {
        let mut grid = NGrid::periodic(5, 5, 8);
        let center_values: Vec<f64> = (0..8).map(|i| i as f64).collect();
        grid.set_cell(2, 2, &center_values);

        let n = grid.neighborhood(2, 2);
        let center = n.center();

        for i in 0..8 {
            assert_relative_eq!(center[i], i as f64);
        }
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_minimal_grid() {
        let grid = NGrid::periodic(1, 1, 1);
        assert_eq!(grid.num_cells(), 1);

        // All neighbors should be the same cell (wrapping)
        let n = grid.neighborhood(0, 0);
        for i in 0..9 {
            assert_relative_eq!(n.get(i, 0), 0.0);
        }
    }

    #[test]
    fn test_2x2_grid_wrapping() {
        let mut grid = NGrid::periodic(2, 2, 1);
        grid.set(0, 0, 0, 1.0); // Top-left
        grid.set(1, 1, 0, 1.0); // Bottom-right

        // Neighborhood at (0,0) should wrap correctly
        let n = grid.neighborhood(0, 0);

        // Center is (0,0) = 1.0
        assert_relative_eq!(n.get(4, 0), 1.0);

        // SE is (1,1) = 1.0
        assert_relative_eq!(n.get(8, 0), 1.0);

        // NW is (-1,-1) = (1,1) = 1.0 due to wrapping
        assert_relative_eq!(n.get(0, 0), 1.0);
    }

    #[test]
    #[should_panic(expected = "Channels must be 1-128")]
    fn test_invalid_channels_0() {
        NGrid::periodic(3, 3, 0);
    }

    #[test]
    #[should_panic(expected = "Channels must be 1-128")]
    fn test_invalid_channels_129() {
        NGrid::periodic(3, 3, 129);
    }
}
