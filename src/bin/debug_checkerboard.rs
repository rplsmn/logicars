//! Debug checkerboard target - saves image for visual inspection
use image::{GrayImage, Luma};
use logicars::{
    create_checkerboard, create_random_seed, create_small_checkerboard_model, NGrid, SimpleRng,
    TrainingConfig, TrainingLoop, CHECKERBOARD_CHANNELS, CHECKERBOARD_GRID_SIZE,
    CHECKERBOARD_SQUARE_SIZE, CHECKERBOARD_SYNC_STEPS,
};

fn save_grid_as_image(grid: &NGrid, filename: &str, scale: u32) {
    let w = grid.width as u32;
    let h = grid.height as u32;
    let mut img = GrayImage::new(w * scale, h * scale);

    for y in 0..h {
        for x in 0..w {
            // Channel 0 only (the one we train on)
            let val = grid.get(x as isize, y as isize, 0);
            let pixel = (val * 255.0).clamp(0.0, 255.0) as u8;

            // Scale up for visibility
            for sy in 0..scale {
                for sx in 0..scale {
                    img.put_pixel(x * scale + sx, y * scale + sy, Luma([pixel]));
                }
            }
        }
    }

    img.save(filename).expect("Failed to save image");
    println!("Saved: {}", filename);
}

fn main() {
    println!("=== Checkerboard Debug ===\n");

    // Create and save target
    let target = create_checkerboard(
        CHECKERBOARD_GRID_SIZE,
        CHECKERBOARD_SQUARE_SIZE,
        CHECKERBOARD_CHANNELS,
    );
    save_grid_as_image(&target, "debug_target.png", 16);

    // Create random seed and save it
    let mut rng = SimpleRng::new(42);
    let seed = create_random_seed(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_CHANNELS, &mut rng);
    save_grid_as_image(&seed, "debug_seed.png", 16);

    // Run untrained model and save output
    let model = create_small_checkerboard_model();
    let config = TrainingConfig::checkerboard_sync();
    let mut training = TrainingLoop::new(model, config);

    let output = training.run_steps(&seed, CHECKERBOARD_SYNC_STEPS);
    save_grid_as_image(&output, "debug_output_untrained.png", 16);

    // Print ASCII version too
    println!("\nTarget (channel 0, 16x16):");
    for y in 0..16 {
        for x in 0..16 {
            let v = target.get(x, y, 0);
            print!("{}", if v > 0.5 { "█" } else { "░" });
        }
        println!();
    }

    println!("\nSeed (channel 0, 16x16):");
    for y in 0..16 {
        for x in 0..16 {
            let v = seed.get(x, y, 0);
            print!("{}", if v > 0.5 { "█" } else { "░" });
        }
        println!();
    }

    println!("\nOutput untrained (channel 0, 16x16):");
    for y in 0..16 {
        for x in 0..16 {
            let v = output.get(x as isize, y as isize, 0);
            print!("{}", if v > 0.5 { "█" } else { "░" });
        }
        println!();
    }

    println!("\nImages saved: debug_target.png, debug_seed.png, debug_output_untrained.png");

    // Run boundary check
    debug_boundary();
}

// Additional debug: verify boundary zero-padding in perception
fn debug_boundary() {
    use logicars::BoundaryCondition;
    use logicars::NGrid;

    println!("\n=== Boundary Zero-Padding Check ===");

    let mut grid = NGrid::new(4, 4, 2, BoundaryCondition::NonPeriodic);
    // Set corner to [1.0, 0.5]
    grid.set(0, 0, 0, 1.0);
    grid.set(0, 0, 1, 0.5);

    // Get neighborhood at (0,0) - top-left corner
    let n = grid.neighborhood(0, 0);

    println!("Neighborhood at (0,0) corner:");
    println!("  Position 0 (NW, out of bounds): {:?}", n.get_cell(0));
    println!("  Position 1 (N, out of bounds):  {:?}", n.get_cell(1));
    println!("  Position 2 (NE, out of bounds): {:?}", n.get_cell(2));
    println!("  Position 3 (W, out of bounds):  {:?}", n.get_cell(3));
    println!("  Position 4 (C, the cell):       {:?}", n.get_cell(4));
    println!("  Position 5 (E, in bounds):      {:?}", n.get_cell(5));

    // Verify
    let nw = n.get_cell(0);
    if nw[0] == 0.0 && nw[1] == 0.0 {
        println!("\n✓ Zero-padding is working correctly!");
    } else {
        println!("\n✗ ERROR: Zero-padding NOT working! Got {:?}", nw);
    }
}
