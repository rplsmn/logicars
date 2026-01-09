//! Generate animated GIF of checkerboard model rollout.
//!
//! Usage:
//!   cargo run --bin visualize_checkerboard --release -- <model.json> [output.gif]
//!
//! Creates an animated GIF showing the pattern emerging from random noise.

use image::{RgbaImage, Rgba, codecs::gif::{GifEncoder, Repeat}};
use image::Frame;
use logicars::{
    create_random_seed, HardCircuit, NGrid, SimpleRng,
    CHECKERBOARD_CHANNELS, CHECKERBOARD_GRID_SIZE,
};
use std::env;
use std::fs::File;

/// Convert grid channel 0 to RGBA image (scaled up for visibility)
fn grid_to_image(grid: &NGrid, scale: u32) -> RgbaImage {
    let w = grid.width as u32;
    let h = grid.height as u32;
    let mut img = RgbaImage::new(w * scale, h * scale);

    for y in 0..h {
        for x in 0..w {
            let val = grid.get(x as isize, y as isize, 0);
            let pixel = if val > 0.5 {
                Rgba([255u8, 255u8, 255u8, 255u8]) // White
            } else {
                Rgba([0u8, 0u8, 0u8, 255u8]) // Black
            };

            // Fill scaled pixel
            for dy in 0..scale {
                for dx in 0..scale {
                    img.put_pixel(x * scale + dx, y * scale + dy, pixel);
                }
            }
        }
    }

    img
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse model path (required)
    let model_path = args.get(1).filter(|a| !a.starts_with("--"));
    if model_path.is_none() {
        eprintln!("Usage: visualize_checkerboard <model.json> [output.gif]");
        eprintln!("\nGenerate animated GIF of model rollout.");
        std::process::exit(1);
    }
    let model_path = model_path.unwrap();

    // Parse output path (optional, default: checkerboard_rollout.gif)
    let output_path = args.get(2)
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "checkerboard_rollout.gif".to_string());

    println!("=== Checkerboard Visualization ===\n");
    println!("Loading model from: {}", model_path);

    // Load model
    let circuit = match HardCircuit::load(model_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            std::process::exit(1);
        }
    };

    println!("Model loaded: {} channels, {} gates", circuit.channels, circuit.total_gate_count());

    // Parameters
    let grid_size = CHECKERBOARD_GRID_SIZE;
    let num_steps = 25; // 20 steps + 5 extra to show stable pattern
    let scale = 8u32;   // Scale up for visibility
    let frame_delay_ms = 150; // 150ms per frame

    println!("\n=== Generating Frames ===");
    println!("  Grid size: {}×{}", grid_size, grid_size);
    println!("  Steps: {}", num_steps);
    println!("  Scale: {}×", scale);
    println!("  Frame delay: {}ms", frame_delay_ms);

    // Create random seed
    let mut rng = SimpleRng::new(42);
    let seed = create_random_seed(grid_size, CHECKERBOARD_CHANNELS, &mut rng);

    // Collect frames
    let mut frames: Vec<Frame> = Vec::new();

    // Add initial frame
    let initial_img = grid_to_image(&seed, scale);
    frames.push(Frame::new(initial_img.into()));
    println!("  Frame 0: initial random state");

    // Run steps and capture frames
    let mut current = seed;
    for step in 1..=num_steps {
        current = circuit.step(&current);
        let img = grid_to_image(&current, scale);
        frames.push(Frame::new(img.into()));

        if step % 5 == 0 || step == num_steps {
            println!("  Frame {}: step {}", step, step);
        }
    }

    // Write GIF
    println!("\n=== Writing GIF ===");
    println!("  Output: {}", output_path);

    let file = File::create(&output_path).expect("Failed to create output file");
    let mut encoder = GifEncoder::new(file);
    encoder.set_repeat(Repeat::Infinite).expect("Failed to set repeat");

    for frame in frames {
        encoder.encode_frame(frame).expect("Failed to encode frame");
    }

    println!("  Done! {} frames written", num_steps + 1);
    println!("\n=== Complete ===");
}
