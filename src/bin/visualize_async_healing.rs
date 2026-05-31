//! Visualize the async checkerboard model's recurrent dynamics, generalization, and
//! self-healing — mirroring the reference notebook (cells 31 / 33 / 34) but rendered to GIFs.
//!
//! Run the model ASYNCHRONOUSLY at inference (fire-rate masking), which is how the async
//! checkerboard is meant to be evaluated and where self-healing emerges.
//!
//! Usage:
//!   cargo run --release --bin visualize_async_healing -- <model.json> [--prefix=NAME]
//!       [--fire-rate=0.6] [--seed=42] [--damage=10] [--scale=8]
//!
//! Produces three GIFs (NAME defaults to "async"):
//!   NAME_rollout.gif     14x14, 50 async steps — pattern emerging from random noise
//!   NAME_generalize.gif  56x56 (4x), 200 steps — generalization to a larger grid
//!   NAME_heal.gif        56x56, 200 steps — a 10x10 center is zeroed for the first
//!                        half (damage held), then released; watch the pattern regrow.

use image::{codecs::gif::{GifEncoder, Repeat}, Frame, Rgba, RgbaImage};
use logicars::{
    create_checkerboard, create_random_seed, compute_checkerboard_accuracy, Float, HardCircuit,
    NGrid, SimpleRng, CHECKERBOARD_ASYNC_GRID_SIZE, CHECKERBOARD_ASYNC_STEPS,
    CHECKERBOARD_CHANNELS, CHECKERBOARD_SQUARE_SIZE,
};
use std::env;
use std::fs::File;

/// Render channel 0 of a grid to a scaled black/white RGBA image.
fn grid_to_image(grid: &NGrid, scale: u32) -> RgbaImage {
    let w = grid.width as u32;
    let h = grid.height as u32;
    let mut img = RgbaImage::new(w * scale, h * scale);
    for y in 0..h {
        for x in 0..w {
            let on = grid.get(x as isize, y as isize, 0) > 0.5;
            let px = if on {
                Rgba([255, 255, 255, 255])
            } else {
                Rgba([0, 0, 0, 255])
            };
            for dy in 0..scale {
                for dx in 0..scale {
                    img.put_pixel(x * scale + dx, y * scale + dy, px);
                }
            }
        }
    }
    img
}

/// Zero a centered `size`x`size` square across all channels (cell "damage").
fn damage_center(grid: &mut NGrid, size: usize) {
    let (w, h) = (grid.width, grid.height);
    let x0 = w / 2 - size / 2;
    let y0 = h / 2 - size / 2;
    for y in y0..(y0 + size).min(h) {
        for x in x0..(x0 + size).min(w) {
            for c in 0..grid.channels {
                grid.set(x, y, c, 0.0);
            }
        }
    }
}

fn write_gif(path: &str, frames: Vec<Frame>) {
    let file = File::create(path).expect("create gif");
    let mut enc = GifEncoder::new(file);
    enc.set_repeat(Repeat::Infinite).ok();
    for f in frames {
        enc.encode_frame(f).expect("encode frame");
    }
    println!("  wrote {path}");
}

fn parse<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T {
    args.iter()
        .find(|a| a.starts_with(key))
        .and_then(|a| a.strip_prefix(key))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_path = match args.get(1).filter(|a| !a.starts_with("--")) {
        Some(p) => p.clone(),
        None => {
            eprintln!(
                "Usage: visualize_async_healing <model.json> [--prefix=NAME] \
                 [--fire-rate=0.6] [--seed=42] [--damage=10] [--scale=8]"
            );
            std::process::exit(1);
        }
    };

    let prefix: String = args
        .iter()
        .find(|a| a.starts_with("--prefix="))
        .and_then(|a| a.strip_prefix("--prefix="))
        .unwrap_or("async")
        .to_string();
    let fire_rate: Float = parse(&args, "--fire-rate=", 0.6);
    let seed: u64 = parse(&args, "--seed=", 42);
    let damage: usize = parse(&args, "--damage=", 10);
    let big_scale: u32 = parse(&args, "--scale=", 8);

    let circuit = HardCircuit::load(&model_path).unwrap_or_else(|e| {
        eprintln!("Error loading {model_path}: {e}");
        std::process::exit(1);
    });
    println!(
        "Loaded {model_path}: {} channels, {} gates, fire_rate={fire_rate}\n",
        circuit.channels,
        circuit.total_gate_count()
    );

    let mut rng = SimpleRng::new(seed);
    let frame_delay = image::Delay::from_numer_denom_ms(120, 1);

    // ---- 1. Basic async rollout on the training-size grid (14x14) ----
    {
        let n = CHECKERBOARD_ASYNC_GRID_SIZE;
        let steps = CHECKERBOARD_ASYNC_STEPS;
        let mut grid = create_random_seed(n, CHECKERBOARD_CHANNELS, &mut rng);
        let target = create_checkerboard(n, CHECKERBOARD_SQUARE_SIZE, CHECKERBOARD_CHANNELS);
        let mut frames = vec![Frame::from_parts(grid_to_image(&grid, 16), 0, 0, frame_delay)];
        for _ in 0..steps {
            grid = circuit.step_async(&grid, fire_rate, &mut rng);
            frames.push(Frame::from_parts(grid_to_image(&grid, 16), 0, 0, frame_delay));
        }
        println!(
            "rollout: final hard acc = {:.3}",
            compute_checkerboard_accuracy(&grid, &target)
        );
        write_gif(&format!("{prefix}_rollout.gif"), frames);
    }

    // ---- 2. Generalization to a 4x-larger grid (56x56), no damage ----
    {
        let n = CHECKERBOARD_ASYNC_GRID_SIZE * 4;
        let steps = CHECKERBOARD_ASYNC_STEPS * 4;
        let mut grid = create_random_seed(n, CHECKERBOARD_CHANNELS, &mut rng);
        let mut frames = vec![Frame::from_parts(grid_to_image(&grid, big_scale), 0, 0, frame_delay)];
        for _ in 0..steps {
            grid = circuit.step_async(&grid, fire_rate, &mut rng);
            frames.push(Frame::from_parts(grid_to_image(&grid, big_scale), 0, 0, frame_delay));
        }
        write_gif(&format!("{prefix}_generalize.gif"), frames);
    }

    // ---- 3. Self-healing: damage held for first half, then released ----
    {
        let n = CHECKERBOARD_ASYNC_GRID_SIZE * 4;
        let steps = CHECKERBOARD_ASYNC_STEPS * 4;
        let release = steps / 2; // re-activate damaged region after this many steps
        let mut grid = create_random_seed(n, CHECKERBOARD_CHANNELS, &mut rng);
        let mut frames = vec![Frame::from_parts(grid_to_image(&grid, big_scale), 0, 0, frame_delay)];
        for step in 0..steps {
            grid = circuit.step_async(&grid, fire_rate, &mut rng);
            if step < release {
                damage_center(&mut grid, damage); // hold the lesion open
            }
            frames.push(Frame::from_parts(grid_to_image(&grid, big_scale), 0, 0, frame_delay));
        }
        println!("heal: damage {damage}x{damage} held for {release} steps, then released");
        write_gif(&format!("{prefix}_heal.gif"), frames);
    }

    println!("\nDone. Open the GIFs to see rollout / generalization / self-healing.");
}
