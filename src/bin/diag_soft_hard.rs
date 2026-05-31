//! DIAGNOSTIC: does the soft forward pass agree with the hard (argmax) forward pass?
//!
//! The async checkerboard symptom is soft_loss -> 0 while hard_loss stays at chance,
//! even though gates are ~99% committed (mean dominant-prob ~0.99). If a committed-gate
//! network on (near-)binary inputs produced the same thing soft and hard, soft_loss and
//! hard_loss would track. This probe measures, step by step, how far the soft rollout and
//! the hard rollout drift apart on the SAME model + SAME input, with NO fire masking
//! (sync, every cell fires) so the only difference is soft-execute vs hard-argmax.

use logicars::{
    create_checkerboard, create_checkerboard_async_model, create_random_seed, Float,
    DiffLogicCA, NGrid, SimpleRng, CHECKERBOARD_ASYNC_GRID_SIZE, CHECKERBOARD_ASYNC_STEPS,
    CHECKERBOARD_CHANNELS, CHECKERBOARD_SQUARE_SIZE,
};

/// One synchronous SOFT step over the whole grid (all cells fire). Cell output replaces state.
fn soft_step(model: &DiffLogicCA, grid: &NGrid) -> NGrid {
    let mut out = grid.clone();
    for y in 0..grid.height {
        for x in 0..grid.width {
            let nb = grid.neighborhood(x, y);
            let (cell, _p, _u) = model.forward_soft(&nb);
            out.set_cell(x, y, &cell);
        }
    }
    out
}

/// One synchronous HARD step over the whole grid (all cells fire).
fn hard_step(model: &DiffLogicCA, grid: &NGrid) -> NGrid {
    let mut out = grid.clone();
    for y in 0..grid.height {
        for x in 0..grid.width {
            let nb = grid.neighborhood(x, y);
            let cell = model.forward_hard(&nb);
            out.set_cell(x, y, &cell);
        }
    }
    out
}

/// Saturate every gate: push the currently-dominant logit far above the rest so soft -> argmax.
fn saturate(model: &mut DiffLogicCA) {
    let bump = |g: &mut logicars::ProbabilisticGate| {
        let mut max_i = 0;
        for i in 1..16 {
            if g.logits[i] > g.logits[max_i] {
                max_i = i;
            }
        }
        for i in 0..16 {
            g.logits[i] = if i == max_i { 30.0 } else { -30.0 };
        }
        g.invalidate_cache();
    };
    for k in &mut model.perception.kernels {
        for l in &mut k.layers {
            for g in &mut l.gates {
                bump(g);
            }
        }
    }
    for l in &mut model.update.layers {
        for g in &mut l.gates {
            bump(g);
        }
    }
}

fn max_abs_diff(a: &NGrid, b: &NGrid) -> (Float, usize) {
    let mut maxd: Float = 0.0;
    let mut hard_disagree = 0usize; // cells whose channel-0 thresholds differ
    for y in 0..a.height {
        for x in 0..a.width {
            for c in 0..a.channels {
                let d: Float = (a.get(x as isize, y as isize, c) - b.get(x as isize, y as isize, c)).abs();
                if d > maxd {
                    maxd = d;
                }
            }
            let pa = if a.get(x as isize, y as isize, 0) > 0.5 { 1 } else { 0 };
            let pb = if b.get(x as isize, y as isize, 0) > 0.5 { 1 } else { 0 };
            if pa != pb {
                hard_disagree += 1;
            }
        }
    }
    (maxd, hard_disagree)
}

fn run_probe(label: &str, model: &DiffLogicCA) {
    let mut rng = SimpleRng::new(777);
    let input = create_random_seed(CHECKERBOARD_ASYNC_GRID_SIZE, CHECKERBOARD_CHANNELS, &mut rng);
    let target = create_checkerboard(
        CHECKERBOARD_ASYNC_GRID_SIZE,
        CHECKERBOARD_SQUARE_SIZE,
        CHECKERBOARD_CHANNELS,
    );

    println!("\n=== {label} ===");
    let mut soft = input.clone();
    let mut hard = input.clone();
    for step in 0..CHECKERBOARD_ASYNC_STEPS {
        soft = soft_step(model, &soft);
        hard = hard_step(model, &hard);
        let (maxd, disagree) = max_abs_diff(&soft, &hard);
        // Also: does soft, thresholded, ever stop matching hard?
        if step < 5 || step % 10 == 9 {
            println!(
                "step {:2}: max|soft-hard|={:.4}  ch0 threshold disagreements={}/{}",
                step + 1,
                maxd,
                disagree,
                CHECKERBOARD_ASYNC_GRID_SIZE * CHECKERBOARD_ASYNC_GRID_SIZE
            );
        }
    }
    // Final channel-0 accuracy of each path vs target.
    let acc = |g: &NGrid| {
        let mut c = 0;
        let n = g.width * g.height;
        for y in 0..g.height {
            for x in 0..g.width {
                let p: Float = if g.get(x as isize, y as isize, 0) > 0.5 { 1.0 } else { 0.0 };
                let t: Float = if target.get(x as isize, y as isize, 0) > 0.5 { 1.0 } else { 0.0 };
                if (p - t).abs() < 0.01 {
                    c += 1;
                }
            }
        }
        c as Float / n as Float
    };
    println!(
        "FINAL: soft ch0 acc={:.3}  hard ch0 acc={:.3}",
        acc(&soft),
        acc(&hard)
    );
}

fn main() {
    // Probe 1: fresh model as-is (mean sat ~0.999 reported by training binary).
    let model = create_checkerboard_async_model();
    run_probe("fresh async model (as initialised)", &model);

    // Probe 2: same model, gates hard-saturated to their argmax (soft MUST equal hard if
    // the two code paths agree on binary inputs). Any drift here is a forward-pass bug.
    let mut sat = create_checkerboard_async_model();
    saturate(&mut sat);
    run_probe("fully saturated model (soft should == hard)", &sat);
}
