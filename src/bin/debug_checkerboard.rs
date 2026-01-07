//! Debug training dynamics
use logicars::{
    create_checkerboard, create_random_seed, create_small_checkerboard_model, update::DiffLogicCA,
    NGrid, SimpleRng, TrainingConfig, TrainingLoop, CHECKERBOARD_CHANNELS, CHECKERBOARD_GRID_SIZE,
    CHECKERBOARD_SQUARE_SIZE,
};

fn main() {
    let target = create_checkerboard(
        CHECKERBOARD_GRID_SIZE,
        CHECKERBOARD_SQUARE_SIZE,
        CHECKERBOARD_CHANNELS,
    );

    println!("=== Training Dynamics Debug ===\n");
    println!("{:?}", target);
}
