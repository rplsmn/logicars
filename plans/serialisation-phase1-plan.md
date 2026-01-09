# Serialisation Phase 1: Core Checkpoint System

## Goal

Enable saving and resuming training runs. After Phase 1:
- Training can be stopped and resumed without loss
- Model weights persist between sessions
- Optimizer state (momentum) is preserved for stable continuation

---

## Phase 1.1: Model Weight Serialisation

### Objective
Add `Serialize`/`Deserialize` derives to all model components.

### Files to Modify

| File | Components | Changes |
|------|------------|---------|
| `phase_0_1.rs` | `ProbabilisticGate` | Add `#[derive(Serialize, Deserialize)]` |
| `perception.rs` | `GateLayer`, `PerceptionKernel`, `PerceptionModule`, `Wires`, `ConnectionType` | Add derives |
| `update.rs` | `UpdateModule`, `DiffLogicCA` | Add derives |

### Implementation Steps

1. **Add bincode dependency** to `Cargo.toml`:
   ```toml
   bincode = "1.3"
   ```

2. **Update phase_0_1.rs**:
   ```rust
   use serde::{Serialize, Deserialize};
   
   #[derive(Clone, Serialize, Deserialize)]
   pub struct ProbabilisticGate {
       pub logits: [f64; 16],
   }
   ```

3. **Update perception.rs**:
   ```rust
   #[derive(Clone, Serialize, Deserialize)]
   pub struct Wires { pub a: Vec<usize>, pub b: Vec<usize> }
   
   #[derive(Clone, Copy, Serialize, Deserialize)]
   pub enum ConnectionType { FirstKernel, Unique }
   
   #[derive(Clone, Serialize, Deserialize)]
   pub struct GateLayer { pub gates: Vec<ProbabilisticGate>, pub wires: Wires }
   
   #[derive(Clone, Serialize, Deserialize)]
   pub struct PerceptionKernel { pub layers: Vec<GateLayer>, pub input_size: usize }
   
   #[derive(Clone, Serialize, Deserialize)]
   pub struct PerceptionModule { ... }
   ```

4. **Update update.rs**:
   ```rust
   #[derive(Clone, Serialize, Deserialize)]
   pub struct UpdateModule { ... }
   
   #[derive(Clone, Serialize, Deserialize)]
   pub struct DiffLogicCA { ... }
   ```

5. **Add save/load methods to DiffLogicCA**:
   ```rust
   impl DiffLogicCA {
       pub fn save(&self, path: &str) -> std::io::Result<()> {
           let data = bincode::serialize(self)
               .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
           std::fs::write(path, data)
       }
       
       pub fn load(path: &str) -> std::io::Result<Self> {
           let data = std::fs::read(path)?;
           bincode::deserialize(&data)
               .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
       }
   }
   ```

### Tests

```rust
#[test]
fn test_model_serialization_roundtrip() {
    let model = create_small_checkerboard_model();
    let path = "/tmp/test_model.bin";
    
    model.save(path).unwrap();
    let loaded = DiffLogicCA::load(path).unwrap();
    
    assert_eq!(model.total_gates(), loaded.total_gates());
    // Verify weights match
    let orig_logits = model.perception.kernels[0].layers[0].gates[0].logits;
    let load_logits = loaded.perception.kernels[0].layers[0].gates[0].logits;
    assert_eq!(orig_logits, load_logits);
    
    std::fs::remove_file(path).ok();
}

#[test]
fn test_model_serialization_preserves_training_state() {
    let mut model = create_small_checkerboard_model();
    // Modify a logit
    model.perception.kernels[0].layers[0].gates[0].logits[0] = 5.0;
    
    let path = "/tmp/test_model_modified.bin";
    model.save(path).unwrap();
    let loaded = DiffLogicCA::load(path).unwrap();
    
    assert_eq!(loaded.perception.kernels[0].layers[0].gates[0].logits[0], 5.0);
    std::fs::remove_file(path).ok();
}
```

### Exit Criteria
- [ ] `cargo test --lib` passes with new serialization tests
- [ ] `DiffLogicCA::save()` creates valid bincode file
- [ ] `DiffLogicCA::load()` restores exact weights
- [ ] File size is reasonable (<1MB for checkerboard model)

---

## Phase 1.2: Optimizer State Serialisation

### Objective
Save/load AdamW momentum buffers (m, v) and timestep (t).

### Files to Modify

| File | Component | Changes |
|------|-----------|---------|
| `optimizer.rs` | `AdamW` | Add derives, save/load methods |

### Current AdamW Structure (from optimizer.rs)
```rust
pub struct AdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
    pub eps: f64,
    m: Vec<f64>,   // First moment estimates
    v: Vec<f64>,   // Second moment estimates
    t: usize,      // Timestep
}
```

### Implementation Steps

1. **Add derives**:
   ```rust
   #[derive(Clone, Serialize, Deserialize)]
   pub struct AdamW { ... }
   ```

2. **Add save/load methods**:
   ```rust
   impl AdamW {
       pub fn save(&self, path: &str) -> std::io::Result<()> {
           let data = bincode::serialize(self)
               .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
           std::fs::write(path, data)
       }
       
       pub fn load(path: &str) -> std::io::Result<Self> {
           let data = std::fs::read(path)?;
           bincode::deserialize(&data)
               .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
       }
   }
   ```

### Tests

```rust
#[test]
fn test_optimizer_serialization_roundtrip() {
    let mut optim = AdamW::new(1000, 0.05, 0.9, 0.99, 0.005);
    
    // Simulate some training steps
    let grads = vec![0.1; 1000];
    for _ in 0..10 {
        optim.step(&grads);
    }
    
    let path = "/tmp/test_optimizer.bin";
    optim.save(path).unwrap();
    let loaded = AdamW::load(path).unwrap();
    
    assert_eq!(optim.t, loaded.t);
    assert_eq!(optim.m, loaded.m);
    assert_eq!(optim.v, loaded.v);
    
    std::fs::remove_file(path).ok();
}
```

### Exit Criteria
- [ ] AdamW serialization round-trips correctly
- [ ] Momentum buffers (m, v) are preserved exactly
- [ ] Timestep (t) is preserved

---

## Phase 1.3: Checkpoint Integration

### Objective
Create unified checkpoint system that saves all training state.

### New File: `checkpoint.rs`

```rust
//! Training checkpoint system
//!
//! Saves and loads complete training state including model, optimizer, and metadata.

use serde::{Serialize, Deserialize};
use crate::{DiffLogicCA, TrainingConfig};
use crate::optimizer::AdamW;

/// Checkpoint metadata
#[derive(Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Checkpoint format version
    pub version: u32,
    /// Epoch at checkpoint
    pub epoch: usize,
    /// Best accuracy achieved
    pub best_accuracy: f64,
    /// RNG state for reproducibility
    pub rng_state: u64,
    /// Unix timestamp
    pub timestamp: u64,
    /// Training config (for reference)
    pub config: TrainingConfig,
}

/// Complete training checkpoint
#[derive(Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub metadata: CheckpointMetadata,
    pub model: DiffLogicCA,
    pub optimizer: AdamW,
}

pub const CHECKPOINT_VERSION: u32 = 1;

impl Checkpoint {
    /// Save checkpoint to file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let data = bincode::serialize(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, data)
    }
    
    /// Load checkpoint from file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        let checkpoint: Checkpoint = bincode::deserialize(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        
        // Version check
        if checkpoint.metadata.version != CHECKPOINT_VERSION {
            eprintln!("Warning: Checkpoint version {} differs from current {}",
                     checkpoint.metadata.version, CHECKPOINT_VERSION);
        }
        
        Ok(checkpoint)
    }
}
```

### Modify: `training.rs`

Add checkpoint methods to `TrainingLoop`:

```rust
impl TrainingLoop {
    /// Save training state to checkpoint
    pub fn save_checkpoint(
        &self, 
        path: &str, 
        epoch: usize, 
        best_accuracy: f64,
        rng_state: u64,
    ) -> std::io::Result<()> {
        let checkpoint = Checkpoint {
            metadata: CheckpointMetadata {
                version: CHECKPOINT_VERSION,
                epoch,
                best_accuracy,
                rng_state,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                config: self.config.clone(),
            },
            model: self.model.clone(),
            optimizer: self.optimizer.clone(),
        };
        checkpoint.save(path)
    }
    
    /// Load training state from checkpoint
    pub fn from_checkpoint(path: &str) -> std::io::Result<(Self, CheckpointMetadata)> {
        let checkpoint = Checkpoint::load(path)?;
        
        let training_loop = TrainingLoop {
            model: checkpoint.model,
            optimizer: checkpoint.optimizer,
            config: checkpoint.metadata.config.clone(),
        };
        
        Ok((training_loop, checkpoint.metadata))
    }
}
```

### Tests

```rust
#[test]
fn test_checkpoint_save_load() {
    let model = create_small_checkerboard_model();
    let config = TrainingConfig::checkerboard_sync();
    let training_loop = TrainingLoop::new(model, config);
    
    let path = "/tmp/test_checkpoint.bin";
    training_loop.save_checkpoint(path, 100, 0.75, 42).unwrap();
    
    let (loaded, metadata) = TrainingLoop::from_checkpoint(path).unwrap();
    
    assert_eq!(metadata.epoch, 100);
    assert_eq!(metadata.best_accuracy, 0.75);
    assert_eq!(metadata.rng_state, 42);
    assert_eq!(loaded.model.total_gates(), training_loop.model.total_gates());
    
    std::fs::remove_file(path).ok();
}

#[test]
fn test_resume_training_from_checkpoint() {
    // Train for 10 epochs
    let model = create_small_checkerboard_model();
    let config = TrainingConfig::checkerboard_sync();
    let mut training_loop = TrainingLoop::new(model, config);
    
    let target = create_checkerboard(16, 2, 8);
    let mut rng = SimpleRng::new(42);
    
    for _ in 0..10 {
        let input = create_random_seed(16, 8, &mut rng);
        training_loop.train_step_batch(&[input], &target);
    }
    
    // Save checkpoint
    let path = "/tmp/test_resume.bin";
    training_loop.save_checkpoint(path, 10, 0.5, rng.state()).unwrap();
    
    // Load and continue
    let (mut loaded, metadata) = TrainingLoop::from_checkpoint(path).unwrap();
    assert_eq!(metadata.epoch, 10);
    
    // Continue training for 10 more epochs
    let mut rng2 = SimpleRng::from_state(metadata.rng_state);
    for _ in 0..10 {
        let input = create_random_seed(16, 8, &mut rng2);
        loaded.train_step_batch(&[input], &target);
    }
    
    std::fs::remove_file(path).ok();
}
```

### Exit Criteria
- [ ] `Checkpoint` saves model, optimizer, and metadata
- [ ] `TrainingLoop::save_checkpoint()` creates valid file
- [ ] `TrainingLoop::from_checkpoint()` restores training state
- [ ] Training can resume from checkpoint and continue correctly
- [ ] RNG state allows reproducible continuation

---

## Phase 1.4: CLI Integration

### Objective
Add checkpoint options to training binary.

### Modify: `src/bin/train_checkerboard.rs`

Add new CLI options:
- `--save-checkpoint=PATH` - Save checkpoint every N epochs
- `--load-checkpoint=PATH` - Resume from checkpoint
- `--checkpoint-interval=N` - How often to save (default: 100 epochs)

### Implementation Sketch

```rust
// Parse CLI args
let save_checkpoint: Option<String> = args
    .iter()
    .find(|a| a.starts_with("--save-checkpoint="))
    .and_then(|a| a.strip_prefix("--save-checkpoint="))
    .map(|s| s.to_string());

let load_checkpoint: Option<String> = args
    .iter()
    .find(|a| a.starts_with("--load-checkpoint="))
    .and_then(|a| a.strip_prefix("--load-checkpoint="))
    .map(|s| s.to_string());

let checkpoint_interval: usize = args
    .iter()
    .find(|a| a.starts_with("--checkpoint-interval="))
    .and_then(|a| a.strip_prefix("--checkpoint-interval="))
    .and_then(|s| s.parse().ok())
    .unwrap_or(100);

// Initialize from checkpoint or fresh
let (mut training_loop, mut start_epoch, mut best_accuracy, mut rng) = 
    if let Some(ref path) = load_checkpoint {
        let (loop_, meta) = TrainingLoop::from_checkpoint(path)?;
        println!("Resuming from epoch {} (best: {:.2}%)", meta.epoch, meta.best_accuracy * 100.0);
        (loop_, meta.epoch, meta.best_accuracy, SimpleRng::from_state(meta.rng_state))
    } else {
        let model = create_checkerboard_model();
        let config = TrainingConfig::checkerboard_sync();
        (TrainingLoop::new(model, config), 0, 0.0, SimpleRng::new(23))
    };

// In training loop, save checkpoint periodically
if let Some(ref path) = save_checkpoint {
    if epoch % checkpoint_interval == 0 && epoch > 0 {
        training_loop.save_checkpoint(path, epoch, best_accuracy, rng.state())?;
        println!("  [Checkpoint saved to {}]", path);
    }
}
```

### Usage Examples

```bash
# Start fresh training with checkpoints every 100 epochs
cargo run --bin train_checkerboard --release -- \
    --epochs=1000 \
    --save-checkpoint=checkpoints/checkerboard.ckpt \
    --checkpoint-interval=100

# Resume from checkpoint
cargo run --bin train_checkerboard --release -- \
    --load-checkpoint=checkpoints/checkerboard.ckpt \
    --save-checkpoint=checkpoints/checkerboard.ckpt \
    --epochs=2000
```

### Exit Criteria
- [ ] `--save-checkpoint` creates checkpoint files during training
- [ ] `--load-checkpoint` resumes training from checkpoint
- [ ] Epoch numbering continues correctly after resume
- [ ] Best accuracy is preserved across sessions
- [ ] RNG state allows reproducible results

---

## Testing Checklist

### Unit Tests
- [ ] `test_model_serialization_roundtrip`
- [ ] `test_model_serialization_preserves_training_state`
- [ ] `test_optimizer_serialization_roundtrip`
- [ ] `test_checkpoint_save_load`
- [ ] `test_resume_training_from_checkpoint`

### Integration Tests
- [ ] Train for 50 epochs, save checkpoint
- [ ] Load checkpoint, continue to 100 epochs
- [ ] Verify loss trajectory is smooth (no discontinuity)
- [ ] Compare with uninterrupted 100-epoch run

### Manual Verification
- [ ] Run `--save-checkpoint` and verify file is created
- [ ] Run `--load-checkpoint` and verify training continues
- [ ] Interrupt training with Ctrl+C, resume from last checkpoint

---

## Estimated Effort

| Phase | Tasks | Time |
|-------|-------|------|
| 1.1 | Model serialization | 2-3 hours |
| 1.2 | Optimizer serialization | 1-2 hours |
| 1.3 | Checkpoint integration | 2-3 hours |
| 1.4 | CLI integration | 1-2 hours |
| **Total** | | **6-10 hours** |

---

## Commands for Implementation

```bash
# Create feature branch
git checkout -b feature/model-serialisation main

# After each phase, run tests
cargo test --lib

# Test checkpoint manually
cargo run --bin train_checkerboard --release -- \
    --small --epochs=20 \
    --save-checkpoint=/tmp/test.ckpt \
    --checkpoint-interval=10

# Verify checkpoint file
ls -la /tmp/test.ckpt
```
