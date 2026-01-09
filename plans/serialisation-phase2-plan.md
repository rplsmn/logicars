# Serialisation Phase 2: Robustness & Convenience

## Goal

Make the checkpoint system production-ready:
- Version compatibility for future format changes
- Single-file checkpoints for easy sharing
- Compression to reduce storage

---

## Prerequisites

Phase 1 must be complete:
- [ ] Model serialization works
- [ ] Optimizer serialization works
- [ ] Checkpoint save/load works
- [ ] CLI integration done

---

## Phase 2.1: Version Compatibility

### Objective
Handle checkpoint format changes gracefully across versions.

### Problem
As Logicars evolves, checkpoint format may change:
- New fields added to `CheckpointMetadata`
- Model architecture changes (new layer types)
- Optimizer changes (new hyperparameters)

Without versioning, old checkpoints become unloadable.

### Implementation

#### 1. Explicit Version in Metadata (already in Phase 1)
```rust
pub const CHECKPOINT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub version: u32,  // This enables migration
    // ...
}
```

#### 2. Migration System

Create `checkpoint_migration.rs`:

```rust
//! Checkpoint migration for backward compatibility

use crate::checkpoint::{Checkpoint, CheckpointMetadata};

/// Migrate checkpoint from old version to current
pub fn migrate_checkpoint(data: &[u8]) -> Result<Checkpoint, MigrationError> {
    // Try to read version first
    let version = read_version(data)?;
    
    match version {
        1 => {
            // Current version, no migration needed
            bincode::deserialize(data)
                .map_err(|e| MigrationError::Deserialize(e.to_string()))
        }
        // Future migrations go here:
        // 2 => migrate_v1_to_v2(data),
        _ => Err(MigrationError::UnsupportedVersion(version))
    }
}

#[derive(Debug)]
pub enum MigrationError {
    UnsupportedVersion(u32),
    Deserialize(String),
    VersionRead(String),
}
```

#### 3. Architecture Hash Validation

Detect model architecture mismatches:

```rust
impl DiffLogicCA {
    /// Compute hash of model architecture (not weights)
    pub fn architecture_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash perception architecture
        self.perception.num_kernels.hash(&mut hasher);
        self.perception.channels.hash(&mut hasher);
        for kernel in &self.perception.kernels {
            kernel.layers.len().hash(&mut hasher);
            for layer in &kernel.layers {
                layer.gates.len().hash(&mut hasher);
            }
        }
        
        // Hash update architecture
        self.update.layers.len().hash(&mut hasher);
        for layer in &self.update.layers {
            layer.gates.len().hash(&mut hasher);
        }
        
        hasher.finish()
    }
}

// Add to CheckpointMetadata
pub struct CheckpointMetadata {
    pub architecture_hash: u64,
    // ...
}

// Validate on load
impl Checkpoint {
    pub fn load_and_validate(path: &str, expected_model: &DiffLogicCA) -> std::io::Result<Self> {
        let checkpoint = Self::load(path)?;
        
        let loaded_hash = checkpoint.model.architecture_hash();
        let expected_hash = expected_model.architecture_hash();
        
        if loaded_hash != expected_hash {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Architecture mismatch: checkpoint={:x}, expected={:x}",
                       loaded_hash, expected_hash)
            ));
        }
        
        Ok(checkpoint)
    }
}
```

### Tests

```rust
#[test]
fn test_version_warning_on_mismatch() {
    // Create checkpoint with future version
    let mut checkpoint = create_test_checkpoint();
    checkpoint.metadata.version = 999;
    
    // Save and load should warn but succeed
    let path = "/tmp/test_version.bin";
    checkpoint.save(path).unwrap();
    
    // This should print warning but load successfully
    let loaded = Checkpoint::load(path).unwrap();
    assert_eq!(loaded.metadata.version, 999);
    
    std::fs::remove_file(path).ok();
}

#[test]
fn test_architecture_hash_different_models() {
    let model1 = create_small_checkerboard_model();
    let model2 = create_checkerboard_model();
    
    assert_ne!(model1.architecture_hash(), model2.architecture_hash());
}

#[test]
fn test_architecture_hash_same_model() {
    let model1 = create_small_checkerboard_model();
    let model2 = create_small_checkerboard_model();
    
    assert_eq!(model1.architecture_hash(), model2.architecture_hash());
}
```

### Exit Criteria
- [ ] Version field is saved and loaded
- [ ] Warning printed for version mismatch
- [ ] Architecture hash prevents loading wrong model type
- [ ] Migration system is in place (even if no migrations yet)

---

## Phase 2.2: Single-File Checkpoints

### Objective
Package checkpoint as single `.logicars` file instead of directory.

### Current Structure (Phase 1)
```
checkpoint.bin   # Single bincode file with everything
```

### Enhanced Structure (Phase 2)
```
checkpoint.logicars   # ZIP archive containing:
├── metadata.json     # Human-readable metadata
├── model.bin         # Model weights (bincode)
├── optimizer.bin     # Optimizer state (bincode)
└── README.md         # What this checkpoint is
```

### Why?
1. **Inspectable**: JSON metadata readable without code
2. **Modular**: Can load just model without optimizer
3. **Extensible**: Easy to add new components
4. **Shareable**: Single file with context

### Implementation

Add dependency:
```toml
[dependencies]
zip = "0.6"
```

Create `checkpoint_archive.rs`:

```rust
use std::io::{Read, Write};
use zip::{ZipArchive, ZipWriter, CompressionMethod};

impl Checkpoint {
    /// Save as .logicars archive
    pub fn save_archive(&self, path: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut zip = ZipWriter::new(file);
        
        // Write metadata.json
        let options = zip::write::FileOptions::default()
            .compression_method(CompressionMethod::Deflated);
        zip.start_file("metadata.json", options)?;
        let metadata_json = serde_json::to_string_pretty(&self.metadata)?;
        zip.write_all(metadata_json.as_bytes())?;
        
        // Write model.bin
        zip.start_file("model.bin", options)?;
        let model_bin = bincode::serialize(&self.model)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        zip.write_all(&model_bin)?;
        
        // Write optimizer.bin
        zip.start_file("optimizer.bin", options)?;
        let optim_bin = bincode::serialize(&self.optimizer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        zip.write_all(&optim_bin)?;
        
        // Write README
        zip.start_file("README.md", options)?;
        let readme = format!(
            "# Logicars Checkpoint\n\n\
             - **Version**: {}\n\
             - **Epoch**: {}\n\
             - **Best Accuracy**: {:.2}%\n\
             - **Gates**: {}\n\
             - **Created**: {}\n",
            self.metadata.version,
            self.metadata.epoch,
            self.metadata.best_accuracy * 100.0,
            self.model.total_gates(),
            chrono_timestamp(self.metadata.timestamp),
        );
        zip.write_all(readme.as_bytes())?;
        
        zip.finish()?;
        Ok(())
    }
    
    /// Load from .logicars archive
    pub fn load_archive(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut archive = ZipArchive::new(file)?;
        
        // Read metadata
        let mut metadata_json = String::new();
        archive.by_name("metadata.json")?.read_to_string(&mut metadata_json)?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)?;
        
        // Read model
        let mut model_bin = Vec::new();
        archive.by_name("model.bin")?.read_to_end(&mut model_bin)?;
        let model: DiffLogicCA = bincode::deserialize(&model_bin)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        
        // Read optimizer
        let mut optim_bin = Vec::new();
        archive.by_name("optimizer.bin")?.read_to_end(&mut optim_bin)?;
        let optimizer: AdamW = bincode::deserialize(&optim_bin)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        
        Ok(Checkpoint { metadata, model, optimizer })
    }
}
```

### Auto-detect Format

```rust
impl Checkpoint {
    /// Load from path, auto-detecting format
    pub fn load_auto(path: &str) -> std::io::Result<Self> {
        if path.ends_with(".logicars") {
            Self::load_archive(path)
        } else {
            Self::load(path)  // bincode format
        }
    }
    
    /// Save to path, using format based on extension
    pub fn save_auto(&self, path: &str) -> std::io::Result<()> {
        if path.ends_with(".logicars") {
            self.save_archive(path)
        } else {
            self.save(path)  // bincode format
        }
    }
}
```

### Tests

```rust
#[test]
fn test_archive_save_load_roundtrip() {
    let checkpoint = create_test_checkpoint();
    let path = "/tmp/test.logicars";
    
    checkpoint.save_archive(path).unwrap();
    let loaded = Checkpoint::load_archive(path).unwrap();
    
    assert_eq!(checkpoint.metadata.epoch, loaded.metadata.epoch);
    assert_eq!(checkpoint.model.total_gates(), loaded.model.total_gates());
    
    std::fs::remove_file(path).ok();
}

#[test]
fn test_archive_is_valid_zip() {
    let checkpoint = create_test_checkpoint();
    let path = "/tmp/test_zip.logicars";
    
    checkpoint.save_archive(path).unwrap();
    
    // Verify it's a valid ZIP
    let file = std::fs::File::open(path).unwrap();
    let archive = ZipArchive::new(file).unwrap();
    
    assert!(archive.file_names().any(|n| n == "metadata.json"));
    assert!(archive.file_names().any(|n| n == "model.bin"));
    assert!(archive.file_names().any(|n| n == "optimizer.bin"));
    assert!(archive.file_names().any(|n| n == "README.md"));
    
    std::fs::remove_file(path).ok();
}

#[test]
fn test_archive_metadata_readable() {
    let checkpoint = create_test_checkpoint();
    let path = "/tmp/test_readable.logicars";
    
    checkpoint.save_archive(path).unwrap();
    
    // Can read metadata without Rust
    let file = std::fs::File::open(path).unwrap();
    let mut archive = ZipArchive::new(file).unwrap();
    
    let mut metadata_json = String::new();
    archive.by_name("metadata.json").unwrap().read_to_string(&mut metadata_json).unwrap();
    
    assert!(metadata_json.contains("\"epoch\""));
    assert!(metadata_json.contains("\"best_accuracy\""));
    
    std::fs::remove_file(path).ok();
}
```

### Exit Criteria
- [ ] `.logicars` files are valid ZIP archives
- [ ] Metadata is human-readable JSON inside archive
- [ ] Model and optimizer load correctly from archive
- [ ] Auto-detection works based on extension
- [ ] Compression reduces file size (measure actual reduction)

---

## Estimated Effort

| Phase | Tasks | Time |
|-------|-------|------|
| 2.1 | Version compatibility | 1-2 hours |
| 2.2 | Single-file checkpoints | 2-3 hours |
| **Total** | | **3-5 hours** |

---

## Dependencies to Add

```toml
[dependencies]
zip = "0.6"     # For archive format
# chrono = "0.4"  # Optional, for timestamp formatting
```

---

## Commands for Implementation

```bash
# Continue on feature branch
git checkout feature/model-serialisation

# After Phase 2.1
cargo test --lib -- version
cargo test --lib -- architecture_hash

# After Phase 2.2
cargo test --lib -- archive

# Test manually
cargo run --bin train_checkerboard --release -- \
    --small --epochs=20 \
    --save-checkpoint=test.logicars

# Verify archive content
unzip -l test.logicars
unzip -p test.logicars metadata.json | jq .
```
