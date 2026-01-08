// Gate layer forward pass shader.
// Computes softmax-weighted sum of all 16 binary operations.

struct LayerConfig {
    num_gates: u32,
    input_offset: u32,
    output_offset: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> config: LayerConfig;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read> logits: array<f32>;  // [gate_idx * 16 + op]
@group(0) @binding(3) var<storage, read> wires: array<u32>;   // [gate_idx * 2 + {0,1}]
@group(0) @binding(4) var<storage, read_write> outputs: array<f32>;

// All 16 binary operations (matching BinaryOp::ALL order in Rust)
fn binary_op(op: u32, a: f32, b: f32) -> f32 {
    switch (op) {
        case 0u: { return 0.0; }                    // False
        case 1u: { return a * b; }                  // And
        case 2u: { return a * (1.0 - b); }          // A and not B
        case 3u: { return a; }                      // A (pass-through)
        case 4u: { return (1.0 - a) * b; }          // Not A and B
        case 5u: { return b; }                      // B
        case 6u: { return a + b - 2.0 * a * b; }    // Xor
        case 7u: { return a + b - a * b; }          // Or
        case 8u: { return 1.0 - (a + b - a * b); }  // Nor
        case 9u: { return 1.0 - (a + b - 2.0 * a * b); } // Xnor
        case 10u: { return 1.0 - b; }               // Not B
        case 11u: { return 1.0 - (1.0 - a) * b; }   // A or not B (B implies A)
        case 12u: { return 1.0 - a; }               // Not A
        case 13u: { return 1.0 - a * (1.0 - b); }   // Not A or B (A implies B)
        case 14u: { return 1.0 - a * b; }           // Nand
        default: { return 1.0; }                    // True (case 15)
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gate_idx = gid.x;
    if (gate_idx >= config.num_gates) {
        return;
    }
    
    // Read input wires
    let wire_a = wires[gate_idx * 2u];
    let wire_b = wires[gate_idx * 2u + 1u];
    let a = inputs[config.input_offset + wire_a];
    let b = inputs[config.input_offset + wire_b];
    
    // Compute softmax over 16 logits for numerical stability
    let logit_base = gate_idx * 16u;
    var max_logit = logits[logit_base];
    for (var i = 1u; i < 16u; i++) {
        max_logit = max(max_logit, logits[logit_base + i]);
    }
    
    var exp_sum = 0.0;
    for (var i = 0u; i < 16u; i++) {
        exp_sum += exp(logits[logit_base + i] - max_logit);
    }
    
    // Weighted sum of all 16 operations
    var result = 0.0;
    for (var i = 0u; i < 16u; i++) {
        let prob = exp(logits[logit_base + i] - max_logit) / exp_sum;
        let op_result = binary_op(i, a, b);
        result += prob * op_result;
    }
    
    outputs[config.output_offset + gate_idx] = result;
}
