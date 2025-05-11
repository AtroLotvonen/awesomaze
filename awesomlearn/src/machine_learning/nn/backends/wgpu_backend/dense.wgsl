const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;

struct Constants {
    m: u32, // layer_size in forward
    n: u32, // batch_size in forward
    k: u32, // input_size in forward
};

@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> constants: Constants;

const BLOCK_SIZE = WORKGROUP_SIZE_X * WORKGROUP_SIZE_X; 

var<workgroup> shared_weights: array<f32, BLOCK_SIZE>;
var<workgroup> shared_inputs: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn dense_forward(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let col = global_id.x; // Which batch 
    let row = global_id.y; // Which layer

    // Returns if outside the output dimensions
    if row >= constants.m || col >= constants.n {
        return;
    }
    let input_offset = col * constants.k; // Take the batch with offset
    let weight_offset = row * constants.k; // Take the layer with offset
    let local_x = local_idx % WORKGROUP_SIZE_X; // which local col or input
    let local_y = local_idx / WORKGROUP_SIZE_Y; // which local row or layer

    let iters = u32(ceil(f32(constants.k) / f32(WORKGROUP_SIZE_X)));
    var acc = 0.0f;
    // for (var k_block: u32 = 0u; k_block < iters; k_block++) {
    //     // Read shared data
    //     let block_offset_x = (k_block * WORKGROUP_SIZE_X) + local_x; // Only x (the input index) can be over anymore
    //     var weight = 0.0;
    //     var input = 0.0;
    //     if block_offset_x < constants.k {
    //         weight = weights[weight_offset + block_offset_x];
    //         input = inputs[input_offset + block_offset_x];
    //     }
    //     shared_weights[local_idx] = weight;
    //     shared_inputs[local_idx] = input;
    //
    //     workgroupBarrier();
    //
    //     // Multiply-Accumulate
    //     for (var i: u32 = 0u; i < WORKGROUP_SIZE_X; i++) {
    //         acc += shared_weights[local_y * WORKGROUP_SIZE_X + i] * shared_inputs[local_x + i * WORKGROUP_SIZE_X];
    //     }
    //     workgroupBarrier();
    // }
    for (var i: u32 = 0u; i < constants.k; i++) {
        acc += weights[weight_offset + i] * inputs[input_offset + i];
    }
    output[row * constants.n + col] = acc + biases[row];
}
