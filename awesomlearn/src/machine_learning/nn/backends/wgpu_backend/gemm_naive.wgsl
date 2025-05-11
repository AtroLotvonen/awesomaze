const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;

struct Constants {
    m: u32, // layer_size in forward
    n: u32, // batch_size in forward
    k: u32, // input_size in forward
};

@group(0) @binding(0) var<uniform> constants: Constants;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const BLOCK_SIZE = WORKGROUP_SIZE_X * WORKGROUP_SIZE_X; 

var<workgroup> shared_weights: array<f32, BLOCK_SIZE>;
var<workgroup> shared_inputs: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let global_row = global_id.x;
    let global_col = global_id.y; 

    if global_row < constants.m && global_col < constants.n {
        var acc = 0.0f;
        for (var k: u32 = 0u; k < constants.k; k++) {
            acc += A[k * constants.m + global_row] * B[global_col * constants.k + k];
        }
        C[global_col * constants.m + global_row] = acc;
    }
}
