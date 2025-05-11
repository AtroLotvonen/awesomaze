const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const M = 16u;
const N = 16u;
const K = 16u;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let global_col = global_id.x;
    let global_row = global_id.y; 

    if global_row < M && global_col < N {
        var acc = 0.0f;
        for (var k: u32 = 0u; k < K; k++) {
            acc += A[global_row * K + k] * B[global_col * K + k];//B[k * N + global_col];
        }
        C[global_row * N + global_col] = acc;
    }
}
