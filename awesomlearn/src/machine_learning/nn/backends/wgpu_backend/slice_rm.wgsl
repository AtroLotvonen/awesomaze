const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const GLOBAL_W = 16u;
const GLOBAL_H = 16u;
const P = 16u; // image height
const Q = 16u; // image width
const N = 16u; // image batches
const C = 16u; // image channels
const P_XL = 128u;
const Q_XL = 128u;
const N_XL = 128u;
const C_XL = 128u;
const START_OFFSET_X = 0u;
const START_OFFSET_Y = 0u;
const START_OFFSET_Z = 0u;
const START_OFFSET_N = 0u;
const LENGTH = 100u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let global_index = global_id.z * GLOBAL_H * GLOBAL_W + global_id.y * GLOBAL_W + global_id.x;

    if global_index < LENGTH {

        let global_row_i = global_index % Q;
        let global_col_i = (global_index / Q) % P;
        let global_channel_i = (global_index / (Q * P)) % C;
        let global_batch_i = (global_index / (Q * P * C));

        let shifted_row = global_row_i + START_OFFSET_X;
        let shifted_col = global_col_i + START_OFFSET_Y;
        let shifted_channel = global_channel_i + START_OFFSET_Z;
        let shifted_batch = global_batch_i + START_OFFSET_N;

        let input_index = shifted_batch * C_XL * P_XL * Q_XL + shifted_channel * P_XL * Q_XL + shifted_col * Q_XL + shifted_row;

        // Store the unpadded result
        output[global_index] = input[input_index];
    }
}

