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

const QS = Q + START_OFFSET_X;
const PS = P + START_OFFSET_Y;
const CS = C + START_OFFSET_Z;
const NS = N + START_OFFSET_N;

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

        let global_row_i = global_index % Q_XL;
        let global_col_i = (global_index / Q_XL) % P_XL;
        let global_channel_i = (global_index / (Q_XL * P_XL)) % C_XL;
        let global_batch_i = (global_index / (Q_XL * P_XL * C_XL));

        var value: f32;

        if global_row_i >= START_OFFSET_X && global_col_i >= START_OFFSET_Y 
            && global_channel_i >= START_OFFSET_Z && global_batch_i >= START_OFFSET_N 
            && global_row_i < QS && global_col_i < PS && global_channel_i < CS && global_batch_i < NS {
            
            let shifted_row = global_row_i - START_OFFSET_X;
            let shifted_col = global_col_i - START_OFFSET_Y;
            let shifted_channel = global_channel_i - START_OFFSET_Z;
            let shifted_batch = global_batch_i - START_OFFSET_N;

            let input_index = shifted_batch * C * P * Q + shifted_channel * P * Q + shifted_col * Q + shifted_row;

            value = input[input_index];
        } else {
            value = 0.0f;
        }

        // Store the padded result
        output[global_index] = value;
    }
}

