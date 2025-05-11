const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const P = 128u;
const Q = 128u;
const P_XL = 128u;
const Q_XL = 128u;
const START_OFFSET_X = 0u;
const START_OFFSET_Y = 0u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tx = workgroup_id.x * WORKGROUP_SIZE_X + local_id.x;
    let ty = workgroup_id.y * WORKGROUP_SIZE_Y + local_id.y;

    if tx < Q && ty < P {

        // shift the tx and ty with offset
        let tx_shifted = tx + START_OFFSET_X;
        let ty_shifted = ty + START_OFFSET_Y;
        // Store the unpadded result
        output[ty * Q + tx] = input[ty_shifted * Q_XL + tx_shifted];
    }
}

