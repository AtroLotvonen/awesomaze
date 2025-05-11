const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const GLOBAL_W = 32u;
const GLOBAL_H = 32u;
const A_W = 32u;
const A_H = 32u;
const A_C = 32u;
const A_N = 32u;
const B_W = 32u;
const B_H = 32u;
const B_C = 32u;
const B_N = 32u;
const DIM0 = 0u;
const DIM1 = 1u;
const DIM2 = 2u;
const DIM3 = 3u;
const LENGTH = 100u;

const TS = WORKGROUP_SIZE_X;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {

    let input_index = global_id.z * GLOBAL_H * GLOBAL_W + global_id.y * GLOBAL_W + global_id.x;

    if input_index < LENGTH {
        // var<workgroup> sharedMem: array<array<f32, WORKGROUP_SIZE_X>, WORKGROUP_SIZE_Y>;
        // sharedMem[local_id.y][local_id.x] = input[input_index];

        let value = input[input_index];
        // input dimension indices
        let input_row_i = input_index % A_W;
        let input_col_i = (input_index / A_W) % A_H;
        let input_channel_i = (input_index / (A_W * A_H)) % A_C;
        let input_batch_i = input_index / (A_W * A_H * A_C);
        // switcharoo the indices
        var t_dim = array<u32, 4>();
        t_dim[DIM0] = input_batch_i;
        t_dim[DIM1] = input_channel_i;
        t_dim[DIM2] = input_col_i;
        t_dim[DIM3] = input_row_i;

        let output_index = t_dim[0] * B_C * B_H * B_W + t_dim[1] * B_H * B_W + t_dim[2] * B_W + t_dim[3];

        output[output_index] = value;
        // output[output_index] = sharedMem[local_id.x][local_id.y];
    }
}
