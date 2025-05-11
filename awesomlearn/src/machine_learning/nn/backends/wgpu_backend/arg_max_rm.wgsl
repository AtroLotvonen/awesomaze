const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const P = 32u;
const Q = 32u;
const A_W = 32u; // reduced dims 
const A_H = 32u;
const A_C = 32u;
const A_N = 32u;
const B_W = 32u; // orig dims
const B_H = 32u;
const B_C = 32u;
const B_N = 32u;
const LENGTH = 32u;
const STEP = 32u;
const COUNT = 32u;


const NEG_INFINITY: f32 = -1e38;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {


    let output_index = global_id.z * P * Q + global_id.y * Q + global_id.x;

    if output_index < LENGTH {

        let w = output_index % A_W;
        let h = (output_index / A_W) % A_H;
        let c = (output_index / (A_W * A_H)) % A_C;
        let n = output_index / (A_W * A_H * A_C);

        let input_index = n * B_C * B_H * B_W + c * B_H * B_W + h * B_W + w;
        var max_dim = NEG_INFINITY;
        var arg_max = 0u;

        for (var i = 0u; i < COUNT; i++) {
            let acc_index = input_index + i * STEP;
            if input[acc_index] > max_dim {
                max_dim = input[acc_index];
                arg_max = i;
            }
        }

        output[output_index] = f32(arg_max);
    }
}
