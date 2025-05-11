const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const P = 32u;
const Q = 32u;
const N = 32u;
const LENGTH = 100u;


@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {

    let global_index = global_id.z * P * Q + global_id.y * Q + global_id.x;

    if global_index < LENGTH {
        let x = input[global_index];

        output[global_index] = log(x);
    }
}
