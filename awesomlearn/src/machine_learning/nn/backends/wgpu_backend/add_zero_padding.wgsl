const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;

struct Constants {
    p: u32, // width
    q: u32, // height
    p_xl: u32, // width
    q_xl: u32, // height
};

@group(0) @binding(0) var<uniform> constants: Constants;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tx = workgroup_id.x * WORKGROUP_SIZE_X + local_id.x;
    let ty = workgroup_id.y * WORKGROUP_SIZE_Y + local_id.y;

    if tx < constants.p_xl && ty < constants.q_xl {

        var value: f32;

        if tx < constants.p && ty < constants.q {
            value = input[ty * constants.p + tx];
        } else {
            value = 0.0f;
        }

        // Store the padded result
        output[ty * constants.p_xl + tx] = value;
    }
}

