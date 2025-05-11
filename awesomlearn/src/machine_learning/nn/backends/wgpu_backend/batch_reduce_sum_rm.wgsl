const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const N = 32u;
const C = 32u;
const H = 32u;
const W = 32u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {

    let c = workgroup_id.z;

    let ty = local_id.y;
    let tx = local_id.x;

    let h = workgroup_id.y * WORKGROUP_SIZE_Y + ty;
    let w = workgroup_id.x * WORKGROUP_SIZE_X + tx;

    if h < H && w < W {
        var acc = 0.0f;
        for (var n = 0u; n < N; n++) {
            let input_index = n * C * H * W + c * H * W + h * W + w;
            acc += input[input_index];
        }

        let output_index = c * H * W + h * W + w;

        output[output_index] = acc;
    }
}
