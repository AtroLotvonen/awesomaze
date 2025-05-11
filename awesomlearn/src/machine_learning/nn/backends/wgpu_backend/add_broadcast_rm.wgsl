const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const N = 32u;
const C = 32u;
const H = 32u;
const W = 32u;
const A_N = 32u;
const A_C = 32u;
const A_H = 32u;
const A_W = 32u;
const B_N = 32u;
const B_C = 32u;
const B_H = 32u;
const B_W = 32u;
const BROAD_A_N = 32u;
const BROAD_A_C = 32u;
const BROAD_A_H = 32u;
const BROAD_A_W = 32u;
const BROAD_B_N = 32u;
const BROAD_B_C = 32u;
const BROAD_B_H = 32u;
const BROAD_B_W = 32u;


@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {

    let nc = workgroup_id.z;
    let n = nc / C;
    let c = nc % C;

    let ty = local_id.y;
    let tx = local_id.x;

    let h = workgroup_id.y * WORKGROUP_SIZE_Y + ty;
    let w = workgroup_id.x * WORKGROUP_SIZE_X + tx;



    if h < H  && w < W {
        let global_index = n * C * H * W + c * H * W + h * W + w;

        let a_index = (BROAD_A_N * n) * A_C * A_H * A_W + ( BROAD_A_C * c ) * A_H * A_W + ( BROAD_A_H * h ) * A_W + (BROAD_A_W * w);
        let b_index = (BROAD_B_N * n) * B_C * B_H * B_W + ( BROAD_B_C * c ) * B_H * B_W + ( BROAD_B_H * h ) * B_W + (BROAD_B_W * w);

        output[global_index] = A[a_index] + B[b_index];
    }
}
