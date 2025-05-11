const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const P = 128u;
const Q = 128u;
const N = 128u;
const UPSAMPLE_SIZE_X = 2u;
const UPSAMPLE_SIZE_Y = 2u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const IN_P = P / UPSAMPLE_SIZE_X;
const IN_Q = Q / UPSAMPLE_SIZE_Y;
const DOWNSAMPLED_X = WORKGROUP_SIZE_X / UPSAMPLE_SIZE_X;
const DOWNSAMPLED_Y = WORKGROUP_SIZE_Y / UPSAMPLE_SIZE_Y;

var<workgroup> down_sampled: array<array<f32, DOWNSAMPLED_X>, DOWNSAMPLED_Y>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    // FIXME: this kernel currently supports only pooling of multiple of 2 and <= 16
    // to support odd numbered and larger pooling the input and output indices should
    // be computed with the fact in mind that one workgroup can compute the max pool 
    // for (WORKGROUP_SIZE_X / POOL_SIZE_X) * POOL_SIZE_X or WORKGROUP_SIZE_X - mod(WORKGROUP_SIZE_X, POOL_SIZE_X)
    // and the same for Y
    let tx = local_id.x;
    let ty = local_id.y;
    let tz = local_id.z;

    let in_idx = (WORKGROUP_SIZE_X / UPSAMPLE_SIZE_X) * workgroup_id.x + tx;
    let in_idy = (WORKGROUP_SIZE_Y / UPSAMPLE_SIZE_Y) * workgroup_id.y + ty;

    
    // Channel offset
    let in_z_offset = workgroup_id.z * IN_P * IN_Q;

    if in_idx < IN_Q && in_idy < IN_P && tx < DOWNSAMPLED_X && ty < DOWNSAMPLED_Y {
        down_sampled[ty][tx] = input[in_z_offset + in_idy * IN_Q + in_idx];
    }

    workgroupBarrier();

    let out_idx = WORKGROUP_SIZE_X * workgroup_id.x + tx;
    let out_idy = WORKGROUP_SIZE_Y * workgroup_id.y + ty;

    if out_idx < Q && out_idy < P {
        let out_z_offset = workgroup_id.z * P * Q;

        let dtx = tx / UPSAMPLE_SIZE_X;
        let dty = ty / UPSAMPLE_SIZE_Y;
        let input_value = down_sampled[dty][dtx];

        output[out_z_offset + out_idy * Q + out_idx] = input_value;
    }
}
