const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const P = 128u;
const Q = 128u;
const N = 128u;
const POOL_SIZE_X = 2u;
const POOL_SIZE_Y = 2u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const NEG_INFINITY: f32 = -1e38;
var<workgroup> max_pool: array<array<f32, WORKGROUP_SIZE_X>, WORKGROUP_SIZE_Y>;

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
    let idx = WORKGROUP_SIZE_X * workgroup_id.x + tx;
    let idy = WORKGROUP_SIZE_Y * workgroup_id.y + ty;
    
    let out_idx = (WORKGROUP_SIZE_X / POOL_SIZE_X) * workgroup_id.x + tx;
    let out_idy = (WORKGROUP_SIZE_Y / POOL_SIZE_Y) * workgroup_id.y + ty;
    // Channel offset
    let z_offset = workgroup_id.z * P * Q;

    // Read the input to local memory. All threads must write so the barrier works.
    var inp = 0.0f;
    if idx < Q && idy < P {
        inp = input[z_offset + idy * Q + idx];
    }
    max_pool[ty][tx] = inp;

    workgroupBarrier();

    // Use only the first part of the threads in the pool for writing to the output
    // and also check that no threads outsize the output size are used for writing.
    if tx < (WORKGROUP_SIZE_X / POOL_SIZE_X) && ty < (WORKGROUP_SIZE_Y / POOL_SIZE_Y) && out_idx < (Q / POOL_SIZE_X) && out_idy < (P / POOL_SIZE_Y) {


        let pool_idx = tx * POOL_SIZE_X;
        let pool_idy = ty * POOL_SIZE_Y;
        var max_value = NEG_INFINITY;

        for (var y = 0u; y < POOL_SIZE_Y; y++) {
            for (var x = 0u; x < POOL_SIZE_X; x++) {
                let value = max_pool[pool_idy + y][pool_idx + x];
                max_value = max(max_value, value);
            }
        }

            // Out channel offset
        let out_z_offset = workgroup_id.z * (P / POOL_SIZE_Y) * (Q / POOL_SIZE_X);

        output[out_z_offset + out_idy * (Q / POOL_SIZE_X) + out_idx] = max_value;
    }
    // storageBarrier();
}
