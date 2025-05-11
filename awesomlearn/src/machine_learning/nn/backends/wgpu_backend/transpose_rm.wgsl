const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;

struct Constants {
    p: u32, // width
    q: u32, // height
};

@group(0) @binding(0) var<uniform> constants: Constants;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> buffer: array<array<f32, WORKGROUP_SIZE_Y>, WORKGROUP_SIZE_X>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tx = local_id.x;
    let ty = local_id.y;
    let id0 = WORKGROUP_SIZE_X * workgroup_id.x + tx; // 0..P
    let id1 = WORKGROUP_SIZE_Y * workgroup_id.y + ty; // 0..Q

    // Swapping the x and y coordinates to perform the rotation (coalesced)
    if id0 < constants.p && id1 < constants.q {
        buffer[ty][tx] = input[id1 * constants.p + id0];
    }

    workgroupBarrier();

    // The swap is already done in the local memory
    let new_id0 = workgroup_id.y * WORKGROUP_SIZE_Y + tx;
    let new_id1 = workgroup_id.x * WORKGROUP_SIZE_X + ty;

    // Store the transposed result (coalesed)
    if new_id0 < constants.q && new_id1 < constants.p {
        output[new_id1 * constants.q + new_id0] = buffer[tx][ty];
    }
}
