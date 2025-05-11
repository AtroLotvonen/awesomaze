const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;

struct Constants {
    m: u32, // layer_size in forward
    n: u32, // batch_size in forward
    k: u32, // input_size in forward
};

@group(0) @binding(0) var<uniform> constants: Constants;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const BLOCK_SIZE = WORKGROUP_SIZE_X * WORKGROUP_SIZE_X; 

const TS = WORKGROUP_SIZE_X;

// var<workgroup> Asub: array<f32, BLOCK_SIZE>;
// var<workgroup> Bsub: array<f32, BLOCK_SIZE>;

var<workgroup> Asub: array<array<f32, TS>, TS>;
var<workgroup> Bsub: array<array<f32, TS>, TS>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = local_id.x;
    let col = local_id.y;
    let TS = WORKGROUP_SIZE_X;
    let global_row = TS * workgroup_id.x + row;
    let global_col = TS * workgroup_id.y + col;

    // local memory is defined up
    var acc: f32 = 0.0;

    let num_tiles = constants.k / TS;
    for (var t: u32 = 0u; t < num_tiles; t++) {
        
        // Load one tile of A and B into local memory
        let tiled_row = TS * t + row;
        let tiled_col = TS * t + col;
        Asub[col][row] = A[tiled_col * constants.m + global_row];
        Bsub[col][row] = B[global_col * constants.k + tiled_row];

        workgroupBarrier();

        for (var k: u32 = 0u; k < TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }
        workgroupBarrier();
    }
    C[global_col * constants.m + global_row] = acc;
}
