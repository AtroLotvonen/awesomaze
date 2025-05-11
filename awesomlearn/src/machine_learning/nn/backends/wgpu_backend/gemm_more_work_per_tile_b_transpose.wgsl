const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const TS = 32u;
const WPT = 8u; // Work Per Thread
const RTS = TS / WPT; // TS/WPT == WORKGROUP_SIZE_Y 
const TSDK = 16u; // The tile-size in dimension K (for this kernel, not needed elsewhere)
const LPT = (TSDK * WPT) / TS;
// const LPT = (TSDK * TS) / (TS * RTS);


struct Constants {
    m: u32, // layer_size in forward
    n: u32, // batch_size in forward
    k: u32, // input_size in forward
};

@group(0) @binding(0) var<uniform> constants: Constants;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

// Padding to reduce the bank conflicts
const B_SUB_SIZE = TSDK + 2u;

var<workgroup> Asub: array<array<f32, TS>, TSDK>;
var<workgroup> Bsub: array<array<f32, B_SUB_SIZE>, TS>;

@compute @workgroup_size(TS, RTS, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = local_id.x;
    let col = local_id.y;
    let global_row = TS * workgroup_id.x + row;
    let global_col = TS * workgroup_id.y + col;

    // local memory is defined up
    var acc = array<f32, WPT>();
    for (var w: u32 = 0u; w < WPT; w++) {
        acc[w] = 0.0f;
    }

    let num_tiles = constants.k / TSDK;
    for (var t: u32 = 0u; t < num_tiles; t++) {
        
        // Load one tile of A and B into local memory
        for (var l: u32 = 0u; l < LPT; l++) {
            let tiled_index = TSDK * t + col + l * RTS;
            let index_A = tiled_index * constants.m + TS * workgroup_id.x + row;
            let index_B = tiled_index * constants.n + TS * workgroup_id.y + row;
            Asub[col + l * RTS][row] = A[index_A];
            Bsub[row][col + l * RTS] = B[index_B];
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TSDK; k++) {
            for (var w: u32 = 0u; w < WPT; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w * RTS][k];
            }
        }
        workgroupBarrier();
    }
    for (var w: u32 = 0u; w < WPT; w++) {
        C[(global_col + w * RTS) * constants.m + global_row] = acc[w];
    }
}
