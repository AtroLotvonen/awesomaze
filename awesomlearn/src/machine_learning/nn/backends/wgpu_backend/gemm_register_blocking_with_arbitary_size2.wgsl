const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const M = 16u;
const N = 16u;
const K = 16u;
const TSM = 128u; // width ?
const TSN = 128u; // height ?
const TSK = 16u;
const WPTM = 8u;
const WPTN = 8u;

const LPTA = (TSK * WPTM * WPTN) / TSN;
const LPTB = (TSK * WPTM * WPTN) / TSM;
const RTSM = WORKGROUP_SIZE_X; //16u; // The reduced tile-size in dimension M (TSM/WPTM number of threads)
const RTSN = WORKGROUP_SIZE_Y; //16u; // The reduced tile-size in dimension N (TSN/WPTN number of threads)

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

// Padding to reduce the bank conflicts
const B_SUB_SIZE = TSK + 2u;

var<workgroup> Asub: array<array<f32, TSM>, TSK>;
var<workgroup> Bsub: array<array<f32, B_SUB_SIZE>, TSN>;

@compute @workgroup_size(RTSM, RTSN, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {

    let tidm = local_id.x;
    let tidn = local_id.y;
    let offset_M = TSM * workgroup_id.x;
    let offset_N = TSN * workgroup_id.y;

    var A_reg: f32;
    var B_reg = array<f32, WPTN>();
    // local memory is defined up
    var acc = array<array<f32, WPTN>, WPTM>();
    for (var wm: u32 = 0u; wm < WPTM; wm++) {
        for (var wn: u32 = 0u; wn < WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    let num_tiles = K / TSK;
    for (var t: u32 = 0u; t < num_tiles; t++) {
        
        // Load one tile of A and B into local memory
        for (var la: u32 = 0u; la < LPTA; la++) {
            let tid = tidn * RTSM + tidm;
            let id = la * RTSN * RTSM + tid;
            let row = id % TSM;
            let col = id / TSM;
            let tiled_index = TSK * t + col;
            Asub[col][row] = A[tiled_index * M + offset_M + row];
        }

        for (var lb: u32 = 0u; lb < LPTB; lb++) {
            let tid = tidn * RTSM + tidm;
            let id = lb * RTSN * RTSM + tid;
            let row = id % TSN;
            let col = id / TSN;
            let tiled_index = TSK * t + col;
            Bsub[row][col] = B[tiled_index * N + offset_N + row];
        }

        // Synchronise
        workgroupBarrier();

        for (var k = 0u; k < TSK; k++) {

            // Cache the values if Bsub in registers
            for (var wn = 0u; wn < WPTN; wn++) {
                let col = tidn + wn * RTSN;
                B_reg[wn] = Bsub[col][k];
            }

            // Perform computation
            for (var wm = 0u; wm < WPTM; wm++) {
                let row = tidm + wm * RTSM;
                A_reg = Asub[k][row];
                for (var wn = 0u; wn < WPTN; wn++) {
                    acc[wm][wn] += A_reg * B_reg[wn];
                }
            }
        }

        // Synchronise
        workgroupBarrier();
    }
    
    // Store result to C
    for (var wm: u32 = 0u; wm < WPTM; wm++) {
        let global_row = offset_M + tidm + wm * RTSM;
        for (var wn: u32 = 0u; wn < WPTN; wn++) {
            let global_col = offset_N + tidn + wn * RTSN;
            C[global_col * M + global_row] = acc[wm][wn];
        }
    }
}
