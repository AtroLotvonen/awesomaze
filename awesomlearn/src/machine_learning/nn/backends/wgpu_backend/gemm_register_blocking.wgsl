const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const TSM = 128u; // width ?
const TSN = 128u; // height ?
const TSK = 16u;
const WPTM = 8u;
const WPTN = 8u;
const LPTA = (TSK * TSM) / (RTSM * RTSN);
const LPTB = (TSK * TSN) / (RTSM * RTSN);
const RTSM = WORKGROUP_SIZE_X; //16u; // The reduced tile-size in dimension M (TSM/WPTM number of threads)
const RTSN = WORKGROUP_SIZE_Y; //16u; // The reduced tile-size in dimension N (TSN/WPTN number of threads)

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
const A_SUB_SIZE = TSK + TSM;
const B_SUB_SIZE = TSK + TSN;

var<workgroup> Asub: array<array<f32, A_SUB_SIZE>, 2>;
var<workgroup> Bsub: array<array<f32, B_SUB_SIZE>, 2>;

@compute @workgroup_size(RTSM, RTSN, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let M = constants.m;
    let N = constants.n;
    let K = constants.k;

    let tidm = local_id.x;
    let tidn = local_id.y;
    let gidm = workgroup_id.x;
    let gidn = workgroup_id.y;
    let tid = tidn * RTSM + tidm; // global thread id

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
            let id = la * RTSN * RTSM + tid;
            let row = id % TSM;
            let col = id / TSM;
            let tiled_index = TSK * t + col;
            let index_A = tiled_index * M + gidm * TSM + row;
            Asub[t % 2u][col * TSM + row] = A[index_A];
        }

        for (var lb: u32 = 0u; lb < LPTB; lb++) {
            let id = lb * RTSN * RTSM + tid;
            let row = id % TSN;
            let col = id / TSN;
            let tiled_index = TSK * t + col;
            let index_B = tiled_index * N + gidn * TSN + row;
            Bsub[t % 2u][col * TSN + row] = B[index_B];
        }

        // Synchronise
        workgroupBarrier();

        for (var k = 0u; k < TSK; k++) {

            // Cache the values if Bsub in registers
            for (var wn = 0u; wn < WPTN; wn++) {
                let col = tidn + wn * RTSN;
                B_reg[wn] = Bsub[t % 2u][k * TSN + col];
            }

            // Perform computation
            for (var wm = 0u; wm < WPTM; wm++) {
                let row = tidm + wm * RTSM;
                A_reg = Asub[t % 2u][k * TSM + row];
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
        let global_row = gidm * TSM + tidm + wm * RTSM;
        for (var wn: u32 = 0u; wn < WPTN; wn++) {
            let global_col = gidn * TSN + tidn + wn * RTSN;
            C[global_col * M + global_row] = acc[wm][wn];
        }
    }
}
