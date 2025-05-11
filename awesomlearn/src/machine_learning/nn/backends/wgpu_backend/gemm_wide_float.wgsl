const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const TS = WORKGROUP_SIZE_Y;
const WIDTH = 4u; // Float Width
const TSW = WORKGROUP_SIZE_X; // TS/WIDTH, this is computed to the X value when compiling the source already

struct Constants {
    m: u32, // layer_size in forward
    n: u32, // batch_size in forward
    k: u32, // input_size in forward
};

@group(0) @binding(0) var<uniform> constants: Constants;
@group(0) @binding(1) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> C: array<vec4<f32>>;

const BLOCK_SIZE = WORKGROUP_SIZE_X * WORKGROUP_SIZE_X; 

var<workgroup> Asub: array<array<vec4<f32>, TSW>, TS>;
var<workgroup> Bsub: array<array<vec4<f32>, TSW>, TS>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = local_id.x;
    let col = local_id.y;
    let global_row = TSW * workgroup_id.x + row;
    let global_col = TS * workgroup_id.y + col;

    // local memory is defined up
    var acc = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);


    let num_tiles = constants.k / TS;
    for (var t: u32 = 0u; t < num_tiles; t++) {
        
        // Load one tile of A and B into local memory
        let tiled_row = TSW * t + row;
        let tiled_col = TS * t + col;
        Asub[col][row] = A[tiled_col * (constants.m / WIDTH) + global_row];
        Bsub[col][row] = B[global_col * (constants.k / WIDTH) + tiled_row];

        workgroupBarrier();

        for (var k: u32 = 0u; k < TSW; k++) {
            let vec_B = Bsub[col][k];
            // for (var w: u32 = 0u; w < WIDTH; w++) {
            //     let vec_A = Asub[WIDTH * k + w][row];
            //     var val_B = 0.0f;
            //     switch (w) {
            //         case 0u:{ val_B = vec_B.x; break;}
            //         case 1u:{ val_B = vec_B.y; break;}
            //         case 2u:{ val_B = vec_B.z; break;}
            //         default: { val_B = vec_B.w; break;}
            //     }
            //     acc.x += vec_A.x * val_B;
            //     acc.y += vec_A.y * val_B;
            //     acc.z += vec_A.z * val_B;
            //     acc.w += vec_A.w * val_B;
            acc += Asub[WIDTH * k][row] * vec_B.x;
            acc += Asub[WIDTH * k + 1u][row] * vec_B.y;
            acc += Asub[WIDTH * k + 2u][row] * vec_B.z;
            acc += Asub[WIDTH * k + 3u][row] * vec_B.w;
            // }
        }
        workgroupBarrier();
    }

    C[global_col * (constants.m / WIDTH) + global_row] = acc;
}
