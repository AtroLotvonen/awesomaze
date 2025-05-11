const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const GLOBAL_W = 32u;
const GLOBAL_H = 32u;
const W = 16u; // image width
const H = 16u; // image height
const C = 16u; // kernel and image channels
const N = 16u; // image batches
const S = 16u; // kernel width
const R = 16u; // kernel height
const LENGTH = 100u; // output_size

const COL_HEIGHT = C * R * S;
const COLS_PER_W = W - S + 1u;
const COLS_PER_H = H - R + 1u;
const COLS_PER_BATCH = COLS_PER_W * COLS_PER_H;

@group(0) @binding(0) var<storage, read> im: array<f32>;
@group(0) @binding(1) var<storage, read_write> col: array<f32>;


const TS = WORKGROUP_SIZE_X;

var<workgroup> sub_input: array<array<f32, TS>, TS>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {

    let out_i = global_id.z * GLOBAL_H * GLOBAL_W + global_id.y * GLOBAL_W + global_id.x;

    if out_i < LENGTH {
        // output col indices
        let col_col = out_i % COL_HEIGHT;
        let col_i = (out_i / COL_HEIGHT) % COLS_PER_BATCH;
        let batch_i = out_i / (COLS_PER_BATCH * COL_HEIGHT);
        // find image indices for the column when moving the kernel over the image row by row
        let image_offset_w = col_i % COLS_PER_W;
        let image_offset_h = col_i / COLS_PER_W;
        let kernel_offset_w = col_col % S;
        let kernel_offset_h = (col_col / S) % R;
        let kernel_offset_c = col_col / (R * S);

        let im_x = image_offset_w + kernel_offset_w;
        let im_y = image_offset_h + kernel_offset_h;
        let im_z = kernel_offset_c;
        let im_w = batch_i;
        let im_i = im_w * C * H * W + im_z * H * W + im_y * W + im_x;
        col[out_i] = im[im_i];
    }
}

