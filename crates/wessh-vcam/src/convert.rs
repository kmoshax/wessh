//! BGR (the pipeline's frames) into what virtual cameras take: YUYV for v4l2loopback, NV12 for
//! Windows. One pass, into caller-owned buffers.

/// Bytes per BGR pixel.
const BGR_CHANNELS: usize = 3;

/// Studio-swing BT.601 RGB → YUV in 8.8 fixed point.
mod bt601 {
    pub const LUMA_OFFSET: i32 = 16;
    pub const CHROMA_OFFSET: i32 = 128;
    pub const ROUND: i32 = 128;
    pub const SHIFT: u32 = 8;
    pub const Y: [i32; 3] = [66, 129, 25];
    pub const U: [i32; 3] = [-38, -74, 112];
    pub const V: [i32; 3] = [112, -94, -18];
}

/// Convert BGR into packed YUYV (`width` must be even), chroma averaged over each pixel pair.
pub fn bgr_to_yuyv(src: &[u8], width: usize, height: usize, out: &mut [u8]) {
    use bt601::{CHROMA_OFFSET, LUMA_OFFSET, ROUND, SHIFT, U, V, Y};
    let dot = |m: [i32; 3], [b, g, r]: [i32; 3]| (m[0] * r + m[1] * g + m[2] * b + ROUND) >> SHIFT;
    let pairs = width / 2 * height;
    for (pair, quad) in src.as_chunks::<6>().0.iter().zip(out.as_chunks_mut::<4>().0).take(pairs) {
        let first = [pair[0], pair[1], pair[2]].map(i32::from);
        let second = [pair[3], pair[4], pair[5]].map(i32::from);
        let mean = std::array::from_fn(|c| (first[c] + second[c] + 1) / 2);
        let byte = |value: i32| value.clamp(0, 255) as u8;
        *quad = [
            byte(dot(Y, first) + LUMA_OFFSET),
            byte(dot(U, mean) + CHROMA_OFFSET),
            byte(dot(Y, second) + LUMA_OFFSET),
            byte(dot(V, mean) + CHROMA_OFFSET),
        ];
    }
}

/// Convert BGR into NV12 (`width` and `height` even): full-resolution luma, then `U V` pairs
/// averaged over each 2×2 block. `out` holds `width * height * 3 / 2` bytes.
pub fn bgr_to_nv12(src: &[u8], width: usize, height: usize, out: &mut [u8]) {
    use bt601::{CHROMA_OFFSET, LUMA_OFFSET, ROUND, SHIFT, U, V, Y};
    let dot = |m: [i32; 3], [b, g, r]: [i32; 3]| (m[0] * r + m[1] * g + m[2] * b + ROUND) >> SHIFT;
    let byte = |value: i32| value.clamp(0, 255) as u8;
    let luma_of = |pixel: [i32; 3]| byte(dot(Y, pixel) + LUMA_OFFSET);
    let stride = width * BGR_CHANNELS;
    let (luma, chroma) = out.split_at_mut((width * height).min(out.len()));
    let row_pairs = src.chunks_exact(stride * 2).zip(luma.chunks_exact_mut(width * 2)).zip(chroma.chunks_exact_mut(width));
    for ((rows, luma_rows), chroma_row) in row_pairs {
        let (top, bottom) = rows.split_at(stride);
        let (luma_top, luma_bottom) = luma_rows.split_at_mut(width);
        let blocks = top
            .as_chunks::<6>()
            .0
            .iter()
            .zip(bottom.as_chunks::<6>().0)
            .zip(luma_top.as_chunks_mut::<2>().0.iter_mut().zip(luma_bottom.as_chunks_mut::<2>().0));
        for (((upper, lower), (luma_upper, luma_lower)), uv) in blocks.zip(chroma_row.as_chunks_mut::<2>().0) {
            let pixel = |pair: &[u8; 6], i: usize| [pair[i * 3], pair[i * 3 + 1], pair[i * 3 + 2]].map(i32::from);
            let block = [pixel(upper, 0), pixel(upper, 1), pixel(lower, 0), pixel(lower, 1)];
            *luma_upper = [luma_of(block[0]), luma_of(block[1])];
            *luma_lower = [luma_of(block[2]), luma_of(block[3])];
            let mean = std::array::from_fn(|c| (block.iter().map(|p| p[c]).sum::<i32>() + 2) / 4);
            *uv = [byte(dot(U, mean) + CHROMA_OFFSET), byte(dot(V, mean) + CHROMA_OFFSET)];
        }
    }
}
