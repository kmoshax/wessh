//! Minimal protobuf reader that extracts one initializer from an ONNX file (inswapper keeps its
//! embedding map there), so no protobuf dependency is needed.

use crate::{Error, Result};

fn invalid(msg: &str) -> Error {
    Error::InvalidOnnx(msg.to_owned())
}

fn varint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut out = 0u64;
    for shift in (0..64).step_by(7) {
        let b = *buf.get(*pos).ok_or_else(|| invalid("truncated varint"))?;
        *pos += 1;
        out |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Ok(out);
        }
    }
    Err(invalid("varint too long"))
}

enum Value<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Fixed,
}

/// Iterate `(field_number, value)` pairs of one protobuf message.
fn fields(buf: &[u8]) -> impl Iterator<Item = Result<(u64, Value<'_>)>> {
    let mut pos = 0;
    std::iter::from_fn(move || {
        if pos >= buf.len() {
            return None;
        }
        let res = (|| -> Result<(u64, Value<'_>)> {
            let key = varint(buf, &mut pos)?;
            let value = match key & 7 {
                0 => Value::Varint(varint(buf, &mut pos)?),
                1 => {
                    pos += 8;
                    Value::Fixed
                }
                2 => {
                    let len = varint(buf, &mut pos)? as usize;
                    let slice = buf.get(pos..pos + len).ok_or_else(|| invalid("truncated field"))?;
                    pos += len;
                    Value::Bytes(slice)
                }
                5 => {
                    pos += 4;
                    Value::Fixed
                }
                wt => return Err(invalid(&format!("unsupported wire type {wt}"))),
            };
            Ok((key >> 3, value))
        })();
        if res.is_err() {
            pos = buf.len();
        }
        Some(res)
    })
}

pub struct Initializer {
    pub dims: Vec<i64>,
    pub data: Vec<f32>,
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    let bits = match exp {
        0 if frac == 0 => sign,
        0 => {
            // subnormal: normalize
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

fn parse_tensor(buf: &[u8]) -> Result<Initializer> {
    let (mut dims, mut dtype, mut raw, mut floats) = (Vec::new(), 0u64, None, Vec::new());
    for f in fields(buf) {
        match f? {
            (1, Value::Varint(d)) => dims.push(d as i64),
            (1, Value::Bytes(packed)) => {
                let mut p = 0;
                while p < packed.len() {
                    dims.push(varint(packed, &mut p)? as i64);
                }
            }
            (2, Value::Varint(t)) => dtype = t,
            (4, Value::Bytes(packed)) => {
                floats.extend(packed.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)));
            }
            (9, Value::Bytes(r)) => raw = Some(r),
            _ => {}
        }
    }
    let data = match (dtype, raw) {
        (1, Some(r)) => r.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        (10, Some(r)) => r.as_chunks::<2>().0.iter().map(|c| f16_to_f32(u16::from_le_bytes(*c))).collect(),
        (1, None) => floats,
        (t, _) => return Err(invalid(&format!("unsupported initializer dtype {t}"))),
    };
    Ok(Initializer { dims, data })
}

/// Return the last initializer of the model graph (where inswapper keeps its `emap`).
pub fn last_initializer(model: &[u8]) -> Result<Initializer> {
    for f in fields(model) {
        if let (7, Value::Bytes(graph)) = f? {
            let mut last = None;
            for g in fields(graph) {
                if let (5, Value::Bytes(t)) = g? {
                    last = Some(t);
                }
            }
            return parse_tensor(last.ok_or_else(|| invalid("graph has no initializers"))?);
        }
    }
    Err(invalid("no graph in model"))
}

#[cfg(test)]
mod tests {
    use super::f16_to_f32;

    #[test]
    fn half_conversion() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert!((f16_to_f32(0x3555) - 0.33325).abs() < 1e-4);
        assert!(f16_to_f32(0x0001) > 0.0);
    }
}
