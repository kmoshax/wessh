//! Minimal protobuf reader / writer for ONNX files, so no protobuf dependency is needed. It
//! extracts one initializer (inswapper keeps its embedding map there) and patches graphs:
//! pinning input shapes, replacing the outputs, appending nodes and pruning every node the new
//! outputs don't need.

use crate::{Error, Result};
use std::collections::HashSet;

fn invalid(msg: &str) -> Error {
    Error::InvalidOnnx(msg.to_owned())
}

// Field numbers from `onnx.proto`.
/// `ModelProto.graph`.
const MODEL_GRAPH: u64 = 7;
/// `GraphProto.node`, `.initializer`, `.input`, `.output`.
const GRAPH_NODE: u64 = 1;
const GRAPH_INITIALIZER: u64 = 5;
const GRAPH_INPUT: u64 = 11;
const GRAPH_OUTPUT: u64 = 12;
/// `NodeProto.input`, `.output`, `.name`, `.op_type`, `.attribute`.
const NODE_INPUT: u64 = 1;
const NODE_OUTPUT: u64 = 2;
const NODE_NAME: u64 = 3;
const NODE_OP_TYPE: u64 = 4;
const NODE_ATTRIBUTE: u64 = 5;
/// `AttributeProto.name`, `.i`, `.ints`, `.type` (`INT` = 2).
const ATTRIBUTE_NAME: u64 = 1;
const ATTRIBUTE_INT: u64 = 3;
const ATTRIBUTE_INTS: u64 = 8;
const ATTRIBUTE_TYPE: u64 = 20;
const ATTRIBUTE_TYPE_INT: u64 = 2;
/// `ValueInfoProto.name`, `.type`; `TypeProto.tensor_type`; `TypeProto.Tensor.elem_type`, `.shape`;
/// `TensorShapeProto.dim`; `Dimension.dim_value`.
const VALUE_INFO_NAME: u64 = 1;
const VALUE_INFO_TYPE: u64 = 2;
const TYPE_TENSOR: u64 = 1;
const TENSOR_ELEM_TYPE: u64 = 1;
const TENSOR_SHAPE: u64 = 2;
const SHAPE_DIM: u64 = 1;
const DIM_VALUE: u64 = 1;
/// `TensorProto.dims`, `.data_type`, `.float_data`, `.name`, `.raw_data`.
const TENSOR_DIMS: u64 = 1;
const TENSOR_DATA_TYPE: u64 = 2;
const TENSOR_FLOAT_DATA: u64 = 4;
const TENSOR_NAME: u64 = 8;
const TENSOR_RAW_DATA: u64 = 9;

/// Wire types.
const WIRE_VARINT: u64 = 0;
const WIRE_FIXED64: u64 = 1;
const WIRE_BYTES: u64 = 2;
const WIRE_FIXED32: u64 = 5;

/// `TensorProto.DataType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    Float = 1,
    Int64 = 7,
    Float16 = 10,
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

/// One field of a message: number, value and its complete encoding (key included).
type RawField<'a> = (u64, Value<'a>, &'a [u8]);

/// Iterate `(field_number, value)` pairs of one protobuf message.
fn fields(buf: &[u8]) -> impl Iterator<Item = Result<(u64, Value<'_>)>> {
    raw_fields(buf).map(|f| f.map(|(number, value, _)| (number, value)))
}

/// Like [`fields`], plus each field's encoding so it can be copied verbatim.
fn raw_fields(buf: &[u8]) -> impl Iterator<Item = Result<RawField<'_>>> {
    let mut pos = 0;
    std::iter::from_fn(move || {
        if pos >= buf.len() {
            return None;
        }
        let start = pos;
        let res = (|| -> Result<(u64, Value<'_>)> {
            let key = varint(buf, &mut pos)?;
            let value = match key & 7 {
                WIRE_VARINT => Value::Varint(varint(buf, &mut pos)?),
                WIRE_FIXED64 => {
                    pos += 8;
                    Value::Fixed
                }
                WIRE_BYTES => {
                    let len = varint(buf, &mut pos)? as usize;
                    let slice = buf.get(pos..pos + len).ok_or_else(|| invalid("truncated field"))?;
                    pos += len;
                    Value::Bytes(slice)
                }
                WIRE_FIXED32 => {
                    pos += 4;
                    Value::Fixed
                }
                wt => return Err(invalid(&format!("unsupported wire type {wt}"))),
            };
            if pos > buf.len() {
                return Err(invalid("truncated field"));
            }
            Ok((key >> 3, value))
        })();
        match res {
            Ok((number, value)) => Some(Ok((number, value, &buf[start..pos]))),
            Err(e) => {
                pos = buf.len();
                Some(Err(e))
            }
        }
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
    const FLOAT: u64 = DataType::Float as u64;
    const FLOAT16: u64 = DataType::Float16 as u64;
    let (mut dims, mut dtype, mut raw, mut floats) = (Vec::new(), 0u64, None, Vec::new());
    for f in fields(buf) {
        match f? {
            (TENSOR_DIMS, Value::Varint(d)) => dims.push(d as i64),
            (TENSOR_DIMS, Value::Bytes(packed)) => {
                let mut p = 0;
                while p < packed.len() {
                    dims.push(varint(packed, &mut p)? as i64);
                }
            }
            (TENSOR_DATA_TYPE, Value::Varint(t)) => dtype = t,
            (TENSOR_FLOAT_DATA, Value::Bytes(packed)) => {
                floats.extend(packed.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)));
            }
            (TENSOR_RAW_DATA, Value::Bytes(r)) => raw = Some(r),
            _ => {}
        }
    }
    let data = match (dtype, raw) {
        (FLOAT, Some(r)) => r.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        (FLOAT16, Some(r)) => r.as_chunks::<2>().0.iter().map(|c| f16_to_f32(u16::from_le_bytes(*c))).collect(),
        (FLOAT, None) => floats,
        (t, _) => return Err(invalid(&format!("unsupported initializer dtype {t}"))),
    };
    Ok(Initializer { dims, data })
}

/// The model's main graph message.
fn graph(model: &[u8]) -> Result<&[u8]> {
    for f in fields(model) {
        if let (MODEL_GRAPH, Value::Bytes(graph)) = f? {
            return Ok(graph);
        }
    }
    Err(invalid("no graph in model"))
}

/// Return the last initializer of the model graph (where inswapper keeps its `emap`).
pub fn last_initializer(model: &[u8]) -> Result<Initializer> {
    let mut last = None;
    for g in fields(graph(model)?) {
        if let (GRAPH_INITIALIZER, Value::Bytes(t)) = g? {
            last = Some(t);
        }
    }
    parse_tensor(last.ok_or_else(|| invalid("graph has no initializers"))?)
}

/// A graph value with a fully static tensor type.
#[derive(Clone, Copy, Debug)]
pub struct TensorInfo<'a> {
    pub name: &'a str,
    pub data_type: DataType,
    pub dims: &'a [i64],
}

/// A node to append; only integer attributes are supported.
#[derive(Clone, Copy, Debug)]
pub struct Node<'a> {
    pub op_type: &'a str,
    pub name: &'a str,
    pub inputs: &'a [&'a str],
    pub outputs: &'a [&'a str],
    pub attributes: &'a [(&'a str, i64)],
}

/// Edits applied to a model's main graph by [`patch_graph`].
#[derive(Clone, Copy, Debug, Default)]
pub struct GraphPatch<'a> {
    /// Graph inputs whose declared type is replaced (matched by name).
    pub inputs: &'a [TensorInfo<'a>],
    /// `(from, to)`: nodes of op type `from` become `to`, without attributes — e.g.
    /// `AveragePool` whose kernel spans the whole feature map → `GlobalAveragePool`, which
    /// works at any input size.
    pub retype: &'a [(&'a str, &'a str)],
    /// Rewrite every `ConvTranspose` padded only at the end — which ONNX Runtime's CUDA provider
    /// can't run, so it silently falls back to the CPU — into an unpadded `ConvTranspose`
    /// followed by a `Slice` that trims the end: the same result, entirely on the GPU.
    pub unpad_conv_transpose: bool,
    /// Nodes appended after the existing ones.
    pub nodes: &'a [Node<'a>],
    /// The new graph outputs; every existing output is dropped, and so is every node none of
    /// these depends on. Empty keeps the outputs (and every node).
    pub outputs: &'a [TensorInfo<'a>],
}

/// Apply `patch` to the serialized `model`, returning the new serialized model.
pub fn patch_graph(model: &[u8], patch: &GraphPatch<'_>) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(model.len());
    for f in raw_fields(model) {
        match f? {
            (MODEL_GRAPH, Value::Bytes(graph), _) => put_bytes(&mut out, MODEL_GRAPH, &patch_graph_message(graph, patch)?),
            (_, _, raw) => out.extend_from_slice(raw),
        }
    }
    Ok(out)
}

fn patch_graph_message(graph: &[u8], patch: &GraphPatch<'_>) -> Result<Vec<u8>> {
    // Walk the nodes backwards from the outputs to find the ones still needed.
    let mut nodes = Vec::new();
    for f in raw_fields(graph) {
        if let (GRAPH_NODE, Value::Bytes(node), _) = f? {
            nodes.push(node_io(node)?);
        }
    }
    let mut appended: Vec<Vec<u8>> = patch.nodes.iter().map(encode_node).collect();
    let mut needed: HashSet<&str> = patch.outputs.iter().map(|o| o.name).collect();
    for node in patch.nodes.iter().rev() {
        if node.outputs.iter().any(|o| needed.contains(o)) {
            needed.extend(node.inputs.iter().copied());
        }
    }
    let mut keep = vec![patch.outputs.is_empty(); nodes.len()];
    for (index, (inputs, outputs, _)) in nodes.iter().enumerate().rev() {
        if outputs.iter().any(|o| needed.contains(o)) {
            keep[index] = true;
            needed.extend(inputs.iter().copied());
        }
    }

    let mut out = Vec::with_capacity(graph.len());
    let mut initializers = Vec::new();
    let mut node_index = 0;
    for f in raw_fields(graph) {
        match f? {
            (GRAPH_NODE, Value::Bytes(node), raw) => {
                let op_type = nodes[node_index].2;
                if !keep[node_index] {
                } else if let Some((_, to)) = patch.retype.iter().find(|(from, _)| *from == op_type) {
                    put_bytes(&mut out, GRAPH_NODE, &retype_node(node, to)?);
                } else if !(patch.unpad_conv_transpose
                    && op_type == "ConvTranspose"
                    && unpad_conv_transpose(node, &mut out, &mut initializers)?)
                {
                    out.extend_from_slice(raw);
                }
                node_index += 1;
            }
            (GRAPH_OUTPUT, _, raw) => {
                if patch.outputs.is_empty() {
                    out.extend_from_slice(raw);
                }
            }
            (GRAPH_INPUT, Value::Bytes(info), raw) => {
                let name = value_info_name(info)?;
                match patch.inputs.iter().find(|i| i.name == name) {
                    Some(replacement) => put_bytes(&mut out, GRAPH_INPUT, &encode_value_info(replacement)),
                    None => out.extend_from_slice(raw),
                }
            }
            (_, _, raw) => out.extend_from_slice(raw),
        }
    }
    for node in appended.drain(..) {
        put_bytes(&mut out, GRAPH_NODE, &node);
    }
    out.extend_from_slice(&initializers);
    for output in patch.outputs {
        put_bytes(&mut out, GRAPH_OUTPUT, &encode_value_info(output));
    }
    Ok(out)
}

/// Inputs, outputs and op type of a node.
type NodeIo<'a> = (Vec<&'a str>, Vec<&'a str>, &'a str);

/// Input and output value names and op type of a serialized node.
fn node_io(node: &[u8]) -> Result<NodeIo<'_>> {
    let (mut inputs, mut outputs, mut op_type) = (Vec::new(), Vec::new(), "");
    for f in fields(node) {
        match f? {
            (NODE_INPUT, Value::Bytes(name)) => inputs.push(utf8(name)?),
            (NODE_OUTPUT, Value::Bytes(name)) => outputs.push(utf8(name)?),
            (NODE_OP_TYPE, Value::Bytes(name)) => op_type = utf8(name)?,
            _ => {}
        }
    }
    Ok((inputs, outputs, op_type))
}

/// Write `node` (a `ConvTranspose`) as an unpadded `ConvTranspose` plus a `Slice` trimming the
/// end padding into `out`, with the slice parameters into `initializers`. Returns `false`, and
/// writes nothing, when the node has start padding (which this rewrite doesn't cover) or none.
fn unpad_conv_transpose(node: &[u8], out: &mut Vec<u8>, initializers: &mut Vec<u8>) -> Result<bool> {
    let (mut name, mut output, mut pads) = ("", "", Vec::new());
    for f in fields(node) {
        match f? {
            (NODE_NAME, Value::Bytes(n)) => name = utf8(n)?,
            (NODE_OUTPUT, Value::Bytes(n)) => output = utf8(n)?,
            (NODE_ATTRIBUTE, Value::Bytes(attribute)) if attribute_name(attribute)? == "pads" => pads = attribute_ints(attribute)?,
            _ => {}
        }
    }
    let (begin, end) = pads.split_at(pads.len() / 2);
    if pads.is_empty() || begin.iter().any(|&p| p != 0) || end.iter().all(|&p| p == 0) {
        return Ok(false);
    }
    // Spatial axes start after batch and channels; `i64::MAX` slices to the end.
    let axes: Vec<i64> = (2..).take(end.len()).collect();
    let ends: Vec<i64> = end.iter().map(|&p| if p == 0 { i64::MAX } else { -p }).collect();
    let starts = vec![0; end.len()];
    let unpadded = format!("{output}__unpadded");
    let names = ["starts", "ends", "axes"].map(|what| format!("{name}__{what}"));
    for (tensor, values) in names.iter().zip([&starts, &ends, &axes]) {
        put_bytes(initializers, GRAPH_INITIALIZER, &encode_int64_tensor(tensor, values));
    }

    let mut rewritten = Vec::with_capacity(node.len());
    for f in raw_fields(node) {
        match f? {
            (NODE_OUTPUT, _, _) => put_bytes(&mut rewritten, NODE_OUTPUT, unpadded.as_bytes()),
            (NODE_ATTRIBUTE, Value::Bytes(attribute), _) if attribute_name(attribute)? == "pads" => {}
            (_, _, raw) => rewritten.extend_from_slice(raw),
        }
    }
    put_bytes(out, GRAPH_NODE, &rewritten);
    let slice_name = format!("{name}__slice");
    let inputs = [unpadded.as_str(), &names[0], &names[1], &names[2]];
    let slice = Node { op_type: "Slice", name: &slice_name, inputs: &inputs, outputs: &[output], attributes: &[] };
    put_bytes(out, GRAPH_NODE, &encode_node(&slice));
    Ok(true)
}

fn attribute_name(attribute: &[u8]) -> Result<&str> {
    for f in fields(attribute) {
        if let (ATTRIBUTE_NAME, Value::Bytes(name)) = f? {
            return utf8(name);
        }
    }
    Err(invalid("attribute without a name"))
}

/// The `ints` of an attribute, packed or not.
fn attribute_ints(attribute: &[u8]) -> Result<Vec<i64>> {
    let mut ints = Vec::new();
    for f in fields(attribute) {
        match f? {
            (ATTRIBUTE_INTS, Value::Varint(v)) => ints.push(v as i64),
            (ATTRIBUTE_INTS, Value::Bytes(packed)) => {
                let mut p = 0;
                while p < packed.len() {
                    ints.push(varint(packed, &mut p)? as i64);
                }
            }
            _ => {}
        }
    }
    Ok(ints)
}

/// A 1-D `int64` initializer.
fn encode_int64_tensor(name: &str, values: &[i64]) -> Vec<u8> {
    let mut out = Vec::new();
    put_uint(&mut out, TENSOR_DIMS, values.len() as u64);
    put_uint(&mut out, TENSOR_DATA_TYPE, DataType::Int64 as u64);
    put_bytes(&mut out, TENSOR_NAME, name.as_bytes());
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    put_bytes(&mut out, TENSOR_RAW_DATA, &raw);
    out
}

/// `node` with op type `to` and no attributes.
fn retype_node(node: &[u8], to: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(node.len());
    for f in raw_fields(node) {
        match f? {
            (NODE_OP_TYPE | NODE_ATTRIBUTE, _, _) => {}
            (_, _, raw) => out.extend_from_slice(raw),
        }
    }
    put_bytes(&mut out, NODE_OP_TYPE, to.as_bytes());
    Ok(out)
}

fn value_info_name(info: &[u8]) -> Result<&str> {
    for f in fields(info) {
        if let (VALUE_INFO_NAME, Value::Bytes(name)) = f? {
            return utf8(name);
        }
    }
    Err(invalid("value without a name"))
}

fn utf8(bytes: &[u8]) -> Result<&str> {
    std::str::from_utf8(bytes).map_err(|_| invalid("name is not UTF-8"))
}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn put_key(out: &mut Vec<u8>, field: u64, wire: u64) {
    put_varint(out, (field << 3) | wire);
}

fn put_bytes(out: &mut Vec<u8>, field: u64, bytes: &[u8]) {
    put_key(out, field, WIRE_BYTES);
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn put_uint(out: &mut Vec<u8>, field: u64, value: u64) {
    put_key(out, field, WIRE_VARINT);
    put_varint(out, value);
}

fn encode_value_info(info: &TensorInfo<'_>) -> Vec<u8> {
    let mut shape = Vec::new();
    for &dim in info.dims {
        let mut d = Vec::new();
        // Negative int64s are encoded as their two's complement, like protobuf does.
        put_uint(&mut d, DIM_VALUE, dim as u64);
        put_bytes(&mut shape, SHAPE_DIM, &d);
    }
    let mut tensor = Vec::new();
    put_uint(&mut tensor, TENSOR_ELEM_TYPE, info.data_type as u64);
    put_bytes(&mut tensor, TENSOR_SHAPE, &shape);
    let mut ty = Vec::new();
    put_bytes(&mut ty, TYPE_TENSOR, &tensor);
    let mut out = Vec::new();
    put_bytes(&mut out, VALUE_INFO_NAME, info.name.as_bytes());
    put_bytes(&mut out, VALUE_INFO_TYPE, &ty);
    out
}

fn encode_node(node: &Node<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    for input in node.inputs {
        put_bytes(&mut out, NODE_INPUT, input.as_bytes());
    }
    for output in node.outputs {
        put_bytes(&mut out, NODE_OUTPUT, output.as_bytes());
    }
    put_bytes(&mut out, NODE_NAME, node.name.as_bytes());
    put_bytes(&mut out, NODE_OP_TYPE, node.op_type.as_bytes());
    for &(name, value) in node.attributes {
        let mut attribute = Vec::new();
        put_bytes(&mut attribute, ATTRIBUTE_NAME, name.as_bytes());
        put_uint(&mut attribute, ATTRIBUTE_INT, value as u64);
        put_uint(&mut attribute, ATTRIBUTE_TYPE, ATTRIBUTE_TYPE_INT);
        put_bytes(&mut out, NODE_ATTRIBUTE, &attribute);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversion() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert!((f16_to_f32(0x3555) - 0.33325).abs() < 1e-4);
        assert!(f16_to_f32(0x0001) > 0.0);
    }

    fn names(graph: &[u8], field: u64) -> Vec<String> {
        fields(graph)
            .filter_map(|f| match f.expect("valid") {
                (n, Value::Bytes(b)) if n == field && field == GRAPH_NODE => Some(fields(b).find_map(|f| match f.ok()? {
                    (NODE_NAME, Value::Bytes(name)) => Some(String::from_utf8_lossy(name).into_owned()),
                    _ => None,
                })?),
                (n, Value::Bytes(b)) if n == field => Some(value_info_name(b).expect("named").to_owned()),
                _ => None,
            })
            .collect()
    }

    /// Every `field` sub-message of `buf`.
    fn nested(buf: &[u8], field: u64) -> Vec<&[u8]> {
        fields(buf)
            .filter_map(|f| match f.ok()? {
                (n, Value::Bytes(b)) if n == field => Some(b),
                _ => None,
            })
            .collect()
    }

    fn dims(graph: &[u8], name: &str) -> Vec<i64> {
        for f in fields(graph) {
            if let (GRAPH_INPUT | GRAPH_OUTPUT, Value::Bytes(info)) = f.expect("valid")
                && value_info_name(info).expect("named") == name
            {
                // value info → type → tensor → shape → dims
                let tensor = nested(nested(info, VALUE_INFO_TYPE)[0], TYPE_TENSOR)[0];
                return nested(nested(tensor, TENSOR_SHAPE)[0], SHAPE_DIM)
                    .into_iter()
                    .map(|d| match fields(d).next().expect("dim").expect("valid") {
                        (DIM_VALUE, Value::Varint(v)) => v as i64,
                        _ => -1,
                    })
                    .collect();
            }
        }
        panic!("no value {name}")
    }

    /// `x → A → a → B → b` plus a side branch `a → C → c`; outputs `b` and `c`.
    fn model() -> Vec<u8> {
        let node =
            |name, input, output| encode_node(&Node { op_type: "Relu", name, inputs: &[input], outputs: &[output], attributes: &[] });
        let info = |name| encode_value_info(&TensorInfo { name, data_type: DataType::Float, dims: &[1, 3, 8, 8] });
        let mut graph = Vec::new();
        put_bytes(&mut graph, GRAPH_NODE, &node("A", "x", "a"));
        put_bytes(&mut graph, GRAPH_NODE, &node("B", "a", "b"));
        put_bytes(&mut graph, GRAPH_NODE, &node("C", "a", "c"));
        put_bytes(&mut graph, GRAPH_INPUT, &info("x"));
        put_bytes(&mut graph, GRAPH_OUTPUT, &info("b"));
        put_bytes(&mut graph, GRAPH_OUTPUT, &info("c"));
        let mut model = Vec::new();
        put_uint(&mut model, 1, 8); // ir_version
        put_bytes(&mut model, MODEL_GRAPH, &graph);
        model
    }

    #[test]
    fn patch_replaces_outputs_and_prunes_dead_nodes() {
        let patch = GraphPatch {
            inputs: &[TensorInfo { name: "x", data_type: DataType::Float, dims: &[1, 3, 4, 4] }],
            retype: &[("Relu", "Sigmoid")],
            unpad_conv_transpose: false,
            nodes: &[Node { op_type: "ArgMax", name: "D", inputs: &["b"], outputs: &["d"], attributes: &[("axis", 1), ("keepdims", 0)] }],
            outputs: &[TensorInfo { name: "d", data_type: DataType::Int64, dims: &[1, 4, 4] }],
        };
        let patched = patch_graph(&model(), &patch).expect("patch");
        let graph = super::graph(&patched).expect("graph");
        assert_eq!(names(graph, GRAPH_NODE), ["A", "B", "D"]);
        let ops: Vec<&str> = nested(graph, GRAPH_NODE).into_iter().map(|n| node_io(n).expect("node").2).collect();
        assert_eq!(ops, ["Sigmoid", "Sigmoid", "ArgMax"]);
        assert_eq!(names(graph, GRAPH_OUTPUT), ["d"]);
        assert_eq!(names(graph, GRAPH_INPUT), ["x"]);
        assert_eq!(dims(graph, "x"), [1, 3, 4, 4]);
        assert_eq!(dims(graph, "d"), [1, 4, 4]);
        // Fields outside the graph are kept.
        assert_eq!(patched[..2], [0x08, 8]);
    }

    #[test]
    fn empty_patch_keeps_the_graph_but_its_outputs() {
        let patch = GraphPatch { outputs: &[TensorInfo { name: "c", data_type: DataType::Float, dims: &[1] }], ..GraphPatch::default() };
        let patched = patch_graph(&model(), &patch).expect("patch");
        assert_eq!(names(super::graph(&patched).expect("graph"), GRAPH_NODE), ["A", "C"]);
    }

    #[test]
    fn end_padded_conv_transpose_becomes_conv_transpose_and_slice() {
        let mut pads = Vec::new();
        put_bytes(&mut pads, ATTRIBUTE_NAME, b"pads");
        let mut packed = Vec::new();
        for p in [0, 0, 1, 1] {
            put_varint(&mut packed, p);
        }
        put_bytes(&mut pads, ATTRIBUTE_INTS, &packed);
        let mut node =
            encode_node(&Node { op_type: "ConvTranspose", name: "up", inputs: &["x", "w"], outputs: &["y"], attributes: &[("group", 1)] });
        put_bytes(&mut node, NODE_ATTRIBUTE, &pads);
        let mut graph = Vec::new();
        put_bytes(&mut graph, GRAPH_NODE, &node);
        put_bytes(&mut graph, GRAPH_OUTPUT, &encode_value_info(&TensorInfo { name: "y", data_type: DataType::Float, dims: &[1] }));
        let mut model = Vec::new();
        put_bytes(&mut model, MODEL_GRAPH, &graph);

        let patched = patch_graph(&model, &GraphPatch { unpad_conv_transpose: true, ..GraphPatch::default() }).expect("patch");
        let graph = super::graph(&patched).expect("graph");
        let nodes: Vec<NodeIo<'_>> = nested(graph, GRAPH_NODE).into_iter().map(|n| node_io(n).expect("node")).collect();
        assert_eq!(nodes[0], (vec!["x", "w"], vec!["y__unpadded"], "ConvTranspose"));
        assert_eq!(nodes[1], (vec!["y__unpadded", "up__starts", "up__ends", "up__axes"], vec!["y"], "Slice"));
        let conv = nested(graph, GRAPH_NODE)[0];
        let attributes: Vec<&str> = nested(conv, NODE_ATTRIBUTE).into_iter().map(|a| attribute_name(a).expect("named")).collect();
        assert_eq!(attributes, ["group"]);
        let ends = nested(graph, GRAPH_INITIALIZER).into_iter().find(|t| nested(t, TENSOR_NAME) == [b"up__ends"]).expect("ends");
        let raw = nested(ends, TENSOR_RAW_DATA)[0];
        assert_eq!(raw.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect::<Vec<_>>(), [-1, -1]);
        assert_eq!(names(graph, GRAPH_OUTPUT), ["y"]);
    }

    #[test]
    fn truncated_input_is_an_error() {
        let model = model();
        assert!(patch_graph(&model[..model.len() - 3], &GraphPatch::default()).is_err());
    }
}
