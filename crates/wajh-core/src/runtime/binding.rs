//! Models with fixed input/output shapes, run through ONNX Runtime's `IoBinding` with every
//! buffer allocated once and reused for every frame.
//!
//! On a GPU backend inputs are written into page-locked (pinned) host memory and uploaded into
//! device tensors whose addresses never change; outputs are downloaded by ONNX Runtime into
//! bound host tensors. Fixed addresses are what CUDA graphs need, so the whole network can be
//! replayed with one launch instead of hundreds of kernel launches.
//!
//! Outputs are deliberately bound to ordinary (pageable) host memory: ONNX Runtime 1.28 fails
//! with `dst_data != src_data` when a CUDA session's output is bound to pinned memory. The
//! outputs are small (a face crop), so the pageable copy costs a few microseconds.

use crate::{Error, Result};
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{IoBinding, Session};
use ort::value::{Outlet, Tensor, TensorValueType};

/// Where a [`StaticModel`] keeps its tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Placement {
    /// CPU execution provider: everything lives in ordinary host memory.
    Host,
    /// CUDA / TensorRT: pinned host staging plus fixed-address device input tensors.
    Device,
}

/// One model input: its name, the host tensor callers write, and (on GPU) its device copy.
struct InputSlot {
    name: String,
    host: Tensor<f32>,
    device: Option<Tensor<f32>>,
    /// Host data changed since the last upload.
    dirty: bool,
}

/// A model whose every input and output has a static shape. Fill inputs with
/// [`StaticModel::input_mut`], call [`StaticModel::run`], read [`StaticModel::output`].
pub struct StaticModel {
    // Field order is drop order: the binding and tensors go before their allocators.
    binding: IoBinding,
    inputs: Vec<InputSlot>,
    /// Host copies of the outputs, refreshed by every run.
    outputs: Vec<Vec<f32>>,
    session: Session,
    _pinned: Option<Allocator>,
    _device: Option<Allocator>,
    name: &'static str,
}

impl StaticModel {
    /// `output_shapes` supplies the shapes of outputs whose metadata is symbolic (by position).
    pub(crate) fn new(session: Session, name: &'static str, placement: Placement, output_shapes: &[Vec<usize>]) -> Result<Self> {
        let (pinned, device) = match placement {
            Placement::Host => (None, None),
            Placement::Device => {
                let pinned = MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUInput)?;
                let device = MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?;
                (Some(Allocator::new(&session, pinned)?), Some(Allocator::new(&session, device)?))
            }
        };
        let default_allocator = Allocator::default();
        let host = pinned.as_ref().unwrap_or(&default_allocator);

        let mut binding = session.create_binding()?;
        let mut inputs = Vec::with_capacity(session.inputs().len());
        for outlet in session.inputs() {
            let dims = static_dims(outlet, name)?;
            let mut host_tensor = Tensor::<f32>::new(host, dims.as_slice())?;
            host_tensor.extract_tensor_mut().1.fill(0.0);
            let device_tensor = device.as_ref().map(|a| Tensor::<f32>::new(a, dims.as_slice())).transpose()?;
            binding.bind_input(outlet.name(), device_tensor.as_ref().unwrap_or(&host_tensor))?;
            inputs.push(InputSlot { name: outlet.name().to_owned(), host: host_tensor, device: device_tensor, dirty: true });
        }
        let mut outputs = Vec::with_capacity(session.outputs().len());
        for (index, outlet) in session.outputs().iter().enumerate() {
            let dims = match output_shapes.get(index) {
                Some(dims) => dims.clone(),
                None => static_dims(outlet, name)?,
            };
            outputs.push(vec![0.0; dims.iter().product()]);
            binding.bind_output(outlet.name(), Tensor::<f32>::new(&default_allocator, dims.as_slice())?)?;
        }
        Ok(Self { binding, inputs, outputs, session, _pinned: pinned, _device: device, name })
    }

    /// Index of the input called `name`.
    #[must_use]
    pub fn input_index(&self, name: &str) -> Option<usize> {
        self.inputs.iter().position(|s| s.name == name)
    }

    #[must_use]
    pub fn input_names(&self) -> Vec<&str> {
        self.inputs.iter().map(|s| s.name.as_str()).collect()
    }

    /// Writable host buffer of input `index`. Inputs keep their value between runs, so a
    /// constant input (like an identity embedding) only needs writing — and uploading — once.
    pub fn input_mut(&mut self, index: usize) -> &mut [f32] {
        let slot = &mut self.inputs[index];
        slot.dirty = true;
        slot.host.extract_tensor_mut().1
    }

    /// Output `index` as of the last [`StaticModel::run`].
    #[must_use]
    pub fn output(&self, index: usize) -> &[f32] {
        &self.outputs[index]
    }

    /// Upload changed inputs, run the network and collect every output.
    pub fn run(&mut self) -> Result<()> {
        for slot in &mut self.inputs {
            if let Some(device) = slot.device.as_mut()
                && slot.dirty
            {
                slot.host.copy_into(device)?;
            }
            slot.dirty = false;
        }
        let results = self.session.run_binding(&self.binding)?;
        if results.len() != self.outputs.len() {
            return Err(Error::UnexpectedOutput { model: self.name, detail: format!("{} outputs", results.len()) });
        }
        for (out, (_, value)) in self.outputs.iter_mut().zip(results.iter()) {
            out.copy_from_slice(value.downcast_ref::<TensorValueType<f32>>()?.extract_tensor().1);
        }
        Ok(())
    }
}

impl Drop for StaticModel {
    fn drop(&mut self) {
        // Flush the trace requested with `WAJH_ORT_PROFILE` (see `Engine`).
        if std::env::var_os("WAJH_ORT_PROFILE").is_some() {
            let _ = self.session.end_profiling();
        }
    }
}

/// Dimensions of a tensor input/output, which must all be fixed.
fn static_dims(outlet: &Outlet, model: &'static str) -> Result<Vec<usize>> {
    let shape = outlet
        .dtype()
        .tensor_shape()
        .ok_or_else(|| Error::UnexpectedOutput { model, detail: format!("{} is not a tensor", outlet.name()) })?;
    if shape.iter().any(|&d| d <= 0) {
        return Err(Error::DynamicShape { model, detail: format!("{} is {shape:?}", outlet.name()) });
    }
    Ok(shape.iter().map(|&d| d as usize).collect())
}
