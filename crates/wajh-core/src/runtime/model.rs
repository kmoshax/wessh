use crate::{Error, Result};
use ort::session::{Session, SessionInputValue};
use ort::value::TensorRef;
use std::borrow::Cow;

/// A named `f32` input tensor, borrowed from the caller so buffers can be reused across frames.
#[derive(Debug)]
pub struct Input<'a> {
    pub name: &'a str,
    pub shape: Vec<i64>,
    pub data: &'a [f32],
}

impl<'a> Input<'a> {
    /// `1×3×size×size` image tensor.
    #[must_use]
    pub fn image(name: &'a str, size: usize, data: &'a [f32]) -> Self {
        Self { name, shape: vec![1, 3, size as i64, size as i64], data }
    }

    /// `1×N` vector tensor.
    #[must_use]
    pub fn vector(name: &'a str, data: &'a [f32]) -> Self {
        Self { name, shape: vec![1, data.len() as i64], data }
    }
}

/// A loaded ONNX model with `f32` inputs and outputs of any shape (see
/// [`StaticModel`](super::StaticModel) for fixed-shape models on the hot path).
pub struct Model {
    session: Session,
    name: &'static str,
}

impl Model {
    pub(crate) const fn new(session: Session, name: &'static str) -> Self {
        Self { session, name }
    }

    #[must_use]
    pub fn input_names(&self) -> Vec<&str> {
        self.session.inputs().iter().map(ort::value::Outlet::name).collect()
    }

    /// Run the model and copy every output out as a flat `Vec<f32>`.
    pub fn run(&mut self, inputs: &[Input<'_>]) -> Result<Vec<Vec<f32>>> {
        let values = inputs
            .iter()
            .map(|input| {
                let tensor = TensorRef::from_array_view((input.shape.as_slice(), input.data))?;
                Ok((Cow::Borrowed(input.name), SessionInputValue::from(tensor)))
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = self.session.run(values)?;
        let tensors = outputs.iter().map(|(_, value)| Ok(value.try_extract_tensor::<f32>()?.1.to_vec())).collect::<Result<Vec<_>>>()?;
        if tensors.is_empty() {
            return Err(Error::UnexpectedOutput { model: self.name, detail: "no outputs".into() });
        }
        Ok(tensors)
    }

    /// Run and return the first output.
    pub fn run_single(&mut self, inputs: &[Input<'_>]) -> Result<Vec<f32>> {
        Ok(self.run(inputs)?.swap_remove(0))
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // Flush the trace requested with `WAJH_ORT_PROFILE` (see `Engine`).
        if std::env::var_os("WAJH_ORT_PROFILE").is_some() {
            let _ = self.session.end_profiling();
        }
    }
}
