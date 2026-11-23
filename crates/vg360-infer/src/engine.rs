//! ONNX engines. `Tract` (default) is pure Rust; `Ort` (feature `ort`) loads ONNX Runtime from
//! `ORT_DYLIB_PATH` at run time. Both take a fixed NCHW input shape and return the first output.
use std::path::Path;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;

pub trait Engine: Send + Sync {
    /// `input` is NCHW f32 of `shape`; returns the first output tensor, flattened.
    fn run(&self, input: Vec<f32>, shape: [usize; 4]) -> Result<Vec<f32>>;
    fn name(&self) -> &'static str;
}

#[cfg(feature = "tract")]
pub struct Tract { plan: tract_onnx::prelude::TypedRunnableModel<tract_onnx::prelude::TypedModel>, shape: [usize; 4] }

#[cfg(feature = "tract")]
impl Tract {
    pub fn load(path: impl AsRef<Path>, shape: [usize; 4]) -> Result<Self> {
        use tract_onnx::prelude::*;
        let plan = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(0, f32::fact(shape).into())?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { plan, shape })
    }
}

#[cfg(feature = "tract")]
impl Engine for Tract {
    fn run(&self, input: Vec<f32>, shape: [usize; 4]) -> Result<Vec<f32>> {
        use tract_onnx::prelude::*;
        if shape != self.shape { return Err(format!("model planned for {:?}, got {shape:?}", self.shape).into()); }
        let t: Tensor = tract_ndarray::Array4::from_shape_vec(shape, input)?.into();
        let out = self.plan.run(tvec!(t.into()))?;
        Ok(out[0].to_array_view::<f32>()?.iter().copied().collect())
    }
    fn name(&self) -> &'static str { "tract (pure Rust)" }
}

#[cfg(feature = "ort")]
pub struct Ort { sess: std::sync::Mutex<ort::session::Session>, input: String }

#[cfg(feature = "ort")]
impl Ort {
    /// Needs `ORT_DYLIB_PATH` pointing at libonnxruntime (e.g. the one shipped by `pip install onnxruntime`).
    pub fn load(path: impl AsRef<Path>, threads: usize) -> Result<Self> {
        static ENV: std::sync::OnceLock<std::result::Result<ort::environment::Environment, String>> = std::sync::OnceLock::new();
        let env = ENV.get_or_init(|| ort::init().with_name("vg360").build().map_err(|e| e.to_string()))
            .as_ref().map_err(|e| format!("onnxruntime: {e}"))?;
        let s = |e: ort::Error<ort::session::builder::SessionBuilder>| e.to_string();
        let sess = ort::session::Session::builder(env)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3).map_err(s)?
            .with_intra_threads(threads).map_err(s)?
            .commit_from_file(path)?;
        let input = sess.inputs()[0].name().to_string();
        Ok(Self { sess: std::sync::Mutex::new(sess), input })
    }
}

#[cfg(feature = "ort")]
impl Engine for Ort {
    fn run(&self, input: Vec<f32>, shape: [usize; 4]) -> Result<Vec<f32>> {
        let t = ort::value::Tensor::from_array((shape.map(|d| d as i64), input))?;
        let mut s = self.sess.lock().map_err(|_| "poisoned")?;
        let out = s.run(ort::inputs![self.input.as_str() => t])?;
        let (_, data) = out[0].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }
    fn name(&self) -> &'static str { "onnxruntime" }
}
