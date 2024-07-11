use candle_core::backend::BackendDevice;
use candle_core::cpu::kernels::par_range;
use candle_core::{DType, Device, MetalDevice, Result, Tensor};
use candle_nn::loss::mse;
use candle_nn::{
    batch_norm, linear, seq, Activation, AdamW, BatchNormConfig, Init, Linear, Module, Optimizer,
    ParamsAdamW, Sequential, VarBuilder, VarMap,
};
use std::time::Instant;

/// Objective function
/// high dimensional simple quadratic function
fn objective(x: &Tensor) -> f32 {
    x.mul(&x)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_vec0::<f32>()
        .unwrap()
}

pub struct Proxy {
    pub model: Sequential,
}

impl Proxy {
    pub fn new(input_dim: usize, var_builder: VarBuilder) -> Result<Self> {
        let vb = var_builder;
        let model = seq()
            .add(linear(input_dim, 64, vb.pp("proxy/linear1"))?)
            .add(Activation::LeakyRelu(0.01))
            .add(linear(64, 64, vb.pp("proxy/linear2"))?)
            .add(Activation::LeakyRelu(0.01))
            .add(linear(64, 64, vb.pp("proxy/linear3"))?)
            .add(Activation::LeakyRelu(0.01))
            .add(linear(64, 1, vb.pp("proxy/out"))?)
            .add(Activation::LeakyRelu(0.01));
        Ok(Self { model })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.model.forward(x)
    }
}

fn main() -> Result<()> {
    // Use backprop to run a linear regression between samples and get the coefficients back.
    let device = Device::Metal(MetalDevice::new(0).unwrap());
    // let device = Device::Cpu;
    let varmap = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let ndim = 100000;
    let proxy = Proxy::new(ndim, var_builder)?;

    let mut param_varmap = VarMap::new();
    param_varmap
        .get(&[ndim], "x", Init::Const(0.0), DType::F32, &device)
        .unwrap();
    param_varmap
        .set_one("x", (Tensor::ones(&[ndim], DType::F32, &device)? * 10.0)?)
        .unwrap();

    let x = param_varmap.get(&[ndim], "x", Init::Const(0.0), DType::F32, &device)?;

    let mut nn_opt_params = ParamsAdamW::default();
    nn_opt_params.lr = 5e-1;
    let mut nn_opt = AdamW::new(param_varmap.all_vars(), nn_opt_params)?;

    {
        // This is my custom code for Warmup the surrogate model
        let batchsize = 1;
        let x_batched = x.expand(&[batchsize, ndim])?;
        let xs = x_batched;
        let fs = (0..xs.shape().dims()[0])
            .map(|i| objective(&xs.get(i).unwrap()))
            .collect::<Vec<f32>>();

        let fs_tensor = Tensor::from_vec(fs.clone(), &[fs.len()], &device)?;
        for _i in 0..100 {
            let nn_pred = proxy.forward(&xs)?;
            let p_loss = mse(&nn_pred.flatten_all().unwrap(), &fs_tensor)?;
            nn_opt.backward_step(&p_loss)?;
        }
    }
    let mut nn_opt_params = ParamsAdamW::default();
    nn_opt_params.lr = 5e-1;
    let mut nn_opt = AdamW::new(param_varmap.all_vars(), nn_opt_params)?;

    let mut x_opt_params = ParamsAdamW::default();
    x_opt_params.lr = 5e-5;
    let mut x_opt = AdamW::new(dbg!(param_varmap.all_vars()), x_opt_params)?;

    proxy.forward(&x.expand(&[1, ndim]).unwrap()).unwrap();

    let sigma = 0.025_f32;
    let mut sum_time = 0.0;

    for i in 0..100_00 {
        let start = Instant::now();
        let batchsize = 2;
        let samples = Tensor::randn(0.0, sigma, &[batchsize, ndim], &device)?;

        let x_batched = x.expand(&[batchsize, ndim])?;

        let xs = Tensor::cat(&[&(&x_batched + &samples)?, &(&x_batched - &samples)?], 0)?;

        let fs = (0..xs.shape().dims()[0])
            .map(|i| objective(&xs.get(i).unwrap()))
            .collect::<Vec<f32>>();

        let fs_tensor = Tensor::from_vec(fs.clone(), &[fs.len()], &device)?;

        for _ in 0..3 {
            let xs_pert =
                (&xs - &Tensor::randn(0.0, 0.1 * sigma, &[batchsize * 2, ndim], &device)?)?;
            let nn_pred = proxy.forward(&xs_pert)?;
            let p_loss = mse(&nn_pred.flatten_all().unwrap(), &fs_tensor)?;
            nn_opt.backward_step(&p_loss)?;
        }

        let input_x = x.expand(&[1, ndim])?;
        let nn_pred = proxy.forward(&input_x)?;
        let grads = nn_pred.backward()?;

        x_opt.step(&grads)?;
        let elapsed = start.elapsed().as_micros();
        sum_time += elapsed as f64;

        if i % 100 == 0 {
            let f = objective(&x);
            // dbg!(&x.to_vec1::<f32>().unwrap());
            // dbg!(grads.get(&x).unwrap().to_vec1::<f32>().unwrap());
            dbg!(f);
            dbg!(sum_time / i as f64);
        }
    }
    //
    // dbg!(f);

    Ok(())
}
