use crate::{
    layers::LayerDims,
    tensors::{Input, Output},
};
use num_traits::{One, Zero};
use std::{
    marker::PhantomData,
    ops::{AddAssign, Mul, MulAssign},
};

use crate::Evaluate;
use NonLinearLayer::*;

#[derive(Debug, Clone)]
pub enum NonLinearLayer<F, C = F> {
    ReLU {
        dims: LayerDims,
        _c: PhantomData<C>,
        _f: PhantomData<F>,
    },
}

#[derive(Debug, Clone)]
pub enum NonLinearLayerInfo<F, C> {
    ReLU {
        _c: PhantomData<C>,
        _f: PhantomData<F>,
    },
}

impl<F, C> NonLinearLayer<F, C> {
    pub fn dimensions(&self) -> LayerDims {
        match self {
            ReLU { dims, .. } => *dims,
        }
    }

    pub fn input_dimensions(&self) -> (usize, usize, usize, usize) {
        self.dimensions().input_dimensions()
    }

    pub fn output_dimensions(&self) -> (usize, usize, usize, usize) {
        self.dimensions().input_dimensions()
    }
}
impl<F, C> Evaluate<F> for NonLinearLayer<F, C>
where
    F: One + Zero + Mul<C, Output = F> + AddAssign + MulAssign + PartialOrd<C> + Copy,
    C: Copy + From<f64> + Zero,
{
    fn evaluate_with_method(&self, _: crate::EvalMethod, input: &Input<F>) -> Output<F> {
        assert_eq!(self.input_dimensions(), input.dim());
        let mut output = Output::zeros(self.output_dimensions());
        match self {
            ReLU { .. } => {
                let zero = C::zero();
                let f_zero = F::zero();
                let f_one = F::one();
                for (&inp, out) in input.iter().zip(&mut output) {
                    *out = if inp > C::from(6.0) {
                        f_one * C::from(6.0)
                    } else if inp > zero {
                        inp
                    } else {
                        f_zero
                    }
                }
            }
        };
        output
    }
}
