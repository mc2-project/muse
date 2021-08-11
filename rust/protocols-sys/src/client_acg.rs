use crate::*;
use algebra::{fixed_point::*, fp_64::Fp64Parameters, FpParameters, PrimeField};
use crypto_primitives::additive_share::{AuthAdditiveShare, AuthShare};
use neural_network::{
    layers::{convolution::Padding, LinearLayerInfo},
    tensors::{Input, Output},
};
use std::os::raw::c_char;

pub struct Conv2D<'a> {
    data: Metadata,
    cfhe: &'a ClientFHE,
    shares: Option<ClientShares>,
}

pub struct FullyConnected<'a> {
    data: Metadata,
    cfhe: &'a ClientFHE,
    shares: Option<ClientShares>,
}

pub enum SealClientACG<'a> {
    Conv2D(Conv2D<'a>),
    FullyConnected(FullyConnected<'a>),
}

pub trait ClientACG {
    type Keys;

    fn new<F, C>(
        cfhe: Self::Keys,
        layer_info: &LinearLayerInfo<F, C>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
    ) -> Self
    where
        Self: std::marker::Sized;

    fn preprocess(&mut self, r: &Input<u64>) -> Vec<c_char>;

    fn decrypt(
        &mut self,
        linear_ct: Vec<c_char>,
        linear_mac_ct: Vec<c_char>,
        r_mac_ct: Vec<c_char>,
    );

    fn postprocess<P>(
        &self,
        linear_auth: &mut Output<AuthAdditiveShare<P::Field>>,
        r_mac_share: &mut Input<P::Field>,
    ) where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
        P::Field: AuthShare;
}

impl<'a> SealClientACG<'a> {
    pub fn preprocess(&mut self, r: &Input<u64>) -> Vec<c_char> {
        match self {
            Self::Conv2D(s) => s.preprocess(r),
            Self::FullyConnected(s) => s.preprocess(r),
        }
    }

    pub fn decrypt(
        &mut self,
        linear_ct: Vec<c_char>,
        linear_mac_ct: Vec<c_char>,
        r_mac_ct: Vec<c_char>,
    ) {
        match self {
            Self::Conv2D(s) => s.decrypt(linear_ct, linear_mac_ct, r_mac_ct),
            Self::FullyConnected(s) => s.decrypt(linear_ct, linear_mac_ct, r_mac_ct),
        };
    }

    pub fn postprocess<P>(
        &self,
        linear_auth: &mut Output<AuthAdditiveShare<P::Field>>,
        r_mac_share: &mut Input<P::Field>,
    ) where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
        P::Field: AuthShare,
    {
        match self {
            Self::Conv2D(s) => ClientACG::postprocess::<P>(s, linear_auth, r_mac_share),
            Self::FullyConnected(s) => ClientACG::postprocess::<P>(s, linear_auth, r_mac_share),
        };
    }
}

impl<'a> ClientACG for Conv2D<'a> {
    type Keys = &'a ClientFHE;

    fn new<F, C>(
        cfhe: &'a ClientFHE,
        layer_info: &LinearLayerInfo<F, C>,
        input_dims: (usize, usize, usize, usize),
        _output_dims: (usize, usize, usize, usize),
    ) -> Self {
        let (kernel, padding, stride) = match layer_info {
            LinearLayerInfo::Conv2d {
                kernel,
                padding,
                stride,
            } => (kernel, padding, stride),
            _ => panic!("Incorrect Layer Type"),
        };
        let data = unsafe {
            conv_metadata(
                cfhe.encoder,
                input_dims.2 as i32,
                input_dims.3 as i32,
                kernel.2 as i32,
                kernel.3 as i32,
                kernel.1 as i32,
                kernel.0 as i32,
                *stride as i32,
                *stride as i32,
                *padding == Padding::Valid,
            )
        };
        Self {
            data,
            cfhe,
            shares: None,
        }
    }

    fn preprocess(&mut self, r: &Input<u64>) -> Vec<c_char> {
        // Convert client secret share to raw pointers for C FFI
        let r_c: Vec<*const u64> = (0..self.data.inp_chans)
            .into_iter()
            .map(|inp_c| {
                r.slice(s![0, inp_c, .., ..])
                    .as_slice()
                    .expect("Error converting client share")
                    .as_ptr()
            })
            .collect();
        let shares = unsafe { client_conv_preprocess(self.cfhe, &self.data, r_c.as_ptr()) };
        let ct_vec = unsafe {
            std::slice::from_raw_parts(shares.input_ct.inner, shares.input_ct.size as usize)
                .to_vec()
        };
        self.shares = Some(shares);
        ct_vec
    }

    fn decrypt(
        &mut self,
        mut linear_ct: Vec<c_char>,
        mut linear_mac_ct: Vec<c_char>,
        mut r_mac_ct: Vec<c_char>,
    ) {
        let mut shares = self.shares.unwrap();
        // Copy the received ciphertexts into share struct
        shares.linear_ct = SerialCT {
            inner: linear_ct.as_mut_ptr(),
            size: linear_ct.len() as u64,
        };
        shares.linear_mac_ct = SerialCT {
            inner: linear_mac_ct.as_mut_ptr(),
            size: linear_mac_ct.len() as u64,
        };
        shares.r_mac_ct = SerialCT {
            inner: r_mac_ct.as_mut_ptr(),
            size: r_mac_ct.len() as u64,
        };
        // Decrypt everything
        unsafe { client_conv_decrypt(self.cfhe, &self.data, &mut shares) };
        self.shares = Some(shares);
    }

    fn postprocess<P>(
        &self,
        linear_auth: &mut Output<AuthAdditiveShare<P::Field>>,
        r_mac_share: &mut Input<P::Field>,
    ) where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
        P::Field: AuthShare,
    {
        let shares = self.shares.unwrap();
        for chan in 0..self.data.out_chans as usize {
            for row in 0..self.data.output_h as usize {
                for col in 0..self.data.output_w as usize {
                    let idx = (row * (self.data.output_w as usize) + col) as isize;
                    let linear_val =
                        unsafe { *(*(shares.linear.offset(chan as isize))).offset(idx as isize) };
                    let linear_mac_val = unsafe {
                        *(*(shares.linear_mac.offset(chan as isize))).offset(idx as isize)
                    };
                    linear_auth[[0, chan, row, col]] = AuthAdditiveShare::new(
                        P::Field::from_repr(linear_val.into()),
                        P::Field::from_repr(linear_mac_val.into()),
                    );
                }
            }
        }
        for chan in 0..self.data.inp_chans as usize {
            for row in 0..self.data.image_h as usize {
                for col in 0..self.data.image_w as usize {
                    let idx = (row * (self.data.image_w as usize) + col) as isize;
                    let r_mac_val =
                        unsafe { *(*(shares.r_mac.offset(chan as isize))).offset(idx as isize) };
                    r_mac_share[[0, chan, row, col]] = P::Field::from_repr(r_mac_val.into());
                }
            }
        }
    }
}

impl<'a> ClientACG for FullyConnected<'a> {
    type Keys = &'a ClientFHE;

    fn new<F, C>(
        cfhe: &'a ClientFHE,
        _layer_info: &LinearLayerInfo<F, C>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
    ) -> Self {
        let data = unsafe {
            fc_metadata(
                cfhe.encoder,
                (input_dims.1 * input_dims.2 * input_dims.3) as i32,
                output_dims.1 as i32,
            )
        };
        Self {
            data,
            cfhe,
            shares: None,
        }
    }

    fn preprocess(&mut self, r: &Input<u64>) -> Vec<c_char> {
        // Convert client secret share to raw pointers for C FFI
        let r_c: *const u64 = r
            .slice(s![0, .., .., ..])
            .as_slice()
            .expect("Error converting client share")
            .as_ptr();
        let shares = unsafe { client_fc_preprocess(self.cfhe, &self.data, r_c) };
        let ct_vec = unsafe {
            std::slice::from_raw_parts(shares.input_ct.inner, shares.input_ct.size as usize)
                .to_vec()
        };
        self.shares = Some(shares);
        ct_vec
    }

    fn decrypt(
        &mut self,
        mut linear_ct: Vec<c_char>,
        mut linear_mac_ct: Vec<c_char>,
        mut r_mac_ct: Vec<c_char>,
    ) {
        let mut shares = self.shares.unwrap();
        // Copy the received ciphertexts into share struct
        shares.linear_ct = SerialCT {
            inner: linear_ct.as_mut_ptr(),
            size: linear_ct.len() as u64,
        };
        shares.linear_mac_ct = SerialCT {
            inner: linear_mac_ct.as_mut_ptr(),
            size: linear_mac_ct.len() as u64,
        };
        shares.r_mac_ct = SerialCT {
            inner: r_mac_ct.as_mut_ptr(),
            size: r_mac_ct.len() as u64,
        };
        // Decrypt everything
        unsafe { client_fc_decrypt(self.cfhe, &self.data, &mut shares) };
        self.shares = Some(shares);
    }

    fn postprocess<P>(
        &self,
        linear_auth: &mut Output<AuthAdditiveShare<P::Field>>,
        r_mac_share: &mut Input<P::Field>,
    ) where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
        P::Field: AuthShare,
    {
        let shares = self.shares.unwrap();
        for row in 0..self.data.filter_h as usize {
            let linear_val = unsafe { *(*(shares.linear.offset(0))).offset(row as isize) };
            let linear_mac_val = unsafe { *(*(shares.linear_mac.offset(0))).offset(row as isize) };
            linear_auth[[0, row, 0, 0]] = AuthAdditiveShare::new(
                P::Field::from_repr(linear_val.into()),
                P::Field::from_repr(linear_mac_val.into()),
            );
        }
        r_mac_share.iter_mut().enumerate().for_each(|(col, e)| {
            let r_mac_val = unsafe { *(*(shares.r_mac.offset(0))).offset(col as isize) };
            *e = P::Field::from_repr(r_mac_val.into());
        });
    }
}

impl<'a> Drop for Conv2D<'a> {
    fn drop(&mut self) {
        unsafe { client_conv_free(&self.data, &mut self.shares.unwrap()) }
    }
}

impl<'a> Drop for FullyConnected<'a> {
    fn drop(&mut self) {
        unsafe { client_fc_free(&mut self.shares.unwrap()) };
    }
}
