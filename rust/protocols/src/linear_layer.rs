use crate::{AdditiveShare, AuthAdditiveShare, InMessage, OutMessage};
use algebra::{
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::Fp64Parameters,
    FpParameters, PrimeField, UniformRandom,
};
use crypto_primitives::additive_share::{AuthShare, Share};
use io_utils::imux::IMuxAsync;
use neural_network::{
    layers::*,
    tensors::{Input, Output},
    Evaluate,
};
use protocols_sys::{SealClientACG, SealServerACG, *};
use rand::{CryptoRng, RngCore};
use std::{marker::PhantomData, os::raw::c_char};

use async_std::io::{Read, Write};

pub struct LinearProtocol<P: FixedPointParameters> {
    _share: PhantomData<P>,
}

pub struct LinearProtocolType;

pub type OfflineServerMsgSend<'a> = OutMessage<'a, Vec<c_char>, LinearProtocolType>;
pub type OfflineServerMsgRcv = InMessage<Vec<c_char>, LinearProtocolType>;

pub type OfflineClientMsgSend<'a> = OutMessage<'a, Vec<c_char>, LinearProtocolType>;
pub type OfflineClientMsgRcv = InMessage<Vec<c_char>, LinearProtocolType>;

pub type MsgSend<'a, P> = OutMessage<'a, Input<AdditiveShare<P>>, LinearProtocolType>;
pub type MsgRcv<P> = InMessage<Input<AdditiveShare<P>>, LinearProtocolType>;

impl<P: FixedPointParameters> LinearProtocol<P>
where
    P: FixedPointParameters,
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
    P::Field: AuthShare,
{
    /// Runs server ACG protocol. Receives client input `r`, and homomorphically
    /// evaluates `Lr`, returning authenticated shares of `r`, shares of `Lr`,
    /// and authenticated shares of shares of `Lr` --> [[r]]_2, <Lr>_2,
    /// [[<Lr>_2]]_2
    pub fn offline_server_acg_protocol<
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: RngCore + CryptoRng,
    >(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        server_acg: &mut SealServerACG,
        rng: &mut RNG,
    ) -> Result<
        (
            (
                Input<AuthAdditiveShare<P::Field>>,
                Output<P::Field>,
                Output<AuthAdditiveShare<P::Field>>,
            ),
            (P::Field, P::Field),
        ),
        bincode::Error,
    > {
        // TODO: Add batch size
        let start_time = timer_start!(|| "Server linear offline protocol");
        let preprocess_time = timer_start!(|| "Preprocessing");

        // Sample MAC keys
        let mac_key_r = P::Field::uniform(rng);
        let mac_key_y = P::Field::uniform(rng);

        // Sample server's randomness for randomizing the i-th
        // layer MAC share and the i+1-th layer/MAC shares
        let mut linear_share = Output::zeros(output_dims);
        let mut linear_mac_share = Output::zeros(output_dims);
        let mut r_mac_share = Input::zeros(input_dims);

        linear_share
            .iter_mut()
            .for_each(|e| *e = P::Field::uniform(rng));
        linear_mac_share
            .iter_mut()
            .for_each(|e| *e = P::Field::uniform(rng));
        r_mac_share
            .iter_mut()
            .for_each(|e| *e = P::Field::uniform(rng));

        // Create SEALServer object for C++ interopt
        // Preprocess filter rotations and noise masks
        server_acg.preprocess(
            &linear_share.to_u64(),
            &linear_mac_share.to_u64(),
            &r_mac_share.to_u64(),
            mac_key_y.into_repr().0,
            mac_key_r.into_repr().0,
        );
        timer_end!(preprocess_time);

        // Receive client Enc(r_i)
        let rcv_time = timer_start!(|| "Receiving Input");
        let client_share: OfflineServerMsgRcv = crate::bytes::deserialize(reader)?;
        let client_share_i = client_share.msg();
        timer_end!(rcv_time);

        // Compute client's MAC share for layer `i`, and share + MAC share for layer `i
        // + 1`, That is, compute Lr - s, [a(Lr-s)]_1, [ar]_1
        let processing = timer_start!(|| "Processing Layer");
        let (linear_ct_vec, linear_mac_ct_vec, r_mac_ct_vec) = server_acg.process(client_share_i);
        timer_end!(processing);

        // Send shares to client
        let send_time = timer_start!(|| "Sending result");
        let sent_message = OfflineServerMsgSend::new(&linear_ct_vec);
        crate::bytes::serialize(&mut *writer, &sent_message)?;
        let sent_message = OfflineServerMsgSend::new(&linear_mac_ct_vec);
        crate::bytes::serialize(&mut *writer, &sent_message)?;
        let sent_message = OfflineServerMsgSend::new(&r_mac_ct_vec);
        crate::bytes::serialize(writer, &sent_message)?;
        timer_end!(send_time);

        // Collect shares and MACs into AuthAdditiveShares
        let linear_auth =
            Output::auth_share_from_parts(Output::zeros(output_dims), linear_mac_share);
        // Negate `r_mac_share` since client locally negates to get correct online share
        let r_auth = Input::auth_share_from_parts(Input::zeros(input_dims), -r_mac_share);

        timer_end!(start_time);
        Ok(((r_auth, linear_share, linear_auth), (mac_key_r, mac_key_y)))
    }

    /// Runs client ACG protocol. Generates random input `r` and receives back
    /// authenticated shares of `r`, shares of `Lr`, and authenticated shares of
    /// shares of `Lr` --> [[r]]_1, <Lr>_1, [[<Lr>_1]]_1
    pub fn offline_client_acg_protocol<
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: RngCore + CryptoRng,
    >(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        client_acg: &mut SealClientACG,
        rng: &mut RNG,
    ) -> Result<
        (
            Input<AuthAdditiveShare<P::Field>>,
            Output<AuthAdditiveShare<P::Field>>,
        ),
        bincode::Error,
    > {
        // TODO: Add batch size
        let start_time = timer_start!(|| "Linear offline protocol");
        let preprocess_time = timer_start!(|| "Client preprocessing");
        // Generate random share r
        let mut r: Input<FixedPoint<P>> = Input::zeros(input_dims);
        r.iter_mut()
            .for_each(|e| *e = FixedPoint::new(P::Field::uniform(&mut *rng)));

        // Create SEALClient object for C++ interopt
        // Preprocess and encrypt client secret share for sending
        let ct_vec = client_acg.preprocess(&r.to_repr());
        timer_end!(preprocess_time);

        // Send layer_i randomness for processing by server.
        let send_time = timer_start!(|| "Sending input");
        let sent_message = OfflineClientMsgSend::new(&ct_vec);
        crate::bytes::serialize(writer, &sent_message)?;
        timer_end!(send_time);

        // Receive shares for layer `i + 1` output, MAC, and layer `i` MAC
        let rcv_time = timer_start!(|| "Receiving Result");
        let linear_ct: OfflineClientMsgRcv = crate::bytes::deserialize(&mut *reader)?;
        let linear_mac_ct: OfflineClientMsgRcv = crate::bytes::deserialize(&mut *reader)?;
        let r_mac_ct: OfflineClientMsgRcv = crate::bytes::deserialize(reader)?;
        timer_end!(rcv_time);

        let post_time = timer_start!(|| "Post-processing");
        let mut linear_auth = Output::zeros(output_dims);
        let mut r_mac = Input::zeros(input_dims);
        // Decrypt + reshape resulting ciphertext and free C++ allocations
        client_acg.decrypt(linear_ct.msg(), linear_mac_ct.msg(), r_mac_ct.msg());
        client_acg.postprocess::<P>(&mut linear_auth, &mut r_mac);

        // Negate both shares here so that we receive the correct
        // labels for the online phase
        let r_auth = Input::auth_share_from_parts(-r.to_base(), -r_mac);

        timer_end!(post_time);
        timer_end!(start_time);

        Ok((r_auth, linear_auth))
    }

    /// Client sends a value to the server and receives back a share of it's
    /// MAC'd value
    pub fn offline_client_auth_share<R: Read + Send + Unpin, W: Write + Send + Unpin>(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        input: Input<P::Field>,
        cfhe: &ClientFHE,
    ) -> Result<Input<AuthAdditiveShare<P::Field>>, bincode::Error> {
        let start_time = timer_start!(|| "Linear offline protocol");

        // Encrypt input and send to the server
        let mut share = SealCT::new();
        let ct = share.encrypt_vec(cfhe, input.to_u64().as_slice().unwrap().to_vec());

        let send_time = timer_start!(|| "Sending input");
        let sent_message = OfflineClientMsgSend::new(&ct);
        crate::bytes::serialize(writer, &sent_message)?;
        timer_end!(send_time);

        // Receive the result and decrypt
        let rcv_time = timer_start!(|| "Receiving Result");
        let auth_ct: OfflineClientMsgRcv = crate::bytes::deserialize(&mut *reader)?;
        timer_end!(rcv_time);

        let result = share
            .decrypt_vec(cfhe, auth_ct.msg(), input.len())
            .iter()
            .map(|e| P::Field::from_repr((*e).into()))
            .collect();
        let input_mac = Input::from_shape_vec(input.dim(), result).expect("Shapes should be same");
        timer_end!(start_time);
        Ok(Input::auth_share_from_parts(input, input_mac))
    }

    /// Server receives an encrypted vector from the client and shares its MAC
    pub fn offline_server_auth_share<
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: RngCore + CryptoRng,
    >(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        input_dims: (usize, usize, usize, usize),
        sfhe: &ServerFHE,
        rng: &mut RNG,
    ) -> Result<(P::Field, Input<AuthAdditiveShare<P::Field>>), bincode::Error> {
        let start_time = timer_start!(|| "Linear offline protocol");

        // Sample MAC key and MAC share
        let mac_key = P::Field::uniform(rng);
        let mut mac_share = Input::zeros(input_dims);
        mac_share
            .iter_mut()
            .for_each(|e| *e = P::Field::uniform(rng));
        let mac_share_c: Vec<u64> = mac_share.to_u64().as_slice().unwrap().to_vec();

        // Receive client input and compute MAC share
        let mut share = SealCT::new();

        let rcv_time = timer_start!(|| "Receiving Input");
        let input: OfflineServerMsgRcv = crate::bytes::deserialize(&mut *reader)?;
        let mut input_ct = input.msg();
        timer_end!(rcv_time);

        share.inner.inner = input_ct.as_mut_ptr();
        share.inner.size = input_ct.len() as u64;
        let result_ct = share.gen_mac_share(sfhe, mac_share_c, mac_key.into_repr().0);

        // Send result back to client
        let send_time = timer_start!(|| "Sending Result");
        let sent_message = OfflineClientMsgSend::new(&result_ct);
        crate::bytes::serialize(writer, &sent_message)?;
        timer_end!(send_time);
        timer_end!(start_time);
        Ok((
            mac_key,
            Input::auth_share_from_parts(Input::zeros(input_dims), mac_share),
        ))
    }

    pub fn online_client_protocol<W: Write + Send + Unpin>(
        writer: &mut IMuxAsync<W>,
        x_s: &Input<AdditiveShare<P>>,
        layer: &LinearLayerInfo<AdditiveShare<P>, FixedPoint<P>>,
    ) -> Result<(), bincode::Error> {
        let start = timer_start!(|| "Linear online protocol");
        match layer {
            LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                let sent_message = MsgSend::new(x_s);
                crate::bytes::serialize(&mut *writer, &sent_message)?;
            }
            _ => {}
        }
        timer_end!(start);
        Ok(())
    }

    pub fn online_server_protocol<R: Read + Send + Unpin>(
        reader: &mut IMuxAsync<R>,
        layer: &LinearLayer<AdditiveShare<P>, FixedPoint<P>>,
        output_rerandomizer: &Output<P::Field>,
        input_derandomizer: &Input<P::Field>,
        output: &mut Output<AdditiveShare<P>>,
    ) -> Result<(), bincode::Error> {
        let start = timer_start!(|| "Linear online protocol");
        let mut input: Input<AdditiveShare<P>> = match &layer {
            LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                let recv: MsgRcv<P> = crate::bytes::deserialize(reader).unwrap();
                recv.msg()
            }
            _ => Input::zeros(input_derandomizer.dim()),
        };
        input.randomize_local_share(input_derandomizer);
        *output = layer.evaluate(&input);
        output.zip_mut_with(output_rerandomizer, |out, s| {
            *out = FixedPoint::randomize_local_share(out, s)
        });
        timer_end!(start);
        Ok(())
    }
}
