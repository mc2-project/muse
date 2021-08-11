use crate::{bytes, cds, error::MpcError, AdditiveShare, InMessage, OutMessage};
use algebra::{
    fields::PrimeField,
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::{Fp64, Fp64Parameters},
    BigInteger64, FpParameters, UniformRandom,
};
use crypto_primitives::{
    gc::{
        fancy_garbling,
        fancy_garbling::{
            circuit::{Circuit, CircuitBuilder},
            Encoder, GarbledCircuit, Wire,
        },
    },
    AuthShare, Share,
};
use io_utils::imux::IMuxAsync;
use itertools::interleave;
use protocols_sys::{ClientFHE, ServerFHE};
use rand::{CryptoRng, RngCore};
use rayon::prelude::*;
use scuttlebutt::Block;
use std::marker::PhantomData;

use async_std::io::{Read, Write};

#[derive(Default)]
pub struct ReluProtocol<P: FixedPointParameters> {
    _share: PhantomData<P>,
}

pub struct ReluProtocolType;

pub type ServerGcMsgSend<'a> = OutMessage<'a, (&'a [GarbledCircuit], &'a [Wire]), ReluProtocolType>;
pub type ClientGcMsgRcv = InMessage<(Vec<GarbledCircuit>, Vec<Wire>), ReluProtocolType>;

// The message is a slice of (vectors of) input labels;
pub type ServerLabelMsgSend<'a> = OutMessage<'a, [Vec<Wire>], ReluProtocolType>;
pub type ClientLabelMsgRcv = InMessage<Vec<Vec<Wire>>, ReluProtocolType>;

pub type ClientShareMsgSend<'a, P> = OutMessage<'a, [AdditiveShare<P>], ReluProtocolType>;
pub type ServerShareMsgRcv<P> = InMessage<Vec<AdditiveShare<P>>, ReluProtocolType>;

fn make_relu<P: FixedPointParameters>() -> Circuit
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    let mut b = CircuitBuilder::new();
    crypto_primitives::gc::relu::<P>(&mut b, 1).unwrap();
    b.finish()
}

pub fn make_truncated_relu<P: FixedPointParameters>(trunc_bits: u8) -> Circuit
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    let mut b = CircuitBuilder::new();
    crypto_primitives::gc::truncated_relu::<P>(&mut b, 1, trunc_bits).unwrap();
    b.finish()
}

fn u128_from_share<P: FixedPointParameters>(s: AdditiveShare<P>) -> u128
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = BigInteger64>,
{
    let s: u64 = s.inner.inner.into_repr().into();
    s.into()
}

pub struct ServerState<P: FixedPointParameters> {
    pub encoders: Vec<Encoder>,
    pub output_randomizers: Vec<P::Field>,
}

pub struct ClientState {
    pub gc_s: Vec<GarbledCircuit>,
    pub server_randomizer_labels: Vec<Wire>,
    pub client_input_labels: Vec<Wire>,
}

impl<P: FixedPointParameters, F: Fp64Parameters> ReluProtocol<P>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P: FixedPointParameters<Field = Fp64<F>>,
    P::Field: Share<
        Constant = <P as FixedPointParameters>::Field,
        Ring = <P as FixedPointParameters>::Field,
    >,
    P::Field: AuthShare,
{
    #[inline]
    pub fn size_of_client_inputs() -> usize {
        make_relu::<P>().num_evaluator_inputs()
    }

    pub fn offline_server_protocol<
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: CryptoRng + RngCore,
    >(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        number_of_relus: usize,
        sfhe: &ServerFHE,
        layer_sizes: &[usize],
        output_mac_keys: &[P::Field],
        output_mac_shares: &[P::Field],
        output_truncations: &[u8],
        input_mac_keys: &[P::Field],
        input_mac_shares: &[P::Field],
        rng: &mut RNG,
    ) -> Result<ServerState<P>, MpcError> {
        let start_time = timer_start!(|| "ReLU offline protocol");

        let mut gc_s = Vec::with_capacity(number_of_relus);
        let mut encoders = Vec::with_capacity(number_of_relus);
        let p = (<<P::Field as PrimeField>::Params>::MODULUS.0).into();

        // let c = make_relu::<P>();
        assert_eq!(
            number_of_relus,
            layer_sizes.iter().fold(0, |sum, &x| sum + x)
        );
        let garble_time = timer_start!(|| "Garbling");

        // For each layer, garbled a circuit with the correct number of truncations
        for (i, num) in layer_sizes.iter().enumerate() {
            let c = make_truncated_relu::<P>(P::EXPONENT_CAPACITY * output_truncations[i]);
            let (en, gc): (Vec<Encoder>, Vec<GarbledCircuit>) = (0..*num)
                .into_par_iter()
                .map(move |_| {
                    let mut c = c.clone();
                    let (en, gc) = fancy_garbling::garble(&mut c).unwrap();
                    (en, gc)
                })
                .unzip();
            encoders.extend(en);
            gc_s.extend(gc);
        }
        timer_end!(garble_time);

        let encode_time = timer_start!(|| "Encoding inputs");
        let (num_garbler_inputs, num_evaluator_inputs) = if number_of_relus > 0 {
            (
                encoders[0].num_garbler_inputs(),
                encoders[0].num_evaluator_inputs(),
            )
        } else {
            (0, 0)
        };

        let zero_inputs = vec![0u16; num_evaluator_inputs];
        let one_inputs = vec![1u16; num_evaluator_inputs];
        let mut labels = Vec::with_capacity(number_of_relus * num_evaluator_inputs);
        let mut randomizer_labels = Vec::with_capacity(number_of_relus);
        let mut output_randomizers = Vec::with_capacity(number_of_relus);
        for enc in encoders.iter() {
            // Output server randomization share
            let r = P::Field::uniform(rng);
            output_randomizers.push(r);
            let r_bits: u64 = ((-r).into_repr()).into();
            let r_bits = fancy_garbling::util::u128_to_bits(
                r_bits.into(),
                crypto_primitives::gc::num_bits(p),
            );
            for w in ((num_garbler_inputs / 2)..num_garbler_inputs)
                .zip(r_bits)
                .map(|(i, r_i)| enc.encode_garbler_input(r_i, i))
            {
                randomizer_labels.push(w);
            }

            let all_zeros = enc.encode_evaluator_inputs(&zero_inputs);
            let all_ones = enc.encode_evaluator_inputs(&one_inputs);
            all_zeros
                .into_iter()
                .zip(all_ones)
                .for_each(|(label_0, label_1)| {
                    labels.push((label_0.as_block(), label_1.as_block()))
                });
        }

        // Extract out the zero labels for the carry bits since these aren't
        // used in CDS
        let (carry_labels, input_labels): (Vec<_>, Vec<_>) = labels
            .into_iter()
            .enumerate()
            .partition(|(i, _)| (i + 1) % (P::Field::size_in_bits() + 1) == 0);

        let carry_labels: Vec<Wire> = carry_labels
            .into_iter()
            .map(|(_, (zero, _))| Wire::from_block(zero, 2))
            .collect();
        let input_labels: Vec<(Block, Block)> = input_labels.into_iter().map(|(_, l)| l).collect();
        timer_end!(encode_time);

        let send_gc_time = timer_start!(|| "Sending GCs");
        let randomizer_label_per_relu = if number_of_relus == 0 {
            8192
        } else {
            randomizer_labels.len() / number_of_relus
        };
        for msg_contents in gc_s
            .chunks(8192)
            .zip(randomizer_labels.chunks(randomizer_label_per_relu * 8192))
        {
            let sent_message = ServerGcMsgSend::new(&msg_contents);
            bytes::serialize(writer, &sent_message)?;
        }
        timer_end!(send_gc_time);

        let cds_time = timer_start!(|| "CDS Protocol");
        if number_of_relus > 0 {
            cds::CDSProtocol::<P>::server_cds(
                reader,
                writer,
                sfhe,
                layer_sizes,
                output_mac_keys,
                output_mac_shares,
                input_mac_keys,
                input_mac_shares,
                input_labels.as_slice(),
                rng,
            )?;
        }
        timer_end!(cds_time);

        // Send carry labels to client
        let send_time = timer_start!(|| "Sending carry labels");
        let tmp = vec![carry_labels];
        let send_message = ServerLabelMsgSend::new(&tmp);
        bytes::serialize(writer, &send_message)?;

        timer_end!(send_time);
        timer_end!(start_time);
        Ok(ServerState {
            encoders,
            output_randomizers,
        })
    }

    pub fn offline_client_protocol<
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: CryptoRng + RngCore,
    >(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        number_of_relus: usize,
        cfhe: &ClientFHE,
        layer_sizes: &[usize],
        output_mac_shares: &[P::Field],
        output_shares: &[P::Field],
        input_mac_shares: &[P::Field],
        input_rands: &[P::Field],
        rng: &mut RNG,
    ) -> Result<ClientState, MpcError> {
        let start_time = timer_start!(|| "ReLU offline protocol");
        let rcv_gc_time = timer_start!(|| "Receiving GCs");
        let mut gc_s = Vec::with_capacity(number_of_relus);
        let mut r_wires = Vec::with_capacity(number_of_relus);

        let num_chunks = (number_of_relus as f64 / 8192.0).ceil() as usize;
        for i in 0..num_chunks {
            let in_msg: ClientGcMsgRcv = bytes::deserialize(reader)?;
            let (gc_chunks, r_wire_chunks) = in_msg.msg();
            if i < (num_chunks - 1) {
                assert_eq!(gc_chunks.len(), 8192);
            }
            gc_s.extend(gc_chunks);
            r_wires.extend(r_wire_chunks);
        }
        timer_end!(rcv_gc_time);

        assert_eq!(gc_s.len(), number_of_relus);

        let cds_time = timer_start!(|| "CDS Protocol");
        let labels = if number_of_relus > 0 {
            cds::CDSProtocol::<P>::client_cds(
                reader,
                writer,
                cfhe,
                layer_sizes,
                output_mac_shares,
                output_shares,
                input_mac_shares,
                input_rands,
                rng,
            )?
        } else {
            Vec::new()
        };
        timer_end!(cds_time);

        // Receive carry labels
        let recv_time = timer_start!(|| "Receiving carry labels");

        let recv_msg: ClientLabelMsgRcv = bytes::deserialize(reader)?;
        let carry_labels: Vec<Wire> = recv_msg.msg().remove(0);

        // Interleave received labels with carry labels
        let labels = interleave(
            labels.chunks(P::Field::size_in_bits()),
            carry_labels.chunks(1),
        )
        .flatten()
        .cloned()
        .collect();
        timer_end!(recv_time);
        timer_end!(start_time);

        Ok(ClientState {
            gc_s,
            server_randomizer_labels: r_wires,
            client_input_labels: labels,
        })
    }

    pub fn online_server_protocol<'a, R: Read + Send + Unpin, W: Write + Send + Unpin>(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        shares: &[AdditiveShare<P>],
        encoders: &[Encoder],
    ) -> Result<Vec<AdditiveShare<P>>, MpcError> {
        let p = u128::from(u64::from(P::Field::characteristic()));
        let start_time = timer_start!(|| "ReLU online protocol");
        let encoding_time = timer_start!(|| "Encoding inputs");

        let field_size = (p.next_power_of_two() * 2).trailing_zeros() as usize;
        let wires = shares
            .iter()
            .map(|share| {
                let share = u128_from_share(*share);
                fancy_garbling::util::u128_to_bits(share, field_size)
            })
            .zip(encoders)
            .map(|(share_bits, encoder)| encoder.encode_garbler_inputs(&share_bits))
            .collect::<Vec<Vec<_>>>();
        timer_end!(encoding_time);

        let send_time = timer_start!(|| "Sending inputs");
        let sent_message = ServerLabelMsgSend::new(wires.as_slice());
        timer_end!(send_time);
        bytes::serialize(&mut *writer, &sent_message)?;

        let rcv_time = timer_start!(|| "Receiving shares");
        let _: ClientLabelMsgRcv = bytes::deserialize(&mut *reader)?;
        let in_msg: ServerShareMsgRcv<P> = bytes::deserialize(reader)?;
        timer_end!(rcv_time);
        timer_end!(start_time);
        Ok(in_msg.msg())
    }

    /// Outputs shares for the next round's input.
    pub fn online_client_protocol<R: Read + Send + Unpin, W: Write + Send + Unpin>(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        num_relus: usize,
        num_trunc: u8,
        server_input_wires: &[Wire],
        client_input_wires: &[Wire],
        evaluators: &[GarbledCircuit],
    ) -> Result<(), MpcError> {
        let start_time = timer_start!(|| "ReLU online protocol");

        let rcv_time = timer_start!(|| "Receiving inputs");
        let in_msg: ClientLabelMsgRcv = bytes::deserialize(reader)?;
        let mut garbler_wires = in_msg.msg();
        timer_end!(rcv_time);

        let eval_time = timer_start!(|| "Evaluating GCs");
        // let c = make_relu::<P>();
        let c = make_truncated_relu::<P>(P::EXPONENT_CAPACITY * num_trunc);
        let num_evaluator_inputs = c.num_evaluator_inputs();
        let num_garbler_inputs = c.num_garbler_inputs();
        garbler_wires
            .iter_mut()
            .zip(server_input_wires.chunks(num_garbler_inputs / 2))
            .for_each(|(w1, w2)| w1.extend_from_slice(w2));

        assert_eq!(num_relus, garbler_wires.len());
        assert_eq!(num_evaluator_inputs * num_relus, client_input_wires.len());
        // We access the input wires in reverse.
        // TODO: Make sure the client only get's labels to the first bits
        // and then actually sends those labels
        let outputs = client_input_wires
            .par_chunks(num_evaluator_inputs)
            .zip(garbler_wires)
            .zip(evaluators)
            .map(|((eval_inps, garbler_inps), gc)| {
                let mut c = c.clone();
                let result = gc
                    .eval(&mut c, &garbler_inps, eval_inps)
                    .expect("evaluation failed");
                let output_fp = fancy_garbling::util::u128_from_bits(&result[2..]);
                let output_as: AdditiveShare<P> =
                    FixedPoint::new(P::Field::from_repr((output_fp as u64).into())).into();
                //((result[0], result[1]), output_as)
                output_as
            })
            .collect::<Vec<_>>();
        timer_end!(eval_time);

        // TODO: Hash
        let send_time = timer_start!(|| "Sending inputs");
        // TODO: For now send dummy wires corresponding to comparisons to simulate bandwidth
        let dummy = vec![(0..2 * outputs.len())
            .map(|_| Wire::from_block((0 as u128).into(), 2))
            .collect::<Vec<_>>()];
        let send_message = ServerLabelMsgSend::new(dummy.as_slice());
        bytes::serialize(&mut *writer, &send_message)?;
        let send_message = ClientShareMsgSend::new(outputs.as_slice());
        bytes::serialize(&mut *writer, &send_message)?;

        timer_end!(send_time);
        timer_end!(start_time);
        Ok(())
    }
}
