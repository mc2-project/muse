use crate::bytes;
use crate::{error::MpcError, mpc::*, InMessage, OutMessage};
use algebra::{
    fields::{Fp64, Fp64Parameters, PrimeField},
    fixed_point::FixedPointParameters,
    UniformRandom,
};
use crypto_primitives::{
    additive_share::{AuthAdditiveShare, AuthShare, Share},
    beavers_mul::Triple,
    gc::{fancy_garbling, fancy_garbling::Wire},
    PBeaversMul,
};
use io_utils::{
    imux::IMuxAsync,
    threaded::{ThreadedReader, ThreadedWriter},
};
use itertools::{interleave, izip};
use num_traits::{One, Zero};
use protocols_sys::{ClientFHE, ServerFHE};
use rand::{ChaChaRng, CryptoRng, RngCore, SeedableRng};
use rayon::prelude::*;
use scuttlebutt::Block;
use std::{
    marker::PhantomData,
    sync::{Arc, Condvar, Mutex, RwLock},
};

use async_std::io::{Read, Write};

const RANDOMNESS: [u8; 32] = [
    0x99, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0x62, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

#[derive(Default)]
pub struct CDSProtocol<P: FixedPointParameters> {
    _field: PhantomData<P>,
}

pub struct InsecureCDSType;

// The message is a slice of (vectors of) input labels;
pub type InsecureMsgSend<'a, P> =
    OutMessage<'a, [<P as FixedPointParameters>::Field], InsecureCDSType>;
pub type InsecureMsgRcv<P> = InMessage<Vec<<P as FixedPointParameters>::Field>, InsecureCDSType>;

pub type InsecureBlockSend<'a> = OutMessage<'a, [Block], InsecureCDSType>;
pub type InsecureBlockRcv = InMessage<Vec<Block>, InsecureCDSType>;

/// TODO: In-depth explanation
/// TODO: Optimizations
///     * Only commit to one MAC key on AvgPool/Linear layers
///
/// Server private inputs per layer:
///     * Two MAC keys
///     * MAC shares of output share and next input
///     * Labels for output and next input (Each label is 128/field_modulus
///       elements)
///
/// Client private inputs per layer:
///     * Shares of output and input (Bit decomposition)
///     * MAC shares of output share and input
///
/// Triples needed:
///     * (128/field_modulus) multiplications are needed for every bit of the
///       input and output shares
///     * One multiplication is needed for every input and output share
///
/// TODO: For full security the client would input both ACG outputs, and the MAC check
///       slightly changes to include the second MAC check. The performance impact
///       of this is negligble (we do, however, generate the correct amount of input
///       randomness since this is non-negl.)
impl<P: FixedPointParameters, F: Fp64Parameters> CDSProtocol<P>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P: FixedPointParameters<Field = Fp64<F>>,
    P::Field: Share<
        Constant = <P as FixedPointParameters>::Field,
        Ring = <P as FixedPointParameters>::Field,
    >,
    P::Field: AuthShare,
{
    /// Returns the number of rand pairs and triples needed for the CDS circuit
    pub fn num_rands_triples(
        _num_layers: usize,
        total_size: usize,
        modulus_bits: usize,
        elems_per_label: usize,
    ) -> (usize, usize) {
        // TODO: Explain
        let rands = 2 * (2 * total_size + total_size * modulus_bits);
        let triples = 2
            * (
                total_size * modulus_bits   // Check that Client's input is bits
            + modulus_bits * total_size * elems_per_label
                // Share GC labels
            );
        (rands, triples)
    }

    /// Embed a GC label as a vector of field elements
    fn embed_label(label: Block, modulus_bits: usize) -> Vec<P::Field> {
        let label_bits: Vec<u16> = (0..128)
            .map(|i| ((u128::from(label) >> i) as u16) & 1)
            .collect();
        // We separate label_bits into chunks of size modulus_bits-1
        // since if we did modulus_bits there's a chance of overflow
        label_bits
            .chunks(modulus_bits - 1)
            .map(|bits| {
                let val = fancy_garbling::util::u128_from_bits(bits);
                P::Field::from_repr((val as u64).into())
            })
            .collect()
    }

    /// Extract a GC label from a vector of field elements
    fn extract_label(elems: &[P::Field], modulus_bits: usize) -> Block {
        let bits: Vec<u16> = elems
            .iter()
            .flat_map(|e| {
                let s: u64 = e.into_repr().into();
                fancy_garbling::util::u128_to_bits(s as u128, modulus_bits - 1)
            })
            .collect();
        fancy_garbling::util::u128_from_bits(&bits[..128]).into()
    }

    fn cds_subcircuit<W, M>(
        reader: &mut ThreadedReader,
        writer: &mut ThreadedWriter<W>,
        mpc: &mut M,
        modulus_bits: usize,
        elems_per_label: usize,
        zero_labels: &[AuthAdditiveShare<P::Field>],
        one_labels: &[AuthAdditiveShare<P::Field>],
        out_bits: &[AuthAdditiveShare<P::Field>],
        inp_bits: &[AuthAdditiveShare<P::Field>],
        mut challenge_1: P::Field,
        mut challenge_2: P::Field,
        layer_idx: usize,
    ) -> Result<
        (
            Vec<AuthAdditiveShare<P::Field>>,
            AuthAdditiveShare<P::Field>,
            AuthAdditiveShare<P::Field>,
        ),
        MpcError,
    >
    where
        W: Write + Send + Unpin,
        M: MPC<P::Field, PBeaversMul<<P::Field as PrimeField>::Params>>,
    {
        let bit_time = timer_start!(|| "Secret sharing GC labels");
        // Interleave the output and input shares to correctly
        // align with garbled circuit input wires
        let gc_input = interleave(out_bits.chunks(modulus_bits), inp_bits.chunks(modulus_bits))
            .flatten()
            .collect::<Vec<_>>();
        // We need to do a multiplication with each bit `elems_per_label` times
        let repeated_bits: Vec<AuthAdditiveShare<_>> = gc_input
            .into_iter()
            .flat_map(|e| vec![*e; elems_per_label])
            .collect();
        // Each label share = l0 + (l1 - l0) * bit
        let l1_minus_l0 = mpc.sub(one_labels, zero_labels)?;
        let rh = mpc.mul_t(
            reader,
            writer,
            l1_minus_l0.as_slice(),
            repeated_bits.as_slice(),
            layer_idx,
        )?;
        let label_shares = mpc.add(zero_labels, rh.as_slice())?;
        timer_end!(bit_time);

        let reconstruct_time = timer_start!(|| "Reconstructing Client input");
        // Reconstruct client input from bit decomposition
        let pows_of_two: Vec<P::Field> = (0..modulus_bits)
            .map(|n| P::Field::from_repr(2_u64.pow(n as u32).into()))
            .collect();
        let out: Vec<AuthAdditiveShare<_>> = out_bits
            .par_chunks(modulus_bits)
            .map(|bits| {
                let terms = mpc.mul_const(bits, pows_of_two.as_slice()).unwrap();
                mpc.sum(&terms)
            })
            .collect();
        let inp: Vec<AuthAdditiveShare<_>> = inp_bits
            .par_chunks(modulus_bits)
            .map(|bits| {
                let terms = mpc.mul_const(bits, pows_of_two.as_slice()).unwrap();
                mpc.sum(&terms)
            })
            .collect();
        timer_end!(reconstruct_time);

        // Compute random linear combinations of reconstructed input
        let comb_time = timer_start!(|| "Computing rhos");
        let mut coeffs_1 = Vec::with_capacity(inp.len());
        let mut coeffs_2 = Vec::with_capacity(inp.len());
        for _ in 0..inp.len() {
            coeffs_1.push(challenge_1);
            coeffs_2.push(challenge_2);
            challenge_1 *= challenge_1;
            challenge_2 *= challenge_2;
        }
        let out_lc = mpc.mul_const(&out, &coeffs_1)?;
        let inp_lc = mpc.mul_const(&inp, &coeffs_2)?;
        let rho_1 = mpc.sum(&out_lc);
        let rho_2 = mpc.sum(&inp_lc);
        timer_end!(comb_time);
        Ok((label_shares, rho_1, rho_2))
    }

    // TODO: clarify that you need to preprocess enough stuff before
    pub fn server_cds<R, W, RNG>(
        mut reader: IMuxAsync<R>,
        mut writer: IMuxAsync<W>,
        sfhe: &ServerFHE,
        triples: Arc<(Mutex<Vec<Triple<P::Field>>>, Condvar)>,
        rands: Arc<Mutex<Vec<AuthAdditiveShare<P::Field>>>>,
        mac_key: P::Field,
        layer_sizes: &[usize],
        out_mac_keys: &[P::Field],
        out_mac_shares: &[P::Field],
        inp_mac_keys: &[P::Field],
        inp_mac_shares: &[P::Field],
        labels: &[(Block, Block)],
        rng: &mut RNG,
    ) -> Result<(), MpcError>
    where
        R: Read + Send + Unpin + 'static,
        W: Write + Send + Unpin + 'static,
        RNG: CryptoRng + RngCore,
    {
        let modulus_bits = <P::Field as PrimeField>::size_in_bits();
        let total_size = out_mac_shares.len();
        let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;

        let (zero_labels, one_labels): (Vec<P::Field>, Vec<P::Field>) = labels
            .iter()
            .flat_map(|(b0, b1)| {
                izip!(
                    CDSProtocol::<P>::embed_label(*b0, modulus_bits),
                    CDSProtocol::<P>::embed_label(*b1, modulus_bits),
                )
            })
            .unzip();

        // Create an MPC object for each layer
        let idx = Arc::new(RwLock::new(layer_sizes.len() - 1));
        let mut layer_mpc: Vec<_> = layer_sizes
            .iter()
            .map(|_| ServerMPC::new(rands.clone(), triples.clone(), mac_key, idx.clone()))
            .collect();

        // Share inputs
        let share_time = timer_start!(|| "Server sharing inputs");
        let zero_labels =
            layer_mpc[0].private_inputs(&mut reader, &mut writer, zero_labels.as_slice(), rng)?;
        let one_labels =
            layer_mpc[0].private_inputs(&mut reader, &mut writer, one_labels.as_slice(), rng)?;
        timer_end!(share_time);

        // Receive client shares
        let recv_time = timer_start!(|| "Server receiving inputs");
        let out_bits = layer_mpc[0].recv_private_inputs(
            &mut reader,
            &mut writer,
            total_size * modulus_bits,
        )?;
        let inp_bits = layer_mpc[0].recv_private_inputs(
            &mut reader,
            &mut writer,
            total_size * modulus_bits,
        )?;
        timer_end!(recv_time);

        // Check that inputs are bits
        // TODO: Could do some sort of linear combo here if opening is expensive
        let one_minus_out_bits = out_bits
            .iter()
            .map(|b| b.sub_constant(P::Field::one()))
            .collect::<Vec<_>>();
        let one_minus_inp_bits = inp_bits
            .iter()
            .map(|b| b.sub_constant(P::Field::one()))
            .collect::<Vec<_>>();
        // Multiply
        let out_are_bits =
            layer_mpc[0].mul(&mut reader, &mut writer, &out_bits, &one_minus_out_bits)?;
        let inp_are_bits =
            layer_mpc[0].mul(&mut reader, &mut writer, &inp_bits, &one_minus_inp_bits)?;
        // Receive opening
        let out_are_bits = layer_mpc[0].private_recv(&mut reader, &out_are_bits)?;
        let inp_are_bits = layer_mpc[0].private_recv(&mut reader, &inp_are_bits)?;
        if !(out_are_bits.iter().all(|e| e.is_zero()) && inp_are_bits.iter().all(|e| e.is_zero())) {
            return Err(MpcError::NotBits);
        }

        let cds_time = timer_start!(|| "CDS Protocol");

        //// Calculate number of layer per thread
        let num_threads = std::cmp::min(layer_sizes.len(), rayon::current_num_threads() - 1);
        //// TODO: Some layers may not do all of this

        let threaded_reader = ThreadedReader::new(reader);
        let threaded_writer = ThreadedWriter::new(writer);

        // Create a reader/writer for each layer
        let layer_io: Vec<(ThreadedReader, ThreadedWriter<_>)> = layer_sizes
            .iter()
            .map(|_| (threaded_reader.clone(), threaded_writer.clone()))
            .collect();

        layer_sizes
            .par_iter()
            .enumerate()
            .zip(layer_io.into_par_iter())
            .zip(layer_mpc.into_par_iter())
            .for_each(|(((i, layer_size), (mut reader, mut writer)), mut mpc)| {
                let layer_time = timer_start!(|| format!("Server layer {} CDS subcircuit", i));

                let rng = &mut ChaChaRng::from_seed(RANDOMNESS);

                let processed: usize = layer_sizes[0..i].iter().sum();
                let bits_processed: usize =
                    layer_sizes[0..i].iter().map(|e| e * modulus_bits).sum();
                let labels_processed: usize = layer_sizes[0..i]
                    .iter()
                    .map(|e| 2 * e * modulus_bits * elems_per_label)
                    .sum();

                let layer_range = processed..(processed + layer_size);
                let bit_range = bits_processed..(bits_processed + layer_size * modulus_bits);
                let label_range = labels_processed
                    ..(labels_processed + 2 * layer_size * modulus_bits * elems_per_label);

                // Send random challenges
                let mut challenge_1 = P::Field::uniform(rng);
                let mut challenge_2 = P::Field::uniform(rng);
                let challenge = &[challenge_1, challenge_2];
                let send_message = InsecureMsgSend::<P>::new(challenge);
                bytes::serialize_t(&mut writer, &send_message).unwrap();

                let (label_shares, rho_1, rho_2) = CDSProtocol::<P>::cds_subcircuit(
                    &mut reader,
                    &mut writer,
                    &mut mpc,
                    modulus_bits,
                    elems_per_label,
                    &zero_labels[label_range.clone()],
                    &one_labels[label_range.clone()],
                    &out_bits[bit_range.clone()],
                    &inp_bits[bit_range],
                    challenge_1,
                    challenge_2,
                    i,
                )
                .unwrap();

                // Compute sigmas
                let comb_time = timer_start!(|| "Computing sigmas");
                let (server_sigma_1, server_sigma_2) = izip!(
                    &out_mac_shares[layer_range.clone()],
                    &inp_mac_shares[layer_range]
                )
                .map(|(out_mac, inp_mac)| {
                    let output = (challenge_1 * *out_mac, challenge_2 * *inp_mac);
                    challenge_1 *= challenge_1;
                    challenge_2 *= challenge_2;
                    output
                })
                .fold((P::Field::zero(), P::Field::zero()), |acc, sum| {
                    (acc.0 + sum.0, acc.1 + sum.1)
                });
                timer_end!(comb_time);

                // Receive opening of rho_1, rho_2
                let recv_time = timer_start!(|| "Server receiving rho");
                let rho_open = mpc.private_recv_t(&mut reader, &[rho_1, rho_2]).unwrap();
                timer_end!(recv_time);

                // Receive omega_1, omega_2
                // TODO: Rename insecure
                let recv_time = timer_start!(|| "Server receiving sigma");
                let recv_message: InsecureMsgRcv<P> = bytes::deserialize_t(&mut reader).unwrap();
                let msg = recv_message.msg();
                let client_sigma_1 = msg[0];
                let client_sigma_2 = msg[1];
                timer_end!(recv_time);

                // Check relations
                let result_1 = out_mac_keys[i] * rho_open[0] - (client_sigma_1 + server_sigma_1);
                let result_2 = inp_mac_keys[i] * rho_open[1] - (client_sigma_2 + server_sigma_2);

                // Send label shares if shares are zero and all MACs are correct
                let send_time = timer_start!(|| "Server sending label shares");
                if result_1.is_zero() && result_2.is_zero() && mpc.check_macs().is_ok() {
                    mpc.private_open_t(&mut writer, label_shares.as_slice())
                        .unwrap();
                } else {
                    //return Err(MpcError::InvalidMAC);
                    panic!("Invalid MAC");
                }
                timer_end!(send_time);
                timer_end!(layer_time);
            });

        //        for (i, layer_size) in layer_sizes.iter().enumerate() {
        //            let layer_time = timer_start!(|| "Server layer CDS subcircuit");
        //
        //            let processed: usize = layer_sizes[0..i].iter().sum();
        //            let bits_processed: usize = layer_sizes[0..i].iter().map(|e| e * modulus_bits).sum();
        //            let labels_processed: usize = layer_sizes[0..i].iter().map(|e| 2 * e * modulus_bits * elems_per_label).sum();
        //
        //            let layer_range = processed..(processed + layer_size);
        //            let bit_range = bits_processed..(bits_processed + layer_size * modulus_bits);
        //            let label_range = labels_processed
        //                ..(labels_processed + 2 * layer_size * modulus_bits * elems_per_label);
        //
        //            // Send random challenges
        //            let mut challenge_1 = P::Field::uniform(rng);
        //            let mut challenge_2 = P::Field::uniform(rng);
        //            let challenge = &[challenge_1, challenge_2];
        //            let send_message = InsecureMsgSend::<P>::new(challenge);
        //            bytes::serialize_t(&mut threaded_writer, &send_message).unwrap();
        //
        //            let (label_shares, rho_1, rho_2) = CDSProtocol::<P>::cds_subcircuit(
        //                &mut threaded_reader,
        //                &mut threaded_writer,
        //                mpc,
        //                modulus_bits,
        //                elems_per_label,
        //                &zero_labels[label_range.clone()],
        //                &one_labels[label_range.clone()],
        //                &out_bits[bit_range.clone()],
        //                &inp_bits[bit_range],
        //                challenge_1,
        //                challenge_2,
        //            )?;
        //
        //            // Compute sigmas
        //            let comb_time = timer_start!(|| "Computing sigmas");
        //            let (server_sigma_1, server_sigma_2) = izip!(
        //                &out_mac_shares[layer_range.clone()],
        //                &inp_mac_shares[layer_range]
        //            )
        //            .map(|(out_mac, inp_mac)| {
        //                let output = (challenge_1 * *out_mac, challenge_2 * *inp_mac);
        //                challenge_1 *= challenge_1;
        //                challenge_2 *= challenge_2;
        //                output
        //            })
        //            .fold((P::Field::zero(), P::Field::zero()), |acc, sum| {
        //                (acc.0 + sum.0, acc.1 + sum.1)
        //            });
        //            timer_end!(comb_time);
        //
        //            // Receive opening of rho_1, rho_2
        //            let recv_time = timer_start!(|| "Server receiving rho");
        //            let rho_open = mpc.private_recv_t(&mut threaded_reader, &[rho_1, rho_2])?;
        //            timer_end!(recv_time);
        //
        //            // Receive omega_1, omega_2
        //            // TODO: Rename insecure
        //            let recv_time = timer_start!(|| "Server receiving sigma");
        //            let recv_message: InsecureMsgRcv<P> = bytes::deserialize_t(&mut threaded_reader)?;
        //            let msg = recv_message.msg();
        //            let client_sigma_1 = msg[0];
        //            let client_sigma_2 = msg[1];
        //            timer_end!(recv_time);
        //
        //            // Check relations
        //            let result_1 = out_mac_keys[i] * rho_open[0] - (client_sigma_1 + server_sigma_1);
        //            let result_2 = inp_mac_keys[i] * rho_open[1] - (client_sigma_2 + server_sigma_2);
        //
        //            // Send label shares if shares are zero and all MACs are correct
        //            let send_time = timer_start!(|| "Server sending label shares");
        //            if result_1.is_zero() && result_2.is_zero() && mpc.check_macs().is_ok() {
        //                mpc.private_open_t(&mut threaded_writer, label_shares.as_slice())?;
        //            } else {
        //                return Err(MpcError::InvalidMAC);
        //            }
        //            timer_end!(send_time);
        //
        //            //processed += layer_size;
        //            //bits_processed += layer_size * modulus_bits;
        //            //labels_processed += 2 * layer_size * modulus_bits * elems_per_label;
        //            timer_end!(layer_time);
        //        }
        timer_end!(cds_time);
        Ok(())
    }

    pub fn client_cds<R, W, RNG>(
        mut reader: IMuxAsync<R>,
        mut writer: IMuxAsync<W>,
        cfhe: &ClientFHE,
        triples: Arc<(Mutex<Vec<Triple<P::Field>>>, Condvar)>,
        rands: Arc<Mutex<Vec<AuthAdditiveShare<P::Field>>>>,
        layer_sizes: &[usize],
        out_mac_shares: &[P::Field],
        out_shares: &[P::Field],
        inp_mac_shares: &[P::Field],
        inp_rands: &[P::Field],
        rng: &mut RNG,
    ) -> Result<Vec<Wire>, MpcError>
    where
        R: Read + Send + Unpin + 'static,
        W: Write + Send + Unpin + 'static,
        RNG: CryptoRng + RngCore,
    {
        // TODO: Generate triples in background during linear layers
        let modulus_bits = <P::Field as PrimeField>::size_in_bits();
        let total_size = out_mac_shares.len();
        let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;

        // Compute bit decomposition of shares
        let out_bits: Vec<P::Field> = out_shares
            .iter()
            .flat_map(|e| {
                let s: u64 = e.into_repr().into();
                fancy_garbling::util::u128_to_bits(s.into(), modulus_bits)
            })
            .map(|b| P::Field::from_repr((b as u64).into()))
            .collect();

        let inp_bits: Vec<P::Field> = inp_rands
            .iter()
            .flat_map(|e| {
                let s: u64 = e.into_repr().into();
                fancy_garbling::util::u128_to_bits(s.into(), modulus_bits)
            })
            .map(|b| P::Field::from_repr((b as u64).into()))
            .collect();

        // Create an MPC object for each layer
        let idx = Arc::new(RwLock::new(layer_sizes.len() - 1));
        let mut layer_mpc: Vec<_> = layer_sizes
            .iter()
            .map(|_| ClientMPC::new(rands.clone(), triples.clone(), idx.clone()))
            .collect();

        // Receive server inputs
        let recv_time = timer_start!(|| "Client receiving inputs");
        let zero_labels = layer_mpc[0].recv_private_inputs(
            &mut reader,
            &mut writer,
            2 * total_size * modulus_bits * elems_per_label,
        )?;
        let one_labels = layer_mpc[0].recv_private_inputs(
            &mut reader,
            &mut writer,
            2 * total_size * modulus_bits * elems_per_label,
        )?;
        timer_end!(recv_time);

        // Share inputs
        let send_time = timer_start!(|| "Client sharing inputs");
        let out_bits =
            layer_mpc[0].private_inputs(&mut reader, &mut writer, out_bits.as_slice(), rng)?;
        let inp_bits =
            layer_mpc[0].private_inputs(&mut reader, &mut writer, inp_bits.as_slice(), rng)?;
        timer_end!(send_time);

        // Check that inputs are bits
        // Currently for constants, only one party does it
        let one_minus_out_bits = out_bits.clone();
        let one_minus_inp_bits = inp_bits.clone();
        // Multiply
        let out_are_bits =
            layer_mpc[0].mul(&mut reader, &mut writer, &out_bits, &one_minus_out_bits)?;
        let inp_are_bits =
            layer_mpc[0].mul(&mut reader, &mut writer, &inp_bits, &one_minus_inp_bits)?;
        // Send opening
        layer_mpc[0].private_open(&mut writer, &out_are_bits)?;
        layer_mpc[0].private_open(&mut writer, &inp_are_bits)?;

        // TODO: Parallelize this
        let cds_time = timer_start!(|| "CDS Protocol");

        let mut threaded_reader = ThreadedReader::new(reader);
        let mut threaded_writer = ThreadedWriter::new(writer);

        // Create a reader/writer for each layer
        let layer_io: Vec<(ThreadedReader, ThreadedWriter<_>)> = layer_sizes
            .iter()
            .map(|_| (threaded_reader.clone(), threaded_writer.clone()))
            .collect();

        let labels: Vec<Block> = layer_sizes
            .par_iter()
            .enumerate()
            .zip(layer_io.into_par_iter())
            .zip(layer_mpc.into_par_iter())
            .flat_map(|(((i, layer_size), (mut reader, mut writer)), mut mpc)| {
                let layer_time = timer_start!(|| format!("Server layer {} CDS subcircuit", i));

                let rng = &mut ChaChaRng::from_seed(RANDOMNESS);

                let processed: usize = layer_sizes[0..i].iter().sum();
                let bits_processed: usize =
                    layer_sizes[0..i].iter().map(|e| e * modulus_bits).sum();
                let labels_processed: usize = layer_sizes[0..i]
                    .iter()
                    .map(|e| 2 * e * modulus_bits * elems_per_label)
                    .sum();

                let layer_range = processed..(processed + layer_size);
                let bit_range = bits_processed..(bits_processed + layer_size * modulus_bits);
                let label_range = labels_processed
                    ..(labels_processed + 2 * layer_size * modulus_bits * elems_per_label);

                // Receive random challenges
                let recv_message: InsecureMsgRcv<P> = bytes::deserialize_t(&mut reader).unwrap();
                let msg = recv_message.msg();
                let mut challenge_1 = msg[0];
                let mut challenge_2 = msg[1];

                let (label_shares, rho_1, rho_2) = CDSProtocol::<P>::cds_subcircuit(
                    &mut reader,
                    &mut writer,
                    &mut mpc,
                    modulus_bits,
                    elems_per_label,
                    &zero_labels[label_range.clone()],
                    &one_labels[label_range],
                    &out_bits[bit_range.clone()],
                    &inp_bits[bit_range],
                    challenge_1,
                    challenge_2,
                    i,
                )
                .unwrap();

                // TODO: Compute omega_1, omega_2
                // TODO: Ensure that subcircuit does mutate these challenges
                let comb_time = timer_start!(|| "Computing sigmas");
                let (sigma_1, sigma_2) = izip!(
                    &out_mac_shares[layer_range.clone()],
                    &inp_mac_shares[layer_range]
                )
                .map(|(out_mac, inp_mac)| {
                    let output = (challenge_1 * *out_mac, challenge_2 * *inp_mac);
                    challenge_1 *= challenge_1;
                    challenge_2 *= challenge_2;
                    output
                })
                .fold((P::Field::zero(), P::Field::zero()), |acc, sum| {
                    (acc.0 + sum.0, acc.1 + sum.1)
                });
                timer_end!(comb_time);

                // Send opening of rho_1, rho_2
                let open_time = timer_start!(|| "Client opening rho");
                mpc.private_open_t(&mut writer, &[rho_1, rho_2]).unwrap();
                timer_end!(open_time);

                // Send sigma_1, sigma_2
                let open_time = timer_start!(|| "Client sending sigmas");
                let sigma = &[sigma_1, sigma_2];
                let send_message = InsecureMsgSend::<P>::new(sigma);
                bytes::serialize_t(&mut writer, &send_message).unwrap();
                timer_end!(open_time);

                // Receive label shares
                let recv_time = timer_start!(|| "Client receiving label shares");
                let label_elems = mpc
                    .private_recv_t(&mut reader, label_shares.as_slice())
                    .map_err(|_| {
                        MpcError::CommunicationError("Server MAC check failed".to_string())
                    })
                    .unwrap();
                let result: Vec<Block> = label_elems
                    .chunks(elems_per_label)
                    .map(|e| CDSProtocol::<P>::extract_label(e, modulus_bits))
                    .collect();
                timer_end!(recv_time);
                timer_end!(layer_time);
                result
            })
            .collect();

        timer_end!(cds_time);
        Ok(labels
            .into_iter()
            .map(|l| Wire::from_block(l, 2))
            .collect::<Vec<_>>())
    }

    /// Insecure protocol where server receives client input in cleartext,
    /// checks MACs, and sends back correct labels
    pub fn insecure_server_cds<
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: CryptoRng + RngCore,
    >(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        _sfhe: &ServerFHE,
        layer_sizes: &[usize],
        output_mac_keys: &[P::Field],
        output_mac_shares: &[P::Field],
        input_mac_keys: &[P::Field],
        input_mac_shares: &[P::Field],
        labels: &[(Block, Block)],
        _rng: &mut RNG,
    ) -> Result<(), MpcError> {
        let modulus_bits = <P::Field as PrimeField>::size_in_bits();
        // Receive inputs
        let recv_message: InsecureMsgRcv<P> = bytes::deserialize(&mut *reader)?;
        let client_output_mac_shares = recv_message.msg();
        let recv_message: InsecureMsgRcv<P> = bytes::deserialize(&mut *reader)?;
        let client_output_shares = recv_message.msg();
        let recv_message: InsecureMsgRcv<P> = bytes::deserialize(&mut *reader)?;
        let client_input_mac_shares = recv_message.msg();
        let recv_message: InsecureMsgRcv<P> = bytes::deserialize(&mut *reader)?;
        let client_input_rands = recv_message.msg();

        // Check MACs
        let mut num_checked = 0;
        output_mac_keys
            .iter()
            .enumerate()
            .flat_map(|(i, key)| {
                let server_mac = &output_mac_shares[num_checked..(num_checked + layer_sizes[i])];
                let client_mac =
                    &client_output_mac_shares[num_checked..(num_checked + layer_sizes[i])];
                let client_share =
                    &client_output_shares[num_checked..(num_checked + layer_sizes[i])];
                num_checked += layer_sizes[i];
                izip!(server_mac, client_mac, client_share).map(move |(s_m, c_m, c_s)| {
                    AuthShare::open(AuthAdditiveShare::new(*c_s, *s_m + *c_m), key)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        num_checked = 0;
        input_mac_keys
            .iter()
            .enumerate()
            .flat_map(|(i, key)| {
                let server_mac = &input_mac_shares[num_checked..(num_checked + layer_sizes[i])];
                let client_mac =
                    &client_input_mac_shares[num_checked..(num_checked + layer_sizes[i])];
                let client_share = &client_input_rands[num_checked..(num_checked + layer_sizes[i])];
                num_checked += layer_sizes[i];
                izip!(server_mac, client_mac, client_share).map(move |(s_m, c_m, c_s)| {
                    AuthShare::open(AuthAdditiveShare::new(*c_s, *s_m + *c_m), key)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Each GC is executed one at a time, so we need to interleave pairs
        // of inputs
        let gc_inputs =
            interleave(client_output_shares.iter(), client_input_rands.iter()).collect::<Vec<_>>();

        let gc_bits: Vec<bool> = gc_inputs
            .iter()
            .flat_map(|e| {
                let s: u64 = e.into_repr().into();
                fancy_garbling::util::u128_to_bits(s.into(), modulus_bits)
            })
            .map(|b| b == 1)
            .collect();

        let client_labels: Vec<Block> = izip!(gc_bits, labels)
            .map(
                |(b, (zero, one))| {
                    if b {
                        one.clone()
                    } else {
                        zero.clone()
                    }
                },
            )
            .collect();
        let send_message = InsecureBlockSend::new(&client_labels);
        bytes::serialize(&mut *writer, &send_message)?;
        Ok(())
    }

    /// Insecure protocol where server receives client input in cleartext,
    /// checks MACs, and sends back correct labels
    pub fn insecure_client_cds<
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: CryptoRng + RngCore,
    >(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        _cfhe: &ClientFHE,
        _layer_sizes: &[usize],
        output_mac_shares: &[P::Field],
        output_shares: &[P::Field],
        input_mac_shares: &[P::Field],
        input_rands: &[P::Field],
        _rng: &mut RNG,
    ) -> Result<Vec<Wire>, MpcError> {
        let modulus_bits = <P::Field as PrimeField>::size_in_bits();
        // Send everything to the server
        let send_message = InsecureMsgSend::<P>::new(&output_mac_shares);
        bytes::serialize(&mut *writer, &send_message)?;
        let send_message = InsecureMsgSend::<P>::new(&output_shares);
        bytes::serialize(&mut *writer, &send_message)?;
        let send_message = InsecureMsgSend::<P>::new(&input_mac_shares);
        bytes::serialize(&mut *writer, &send_message)?;
        let send_message = InsecureMsgSend::<P>::new(&input_rands);
        bytes::serialize(&mut *writer, &send_message)?;

        let recv_message: InsecureBlockRcv = bytes::deserialize(&mut *reader)?;
        let labels = recv_message.msg();
        Ok(labels
            .into_iter()
            .map(|l| Wire::from_block(l, 2))
            .collect::<Vec<_>>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebra::{fields::near_mersenne_64::F, fixed_point::FixedPointParameters};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaChaRng;

    // TODO: Add more tests here

    struct TenBitExpParams {}
    impl FixedPointParameters for TenBitExpParams {
        type Field = F;
        const MANTISSA_CAPACITY: u8 = 3;
        const EXPONENT_CAPACITY: u8 = 8;
    }

    const RANDOMNESS: [u8; 32] = [
        0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4,
        0x76, 0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a,
        0x52, 0xd2,
    ];

    #[test]
    fn test_label_embedding() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let modulus_bits = F::size_in_bits();
        let num_labels = 10;

        let mut blocks: Vec<Block> = Vec::with_capacity(num_labels);
        for _ in 0..num_labels {
            let val: u128 = rng.gen();
            blocks.push(val.into());
        }
        let embedding = blocks
            .iter()
            .map(|b| CDSProtocol::<TenBitExpParams>::embed_label(*b, modulus_bits))
            .collect::<Vec<Vec<_>>>();
        let recovered = embedding
            .iter()
            .map(|e| CDSProtocol::<TenBitExpParams>::extract_label(e.as_slice(), modulus_bits))
            .collect::<Vec<_>>();
        izip!(blocks, recovered).for_each(|(b, r)| assert_eq!(b, r));
    }
}
