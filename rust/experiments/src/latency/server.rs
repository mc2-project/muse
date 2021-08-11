use crate::*;
use algebra::{
    fields::near_mersenne_64::F,
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::{Fp64, Fp64Parameters},
    FpParameters, PrimeField, UniformRandom,
};
use crypto_primitives::{
    gc::fancy_garbling::{Encoder, GarbledCircuit, Wire},
    AuthAdditiveShare, AuthShare, Share,
};
use futures::stream::StreamExt;
use io_utils::{counting::*, imux::*, threaded::*};
use neural_network::{
    layers::*,
    tensors::{Input, Output},
    NeuralArchitecture, NeuralNetwork,
};
use num_traits::identities::Zero;
use protocols::{
    cds::*, client_keygen, gc::ServerGcMsgSend, linear_layer::LinearProtocol, mpc::*,
    mpc_offline::*, neural_network::NNProtocol, server_keygen,
};
use protocols_sys::{server_acg, SealClientACG, SealServerACG, ServerACG, ServerFHE};
use rand::{CryptoRng, RngCore};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use scuttlebutt::Block;
use std::{
    collections::BTreeMap,
    sync::{Arc, Condvar, Mutex, RwLock},
};

use async_std::{
    io::{BufReader, BufWriter, Write},
    net::{TcpListener, TcpStream},
    task,
};

pub fn server_connect(
    addr: &str,
) -> (
    IMuxAsync<CountingIO<BufReader<TcpStream>>>,
    IMuxAsync<CountingIO<BufWriter<TcpStream>>>,
) {
    task::block_on(async {
        // TODO: Maybe change to rayon_num_threads
        let listener = TcpListener::bind("0.0.0.0:8000").await.unwrap();
        let mut incoming = listener.incoming();
        let mut readers = Vec::with_capacity(16);
        let mut writers = Vec::with_capacity(16);
        for _ in 0..16 {
            let stream = incoming.next().await.unwrap().unwrap();
            readers.push(CountingIO::new(BufReader::new(stream.clone())));
            writers.push(CountingIO::new(BufWriter::new(stream)));
        }
        (IMuxAsync::new(readers), IMuxAsync::new(writers))
    })
}

pub fn server_connect_3(
    addr: &str,
) -> (
    IMuxAsync<CountingIO<BufReader<TcpStream>>>,
    IMuxAsync<CountingIO<BufWriter<TcpStream>>>,
    IMuxAsync<CountingIO<BufReader<TcpStream>>>,
    IMuxAsync<CountingIO<BufWriter<TcpStream>>>,
    IMuxAsync<CountingIO<BufReader<TcpStream>>>,
    IMuxAsync<CountingIO<BufWriter<TcpStream>>>,
) {
    task::block_on(async {
        // TODO: Maybe change to rayon_num_threads
        let listener = TcpListener::bind("0.0.0.0:8000").await.unwrap();
        let mut incoming = listener.incoming();
        let mut readers = Vec::with_capacity(16);
        let mut writers = Vec::with_capacity(16);
        let mut readers_2 = Vec::with_capacity(16);
        let mut writers_2 = Vec::with_capacity(16);
        let mut readers_3 = Vec::with_capacity(16);
        let mut writers_3 = Vec::with_capacity(16);
        for _ in 0..16 {
            let stream = incoming.next().await.unwrap().unwrap();
            readers.push(CountingIO::new(BufReader::new(stream.clone())));
            writers.push(CountingIO::new(BufWriter::new(stream)));
        }
        for _ in 0..16 {
            let stream = incoming.next().await.unwrap().unwrap();
            readers_2.push(CountingIO::new(BufReader::new(stream.clone())));
            writers_2.push(CountingIO::new(BufWriter::new(stream)));
        }
        for _ in 0..16 {
            let stream = incoming.next().await.unwrap().unwrap();
            readers_3.push(CountingIO::new(BufReader::new(stream.clone())));
            writers_3.push(CountingIO::new(BufWriter::new(stream)));
        }
        (
            IMuxAsync::new(readers),
            IMuxAsync::new(writers),
            IMuxAsync::new(readers_2),
            IMuxAsync::new(writers_2),
            IMuxAsync::new(readers_3),
            IMuxAsync::new(writers_3),
        )
    })
}

pub fn nn_server<R: RngCore + CryptoRng + Send>(
    server_addr: &str,
    nn: NeuralNetwork<TenBitAS, TenBitExpFP>,
    rng: &mut R,
    rng_2: &mut R,
    rng_3: &mut R,
) {
    let (server_offline_state, offline_read, offline_write) = {
        let (reader, writer, reader_2, writer_2, _, writer_3) = server_connect_3(server_addr);

        (
            NNProtocol::offline_server_protocol(
                reader, writer, reader_2, writer_2, writer_3, &nn, rng, rng_2, rng_3,
            )
            .unwrap(),
            0,
            0,
        )
    };

    let (_, online_read, online_write) = {
        let (mut reader, mut writer) = server_connect(server_addr);
        (
            NNProtocol::online_server_protocol(
                &mut reader,
                &mut writer,
                &nn,
                &server_offline_state,
            )
            .unwrap(),
            reader.count(),
            writer.count(),
        )
    };

    // TODO: The multi-threading for the CDS currently requires moving the reader/writer so we
    // can't get the communication count
    //
    //add_to_trace!(|| "Offline Communication", || format!(
    //    "Read {} bytes\nWrote {} bytes",
    //    offline_read, offline_write
    //));
    add_to_trace!(|| "Online Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        online_read, online_write
    ));
}

// TODO: Pull out this functionality in `neural_network.rs` so this is clean
pub fn acg<R: RngCore + CryptoRng>(
    server_addr: &str,
    nn: NeuralNetwork<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    let (mut reader, mut writer) = server_connect(server_addr);

    // Keygen
    let sfhe = server_keygen(&mut reader).unwrap();
    let mac_key = F::uniform(rng);
    reader.reset();

    let _ = NNProtocol::offline_server_acg(&mut reader, &mut writer, &sfhe, &nn, rng).unwrap();

    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}

pub fn garbling<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], rng: &mut R) {
    let (mut reader, mut writer) = server_connect(server_addr);

    // Keygen
    let sfhe = server_keygen(&mut reader).unwrap();
    reader.count();

    let activations: usize = layers.iter().map(|e| *e).sum();
    let output_truncations: Vec<u8> = layers.iter().map(|_| 1).collect();

    let garble_time = timer_start!(|| "Garbling Time");
    let _ = protocols::gc::ReluProtocol::<TenBitExpParams>::offline_server_garbling(
        &mut writer,
        activations,
        &sfhe,
        layers,
        output_truncations.as_slice(),
        rng,
    )
    .unwrap();
    timer_end!(garble_time);
    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}

pub fn triples_gen<R: RngCore + CryptoRng>(server_addr: &str, num: usize, rng: &mut R) {
    let (mut reader, mut writer) = server_connect(server_addr);

    // Keygen
    let sfhe = server_keygen(&mut reader).unwrap();
    let mac_key = F::uniform(rng);
    reader.reset();

    // Generate triples
    let server_gen = ServerOfflineMPC::<F, _>::new(&sfhe, mac_key.into_repr().0);
    let triples = timer_start!(|| "Generating triples");
    server_gen.triples_gen(&mut reader, &mut writer, rng, num);
    timer_end!(triples);
    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}

pub fn cds<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], rng: &mut R) {
    let (mut reader, mut writer) = server_connect(server_addr);

    // Keygen
    let sfhe = server_keygen(&mut reader).unwrap();
    reader.reset();

    // Generate dummy labels/layer for CDS
    let activations: usize = layers.iter().map(|e| *e).sum();
    let modulus_bits = <F as PrimeField>::size_in_bits();
    let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;

    let out_mac_keys = vec![F::zero(); layers.len()];
    let out_mac_shares = vec![F::zero(); activations];
    let inp_mac_keys = vec![F::zero(); layers.len()];
    let inp_mac_shares = vec![F::zero(); activations];
    let labels: Vec<(Block, Block)> = vec![(0.into(), 0.into()); 2 * activations * modulus_bits];

    // Generate rands and triples
    let (num_rands, num_triples) = CDSProtocol::<TenBitExpParams>::num_rands_triples(
        layers.len(),
        activations,
        modulus_bits,
        elems_per_label,
    );
    let mac_key = F::uniform(rng);
    let gen = InsecureServerOfflineMPC::new(&sfhe, mac_key.into_repr().0);
    let rands = gen.rands_gen(&mut reader, &mut writer, rng, num_rands);
    let triples = gen.triples_gen(&mut reader, &mut writer, rng, num_triples);

    // Generate triples
    protocols::cds::CDSProtocol::<TenBitExpParams>::server_cds(
        reader,
        writer,
        &sfhe,
        Arc::new((Mutex::new(triples), Condvar::new())),
        Arc::new((Mutex::new(rands))),
        mac_key,
        layers,
        &out_mac_keys,
        &out_mac_shares,
        &inp_mac_keys,
        &inp_mac_shares,
        labels.as_slice(),
        rng,
    )
    .unwrap();
    // TODO: The multi-threading for the CDS currently requires moving the reader/writer so we
    // can't get the communication count
    //
    //add_to_trace!(|| "Communication", || format!(
    //    "Read {} bytes\nWrote {} bytes",
    //    reader.count(),
    //    writer.count()
    //));
}

pub fn input_auth<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], rng: &mut R) {
    let (mut reader, mut writer) = server_connect(server_addr);

    // Keygen
    let sfhe = server_keygen(&mut reader).unwrap();
    let mut cfhe = client_keygen(&mut writer).unwrap();
    reader.reset();
    writer.reset();

    // Generate dummy labels/layer for CDS
    let activations: usize = layers.iter().map(|e| *e).sum();
    let modulus_bits = <F as PrimeField>::size_in_bits();
    let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;

    let out_mac_keys = vec![F::zero(); layers.len()];
    let out_mac_shares = vec![F::zero(); 2 * activations];
    let inp_mac_keys = vec![F::zero(); layers.len()];
    let inp_mac_shares = vec![F::zero(); 2 * activations];
    let zero_labels = vec![F::zero(); 2 * activations * modulus_bits * elems_per_label];
    let one_labels = vec![F::zero(); 2 * activations * modulus_bits * elems_per_label];

    let (num_rands, _) = CDSProtocol::<TenBitExpParams>::num_rands_triples(
        layers.len(),
        activations,
        modulus_bits,
        elems_per_label,
    );

    // Generate rands
    let mac_key = F::uniform(rng);
    let gen = ServerOfflineMPC::new(&sfhe, mac_key.into_repr().0);

    let input_time = timer_start!(|| "Input Auth");
    let rands = gen.rands_gen(&mut reader, &mut writer, rng, num_rands);
    let mut mpc = ServerMPC::new(
        Arc::new((Mutex::new(rands))),
        Arc::new((Mutex::new(Vec::new()), Condvar::new())),
        mac_key,
        Arc::new(RwLock::new(0)),
    );

    // Share inputs
    let share_time = timer_start!(|| "Server sharing inputs");
    let out_mac_keys = mpc
        .private_inputs(&mut reader, &mut writer, out_mac_keys.as_slice(), rng)
        .unwrap();
    let inp_mac_keys = mpc
        .private_inputs(&mut reader, &mut writer, inp_mac_keys.as_slice(), rng)
        .unwrap();
    let out_mac_shares = mpc
        .private_inputs(&mut reader, &mut writer, out_mac_shares.as_slice(), rng)
        .unwrap();
    let inp_mac_shares = mpc
        .private_inputs(&mut reader, &mut writer, inp_mac_shares.as_slice(), rng)
        .unwrap();
    let zero_labels = mpc
        .private_inputs(&mut reader, &mut writer, zero_labels.as_slice(), rng)
        .unwrap();
    let one_labels = mpc
        .private_inputs(&mut reader, &mut writer, one_labels.as_slice(), rng)
        .unwrap();
    timer_end!(share_time);

    // Receive client shares
    let recv_time = timer_start!(|| "Server receiving inputs");
    let out_bits = mpc
        .recv_private_inputs(&mut reader, &mut writer, activations * modulus_bits)
        .unwrap();
    let inp_bits = mpc
        .recv_private_inputs(&mut reader, &mut writer, activations * modulus_bits)
        .unwrap();
    let c_out_mac_shares = mpc
        .recv_private_inputs(&mut reader, &mut writer, 2 * activations)
        .unwrap();
    let c_inp_mac_shares = mpc
        .recv_private_inputs(&mut reader, &mut writer, 2 * activations)
        .unwrap();
    timer_end!(recv_time);
    timer_end!(input_time);
    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}

//pub fn async_input_auth<R: RngCore + CryptoRng>(
//    server_addr: &str,
//    server_addr_2: &str,
//    layers: &[usize],
//    rng: &mut R,
//) {
//    use protocols::async_client_keygen;
//    use protocols_sys::SealCT;
//
//    // TODO: Need a sync stream for now because async is not implemented for online MPC
//    let (mut sync_reader, mut sync_writer) = server_connect_sync(server_addr);
//
//    let (mut reader, mut writer) =
//        task::block_on(async { server_connect_async(server_addr).await });
//
//    let (sfhe, mut cfhe) = task::block_on(async {
//        (
//            async_server_keygen(&mut reader).await.unwrap(),
//            async_client_keygen(&mut writer).await.unwrap(),
//        )
//    });
//
//    writer.reset();
//
//    // Generate dummy labels/layer for CDS
//    let activations: usize = layers.iter().map(|e| *e).sum();
//    let modulus_bits = <F as PrimeField>::size_in_bits();
//    let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;
//
//    let out_mac_keys = vec![F::zero(); layers.len()];
//    let out_mac_shares = vec![F::zero(); activations];
//    let inp_mac_keys = vec![F::zero(); layers.len()];
//    let inp_mac_shares = vec![F::zero(); activations];
//    let zero_labels = vec![F::zero(); 2 * activations * modulus_bits * elems_per_label];
//    let one_labels = vec![F::zero(); 2 * activations * modulus_bits * elems_per_label];
//
//    let num_rands = 2 * (activations + activations * modulus_bits);
//
//    // Generate rands
//    let mac_key = F::uniform(rng);
//    let gen = ServerOfflineMPC::new(&sfhe, mac_key.into_repr().0);
//
//    let input_time = timer_start!(|| "Input Auth");
//    let rands = gen.async_rands_gen(&mut reader, &mut writer, rng, num_rands);
//    let mut mpc = ServerMPC::new(rands, Vec::new(), mac_key);
//
//    // Share inputs
//    let share_time = timer_start!(|| "Server sharing inputs");
//    let out_mac_keys = mpc
//        .private_inputs(
//            &mut sync_reader,
//            &mut sync_writer,
//            out_mac_keys.as_slice(),
//            rng,
//        )
//        .unwrap();
//    let inp_mac_keys = mpc
//        .private_inputs(
//            &mut sync_reader,
//            &mut sync_writer,
//            inp_mac_keys.as_slice(),
//            rng,
//        )
//        .unwrap();
//    let out_mac_shares = mpc
//        .private_inputs(
//            &mut sync_reader,
//            &mut sync_writer,
//            out_mac_shares.as_slice(),
//            rng,
//        )
//        .unwrap();
//    let inp_mac_shares = mpc
//        .private_inputs(
//            &mut sync_reader,
//            &mut sync_writer,
//            inp_mac_shares.as_slice(),
//            rng,
//        )
//        .unwrap();
//    let zero_labels = mpc
//        .private_inputs(
//            &mut sync_reader,
//            &mut sync_writer,
//            zero_labels.as_slice(),
//            rng,
//        )
//        .unwrap();
//    let one_labels = mpc
//        .private_inputs(
//            &mut sync_reader,
//            &mut sync_writer,
//            one_labels.as_slice(),
//            rng,
//        )
//        .unwrap();
//    timer_end!(share_time);
//
//    // Receive client shares
//    let recv_time = timer_start!(|| "Server receiving inputs");
//    let out_bits = mpc
//        .recv_private_inputs(
//            &mut sync_reader,
//            &mut sync_writer,
//            activations * modulus_bits,
//        )
//        .unwrap();
//    let inp_bits = mpc
//        .recv_private_inputs(
//            &mut sync_reader,
//            &mut sync_writer,
//            activations * modulus_bits,
//        )
//        .unwrap();
//    let c_out_mac_shares = mpc
//        .recv_private_inputs(&mut sync_reader, &mut sync_writer, activations)
//        .unwrap();
//    let c_inp_mac_shares = mpc
//        .recv_private_inputs(&mut sync_reader, &mut sync_writer, activations)
//        .unwrap();
//    timer_end!(recv_time);
//    timer_end!(input_time);
//    add_to_trace!(|| "Bytes written: ", || format!(
//        "{}",
//        writer.count() + sync_writer.count()
//    ));
//}
//
//pub fn input_auth_ltme<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], rng: &mut R) {
//    use protocols::client_keygen;
//    use protocols_sys::SealCT;
//
//    let (mut reader, mut writer) = server_connect_sync(server_addr);
//
//    // Keygen
//    let sfhe = server_keygen(&mut reader).unwrap();
//    let mut cfhe = client_keygen(&mut writer).unwrap();
//
//    // Generate and send MAC key to client
//    let mac_key = F::uniform(rng);
//    let mut mac_ct_seal = SealCT::new();
//    let mac_ct = mac_ct_seal.encrypt_vec(&cfhe, vec![mac_key.into_repr().0, 8192]);
//
//    let gen = ServerOfflineMPC::<F, _>::new(&sfhe, mac_key.into_repr().0);
//    gen.send_mac(&mut writer, mac_ct);
//    let mut mpc = ServerMPC::new(Vec::new(), Vec::new(), mac_key);
//
//    // Generate dummy labels/layer for CDS
//    let activations: usize = layers.iter().map(|e| *e).sum();
//    let modulus_bits = <F as PrimeField>::size_in_bits();
//    let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;
//
//    let out_mac_keys = vec![F::zero(); layers.len()];
//    let out_mac_shares = vec![F::zero(); activations];
//    let inp_mac_keys = vec![F::zero(); layers.len()];
//    let inp_mac_shares = vec![F::zero(); activations];
//    let zero_labels = vec![F::zero(); 2 * activations * modulus_bits * elems_per_label];
//    let one_labels = vec![F::zero(); 2 * activations * modulus_bits * elems_per_label];
//
//    let input_time = timer_start!(|| "Input Auth");
//
//    // Share inputs
//    let share_time = timer_start!(|| "Server sharing inputs");
//    let out_mac_keys = mpc
//        .private_inputs(&mut reader, &mut writer, out_mac_keys.as_slice(), rng)
//        .unwrap();
//    let inp_mac_keys = mpc
//        .private_inputs(&mut reader, &mut writer, inp_mac_keys.as_slice(), rng)
//        .unwrap();
//    let out_mac_shares = mpc
//        .private_inputs(&mut reader, &mut writer, out_mac_shares.as_slice(), rng)
//        .unwrap();
//    let inp_mac_shares = mpc
//        .private_inputs(&mut reader, &mut writer, inp_mac_shares.as_slice(), rng)
//        .unwrap();
//    let zero_labels = mpc
//        .private_inputs(&mut reader, &mut writer, zero_labels.as_slice(), rng)
//        .unwrap();
//    let one_labels = mpc
//        .private_inputs(&mut reader, &mut writer, one_labels.as_slice(), rng)
//        .unwrap();
//    timer_end!(share_time);
//
//    // Receive client shares
//    let recv_time = timer_start!(|| "Server receiving inputs");
//    let out_bits = gen.recv_optimized_input(&cfhe, &mut reader, activations * modulus_bits);
//    let inp_bits = gen.recv_optimized_input(&cfhe, &mut reader, activations * modulus_bits);
//    let c_out_mac_shares = gen.recv_optimized_input(&cfhe, &mut reader, activations);
//    let c_inp_mac_shares = gen.recv_optimized_input(&cfhe, &mut reader, activations);
//    timer_end!(recv_time);
//    timer_end!(input_time);
//}
