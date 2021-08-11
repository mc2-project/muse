use crate::*;
use ::neural_network::{
    layers::*,
    tensors::{Input, Output},
    NeuralArchitecture, NeuralNetwork,
};
use algebra::{fields::near_mersenne_64::F, Field, PrimeField, UniformRandom};
use async_std::{
    io::{BufReader, BufWriter},
    net::TcpStream,
    prelude::*,
    task,
};
use crypto_primitives::{
    gc::fancy_garbling::{Encoder, GarbledCircuit, Wire},
    AuthAdditiveShare, AuthShare, Share,
};
use io_utils::{counting::*, imux::*, threaded::*};
use num_traits::identities::Zero;
use protocols::{
    cds::CDSProtocol, client_keygen, gc::ClientGcMsgRcv, linear_layer::LinearProtocol, mpc::*,
    mpc_offline::*, neural_network::NNProtocol,
};
use protocols_sys::*;
use rayon::ThreadPoolBuilder;
use std::{
    collections::BTreeMap,
    sync::{Arc, Condvar, Mutex, RwLock},
};

pub fn client_connect(
    addr: &str,
) -> (
    IMuxAsync<CountingIO<BufReader<TcpStream>>>,
    IMuxAsync<CountingIO<BufWriter<TcpStream>>>,
) {
    // TODO: Maybe change to rayon_num_threads
    let mut readers = Vec::with_capacity(16);
    let mut writers = Vec::with_capacity(16);
    task::block_on(async {
        for _ in 0..16 {
            let stream = TcpStream::connect(addr).await.unwrap();
            readers.push(CountingIO::new(BufReader::new(stream.clone())));
            writers.push(CountingIO::new(BufWriter::new(stream)));
        }
        (IMuxAsync::new(readers), IMuxAsync::new(writers))
    })
}

pub fn client_connect_3(
    addr: &str,
) -> (
    IMuxAsync<CountingIO<BufReader<TcpStream>>>,
    IMuxAsync<CountingIO<BufWriter<TcpStream>>>,
    IMuxAsync<CountingIO<BufReader<TcpStream>>>,
    IMuxAsync<CountingIO<BufWriter<TcpStream>>>,
    IMuxAsync<CountingIO<BufReader<TcpStream>>>,
    IMuxAsync<CountingIO<BufWriter<TcpStream>>>,
) {
    // TODO: Maybe change to rayon_num_threads
    let mut readers = Vec::with_capacity(16);
    let mut writers = Vec::with_capacity(16);
    let mut readers_2 = Vec::with_capacity(16);
    let mut writers_2 = Vec::with_capacity(16);
    let mut readers_3 = Vec::with_capacity(16);
    let mut writers_3 = Vec::with_capacity(16);
    task::block_on(async {
        for _ in 0..16 {
            let stream = TcpStream::connect(addr).await.unwrap();
            readers.push(CountingIO::new(BufReader::new(stream.clone())));
            writers.push(CountingIO::new(BufWriter::new(stream)));
        }
        for _ in 0..16 {
            let stream = TcpStream::connect(addr).await.unwrap();
            readers_2.push(CountingIO::new(BufReader::new(stream.clone())));
            writers_2.push(CountingIO::new(BufWriter::new(stream)));
        }
        for _ in 0..16 {
            let stream = TcpStream::connect(addr).await.unwrap();
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

pub fn nn_client<R: RngCore + CryptoRng + Send>(
    server_addr: &str,
    architecture: NeuralArchitecture<TenBitAS, TenBitExpFP>,
    rng: &mut R,
    rng_2: &mut R,
) {
    // Sample a random input.
    let input_dims = architecture.layers.first().unwrap().input_dimensions();
    let mut input = Input::zeros(input_dims);
    input
        .iter_mut()
        .for_each(|in_i| *in_i = generate_random_number(rng).1);

    let (client_state, offline_read, offline_write) = {
        let (reader, writer, reader_2, writer_2, reader_3, _) = client_connect_3(server_addr);
        (
            NNProtocol::offline_client_protocol(
                reader,
                writer,
                reader_2,
                writer_2,
                reader_3,
                &architecture,
                rng,
                rng_2,
            )
            .unwrap(),
            0,
            0,
        )
    };

    let (_client_output, online_read, online_write) = {
        let (mut reader, mut writer) = client_connect(server_addr);
        (
            NNProtocol::online_client_protocol(
                &mut reader,
                &mut writer,
                &input,
                &architecture,
                &client_state,
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
    architecture: NeuralArchitecture<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    let (mut reader, mut writer) = client_connect(server_addr);

    // Keygen
    let cfhe = client_keygen(&mut writer).unwrap();
    writer.reset();

    let _ = NNProtocol::offline_client_acg(&mut reader, &mut writer, &cfhe, &architecture, rng)
        .unwrap();

    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}

// TODO: Pull out this functionality in `neural_network.rs` so this is clean
pub fn garbling<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], rng: &mut R) {
    let (mut reader, mut writer) = client_connect(server_addr);

    // Keygen
    let cfhe = client_keygen(&mut writer).unwrap();
    writer.reset();

    let rcv_gc_time = timer_start!(|| "Receiving GCs");
    let activations: usize = layers.iter().map(|e| *e).sum();
    let _ = protocols::gc::ReluProtocol::<TenBitExpParams>::offline_client_garbling(
        &mut reader,
        activations,
    )
    .unwrap();
    timer_end!(rcv_gc_time);
    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}

pub fn triples_gen<R: RngCore + CryptoRng>(server_addr: &str, num: usize, rng: &mut R) {
    let (mut reader, mut writer) = client_connect(server_addr);

    // Keygen
    let cfhe = client_keygen(&mut writer).unwrap();
    writer.reset();

    // Generate triples
    let client_gen = ClientOfflineMPC::<F, _>::new(&cfhe);
    let triples = timer_start!(|| "Generating triples");
    client_gen.triples_gen(&mut reader, &mut writer, rng, num);
    timer_end!(triples);
    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}

pub fn cds<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], rng: &mut R) {
    let (mut reader, mut writer) = client_connect(server_addr);

    // Keygen
    let cfhe = client_keygen(&mut writer).unwrap();
    writer.reset();

    // TODO: Reorder
    let activations: usize = layers.iter().map(|e| *e).sum();
    let modulus_bits = <F as PrimeField>::size_in_bits();
    let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;

    // Generate dummy labels/layer for CDS
    let out_mac_shares = vec![F::zero(); activations];
    let out_shares = vec![F::zero(); activations];
    let inp_mac_shares = vec![F::zero(); activations];
    let inp_rands = vec![F::zero(); activations];

    // TODO
    let (num_rands, num_triples) = CDSProtocol::<TenBitExpParams>::num_rands_triples(
        layers.len(),
        activations,
        modulus_bits,
        elems_per_label,
    );

    // Generate rands and triples
    let gen = InsecureClientOfflineMPC::new(&cfhe);
    let rands = gen.rands_gen(&mut reader, &mut writer, rng, num_rands);
    let triples = gen.triples_gen(&mut reader, &mut writer, rng, num_triples);

    // Generate triples
    protocols::cds::CDSProtocol::<TenBitExpParams>::client_cds(
        reader,
        writer,
        &cfhe,
        Arc::new((Mutex::new(triples), Condvar::new())),
        Arc::new(Mutex::new(rands)),
        layers,
        &out_mac_shares,
        &out_shares,
        &inp_mac_shares,
        &inp_rands,
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
    use protocols::server_keygen;
    use protocols_sys::{SealCT, SerialCT};

    let (mut reader, mut writer) = client_connect(server_addr);

    // Keygen
    let cfhe = client_keygen(&mut writer).unwrap();
    let mut sfhe = server_keygen(&mut reader).unwrap();
    reader.reset();
    writer.reset();

    // Generate dummy labels/layer for CDS
    let activations: usize = layers.iter().map(|e| *e).sum();
    let modulus_bits = <F as PrimeField>::size_in_bits();
    let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;

    // TODO: Explain
    let out_mac_shares = vec![F::zero(); 2 * activations];
    let out_shares_bits = vec![F::zero(); activations * modulus_bits];
    let inp_mac_shares = vec![F::zero(); 2 * activations];
    let inp_rands_bits = vec![F::zero(); activations * modulus_bits];

    let (num_rands, _) = CDSProtocol::<TenBitExpParams>::num_rands_triples(
        layers.len(),
        activations,
        modulus_bits,
        elems_per_label,
    );

    // Generate rands
    let gen = ClientOfflineMPC::new(&cfhe);

    let input_time = timer_start!(|| "Input Auth");
    let rands = gen.rands_gen(&mut reader, &mut writer, rng, num_rands);
    let mut mpc = ClientMPC::new(
        Arc::new(Mutex::new(rands)),
        Arc::new((Mutex::new(Vec::new()), Condvar::new())),
        Arc::new(RwLock::new(0)),
    );

    // Share inputs
    let share_time = timer_start!(|| "Client receiving inputs");
    let s_out_mac_keys = mpc
        .recv_private_inputs(&mut reader, &mut writer, layers.len())
        .unwrap();
    let s_inp_mac_keys = mpc
        .recv_private_inputs(&mut reader, &mut writer, layers.len())
        .unwrap();
    let s_out_mac_shares = mpc
        .recv_private_inputs(&mut reader, &mut writer, 2 * activations)
        .unwrap();
    let s_inp_mac_shares = mpc
        .recv_private_inputs(&mut reader, &mut writer, 2 * activations)
        .unwrap();
    let zero_labels = mpc
        .recv_private_inputs(
            &mut reader,
            &mut writer,
            2 * activations * modulus_bits * elems_per_label,
        )
        .unwrap();
    let one_labels = mpc
        .recv_private_inputs(
            &mut reader,
            &mut writer,
            2 * activations * modulus_bits * elems_per_label,
        )
        .unwrap();
    timer_end!(share_time);

    // Receive client shares
    let recv_time = timer_start!(|| "Client sending inputs");
    let out_bits = mpc
        .private_inputs(&mut reader, &mut writer, out_shares_bits.as_slice(), rng)
        .unwrap();
    let inp_bits = mpc
        .private_inputs(&mut reader, &mut writer, inp_rands_bits.as_slice(), rng)
        .unwrap();
    let c_out_mac_shares = mpc
        .private_inputs(&mut reader, &mut writer, out_mac_shares.as_slice(), rng)
        .unwrap();
    let c_inp_mac_shares = mpc
        .private_inputs(&mut reader, &mut writer, inp_mac_shares.as_slice(), rng)
        .unwrap();
    timer_end!(recv_time);
    timer_end!(input_time);
    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}

//pub fn input_auth_ltme<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], rng: &mut R) {
//    use protocols::server_keygen;
//    use protocols_sys::{SealCT, SerialCT};
//
//    let (mut reader, mut writer) = client_connect(server_addr);
//
//    // Keygen
//    let cfhe = client_keygen(&mut writer).unwrap();
//    let mut sfhe = server_keygen(&mut reader).unwrap();
//
//    let gen = ClientOfflineMPC::new(&cfhe);
//    let mut mpc = ClientMPC::<F>::new(Vec::new(), Vec::new());
//
//    let mut ct = gen.recv_mac(&mut reader);
//    let mut mac_ct = SealCT {
//        inner: SerialCT {
//            inner: ct.as_mut_ptr(),
//            size: ct.len() as u64,
//        },
//    };
//
//    // Generate dummy labels/layer for CDS
//    let activations: usize = layers.iter().map(|e| *e).sum();
//    let modulus_bits = <F as PrimeField>::size_in_bits();
//    let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;
//
//    let out_mac_shares = vec![F::zero(); activations];
//    let out_shares_bits = vec![F::zero(); activations * modulus_bits];
//    let inp_mac_shares = vec![F::zero(); activations];
//    let inp_rands_bits = vec![F::zero(); activations * modulus_bits];
//
//    let input_time = timer_start!(|| "Input Auth");
//
//    // Receive server inputs
//    let share_time = timer_start!(|| "Client receiving inputs");
//    let s_out_mac_keys = mpc
//        .recv_private_inputs(&mut reader, &mut writer, layers.len())
//        .unwrap();
//    let s_inp_mac_keys = mpc
//        .recv_private_inputs(&mut reader, &mut writer, layers.len())
//        .unwrap();
//    let s_out_mac_shares = mpc
//        .recv_private_inputs(&mut reader, &mut writer, activations)
//        .unwrap();
//    let s_inp_mac_shares = mpc
//        .recv_private_inputs(&mut reader, &mut writer, activations)
//        .unwrap();
//    let zero_labels = mpc
//        .recv_private_inputs(
//            &mut reader,
//            &mut writer,
//            2 * activations * modulus_bits * elems_per_label,
//        )
//        .unwrap();
//    let one_labels = mpc
//        .recv_private_inputs(
//            &mut reader,
//            &mut writer,
//            2 * activations * modulus_bits * elems_per_label,
//        )
//        .unwrap();
//    timer_end!(share_time);
//
//    // Share inputs
//    let recv_time = timer_start!(|| "Client sending inputs");
//    let out_bits = gen.optimized_input(
//        &mut sfhe,
//        &mut writer,
//        out_shares_bits.as_slice(),
//        &mut mac_ct,
//        rng,
//    );
//    let inp_bits = gen.optimized_input(
//        &mut sfhe,
//        &mut writer,
//        inp_rands_bits.as_slice(),
//        &mut mac_ct,
//        rng,
//    );
//    let c_out_mac_shares = gen.optimized_input(
//        &mut sfhe,
//        &mut writer,
//        out_mac_shares.as_slice(),
//        &mut mac_ct,
//        rng,
//    );
//    let c_inp_mac_shares = gen.optimized_input(
//        &mut sfhe,
//        &mut writer,
//        inp_mac_shares.as_slice(),
//        &mut mac_ct,
//        rng,
//    );
//    timer_end!(recv_time);
//    timer_end!(input_time);
//}
