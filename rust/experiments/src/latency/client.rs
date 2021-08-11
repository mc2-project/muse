use crate::*;
use ::neural_network::{
    layers::*,
    tensors::{Input, Output},
    NeuralArchitecture,
};
use algebra::{fields::near_mersenne_64::F, PrimeField, UniformRandom};
use async_std::{
    io::{BufReader, BufWriter},
    net::TcpStream,
    task,
};
use crypto_primitives::AuthAdditiveShare;
use io_utils::{counting::CountingIO, imux::IMuxAsync};
use num_traits::identities::Zero;
use protocols::{
    client_keygen,
    gc::ClientGcMsgRcv,
    linear_layer::LinearProtocol,
    mpc::{ClientMPC, MPC},
    mpc_offline::{ClientOfflineMPC, OfflineMPC},
    neural_network::NNProtocol,
};
use protocols_sys::{client_acg, ClientACG, SealClientACG};
use std::collections::BTreeMap;

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

pub fn nn_client<R: RngCore + CryptoRng>(
    server_addr: &str,
    architecture: NeuralArchitecture<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    // Sample a random input.
    let input_dims = architecture.layers.first().unwrap().input_dimensions();
    let mut input = Input::zeros(input_dims);
    input
        .iter_mut()
        .for_each(|in_i| *in_i = generate_random_number(rng).1);

    let (client_state, offline_read, offline_write) = {
        let (mut reader, mut writer) = client_connect(server_addr);
        (
            NNProtocol::offline_client_protocol(&mut reader, &mut writer, &architecture, rng)
                .unwrap(),
            reader.count(),
            writer.count(),
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
    add_to_trace!(|| "Offline Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        offline_read, offline_write
    ));
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

    let mut in_shares = BTreeMap::new();
    let mut out_shares: BTreeMap<usize, Output<AuthAdditiveShare<F>>> = BTreeMap::new();
    let linear_time = timer_start!(|| "Linear layers offline phase");
    for (i, layer) in architecture.layers.iter().enumerate() {
        match layer {
            LayerInfo::NLL(_dims, NonLinearLayerInfo::ReLU { .. }) => {}
            LayerInfo::LL(dims, linear_layer_info) => {
                let input_dims = dims.input_dimensions();
                let output_dims = dims.output_dimensions();
                let (in_share, out_share) = match &linear_layer_info {
                    LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                        let mut acg_handler = match &linear_layer_info {
                            LinearLayerInfo::Conv2d { .. } => {
                                SealClientACG::Conv2D(client_acg::Conv2D::new(
                                    &cfhe,
                                    &linear_layer_info,
                                    input_dims,
                                    output_dims,
                                ))
                            }
                            LinearLayerInfo::FullyConnected => {
                                SealClientACG::FullyConnected(client_acg::FullyConnected::new(
                                    &cfhe,
                                    &linear_layer_info,
                                    input_dims,
                                    output_dims,
                                ))
                            }
                            _ => unreachable!(),
                        };
                        LinearProtocol::<TenBitExpParams>::offline_client_acg_protocol(
                            &mut reader,
                            &mut writer,
                            layer.input_dimensions(),
                            layer.output_dimensions(),
                            &mut acg_handler,
                            rng,
                        )
                        .unwrap()
                    }
                    _ => {
                        let inp_zero = Input::zeros(input_dims);
                        let mut output_share = Output::zeros(output_dims);
                        if out_shares.keys().any(|k| k == &(i - 1)) {
                            // If the layer comes after a linear layer, apply the function to
                            // the last layer's output share MAC
                            let prev_output_share = out_shares.get(&(i - 1)).unwrap();
                            linear_layer_info
                                .evaluate_naive_auth(&prev_output_share, &mut output_share);
                            (
                                Input::auth_share_from_parts(inp_zero.clone(), inp_zero),
                                output_share,
                            )
                        } else {
                            // If the layer comes after a non-linear layer, generate a
                            // randomizer, send it to the server to receive back an
                            // authenticated share, and apply the function to that share
                            let mut randomizer = Input::zeros(input_dims);
                            randomizer.iter_mut().for_each(|e| *e = F::uniform(rng));
                            let randomizer =
                                LinearProtocol::<TenBitExpParams>::offline_client_auth_share(
                                    &mut reader,
                                    &mut writer,
                                    randomizer,
                                    &cfhe,
                                )
                                .unwrap();
                            linear_layer_info.evaluate_naive_auth(&randomizer, &mut output_share);
                            (-randomizer, output_share)
                        }
                    }
                };
                // r
                in_shares.insert(i, in_share);
                // -(Lr + s)
                out_shares.insert(i, out_share);
            }
        }
    }
    timer_end!(linear_time);
    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}

// TODO: Pull out this functionality in `neural_network.rs` so this is clean
pub fn garbling<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], _rng: &mut R) {
    let (mut reader, writer) = client_connect(server_addr);

    let activations: usize = layers.iter().map(|e| *e).sum();
    let rcv_gc_time = timer_start!(|| "Receiving GCs");
    let mut gc_s = Vec::with_capacity(activations);
    let mut r_wires = Vec::with_capacity(activations);

    let num_chunks = (activations as f64 / 8192.0).ceil() as usize;
    for i in 0..num_chunks {
        let in_msg: ClientGcMsgRcv = protocols::bytes::deserialize(&mut reader).unwrap();

        let (gc_chunks, r_wire_chunks) = in_msg.msg();
        if i < (num_chunks - 1) {
            assert_eq!(gc_chunks.len(), 8192);
        }
        gc_s.extend(gc_chunks);
        r_wires.extend(r_wire_chunks);
    }
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

    // Generate dummy labels/layer for CDS
    let activations: usize = layers.iter().map(|e| *e).sum();
    let out_mac_shares = vec![F::zero(); activations];
    let out_shares = vec![F::zero(); activations];
    let inp_mac_shares = vec![F::zero(); activations];
    let inp_rands = vec![F::zero(); activations];

    // Generate triples
    protocols::cds::CDSProtocol::<TenBitExpParams>::client_cds(
        &mut reader,
        &mut writer,
        &cfhe,
        layers,
        &out_mac_shares,
        &out_shares,
        &inp_mac_shares,
        &inp_rands,
        rng,
    )
    .unwrap();
    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}

pub fn input_auth<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], rng: &mut R) {
    let (mut reader, mut writer) = client_connect(server_addr);

    // Keygen
    let cfhe = client_keygen(&mut writer).unwrap();
    writer.reset();

    // Generate dummy labels/layer for CDS
    let activations: usize = layers.iter().map(|e| *e).sum();
    let modulus_bits = <F as PrimeField>::size_in_bits();
    let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;

    let out_mac_shares = vec![F::zero(); activations];
    let out_shares_bits = vec![F::zero(); activations * modulus_bits];
    let inp_mac_shares = vec![F::zero(); activations];
    let inp_rands_bits = vec![F::zero(); activations * modulus_bits];

    let num_rands = 2 * (activations + activations * modulus_bits);

    // Generate rands
    let gen = ClientOfflineMPC::new(&cfhe);

    let input_time = timer_start!(|| "Input Auth");
    let rands = gen.rands_gen(&mut reader, &mut writer, rng, num_rands);
    let mut mpc = ClientMPC::new(rands, Vec::new());

    // Share inputs
    let share_time = timer_start!(|| "Client receiving inputs");
    let _s_out_mac_keys = mpc
        .recv_private_inputs(&mut reader, &mut writer, layers.len())
        .unwrap();
    let _s_inp_mac_keys = mpc
        .recv_private_inputs(&mut reader, &mut writer, layers.len())
        .unwrap();
    let _s_out_mac_shares = mpc
        .recv_private_inputs(&mut reader, &mut writer, activations)
        .unwrap();
    let _s_inp_mac_shares = mpc
        .recv_private_inputs(&mut reader, &mut writer, activations)
        .unwrap();
    let _zero_labels = mpc
        .recv_private_inputs(
            &mut reader,
            &mut writer,
            2 * activations * modulus_bits * elems_per_label,
        )
        .unwrap();
    let _one_labels = mpc
        .recv_private_inputs(
            &mut reader,
            &mut writer,
            2 * activations * modulus_bits * elems_per_label,
        )
        .unwrap();
    timer_end!(share_time);

    // Receive client shares
    let recv_time = timer_start!(|| "Client sending inputs");
    let _out_bits = mpc
        .private_inputs(&mut reader, &mut writer, out_shares_bits.as_slice(), rng)
        .unwrap();
    let _inp_bits = mpc
        .private_inputs(&mut reader, &mut writer, inp_rands_bits.as_slice(), rng)
        .unwrap();
    let _c_out_mac_shares = mpc
        .private_inputs(&mut reader, &mut writer, out_mac_shares.as_slice(), rng)
        .unwrap();
    let _c_inp_mac_shares = mpc
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

//pub fn async_input_auth<R: RngCore + CryptoRng>(
//    server_addr: &str,
//    server_addr_2: &str,
//    layers: &[usize],
//    rng: &mut R,
//) {
//    use protocols::async_server_keygen;
//    use protocols_sys::{SealCT, SerialCT};
//
//    let (mut sync_reader, mut sync_writer) = client_connect_sync(server_addr);
//
//    // Give server time to start async listener
//    std::thread::sleep_ms(1000);
//
//    let (mut reader, mut writer) =
//        task::block_on(async { client_connect_async(server_addr).await });
//
//    // Keygen
//    let (cfhe, mut sfhe) = task::block_on(async {
//        (
//            async_client_keygen(&mut writer).await.unwrap(),
//            async_server_keygen(&mut reader).await.unwrap(),
//        )
//    });
//    writer.reset();
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
//    let num_rands = 2 * (activations + activations * modulus_bits);
//
//    // Generate rands
//    let gen = ClientOfflineMPC::new(&cfhe);
//
//    let input_time = timer_start!(|| "Input Auth");
//    let rands = gen.async_rands_gen(&mut reader, &mut writer, rng, num_rands);
//    let mut mpc = ClientMPC::new(rands, Vec::new());
//
//    // Share inputs
//    let share_time = timer_start!(|| "Client receiving inputs");
//    let s_out_mac_keys = mpc
//        .recv_private_inputs(&mut sync_reader, &mut sync_writer, layers.len())
//        .unwrap();
//    let s_inp_mac_keys = mpc
//        .recv_private_inputs(&mut sync_reader, &mut sync_writer, layers.len())
//        .unwrap();
//    let s_out_mac_shares = mpc
//        .recv_private_inputs(&mut sync_reader, &mut sync_writer, activations)
//        .unwrap();
//    let s_inp_mac_shares = mpc
//        .recv_private_inputs(&mut sync_reader, &mut sync_writer, activations)
//        .unwrap();
//    let zero_labels = mpc
//        .recv_private_inputs(
//            &mut sync_reader,
//            &mut sync_writer,
//            2 * activations * modulus_bits * elems_per_label,
//        )
//        .unwrap();
//    let one_labels = mpc
//        .recv_private_inputs(
//            &mut sync_reader,
//            &mut sync_writer,
//            2 * activations * modulus_bits * elems_per_label,
//        )
//        .unwrap();
//    timer_end!(share_time);
//
//    // Receive client shares
//    let recv_time = timer_start!(|| "Client sending inputs");
//    let out_bits = mpc
//        .private_inputs(
//            &mut sync_reader,
//            &mut sync_writer,
//            out_shares_bits.as_slice(),
//            rng,
//        )
//        .unwrap();
//    let inp_bits = mpc
//        .private_inputs(
//            &mut sync_reader,
//            &mut sync_writer,
//            inp_rands_bits.as_slice(),
//            rng,
//        )
//        .unwrap();
//    let c_out_mac_shares = mpc
//        .private_inputs(
//            &mut sync_reader,
//            &mut sync_writer,
//            out_mac_shares.as_slice(),
//            rng,
//        )
//        .unwrap();
//    let c_inp_mac_shares = mpc
//        .private_inputs(
//            &mut sync_reader,
//            &mut sync_writer,
//            inp_mac_shares.as_slice(),
//            rng,
//        )
//        .unwrap();
//    timer_end!(recv_time);
//    timer_end!(input_time);
//    add_to_trace!(|| "Bytes written: ", || format!("{}", writer.count()));
//}

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
