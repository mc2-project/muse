use crate::{linear_layer::*, AdditiveShare, ClientKeySend, ServerKeyRcv};
use algebra::{
    fields::near_mersenne_64::F,
    fixed_point::{FixedPoint, FixedPointParameters},
    UniformRandom,
};
use crypto_primitives::additive_share::{AuthAdditiveShare, AuthShare, Share};
use io_utils::IMuxSync;
use neural_network::{
    layers::{average_pooling::*, convolution::*, fully_connected::*, *},
    tensors::*,
    NeuralNetwork,
};
use num_traits::identities::Zero;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use std::{
    io::{BufReader, BufWriter},
    net::{TcpListener, TcpStream},
};

struct TenBitExpParams {}
impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 3;
    const EXPONENT_CAPACITY: u8 = 8;
}

type TenBitExpFP = FixedPoint<TenBitExpParams>;
type TenBitAS = AdditiveShare<TenBitExpParams>;

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -10.0 } else { 10.0 };
    let mut float: f64 = rng.gen();
    float += 1.0;
    let f = TenBitExpFP::truncate_float(float * mul);
    let n = TenBitExpFP::from(f);
    (f, n)
}

fn generate_random_weight<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -10.0 } else { 10.0 };
    let float: f64 = rng.gen_range(-1.0, 1.0);
    let f = TenBitExpFP::truncate_float(float * mul);
    let n = TenBitExpFP::from(f);
    (f, n)
}

fn generate_random_image<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    let pixel: u32 = rng.gen_range(0, 255);
    let float = (pixel as f64) / 255.;
    let f = TenBitExpFP::truncate_float(float * 10.0);
    let n = TenBitExpFP::from(f);
    (f, n)
}

fn sample_conv_layer<R: Rng>(
    input_dims: (usize, usize, usize, usize),
    kernel_dims: (usize, usize, usize, usize),
    stride: usize,
    padding: Padding,
    rng: &mut R,
) -> (
    LinearLayer<TenBitAS, TenBitExpFP>,
    LinearLayer<TenBitExpFP, TenBitExpFP>,
) {
    let mut kernel = Kernel::zeros(kernel_dims);
    let mut bias = Kernel::zeros((kernel_dims.0, 1, 1, 1));
    kernel
        .iter_mut()
        .for_each(|ker_i| *ker_i = generate_random_weight(rng).1);
    bias.iter_mut()
        .for_each(|bias_i| *bias_i = generate_random_weight(rng).1);

    let layer_params =
        Conv2dParams::<TenBitAS, _>::new(padding, stride, kernel.clone(), bias.clone());
    let pt_layer_params =
        Conv2dParams::<TenBitExpFP, _>::new(padding, stride, kernel.clone(), bias.clone());
    let layer_dims = LayerDims {
        input_dims,
        output_dims: layer_params.calculate_output_size(input_dims),
    };
    let layer = LinearLayer::Conv2d {
        dims: layer_dims,
        params: layer_params,
    };
    let pt_layer = LinearLayer::Conv2d {
        dims: layer_dims,
        params: pt_layer_params,
    };
    (layer, pt_layer)
}

fn sample_fc_layer<R: Rng>(
    input_dims: (usize, usize, usize, usize),
    out_chn: usize,
    rng: &mut R,
) -> (
    LinearLayer<TenBitAS, TenBitExpFP>,
    LinearLayer<TenBitExpFP, TenBitExpFP>,
) {
    let mut weights = Kernel::zeros((out_chn, input_dims.1, input_dims.2, input_dims.3));
    let mut bias = Kernel::zeros((out_chn, 1, 1, 1));
    weights
        .iter_mut()
        .for_each(|w_i| *w_i = generate_random_weight(rng).1);
    bias.iter_mut()
        .for_each(|w_i| *w_i = generate_random_weight(rng).1);

    let layer_params = FullyConnectedParams::new(weights.clone(), bias.clone());
    let pt_params = FullyConnectedParams::new(weights.clone(), bias.clone());
    let layer_dims = LayerDims {
        input_dims,
        output_dims: layer_params.calculate_output_size(input_dims),
    };
    let layer = LinearLayer::FullyConnected {
        dims: layer_dims,
        params: layer_params,
    };
    let pt_layer = LinearLayer::FullyConnected {
        dims: layer_dims,
        params: pt_params,
    };
    (layer, pt_layer)
}

fn sample_iden_layer(
    input_dims: (usize, usize, usize, usize),
) -> (
    LinearLayer<TenBitAS, TenBitExpFP>,
    LinearLayer<TenBitExpFP, TenBitExpFP>,
) {
    let layer_dims = LayerDims {
        input_dims,
        output_dims: input_dims,
    };
    let layer = LinearLayer::Identity { dims: layer_dims };
    let pt_layer = LinearLayer::Identity { dims: layer_dims };
    (layer, pt_layer)
}

fn sample_avg_pool_layer(
    input_dims: (usize, usize, usize, usize),
    (pool_h, pool_w): (usize, usize),
    stride: usize,
) -> (
    LinearLayer<TenBitAS, TenBitExpFP>,
    LinearLayer<TenBitExpFP, TenBitExpFP>,
) {
    let size = (pool_h * pool_w) as f64;
    let layer_params = AvgPoolParams::new(pool_h, pool_w, stride, TenBitExpFP::from(1.0 / size));
    let pt_params = AvgPoolParams::new(pool_h, pool_w, stride, TenBitExpFP::from(1.0 / size));
    let pool_dims = LayerDims {
        input_dims,
        output_dims: layer_params.calculate_output_size(input_dims),
    };
    let layer = LinearLayer::AvgPool {
        dims: pool_dims,
        params: layer_params,
    };
    let pt_layer = LinearLayer::AvgPool {
        dims: pool_dims,
        params: pt_params,
    };
    (layer, pt_layer)
}

fn add_activation_layer<P, Q>(nn: &mut NeuralNetwork<P, Q>) {
    let cur_input_dims = nn.layers.last().as_ref().unwrap().output_dimensions();
    let layer_dims = LayerDims {
        input_dims: cur_input_dims,
        output_dims: cur_input_dims,
    };
    let layer = Layer::NLL(NonLinearLayer::ReLU {
        dims: layer_dims,
        _f: std::marker::PhantomData,
        _c: std::marker::PhantomData,
    });
    nn.layers.push(layer);
}

mod gc {
    use super::*;
    use crate::gc::*;
    use protocols_sys::*;

    #[test]
    fn test_relu() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let num_relus = 1;

        let mac_key_a = F::uniform(&mut rng);
        let mac_key_b = F::uniform(&mut rng);

        let mut plain_x_s = Vec::with_capacity(num_relus);
        let mut plain_results = Vec::with_capacity(num_relus);
        // Server shares
        let mut server_output_shares = Vec::with_capacity(num_relus);
        let mut server_output_mac_shares = Vec::with_capacity(num_relus);
        let mut server_input_mac_shares = Vec::with_capacity(num_relus);
        // Client shares
        let mut client_output_shares = Vec::with_capacity(num_relus);
        let mut client_inputs = Vec::with_capacity(num_relus);
        let mut client_output_mac_shares = Vec::with_capacity(num_relus);
        let mut client_input_mac_shares = Vec::with_capacity(num_relus);
        for _ in 0..num_relus {
            let (f1, n1) = generate_random_number(&mut rng);
            plain_x_s.push(n1);
            let f2 = if f1 < 0.0 {
                0.0
            } else if f1 > 6.0 {
                6.0
            } else {
                f1
            };
            let n2 = TenBitExpFP::from(f2);
            plain_results.push(n2);

            // Parties have shares of ReLU input
            let (s11, s12) = n1.share(&mut rng);

            // Parties hold authenticated shares of client's output share
            let client_output_mac_share = F::uniform(&mut rng);
            let server_output_mac_share = (s12.inner.inner * mac_key_a) - client_output_mac_share;
            server_output_shares.push(s11);
            client_output_shares.push(s12.inner.inner);
            server_output_mac_shares.push(server_output_mac_share);
            client_output_mac_shares.push(client_output_mac_share);

            // Parties hold authenticated share of client's input
            let client_input = F::uniform(&mut rng);
            let client_input_mac_share = F::uniform(&mut rng);
            let server_input_mac_share = (client_input * mac_key_b) - client_input_mac_share;
            client_inputs.push(client_input);
            client_input_mac_shares.push(client_input_mac_share);
            server_input_mac_shares.push(server_input_mac_share);
        }

        let server_addr = "127.0.0.1:8001";
        let client_addr = "127.0.0.1:8002";
        let server_listener = TcpListener::bind(server_addr).unwrap();

        let (server_offline, client_offline) = crossbeam::thread::scope(|s| {
            let server_offline_result = s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);

                for stream in server_listener.incoming() {
                    let stream = stream.expect("Client connection failed!");
                    let mut write_stream = IMuxSync::new(vec![BufWriter::new(&stream)]);
                    let mut read_stream = IMuxSync::new(vec![BufReader::new(&stream)]);
                    // Keygen
                    let key_time = timer_start!(|| "Receiving keys");
                    let keys: ServerKeyRcv = crate::bytes::deserialize(&mut read_stream).unwrap();
                    let mut key_share = KeyShare::new();
                    let sfhe = key_share.receive(keys.msg());
                    timer_end!(key_time);

                    return ReluProtocol::<TenBitExpParams>::offline_server_protocol(
                        &mut read_stream,
                        &mut write_stream,
                        num_relus,
                        &sfhe,
                        vec![num_relus].as_slice(),
                        vec![mac_key_a].as_slice(),
                        server_output_mac_shares.as_slice(),
                        vec![0].as_slice(),
                        vec![mac_key_b].as_slice(),
                        server_input_mac_shares.as_slice(),
                        &mut rng,
                    );
                }
                unreachable!("we should never exit server's loop")
            });

            let client_offline_result = s.spawn(|_| {
                // client's connection to server.
                let stream = TcpStream::connect(server_addr).unwrap();
                let mut write_stream = IMuxSync::new(vec![BufWriter::new(&stream)]);
                let mut read_stream = IMuxSync::new(vec![BufReader::new(&stream)]);

                let mut rng = ChaChaRng::from_seed(RANDOMNESS);

                // Keygen
                let key_time = timer_start!(|| "Client Keygen");
                let mut key_share = KeyShare::new();
                let (cfhe, keys_vec) = key_share.generate();
                let sent_message = ClientKeySend::new(&keys_vec);
                crate::bytes::serialize(&mut write_stream, &sent_message).unwrap();
                timer_end!(key_time);

                return ReluProtocol::<TenBitExpParams>::offline_client_protocol(
                    &mut read_stream,
                    &mut write_stream,
                    num_relus,
                    &cfhe,
                    vec![num_relus].as_slice(),
                    client_output_mac_shares.as_slice(),
                    client_output_shares.as_slice(),
                    client_input_mac_shares.as_slice(),
                    client_inputs.as_slice(),
                    &mut rng,
                );
            });
            (
                server_offline_result.join().unwrap().unwrap(),
                client_offline_result.join().unwrap().unwrap(),
            )
        })
        .unwrap();

        let client_listener = TcpListener::bind(client_addr).unwrap();
        let server_online = crossbeam::thread::scope(|s| {
            // Start thread for client.
            let _ = s.spawn(|_| {
                let gc_s = &client_offline.gc_s;
                let server_labels = &client_offline.server_randomizer_labels;
                let client_labels = &client_offline.client_input_labels;
                for stream in client_listener.incoming() {
                    let stream = stream.expect("Client connection failed!");
                    let mut write_stream = IMuxSync::new(vec![BufWriter::new(&stream)]);
                    let mut read_stream = IMuxSync::new(vec![BufReader::new(&stream)]);
                    return ReluProtocol::<TenBitExpParams>::online_client_protocol(
                        &mut read_stream,
                        &mut write_stream,
                        num_relus,
                        0,
                        &server_labels,
                        &client_labels,
                        &gc_s,
                    );
                }
                unreachable!("we should never reach here")
            });

            // Start thread for the server to make a connection.
            let result = s
                .spawn(|_| {
                    let stream = TcpStream::connect(client_addr).unwrap();
                    let mut write_stream = IMuxSync::new(vec![BufWriter::new(&stream)]);
                    let mut read_stream = IMuxSync::new(vec![BufReader::new(&stream)]);

                    ReluProtocol::online_server_protocol(
                        &mut read_stream,
                        &mut write_stream,
                        server_output_shares.as_slice(),
                        &server_offline.encoders,
                    )
                })
                .join()
                .unwrap();

            result
        })
        .unwrap()
        .unwrap();
        for i in 0..num_relus {
            let randomizer = server_offline.output_randomizers[i] - client_inputs[i];
            let server_share = TenBitExpFP::randomize_local_share(&server_online[i], &randomizer);
            let result = plain_results[i];
            assert_eq!(server_share.inner, result);
        }
    }
}

mod linear {
    use super::*;
    use crypto_primitives::AuthShare;
    use ndarray::s;
    use neural_network::Evaluate;
    use protocols_sys::*;

    // Evaluates given `layer` on `input` and outputs client and server
    // output shares
    fn eval_linear_layer(
        server_addr: &str,
        layer: &LinearLayer<TenBitAS, TenBitExpFP>,
        layer_info: &LinearLayerInfo<TenBitAS, TenBitExpFP>,
        input: &Input<TenBitExpFP>,
    ) -> (Output<TenBitAS>, Output<TenBitAS>) {
        let input_dims = layer.input_dimensions();
        let output_dims = layer.output_dimensions();
        let server_listener = TcpListener::bind(server_addr).unwrap();

        // Run the offline phase for both the client and server
        let (client_randomizers, (server_randomizers, server_mac_keys)) =
            crossbeam::thread::scope(|s| {
                let server_offline_result = s.spawn(|_| {
                    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                    for stream in server_listener.incoming() {
                        let stream = stream.expect("server connection failed!");
                        let mut reader = IMuxSync::new(vec![BufReader::new(&stream)]);
                        let mut writer = IMuxSync::new(vec![BufWriter::new(&stream)]);

                        // Keygen
                        let key_time = timer_start!(|| "Receiving keys");

                        let keys: ServerKeyRcv = crate::bytes::deserialize(&mut reader).unwrap();
                        let mut key_share = KeyShare::new();
                        let sfhe = key_share.receive(keys.msg());

                        timer_end!(key_time);
                        let acg_time = timer_start!(|| "Server ACG");

                        // Select the correct acg protocol
                        let mut server_acg = match layer {
                            LinearLayer::Conv2d { .. } => SealServerACG::Conv2D(
                                server_acg::Conv2D::new(&sfhe, layer, &layer.kernel_to_repr()),
                            ),
                            LinearLayer::FullyConnected { .. } => {
                                SealServerACG::FullyConnected(server_acg::FullyConnected::new(
                                    &sfhe,
                                    layer,
                                    &layer.kernel_to_repr(),
                                ))
                            }
                            _ => unreachable!(),
                        };
                        let result = LinearProtocol::<TenBitExpParams>::offline_server_acg_protocol(
                            &mut reader,
                            &mut writer,
                            layer.input_dimensions(),
                            layer.output_dimensions(),
                            &mut server_acg,
                            &mut rng,
                        );
                        timer_end!(acg_time);
                        return result.unwrap();
                    }
                    unreachable!("we should never exit server's loop")
                });

                let client_offline_result = s.spawn(|_| {
                    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

                    // Client's connection to server.
                    let write_stream =
                        TcpStream::connect(server_addr).expect("connecting to server failed");
                    let read_stream = write_stream.try_clone().unwrap();
                    let mut reader = IMuxSync::new(vec![BufReader::new(&read_stream)]);
                    let mut writer = IMuxSync::new(vec![BufWriter::new(&write_stream)]);

                    // Keygen
                    let key_time = timer_start!(|| "Client Keygen");

                    let mut key_share = KeyShare::new();
                    let (cfhe, keys_vec) = key_share.generate();
                    let sent_message = ClientKeySend::new(&keys_vec);
                    crate::bytes::serialize(&mut writer, &sent_message).unwrap();

                    timer_end!(key_time);
                    let acg_time = timer_start!(|| "Client ACG");

                    // Select the correct acg protocol
                    let mut acg_handler = match layer_info {
                        LinearLayerInfo::Conv2d { .. } => SealClientACG::Conv2D(
                            client_acg::Conv2D::new(&cfhe, layer_info, input_dims, output_dims),
                        ),
                        LinearLayerInfo::FullyConnected => {
                            SealClientACG::FullyConnected(client_acg::FullyConnected::new(
                                &cfhe,
                                layer_info,
                                input_dims,
                                output_dims,
                            ))
                        }
                        _ => unreachable!(),
                    };
                    let result = LinearProtocol::<TenBitExpParams>::offline_client_acg_protocol(
                        &mut reader,
                        &mut writer,
                        layer.input_dimensions(),
                        layer.output_dimensions(),
                        &mut acg_handler,
                        &mut rng,
                    );

                    timer_end!(acg_time);
                    return result.unwrap();
                });
                (
                    client_offline_result.join().unwrap(),
                    server_offline_result.join().unwrap(),
                )
            })
            .unwrap();

        // Check MACS
        client_randomizers
            .0
            .iter()
            .zip(server_randomizers.0.iter())
            .for_each(|(s1, s2)| {
                let combined = s1 + s2;
                assert!(
                    AuthShare::open(combined, &server_mac_keys.0).is_ok(),
                    "Invalid r MAC"
                );
            });

        client_randomizers
            .1
            .iter()
            .zip(server_randomizers.2.iter())
            .for_each(|(s1, s2)| {
                let combined = s1 + s2;
                assert!(
                    AuthShare::open(combined, &server_mac_keys.1).is_ok(),
                    "Invalid y MAC"
                );
            });

        // Convert shares
        let client_layer_share: Vec<TenBitAS> = client_randomizers
            .1
            .into_iter()
            .map(|e| FixedPoint::with_num_muls(e.get_value().inner, 1).into())
            .collect();
        let client_layer_share = Input::from_shape_vec(output_dims, client_layer_share).unwrap();

        let online_time = timer_start!(|| "Online Phase");

        // Share the input for layer `1`, computing
        // server_share = x + r.
        // client_share = -r;
        let (server_input_share, _) =
            input.share_with_randomness(&Input::unwrap_auth_value(client_randomizers.0));

        // Run online phase for client and server, return server output share
        let server_layer_share = crossbeam::thread::scope(|s| {
            // Start thread for client.
            let result = s.spawn(|_| {
                let mut write_stream =
                    IMuxSync::new(vec![TcpStream::connect(server_addr).unwrap()]);
                LinearProtocol::online_client_protocol(
                    &mut write_stream,
                    &server_input_share,
                    layer_info,
                )
            });

            // Start thread for the server to make a connection.
            let server_result = s
                .spawn(|_| {
                    for stream in server_listener.incoming() {
                        let mut read_stream =
                            IMuxSync::new(vec![stream.expect("server connection failed!")]);
                        let mut output = Output::zeros(output_dims);
                        return LinearProtocol::online_server_protocol(
                            &mut read_stream,      // we only receive here, no messages to client
                            layer,                 // layer parameters
                            &server_randomizers.1, // this is our `s` from above.
                            &Input::zeros(input_dims),
                            &mut output, // this is where the result will go.
                        )
                        .map(|_| output);
                    }
                    unreachable!("Server should not exit loop");
                })
                .join()
                .unwrap()
                .unwrap();
            let _ = result.join();
            server_result
        })
        .unwrap();

        timer_end!(online_time);

        (client_layer_share, server_layer_share)
    }

    #[test]
    fn test_convolution() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        // Set the parameters for the convolution.
        // let input_dims = (1, 64, 8, 8);
        // let kernel_dims = (64, 64, 3, 3);
        let input_dims = (1, 1, 8, 8);
        let kernel_dims = (1, 1, 3, 3);
        let stride = 1;
        let padding = Padding::Same;

        // Sample the layer
        let (layer, pt_layer) =
            sample_conv_layer(input_dims, kernel_dims, stride, padding, &mut rng);
        let layer_info = (&layer).into();

        // Sample a random input.
        let mut input = Input::zeros(input_dims);
        input
            .iter_mut()
            .for_each(|in_i| *in_i = generate_random_image(&mut rng).1);

        // Evaluate convolution layer on plaintext, so that we can check results later.
        let output = pt_layer.evaluate(&input);

        // Evaluate convolution on actual layer
        let (client_share, server_share) =
            eval_linear_layer("127.0.0.1:8003", &layer, &layer_info, &input);

        // Debug results
        let mut result: Input<TenBitExpFP> = Input::zeros(layer.output_dimensions());
        result
            .iter_mut()
            .zip(client_share.iter())
            .zip(server_share.iter())
            .for_each(|((r, s1), s2)| {
                *r = (*s1).combine(s2);
            });

        println!("Result:");
        println!("DIM: {:?}", result.dim());
        let chan_size = result.dim().2 * result.dim().3;
        let row_size = result.dim().2;
        let mut success = true;
        result
            .slice(s![0, .., .., ..])
            .outer_iter()
            .zip(output.slice(s![0, .., .., ..]).outer_iter())
            .enumerate()
            .for_each(|(chan_idx, (res_c, out_c))| {
                println!("Channel {}: ", chan_idx);

                res_c
                    .outer_iter()
                    .zip(out_c.outer_iter())
                    .enumerate()
                    .for_each(|(inp_idx, (inp_r, inp_out))| {
                        println!("    Row {}: ", inp_idx);

                        inp_r
                            .iter()
                            .zip(inp_out.iter())
                            .enumerate()
                            .for_each(|(i, (r, out))| {
                                println!(
                                    "IDX {}:           {}        {}",
                                    i + inp_idx * row_size + chan_idx * chan_size,
                                    r,
                                    out
                                );
                                let delta = f64::from(*r) - f64::from(*out);
                                if delta.abs() > 0.5 {
                                    println!(
                                        "{:?}-th index failed {:?} {:?} {} {}",
                                        i,
                                        r.signed_reduce(),
                                        out.signed_reduce(),
                                        r,
                                        out
                                    );
                                    success = false;
                                }
                            });
                    });
            });
        assert!(success);
    }

    #[test]
    fn test_fully_connected() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        // Set the parameters for the layer
        let input_dims = (1, 3, 32, 32);
        let out_chn = 10;

        // Sample the layer
        let (layer, pt_layer) = sample_fc_layer(input_dims, out_chn, &mut rng);
        let layer_info = (&layer).into();

        // Sample a random input.
        let mut input = Input::zeros(input_dims);
        input
            .iter_mut()
            .for_each(|in_i| *in_i = generate_random_number(&mut rng).1);

        // Evaluate plaintext layer, so that we can check results later.
        let output = pt_layer.evaluate(&input);

        // Evaluate convolution on actual layer
        let (client_share, server_share) =
            eval_linear_layer("127.0.0.1:8004", &layer, &layer_info, &input);

        // Debug results
        let mut result: Input<TenBitExpFP> = Input::zeros(layer.output_dimensions());
        result
            .iter_mut()
            .zip(client_share.iter())
            .zip(server_share.iter())
            .for_each(|((r, s1), s2)| {
                *r = (*s1).combine(s2);
            });

        output
            .iter()
            .zip(result.iter())
            .enumerate()
            .for_each(|(i, (o, r))| {
                assert_eq!(o, r, "{:?}-th index failed", i);
            });
    }
}

mod network {
    use super::*;
    use crate::neural_network::{ClientState, NNProtocol, ServerState};
    use ndarray::s;
    use neural_network::{Evaluate, NeuralArchitecture, NeuralNetwork};

    // Evaluate given network
    fn eval_network(
        server_addr: &str,
        network: &NeuralNetwork<TenBitAS, TenBitExpFP>,
        architecture: &NeuralArchitecture<TenBitAS, TenBitExpFP>,
        input: &Input<TenBitExpFP>,
    ) -> (
        ClientState<TenBitExpParams>,
        ServerState<TenBitExpParams>,
        Output<TenBitExpFP>,
    ) {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        // Run the offline phase for both the client and server
        let (client_state, server_state) = crossbeam::thread::scope(|s| {
            let server_state = s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                let server_listener = TcpListener::bind(server_addr).unwrap();
                let stream = server_listener
                    .incoming()
                    .next()
                    .unwrap()
                    .expect("Server connection failed!");
                let mut write_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                let mut read_stream = IMuxSync::new(vec![BufReader::new(stream)]);
                NNProtocol::offline_server_protocol(
                    &mut read_stream,
                    &mut write_stream,
                    &network,
                    &mut rng,
                )
            });

            let client_state = s.spawn(|_| {
                // This is mainly for Valgrind debugging to not fail
                std::thread::sleep(std::time::Duration::new(1, 0));
                let stream = TcpStream::connect(server_addr).expect("Client connection failed!");
                let mut read_stream =
                    IMuxSync::new(vec![BufReader::new(stream.try_clone().unwrap())]);
                let mut write_stream = IMuxSync::new(vec![stream]);
                NNProtocol::offline_client_protocol(
                    &mut read_stream,
                    &mut write_stream,
                    &architecture,
                    &mut rng,
                )
            });
            (
                client_state.join().unwrap().unwrap(),
                server_state.join().unwrap().unwrap(),
            )
        })
        .unwrap();

        // Run online phase for client and server, return server output share
        let result = crossbeam::thread::scope(|s| {
            // Start thread for the server to make a connection.
            let server_thread = s.spawn(|_| {
                let server_listener = TcpListener::bind(server_addr).unwrap();
                let stream = server_listener
                    .incoming()
                    .next()
                    .unwrap()
                    .expect("server connection failed!");
                let mut write_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                let mut read_stream = IMuxSync::new(vec![BufReader::new(stream)]);
                NNProtocol::online_server_protocol(
                    &mut read_stream,
                    &mut write_stream,
                    &network,
                    &server_state,
                )
            });

            let result = s.spawn(|_| {
                // This is mainly for Valgrind debugging to not fail
                std::thread::sleep(std::time::Duration::new(1, 0));
                let stream = TcpStream::connect(server_addr).expect("Client connection failed!");
                let mut write_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                let mut read_stream = IMuxSync::new(vec![BufReader::new(stream)]);
                NNProtocol::online_client_protocol(
                    &mut read_stream,
                    &mut write_stream,
                    &input,
                    &architecture,
                    &client_state,
                )
            });
            server_thread.join().unwrap().unwrap();
            let result = result.join().unwrap();
            result
        })
        .unwrap()
        .unwrap();

        (client_state, server_state, result)
    }

    fn print_result(
        output: Output<TenBitExpFP>,
        result: Output<TenBitExpFP>,
        correct_delta: f64,
    ) -> bool {
        let row_size = output.dim().2;
        let chan_size = output.dim().2 * output.dim().3;
        let mut success = true;
        result
            .slice(s![0, .., .., ..])
            .outer_iter()
            .zip(output.slice(s![0, .., .., ..]).outer_iter())
            .enumerate()
            .for_each(|(chan_idx, (res_c, out_c))| {
                println!("Channel {}: ", chan_idx);

                res_c
                    .outer_iter()
                    .zip(out_c.outer_iter())
                    .enumerate()
                    .for_each(|(inp_idx, (inp_r, inp_out))| {
                        println!("    Row {}: ", inp_idx);

                        inp_r
                            .iter()
                            .zip(inp_out.iter())
                            .enumerate()
                            .for_each(|(i, (r, out))| {
                                println!(
                                    "IDX {}:           {}        {}",
                                    i + inp_idx * row_size + chan_idx * chan_size,
                                    r,
                                    out
                                );
                                let delta = f64::from(*r) - f64::from(*out);
                                if delta.abs() > correct_delta {
                                    println!(
                                        "{:?}-th index failed {:?} {:?} {} {}",
                                        i,
                                        r.signed_reduce(),
                                        out.signed_reduce(),
                                        r,
                                        out
                                    );
                                    success = false;
                                }
                            });
                    });
            });
        success
    }

    #[test]
    fn test_conv_iden() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        // Build networks
        let mut network = NeuralNetwork {
            layers: vec![],
            ..Default::default()
        };
        let mut pt_network = NeuralNetwork {
            layers: vec![],
            ..Default::default()
        };

        // Dimensions of input image.
        let input_dims = (1, 3, 32, 32);

        // Conv Layer
        let kernel_dims = (1, 3, 3, 3);
        let (conv, pt_conv) =
            sample_conv_layer(input_dims, kernel_dims, 1, Padding::Same, &mut rng);
        network.layers.push(Layer::LL(conv));
        pt_network.layers.push(Layer::LL(pt_conv));

        // Identity Layer
        let next_input_dims = network.layers.last().unwrap().output_dimensions();
        let (iden, pt_iden) = sample_iden_layer(next_input_dims);
        network.layers.push(Layer::LL(iden));
        pt_network.layers.push(Layer::LL(pt_iden));

        assert!(network.validate());
        assert!(pt_network.validate());

        let architecture = (&network).into();

        // Sample a random input.
        let mut input = Input::zeros(input_dims);
        input
            .iter_mut()
            .for_each(|in_i| *in_i = generate_random_image(&mut rng).1);

        // Evaluate networks
        let output = pt_network.evaluate(&input);
        let (_, _, result) = eval_network("127.0.0.1:8006", &network, &architecture, &input);

        // Print and verify output
        assert!(print_result(output, result, 0.5));
    }

    #[test]
    fn test_conv_avgpool() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        // Build networks
        let mut network = NeuralNetwork {
            layers: vec![],
            ..Default::default()
        };
        let mut pt_network = NeuralNetwork {
            layers: vec![],
            ..Default::default()
        };

        // Dimensions of input image.
        let input_dims = (1, 1, 32, 32);

        // Conv Layer
        let kernel_dims = (1, 1, 3, 3);
        let (conv, pt_conv) =
            sample_conv_layer(input_dims, kernel_dims, 1, Padding::Same, &mut rng);
        network.layers.push(Layer::LL(conv));
        pt_network.layers.push(Layer::LL(pt_conv));

        // AvgPool Layer
        let next_input_dims = network.layers.last().unwrap().output_dimensions();
        let (avg, pt_avg) = sample_avg_pool_layer(next_input_dims, (2, 2), 2);
        network.layers.push(Layer::LL(avg));
        pt_network.layers.push(Layer::LL(pt_avg));

        assert!(network.validate());
        assert!(pt_network.validate());

        let architecture = (&network).into();

        // Sample a random input.
        let mut input = Input::zeros(input_dims);
        input
            .iter_mut()
            .for_each(|in_i| *in_i = generate_random_image(&mut rng).1);

        // Evaluate networks
        let output = pt_network.evaluate(&input);
        let (_, _, result) = eval_network("127.0.0.1:8007", &network, &architecture, &input);

        // Print and verify output
        assert!(print_result(output, result, 0.5));
    }

    #[test]
    fn test_conv_relu_fc() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        // Build networks
        let mut network = NeuralNetwork {
            layers: vec![],
            ..Default::default()
        };
        let mut pt_network = NeuralNetwork {
            layers: vec![],
            ..Default::default()
        };

        // Dimensions of input image.
        let input_dims = (1, 3, 32, 32);

        // Conv Layer
        let kernel_dims = (3, 3, 3, 3);
        let (conv, pt_conv) =
            sample_conv_layer(input_dims, kernel_dims, 1, Padding::Same, &mut rng);
        network.layers.push(Layer::LL(conv));
        pt_network.layers.push(Layer::LL(pt_conv));

        // ReLU Layer
        add_activation_layer(&mut network);
        add_activation_layer(&mut pt_network);

        // FC Layer
        let next_input_dims = network.layers.last().unwrap().output_dimensions();
        let (fc, pt_fc) = sample_fc_layer(next_input_dims, 10, &mut rng);
        network.layers.push(Layer::LL(fc));
        pt_network.layers.push(Layer::LL(pt_fc));

        assert!(network.validate());
        assert!(pt_network.validate());

        let architecture = (&network).into();

        // Sample a random input.
        let mut input = Input::zeros(input_dims);
        input
            .iter_mut()
            .for_each(|in_i| *in_i = generate_random_image(&mut rng).1);

        // Evaluate networks
        let output = pt_network.evaluate(&input);
        let (_, _, result) = eval_network("127.0.0.1:8008", &network, &architecture, &input);

        // Print and verify output
        assert!(print_result(output, result, 0.5));
    }

    #[test]
    fn test_conv_avg_relu_iden_fc() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        // Build networks
        let mut network = NeuralNetwork {
            layers: vec![],
            ..Default::default()
        };
        let mut pt_network = NeuralNetwork {
            layers: vec![],
            ..Default::default()
        };

        // Dimensions of input image.
        let input_dims = (1, 3, 32, 32);

        // Conv Layer
        let kernel_dims = (3, 3, 3, 3);
        let (conv, pt_conv) =
            sample_conv_layer(input_dims, kernel_dims, 1, Padding::Same, &mut rng);
        network.layers.push(Layer::LL(conv));
        pt_network.layers.push(Layer::LL(pt_conv));

        // AvgPool Layer
        let next_input_dims = network.layers.last().unwrap().output_dimensions();
        let (avg, pt_avg) = sample_avg_pool_layer(next_input_dims, (2, 2), 2);
        network.layers.push(Layer::LL(avg));
        pt_network.layers.push(Layer::LL(pt_avg));

        // ReLU Layer
        add_activation_layer(&mut network);
        add_activation_layer(&mut pt_network);

        // Iden Layer
        let next_input_dims = network.layers.last().unwrap().output_dimensions();
        let (iden, pt_iden) = sample_iden_layer(next_input_dims);
        network.layers.push(Layer::LL(iden));
        pt_network.layers.push(Layer::LL(pt_iden));

        // FC Layer
        let next_input_dims = network.layers.last().unwrap().output_dimensions();
        let (fc, pt_fc) = sample_fc_layer(next_input_dims, 10, &mut rng);
        network.layers.push(Layer::LL(fc));
        pt_network.layers.push(Layer::LL(pt_fc));

        assert!(network.validate());
        assert!(pt_network.validate());

        let architecture = (&network).into();

        // Sample a random input.
        let mut input = Input::zeros(input_dims);
        input
            .iter_mut()
            .for_each(|in_i| *in_i = generate_random_image(&mut rng).1);

        // Evaluate networks
        let output = pt_network.evaluate(&input);
        let (_, _, result) = eval_network("127.0.0.1:8009", &network, &architecture, &input);

        // Print and verify output
        // The delta can be higher since this is a deeper network
        assert!(print_result(output, result, 10.0));
    }

    #[test]
    fn test_mnist() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        // Build networks
        let mut network = NeuralNetwork {
            layers: vec![],
            ..Default::default()
        };
        let mut pt_network = NeuralNetwork {
            layers: vec![],
            ..Default::default()
        };

        // Dimensions of input image.
        let input_dims = (1, 1, 28, 28);

        // Conv Layer
        let kernel_dims = (16, 1, 5, 5);
        let (conv, pt_conv) =
            sample_conv_layer(input_dims, kernel_dims, 1, Padding::Same, &mut rng);
        network.layers.push(Layer::LL(conv));
        pt_network.layers.push(Layer::LL(pt_conv));

        // ReLU Layer
        add_activation_layer(&mut network);
        add_activation_layer(&mut pt_network);

        // AvgPool Layer
        let next_input_dims = network.layers.last().unwrap().output_dimensions();
        let (avg, pt_avg) = sample_avg_pool_layer(next_input_dims, (2, 2), 2);
        network.layers.push(Layer::LL(avg));
        pt_network.layers.push(Layer::LL(pt_avg));

        // Conv Layer
        let next_input_dims = network.layers.last().unwrap().output_dimensions();
        let kernel_dims = (16, 16, 5, 5);
        let (conv, pt_conv) =
            sample_conv_layer(next_input_dims, kernel_dims, 1, Padding::Valid, &mut rng);
        network.layers.push(Layer::LL(conv));
        pt_network.layers.push(Layer::LL(pt_conv));

        // ReLU Layer
        add_activation_layer(&mut network);
        add_activation_layer(&mut pt_network);

        // AvgPool Layer
        let next_input_dims = network.layers.last().unwrap().output_dimensions();
        let (avg, pt_avg) = sample_avg_pool_layer(next_input_dims, (2, 2), 2);
        network.layers.push(Layer::LL(avg));
        pt_network.layers.push(Layer::LL(pt_avg));

        // FC Layer
        let next_input_dims = network.layers.last().unwrap().output_dimensions();
        let (fc, pt_fc) = sample_fc_layer(next_input_dims, 100, &mut rng);
        network.layers.push(Layer::LL(fc));
        pt_network.layers.push(Layer::LL(pt_fc));

        // ReLU Layer
        add_activation_layer(&mut network);
        add_activation_layer(&mut pt_network);

        // FC Layer
        let next_input_dims = network.layers.last().unwrap().output_dimensions();
        let (fc, pt_fc) = sample_fc_layer(next_input_dims, 10, &mut rng);
        network.layers.push(Layer::LL(fc));
        pt_network.layers.push(Layer::LL(pt_fc));

        assert!(network.validate());
        assert!(pt_network.validate());

        let architecture = (&network).into();

        // Sample a random input.
        let mut input = Input::zeros(input_dims);
        input
            .iter_mut()
            .for_each(|in_i| *in_i = generate_random_image(&mut rng).1);

        // Evaluate networks
        let output = pt_network.evaluate(&input);
        let (_, _, result) = eval_network("127.0.0.1:8010", &network, &architecture, &input);

        // Print and verify output
        // The delta can be higher since this is a deeper network
        assert!(print_result(output, result, 200.0));
    }
}
