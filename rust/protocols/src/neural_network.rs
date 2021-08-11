use crate::{bytes, error::MpcError, AdditiveShare, AuthAdditiveShare, InMessage, OutMessage};
use bench_utils::{timer_end, timer_start};
use neural_network::{
    layers::{Layer, LayerInfo, NonLinearLayer, NonLinearLayerInfo},
    NeuralArchitecture, NeuralNetwork,
};

use async_std::io::{Read, Write};
use rand::{CryptoRng, RngCore};
use std::{
    collections::BTreeMap,
    marker::PhantomData,
    sync::{Arc, Condvar, Mutex},
};

use algebra::{
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::{Fp64, Fp64Parameters},
    PrimeField, UniformRandom,
};

use neural_network::{
    layers::*,
    tensors::{Input, Output},
};

use crypto_primitives::{
    gc::fancy_garbling::{Encoder, GarbledCircuit, Wire},
    AuthShare, Share,
};

use crate::{cds::CDSProtocol, gc::ReluProtocol, linear_layer::LinearProtocol, mpc_offline::*};

use io_utils::imux::IMuxAsync;
use protocols_sys::{
    client_acg, server_acg, ClientACG, ClientFHE, SealClientACG, SealServerACG, ServerACG,
    ServerFHE,
};

use rayon::ThreadPoolBuilder;

pub struct NNProtocol<P: FixedPointParameters> {
    _share: PhantomData<P>,
}

pub const CLIENT: usize = 1;
pub const SERVER: usize = 2;

pub struct ServerState<P: FixedPointParameters>
where
    P::Field: AuthShare,
{
    pub linear_randomizers: BTreeMap<usize, (Input<P::Field>, Output<P::Field>)>,
    pub relu_encoders: Vec<Encoder>,
    pub relu_output_randomizers: Vec<P::Field>,
}

// TODO: Add Online phase MACs
pub struct ClientState<P: FixedPointParameters>
where
    P::Field: AuthShare,
{
    pub relu_circuits: Vec<GarbledCircuit>,
    pub relu_server_labels: Vec<Vec<Wire>>,
    pub relu_client_labels: Vec<Vec<Wire>>,
    /// Randomizers for the input of each linear layer
    pub linear_randomizers: BTreeMap<usize, Input<P::Field>>,
    /// Shares of the output of each linear layer
    pub linear_shares: BTreeMap<usize, Output<AdditiveShare<P>>>,
}

pub struct NNProtocolType;
// The final message from the server to the client, contains a share of the
// output.
pub type MsgRcv<P> = InMessage<Output<AdditiveShare<P>>, NNProtocolType>;
pub type MsgSend<'a, P> = OutMessage<'a, Output<AdditiveShare<P>>, NNProtocolType>;

impl<P: FixedPointParameters, F: Fp64Parameters> NNProtocol<P>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P: FixedPointParameters<Field = Fp64<F>>,
    P::Field: Share<
        Constant = <P as FixedPointParameters>::Field,
        Ring = <P as FixedPointParameters>::Field,
    >,
    P::Field: AuthShare,
{
    /// TODO
    pub fn offline_server_acg<
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: CryptoRng + RngCore,
    >(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        sfhe: &ServerFHE,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    ) -> Result<
        (
            BTreeMap<
                usize,
                (
                    Input<AuthAdditiveShare<P::Field>>,
                    Output<P::Field>,
                    Output<AuthAdditiveShare<P::Field>>,
                ),
            >,
            BTreeMap<usize, (P::Field, P::Field)>,
        ),
        MpcError,
    > {
        let mut linear_shares: BTreeMap<
            usize,
            (
                Input<AuthAdditiveShare<P::Field>>,
                Output<P::Field>,
                Output<AuthAdditiveShare<P::Field>>,
            ),
        > = BTreeMap::new();
        let mut mac_keys: BTreeMap<usize, (P::Field, P::Field)> = BTreeMap::new();

        let linear_time = timer_start!(|| "Linear layers offline phase");
        for (i, layer) in neural_network.layers.iter().enumerate() {
            match layer {
                Layer::NLL(NonLinearLayer::ReLU { .. }) => {}
                Layer::LL(layer) => {
                    let (shares, keys) = match &layer {
                        LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                            let mut acg_handler = match &layer {
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
                            LinearProtocol::<P>::offline_server_acg_protocol(
                                reader,
                                writer,
                                layer.input_dimensions(),
                                layer.output_dimensions(),
                                &mut acg_handler,
                                rng,
                            )?
                        }
                        LinearLayer::AvgPool { dims, .. } | LinearLayer::Identity { dims } => {
                            let in_zero = Output::zeros(dims.input_dimensions());
                            if linear_shares.keys().any(|k| k == &(i - 1)) {
                                // If the layer comes after a linear layer, apply the function to
                                // the last layer's output share MAC
                                let prev_mac_keys = mac_keys.get(&(i - 1)).unwrap();
                                let prev_output_share = &linear_shares.get(&(i - 1)).unwrap().2;
                                let mut output_share = Output::zeros(dims.output_dimensions());
                                layer.evaluate_naive_auth(prev_output_share, &mut output_share);
                                (
                                    (
                                        Input::auth_share_from_parts(
                                            in_zero.clone(),
                                            in_zero.clone(),
                                        ),
                                        Output::zeros(dims.output_dimensions()),
                                        output_share,
                                    ),
                                    prev_mac_keys.clone(),
                                )
                            } else {
                                // If the layer comes after a non-linear layer, receive the
                                // randomizer from the client, authenticate it, and then apply the
                                // function to the MAC share
                                let (key, input_share) =
                                    LinearProtocol::<P>::offline_server_auth_share(
                                        reader,
                                        writer,
                                        dims.input_dimensions(),
                                        &sfhe,
                                        rng,
                                    )
                                    .unwrap();
                                let mut output_share = Output::zeros(dims.output_dimensions());
                                layer.evaluate_naive_auth(&input_share, &mut output_share);
                                (
                                    (
                                        -input_share,
                                        Output::zeros(dims.output_dimensions()),
                                        output_share,
                                    ),
                                    (key, key),
                                )
                            }
                        }
                    };
                    linear_shares.insert(i, shares);
                    mac_keys.insert(i, keys);
                }
            }
        }
        timer_end!(linear_time);
        Ok((linear_shares, mac_keys))
    }

    pub fn offline_server_protocol<
        R: Read + Send + Unpin + 'static,
        W: Write + Send + Unpin + 'static,
        RNG: CryptoRng + RngCore + Send,
    >(
        mut reader: IMuxAsync<R>,
        mut writer: IMuxAsync<W>,
        mut reader_2: IMuxAsync<R>,
        mut writer_2: IMuxAsync<W>,
        mut writer_3: IMuxAsync<W>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
        rng_2: &mut RNG,
        rng_3: &mut RNG,
    ) -> Result<ServerState<P>, MpcError> {
        let sfhe: ServerFHE = crate::server_keygen(&mut reader)?;

        let start_time = timer_start!(|| "Server offline phase");

        // TODO
        let mut num_relu = 0;
        let mut num_truncations = BTreeMap::new();
        let mut output_truncations = Vec::new();
        let mut relu_layers = Vec::new();
        let mut relu_layer_sizes = Vec::new();
        for (i, layer) in neural_network.layers.iter().enumerate() {
            match layer {
                Layer::NLL(NonLinearLayer::ReLU { dims, .. }) => {
                    relu_layers.push(i);
                    let (b, c, h, w) = dims.input_dimensions();
                    let relus = b * c * h * w;
                    num_relu += relus;

                    // TODO
                    relu_layer_sizes.push(relus);
                    output_truncations.push(*num_truncations.get(&(i - 1)).unwrap());
                }
                Layer::LL(_layer) => {
                    // Keep track of the number of truncations needed for the output
                    // shares of each linear layer
                    let mut truncations = 0;
                    // If linear layer is preceded by a linear layer, add a truncation
                    if i != 0 && neural_network.layers[i - 1].is_linear() {
                        truncations += 1;
                    }
                    // If linear layer is followed by a non-linear layer, add a truncation
                    if i != neural_network.layers.len() - 1
                        && neural_network.layers[i + 1].is_non_linear()
                    {
                        truncations += 1;
                    }
                    num_truncations.insert(i, truncations);
                }
            }
        }

        let modulus_bits = <P::Field as PrimeField>::size_in_bits();
        let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;
        let (num_rands, num_triples) = CDSProtocol::<P>::num_rands_triples(
            relu_layers.len(),
            num_relu,
            modulus_bits,
            elems_per_label,
        );

        // TODO
        let mac_key = P::Field::uniform(rng);
        //let gen = crate::mpc_offline::InsecureServerOfflineMPC::new(&sfhe, mac_key.into_repr().0);
        let gen = crate::mpc_offline::ServerOfflineMPC::new(&sfhe, mac_key.into_repr().0);

        let mut linear_shares = BTreeMap::new();
        let mut mac_keys = BTreeMap::new();

        let triples = Arc::new((Mutex::new(Vec::new()), Condvar::new()));

        let mut labels = Vec::new();

        let mut gc_state = None;
        let _ = rayon::scope(|s| {
            s.spawn(|_| {
                let pool = ThreadPoolBuilder::new().num_threads(6).build().unwrap();
                pool.install(|| {
                    // Generate triples in batches
                    let batch_size = num_triples / 4;
                    let batches = (num_triples as f64 / batch_size as f64).ceil() as usize;
                    for i in 0..batches {
                        let triples_batch_size =
                            std::cmp::min(batch_size, num_triples - i * batch_size);
                        let mut triples_batch =
                            gen.triples_gen(&mut reader, &mut writer, rng, triples_batch_size);

                        // Add the batch to the vector of triples
                        let mut triples_vec = triples.0.lock().unwrap();
                        triples_vec.append(&mut triples_batch);
                        triples.1.notify_all();
                    }
                });
            });

            s.spawn(|_| {
                let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
                pool.install(|| {
                    let mut rands = Vec::new();
                    rayon::scope(|s2| {
                        // Generate input rands
                        rands = gen.rands_gen(&mut reader_2, &mut writer_2, rng_2, num_rands);
                        s2.spawn(|_| {
                            // ACG/Garbling
                            let result = NNProtocol::offline_server_acg(
                                &mut reader_2,
                                &mut writer_2,
                                &sfhe,
                                neural_network,
                                rng_2,
                            ).unwrap();
                            linear_shares = result.0;
                            mac_keys = result.1;
                        });

                        // TODO: Add timing stuff
                        let result = ReluProtocol::<P>::offline_server_garbling(
                            &mut writer_3,
                            num_relu,
                            &sfhe,
                            relu_layer_sizes.as_slice(),
                            output_truncations.as_slice(),
                            rng_3,
                        ).unwrap();
                        gc_state = Some(result.0);
                        labels = result.1;
                    });
                    // CDS
                    // Preprocessing for next step with ReLUs; if a ReLU is layer i,
                    // we want to take output mac shares for the (linear) layer i - 1,
                    // and input mac shares for the (linear) layer i + 1.
                    let mut output_mac_keys = Vec::new();
                    let mut output_mac_shares = Vec::new();
                    let mut input_mac_keys = Vec::new();
                    let mut input_mac_shares = Vec::new();
                    for &i in &relu_layers {
                        let output_share = &linear_shares
                            .get(&(i - 1))
                            .expect("should exist because every ReLU should be preceeded by a linear layer")
                            .2;
                        output_mac_keys.push(mac_keys.get(&(i - 1)).unwrap().1);
                        output_mac_shares.extend_from_slice(
                            Input::unwrap_auth_mac(output_share.clone())
                                .as_slice()
                                .unwrap(),
                        );

                        let input_share = &linear_shares
                            .get(&(i + 1))
                            .expect("should exist because every ReLU should be succeeded by a linear layer")
                            .0;
                        input_mac_keys.push(mac_keys.get(&(i + 1)).unwrap().0);
                        input_mac_shares.extend_from_slice(
                            Input::unwrap_auth_mac(input_share.clone())
                                .as_slice()
                                .unwrap(),
                        );
                    }

                    ReluProtocol::<P>::offline_server_cds(
                        reader_2,
                        writer_2,
                        writer_3,
                        &sfhe,
                        triples.clone(),
                        Arc::new(Mutex::new(rands)),
                        mac_key,
                        relu_layer_sizes.as_slice(),
                        output_mac_keys.as_slice(),
                        output_mac_shares.as_slice(),
                        input_mac_keys.as_slice(),
                        input_mac_shares.as_slice(),
                        labels,
                        rng_2,
                    ).unwrap();
                });
            });
        });
        let gc_state = gc_state.unwrap();

        // We no longer need the MACs so unwrap underlying values
        let linear_randomizers = linear_shares
            .into_iter()
            .map(|(k, v)| (k, (Input::unwrap_auth_value(v.0), v.1)))
            .collect();

        timer_end!(start_time);
        Ok(ServerState {
            linear_randomizers,
            relu_encoders: gc_state.encoders,
            relu_output_randomizers: gc_state.output_randomizers,
        })
    }

    pub fn offline_client_acg<
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: CryptoRng + RngCore,
    >(
        mut reader: &mut IMuxAsync<R>,
        mut writer: &mut IMuxAsync<W>,
        cfhe: &ClientFHE,
        neural_network_architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    ) -> Result<
        (
            BTreeMap<usize, Input<AuthAdditiveShare<P::Field>>>,
            BTreeMap<usize, Output<AuthAdditiveShare<P::Field>>>,
        ),
        MpcError,
    > {
        let mut in_shares = BTreeMap::new();
        let mut out_shares: BTreeMap<usize, Output<AuthAdditiveShare<P::Field>>> = BTreeMap::new();
        let linear_time = timer_start!(|| "Linear layers offline phase");
        for (i, layer) in neural_network_architecture.layers.iter().enumerate() {
            match layer {
                LayerInfo::NLL(_, NonLinearLayerInfo::ReLU { .. }) => {}
                LayerInfo::LL(dims, linear_layer_info) => {
                    let input_dims = dims.input_dimensions();
                    let output_dims = dims.output_dimensions();
                    let (in_share, out_share) = match &linear_layer_info {
                        LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                            let mut acg_handler = match &linear_layer_info {
                                LinearLayerInfo::Conv2d { .. } => {
                                    SealClientACG::Conv2D(client_acg::Conv2D::new(
                                        &cfhe,
                                        linear_layer_info,
                                        input_dims,
                                        output_dims,
                                    ))
                                }
                                LinearLayerInfo::FullyConnected => {
                                    SealClientACG::FullyConnected(client_acg::FullyConnected::new(
                                        &cfhe,
                                        linear_layer_info,
                                        input_dims,
                                        output_dims,
                                    ))
                                }
                                _ => unreachable!(),
                            };
                            LinearProtocol::<P>::offline_client_acg_protocol(
                                reader,
                                writer,
                                layer.input_dimensions(),
                                layer.output_dimensions(),
                                &mut acg_handler,
                                rng,
                            )?
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
                                randomizer
                                    .iter_mut()
                                    .for_each(|e| *e = P::Field::uniform(rng));
                                let randomizer = LinearProtocol::<P>::offline_client_auth_share(
                                    reader, writer, randomizer, &cfhe,
                                )
                                .unwrap();
                                linear_layer_info
                                    .evaluate_naive_auth(&randomizer, &mut output_share);
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
        Ok((in_shares, out_shares))
    }

    pub fn offline_client_protocol<
        R: Read + Send + Unpin + 'static,
        W: Write + Send + Unpin + 'static,
        RNG: RngCore + CryptoRng + Send,
    >(
        mut reader: IMuxAsync<R>,
        mut writer: IMuxAsync<W>,
        mut reader_2: IMuxAsync<R>,
        mut writer_2: IMuxAsync<W>,
        mut reader_3: IMuxAsync<R>,
        neural_network_architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
        rng_2: &mut RNG,
    ) -> Result<ClientState<P>, MpcError> {
        let cfhe: ClientFHE = crate::client_keygen(&mut writer)?;

        let start_time = timer_start!(|| "Client offline phase");

        // TODO
        let mut num_relu = 0;
        let mut relu_layers = Vec::new();
        let mut relu_layer_sizes = Vec::new();
        for (i, layer) in neural_network_architecture.layers.iter().enumerate() {
            match layer {
                LayerInfo::NLL(dims, NonLinearLayerInfo::ReLU { .. }) => {
                    relu_layers.push(i);
                    let (b, c, h, w) = dims.input_dimensions();
                    let relus = b * c * h * w;
                    relu_layer_sizes.push(relus);
                    num_relu += relus;
                }
                LayerInfo::LL(dims, linear_layer_info) => {}
            }
        }

        // TODO
        let modulus_bits = <P::Field as PrimeField>::size_in_bits();
        let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;
        let (num_rands, num_triples) = CDSProtocol::<P>::num_rands_triples(
            relu_layer_sizes.len(),
            num_relu,
            modulus_bits,
            elems_per_label,
        );

        // Generate rands and triples
        //let gen = InsecureClientOfflineMPC::new(&cfhe);
        let gen = ClientOfflineMPC::new(&cfhe);

        let mut in_shares = BTreeMap::new();
        let mut out_shares = BTreeMap::new();

        let triples = Arc::new((Mutex::new(Vec::new()), Condvar::new()));

        let mut gc_state = None;
        let _ = rayon::scope(|s| {
            s.spawn(|_| {
                let pool = ThreadPoolBuilder::new().num_threads(6).build().unwrap();
                pool.install(|| {
                    // Generate triples in batches
                    let batch_size = num_triples / 4;
                    let batches = (num_triples as f64 / batch_size as f64).ceil() as usize;
                    for i in 0..batches {
                        let triples_batch_size =
                            std::cmp::min(batch_size, num_triples - i * batch_size);
                        let mut triples_batch =
                            gen.triples_gen(&mut reader, &mut writer, rng, triples_batch_size);

                        // Add the batch to the vector of triples
                        let mut triples_vec = triples.0.lock().unwrap();
                        triples_vec.append(&mut triples_batch);
                        triples.1.notify_all();
                    }
                });
            });

            s.spawn(|_| {
                let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
                pool.install(|| {
                    let mut rands = Vec::new();
                    rayon::scope(|s2| {
                        // Generate input rands
                        rands = gen.rands_gen(&mut reader_2, &mut writer_2, rng_2, num_rands);

                        s2.spawn(|_| {
                            // ACG/Garbling
                            let result = NNProtocol::offline_client_acg(
                                &mut reader_2,
                                &mut writer_2,
                                &cfhe,
                                neural_network_architecture,
                                rng_2,
                            )
                            .unwrap();
                            in_shares = result.0;
                            out_shares = result.1;
                        });

                        // TODO
                        gc_state = Some(
                            ReluProtocol::<P>::offline_client_garbling(&mut reader_3, num_relu)
                                .unwrap(),
                        );
                    });

                    // Preprocessing for next step with ReLUs; if a ReLU is layer i,
                    // we want to take output shares for the (linear) layer i - 1,
                    // and input shares for the (linear) layer i + 1.
                    let mut output_shares = Vec::new();
                    let mut output_mac_shares = Vec::new();
                    let mut input_rands = Vec::new();
                    let mut input_mac_shares = Vec::new();
                    for &i in &relu_layers {
                        let output_share = out_shares.get(&(i - 1)).expect(
                            "should exist because every ReLU should be preceeded by a linear layer",
                        );
                        output_shares.extend_from_slice(
                            Input::unwrap_auth_value(output_share.clone())
                                .as_slice()
                                .unwrap(),
                        );
                        output_mac_shares.extend_from_slice(
                            Input::unwrap_auth_mac(output_share.clone())
                                .as_slice()
                                .unwrap(),
                        );

                        let input_rand = in_shares.get(&(i + 1)).expect(
                            "should exist because every ReLU should be succeeded by a linear layer",
                        );
                        input_rands.extend_from_slice(
                            Input::unwrap_auth_value(input_rand.clone())
                                .as_slice()
                                .unwrap(),
                        );
                        input_mac_shares.extend_from_slice(
                            Input::unwrap_auth_mac(input_rand.clone())
                                .as_slice()
                                .unwrap(),
                        );
                    }

                    // CDS
                    let gc_state = gc_state.as_mut().unwrap();
                    ReluProtocol::<P>::offline_client_cds(
                        reader_2,
                        writer_2,
                        reader_3,
                        &cfhe,
                        triples.clone(),
                        Arc::new(Mutex::new(rands)),
                        gc_state,
                        relu_layer_sizes.as_slice(),
                        output_mac_shares.as_slice(),
                        output_shares.as_slice(),
                        input_mac_shares.as_slice(),
                        input_rands.as_slice(),
                        rng_2,
                    )
                    .unwrap();
                });
            });
        });
        let mut gc_state = gc_state.unwrap();

        let crate::gc::ClientState {
            gc_s: relu_circuits,
            server_randomizer_labels: randomizer_labels,
            client_input_labels: relu_labels,
        } = gc_state;

        let (relu_client_labels, relu_server_labels) = if num_relu != 0 {
            let size_of_client_input = relu_labels.len() / num_relu;
            let size_of_server_input = randomizer_labels.len() / num_relu;

            assert_eq!(
                size_of_client_input,
                ReluProtocol::<P>::size_of_client_inputs(),
                "number of inputs unequal"
            );

            let client_labels = relu_labels
                .chunks(size_of_client_input)
                .map(|chunk| chunk.to_vec())
                .collect();
            let server_labels = randomizer_labels
                .chunks(size_of_server_input)
                .map(|chunk| chunk.to_vec())
                .collect();

            (client_labels, server_labels)
        } else {
            (vec![], vec![])
        };

        // We no longer need the MACs so unwrap underlying values
        let linear_randomizers: BTreeMap<_, _> = in_shares
            .into_iter()
            .map(|(k, v)| (k, Input::unwrap_auth_value(v)))
            .collect();

        let linear_shares: BTreeMap<_, _> =
            out_shares.into_iter().map(|(k, v)| (k, v.into())).collect();

        timer_end!(start_time);
        Ok(ClientState {
            relu_circuits,
            relu_server_labels,
            relu_client_labels,
            linear_randomizers,
            linear_shares,
        })
    }

    pub fn online_server_protocol<R: Read + Send + Unpin, W: Write + Send + Unpin>(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        state: &ServerState<P>,
    ) -> Result<(), MpcError> {
        let (first_layer_in_dims, first_layer_out_dims) = {
            let layer = neural_network.layers.first().unwrap();
            assert!(
                layer.is_linear(),
                "first layer of the network should always be linear."
            );
            (layer.input_dimensions(), layer.output_dimensions())
        };
        let mut num_consumed_relus = 0;

        let mut next_layer_input = Output::zeros(first_layer_out_dims);
        let mut next_layer_derandomizer = Input::zeros(first_layer_in_dims);
        let start_time = timer_start!(|| "Server online phase");
        for (i, layer) in neural_network.layers.iter().enumerate() {
            match layer {
                Layer::NLL(NonLinearLayer::ReLU { dims, .. }) => {
                    let start_time = timer_start!(|| "ReLU layer");
                    // Have the server encode the current input, via the garbled circuit,
                    // and then send the labels over to the other party.
                    let layer_size = next_layer_input.len();
                    assert_eq!(dims.input_dimensions(), next_layer_input.dim());
                    let layer_encoders =
                        &state.relu_encoders[num_consumed_relus..(num_consumed_relus + layer_size)];
                    // The server receives output of ReLU
                    let output = ReluProtocol::online_server_protocol(
                        reader,
                        writer,
                        &next_layer_input.as_slice().unwrap(),
                        layer_encoders,
                    )?;
                    next_layer_input = ndarray::Array1::from_iter(output)
                        .into_shape(dims.output_dimensions())
                        .expect("shape should be correct")
                        .into();
                    let relu_output_randomizers = state.relu_output_randomizers
                        [num_consumed_relus..(num_consumed_relus + layer_size)]
                        .to_vec();
                    num_consumed_relus += layer_size;
                    next_layer_derandomizer = ndarray::Array1::from_iter(relu_output_randomizers)
                        .into_shape(dims.output_dimensions())
                        .expect("shape should be correct")
                        .into();
                    timer_end!(start_time);
                }
                Layer::LL(layer) => {
                    let start_time = timer_start!(|| "Linear layer");
                    // Input for the next layer.
                    let layer_randomizer = &state.linear_randomizers.get(&i).unwrap().1;
                    if i != 0 {
                        next_layer_derandomizer
                            .iter_mut()
                            .zip(&next_layer_input)
                            .for_each(|(l_r, inp)| {
                                *l_r += &inp.inner.inner;
                            });
                    }
                    next_layer_input = Output::zeros(layer.output_dimensions());
                    LinearProtocol::online_server_protocol(
                        reader,
                        layer,
                        layer_randomizer,
                        &next_layer_derandomizer,
                        &mut next_layer_input,
                    )?;
                    next_layer_derandomizer = Output::zeros(layer.output_dimensions());
                    // Since linear operations involve multiplications
                    // by fixed-point constants, we want to truncate appropiately to
                    // ensure that we don't overflow.
                    // TODO: Removed as truncation is currently in garbled circuits
                    // if i != neural_network.layers.len() - 1 {
                    //    if i != 0 && neural_network.layers[i - 1].is_linear() {
                    //        // If the previous layer was linear we reduce twice
                    //        for share in next_layer_input.iter_mut() {
                    //            share.inner = FixedPoint::with_num_muls(share.inner.inner, 2);
                    //            share.inner.signed_reduce_in_place();
                    //        }
                    //    } else if !neural_network.layers[i + 1].is_linear() {
                    //        // If the next layer is non-linear, truncate
                    //        for share in next_layer_input.iter_mut() {
                    //            share.inner.signed_reduce_in_place();
                    //        }
                    //    }
                    //}
                    timer_end!(start_time);
                }
            }
        }

        // Open the final share to the client
        let sent_message = MsgSend::new(&next_layer_input);
        bytes::serialize(writer, &sent_message)?;

        timer_end!(start_time);
        Ok(())
    }

    /// Outputs shares for the next round's input.
    pub fn online_client_protocol<R: Read + Send + Unpin, W: Write + Send + Unpin>(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        input: &Input<FixedPoint<P>>,
        architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        state: &ClientState<P>,
    ) -> Result<Output<FixedPoint<P>>, MpcError> {
        let first_layer_in_dims = {
            let layer = architecture.layers.first().unwrap();
            assert!(
                layer.is_linear(),
                "first layer of the network should always be linear."
            );
            assert_eq!(layer.input_dimensions(), input.dim());
            layer.input_dimensions()
        };
        assert_eq!(first_layer_in_dims, input.dim());

        let mut num_consumed_relus = 0;
        let mut num_muls = 0;

        let start_time = timer_start!(|| "Client online phase");
        let (mut next_layer_input, _) =
            input.share_with_randomness(&(state.linear_randomizers[&0]));

        for (i, layer) in architecture.layers.iter().enumerate() {
            match layer {
                LayerInfo::NLL(dims, nll_info) => {
                    match nll_info {
                        NonLinearLayerInfo::ReLU { .. } => {
                            let start_time = timer_start!(|| "ReLU layer");
                            // The client receives the garbled circuits from the server,
                            // uses its already encoded inputs to get the next linear
                            // layer's input.
                            let layer_size = next_layer_input.len();
                            assert_eq!(dims.input_dimensions(), next_layer_input.dim());

                            let layer_client_labels = &state.relu_client_labels
                                [num_consumed_relus..(num_consumed_relus + layer_size)];
                            let layer_server_labels = &state.relu_server_labels
                                [num_consumed_relus..(num_consumed_relus + layer_size)];
                            let layer_circuits = &state.relu_circuits
                                [num_consumed_relus..(num_consumed_relus + layer_size)];

                            num_consumed_relus += layer_size;

                            let layer_client_labels = layer_client_labels
                                .into_iter()
                                .flat_map(|l| l.clone())
                                .collect::<Vec<_>>();
                            let layer_server_labels = layer_server_labels
                                .into_iter()
                                .flat_map(|l| l.clone())
                                .collect::<Vec<_>>();
                            ReluProtocol::<P>::online_client_protocol(
                                reader,
                                writer,
                                layer_size,           // num_relus
                                num_muls,             // number of truncations
                                &layer_server_labels, // Labels for layer
                                &layer_client_labels, // Labels for layer
                                &layer_circuits,      // circuits for layer.
                            )?;
                            next_layer_input = Output::zeros(dims.output_dimensions());
                            timer_end!(start_time);
                        }
                    };
                    num_muls = 0;
                }
                LayerInfo::LL(_, layer_info) => {
                    let start_time = timer_start!(|| "Linear layer");
                    // Send server secret share if required by the layer
                    let input = next_layer_input;
                    LinearProtocol::online_client_protocol(writer, &input, &layer_info)?;

                    // If this is not the last layer, and if the next layer
                    // is also linear, randomize the output correctly.
                    next_layer_input = state.linear_shares[&i].clone();
                    if i != (architecture.layers.len() - 1)
                        && architecture.layers[i + 1].is_linear()
                    {
                        next_layer_input.randomize_local_share(&state.linear_randomizers[&(i + 1)]);
                    }
                    // Add a multiplication for each linear layer
                    num_muls += 1;
                    timer_end!(start_time);
                }
            }
        }
        let result = bytes::deserialize(reader).map(|output: MsgRcv<P>| {
            // Receive server input and reset multiplication count to
            // avoid an early reduction
            let mut server_output_share = output.msg();
            server_output_share
                .iter_mut()
                .for_each(|e| e.inner = FixedPoint::new(e.inner.inner));
            let mut result = server_output_share.combine(&next_layer_input);
            // Reduce the output based on how many linear layers there have been
            // such last reduction
            result.iter_mut().for_each(|e| {
                *e = FixedPoint::with_num_muls(e.inner, num_muls).signed_reduce();
            });
            result
        })?;
        timer_end!(start_time);
        Ok(result)
    }
}
