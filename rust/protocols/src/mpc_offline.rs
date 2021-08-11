use crate::bytes;
use crate::{InMessage, OutMessage};
use algebra::{Fp64, Fp64Parameters, PrimeField, UniformRandom};
use crypto_primitives::{
    additive_share::{AdditiveShare, AuthAdditiveShare, AuthShare, Share},
    beavers_mul::Triple,
};
use io_utils::imux::IMuxAsync;
use itertools::izip;
use num_traits::Zero;
use protocols_sys::SealCT;
use protocols_sys::{ClientFHE, ClientGen, SealClientGen, SealServerGen, ServerFHE, ServerGen};
use rand::{ChaChaRng, CryptoRng, RngCore, SeedableRng};
use std::{
    cmp::min,
    marker::PhantomData,
    os::raw::c_char,
    sync::{Arc, Mutex},
};

// TODO
use async_std::{
    channel,
    io::{Read, Write},
    sync::RwLock,
    task,
};
use futures::stream::StreamExt;

pub struct OfflineMPCProtocolType;

type MsgSend<'a> = OutMessage<'a, (usize, Vec<c_char>), OfflineMPCProtocolType>;
type MsgRcv = InMessage<(usize, Vec<c_char>), OfflineMPCProtocolType>;

pub type ShareSend<'a, T> = OutMessage<'a, [AdditiveShare<T>], OfflineMPCProtocolType>;
pub type ShareRcv<T> = InMessage<Vec<AdditiveShare<T>>, OfflineMPCProtocolType>;

const RANDOMNESS: [u8; 32] = [
    0x99, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0x62, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

/// Represents a type which implements pairwise randomness and triple generation
/// for a client-malicious SPDZ-style MPC
pub trait OfflineMPC<T: AuthShare> {
    /// Message batch size
    const BATCH_SIZE: usize = 8192;

    /// Generates `num` authenticated pairwise randomness shares
    fn rands_gen<R: Read + Send + Unpin, W: Write + Send + Unpin, RNG: RngCore + CryptoRng>(
        &self,
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        rng: &mut RNG,
        num: usize,
    ) -> Vec<AuthAdditiveShare<T>>;

    /// Generates `num` authenticated triples
    fn triples_gen<R: Read + Send + Unpin, W: Write + Send + Unpin, RNG: RngCore + CryptoRng>(
        &self,
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        rng: &mut RNG,
        num: usize,
    ) -> Vec<Triple<T>>;
}

pub struct ClientOfflineMPC<T: AuthShare, C: ClientGen> {
    backend: C,
    _share: PhantomData<T>,
}

pub struct ServerOfflineMPC<T: AuthShare, S: ServerGen> {
    backend: S,
    _share: PhantomData<T>,
}

impl<'a, P: Fp64Parameters> ClientOfflineMPC<Fp64<P>, SealClientGen<'a>> {
    pub fn new(cfhe: &'a ClientFHE) -> Self {
        Self {
            backend: SealClientGen::new(cfhe),
            _share: PhantomData,
        }
    }
}

impl<'a, P: Fp64Parameters> ServerOfflineMPC<Fp64<P>, SealServerGen<'a>> {
    pub fn new(sfhe: &'a ServerFHE, mac_key: u64) -> Self {
        Self {
            backend: SealServerGen::new(sfhe, mac_key),
            _share: PhantomData,
        }
    }
}

impl<P: Fp64Parameters> ClientOfflineMPC<Fp64<P>, SealClientGen<'_>> {
    // TODO: Explain that this is for LTME assumption
    // TODO
    pub fn optimized_input<W: Write + Send + Unpin, RNG: RngCore + CryptoRng>(
        &self,
        keys: &ServerFHE,
        writer: &mut IMuxAsync<W>,
        input: &[Fp64<P>],
        mac_key_ct: &mut SealCT,
        rng: &mut RNG,
    ) -> Vec<AuthAdditiveShare<Fp64<P>>> {
        // Process shares
        // TODO: Thread
        let batches = (input.len() as f64 / Self::BATCH_SIZE as f64).ceil() as usize;
        let mut client_auth_shares = Vec::with_capacity(input.len());
        let mut server_shares = Vec::with_capacity(input.len());
        let mut server_mac_ct = Vec::with_capacity(batches);

        for input_slice in input.chunks(Self::BATCH_SIZE) {
            let (c_shares, s_shares): (Vec<_>, Vec<_>) =
                input_slice.iter().map(|e| e.share(rng)).unzip();
            let client_mac_shares: Vec<_> = (0..min(input_slice.len(), Self::BATCH_SIZE))
                .map(|_| Fp64::<P>::uniform(rng))
                .collect();
            let c_auth_shares: Vec<AuthAdditiveShare<_>> =
                izip!(c_shares, client_mac_shares.clone())
                    .map(|(s, m)| AuthAdditiveShare::new(s.inner, m))
                    .collect();
            server_shares.extend(s_shares);
            client_auth_shares.extend(c_auth_shares);

            // TODO: Might need to pad
            let input_c = input_slice.iter().map(|e| e.into_repr().0).collect();
            let rand_c = client_mac_shares.iter().map(|e| e.into_repr().0).collect();
            let mut server_ct = SealCT::new();
            server_mac_ct.push(server_ct.input_auth(keys, mac_key_ct, input_c, rand_c));
        }
        // Send shares in batches
        for i in 0..batches {
            let send_message = ShareSend::new(
                &server_shares[i * Self::BATCH_SIZE..min((i + 1) * Self::BATCH_SIZE, input.len())],
            );
            bytes::serialize(&mut *writer, &send_message).unwrap();
            let msg = (i, server_mac_ct[i].clone());
            let send_message = MsgSend::new(&msg);
            bytes::serialize(&mut *writer, &send_message).unwrap();
        }
        client_auth_shares
    }

    // TODO
    //    pub fn recv_mac<R: Read + Send + Unpin>(&self, reader: &mut IMuxAsync<R>) -> Vec<c_char> {
    //        let recv_message: MsgRcv = bytes::deserialize(&mut *reader).unwrap();
    //        recv_message.msg().1
    //    }
}

impl<P: Fp64Parameters> ServerOfflineMPC<Fp64<P>, SealServerGen<'_>> {
    // TODO: Explain that this is for LTME assumption
    // TODO
    pub fn recv_optimized_input<R: Read + Send + Unpin>(
        &self,
        keys: &ClientFHE,
        reader: &mut IMuxAsync<R>,
        num: usize,
    ) -> Vec<AuthAdditiveShare<Fp64<P>>> {
        // Receive batches of shares
        let mut shares = Vec::with_capacity(num);
        for i in 0..((num as f64 / Self::BATCH_SIZE as f64).ceil() as usize) {
            let recv_message: ShareRcv<_> = bytes::deserialize(&mut *reader).unwrap();
            let share = recv_message.msg();
            let recv_message: MsgRcv = bytes::deserialize(&mut *reader).unwrap();
            let (_, mac_share_ct) = recv_message.msg();
            let mut tmp = SealCT::new();
            let mac_shares = if i * Self::BATCH_SIZE > num {
                tmp.decrypt_vec(keys, mac_share_ct, num - (i - 1) * Self::BATCH_SIZE)
            } else {
                tmp.decrypt_vec(keys, mac_share_ct, Self::BATCH_SIZE)
            };
            let auth_shares: Vec<AuthAdditiveShare<_>> = izip!(share, mac_shares)
                .map(|(s, m)| AuthAdditiveShare::new(s.inner, Fp64::from_repr(m.into())))
                .collect();
            shares.extend(auth_shares);
        }
        shares
    }

    // TODO
    //    pub fn send_mac<W: Write + Send + Unpin>(&self, writer: &mut IMuxAsync<W>, mac_key: Vec<c_char>) {
    //        let msg = (0, mac_key);
    //        let send_message = MsgSend::new(&msg);
    //        bytes::serialize(writer, &send_message).unwrap();
    //    }
}

impl<P: Fp64Parameters> OfflineMPC<Fp64<P>> for ClientOfflineMPC<Fp64<P>, SealClientGen<'_>> {
    fn rands_gen<R, W, RNG>(
        &self,
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        _rng: &mut RNG,
        num: usize,
    ) -> Vec<AuthAdditiveShare<Fp64<P>>>
    where
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: RngCore + CryptoRng,
    {
        let start_time = timer_start!(|| "Client pairwise randomness generation");

        // Calculate number of batches to send per thread
        let batches = (num as f64 / Self::BATCH_SIZE as f64).ceil() as usize;
        let num_threads = min(batches, rayon::current_num_threads() - 1);
        let batches_per_thread = (batches as f64 / num_threads as f64).ceil() as usize;

        // Final result vector
        // TODO: Benchmark w/o mutex in single thread
        let result = Arc::new(Mutex::new(vec![AuthAdditiveShare::zero(); num]));

        // Vector which holds states for post processing server result
        let states = RwLock::new(vec![None; batches]);

        rayon::scope(|s| {
            // Create a channel which all threads will push state to be sent to the server
            let (send_1, mut recv_1) = channel::bounded(batches);
            // Create a channel which will contain all state receieved from the server
            let (send_2, recv_2) = channel::bounded(batches);

            for thread_idx in 0..num_threads {
                let send = send_1.clone(); // TODO: Change name
                let mut recv = recv_2.clone();
                let result = result.clone();
                let states = &states;
                s.spawn(move |_| {
                    // TODO: Remove
                    let rng = &mut ChaChaRng::from_seed(RANDOMNESS);
                    // If this is the last thread, only generate as many rands as needed
                    let num_rands = if thread_idx == num_threads - 1 {
                        num - thread_idx * Self::BATCH_SIZE * batches_per_thread
                    } else {
                        Self::BATCH_SIZE * batches_per_thread
                    };
                    let mut rands = Vec::with_capacity(num_rands);
                    for _ in 0..num_rands {
                        rands.push(Fp64::<P>::uniform(rng).into_repr().0);
                    }

                    for (i, rands_batch) in rands.chunks(Self::BATCH_SIZE).enumerate() {
                        let batch_idx = thread_idx * batches_per_thread + i;
                        // Preprocess state and ciphertexts
                        let (seal_state, ct) = self.backend.rands_preprocess(rands_batch);
                        // Push ciphertexts and state to channel
                        task::block_on(async {
                            send.send((batch_idx, seal_state, ct)).await.unwrap()
                        });
                        // TODO Simulate ZK proof time
                        if i % 6 == 0 {
                            // Proving time
                            std::thread::sleep(std::time::Duration::from_millis(385));
                            // Sending time
                            std::thread::sleep(std::time::Duration::from_millis(100));
                        }
                    }

                    task::block_on(async {
                        while let Some(msg) = recv.next().await {
                            let (i, mut r_ct, mut r_mac_ct): (usize, Vec<i8>, Vec<i8>) = msg;

                            // This is guaranteed to be Some(..) since we only receive `i`
                            // that the server has finished processing (and thus received)
                            let states = states.read().await;
                            let mut seal_state = states[i].unwrap();
                            drop(states);

                            let (r_share, r_mac_share) = self.backend.rands_postprocess(
                                &mut seal_state,
                                r_ct.as_mut_slice(),
                                r_mac_ct.as_mut_slice(),
                            );
                            let recv_shares = izip!(r_share, r_mac_share)
                                .map(|(v, m)| {
                                    AuthAdditiveShare::new(
                                        Fp64::from_repr(v.into()),
                                        Fp64::from_repr(m.into()),
                                    )
                                })
                                .collect::<Vec<_>>();
                            let mut result_lock = result.lock().unwrap();
                            for (old, new) in izip!(
                                (*result_lock)
                                    [Self::BATCH_SIZE * i..min(Self::BATCH_SIZE * (i + 1), num)]
                                    .iter_mut(),
                                recv_shares.iter()
                            ) {
                                *old = *new;
                            }
                        }
                    });
                });
            }
            // Drop the initial sending channel
            drop(send_1);

            task::block_on(async {
                // Future for sending ciphertexts to server and pushing states into `states`
                let send_future = async {
                    let send_time = timer_start!(|| "Sending ciphertexts to server");
                    while let Some((batch_idx, mut seal_state, ct)) = recv_1.next().await {
                        let msg = (batch_idx, ct);
                        let send_message = MsgSend::new(&msg);
                        bytes::async_serialize(&mut *writer, &send_message)
                            .await
                            .unwrap();
                        self.backend.rands_free_ct(&mut seal_state);
                        let mut states = states.write().await;
                        states[batch_idx] = Some(seal_state);
                    }
                    timer_end!(send_time);
                };

                // Future for receiving result from server
                let recv_future = async {
                    let send = send_2.clone();
                    let recv_time = timer_start!(|| "Receiving ciphertexts from server");
                    for _ in 0..batches {
                        // Receive ciphertexts from the server
                        let recv_message: MsgRcv =
                            bytes::async_deserialize(&mut *reader).await.unwrap();
                        let (i, r_ct) = recv_message.msg();
                        let recv_message: MsgRcv =
                            bytes::async_deserialize(&mut *reader).await.unwrap();
                        let (_, r_mac_ct) = recv_message.msg();
                        // Push ciphertexts to channel to be processed by a thread
                        send.send((i, r_ct, r_mac_ct)).await.unwrap();
                    }
                    timer_end!(recv_time);
                    // Drop the remaining channel
                    drop(send_2);
                };

                // Run the send/recv futures concurrently
                futures::future::join(send_future, recv_future).await;
            });
        });
        timer_end!(start_time);
        Arc::try_unwrap(result).unwrap().into_inner().unwrap()
    }

    fn triples_gen<R, W, RNG>(
        &self,
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        _rng: &mut RNG,
        num: usize,
    ) -> Vec<Triple<Fp64<P>>>
    where
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: RngCore + CryptoRng,
    {
        let start_time = timer_start!(|| "Client triples generation");

        // Calculate number of batches to send per thread
        let batches = (num as f64 / Self::BATCH_SIZE as f64).ceil() as usize;
        let num_threads = min(batches, rayon::current_num_threads() - 1);
        let batches_per_thread = (batches as f64 / num_threads as f64).ceil() as usize;

        // Final result vector
        // TODO: Benchmark w/o mutex in single thread
        let result = Arc::new(Mutex::new(vec![
            Triple {
                a: AuthAdditiveShare::zero(),
                b: AuthAdditiveShare::zero(),
                c: AuthAdditiveShare::zero(),
            };
            num
        ]));

        // Vector which holds states for post processing server result
        let states = RwLock::new(vec![None; batches]);

        rayon::scope(|s| {
            // Create a channel which all threads will push state to be sent to the server
            let (send_1, mut recv_1) = channel::bounded(batches);
            // Create a channel which will contain all state receieved from the server
            let (send_2, recv_2) = channel::bounded(batches);

            for thread_idx in 0..num_threads {
                let send = send_1.clone(); // TODO: Change name
                let mut recv = recv_2.clone();
                let result = result.clone();
                let states = &states;
                s.spawn(move |_| {
                    // TODO: Remove
                    let rng = &mut ChaChaRng::from_seed(RANDOMNESS);
                    // If this is the last thread, only generate as many rands as needed
                    let num_rands = if thread_idx == num_threads - 1 {
                        num - thread_idx * Self::BATCH_SIZE * batches_per_thread
                    } else {
                        Self::BATCH_SIZE * batches_per_thread
                    };
                    let mut a = Vec::with_capacity(num_rands);
                    let mut b = Vec::with_capacity(num_rands);
                    for _ in 0..num_rands {
                        a.push(Fp64::<P>::uniform(rng).into_repr().0);
                        b.push(Fp64::<P>::uniform(rng).into_repr().0);
                    }

                    for (i, (a_batch, b_batch)) in a
                        .chunks(Self::BATCH_SIZE)
                        .zip(b.chunks(Self::BATCH_SIZE))
                        .enumerate()
                    {
                        let batch_idx = thread_idx * batches_per_thread + i;
                        // Preprocess state and ciphertexts
                        let (seal_state, a_ct, b_ct) =
                            self.backend.triples_preprocess(a_batch, b_batch);
                        // Push ciphertexts and state to channel
                        task::block_on(async {
                            send.send((batch_idx, seal_state, a_ct, b_ct))
                                .await
                                .unwrap()
                        });
                        // TODO Simulate ZK proof time
                        if i % 6 == 0 {
                            // Proving time
                            std::thread::sleep(std::time::Duration::from_millis(385));
                            // Sending time
                            std::thread::sleep(std::time::Duration::from_millis(100));
                        }
                    }

                    task::block_on(async {
                        while let Some(msg) = recv.next().await {
                            let (
                                i,
                                mut a_ct,
                                mut b_ct,
                                mut c_ct,
                                mut a_mac_ct,
                                mut b_mac_ct,
                                mut c_mac_ct,
                            ): (
                                usize,
                                Vec<i8>,
                                Vec<i8>,
                                Vec<i8>,
                                Vec<i8>,
                                Vec<i8>,
                                Vec<i8>,
                            ) = msg;

                            // This is guaranteed to be Some(..) since we only receive `i`
                            // that the server has finished processing (and thus received)
                            let states = states.read().await;
                            let mut seal_state: protocols_sys::ClientTriples = states[i].unwrap();
                            drop(states);

                            let (a_share, b_share, c_share, a_mac_share, b_mac_share, c_mac_share) =
                                self.backend.triples_postprocess(
                                    &mut seal_state,
                                    a_ct.as_mut_slice(),
                                    b_ct.as_mut_slice(),
                                    c_ct.as_mut_slice(),
                                    a_mac_ct.as_mut_slice(),
                                    b_mac_ct.as_mut_slice(),
                                    c_mac_ct.as_mut_slice(),
                                );
                            // Map to Triples and insert into `result`
                            let recv_triples = izip!(
                                a_share,
                                b_share,
                                c_share,
                                a_mac_share,
                                b_mac_share,
                                c_mac_share
                            )
                            .map(|(a, b, c, a_m, b_m, c_m)| Triple {
                                a: AuthAdditiveShare::new(
                                    Fp64::from_repr(a.into()),
                                    Fp64::from_repr(a_m.into()),
                                ),
                                b: AuthAdditiveShare::new(
                                    Fp64::from_repr(b.into()),
                                    Fp64::from_repr(b_m.into()),
                                ),
                                c: AuthAdditiveShare::new(
                                    Fp64::from_repr(c.into()),
                                    Fp64::from_repr(c_m.into()),
                                ),
                            })
                            .collect::<Vec<_>>();
                            let mut result_lock = result.lock().unwrap();
                            for (old, new) in izip!(
                                (*result_lock)
                                    [Self::BATCH_SIZE * i..min(Self::BATCH_SIZE * (i + 1), num)]
                                    .iter_mut(),
                                recv_triples.iter()
                            ) {
                                *old = *new;
                            }
                        }
                    });
                });
            }
            // Drop the initial sending channel
            drop(send_1);

            task::block_on(async {
                // Future for sending ciphertexts to server and pushing states into `states`
                let send_future = async {
                    let send_time = timer_start!(|| "Sending ciphertexts to server");
                    while let Some((batch_idx, mut seal_state, a_ct, b_ct)) = recv_1.next().await {
                        let msg = (batch_idx, a_ct);
                        let send_message = MsgSend::new(&msg);
                        bytes::async_serialize(&mut *writer, &send_message)
                            .await
                            .unwrap();
                        let msg = (batch_idx, b_ct);
                        let send_message = MsgSend::new(&msg);
                        bytes::async_serialize(&mut *writer, &send_message)
                            .await
                            .unwrap();
                        self.backend.triples_free_ct(&mut seal_state);
                        let mut states = states.write().await;
                        states[batch_idx] = Some(seal_state);
                    }
                    timer_end!(send_time);
                };

                // Future for receiving result from server
                let recv_future = async {
                    let send = send_2.clone();
                    let recv_time = timer_start!(|| "Receiving ciphertexts from server");
                    for _ in 0..batches {
                        // Receive ciphertexts from the server
                        let recv_message: MsgRcv =
                            bytes::async_deserialize(&mut *reader).await.unwrap();
                        let (i, a_ct) = recv_message.msg();
                        let recv_message: MsgRcv =
                            bytes::async_deserialize(&mut *reader).await.unwrap();
                        let (_, b_ct) = recv_message.msg();
                        let recv_message: MsgRcv =
                            bytes::async_deserialize(&mut *reader).await.unwrap();
                        let (_, c_ct) = recv_message.msg();
                        let recv_message: MsgRcv =
                            bytes::async_deserialize(&mut *reader).await.unwrap();
                        let (_, a_mac_ct) = recv_message.msg();
                        let recv_message: MsgRcv =
                            bytes::async_deserialize(&mut *reader).await.unwrap();
                        let (_, b_mac_ct) = recv_message.msg();
                        let recv_message: MsgRcv =
                            bytes::async_deserialize(&mut *reader).await.unwrap();
                        let (_, c_mac_ct) = recv_message.msg();
                        // Push ciphertexts to channel to be processed by a thread
                        send.send((i, a_ct, b_ct, c_ct, a_mac_ct, b_mac_ct, c_mac_ct))
                            .await
                            .unwrap();
                    }
                    timer_end!(recv_time);
                    // Drop the remaining channel
                    drop(send_2);
                };
                // Run the send/recv futures concurrently
                futures::future::join(send_future, recv_future).await;
            });
        });
        timer_end!(start_time);
        Arc::try_unwrap(result).unwrap().into_inner().unwrap()
    }
}

impl<P: Fp64Parameters> OfflineMPC<Fp64<P>> for ServerOfflineMPC<Fp64<P>, SealServerGen<'_>> {
    fn rands_gen<R, W, RNG>(
        &self,
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        rng: &mut RNG,
        num: usize,
    ) -> Vec<AuthAdditiveShare<Fp64<P>>>
    where
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: RngCore + CryptoRng,
    {
        let start_time = timer_start!(|| "Server pairwise randomness generation");

        // Calculate number of batches to send per thread
        let batches = (num as f64 / Self::BATCH_SIZE as f64).ceil() as usize;
        let num_threads = min(batches, rayon::current_num_threads() - 1);
        let batches_per_thread = (batches as f64 / num_threads as f64).ceil() as usize;

        let pre_time = timer_start!(|| "Preprocessing");
        let mut rands =
            vec![Vec::with_capacity(Self::BATCH_SIZE * batches_per_thread); num_threads];
        let mut shares =
            vec![Vec::with_capacity(Self::BATCH_SIZE * batches_per_thread); num_threads];
        let mut mac_shares =
            vec![Vec::with_capacity(Self::BATCH_SIZE * batches_per_thread); num_threads];
        let mut result = Vec::with_capacity(num);
        for i in 0..num {
            let idx = i / Self::BATCH_SIZE / batches_per_thread;
            rands[idx].push(Fp64::<P>::uniform(rng).into_repr().0);
            let share = Fp64::<P>::uniform(rng);
            let mac_share = Fp64::<P>::uniform(rng);
            shares[idx].push(share.into_repr().0);
            mac_shares[idx].push(mac_share.into_repr().0);
            result.push(AuthAdditiveShare::new(share, mac_share));
        }
        timer_end!(pre_time);

        // Create channels which will store state received from the client
        let (tx_inp, rx_inp): (
            Vec<channel::Sender<Vec<c_char>>>,
            Vec<channel::Receiver<Vec<c_char>>>,
        ) = (0..num_threads)
            .map(|_| channel::bounded(batches_per_thread))
            .unzip();

        // Create a channel which will contain all state to be sent to the client
        let (tx_out, mut rx_out) = channel::bounded(batches);

        rayon::scope(|s| {
            for thread_idx in (0..num_threads).rev() {
                // Move thread batches into scope
                let rands = rands.pop().unwrap();
                let shares = shares.pop().unwrap();
                let mac_shares = mac_shares.pop().unwrap();

                // Get references to necessary channels
                let rx = rx_inp[thread_idx].clone();
                let tx = tx_out.clone();

                s.spawn(move |_| {
                    for (i, (rand, share, mac_share)) in izip!(
                        rands.chunks(Self::BATCH_SIZE),
                        shares.chunks(Self::BATCH_SIZE),
                        mac_shares.chunks(Self::BATCH_SIZE),
                    )
                    .enumerate()
                    {
                        let batch_idx = thread_idx * batches_per_thread + i;
                        let mut seal_state = self.backend.rands_preprocess(rand, share, mac_share);
                        task::block_on(async {
                            // Receive and process ciphertexts from client
                            let mut ct = rx.recv().await.unwrap();
                            tx.send((
                                batch_idx,
                                self.backend
                                    .rands_online(&mut seal_state, ct.as_mut_slice()),
                            ))
                            .await
                            .unwrap();
                        });
                        // TODO Simulate ZK-proof time
                        if i % 6 == 0 {
                            // Receiving time
                            std::thread::sleep(std::time::Duration::from_millis(100));
                            // Verifying time
                            std::thread::sleep(std::time::Duration::from_millis(100));
                        }
                    }
                });
            }
            // Drop the leftover channels
            drop(tx_out);
            rx_inp.into_iter().for_each(|c| drop(c));

            task::block_on(async {
                // Future for receiving ciphertexts from the client
                let recv_future = async {
                    let recv_time = timer_start!(|| "Receiving client input");
                    for _ in 0..batches {
                        // Receive input from client
                        let recv_message: MsgRcv =
                            bytes::async_deserialize(&mut *reader).await.unwrap();
                        let (i, ct) = recv_message.msg();
                        // Send input over appropriate channel
                        tx_inp[i / batches_per_thread].send(ct).await.unwrap();
                    }
                    timer_end!(recv_time);
                };
                // Future for sending cipehrtexts to the client
                let send_future = async {
                    while let Some((i, (r_share_ct, r_mac_share_ct))) = rx_out.next().await {
                        // Send result to client
                        let msg = (i, r_share_ct);
                        let send_message = MsgSend::new(&msg);
                        bytes::async_serialize(&mut *writer, &send_message)
                            .await
                            .unwrap();
                        let msg = (i, r_mac_share_ct);
                        let send_message = MsgSend::new(&msg);
                        bytes::async_serialize(&mut *writer, &send_message)
                            .await
                            .unwrap();
                    }
                };
                futures::future::join(recv_future, send_future).await;
                // Drop remaining channels
                tx_inp.into_iter().for_each(|c| drop(c));
            });
        });
        timer_end!(start_time);
        result
    }

    fn triples_gen<R, W, RNG>(
        &self,
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        rng: &mut RNG,
        num: usize,
    ) -> Vec<Triple<Fp64<P>>>
    where
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: RngCore + CryptoRng,
    {
        let start_time = timer_start!(|| "Server triples generation");

        // Calculate number of batches to send per thread
        let batches = (num as f64 / Self::BATCH_SIZE as f64).ceil() as usize;
        let num_threads = min(batches, rayon::current_num_threads() - 1);
        let batches_per_thread = (batches as f64 / num_threads as f64).ceil() as usize;

        let rand_time = timer_start!(|| "Generating shares");
        let mut a_rands =
            vec![Vec::with_capacity(Self::BATCH_SIZE * batches_per_thread); num_threads];
        let mut b_rands =
            vec![Vec::with_capacity(Self::BATCH_SIZE * batches_per_thread); num_threads];
        let mut c_rands =
            vec![Vec::with_capacity(Self::BATCH_SIZE * batches_per_thread); num_threads];
        let mut a_shares =
            vec![Vec::with_capacity(Self::BATCH_SIZE * batches_per_thread); num_threads];
        let mut b_shares =
            vec![Vec::with_capacity(Self::BATCH_SIZE * batches_per_thread); num_threads];
        let mut c_shares =
            vec![Vec::with_capacity(Self::BATCH_SIZE * batches_per_thread); num_threads];
        let mut a_mac_shares =
            vec![Vec::with_capacity(Self::BATCH_SIZE * batches_per_thread); num_threads];
        let mut b_mac_shares =
            vec![Vec::with_capacity(Self::BATCH_SIZE * batches_per_thread); num_threads];
        let mut c_mac_shares =
            vec![Vec::with_capacity(Self::BATCH_SIZE * batches_per_thread); num_threads];
        let mut triples = Vec::with_capacity(num);
        for i in 0..num {
            let idx = i / Self::BATCH_SIZE / batches_per_thread;
            let a_rand = Fp64::<P>::uniform(rng);
            let b_rand = Fp64::<P>::uniform(rng);
            let c_rand = a_rand * b_rand;
            let a_share = Fp64::<P>::uniform(rng);
            let b_share = Fp64::<P>::uniform(rng);
            let c_share = Fp64::<P>::uniform(rng);
            let a_mac_share = Fp64::<P>::uniform(rng);
            let b_mac_share = Fp64::<P>::uniform(rng);
            let c_mac_share = Fp64::<P>::uniform(rng);
            a_rands[idx].push(a_rand.into_repr().0);
            b_rands[idx].push(b_rand.into_repr().0);
            c_rands[idx].push(c_rand.into_repr().0);
            a_shares[idx].push(a_share.into_repr().0);
            b_shares[idx].push(b_share.into_repr().0);
            c_shares[idx].push(c_share.into_repr().0);
            a_mac_shares[idx].push(a_mac_share.into_repr().0);
            b_mac_shares[idx].push(b_mac_share.into_repr().0);
            c_mac_shares[idx].push(c_mac_share.into_repr().0);

            triples.push(Triple {
                a: AuthAdditiveShare::new(a_share, a_mac_share),
                b: AuthAdditiveShare::new(b_share, b_mac_share),
                c: AuthAdditiveShare::new(c_share, c_mac_share),
            });
        }
        timer_end!(rand_time);

        // Create channels which will store state received from the client
        let (tx_inp, rx_inp): (
            Vec<channel::Sender<(Vec<c_char>, Vec<c_char>)>>,
            Vec<channel::Receiver<(Vec<c_char>, Vec<c_char>)>>,
        ) = (0..num_threads)
            .map(|_| channel::bounded(batches_per_thread))
            .unzip();

        // Create a channel which will contain all state to be sent to the client
        let (tx_out, mut rx_out) = channel::bounded(batches);

        rayon::scope(|s| {
            for thread_idx in (0..num_threads).rev() {
                // Move thread batches into scope
                let a_rands = a_rands.pop().unwrap();
                let b_rands = b_rands.pop().unwrap();
                let c_rands = c_rands.pop().unwrap();
                let a_shares = a_shares.pop().unwrap();
                let b_shares = b_shares.pop().unwrap();
                let c_shares = c_shares.pop().unwrap();
                let a_mac_shares = a_mac_shares.pop().unwrap();
                let b_mac_shares = b_mac_shares.pop().unwrap();
                let c_mac_shares = c_mac_shares.pop().unwrap();

                // Get references to necessary channels
                let rx = rx_inp[thread_idx].clone();
                let tx = tx_out.clone();

                s.spawn(move |_| {
                    for (
                        i,
                        (
                            a_rand,
                            b_rand,
                            c_rand,
                            a_share,
                            b_share,
                            c_share,
                            a_mac_share,
                            b_mac_share,
                            c_mac_share,
                        ),
                    ) in izip!(
                        a_rands.chunks(Self::BATCH_SIZE),
                        b_rands.chunks(Self::BATCH_SIZE),
                        c_rands.chunks(Self::BATCH_SIZE),
                        a_shares.chunks(Self::BATCH_SIZE),
                        b_shares.chunks(Self::BATCH_SIZE),
                        c_shares.chunks(Self::BATCH_SIZE),
                        a_mac_shares.chunks(Self::BATCH_SIZE),
                        b_mac_shares.chunks(Self::BATCH_SIZE),
                        c_mac_shares.chunks(Self::BATCH_SIZE),
                    )
                    .enumerate()
                    {
                        let batch_idx = thread_idx * batches_per_thread + i;
                        let mut seal_state = self.backend.triples_preprocess(
                            a_rand,
                            b_rand,
                            c_rand,
                            a_share,
                            b_share,
                            c_share,
                            a_mac_share,
                            b_mac_share,
                            c_mac_share,
                        );
                        task::block_on(async {
                            // Receive and process ciphertexts from client
                            let (mut a_rands_ct, mut b_rands_ct) = rx.recv().await.unwrap();
                            tx.send((
                                batch_idx,
                                self.backend.triples_online(
                                    &mut seal_state,
                                    a_rands_ct.as_mut_slice(),
                                    b_rands_ct.as_mut_slice(),
                                ),
                            ))
                            .await
                            .unwrap();
                        });
                        // TODO Simulate ZK-proof time
                        if i % 6 == 0 {
                            // Receiving time
                            std::thread::sleep(std::time::Duration::from_millis(100));
                            // Verifying time
                            std::thread::sleep(std::time::Duration::from_millis(100));
                        }
                    }
                });
            }
            // Drop the leftover channels
            drop(tx_out);
            rx_inp.into_iter().for_each(|c| drop(c));

            task::block_on(async {
                // Future for receiving ciphertexts from the client
                let recv_future = async {
                    let recv_time = timer_start!(|| "Receiving client input");
                    for _ in 0..batches {
                        // Receive input from client
                        let recv_message: MsgRcv =
                            bytes::async_deserialize(&mut *reader).await.unwrap();
                        let (i, a_rands_ct) = recv_message.msg();
                        let recv_message: MsgRcv =
                            bytes::async_deserialize(&mut *reader).await.unwrap();
                        let (j, b_rands_ct) = recv_message.msg();
                        assert_eq!(i, j);
                        // Send input over appropriate channel
                        tx_inp[i / batches_per_thread]
                            .send((a_rands_ct, b_rands_ct))
                            .await
                            .unwrap();
                    }
                    timer_end!(recv_time);
                };

                // Future for sending cipehrtexts to the client
                let send_future = async {
                    while let Some((i, (a_ct, b_ct, c_ct, a_mac_ct, b_mac_ct, c_mac_ct))) =
                        rx_out.next().await
                    {
                        // Send result to client
                        let msg = (i, a_ct);
                        let send_message = MsgSend::new(&msg);
                        bytes::async_serialize(&mut *writer, &send_message)
                            .await
                            .unwrap();
                        let msg = (i, b_ct);
                        let send_message = MsgSend::new(&msg);
                        bytes::async_serialize(&mut *writer, &send_message)
                            .await
                            .unwrap();
                        let msg = (i, c_ct);
                        let send_message = MsgSend::new(&msg);
                        bytes::async_serialize(&mut *writer, &send_message)
                            .await
                            .unwrap();
                        let msg = (i, a_mac_ct);
                        let send_message = MsgSend::new(&msg);
                        bytes::async_serialize(&mut *writer, &send_message)
                            .await
                            .unwrap();
                        let msg = (i, b_mac_ct);
                        let send_message = MsgSend::new(&msg);
                        bytes::async_serialize(&mut *writer, &send_message)
                            .await
                            .unwrap();
                        let msg = (i, c_mac_ct);
                        let send_message = MsgSend::new(&msg);
                        bytes::async_serialize(&mut *writer, &send_message)
                            .await
                            .unwrap();
                        writer.flush().await.unwrap();
                    }
                };
                futures::future::join(recv_future, send_future).await;
                // Drop remaining channels
                tx_inp.into_iter().for_each(|c| drop(c));
            });
        });
        timer_end!(start_time);
        triples
    }
}

/// Insecure offline MPC phase for testing
pub struct InsecureClientOfflineMPC<T: AuthShare, C: ClientGen> {
    _backend: PhantomData<C>,
    _share: PhantomData<T>,
}

pub struct InsecureServerOfflineMPC<T: AuthShare, S: ServerGen> {
    backend: S,
    _share: PhantomData<T>,
}

type InsecureTriplesSend<'a, F> = OutMessage<'a, Vec<Triple<F>>, OfflineMPCProtocolType>;
type InsecureTriplesRcv<F> = InMessage<Vec<Triple<F>>, OfflineMPCProtocolType>;
type InsecureRandsSend<'a, F> = OutMessage<'a, Vec<AuthAdditiveShare<F>>, OfflineMPCProtocolType>;
type InsecureRandsRcv<F> = InMessage<Vec<AuthAdditiveShare<F>>, OfflineMPCProtocolType>;

impl<'a, P: Fp64Parameters> InsecureClientOfflineMPC<Fp64<P>, SealClientGen<'a>> {
    pub fn new(_cfhe: &'a ClientFHE) -> Self {
        Self {
            _backend: PhantomData,
            _share: PhantomData,
        }
    }
}

impl<'a, P: Fp64Parameters> InsecureServerOfflineMPC<Fp64<P>, SealServerGen<'a>> {
    pub fn new(sfhe: &'a ServerFHE, mac_key: u64) -> Self {
        Self {
            backend: SealServerGen::new(sfhe, mac_key),
            _share: PhantomData,
        }
    }
}

impl<P: Fp64Parameters> OfflineMPC<Fp64<P>>
    for InsecureClientOfflineMPC<Fp64<P>, SealClientGen<'_>>
{
    fn rands_gen<R: Read + Send + Unpin, W: Write + Send + Unpin, RNG: RngCore + CryptoRng>(
        &self,
        reader: &mut IMuxAsync<R>,
        _writer: &mut IMuxAsync<W>,
        _rng: &mut RNG,
        _num: usize,
    ) -> Vec<AuthAdditiveShare<Fp64<P>>> {
        let start_time = timer_start!(|| "Insecure Client pairwise randomness generation");
        let recv_message: InsecureRandsRcv<Fp64<_>> = bytes::deserialize(&mut *reader).unwrap();
        let result = recv_message.msg();
        timer_end!(start_time);
        result
    }

    fn triples_gen<R: Read + Send + Unpin, W: Write + Send + Unpin, RNG: RngCore + CryptoRng>(
        &self,
        reader: &mut IMuxAsync<R>,
        _writer: &mut IMuxAsync<W>,
        _rng: &mut RNG,
        _num: usize,
    ) -> Vec<Triple<Fp64<P>>> {
        let start_time = timer_start!(|| "Insecure Client triples generation");
        let recv_message: InsecureTriplesRcv<Fp64<_>> = bytes::deserialize(&mut *reader).unwrap();
        let result = recv_message.msg();
        timer_end!(start_time);
        result
    }
}

impl<P: Fp64Parameters> OfflineMPC<Fp64<P>>
    for InsecureServerOfflineMPC<Fp64<P>, SealServerGen<'_>>
{
    fn rands_gen<R: Read + Send + Unpin, W: Write + Send + Unpin, RNG: RngCore + CryptoRng>(
        &self,
        _reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        rng: &mut RNG,
        num: usize,
    ) -> Vec<AuthAdditiveShare<Fp64<P>>> {
        let start_time = timer_start!(|| "Insecure Server pairwise randomness generation");
        let mut server_rands = Vec::with_capacity(num);
        let mut client_rands = Vec::with_capacity(num);
        let mac_key = Fp64::from_repr(self.backend.mac_key.into());

        for _ in 0..num {
            let value = Fp64::uniform(rng);
            let (s_rand, c_rand) = value.auth_share(&mac_key, rng);
            server_rands.push(s_rand);
            client_rands.push(c_rand);
        }

        let send_message = InsecureRandsSend::new(&client_rands);
        bytes::serialize(&mut *writer, &send_message).unwrap();
        timer_end!(start_time);
        server_rands
    }

    fn triples_gen<R: Read + Send + Unpin, W: Write + Send + Unpin, RNG: RngCore + CryptoRng>(
        &self,
        _reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        _rng: &mut RNG,
        num: usize,
    ) -> Vec<Triple<Fp64<P>>> {
        let start_time = timer_start!(|| "Insecure Server triples generation");
        let mut server_triples = Vec::with_capacity(num);
        let mut client_triples = Vec::with_capacity(num);
        let seed: [u8; 32] = [
            0x99, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda,
            0xf4, 0x76, 0x5d, 0xc9, 0x8d, 0x62, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77,
            0xd3, 0x4a, 0x52, 0xd2,
        ];
        let mut insecure_gen = crypto_primitives::beavers_mul::InsecureTripleGen::new(seed);
        let mac_key = Fp64::from_repr(self.backend.mac_key.into());

        for _ in 0..num {
            let (s_triple, c_triple) = insecure_gen.generate_triple_shares(mac_key);
            server_triples.push(s_triple);
            client_triples.push(c_triple);
        }

        let send_message = InsecureTriplesSend::new(&client_triples);
        bytes::serialize(&mut *writer, &send_message).unwrap();
        timer_end!(start_time);
        server_triples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ClientKeySend, ServerKeyRcv};
    use algebra::{fields::near_mersenne_64::F, PrimeField, UniformRandom};
    use async_std::{
        io::{BufReader, BufWriter, Read, Write},
        net::{TcpListener, TcpStream},
    };
    use io_utils::imux::IMuxAsync;
    use protocols_sys::KeyShare;
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    const RANDOMNESS: [u8; 32] = [
        0x99, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4,
        0x76, 0x5d, 0xc9, 0x8d, 0x62, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a,
        0x52, 0xd2,
    ];

    fn get_connection(
        server_addr: &str,
    ) -> (
        (IMuxAsync<impl Read>, IMuxAsync<impl Write>),
        (IMuxAsync<impl Read>, IMuxAsync<impl Write>),
    ) {
        crossbeam::thread::scope(|s| {
            let server_io = s.spawn(|_| {
                task::block_on(async {
                    let server_listener = TcpListener::bind(server_addr).await.unwrap();
                    let stream = server_listener
                        .incoming()
                        .next()
                        .await
                        .unwrap()
                        .expect("Server connection failed!");
                    let read_stream = IMuxAsync::new(vec![BufReader::new(stream.clone())]);
                    let write_stream = IMuxAsync::new(vec![BufWriter::new(stream)]);
                    (read_stream, write_stream)
                })
            });
            // Sometimes the client thread will start too soon and connection fails so put a
            // small delay
            std::thread::sleep(std::time::Duration::from_millis(10));
            let client_io = s.spawn(|_| {
                task::block_on(async {
                    let stream = TcpStream::connect(server_addr)
                        .await
                        .expect("Client connection failed!");
                    let read_stream = IMuxAsync::new(vec![BufReader::new(stream.clone())]);
                    let write_stream = IMuxAsync::new(vec![BufWriter::new(stream)]);
                    (read_stream, write_stream)
                })
            });
            (client_io.join().unwrap(), server_io.join().unwrap())
        })
        .unwrap()
    }

    #[test]
    fn test_rands_gen() {
        let server_addr = "127.0.0.1:8020";

        let num: usize = 1000000;
        let ((mut client_read, mut client_write), (mut server_read, mut server_write)) =
            get_connection(&server_addr);

        let (client_rands, (server_rands, mac_key)) = crossbeam::thread::scope(|s| {
            let server_result = s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                // Keygen
                let mac_key = F::uniform(&mut rng);
                let key_recv = timer_start!(|| "Receiving Keys");
                let recv_message: ServerKeyRcv = bytes::deserialize(&mut server_read).unwrap();
                let mut key_share = KeyShare::new();
                let sfhe = key_share.receive(recv_message.msg());
                timer_end!(key_recv);

                // Generate rands
                let server_gen = ServerOfflineMPC::<F, _>::new(&sfhe, mac_key.into_repr().0);
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(rayon::current_num_threads() / 2)
                    .build()
                    .unwrap();
                pool.install(|| {
                    (
                        server_gen.rands_gen(&mut server_read, &mut server_write, &mut rng, num),
                        mac_key,
                    )
                })
            });

            let client_rands = s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                // Keygen
                let keygen = timer_start!(|| "Generating Keys");
                let mut key_share = KeyShare::new();
                let (cfhe, keys_vec) = key_share.generate();
                timer_end!(keygen);

                let key_send = timer_start!(|| "Sending Keys");
                let send_message = ClientKeySend::new(&keys_vec);
                bytes::serialize(&mut client_write, &send_message).unwrap();
                timer_end!(key_send);

                // Generate rands
                let client_gen = ClientOfflineMPC::<F, _>::new(&cfhe);
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(rayon::current_num_threads() / 2)
                    .build()
                    .unwrap();
                pool.install(|| {
                    client_gen.rands_gen(&mut client_read, &mut client_write, &mut rng, num)
                })
            });
            (client_rands.join().unwrap(), server_result.join().unwrap())
        })
        .unwrap();

        izip!(client_rands, server_rands)
            .for_each(|(s1, s2)| assert!(s1.combine(&s2, &mac_key).is_ok()));
    }

    #[test]
    fn test_triples_gen() {
        let server_addr = "127.0.0.1:8021";

        let num: usize = 1000000;
        let ((mut client_read, mut client_write), (mut server_read, mut server_write)) =
            get_connection(&server_addr);

        let (client_triples, (server_triples, mac_key)) = crossbeam::thread::scope(|s| {
            let server_result = s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                // Keygen
                let mac_key = F::uniform(&mut rng);
                let key_recv = timer_start!(|| "Receiving Keys");
                let recv_message: ServerKeyRcv = bytes::deserialize(&mut server_read).unwrap();
                let mut key_share = KeyShare::new();
                let sfhe = key_share.receive(recv_message.msg());
                timer_end!(key_recv);

                // Generate triples
                let server_gen = ServerOfflineMPC::<F, _>::new(&sfhe, mac_key.into_repr().0);
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(rayon::current_num_threads() / 2)
                    .build()
                    .unwrap();
                pool.install(|| {
                    (
                        server_gen.triples_gen(&mut server_read, &mut server_write, &mut rng, num),
                        mac_key,
                    )
                })
            });

            let client_triples = s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                // Keygen
                let keygen = timer_start!(|| "Generating Keys");
                let mut key_share = KeyShare::new();
                let (cfhe, keys_vec) = key_share.generate();
                timer_end!(keygen);

                let key_send = timer_start!(|| "Sending Keys");
                let send_message = ClientKeySend::new(&keys_vec);
                bytes::serialize(&mut client_write, &send_message).unwrap();
                timer_end!(key_send);

                // Generate triples
                let client_gen = ClientOfflineMPC::<F, _>::new(&cfhe);
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(rayon::current_num_threads() / 2)
                    .build()
                    .unwrap();
                pool.install(|| {
                    client_gen.triples_gen(&mut client_read, &mut client_write, &mut rng, num)
                })
            });
            (
                client_triples.join().unwrap(),
                server_result.join().unwrap(),
            )
        })
        .unwrap();

        izip!(client_triples, server_triples).for_each(|(c, s)| {
            let a = s.a.combine(&c.a, &mac_key);
            let b = s.b.combine(&c.b, &mac_key);
            let c = s.c.combine(&c.c, &mac_key);
            assert!(a.is_ok());
            assert!(b.is_ok());
            assert!(c.is_ok());
            assert_eq!(c.unwrap(), a.unwrap() * b.unwrap());
        });
    }
}
