use crate::bytes;
use crate::{error::MpcError, InMessage, OutMessage};
use algebra::fields::{Fp64, Fp64Parameters};
use async_std::io::{Read, Write};
use crypto_primitives::{
    additive_share::{AdditiveShare, AuthAdditiveShare, AuthShare, Share},
    beavers_mul::{BeaversMul, BlindedInputs, BlindedSharedInputs, PBeaversMul, Triple},
};
use io_utils::imux::IMuxAsync;
use itertools::izip;
use num_traits::identities::Zero;
use rand::{CryptoRng, RngCore};
use rayon::prelude::*;

pub struct MpcProtocolType;

type AuthShareSend<'a, T> = OutMessage<'a, [AuthAdditiveShare<T>], MpcProtocolType>;
type AuthShareRcv<T> = InMessage<Vec<AuthAdditiveShare<T>>, MpcProtocolType>;

pub type ShareSend<'a, T> = OutMessage<'a, [AdditiveShare<T>], MpcProtocolType>;
pub type ShareRcv<T> = InMessage<Vec<AdditiveShare<T>>, MpcProtocolType>;

type MulMsgSend<'a, T> = OutMessage<'a, [BlindedSharedInputs<T>], MpcProtocolType>;
type MulMsgRcv<T> = InMessage<Vec<BlindedSharedInputs<T>>, MpcProtocolType>;

type ConstantMsgSend<'a, T> = OutMessage<'a, [T], MpcProtocolType>;
type ConstantMsgRcv<T> = InMessage<Vec<T>, MpcProtocolType>;

// TODO: Handle errors better
// TODO: Explore using rayon
// TODO: Add Drop trait for checking MACs + intermediate checks
/// Represents a type which implements a client-malicious SPDZ-style MPC
/// protocol
pub trait MPC<T: AuthShare, M: BeaversMul<T>>: Send + Sync {
    /// Message batch size
    const BATCH_SIZE: usize = 8192;

    /// Party index
    const PARTY_IDX: usize;

    /// Share `inputs` with the other party
    fn private_inputs<R: Read + Send + Unpin, W: Write + Send + Unpin, RNG: RngCore + CryptoRng>(
        &mut self,
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        inputs: &[T],
        rng: &mut RNG,
    ) -> Result<Vec<AuthAdditiveShare<T>>, MpcError>;

    /// Receive `num_recv` shares from the other party
    fn recv_private_inputs<R: Read + Send + Unpin, W: Write + Send + Unpin>(
        &mut self,
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        num_recv: usize,
    ) -> Result<Vec<AuthAdditiveShare<T>>, MpcError>;

    /// Opens `shares` to the other party
    fn private_open<W: Write + Send + Unpin>(
        &self,
        writer: &mut IMuxAsync<W>,
        shares: &[AuthAdditiveShare<T>],
    ) -> Result<(), MpcError>;

    /// Receive `shares` from the other party
    fn private_recv<R: Read + Send + Unpin>(
        &mut self,
        reader: &mut IMuxAsync<R>,
        shares: &[AuthAdditiveShare<T>],
    ) -> Result<Vec<T>, MpcError>;

    /// Opens `shares` publically and returns result
    fn public_open<R: Read + Send + Unpin, W: Write + Send + Unpin>(
        &mut self,
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        shares: &[AuthAdditiveShare<T>],
    ) -> Result<Vec<T>, MpcError>;

    /// Add shares `x` and `y`
    fn add(
        &mut self,
        x: &[AuthAdditiveShare<T>],
        y: &[AuthAdditiveShare<T>],
    ) -> Result<Vec<AuthAdditiveShare<T>>, MpcError> {
        if x.len() != y.len() {
            return Err(MpcError::MismatchedInputLength {
                left: x.len(),
                right: y.len(),
            });
        }
        Ok(izip!(x, y).map(|(l, r)| l + r).collect())
    }

    /// Sub shares `y` from `x`
    fn sub(
        &mut self,
        x: &[AuthAdditiveShare<T>],
        y: &[AuthAdditiveShare<T>],
    ) -> Result<Vec<AuthAdditiveShare<T>>, MpcError> {
        if x.len() != y.len() {
            return Err(MpcError::MismatchedInputLength {
                left: x.len(),
                right: y.len(),
            });
        }
        Ok(izip!(x, y).map(|(l, r)| l - r).collect())
    }

    // TODO
    //    async fn mul<D: EvaluationDomain<F>>(
    //        &mut self,
    //        a: Evaluations<F, D>,
    //        b: Evaluations<F, D>,
    //    ) -> Result<Evaluations<F, D>, DelegationError> {
    //        let domain = a.domain().clone();
    //
    //        // Consume necessary triples
    //        let req_triples = a.evals.len();
    //        let triples = self.get_triples(req_triples)?;
    //
    //        // Compute blinded shares using the triples.
    //        let blinded_shares = a
    //            .evals
    //            .into_iter()
    //            .zip(b.evals.into_iter())
    //            .zip(triples.iter())
    //            .map(|((a, b), t)| FBeaversMul::share_and_blind_inputs(&a, &b, t))
    //            .collect::<Vec<_>>();
    //
    //        // Send blinded shares to all parties
    //        let send_shares_f = self
    //            .writers
    //            .iter_mut()
    //            .map(|w| crate::IO::serialize_write_and_flush(&blinded_shares, w))
    //            .collect::<FuturesUnordered<_>>()
    //            .collect::<Vec<_>>();
    //
    //        // Receive blinded shares from all parties
    //        let recv_shares_f = self
    //            .readers
    //            .iter_mut()
    //            .map(|r| crate::IO::read_and_deserialize::<Vec<BlindedSharedInputs<F>>, R>(r))
    //            .collect::<FuturesUnordered<_>>();
    //
    //        // Combine all vectors of shares together
    //        let blinded_inputs =
    //            recv_shares_f.fold(Ok(blinded_shares.clone()), |a, b| Self::add_entries(a, b));
    //
    //        // Concurrently receive/add shares together and send shares
    //        let (blinded_inputs, send_shares_f) = join(blinded_inputs, send_shares_f).await;
    //
    //        // Unwrap any errors that occured when sending.
    //        send_shares_f.into_iter().collect::<Result<Vec<_>, _>>()?;
    //
    //        // Use blinded_inputs and triples to compute local share
    //        let result = blinded_inputs?
    //            .into_iter()
    //            .zip(triples)
    //            .map(|(bi, t)| FBeaversMul::multiply_blinded_inputs(self.party_idx, bi.into(), &t))
    //            .collect();
    //
    //        Ok(Evaluations::from_vec_and_domain(result, domain))
    //    }

    /// Multiply shares `x` and `y`
    fn mul<R: Read + Send + Unpin, W: Write + Send + Unpin>(
        &mut self,
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        x: &[AuthAdditiveShare<T>],
        y: &[AuthAdditiveShare<T>],
    ) -> Result<Vec<AuthAdditiveShare<T>>, MpcError> {
        if x.len() != y.len() {
            return Err(MpcError::MismatchedInputLength {
                left: x.len(),
                right: y.len(),
            });
        }
        // Consume necessary triples
        let triples = self.get_triples(x.len())?;

        // Compute blinded shares using the triples.
        let self_blinded_and_shared = izip!(x.iter(), y.iter(), triples.iter())
            .map(|(left, right, t)| M::share_and_blind_inputs(left, right, t))
            .collect::<Vec<_>>();

        let mut result = Vec::with_capacity(triples.len());
        let (snd, rcv) = crossbeam::channel::unbounded();
        crossbeam::scope(|s| {
            // Receive blinded shares
            s.spawn(|_| {
                for _ in self_blinded_and_shared.chunks(Self::BATCH_SIZE) {
                    let in_msg: MulMsgRcv<_> = bytes::deserialize(&mut *reader).unwrap();
                    let shares = in_msg.msg();
                    snd.send(shares).unwrap();
                }
                // Need to drop the sending channel so the second thread doesn't
                // block
                drop(snd);
            });
            // Send blinded shares
            s.spawn(|_| {
                for msg_contents in self_blinded_and_shared.chunks(Self::BATCH_SIZE) {
                    let sent_message = MulMsgSend::new(&msg_contents);
                    bytes::serialize(&mut *writer, &sent_message).unwrap();
                }
            });
            // Open blinded shares and perform multiplication
            for (cur_chunk, other_chunk, triple_chunk) in izip!(
                self_blinded_and_shared.chunks(Self::BATCH_SIZE),
                rcv.iter(),
                triples.chunks(Self::BATCH_SIZE)
            ) {
                let result_chunk: Vec<AuthAdditiveShare<T>> = izip!(cur_chunk, other_chunk.iter())
                    .map(|(cur, other)| BlindedInputs {
                        blinded_x: (cur.blinded_x + other.blinded_x).get_value().inner,
                        blinded_y: (cur.blinded_y + other.blinded_y).get_value().inner,
                    })
                    .zip(triple_chunk)
                    .map(|(inp, triple)| M::multiply_blinded_inputs(Self::PARTY_IDX, inp, triple))
                    .collect();
                result.extend_from_slice(result_chunk.as_slice());
            }
        })
        .unwrap();
        Ok(result)
    }

    /// Add shares `x` with constants `c`
    fn add_const(
        &self,
        x: &[AuthAdditiveShare<T>],
        c: &[impl Into<T> + Copy],
    ) -> Result<Vec<AuthAdditiveShare<T>>, MpcError> {
        if x.len() != c.len() {
            return Err(MpcError::MismatchedInputLength {
                left: x.len(),
                right: c.len(),
            });
        }
        Ok(izip!(x, c).map(|(l, r)| l.add_constant(*r)).collect())
    }

    /// Multiply shares `x` with constants `c`
    fn mul_const(
        &self,
        x: &[AuthAdditiveShare<T>],
        c: &[impl Into<<T as Share>::Constant> + Copy],
    ) -> Result<Vec<AuthAdditiveShare<T>>, MpcError> {
        if x.len() != c.len() {
            return Err(MpcError::MismatchedInputLength {
                left: x.len(),
                right: c.len(),
            });
        }
        Ok(izip!(x, c).map(|(l, r)| *l * *r).collect())
    }

    /// Sums a vector of elements together
    fn sum(&self, x: &[AuthAdditiveShare<T>]) -> AuthAdditiveShare<T> {
        x.par_iter()
            .map(|e| *e)
            .reduce(|| AuthAdditiveShare::zero(), |l, r| l + r)
    }

    /// Returns number of available triples
    fn num_triples(&self) -> usize;

    /// Returns number of available rands
    fn num_rands(&self) -> usize;

    /// Returns `num` triples if available
    fn get_triples(&mut self, num: usize) -> Result<Vec<Triple<T>>, MpcError>;

    /// Returns `num` rands if available
    fn get_rands(&mut self, num: usize) -> Result<Vec<AuthAdditiveShare<T>>, MpcError>;
}

/// Client MPC instance
pub struct ClientMPC<T: AuthShare> {
    rands: Vec<AuthAdditiveShare<T>>,
    triples: Vec<Triple<T>>,
}

/// Server MPC instance
pub struct ServerMPC<T: AuthShare> {
    rands: Vec<AuthAdditiveShare<T>>,
    triples: Vec<Triple<T>>,
    mac_key: <T as Share>::Ring,
    /// Opened auth_shares with unchecked MACs
    unchecked: Vec<AuthAdditiveShare<T>>,
}

impl<P: Fp64Parameters> ClientMPC<Fp64<P>> {
    pub fn new(rands: Vec<AuthAdditiveShare<Fp64<P>>>, triples: Vec<Triple<Fp64<P>>>) -> Self {
        Self { rands, triples }
    }
}

impl<P: Fp64Parameters> ServerMPC<Fp64<P>> {
    pub fn new(
        rands: Vec<AuthAdditiveShare<Fp64<P>>>,
        triples: Vec<Triple<Fp64<P>>>,
        mac_key: <Fp64<P> as Share>::Ring,
    ) -> Self {
        Self {
            rands,
            triples,
            mac_key,
            unchecked: Vec::with_capacity(Self::BATCH_SIZE * 100),
        }
    }

    /// Check all AuthAdditiveShares in `unchecked`
    pub fn check_macs(&mut self) -> Result<(), MpcError> {
        if !self
            .unchecked
            .drain(..)
            .collect::<Vec<_>>()
            .par_iter()
            .map(|s| AuthShare::open(*s, &self.mac_key).is_ok())
            .reduce(|| true, |a, b| a && b)
        {
            return Err(MpcError::InvalidMAC);
        }
        Ok(())
    }
}

impl<P: Fp64Parameters> MPC<Fp64<P>, PBeaversMul<P>> for ClientMPC<Fp64<P>> {
    const PARTY_IDX: usize = 2;

    /// Share `inputs` with the server
    fn private_inputs<R: Read + Send + Unpin, W: Write + Send + Unpin, RNG: RngCore + CryptoRng>(
        &mut self,
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        inputs: &[Fp64<P>],
        _: &mut RNG,
    ) -> Result<Vec<AuthAdditiveShare<Fp64<P>>>, MpcError> {
        // Consume necessary random shares
        let rands = self.get_rands(inputs.len())?;
        let (snd, rcv) = crossbeam::channel::unbounded();
        crossbeam::scope(|s| {
            // Receive rand openings
            s.spawn(|_| {
                for r_chunk in rands.chunks(Self::BATCH_SIZE) {
                    snd.send(self.private_recv(reader, r_chunk).unwrap())
                        .unwrap()
                }
                // Need to drop the sending channel so the second thread doesn't
                // block
                drop(snd);
            });
            // Use opened rands to compute epsilon and send to other parties
            for (inp_chunk, open_r_chunk) in izip!(inputs.chunks(Self::BATCH_SIZE), rcv.iter()) {
                let epsilon_vec: Vec<Fp64<P>> = izip!(inp_chunk, open_r_chunk)
                    .map(|(inp, open_r)| {
                        let epsilon = *inp - open_r;
                        epsilon
                    })
                    .collect();
                let send_message = ConstantMsgSend::new(epsilon_vec.as_slice());
                bytes::serialize(&mut *writer, &send_message).unwrap();
            }
        })
        .unwrap();
        Ok(rands)
    }

    /// Receive `num_recv` shares from the server
    fn recv_private_inputs<R: Read + Send + Unpin, W: Write + Send + Unpin>(
        &mut self,
        reader: &mut IMuxAsync<R>,
        _writer: &mut IMuxAsync<W>,
        num_recv: usize,
    ) -> Result<Vec<AuthAdditiveShare<Fp64<P>>>, MpcError> {
        let mut shares = Vec::with_capacity(num_recv);

        // Send shares to client
        // TODO: Thread this
        for _ in 0..((num_recv as f64 / Self::BATCH_SIZE as f64).ceil() as usize) {
            let recv_message: AuthShareRcv<Fp64<P>> = bytes::deserialize(&mut *reader).unwrap();
            shares.extend(recv_message.msg());
        }
        Ok(shares)
    }

    /// To open a share to the server, the client sends full AuthAdditiveShare
    fn private_open<W: Write + Send + Unpin>(
        &self,
        writer: &mut IMuxAsync<W>,
        shares: &[AuthAdditiveShare<Fp64<P>>],
    ) -> Result<(), MpcError> {
        for shares_chunk in shares.chunks(Self::BATCH_SIZE) {
            let send_message = AuthShareSend::new(shares_chunk);
            bytes::serialize(&mut *writer, &send_message)?;
        }
        Ok(())
    }

    /// To receive a share from the server, the client is given an AdditiveShare
    /// which it adds to its AuthAdditiveShare
    fn private_recv<R: Read + Send + Unpin>(
        &mut self,
        reader: &mut IMuxAsync<R>,
        shares: &[AuthAdditiveShare<Fp64<P>>],
    ) -> Result<Vec<Fp64<P>>, MpcError> {
        let mut recv_shares = Vec::with_capacity(shares.len());
        for _ in 0..((shares.len() as f64 / Self::BATCH_SIZE as f64).ceil() as usize) {
            let recv_message: ShareRcv<Fp64<P>> = bytes::deserialize(&mut *reader)?;
            recv_shares.extend(recv_message.msg());
        }
        let result = izip!(shares.iter(), recv_shares.iter())
            .map(|(s1, s2)| (s1 + s2).get_value().inner)
            .collect();
        Ok(result)
    }

    fn public_open<R: Read + Send + Unpin, W: Write + Send + Unpin>(
        &mut self,
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        shares: &[AuthAdditiveShare<Fp64<P>>],
    ) -> Result<Vec<Fp64<P>>, MpcError> {
        self.private_open(writer, shares)?;
        self.private_recv(reader, shares)
    }

    fn num_triples(&self) -> usize {
        self.triples.len()
    }

    fn num_rands(&self) -> usize {
        self.rands.len()
    }

    fn get_triples(&mut self, num: usize) -> Result<Vec<Triple<Fp64<P>>>, MpcError> {
        if self.triples.len() < num {
            return Err(MpcError::InsufficientTriples {
                num: self.triples.len(),
                needed: num,
            });
        }
        Ok(self.triples.split_off(self.triples.len() - num))
    }

    /// Returns `num` rands if available
    fn get_rands(&mut self, num: usize) -> Result<Vec<AuthAdditiveShare<Fp64<P>>>, MpcError> {
        if self.rands.len() < num {
            return Err(MpcError::InsufficientRand {
                num: self.rands.len(),
                needed: num,
            });
        }
        Ok(self.rands.split_off(self.rands.len() - num))
    }
}

impl<P: Fp64Parameters> MPC<Fp64<P>, PBeaversMul<P>> for ServerMPC<Fp64<P>> {
    const PARTY_IDX: usize = 1;

    /// Share `inputs` with the client
    fn private_inputs<R: Read + Send + Unpin, W: Write + Send + Unpin, RNG: RngCore + CryptoRng>(
        &mut self,
        _reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        inputs: &[Fp64<P>],
        rng: &mut RNG,
    ) -> Result<Vec<AuthAdditiveShare<Fp64<P>>>, MpcError> {
        // Secret share all inputs
        let (server_shares, client_shares): (Vec<_>, Vec<_>) = inputs
            .iter()
            .map(|e| e.auth_share(&self.mac_key, rng))
            .unzip();

        // Send shares to client
        // TODO: Thread this
        for client_share in client_shares.chunks(Self::BATCH_SIZE) {
            let send_message = AuthShareSend::new(client_share);
            bytes::serialize(&mut *writer, &send_message)?;
        }
        Ok(server_shares)
    }

    /// Receive `num_recv` shares from the client
    fn recv_private_inputs<R: Read + Send + Unpin, W: Write + Send + Unpin>(
        &mut self,
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        num_recv: usize,
    ) -> Result<Vec<AuthAdditiveShare<Fp64<P>>>, MpcError> {
        // Consume necessary random shares
        let rands = self.get_rands(num_recv)?;
        let mut result = Vec::with_capacity(num_recv);
        crossbeam::scope(|s| {
            // Send rand openings
            s.spawn(|_| {
                for r_chunk in rands.chunks(Self::BATCH_SIZE) {
                    self.private_open(writer, r_chunk).unwrap()
                }
            });
            // Receive epsilon and compute share
            for r_chunk in rands.chunks(Self::BATCH_SIZE) {
                let recv_message: ConstantMsgRcv<Fp64<P>> =
                    bytes::deserialize(&mut *reader).unwrap();
                let epsilon = recv_message.msg();
                izip!(r_chunk, epsilon.iter()).for_each(|(r, e)| result.push(r.add_constant(*e)));
            }
        })
        .unwrap();
        Ok(result)
    }

    /// To open a share to the client, the server sends AdditiveShares
    fn private_open<W: Write + Send + Unpin>(
        &self,
        writer: &mut IMuxAsync<W>,
        shares: &[AuthAdditiveShare<Fp64<P>>],
    ) -> Result<(), MpcError> {
        let stripped_shares: Vec<AdditiveShare<Fp64<P>>> =
            shares.par_iter().map(|e| e.get_value()).collect();
        for shares in stripped_shares.chunks(Self::BATCH_SIZE) {
            let send_message = ShareSend::new(shares);
            bytes::serialize(&mut *writer, &send_message)?;
        }
        Ok(())
    }

    /// To receive a share from the client, the server is sent an
    /// AuthAdditiveShare. It adds the share to `unchecked` and
    /// eagerly opens value
    fn private_recv<R: Read + Send + Unpin>(
        &mut self,
        reader: &mut IMuxAsync<R>,
        shares: &[AuthAdditiveShare<Fp64<P>>],
    ) -> Result<Vec<Fp64<P>>, MpcError> {
        let mut recv_shares = Vec::with_capacity(shares.len());
        for _ in 0..((shares.len() as f64 / Self::BATCH_SIZE as f64).ceil() as usize) {
            let recv_message: AuthShareRcv<Fp64<P>> = bytes::deserialize(&mut *reader)?;
            recv_shares.extend(recv_message.msg());
        }
        let result = izip!(shares.iter(), recv_shares.iter())
            .map(|(s1, s2)| {
                let result = s1 + s2;
                self.unchecked.push(result);
                result.get_value().inner
            })
            .collect();
        Ok(result)
    }

    fn public_open<R: Read + Send + Unpin, W: Write + Send + Unpin>(
        &mut self,
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        shares: &[AuthAdditiveShare<Fp64<P>>],
    ) -> Result<Vec<Fp64<P>>, MpcError> {
        let result = Self::private_recv(self, reader, shares);
        self.private_open(writer, shares)?;
        result
    }

    fn num_triples(&self) -> usize {
        self.triples.len()
    }

    fn num_rands(&self) -> usize {
        self.rands.len()
    }

    fn get_triples(&mut self, num: usize) -> Result<Vec<Triple<Fp64<P>>>, MpcError> {
        if self.triples.len() < num {
            return Err(MpcError::InsufficientTriples {
                num: self.triples.len(),
                needed: num,
            });
        }
        Ok(self.triples.split_off(self.triples.len() - num))
    }

    /// Returns `num` rands if available
    fn get_rands(&mut self, num: usize) -> Result<Vec<AuthAdditiveShare<Fp64<P>>>, MpcError> {
        if self.rands.len() < num {
            return Err(MpcError::InsufficientRand {
                num: self.rands.len(),
                needed: num,
            });
        }
        Ok(self.rands.split_off(self.rands.len() - num))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebra::{fields::near_mersenne_64::F, UniformRandom};
    use async_std::{
        io::{BufReader, BufWriter, Read, Write},
        net::{TcpListener, TcpStream},
        task,
    };
    use crypto_primitives::beavers_mul::InsecureTripleGen;
    use futures::stream::StreamExt;
    use io_utils::imux::IMuxAsync;
    use num_traits::identities::Zero;
    use rand::{Rng, SeedableRng};
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

    fn insecure_gen(
        mac_key: F,
        num_rands: usize,
        num_triples: usize,
    ) -> (
        (Vec<AuthAdditiveShare<F>>, Vec<AuthAdditiveShare<F>>),
        (Vec<Triple<F>>, Vec<Triple<F>>),
    ) {
        let mut gen = InsecureTripleGen::<F>::new(RANDOMNESS);
        let mut triples_1 = Vec::with_capacity(num_triples);
        let mut triples_2 = Vec::with_capacity(num_triples);
        let mut rands_1 = Vec::with_capacity(num_rands);
        let mut rands_2 = Vec::with_capacity(num_rands);
        for _ in 0..num_triples {
            let (t1, t2) = gen.generate_triple_shares(mac_key);
            triples_1.push(t1);
            triples_2.push(t2);
        }
        for _ in 0..num_rands {
            let (t1, t2) = gen.generate_triple_shares(mac_key);
            rands_1.push(t1.a);
            rands_2.push(t2.a);
        }
        ((rands_1, rands_2), (triples_1, triples_2))
    }

    #[test]
    fn test_private_open() {
        let server_addr = "127.0.0.1:8010";
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let mac_key = F::uniform(&mut rng);
        let ((mut client_reader, mut client_writer), (mut server_reader, mut server_writer)) =
            get_connection(server_addr);

        let num_rands = 100000;

        let ((c_rands, s_rands), (..)) = insecure_gen(mac_key, num_rands, 0);
        let mut client_mpc = ClientMPC::new(Vec::new(), Vec::new());
        let mut server_mpc = ServerMPC::new(Vec::new(), Vec::new(), mac_key);

        let client_opened = crossbeam::thread::scope(|s| {
            s.spawn(|_| {
                server_mpc
                    .private_open(&mut server_writer, s_rands.as_slice())
                    .unwrap();
            });
            client_mpc
                .private_recv(&mut client_reader, c_rands.as_slice())
                .unwrap()
        })
        .unwrap();

        let server_opened = crossbeam::thread::scope(|s| {
            s.spawn(|_| {
                client_mpc
                    .private_open(&mut client_writer, c_rands.as_slice())
                    .unwrap()
            });
            server_mpc
                .private_recv(&mut server_reader, s_rands.as_slice())
                .unwrap()
        })
        .unwrap();

        izip!(&c_rands, &s_rands, client_opened).for_each(|(c, s, o)| {
            assert_eq!(
                (c + s).get_value().inner,
                o,
                "Client Opening: Got {:?}, Expected {:?}",
                o,
                (c + s).get_value().inner,
            )
        });
        izip!(c_rands, s_rands, server_opened).for_each(|(c, s, o)| {
            assert_eq!(
                (c + s).get_value().inner,
                o,
                "Server Opening: Got {:?}, Expected {:?}",
                o,
                (c + s).get_value().inner,
            )
        });
    }

    #[test]
    fn test_public_open() {
        let server_addr = "127.0.0.1:8011";
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let mac_key = F::uniform(&mut rng);
        let ((mut client_reader, mut client_writer), (mut server_reader, mut server_writer)) =
            get_connection(server_addr);

        let num_rands = 100000;

        let ((c_rands, s_rands), (..)) = insecure_gen(mac_key, num_rands, 0);
        let mut client_mpc = ClientMPC::new(Vec::new(), Vec::new());
        let mut server_mpc = ServerMPC::new(Vec::new(), Vec::new(), mac_key);

        let (client_opened, server_opened) = crossbeam::thread::scope(|s| {
            let server_result = s.spawn(|_| {
                server_mpc
                    .public_open(&mut server_reader, &mut server_writer, s_rands.as_slice())
                    .unwrap()
            });
            let client_result = s.spawn(|_| {
                client_mpc
                    .public_open(&mut client_reader, &mut client_writer, c_rands.as_slice())
                    .unwrap()
            });
            (client_result.join().unwrap(), server_result.join().unwrap())
        })
        .unwrap();

        izip!(&c_rands, &s_rands, client_opened).for_each(|(c, s, o)| {
            assert_eq!(
                (c + s).get_value().inner,
                o,
                "Client Opening: Got {:?}, Expected {:?}",
                o,
                (c + s).get_value().inner,
            )
        });
        izip!(c_rands, s_rands, server_opened).for_each(|(c, s, o)| {
            assert_eq!(
                (c + s).get_value().inner,
                o,
                "Server Opening: Got {:?}, Expected {:?}",
                o,
                (c + s).get_value().inner,
            )
        });
    }

    #[test]
    fn test_invalid_mac() {
        let server_addr = "127.0.0.1:8012";
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let mac_key = F::uniform(&mut rng);
        let ((mut client_reader, mut client_writer), (mut server_reader, mut server_writer)) =
            get_connection(server_addr);

        let num_rands = 100000;
        let ((mut c_rands, s_rands), (..)) = insecure_gen(mac_key, num_rands, 0);
        let mut client_mpc = ClientMPC::new(Vec::new(), Vec::new());
        let mut server_mpc = ServerMPC::new(Vec::new(), Vec::new(), mac_key);

        // Increment one of the MACs
        c_rands[rng.gen_range(0, num_rands)] +=
            AuthAdditiveShare::new(F::zero(), F::uniform(&mut rng));

        let server_opened = crossbeam::thread::scope(|s| {
            let server_result = s.spawn(|_| {
                server_mpc
                    .public_open(&mut server_reader, &mut server_writer, s_rands.as_slice())
                    .unwrap()
            });
            s.spawn(|_| {
                client_mpc
                    .public_open(&mut client_reader, &mut client_writer, c_rands.as_slice())
                    .unwrap();
            });
            server_result.join().unwrap()
        })
        .unwrap();

        // The opening should be correct
        izip!(&c_rands, &s_rands, server_opened).for_each(|(c, s, o)| {
            assert_eq!(
                (c + s).get_value().inner,
                o,
                "Server Opening: Got {:?}, Expected {:?}",
                o,
                (c + s).get_value().inner,
            )
        });

        // The MAC check should fail
        assert!(server_mpc.check_macs().is_err());
    }

    #[test]
    fn test_private_inputs() {
        let server_addr = "127.0.0.1:8013";
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let mac_key = F::uniform(&mut rng);
        let ((mut client_reader, mut client_writer), (mut server_reader, mut server_writer)) =
            get_connection(server_addr);

        let num_rands = 100000;
        let ((c_rands, s_rands), (..)) = insecure_gen(mac_key, num_rands, 0);
        let mut client_mpc = ClientMPC::new(c_rands, Vec::new());
        let mut server_mpc = ServerMPC::new(s_rands, Vec::new(), mac_key);

        // Generate private inputs
        let mut client_inputs = Vec::with_capacity(num_rands / 2);
        let mut server_inputs = Vec::with_capacity(num_rands / 2);
        for _ in 0..num_rands / 2 {
            client_inputs.push(F::uniform(&mut rng));
            server_inputs.push(F::uniform(&mut rng));
        }

        // Share client private inputs
        let (client_input_1, client_input_2) = crossbeam::thread::scope(|s| {
            let server_result = s.spawn(|_| {
                server_mpc
                    .recv_private_inputs(&mut server_reader, &mut server_writer, num_rands / 2)
                    .unwrap()
            });
            let client_result = s.spawn(|_| {
                client_mpc
                    .private_inputs(
                        &mut client_reader,
                        &mut client_writer,
                        client_inputs.as_slice(),
                        &mut rng,
                    )
                    .unwrap()
            });
            (client_result.join().unwrap(), server_result.join().unwrap())
        })
        .unwrap();

        // Share server private inputs
        let (server_input_1, server_input_2) = crossbeam::thread::scope(|s| {
            let server_result = s.spawn(|_| {
                server_mpc
                    .private_inputs(
                        &mut server_reader,
                        &mut server_writer,
                        server_inputs.as_slice(),
                        &mut rng,
                    )
                    .unwrap()
            });
            let client_result = s.spawn(|_| {
                client_mpc
                    .recv_private_inputs(&mut client_reader, &mut client_writer, num_rands / 2)
                    .unwrap()
            });
            (client_result.join().unwrap(), server_result.join().unwrap())
        })
        .unwrap();

        izip!(client_input_1, client_input_2, client_inputs).for_each(|(s1, s2, e)| {
            let result = s1.combine(&s2, &mac_key);
            assert!(result.is_ok(), "Client inputs: MAC check failure");
            let result = result.unwrap();
            assert_eq!(
                result, e,
                "Client inputs: Got {:?}, Expected {:?}",
                result, e,
            )
        });
        izip!(server_input_1, server_input_2, server_inputs).for_each(|(s1, s2, e)| {
            let result = s1.combine(&s2, &mac_key);
            assert!(result.is_ok(), "Server inputs: MAC check failure");
            let result = result.unwrap();
            assert_eq!(
                result, e,
                "Server inputs: Got {:?}, Expected {:?}",
                result, e,
            )
        });
    }

    #[test]
    fn test_add() {
        let server_addr = "127.0.0.1:8014";
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let mac_key = F::uniform(&mut rng);
        let ((mut client_reader, mut client_writer), (mut server_reader, mut server_writer)) =
            get_connection(server_addr);

        let num_rands = 100000;

        let ((c_rands, s_rands), (..)) = insecure_gen(mac_key, num_rands, 0);
        let mut client_mpc = ClientMPC::new(c_rands, Vec::new());
        let mut server_mpc = ServerMPC::new(s_rands, Vec::new(), mac_key);

        // Generate private inputs
        let mut client_inputs = Vec::with_capacity(num_rands / 2);
        let mut server_inputs = Vec::with_capacity(num_rands / 2);
        for _ in 0..num_rands / 2 {
            client_inputs.push(F::uniform(&mut rng));
            server_inputs.push(F::uniform(&mut rng));
        }

        // Share client private inputs
        let (client_input_1, client_input_2) = crossbeam::thread::scope(|s| {
            let server_result = s.spawn(|_| {
                server_mpc
                    .recv_private_inputs(&mut server_reader, &mut server_writer, num_rands / 2)
                    .unwrap()
            });
            let client_result = s.spawn(|_| {
                client_mpc
                    .private_inputs(
                        &mut client_reader,
                        &mut client_writer,
                        client_inputs.as_slice(),
                        &mut rng,
                    )
                    .unwrap()
            });
            (client_result.join().unwrap(), server_result.join().unwrap())
        })
        .unwrap();

        // Share server private inputs
        let (server_input_1, server_input_2) = crossbeam::thread::scope(|s| {
            let server_result = s.spawn(|_| {
                server_mpc
                    .private_inputs(
                        &mut server_reader,
                        &mut server_writer,
                        server_inputs.as_slice(),
                        &mut rng,
                    )
                    .unwrap()
            });
            let client_result = s.spawn(|_| {
                client_mpc
                    .recv_private_inputs(&mut client_reader, &mut client_writer, num_rands / 2)
                    .unwrap()
            });
            (client_result.join().unwrap(), server_result.join().unwrap())
        })
        .unwrap();

        // Sum shares of inputs
        let sum_1 = client_mpc
            .add(client_input_1.as_slice(), server_input_1.as_slice())
            .unwrap();
        let sum_2 = server_mpc
            .add(client_input_2.as_slice(), server_input_2.as_slice())
            .unwrap();

        izip!(sum_1, sum_2, client_inputs, server_inputs).for_each(|(s1, s2, c, s)| {
            let result = s1.combine(&s2, &mac_key);
            assert!(result.is_ok(), "MAC check failure");
            let result = result.unwrap();
            assert_eq!(result, c + s, "Got {:?}, Expected {:?}", result, c + s,)
        });
    }

    #[test]
    fn test_sub() {
        let server_addr = "127.0.0.1:8015";
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let mac_key = F::uniform(&mut rng);
        let ((mut client_reader, mut client_writer), (mut server_reader, mut server_writer)) =
            get_connection(server_addr);

        let num_rands = 100000;

        let ((c_rands, s_rands), (..)) = insecure_gen(mac_key, num_rands, 0);
        let mut client_mpc = ClientMPC::new(c_rands, Vec::new());
        let mut server_mpc = ServerMPC::new(s_rands, Vec::new(), mac_key);

        // Generate private inputs
        let mut client_inputs = Vec::with_capacity(num_rands / 2);
        let mut server_inputs = Vec::with_capacity(num_rands / 2);
        for _ in 0..num_rands / 2 {
            client_inputs.push(F::uniform(&mut rng));
            server_inputs.push(F::uniform(&mut rng));
        }

        // Share client private inputs
        let (client_input_1, client_input_2) = crossbeam::thread::scope(|s| {
            let server_result = s.spawn(|_| {
                server_mpc
                    .recv_private_inputs(&mut server_reader, &mut server_writer, num_rands / 2)
                    .unwrap()
            });
            let client_result = s.spawn(|_| {
                client_mpc
                    .private_inputs(
                        &mut client_reader,
                        &mut client_writer,
                        client_inputs.as_slice(),
                        &mut rng,
                    )
                    .unwrap()
            });
            (client_result.join().unwrap(), server_result.join().unwrap())
        })
        .unwrap();

        // Share server private inputs
        let (server_input_1, server_input_2) = crossbeam::thread::scope(|s| {
            let server_result = s.spawn(|_| {
                server_mpc
                    .private_inputs(
                        &mut server_reader,
                        &mut server_writer,
                        server_inputs.as_slice(),
                        &mut rng,
                    )
                    .unwrap()
            });
            let client_result = s.spawn(|_| {
                client_mpc
                    .recv_private_inputs(&mut client_reader, &mut client_writer, num_rands / 2)
                    .unwrap()
            });
            (client_result.join().unwrap(), server_result.join().unwrap())
        })
        .unwrap();

        // Sum shares of inputs
        let sum_1 = client_mpc
            .sub(client_input_1.as_slice(), server_input_1.as_slice())
            .unwrap();
        let sum_2 = server_mpc
            .sub(client_input_2.as_slice(), server_input_2.as_slice())
            .unwrap();

        izip!(sum_1, sum_2, client_inputs, server_inputs).for_each(|(s1, s2, c, s)| {
            let result = s1.combine(&s2, &mac_key);
            assert!(result.is_ok(), "MAC check failure");
            let result = result.unwrap();
            assert_eq!(result, c - s, "Got {:?}, Expected {:?}", result, c - s,)
        });
    }

    #[test]
    fn test_mul() {
        let server_addr = "127.0.0.1:8016";
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let mac_key = F::uniform(&mut rng);
        let ((mut client_reader, mut client_writer), (mut server_reader, mut server_writer)) =
            get_connection(server_addr);

        let num_rands = 100000;
        let num_triples = num_rands / 2;

        let ((c_rands, s_rands), (c_triples, s_triples)) =
            insecure_gen(mac_key, num_rands, num_triples);
        let mut client_mpc = ClientMPC::new(c_rands, c_triples);
        let mut server_mpc = ServerMPC::new(s_rands, s_triples, mac_key);

        // Generate private inputs
        let mut client_inputs = Vec::with_capacity(num_rands / 2);
        let mut server_inputs = Vec::with_capacity(num_rands / 2);
        for _ in 0..num_rands / 2 {
            client_inputs.push(F::uniform(&mut rng));
            server_inputs.push(F::uniform(&mut rng));
        }

        // Share client private inputs
        let (client_input_1, client_input_2) = crossbeam::thread::scope(|s| {
            let server_result = s.spawn(|_| {
                server_mpc
                    .recv_private_inputs(&mut server_reader, &mut server_writer, num_rands / 2)
                    .unwrap()
            });
            let client_result = s.spawn(|_| {
                client_mpc
                    .private_inputs(
                        &mut client_reader,
                        &mut client_writer,
                        client_inputs.as_slice(),
                        &mut rng,
                    )
                    .unwrap()
            });
            (client_result.join().unwrap(), server_result.join().unwrap())
        })
        .unwrap();

        // Share server private inputs
        let (server_input_1, server_input_2) = crossbeam::thread::scope(|s| {
            let server_result = s.spawn(|_| {
                server_mpc
                    .private_inputs(
                        &mut server_reader,
                        &mut server_writer,
                        server_inputs.as_slice(),
                        &mut rng,
                    )
                    .unwrap()
            });
            let client_result = s.spawn(|_| {
                client_mpc
                    .recv_private_inputs(&mut client_reader, &mut client_writer, num_rands / 2)
                    .unwrap()
            });
            (client_result.join().unwrap(), server_result.join().unwrap())
        })
        .unwrap();

        // Multiply shares of inputs
        let (mul_1, mul_2) = crossbeam::thread::scope(|s| {
            let client_result = s.spawn(|_| {
                client_mpc
                    .mul(
                        &mut client_reader,
                        &mut client_writer,
                        client_input_1.as_slice(),
                        server_input_1.as_slice(),
                    )
                    .unwrap()
            });
            let server_result = s.spawn(|_| {
                server_mpc
                    .mul(
                        &mut server_reader,
                        &mut server_writer,
                        client_input_2.as_slice(),
                        server_input_2.as_slice(),
                    )
                    .unwrap()
            });
            (client_result.join().unwrap(), server_result.join().unwrap())
        })
        .unwrap();

        izip!(mul_1, mul_2, client_inputs, server_inputs).for_each(|(s1, s2, c, s)| {
            let result = s1.combine(&s2, &mac_key);
            assert!(result.is_ok(), "MAC check failure");
            let result = result.unwrap();
            assert_eq!(result, c * s, "Got {:?}, Expected {:?}", result, c * s,)
        });
    }

    #[test]
    fn test_basic_circuit() {
        let server_addr = "127.0.0.1:8017";
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let mac_key = F::uniform(&mut rng);
        let ((mut client_reader, mut client_writer), (mut server_reader, mut server_writer)) =
            get_connection(server_addr);

        let num_rands = 100000;
        let num_triples = num_rands / 2;

        // Generate private inputs
        let mut client_inputs = Vec::with_capacity(num_rands / 2);
        let mut server_inputs = Vec::with_capacity(num_rands / 2);
        let mut adders = Vec::with_capacity(num_rands);
        let mut multipliers = Vec::with_capacity(num_rands);
        for _ in 0..num_rands / 2 {
            client_inputs.push(F::uniform(&mut rng));
            server_inputs.push(F::uniform(&mut rng));
            adders.push(F::uniform(&mut rng));
            multipliers.push(F::uniform(&mut rng));
        }

        let ((c_rands, s_rands), (c_triples, s_triples)) =
            insecure_gen(mac_key, num_rands, num_triples);
        let mut client_mpc = ClientMPC::new(c_rands, c_triples);
        let mut server_mpc = ServerMPC::new(s_rands, s_triples, mac_key);

        // Share client private inputs
        let (client_input_1, client_input_2) = crossbeam::thread::scope(|s| {
            let server_result = s.spawn(|_| {
                server_mpc
                    .recv_private_inputs(&mut server_reader, &mut server_writer, num_rands / 2)
                    .unwrap()
            });
            let client_result = s.spawn(|_| {
                client_mpc
                    .private_inputs(
                        &mut client_reader,
                        &mut client_writer,
                        client_inputs.as_slice(),
                        &mut rng,
                    )
                    .unwrap()
            });
            (client_result.join().unwrap(), server_result.join().unwrap())
        })
        .unwrap();

        // Share server private inputs
        let (server_input_1, server_input_2) = crossbeam::thread::scope(|s| {
            let server_result = s.spawn(|_| {
                server_mpc
                    .private_inputs(
                        &mut server_reader,
                        &mut server_writer,
                        server_inputs.as_slice(),
                        &mut rng,
                    )
                    .unwrap()
            });
            let client_result = s.spawn(|_| {
                client_mpc
                    .recv_private_inputs(&mut client_reader, &mut client_writer, num_rands / 2)
                    .unwrap()
            });
            (client_result.join().unwrap(), server_result.join().unwrap())
        })
        .unwrap();

        // Multiply shares of inputs
        let (mul_1, mul_2) = crossbeam::thread::scope(|s| {
            let client_result = s.spawn(|_| {
                client_mpc
                    .mul(
                        &mut client_reader,
                        &mut client_writer,
                        client_input_1.as_slice(),
                        server_input_1.as_slice(),
                    )
                    .unwrap()
            });
            let server_result = s.spawn(|_| {
                server_mpc
                    .mul(
                        &mut server_reader,
                        &mut server_writer,
                        client_input_2.as_slice(),
                        server_input_2.as_slice(),
                    )
                    .unwrap()
            });
            (client_result.join().unwrap(), server_result.join().unwrap())
        })
        .unwrap();

        // Add constant to shares
        let added = client_mpc
            .add_const(mul_1.as_slice(), adders.as_slice())
            .unwrap();

        // Multiply shares by a constant
        let mul_1 = client_mpc
            .mul_const(added.as_slice(), multipliers.as_slice())
            .unwrap();
        let mul_2 = server_mpc
            .mul_const(mul_2.as_slice(), multipliers.as_slice())
            .unwrap();

        // Sum all inputs together
        let res_1 = client_mpc.sum(mul_1.as_slice());
        let res_2 = server_mpc.sum(mul_2.as_slice());

        let expected = izip!(client_inputs, server_inputs)
            .map(|(c, s)| c * s)
            .zip(adders.iter())
            .map(|(s, a)| s + a)
            .zip(multipliers.iter())
            .map(|(s, m)| s * m)
            .fold(F::zero(), |a, b| a + b);

        let result = res_1.combine(&res_2, &mac_key);
        assert!(result.is_ok(), "MAC check failure");
        let result = result.unwrap();
        assert_eq!(
            result, expected,
            "Got {:?}, Expected {:?}",
            result, expected
        )
    }
}
