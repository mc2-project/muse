use crate::*;
use std::os::raw::c_char;

/// Represents a type which does pairwise randomness and triple generation as a
/// client for client-malicious SPDZ
pub trait ClientGen {
    type Keys;
    /// The type of messages passed between client and server
    type MsgType;

    /// Create new ClientGen object
    fn new(keys: Self::Keys) -> Self;

    /// Preprocess `a` and `b` randomizers for sending to the server
    fn triples_preprocess(
        &self,
        a_rand: &[u64],
        b_rand: &[u64],
    ) -> (ClientTriples, Vec<Self::MsgType>, Vec<Self::MsgType>);

    /// Preprocess `r` randomizer for sending to the server
    fn rands_preprocess(&self, rand: &[u64]) -> (ClientTriples, Vec<Self::MsgType>);

    /// Free ciphertexts that were sent to the server
    fn triples_free_ct(&self, shares: &mut ClientTriples);

    /// Free ciphertexts that were sent to the server
    fn rands_free_ct(&self, shares: &mut ClientTriples);

    /// Postprocess server's response and return `a`, `b`, `c` shares and MAC
    /// shares
    fn triples_postprocess(
        &self,
        shares: &mut ClientTriples,
        a: &mut [Self::MsgType],
        b: &mut [Self::MsgType],
        c: &mut [Self::MsgType],
        a_mac: &mut [Self::MsgType],
        b_mac: &mut [Self::MsgType],
        c_mac: &mut [Self::MsgType],
    ) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>);

    /// Postprocess server's response and return `r` share and MAC share
    fn rands_postprocess(
        &self,
        shares: &mut ClientTriples,
        r: &mut [Self::MsgType],
        r_mac: &mut [Self::MsgType],
    ) -> (Vec<u64>, Vec<u64>);
}

/// SEAL implementation of ClientGen
pub struct SealClientGen<'a> {
    cfhe: &'a ClientFHE,
}

impl<'a> ClientGen for SealClientGen<'a> {
    type Keys = &'a ClientFHE;
    /// Messages are SEAL ciphertexts which are passed as opaque C pointers
    type MsgType = c_char;

    fn new(cfhe: Self::Keys) -> Self {
        Self { cfhe }
    }

    fn triples_preprocess(
        &self,
        a_rand: &[u64],
        b_rand: &[u64],
    ) -> (ClientTriples, Vec<c_char>, Vec<c_char>) {
        let shares = unsafe {
            client_triples_preprocess(
                self.cfhe,
                a_rand.len() as u32,
                a_rand.as_ptr(),
                b_rand.as_ptr(),
            )
        };
        unsafe {
            (
                shares,
                std::slice::from_raw_parts(shares.a_ct.inner, shares.a_ct.size as usize).to_vec(),
                std::slice::from_raw_parts(shares.b_ct.inner, shares.b_ct.size as usize).to_vec(),
            )
        }
    }

    fn rands_preprocess(&self, rand: &[u64]) -> (ClientTriples, Vec<c_char>) {
        let shares = unsafe { client_rand_preprocess(self.cfhe, rand.len() as u32, rand.as_ptr()) };
        (shares, unsafe {
            std::slice::from_raw_parts(shares.a_ct.inner, shares.a_ct.size as usize).to_vec()
        })
    }

    fn triples_free_ct(&self, shares: &mut ClientTriples) {
        unsafe {
            free_ct(&mut shares.a_ct);
            free_ct(&mut shares.b_ct);
        }
    }

    fn rands_free_ct(&self, shares: &mut ClientTriples) {
        unsafe {
            free_ct(&mut shares.a_ct);
        }
    }

    fn triples_postprocess(
        &self,
        shares: &mut ClientTriples,
        a: &mut [c_char],
        b: &mut [c_char],
        c: &mut [c_char],
        a_mac: &mut [c_char],
        b_mac: &mut [c_char],
        c_mac: &mut [c_char],
    ) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
        let a_ct = SerialCT {
            inner: a.as_mut_ptr(),
            size: a.len() as u64,
        };
        let b_ct = SerialCT {
            inner: b.as_mut_ptr(),
            size: b.len() as u64,
        };
        let c_ct = SerialCT {
            inner: c.as_mut_ptr(),
            size: c.len() as u64,
        };
        let a_mac_ct = SerialCT {
            inner: a_mac.as_mut_ptr(),
            size: a_mac.len() as u64,
        };
        let b_mac_ct = SerialCT {
            inner: b_mac.as_mut_ptr(),
            size: b_mac.len() as u64,
        };
        let c_mac_ct = SerialCT {
            inner: c_mac.as_mut_ptr(),
            size: c_mac.len() as u64,
        };
        unsafe {
            client_triples_decrypt(
                self.cfhe, a_ct, b_ct, c_ct, a_mac_ct, b_mac_ct, c_mac_ct, shares,
            );
        };
        let result = unsafe {
            (
                std::slice::from_raw_parts(shares.a_share, shares.num as usize).to_vec(),
                std::slice::from_raw_parts(shares.b_share, shares.num as usize).to_vec(),
                std::slice::from_raw_parts(shares.c_share, shares.num as usize).to_vec(),
                std::slice::from_raw_parts(shares.a_mac_share, shares.num as usize).to_vec(),
                std::slice::from_raw_parts(shares.b_mac_share, shares.num as usize).to_vec(),
                std::slice::from_raw_parts(shares.c_mac_share, shares.num as usize).to_vec(),
            )
        };
        // Drop all C++ shares
        unsafe { client_triples_free(shares) };
        result
    }

    fn rands_postprocess(
        &self,
        shares: &mut ClientTriples,
        r: &mut [c_char],
        r_mac: &mut [c_char],
    ) -> (Vec<u64>, Vec<u64>) {
        let r_ct = SerialCT {
            inner: r.as_mut_ptr(),
            size: r.len() as u64,
        };
        let r_mac_ct = SerialCT {
            inner: r_mac.as_mut_ptr(),
            size: r_mac.len() as u64,
        };
        unsafe {
            client_rand_decrypt(self.cfhe, r_ct, r_mac_ct, shares);
        };
        let result = unsafe {
            (
                std::slice::from_raw_parts(shares.a_share, shares.num as usize).to_vec(),
                std::slice::from_raw_parts(shares.a_mac_share, shares.num as usize).to_vec(),
            )
        };
        // Drop all C++ shares
        unsafe { client_rand_free(shares) };
        result
    }
}
