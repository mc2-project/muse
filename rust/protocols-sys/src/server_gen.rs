use crate::*;
use std::os::raw::c_char;

/// Represents a type which does pairwise randomness and triple generation as a
/// client for client-malicious SPDZ
pub trait ServerGen {
    type Keys;
    /// The type of messages passed between client and server
    type MsgType;

    /// Create new ServerGen object
    fn new(keys: Self::Keys, mac_key: u64) -> Self;

    /// Preprocess `a`, `b`, and `c` randomizers, shares, and MAC shares
    fn triples_preprocess(
        &self,
        a_rand: &[u64],
        b_rand: &[u64],
        c_rand: &[u64],
        a_share: &[u64],
        b_share: &[u64],
        c_share: &[u64],
        a_mac_share: &[u64],
        b_mac_share: &[u64],
        c_mac_share: &[u64],
    ) -> ServerTriples;

    /// Preprocess `r` randomizer, share, and MAC share
    fn rands_preprocess(&self, rand: &[u64], share: &[u64], mac_share: &[u64]) -> ServerTriples;

    /// Process clients's input and return `a`, `b`, `c` shares and MAC shares
    /// for client
    fn triples_online(
        &self,
        shares: &mut ServerTriples,
        a: &mut [Self::MsgType],
        b: &mut [Self::MsgType],
    ) -> (
        Vec<Self::MsgType>,
        Vec<Self::MsgType>,
        Vec<Self::MsgType>,
        Vec<Self::MsgType>,
        Vec<Self::MsgType>,
        Vec<Self::MsgType>,
    );

    /// Process client's input and return `r` share and MAC share for client
    fn rands_online(
        &self,
        shares: &mut ServerTriples,
        r: &mut [Self::MsgType],
    ) -> (Vec<Self::MsgType>, Vec<Self::MsgType>);
}

/// SEAL implementation of ClientGen
pub struct SealServerGen<'a> {
    sfhe: &'a ServerFHE,
    pub mac_key: u64, // TODO: This is just pub for insecure
}

impl<'a> ServerGen for SealServerGen<'a> {
    type Keys = &'a ServerFHE;
    /// Messages are SEAL ciphertexts which are passed as opaque C pointers
    type MsgType = c_char;

    fn new(sfhe: Self::Keys, mac_key: u64) -> Self {
        Self { sfhe, mac_key }
    }

    fn triples_preprocess(
        &self,
        a_rand: &[u64],
        b_rand: &[u64],
        c_rand: &[u64],
        a_share: &[u64],
        b_share: &[u64],
        c_share: &[u64],
        a_mac_share: &[u64],
        b_mac_share: &[u64],
        c_mac_share: &[u64],
    ) -> ServerTriples {
        unsafe {
            server_triples_preprocess(
                self.sfhe,
                a_rand.len() as u32,
                a_rand.as_ptr(),
                b_rand.as_ptr(),
                c_rand.as_ptr(),
                a_share.as_ptr(),
                b_share.as_ptr(),
                c_share.as_ptr(),
                a_mac_share.as_ptr(),
                b_mac_share.as_ptr(),
                c_mac_share.as_ptr(),
                self.mac_key,
            )
        }
    }

    fn rands_preprocess(&self, rand: &[u64], share: &[u64], mac_share: &[u64]) -> ServerTriples {
        unsafe {
            server_rand_preprocess(
                self.sfhe,
                rand.len() as u32,
                rand.as_ptr(),
                share.as_ptr(),
                mac_share.as_ptr(),
                self.mac_key,
            )
        }
    }

    fn triples_online(
        &self,
        shares: &mut ServerTriples,
        a: &mut [Self::MsgType],
        b: &mut [Self::MsgType],
    ) -> (
        Vec<Self::MsgType>,
        Vec<Self::MsgType>,
        Vec<Self::MsgType>,
        Vec<Self::MsgType>,
        Vec<Self::MsgType>,
        Vec<Self::MsgType>,
    ) {
        let a_ct = SerialCT {
            inner: a.as_mut_ptr(),
            size: a.len() as u64,
        };
        let b_ct = SerialCT {
            inner: b.as_mut_ptr(),
            size: b.len() as u64,
        };
        unsafe { server_triples_online(self.sfhe, a_ct, b_ct, shares) };
        let result = unsafe {
            (
                std::slice::from_raw_parts(shares.a_ct.inner, shares.a_ct.size as usize).to_vec(),
                std::slice::from_raw_parts(shares.b_ct.inner, shares.b_ct.size as usize).to_vec(),
                std::slice::from_raw_parts(shares.c_ct.inner, shares.c_ct.size as usize).to_vec(),
                std::slice::from_raw_parts(shares.a_mac_ct.inner, shares.a_mac_ct.size as usize)
                    .to_vec(),
                std::slice::from_raw_parts(shares.b_mac_ct.inner, shares.b_mac_ct.size as usize)
                    .to_vec(),
                std::slice::from_raw_parts(shares.c_mac_ct.inner, shares.c_mac_ct.size as usize)
                    .to_vec(),
            )
        };
        unsafe { server_triples_free(shares) };
        result
    }

    fn rands_online(
        &self,
        shares: &mut ServerTriples,
        r: &mut [Self::MsgType],
    ) -> (Vec<Self::MsgType>, Vec<Self::MsgType>) {
        let r_ct = SerialCT {
            inner: r.as_mut_ptr(),
            size: r.len() as u64,
        };
        unsafe { server_rand_online(self.sfhe, r_ct, shares) };
        let result = unsafe {
            (
                std::slice::from_raw_parts(shares.a_ct.inner, shares.a_ct.size as usize).to_vec(),
                std::slice::from_raw_parts(shares.a_mac_ct.inner, shares.a_mac_ct.size as usize)
                    .to_vec(),
            )
        };
        unsafe { server_rand_free(shares) };
        result
    }
}
