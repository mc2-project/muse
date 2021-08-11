#![allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code
)]

#[macro_use]
pub extern crate ndarray;

pub mod client_acg;
pub mod client_gen;
pub mod key_share;
pub mod server_acg;
pub mod server_gen;

pub use client_acg::*;
pub use client_gen::*;
pub use key_share::KeyShare;
pub use server_acg::*;
pub use server_gen::*;

use std::os::raw::c_char;

include!("bindings.rs");

pub struct SealCT {
    pub inner: SerialCT,
}

impl SealCT {
    pub fn new() -> Self {
        let inner = SerialCT {
            inner: std::ptr::null_mut(),
            size: 0,
        };
        Self { inner }
    }

    /// TODO
    pub fn input_auth(
        &mut self,
        sfhe: &ServerFHE,
        mac_key: &mut SealCT,
        input: Vec<u64>,
        rand: Vec<u64>,
    ) -> Vec<c_char> {
        self.inner = unsafe {
            client_input_auth(
                sfhe,
                &mut mac_key.inner,
                input.as_ptr(),
                rand.as_ptr(),
                input.len() as u64,
            )
        };

        unsafe { std::slice::from_raw_parts(self.inner.inner, self.inner.size as usize).to_vec() }
    }

    /// Encrypt a vector using SEAL
    pub fn encrypt_vec(&mut self, cfhe: &ClientFHE, input: Vec<u64>) -> Vec<c_char> {
        self.inner = unsafe { encrypt_vec(cfhe, input.as_ptr(), input.len() as u64) };

        unsafe { std::slice::from_raw_parts(self.inner.inner, self.inner.size as usize).to_vec() }
    }

    /// Decrypt a vector of SEAL ciphertexts. Assumes `inner.share_size` is set.
    pub fn decrypt_vec(&mut self, cfhe: &ClientFHE, mut ct: Vec<c_char>, size: usize) -> Vec<u64> {
        // Don't replace the current inner CT, since the received ciphertext was
        // allocated by Rust
        let mut recv_ct = SerialCT {
            inner: ct.as_mut_ptr(),
            size: ct.len() as u64,
        };
        unsafe {
            let raw_vec = decrypt_vec(cfhe, &mut recv_ct, size as u64);
            std::slice::from_raw_parts(raw_vec, size as usize).to_vec()
        }
    }

    /// Generates MAC shares of given SEAL ciphertexts. Assumes `inner.share`
    /// and `inner.share_size` are set.
    pub fn gen_mac_share(
        &mut self,
        sfhe: &ServerFHE,
        share: Vec<u64>,
        mac_key: u64,
    ) -> Vec<c_char> {
        // Swap out the inner CT, since the current inner was allocated by Rust. This
        // way we keep a reference to the C++ allocation and Drop properly
        self.inner = unsafe {
            server_mac_ct(
                sfhe,
                &mut self.inner,
                share.as_ptr(),
                share.len() as u64,
                mac_key,
            )
        };
        unsafe { std::slice::from_raw_parts(self.inner.inner, self.inner.size as usize).to_vec() }
    }
}

//impl Drop for SealCT {
//    fn drop(&mut self) {
//        unsafe { free_ct(&mut self.inner) }
//    }
//}

impl Drop for ClientFHE {
    fn drop(&mut self) {
        unsafe {
            client_free_keys(self);
        }
    }
}
unsafe impl Send for ClientFHE {}
unsafe impl Sync for ClientFHE {}

impl Drop for ServerFHE {
    fn drop(&mut self) {
        unsafe {
            server_free_keys(self);
        }
    }
}
unsafe impl Send for ServerFHE {}
unsafe impl Sync for ServerFHE {}

unsafe impl Send for ClientTriples {}
unsafe impl Sync for ClientTriples {}

unsafe impl Send for ServerTriples {}
unsafe impl Sync for ServerTriples {}
