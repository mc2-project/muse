use crate::additive_share::{AuthAdditiveShare, AuthShare, Share};
use algebra::{Fp64, Fp64Parameters, PrimeField};
use rand_chacha::ChaChaRng;
use rand_core::SeedableRng;
use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, ops::Neg};
/// Shares of a triple `[[a]]`, `[[b]]`, `[[c]]` such that `ab = c`.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "T: AuthShare")]
pub struct Triple<T: AuthShare> {
    /// A share of the `a` part of the triple.
    pub a: AuthAdditiveShare<T>,
    /// A share of the `b` part of the triple.
    pub b: AuthAdditiveShare<T>,
    /// A share of the `c` part of the triple.
    pub c: AuthAdditiveShare<T>,
}

/// Shares of the intermediate step.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(bound = "T: AuthShare")]
pub struct BlindedSharedInputs<T: AuthShare> {
    /// A share of the `x-a`.
    pub blinded_x: AuthAdditiveShare<T>,
    /// A share of the `y-b`.
    pub blinded_y: AuthAdditiveShare<T>,
}

/// Result of combining shares in `BlindedSharedInput`.
#[derive(Serialize, Deserialize)]
#[serde(bound = "T: Share")]
pub struct BlindedInputs<T: Share> {
    /// `x-a`.
    pub blinded_x: T,
    /// `y-b`.
    pub blinded_y: T,
}

/// Objects that can be multiplied via Beaver's triples protocols must implement
/// this trait.
pub trait BeaversMul<T: AuthShare> {
    /// Share inputs by consuming a triple. Each party should independently
    /// process the `BlindedSharedInputs` into `BlindedInputs`
    fn share_and_blind_inputs(
        x: &AuthAdditiveShare<T>,
        y: &AuthAdditiveShare<T>,
        triple: &Triple<T>,
    ) -> BlindedSharedInputs<T> {
        BlindedSharedInputs {
            blinded_x: x + &triple.a.neg(),
            blinded_y: y + &triple.b.neg(),
        }
    }

    /// Multiply blinded inputs.
    fn multiply_blinded_inputs(
        party_index: usize,
        bl: BlindedInputs<T>,
        t: &Triple<T>,
    ) -> AuthAdditiveShare<T>;
}

/// An implementation of Beaver's multiplication algorithm for a malicious
/// client with shares over Fp64 `P`
pub struct PBeaversMul<P: Fp64Parameters>(PhantomData<P>);

impl<P: Fp64Parameters> BeaversMul<Fp64<P>> for PBeaversMul<P> {
    fn multiply_blinded_inputs(
        party_index: usize,
        bl: BlindedInputs<Fp64<P>>,
        t: &Triple<Fp64<P>>,
    ) -> AuthAdditiveShare<Fp64<P>> {
        let epsilon = bl.blinded_x;
        let gamma = bl.blinded_y;
        let res = t.c + (t.b * epsilon) + (t.a * gamma);
        if party_index == 1 {
            res.add_constant(epsilon * gamma)
        } else {
            res
        }
    }
}

/// An **insecure** method of generating triples. This is intended *purely* for
/// testing purposes.
pub struct InsecureTripleGen<T: AuthShare>(ChaChaRng, PhantomData<T>);

impl<T: AuthShare> InsecureTripleGen<T> {
    /// Create a new `Self` from a random seed.
    pub fn new(seed: [u8; 32]) -> Self {
        Self(ChaChaRng::from_seed(seed), PhantomData)
    }
}

impl<T: AuthShare> InsecureTripleGen<T>
where
    T: PrimeField,
{
    /// Sample a triple for both parties.
    pub fn generate_triple_shares(
        &mut self,
        mac_key: <T as Share>::Ring,
    ) -> (Triple<T>, Triple<T>) {
        let a = T::uniform(&mut self.0);
        let b = T::uniform(&mut self.0);
        let c = a * b;
        let (a1, a2) = a.auth_share(&mac_key, &mut self.0);
        let (b1, b2) = b.auth_share(&mac_key, &mut self.0);
        let (c1, c2) = c.auth_share(&mac_key, &mut self.0);

        let party_1_triple = Triple {
            a: a1,
            b: b1,
            c: c1,
        };

        let party_2_triple = Triple {
            a: a2,
            b: b2,
            c: c2,
        };
        (party_1_triple, party_2_triple)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebra::{fields::near_mersenne_64::F, UniformRandom};

    const RANDOMNESS: [u8; 32] = [
        0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4,
        0x76, 0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a,
        0x52, 0xd2,
    ];

    #[test]
    fn test_triple_gen() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let mut gen = InsecureTripleGen::<F>::new(RANDOMNESS);
        for _ in 0..1000 {
            let mac_key = F::uniform(&mut rng);
            let (t1, t2) = gen.generate_triple_shares(mac_key);
            let a = t1.a.combine(&t2.a, &mac_key).unwrap();
            let b = t1.b.combine(&t2.b, &mac_key).unwrap();
            let c = t1.c.combine(&t2.c, &mac_key).unwrap();
            assert_eq!(a * b, c);
        }
    }

    #[test]
    fn test_share_and_blind() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let seed = RANDOMNESS;
        let mut gen = InsecureTripleGen::<F>::new(seed);
        for _ in 0..1000 {
            let mac_key = F::uniform(&mut rng);
            let n1 = F::uniform(&mut rng);
            let n2 = F::uniform(&mut rng);

            let (t1, t2) = gen.generate_triple_shares(mac_key);
            let (s11, s12) = n1.auth_share(&mac_key, &mut rng);
            let (s21, s22) = n2.auth_share(&mac_key, &mut rng);
            let p1_bl_auth = PBeaversMul::share_and_blind_inputs(&s11, &s21, &t1);
            let p2_bl_auth = PBeaversMul::share_and_blind_inputs(&s12, &s22, &t2);

            let a = t1.a.combine(&t2.a, &mac_key).unwrap();
            let b = t1.b.combine(&t2.b, &mac_key).unwrap();

            assert_eq!(
                p1_bl_auth
                    .blinded_x
                    .combine(&p2_bl_auth.blinded_x, &mac_key)
                    .unwrap(),
                n1 - a
            );
            assert_eq!(
                p1_bl_auth
                    .blinded_y
                    .combine(&p2_bl_auth.blinded_y, &mac_key)
                    .unwrap(),
                n2 - b
            );
        }
    }

    #[test]
    fn test_beavers_mul() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let seed = RANDOMNESS;
        let mut gen = InsecureTripleGen::<F>::new(seed);
        for _ in 0..1000 {
            let mac_key = F::uniform(&mut rng);
            let n1 = F::uniform(&mut rng);
            let n2 = F::uniform(&mut rng);
            let n3 = n1 * n2;
            let (t1, t2) = gen.generate_triple_shares(mac_key);
            let (s11, s12) = n1.auth_share(&mac_key, &mut rng);
            let (s21, s22) = n2.auth_share(&mac_key, &mut rng);

            let p1_bl_auth = PBeaversMul::share_and_blind_inputs(&s11, &s21, &t1);
            let p2_bl_auth = PBeaversMul::share_and_blind_inputs(&s12, &s22, &t2);

            // Both parties conver to BlindedInputs. Server checks MAC, Client doesn't.
            let p1_bl_input = BlindedInputs {
                blinded_x: p1_bl_auth
                    .blinded_x
                    .combine(&p2_bl_auth.blinded_x, &mac_key)
                    .unwrap(),
                blinded_y: p1_bl_auth
                    .blinded_y
                    .combine(&p2_bl_auth.blinded_y, &mac_key)
                    .unwrap(),
            };

            let p2_bl_input = BlindedInputs {
                blinded_x: (p1_bl_auth.blinded_x + p2_bl_auth.blinded_x)
                    .get_value()
                    .inner,
                blinded_y: (p1_bl_auth.blinded_y + p2_bl_auth.blinded_y)
                    .get_value()
                    .inner,
            };

            let s31 = PBeaversMul::multiply_blinded_inputs(1, p1_bl_input, &t1);
            let s32 = PBeaversMul::multiply_blinded_inputs(2, p2_bl_input, &t2);
            let n4 = s31.combine(&s32, &mac_key).unwrap();
            assert_eq!(
                n4, n3,
                "test failed with n1 = {:?}, n2 = {:?}, n3 = {:?}",
                n1, n2, n3
            );
        }
    }
}
