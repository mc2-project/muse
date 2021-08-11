use crate::{
    biginteger::BigInteger64 as BigInteger,
    fields::{Fp64, Fp64Parameters, FpParameters},
};

pub type F = Fp64<FParameters>;

pub struct FParameters;

impl Fp64Parameters for FParameters {}
impl FpParameters for FParameters {
    type BigInt = BigInteger;

    const MODULUS: BigInteger = BigInteger(17592060215297);

    const MODULUS_BITS: u32 = 44u32;

    const CAPACITY: u32 = Self::MODULUS_BITS - 1;

    const REPR_SHAVE_BITS: u32 = 20;

    const R: BigInteger = BigInteger(8796972777465);

    const R2: BigInteger = BigInteger(11839330007784);

    const INV: u64 = 18430928698329792511;

    const GENERATOR: BigInteger = BigInteger(7u64);

    const TWO_ADICITY: u32 = 23;

    const MODULUS_MINUS_ONE_DIV_TWO: BigInteger = BigInteger(8796030107648);

    const ROOT_OF_UNITY: BigInteger = BigInteger(11527399278657);

    const T: BigInteger = BigInteger(17623);

    const T_MINUS_ONE_DIV_TWO: BigInteger = BigInteger(8811);
}

#[cfg(test)]
mod tests;
