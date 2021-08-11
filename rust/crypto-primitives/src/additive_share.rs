use algebra::{
    fields::{Fp64, Fp64Parameters},
    fixed_point::{FixedPoint, FixedPointParameters},
    UniformRandom,
};
use num_traits::{One, Zero};
use rand_core::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Represents a type that can be additively shared.
pub trait Share:
    Sized
    + Send
    + Sync
    + Clone
    + Copy
    + std::fmt::Debug
    + Eq
    + Zero
    + One
    + Serialize
    + for<'de> Deserialize<'de>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<<Self as Share>::Constant, Output = Self>
    + Add<<Self as Share>::Constant, Output = Self>
    + Neg<Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<<Self as Share>::Constant>
    + AddAssign<<Self as Share>::Constant>
{
    /// The underlying ring that the shares are created over.
    type Ring: for<'a> Add<&'a Self::Ring, Output = Self::Ring>
        + for<'a> Sub<&'a Self::Ring, Output = Self::Ring>
        + std::hash::Hash
        + Serialize
        + for<'de> Deserialize<'de>
        + Copy
        + Zero
        + Neg<Output = Self::Ring>
        + UniformRandom;

    /// The underlying ring that the shares are created over.
    type Constant: Into<Self> + Copy + Zero + One + Neg<Output = Self::Constant>;

    /// Create shares for `self`.
    fn share<R: RngCore + CryptoRng>(
        &self,
        rng: &mut R,
    ) -> (AdditiveShare<Self>, AdditiveShare<Self>) {
        let r = Self::Ring::uniform(rng);
        self.share_with_randomness(&r)
    }

    /// Create shares for `self` using randomness `r`.
    fn share_with_randomness(&self, r: &Self::Ring) -> (AdditiveShare<Self>, AdditiveShare<Self>);

    /// Randomize a share `s` with randomness `r`.
    fn randomize_local_share(s: &AdditiveShare<Self>, r: &Self::Ring) -> AdditiveShare<Self>;
}

// TODO: Figure out the best strategy to handle adding constants

/// Represents a type that can be additively shared with authentication.
pub trait AuthShare: Share {
    /// Create authenticated shares for `self`.
    fn auth_share<R: RngCore + CryptoRng>(
        &self,
        mac_key: &<Self as Share>::Ring,
        rng: &mut R,
    ) -> (AuthAdditiveShare<Self>, AuthAdditiveShare<Self>) {
        let r1 = <Self as Share>::Ring::uniform(rng);
        let r2 = <Self as Share>::Ring::uniform(rng);
        self.auth_share_with_randomness(mac_key, &r1, &r2)
    }

    /// Create authenticated shares for `self` using randomness `r1` and `r2`.
    fn auth_share_with_randomness(
        &self,
        mac_key: &<Self as Share>::Ring,
        r1: &<Self as Share>::Ring,
        r2: &<Self as Share>::Ring,
    ) -> (AuthAdditiveShare<Self>, AuthAdditiveShare<Self>);

    /// Randomize authenticated share `s` using randomness `r1` and `r2`.
    fn randomize_local_auth_share(
        s: &AuthAdditiveShare<Self>,
        r1: &<Self as Share>::Ring,
        r2: &<Self as Share>::Ring,
    ) -> AuthAdditiveShare<Self>;

    /// Open share `s`, checking that the MAC is correct
    fn open(
        s: AuthAdditiveShare<Self>,
        mac_key: &<Self as Share>::Ring,
    ) -> Result<AdditiveShare<Self>, AuthError>;
}

/// Errors that may occur when working with AuthShares
#[derive(Debug)]
pub enum AuthError {
    /// Attempted to open share with invalid MAC
    InvalidMAC,
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AuthError::InvalidMAC => "Attempted to open share with an invalid MAC".fmt(f),
        }
    }
}

#[derive(Default, Hash, Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(bound = "T: Share")]
#[must_use]
/// Represents an additive share of `T`.
pub struct AdditiveShare<T: Share> {
    /// The secret share.
    pub inner: T,
}

impl<T: Share> AdditiveShare<T> {
    /// Construct a new share from `inner`.
    #[inline]
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Combine two additive shares to obtain the shared value.
    pub fn combine(&self, other: &Self) -> T {
        self.inner + other.inner
    }

    /// Add a constant to the share.
    #[inline]
    pub fn add_constant(mut self, other: T::Constant) -> Self {
        self.inner += other;
        self
    }

    /// Add a constant to the share in place..
    #[inline]
    pub fn add_constant_in_place(&mut self, other: T::Constant) {
        self.inner += other;
    }
}

impl<T: Share + Zero> Zero for AdditiveShare<T> {
    fn zero() -> Self {
        Self::new(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }
}

impl<P: FixedPointParameters> AdditiveShare<FixedPoint<P>> {
    /// Double the share.
    #[inline]
    pub fn double(&self) -> Self {
        let mut result = *self;
        result.inner.double_in_place();
        result
    }

    /// Double the share in place.
    #[inline]
    pub fn double_in_place(&mut self) -> &mut Self {
        self.inner.double_in_place();
        self
    }
}

/// Iterate over `self.inner` as `u64`s
pub struct ShareIterator<T: Share + IntoIterator<Item = u64>> {
    inner: <T as IntoIterator>::IntoIter,
}

impl<T: Share + IntoIterator<Item = u64>> Iterator for ShareIterator<T> {
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T: Share + IntoIterator<Item = u64>> ExactSizeIterator for ShareIterator<T> {}

impl<T: Share + IntoIterator<Item = u64>> IntoIterator for AdditiveShare<T> {
    type Item = u64;
    type IntoIter = ShareIterator<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            inner: self.inner.into_iter(),
        }
    }
}

impl<T: Share + std::iter::FromIterator<u64>> std::iter::FromIterator<u64> for AdditiveShare<T> {
    /// Creates a FixedPoint from an iterator over limbs in little-endian order
    #[inline]
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        Self::new(T::from_iter(iter))
    }
}

impl<T: Share> Add<Self> for AdditiveShare<T> {
    type Output = Self;

    #[inline]
    fn add(mut self, other: Self) -> Self {
        self.inner = self.inner + other.inner;
        self
    }
}

impl<T: Share> AddAssign<Self> for AdditiveShare<T> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.inner += other.inner;
    }
}

impl<T: Share> Sub for AdditiveShare<T> {
    type Output = Self;

    #[inline]
    fn sub(mut self, other: Self) -> Self {
        self.inner -= other.inner;
        self
    }
}

impl<T: Share> SubAssign for AdditiveShare<T> {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.inner -= other.inner;
    }
}

impl<T: Share> Neg for AdditiveShare<T> {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self {
        self.inner = -self.inner;
        self
    }
}

impl<T: Share> Mul<T::Constant> for AdditiveShare<T> {
    type Output = Self;
    #[inline]
    fn mul(mut self, other: T::Constant) -> Self {
        self *= other;
        self
    }
}

impl<T: Share> MulAssign<T::Constant> for AdditiveShare<T> {
    #[inline]
    fn mul_assign(&mut self, other: T::Constant) {
        self.inner *= other;
    }
}

impl<T: Share> From<T> for AdditiveShare<T> {
    #[inline]
    fn from(other: T) -> Self {
        Self { inner: other }
    }
}

impl<P: FixedPointParameters> From<AdditiveShare<FixedPoint<P>>> for FixedPoint<P> {
    #[inline]
    fn from(other: AdditiveShare<FixedPoint<P>>) -> Self {
        other.inner
    }
}

/// Operations on shares mimic those of `FixedPoint<P>` itself.
/// This means that
/// * Multiplication by a constant does not automatically truncate the result;
/// * Addition, subtraction, and addition by a constant automatically
/// promote the result to have the correct number of multiplications (max(in1,
/// in2));
/// * `signed_reduce` behaves the same on `FixedPoint<P>` and
///   `AdditiveShare<FixedPoint<P>>`.
impl<P: FixedPointParameters> Share for FixedPoint<P> {
    type Ring = P::Field;
    type Constant = Self;

    #[inline]
    fn share_with_randomness(&self, r: &Self::Ring) -> (AdditiveShare<Self>, AdditiveShare<Self>) {
        let mut cur = *self;
        cur.inner += r;
        (AdditiveShare::new(cur), AdditiveShare::new(Self::new(-*r)))
    }

    #[inline]
    fn randomize_local_share(cur: &AdditiveShare<Self>, r: &Self::Ring) -> AdditiveShare<Self> {
        let mut cur = *cur;
        cur.inner.inner += r;
        cur
    }
}

impl<P: Fp64Parameters> Share for Fp64<P> {
    type Ring = Self;
    type Constant = Self;

    #[inline]
    fn share_with_randomness(&self, r: &Self::Ring) -> (AdditiveShare<Self>, AdditiveShare<Self>) {
        let mut cur = *self;
        cur += r;
        (AdditiveShare::new(cur), AdditiveShare::new(-*r))
    }

    #[inline]
    fn randomize_local_share(cur: &AdditiveShare<Self>, r: &Self::Ring) -> AdditiveShare<Self> {
        let mut cur = *cur;
        cur.inner += r;
        cur
    }
}

#[derive(Default, Hash, Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(bound = "T: AuthShare")]
/// Represents an authenticated additive share of `T`.
pub struct AuthAdditiveShare<T: AuthShare> {
    epsilon: T,
    value: T,
    mac: T,
}

impl<T: AuthShare> AuthAdditiveShare<T> {
    #[inline]
    /// Construct a new authenticated share from `value` and `mac`
    pub fn new(value: T, mac: T) -> Self {
        Self {
            epsilon: T::zero(),
            value,
            mac,
        }
    }

    /// Combine two authenticated shares to obtain the shared value.
    #[inline]
    pub fn combine(&self, other: &Self, mac_key: &<T as Share>::Ring) -> Result<T, AuthError> {
        let combined = AuthShare::open(self + other, mac_key)?;
        Ok(combined.inner)
    }

    /// Add a constant to the authenticated share.
    #[inline]
    pub fn add_constant(mut self, other: impl Into<T>) -> Self {
        let other = other.into();
        self.value = self.value + other;
        self.epsilon = self.epsilon + other;
        self
    }

    /// Add a constant to the share in place.
    #[inline]
    pub fn add_constant_in_place(&mut self, other: impl Into<T>) {
        let other = other.into();
        self.value += other;
        self.epsilon += other;
    }

    /// Subtract a constant from the authenticated share.
    #[inline]
    pub fn sub_constant(mut self, other: impl Into<T>) -> Self {
        let other = other.into();
        self.epsilon = self.epsilon - other;
        self.value = self.value - other;
        self
    }

    /// Subtract a constant to the share in place.
    #[inline]
    pub fn sub_constant_in_place(&mut self, other: impl Into<T>) {
        let other = other.into();
        self.epsilon -= other;
        self.value -= other;
    }

    /// Get inner value without a MAC check
    #[inline]
    pub fn get_value(self) -> AdditiveShare<T> {
        AdditiveShare::new(self.value)
    }

    /// Get inner MAC value
    #[inline]
    pub fn get_mac(self) -> AdditiveShare<T> {
        AdditiveShare::new(self.mac)
    }
}

impl<T: AuthShare> Zero for AuthAdditiveShare<T> {
    fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        self.epsilon.is_zero() && self.value.is_zero() && self.mac.is_zero()
    }
}

impl<T: AuthShare> Neg for AuthAdditiveShare<T> {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self {
        self.epsilon = -self.epsilon;
        self.value = -self.value;
        self.mac = -self.mac;
        self
    }
}

impl<T: AuthShare> Neg for &AuthAdditiveShare<T> {
    type Output = AuthAdditiveShare<T>;

    #[inline]
    fn neg(self) -> AuthAdditiveShare<T> {
        let mut cur = *self;
        cur.epsilon = -self.epsilon;
        cur.value = -self.value;
        cur.mac = -self.mac;
        cur
    }
}

impl<T: AuthShare> Add<Self> for &AuthAdditiveShare<T> {
    type Output = AuthAdditiveShare<T>;

    #[inline]
    fn add(self, other: Self) -> AuthAdditiveShare<T> {
        let mut cur = *self;
        cur.epsilon += other.epsilon;
        cur.value += other.value;
        cur.mac += other.mac;
        cur
    }
}

impl<T: AuthShare> Add<Self> for AuthAdditiveShare<T> {
    type Output = Self;

    #[inline]
    fn add(mut self, other: Self) -> Self {
        self.epsilon += other.epsilon;
        self.value += other.value;
        self.mac += other.mac;
        self
    }
}

impl<T: AuthShare> AddAssign<Self> for AuthAdditiveShare<T> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.epsilon += other.epsilon;
        self.value += other.value;
        self.mac += other.mac;
    }
}

impl<T: AuthShare> Add<&AdditiveShare<T>> for &AuthAdditiveShare<T> {
    type Output = AuthAdditiveShare<T>;

    #[inline]
    fn add(self, other: &AdditiveShare<T>) -> AuthAdditiveShare<T> {
        let mut cur = *self;
        cur.epsilon += other.inner;
        cur.value += other.inner;
        cur
    }
}

impl<T: AuthShare> Add<AdditiveShare<T>> for AuthAdditiveShare<T> {
    type Output = Self;

    #[inline]
    fn add(mut self, other: AdditiveShare<T>) -> Self {
        self.epsilon += other.inner;
        self.value += other.inner;
        self
    }
}

impl<T: AuthShare> AddAssign<AdditiveShare<T>> for AuthAdditiveShare<T> {
    #[inline]
    fn add_assign(&mut self, other: AdditiveShare<T>) {
        self.epsilon += other.inner;
        self.value += other.inner;
    }
}

impl<T: AuthShare> Sub for AuthAdditiveShare<T> {
    type Output = Self;

    #[inline]
    fn sub(mut self, other: Self) -> Self {
        self.epsilon -= other.epsilon;
        self.value -= other.value;
        self.mac -= other.mac;
        self
    }
}

impl<T: AuthShare> SubAssign for AuthAdditiveShare<T> {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.epsilon -= other.epsilon;
        self.value -= other.value;
        self.mac -= other.mac;
    }
}

impl<T: AuthShare> Sub<Self> for &AuthAdditiveShare<T> {
    type Output = AuthAdditiveShare<T>;

    #[inline]
    fn sub(self, other: Self) -> AuthAdditiveShare<T> {
        let mut cur = *self;
        cur.epsilon -= other.epsilon;
        cur.value -= other.value;
        cur.mac -= other.mac;
        cur
    }
}

impl<T: AuthShare, C: Into<<T as Share>::Constant>> Mul<C> for AuthAdditiveShare<T> {
    type Output = Self;

    #[inline]
    fn mul(mut self, other: C) -> Self {
        let other = other.into();
        self.epsilon *= other;
        self.value *= other;
        self.mac *= other;
        self
    }
}

impl<T: AuthShare, C: Into<<T as Share>::Constant>> MulAssign<C> for AuthAdditiveShare<T> {
    #[inline]
    fn mul_assign(&mut self, other: C) {
        let other = other.into();
        self.epsilon *= other;
        self.value *= other;
        self.mac *= other;
    }
}

impl<P: Fp64Parameters> AuthShare for Fp64<P> {
    fn auth_share_with_randomness(
        &self,
        mac_key: &<Self as Share>::Ring,
        r1: &<Self as Share>::Ring,
        r2: &<Self as Share>::Ring,
    ) -> (AuthAdditiveShare<Self>, AuthAdditiveShare<Self>) {
        let mac = (*self) * (*mac_key);
        let (value1, value2) = self.share_with_randomness(r1);
        let (mac1, mac2) = mac.share_with_randomness(r2);

        let share_1 = AuthAdditiveShare::new(value1.inner, mac1.inner);
        let share_2 = AuthAdditiveShare::new(value2.inner, mac2.inner);
        (share_1, share_2)
    }

    fn randomize_local_auth_share(
        s: &AuthAdditiveShare<Self>,
        r1: &<Self as Share>::Ring,
        r2: &<Self as Share>::Ring,
    ) -> AuthAdditiveShare<Self> {
        let mut cur = *s;
        cur.value += r1;
        cur.mac += r2;
        cur
    }

    fn open(
        s: AuthAdditiveShare<Self>,
        mac_key: &<Self as Share>::Ring,
    ) -> Result<AdditiveShare<Self>, AuthError> {
        if ((s.value - s.epsilon) * mac_key) == s.mac {
            Ok(AdditiveShare::new(s.value))
        } else {
            Err(AuthError::InvalidMAC)
        }
    }
}

impl<P: FixedPointParameters> From<AuthAdditiveShare<P::Field>> for AdditiveShare<FixedPoint<P>>
where
    P::Field: AuthShare,
{
    #[inline]
    fn from(other: AuthAdditiveShare<P::Field>) -> Self {
        FixedPoint::new(other.get_value().inner).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebra::fields::near_mersenne_64::F;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaChaRng;

    struct TenBitExpParams {}
    impl FixedPointParameters for TenBitExpParams {
        type Field = F;
        const MANTISSA_CAPACITY: u8 = 5;
        const EXPONENT_CAPACITY: u8 = 5;
    }

    type TenBitExpFP = FixedPoint<TenBitExpParams>;
    // type FPShare = AdditiveShare<TenBitExpFP>;

    const RANDOMNESS: [u8; 32] = [
        0x99, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4,
        0x76, 0x5d, 0xc9, 0x8d, 0x62, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a,
        0x52, 0xd2,
    ];

    fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
        let is_neg: bool = rng.gen();
        let mul = if is_neg { -10.0 } else { 10.0 };
        let float: f64 = rng.gen();
        let f = TenBitExpFP::truncate_float(float * mul);
        let n = TenBitExpFP::from(f);
        (f, n)
    }

    #[test]
    fn test_share_combine() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (_, n) = generate_random_number(&mut rng);
            let (s1, s2) = n.share(&mut rng);
            assert_eq!(s1.combine(&s2), n);
        }
    }

    #[test]
    fn test_double() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (_, n) = generate_random_number(&mut rng);
            let (mut s1, mut s2) = n.share(&mut rng);
            s1.double_in_place();
            s2.double_in_place();
            assert_eq!(s1.combine(&s2), n.double());
        }
    }

    #[test]
    fn test_neg() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (_, n) = generate_random_number(&mut rng);
            let (mut s1, mut s2) = n.share(&mut rng);
            s1 = -s1;
            s2 = -s2;
            assert_eq!(s1.combine(&s2), -n);
        }
    }

    #[test]
    fn test_mul_by_const() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (_, n1) = generate_random_number(&mut rng);
            let (_, n2) = generate_random_number(&mut rng);
            let (mut s1, mut s2) = n1.share(&mut rng);
            s1 = s1 * n2;
            s2 = s2 * n2;
            assert_eq!(s1.combine(&s2), n1 * n2);
        }
    }

    #[test]
    fn test_mul_by_const_with_trunc() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (_, n1) = generate_random_number(&mut rng);
            let (_, n2) = generate_random_number(&mut rng);
            let (mut s1, mut s2) = n1.share(&mut rng);
            s1 = s1 * n2;
            s2 = s2 * n2;
            s1.inner.signed_reduce_in_place();
            s2.inner.signed_reduce_in_place();
            assert_eq!(s1.combine(&s2), n1 * n2);
        }
    }

    #[test]
    fn test_add() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (f1, n1) = generate_random_number(&mut rng);
            let (f2, n2) = generate_random_number(&mut rng);
            let f3 = f1 + f2;
            let n3 = TenBitExpFP::from(f3);
            let (s11, s12) = n1.share(&mut rng);
            let (s21, s22) = n2.share(&mut rng);
            let s31 = s11 + s21;
            let s32 = s12 + s22;
            assert_eq!(
                s31.combine(&s32),
                n3,
                "test failed with f1 = {:?}, f2 = {:?}, f3 = {:?}",
                f1,
                f2,
                f3
            );
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (f1, n1) = generate_random_number(&mut rng);
            let (f2, n2) = generate_random_number(&mut rng);
            let f3 = f1 - f2;
            let n3 = TenBitExpFP::from(f3);
            let (s11, s12) = n1.share(&mut rng);
            let (s21, s22) = n2.share(&mut rng);
            let s31 = s11 - s21;
            let s32 = s12 - s22;
            assert_eq!(
                s31.combine(&s32),
                n3,
                "test failed with f1 = {:?}, f2 = {:?}, f3 = {:?}",
                f1,
                f2,
                f3
            );
        }
    }

    #[test]
    fn test_auth_neg() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..10 {
            let elem = F::uniform(&mut rng);
            let mac_key = F::uniform(&mut rng);
            let (mut s1, mut s2) = elem.auth_share(&mac_key, &mut rng);
            s1 = -s1;
            s2 = -s2;
            assert_eq!(s1.combine(&s2, &mac_key).unwrap(), -elem);
        }
    }

    #[test]
    fn test_auth_share_combine() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let elem = F::uniform(&mut rng);
            let mac_key = F::uniform(&mut rng);
            let (s1, s2) = elem.auth_share(&mac_key, &mut rng);
            assert_eq!(s1.combine(&s2, &mac_key).unwrap(), elem);
        }
    }

    #[test]
    fn test_auth_invalid_mac() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let elem = F::uniform(&mut rng);
            let mut mac_key = F::uniform(&mut rng);
            // Make sure the mac_key isn't 0
            if mac_key.is_zero() {
                mac_key = F::new(1.into());
            };
            // Shift either the value or the mac by some random amount
            let shift = F::uniform(&mut rng);
            let (mut s1, s2) = elem.auth_share(&mac_key, &mut rng);
            match rng.gen_range(0, 2) {
                d if d >= 1 => s1.value += shift,
                d if d < 1 => s1.mac += shift,
                _ => unreachable!(),
            };
            assert!(s1.combine(&s2, &mac_key).is_err())
        }
    }

    #[test]
    fn test_auth_add_by_const() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let elem = F::uniform(&mut rng);
            let shift = F::uniform(&mut rng);
            let mac_key = F::uniform(&mut rng);
            let (mut s1, s2) = elem.auth_share(&mac_key, &mut rng);
            s1.add_constant_in_place(shift);
            assert_eq!(s1.combine(&s2, &mac_key).unwrap(), elem + shift);
        }
    }

    #[test]
    fn test_auth_mul_by_const() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let elem = F::uniform(&mut rng);
            let multiplier = F::uniform(&mut rng);
            let mac_key = F::uniform(&mut rng);
            let (mut s1, mut s2) = elem.auth_share(&mac_key, &mut rng);
            s1 = s1 * multiplier;
            s2 = s2 * multiplier;
            assert_eq!(s1.combine(&s2, &mac_key).unwrap(), elem * multiplier);
        }
    }

    #[test]
    fn test_auth_add_const_then_neg() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let elem = F::uniform(&mut rng);
            let shift1 = F::uniform(&mut rng);
            let shift2 = F::uniform(&mut rng);
            let mac_key = F::uniform(&mut rng);
            let (mut s1, mut s2) = elem.auth_share(&mac_key, &mut rng);
            s1.add_constant_in_place(shift1);
            s2.add_constant_in_place(shift2);
            s1 = -s1;
            s2 = -s2;
            assert_eq!(
                s1.combine(&s2, &mac_key).unwrap(),
                -(elem + shift1 + shift2)
            );
        }
    }

    #[test]
    fn test_auth_add_const_then_add() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let e1 = F::uniform(&mut rng);
            let e2 = F::uniform(&mut rng);
            let e3 = e1 + e2;
            let shift1 = F::uniform(&mut rng);
            let shift2 = F::uniform(&mut rng);
            let mac_key = F::uniform(&mut rng);
            let (mut s11, s12) = e1.auth_share(&mac_key, &mut rng);
            let (s21, mut s22) = e2.auth_share(&mac_key, &mut rng);
            s11.add_constant_in_place(shift1);
            s22.add_constant_in_place(shift2);
            let s31 = s11 + s21;
            let s32 = s12 + s22;
            assert_eq!(s31.combine(&s32, &mac_key).unwrap(), e3 + shift1 + shift2);
        }
    }
}
