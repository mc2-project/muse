#![allow(non_snake_case)]

use algebra::{BitIterator, FixedPointParameters, Fp64Parameters, FpParameters, PrimeField};
pub use fancy_garbling;

use fancy_garbling::{
    circuit::CircuitBuilder,
    error::{CircuitBuilderError, FancyError},
    util, BinaryBundle, BinaryGadgets, Bundle, BundleGadgets, Fancy,
};

#[inline(always)]
fn mux_single_bit<F: Fancy>(
    f: &mut F,
    b: &F::Item,
    x: &F::Item,
    y: &F::Item,
) -> Result<F::Item, F::Error> {
    let y_plus_x = f.add(x, y)?;
    let res = f.mul(b, &y_plus_x)?;
    f.add(&x, &res)
}

/// If `b = 0` returns `x` else `y`.
///
/// `b` must be mod 2 but `x` and `y` can be have any modulus.
fn mux<F: Fancy>(
    f: &mut F,
    b: &F::Item,
    x: &BinaryBundle<F::Item>,
    y: &BinaryBundle<F::Item>,
) -> Result<Vec<F::Item>, F::Error> {
    x.wires()
        .iter()
        .zip(y.wires())
        .map(|(x, y)| mux_single_bit(f, b, x, y))
        .collect()
}

#[inline]
fn bin_xnor<F: Fancy>(
    b: &mut F,
    xs: &BinaryBundle<F::Item>,
    ys: &BinaryBundle<F::Item>,
) -> Result<BinaryBundle<F::Item>, F::Error> {
    let xor = b.bin_xor(xs, ys)?;
    bin_negate(b, &xor)
}

#[inline]
fn bin_negate<F: Fancy>(
    b: &mut F,
    xs: &BinaryBundle<F::Item>,
) -> Result<BinaryBundle<F::Item>, F::Error> {
    xs.iter()
        .map(|e| b.negate(e))
        .collect::<Result<Vec<_>, _>>()
        .map(BinaryBundle::new)
}

#[inline]
fn mod_p_helper<F: Fancy>(
    b: &mut F,
    neg_p: &BinaryBundle<F::Item>,
    bits: &BinaryBundle<F::Item>,
) -> Result<BinaryBundle<F::Item>, F::Error> {
    let (result, borrow) = b.bin_addition(&bits, &neg_p)?;
    // If p underflowed, then we want the result, otherwise we're fine with the
    // original.
    mux(b, &borrow, &bits, &result).map(BinaryBundle::new)
}

fn neg_p_over_2_helper<F: Fancy>(
    f: &mut F,
    neg_p_over_2: u128,
    neg_p_over_2_bits: &BinaryBundle<F::Item>,
    bits: &BinaryBundle<F::Item>,
) -> Result<BinaryBundle<F::Item>, F::Error> {
    let xwires = bits.wires();
    let ywires = neg_p_over_2_bits.wires();
    let mut neg_p_over_2 = BitIterator::new([neg_p_over_2 as u64]).collect::<Vec<_>>();
    neg_p_over_2.reverse();
    let mut neg_p_over_2 = neg_p_over_2.into_iter();
    let mut seen_one = neg_p_over_2.next().unwrap();

    let (mut z, mut c) = adder_const(f, &xwires[0], &ywires[0], seen_one, None)?;

    let mut bs = vec![z];
    for ((x, y), b) in xwires[1..(xwires.len() - 1)]
        .iter()
        .zip(&ywires[1..])
        .zip(neg_p_over_2)
    {
        seen_one |= b;
        let res = adder_const(f, x, y, seen_one, c.as_ref())?;
        z = res.0;
        c = res.1;
        bs.push(z);
    }

    z = f.add_many(&[
        xwires.last().unwrap().clone(),
        ywires.last().unwrap().clone(),
        c.unwrap(),
    ])?;
    bs.push(z);
    Ok(BinaryBundle::new(bs))
}

/// Compute the number of bits needed to represent `p`, plus one.
#[inline]
pub fn num_bits(p: u128) -> usize {
    (p.next_power_of_two() * 2).trailing_zeros() as usize
}

/// Binary adder. Returns the result and the carry.
fn adder_const<F: Fancy>(
    f: &mut F,
    x: &F::Item,
    y: &F::Item,
    b: bool,
    carry_in: Option<&F::Item>,
) -> Result<(F::Item, Option<F::Item>), F::Error> {
    if let Some(c) = carry_in {
        let z1 = f.xor(x, y)?;
        let z2 = f.xor(&z1, c)?;
        let z3 = f.xor(x, c)?;
        let z4 = f.and(&z1, &z3)?;
        let carry = f.xor(&z4, x)?;
        Ok((z2, Some(carry)))
    } else {
        let z = f.xor(x, y)?;
        let carry = if !b { None } else { Some(f.and(x, y)?) };
        Ok((z, carry))
    }
}

/// Add two numbers with possible unequal size starting from
/// `offset` and returning the lower `out_bits` bits.
/// TODO
/// Note that this function assumes that the bits of `large`
/// greater than `small` are all zeroed out.
fn bin_add_unequal<F: Fancy>(
    b: &mut F,
    large: &BinaryBundle<F::Item>,
    small: &BinaryBundle<F::Item>,
    offset: usize,
    out_bits: usize,
    zeroed: bool,
) -> Result<BinaryBundle<F::Item>, F::Error> {
    if !(large.size() >= small.size()) {
        return Err(F::Error::from(FancyError::UnequalModuli));
    }
    if out_bits <= offset {
        return Err(F::Error::from(FancyError::InvalidArg(
            "bin_add_unequal: out_bits must be greater than offset".to_string(),
        )));
    }
    let mut lwires = large.wires().to_vec();
    let swires = small.wires();

    // Pad `large` to `out_bits` bits
    let zero = b.constant(0, 2)?;
    if lwires.len() < out_bits {
        for _ in 0..(out_bits - lwires.len()) {
            lwires.push(zero.clone());
        }
    }

    // Initial sum
    let (mut z, mut c) = b.adder(&lwires[offset], &swires[offset], None)?;
    lwires[offset] = z;

    // Addition is done in the standard way, except if `out_bits` is
    // greater than the length of `small`, the very last
    // carry bit is simply copied into the wire since we assume
    // that the wire is zero
    for i in (offset + 1)..out_bits {
        if i < swires.len() {
            let res = b.adder(&lwires[i], &swires[i], Some(&c))?;
            z = res.0;
            c = res.1;
            lwires[i] = z;
        } else if i == swires.len() && zeroed {
            lwires[i] = c.clone();
        } else if !zeroed {
            let res = b.adder(&lwires[i], &c, None)?;
            z = res.0;
            c = res.1;
            lwires[i] = z;
        }
    }
    Ok(BinaryBundle::new(lwires.to_vec()))
}

/// Reduces `x` modulo `p` using a technique motivated by
/// this paper by Will and Ko:
///     https://eprint.iacr.org/2014/755.pdf
///
/// Instead of using a table-lookup + addition, this method
/// opts to reduce after every shift as table-lookup is expensive
///
/// This supports `x` being the product of a single multiplication
/// with width 2*(n-1) but could be easily extended to higher bit widths
fn mul_reduce<F: Fancy>(
    b: &mut F,
    neg_p: &BinaryBundle<F::Item>,
    zero: &F::Item,
    x: &BinaryBundle<F::Item>,
) -> Result<BinaryBundle<F::Item>, F::Error> {
    let n = neg_p.size();
    // Split x_wires into two chunks of `n` bits
    //
    // The modulus is `n-1` bits so we add an extra 0
    // on the end for the carry
    let mut G = x
        .wires()
        .chunks(n - 1)
        .map(|s| {
            let mut vec = s.to_vec();
            (vec.len()..n).for_each(|_| vec.push(zero.clone()));
            BinaryBundle::new(vec)
        })
        .collect::<Vec<_>>();

    // For each bit of the modulus size, shift by one and reduce
    let mut sum = G.remove(1).extract();
    for _ in (0..n - 1).rev() {
        sum = b.shift(&sum, 1)?;
        sum = mod_p_helper(b, neg_p, &sum.into())?.extract();
    }
    G[0] = b.bin_addition_no_carry(&G[0], &sum.into())?;
    let out = mod_p_helper(b, neg_p, &G[0])?;
    Ok(out)
}

///// Reduces `x` modulo `p` using Barret's reduction technique.
///// See explanation:
/////   https://www.nayuki.io/page/barrett-reduction-algorithm
/////
///// `x` must be the result of a single multiplication ie. be
///// less than `p**2`
// fn barret_reduce<F:Fancy>(
//    b: &mut F,
//    num_bits: usize,
//    p: &BinaryBundle<F::Item>,
//    neg_p: &BinaryBundle<F::Item>,
//    r: &BinaryBundle<F::Item>,
//    x: &BinaryBundle<F::Item>,
//) -> Result<BinaryBundle<F::Item>, F::Error> {
//    // k = num_bits, m = mod_size
//    let xr = bin_mul(b, &x, &r, num_bits*3)?;
//    // Compute `2k` right shift by removing
//    // lower order wires
//    let mut xr_wires = xr.wires().to_vec();
//    xr_wires.drain(0..2*num_bits);
//    let xr_shift = BinaryBundle::new(xr_wires);
//
//    let xr_shift_p = bin_mul(b, p, &xr_shift, 2*num_bits)?;
//    // This subtraction is guaranteed not to underflow
//    let (t, _) = b.bin_subtraction(&x, &xr_shift_p)?;
//
//    // Reduce `t` to `k` bits
//    let mut t_wires = t.wires().to_vec();
//    t_wires.drain(num_bits..2*num_bits);
//
//    // Do a single final reduction
//    mod_p_helper(b, neg_p, &BinaryBundle::new(t_wires))
//}

/// Binary multiplier. Returns `out_bits` lower order bits
///
/// `xs` and `ys` do not need to have the same bit length.
fn bin_mul<F: Fancy>(
    b: &mut F,
    xs: &BinaryBundle<F::Item>,
    ys: &BinaryBundle<F::Item>,
    out_bits: usize,
) -> Result<BinaryBundle<F::Item>, F::Error> {
    let xwires = xs.wires();
    let ywires = ys.wires();
    let x_bits = xwires.len();
    let y_bits = ywires.len();

    // Algorithm is same as elementary school multiplication
    // Compute n^2 intermediate shifted products and sum all
    //
    // Compute initial sum and pad to `out_bits` wires
    let mut init = xwires
        .iter()
        .map(|x| b.and(x, &ywires[0]))
        .collect::<Result<Vec<_>, _>>()
        .map(Bundle::new)?;
    let zero = b.constant(0, 2)?;
    if out_bits > x_bits {
        init.pad(&zero, out_bits - x_bits);
    }
    let mut sum = Into::<BinaryBundle<F::Item>>::into(init);

    // We only do the exact number of multiplications necessary
    // to correctly compute `out_bits` bits
    for i in 1..std::cmp::min(y_bits, out_bits - x_bits + 2) {
        let mut mul = xwires
            .iter()
            .map(|x| b.and(x, &ywires[i]))
            .collect::<Result<Vec<F::Item>, F::Error>>()
            .map(Bundle::new)?;

        // Pad the multiplication so that shifting retains
        // `out_bits` of information
        if out_bits > x_bits {
            mul.pad(&zero, std::cmp::min(out_bits - x_bits, i));
        }
        let shifted = b.shift(&mul, i).map(BinaryBundle::from)?;
        sum = bin_add_unequal(b, &sum, &shifted, i, out_bits, true)?;
    }
    Ok(sum)
}

/// Binary modular multiplier modulo p.
///
/// `xs` and `ys` do not need to have the same bit length.
///
/// This function assumes that both `xs` and `ys` have a higher 0-bit
/// ie. were not multiplied after an un-reduced addition
#[inline]
fn bin_mod_mul<F: Fancy>(
    b: &mut F,
    neg_p: &BinaryBundle<F::Item>,
    zero: &F::Item,
    xs: &BinaryBundle<F::Item>,
    ys: &BinaryBundle<F::Item>,
) -> Result<BinaryBundle<F::Item>, F::Error> {
    let n = neg_p.size();
    // Since `xs` and `ys` have an extra 0 bit we only need
    // top 2*(n-1) bits of multiplication
    let x = bin_mul(b, xs, ys, 2 * (n - 1))?;
    mul_reduce(b, neg_p, zero, &x)
}

/// Compute the `ReLU` of `n` over the field `P::Field`.
pub fn relu<P: FixedPointParameters>(
    b: &mut CircuitBuilder,
    n: usize,
) -> Result<(), CircuitBuilderError>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    let p = u128::from(<<P::Field as PrimeField>::Params>::MODULUS.0);
    let exponent_size = P::EXPONENT_CAPACITY as usize;

    let p_over_2 = p / 2;
    // Convert to two's complement
    let neg_p_over_2 = !p_over_2 + 1;
    // Convert to two's complement. Equivalent to `let neg_p = -(p as i128) as u128;
    let neg_p = !p + 1;
    let q = 2;
    let num_bits = num_bits(p);

    let moduli = vec![q; num_bits];
    // Construct constant for addition with neg p
    let neg_p = b.bin_constant_bundle(neg_p, num_bits)?;
    let neg_p_over_2_bits = b
        .constant_bundle(&util::u128_to_bits(neg_p_over_2, num_bits), &moduli)?
        .into();
    let zero = b.constant(0, 2)?;
    let one = b.constant(1, 2)?;
    for _ in 0..n {
        let s1 = BinaryBundle::new(b.evaluator_inputs(&moduli));
        let s1_next = BinaryBundle::new(b.evaluator_inputs(&moduli));
        let s2 = BinaryBundle::new(b.garbler_inputs(&moduli));
        let s2_next = BinaryBundle::new(b.garbler_inputs(&moduli));
        // Add secret shares as integers
        let res = b.bin_addition_no_carry(&s1, &s2)?;
        // Take the result mod p;
        let layer_input = mod_p_helper(b, &neg_p, &res).unwrap();

        // Compare with p/2
        // Since we take > p/2 as negative, if the number is less than p/2, it is
        // positive.
        let res = neg_p_over_2_helper(b, neg_p_over_2, &neg_p_over_2_bits, &layer_input)?;
        // Take the sign bit
        let zs_is_positive = res.wires().last().unwrap();

        // Compute the relu
        let mut relu_res = Vec::with_capacity(num_bits);
        let relu_6_size = exponent_size + 3;
        // We choose 5 arbitrarily here; the idea is that we won't see values of
        // greater than 2^8.
        // We then drop the larger bits
        for wire in layer_input.wires().iter().take(relu_6_size + 5) {
            relu_res.push(b.and(&zs_is_positive, wire)?);
        }
        let is_seven = b.and_many(&relu_res[(exponent_size + 1)..relu_6_size])?;
        let some_higher_bit_is_set = b.or_many(&relu_res[relu_6_size..])?;

        let should_be_six = b.or(&some_higher_bit_is_set, &is_seven)?;

        for wire in &mut relu_res[relu_6_size..] {
            *wire = zero;
        }
        let lsb = &mut relu_res[exponent_size];
        *lsb = mux_single_bit(b, &should_be_six, lsb, &zero)?;

        let middle_bit = &mut relu_res[exponent_size + 1];
        *middle_bit = mux_single_bit(b, &should_be_six, middle_bit, &one)?;

        let msb = &mut relu_res[exponent_size + 2];
        *msb = mux_single_bit(b, &should_be_six, msb, &one)?;

        for wire in &mut relu_res[..exponent_size] {
            *wire = mux_single_bit(b, &should_be_six, wire, &zero)?;
        }

        relu_res.extend(std::iter::repeat(zero).take(num_bits - relu_6_size - 5));

        let relu_res = BinaryBundle::new(relu_res);

        // TODO: Need to reduce here?
        let res = b.bin_addition_no_carry(&relu_res, &s1_next)?;
        let next_share = mod_p_helper(b, &neg_p, &res)?;

        let res = b.bin_addition_no_carry(&next_share, &s2_next)?;
        let next_share = mod_p_helper(b, &neg_p, &res)?;

        b.output_bundle(&next_share)?;
    }
    Ok(())
}

/// Truncate `n` by `trunc_bits` and compute its `ReLU`  over the field `P::Field`.
pub fn truncated_relu<P: FixedPointParameters>(
    b: &mut CircuitBuilder,
    n: usize,
    trunc_bits: u8,
) -> Result<(), CircuitBuilderError>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    let p = u128::from(<<P::Field as PrimeField>::Params>::MODULUS.0);
    let exponent_size = P::EXPONENT_CAPACITY as usize;

    let p_over_2 = p / 2;
    // Convert to two's complement
    let neg_p_over_2 = !p_over_2 + 1;
    // Convert to two's complement. Equivalent to `let neg_p = -(p as i128) as u128;
    let neg_p = !p + 1;
    let q = 2;
    let num_bits = num_bits(p);

    let moduli = vec![q; num_bits];
    // Construct contant for comparison with p
    let p = b.bin_constant_bundle(p, num_bits)?;
    // Construct constant for addition with neg p
    let neg_p = b.bin_constant_bundle(neg_p, num_bits)?;
    let neg_p_over_2_bits = b
        .constant_bundle(&util::u128_to_bits(neg_p_over_2, num_bits), &moduli)?
        .into();
    let zero = b.constant(0, 2)?;
    let one = b.constant(1, 2)?;
    for _ in 0..n {
        let s1 = BinaryBundle::new(b.evaluator_inputs(&moduli));
        let s1_next = BinaryBundle::new(b.evaluator_inputs(&moduli));
        let s2 = BinaryBundle::new(b.garbler_inputs(&moduli));
        let s2_next = BinaryBundle::new(b.garbler_inputs(&moduli));

        // Compare client inputs with p
        let s1_less_than_p = b.bin_lt(&s1, &p)?;
        let s1_next_less_than_p = b.bin_lt(&s1_next, &p)?;
        b.output(&s1_less_than_p)?;
        b.output(&s1_next_less_than_p)?;

        // Add secret shares as integers
        let res = b.bin_addition_no_carry(&s1, &s2)?;
        // Take the result mod p;
        let layer_input = mod_p_helper(b, &neg_p, &res).unwrap();

        // Compare with p/2
        // Since we take > p/2 as negative, if the number is less than p/2, it is
        // positive.
        let res = neg_p_over_2_helper(b, neg_p_over_2, &neg_p_over_2_bits, &layer_input)?;
        // Take the sign bit
        let inp_is_positive = res.wires().last().unwrap();

        // Compute the relu
        let mut relu_res = Vec::with_capacity(num_bits);
        let relu_6_size = exponent_size + 3;
        // First we skip all the bits which will be truncated. Next, we take the number of bits
        // needed for ReLU6 along with some arbitrary upperbound (we choose 2^8) for the
        // intermediate values during network evaluation. We then drop the larger bits.
        for wire in layer_input
            .wires()
            .iter()
            .skip(trunc_bits as usize)
            .take(relu_6_size + 5)
        {
            relu_res.push(b.and(&inp_is_positive, wire)?);
        }
        let is_seven = b.and_many(&relu_res[(exponent_size + 1)..relu_6_size])?;
        let some_higher_bit_is_set = b.or_many(&relu_res[relu_6_size..])?;

        let should_be_six = b.or(&some_higher_bit_is_set, &is_seven)?;

        for wire in &mut relu_res[relu_6_size..] {
            *wire = zero;
        }
        let lsb = &mut relu_res[exponent_size];
        *lsb = mux_single_bit(b, &should_be_six, lsb, &zero)?;

        let middle_bit = &mut relu_res[exponent_size + 1];
        *middle_bit = mux_single_bit(b, &should_be_six, middle_bit, &one)?;

        let msb = &mut relu_res[exponent_size + 2];
        *msb = mux_single_bit(b, &should_be_six, msb, &one)?;

        for wire in &mut relu_res[..exponent_size] {
            *wire = mux_single_bit(b, &should_be_six, wire, &zero)?;
        }

        relu_res.extend(std::iter::repeat(zero).take(num_bits - relu_6_size - 5));

        let relu_res = BinaryBundle::new(relu_res);

        let res = b.bin_addition_no_carry(&relu_res, &s1_next)?;
        let next_share = mod_p_helper(b, &neg_p, &res)?;

        let res = b.bin_addition_no_carry(&next_share, &s2_next)?;
        let next_share = mod_p_helper(b, &neg_p, &res)?;
        b.output_bundle(&next_share)?;
    }
    Ok(())
}

/// Sum vector of additive shares
#[inline]
fn reconstruct_shares_vec<F: Fancy>(
    b: &mut F,
    neg_p: &BinaryBundle<F::Item>,
    s1: &Vec<BinaryBundle<F::Item>>,
    s2: &Vec<BinaryBundle<F::Item>>,
) -> Result<Vec<BinaryBundle<F::Item>>, F::Error> {
    s1.iter()
        .zip(s2.iter())
        .map(|(e1, e2)| {
            let sum = b.bin_addition_no_carry(e1, e2)?;
            mod_p_helper(b, neg_p, &sum)
        })
        .collect()
}

/// Multiply vector by element
#[inline]
fn multiply_shares_vec<F: Fancy>(
    b: &mut F,
    neg_p: &BinaryBundle<F::Item>,
    zero: &F::Item,
    s1: &Vec<BinaryBundle<F::Item>>,
    multiplier: &BinaryBundle<F::Item>,
) -> Result<Vec<BinaryBundle<F::Item>>, F::Error> {
    s1.iter()
        .map(|e| bin_mod_mul(b, neg_p, zero, e, multiplier))
        .collect()
}

/// Check whether two vectors of elements are equal to eachother
#[inline]
fn is_equal_vec<F: Fancy>(
    b: &mut F,
    x: &Vec<BinaryBundle<F::Item>>,
    y: &Vec<BinaryBundle<F::Item>>,
) -> Result<Vec<BinaryBundle<F::Item>>, F::Error> {
    // Check equality by taking the XNOR between each wire
    // and running all results through an AND gate
    x.iter()
        .zip(y.iter())
        .map(|(e1, e2)| bin_xnor(b, e1, e2))
        .collect::<Result<Vec<_>, _>>()
}

/// Garbled circuit implementation of conditional disclosure of secrets
/// Checks correctness of additive MAC shares and outputs GC labels
/// to evaluator
pub fn cds<P: FixedPointParameters>(
    b: &mut CircuitBuilder,
    layer_size: usize,
) -> Result<(), CircuitBuilderError>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    let p = u128::from(<<P::Field as PrimeField>::Params>::MODULUS.0);
    let n = num_bits(p);
    let moduli = vec![2; n];
    // neg_p for reductions
    let neg_p = !p + 1;
    // Convert to circuit constants
    let neg_p = b.bin_constant_bundle(neg_p, n)?;
    let zero = b.constant(0, 2)?;

    // Helper closures for defining input vectors
    let eval_input_vec = |b: &mut CircuitBuilder, len: usize, moduli: &[u16]| -> Vec<_> {
        (0..len)
            .map(|_| BinaryBundle::new(b.evaluator_inputs(&moduli)))
            .collect::<Vec<_>>()
    };

    let garb_input_vec = |b: &mut CircuitBuilder, len: usize, moduli: &[u16]| -> Vec<_> {
        (0..len)
            .map(|_| BinaryBundle::new(b.garbler_inputs(&moduli)))
            .collect::<Vec<_>>()
    };

    // Denote y = M_i * r_i - s_i, r = r_i+1
    // Client inputs ay_1, br_1, y, r
    let ay_1 = eval_input_vec(b, layer_size, &moduli);
    let br_1 = eval_input_vec(b, layer_size, &moduli);
    let y = eval_input_vec(b, layer_size, &moduli);
    let r = eval_input_vec(b, layer_size, &moduli);

    // Server inputs ay_2, br_2, a, b, next_layer_shares
    let ay_2 = garb_input_vec(b, layer_size, &moduli);
    let br_2 = garb_input_vec(b, layer_size, &moduli);
    let alpha = BinaryBundle::new(b.garbler_inputs(&moduli));
    let beta = BinaryBundle::new(b.garbler_inputs(&moduli));

    // Reconstruct ay and br  from shares
    let ay_sum = reconstruct_shares_vec(b, &neg_p, &ay_1, &ay_2)?;
    let br_sum = reconstruct_shares_vec(b, &neg_p, &br_1, &br_2)?;

    // Multiply client's inputted y, r by server's a, b
    let ay = multiply_shares_vec(b, &neg_p, &zero, &y, &alpha)?;
    let br = multiply_shares_vec(b, &neg_p, &zero, &r, &beta)?;

    // Take XNOR of ay and br to check equality
    // If equal, result should be vectors of all ones
    let ay_equal = is_equal_vec(b, &ay_sum, &ay)?;
    let br_equal = is_equal_vec(b, &br_sum, &br)?;

    let bin_to_bundle = |elems: Vec<BinaryBundle<_>>| -> Vec<_> {
        elems.into_iter().map(|e| e.extract()).collect::<Vec<_>>()
    };

    b.output_bundles(bin_to_bundle(y).as_slice())?;
    b.output_bundles(bin_to_bundle(r).as_slice())?;
    b.output_bundles(bin_to_bundle(ay_equal).as_slice())?;
    b.output_bundles(bin_to_bundle(br_equal).as_slice())?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Share;
    use algebra::{fields::near_mersenne_64::F, *};
    use fancy_garbling::circuit::CircuitBuilder;
    use rand::{thread_rng, Rng};

    struct TenBitExpParams {}
    impl FixedPointParameters for TenBitExpParams {
        type Field = F;
        const MANTISSA_CAPACITY: u8 = 3;
        const EXPONENT_CAPACITY: u8 = 8;
    }

    type TenBitExpFP = FixedPoint<TenBitExpParams>;

    fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
        let is_neg: bool = rng.gen();
        let mul = if is_neg { -10.0 } else { 10.0 };
        let float: f64 = rng.gen();
        // TODO: Currently only generates positive numbers to stop overflow
        let f = TenBitExpFP::truncate_float(float * mul);
        let n = TenBitExpFP::from(f);
        (f, n)
    }

    /// Compute the product of some u16s as a u128.
    #[inline]
    pub(crate) fn product(xs: &[u16]) -> u128 {
        xs.iter().fold(1, |acc, &x| acc * x as u128)
    }

    #[test]
    pub(crate) fn test_mod_mul() {
        let mut rng = thread_rng();
        let p = <F as PrimeField>::Params::MODULUS.0 as u128;
        let n = num_bits(p);
        let moduli = vec![2; n];
        let neg_p = !p + 1;

        // Construct multiplication circuit
        let mut b = CircuitBuilder::new();
        // Convert to circuit constants
        let neg_p = b.bin_constant_bundle(neg_p, n).unwrap();
        let zero = b.constant(0, 2).unwrap();

        let e1 = BinaryBundle::new(b.garbler_inputs(&moduli));
        let e2 = BinaryBundle::new(b.evaluator_inputs(&moduli));

        // We only need 2*(n-1) bits since the multiplication is between fresh field
        // elements
        let x = bin_mul(&mut b, &e1, &e2, 2 * (n - 1)).unwrap();
        let out = mul_reduce(&mut b, &neg_p, &zero, &x).unwrap();

        b.output_bundle(&out).unwrap();

        let mut c = b.finish();
        let _ = c.print_info();
        let (en, ev) = fancy_garbling::garble(&mut c).unwrap();

        for i in 0..1000 {
            // Input
            let a = generate_random_number(&mut rng).1;
            let b = generate_random_number(&mut rng).1;
            let garb_input = util::u128_to_bits(a.inner.into_repr().0 as u128, n);
            let eval_input = util::u128_to_bits(b.inner.into_repr().0 as u128, n);

            // Encode inputs and evaluate circuit
            let xs = en.encode_garbler_inputs(&garb_input);
            let ys = en.encode_evaluator_inputs(&eval_input);
            let garbled_eval_results = ev.eval(&mut c, &xs, &ys).unwrap();

            let eval_result = util::u128_from_bits(&garbled_eval_results);
            let pt_result = (a * b).inner.into_repr().0 as u128;
            assert_eq!(
                eval_result, pt_result,
                "\nIter {}: Got {} Expected {}",
                i, eval_result, pt_result,
            );
        }
    }

    fn assert_delta(x: u128, y: u128, d: u128) -> bool {
        if x > y {
            x - y <= d
        } else {
            y - x <= d
        }
    }

    #[test]
    pub(crate) fn test_relu() {
        let mut rng = thread_rng();
        let q = 2;
        let p = <F as PrimeField>::Params::MODULUS.0 as u128;
        let n = num_bits(p);
        let Q = product(&vec![q; n]);
        println!("n={} q={} Q={}", n, q, Q);

        let mut b = CircuitBuilder::new();
        relu::<TenBitExpParams>(&mut b, 1).unwrap();
        let mut c = b.finish();
        let _ = c.print_info();

        let zero = TenBitExpFP::zero();
        let six = TenBitExpFP::from(6.0);
        for i in 0..10000 {
            let (_, n1) = generate_random_number(&mut rng);
            let (s1, s2) = n1.share(&mut rng);
            let res_should_be_fp = if n1 <= zero {
                zero
            } else if n1 > six {
                six
            } else {
                n1
            };
            let res_should_be = res_should_be_fp.inner.into_repr().0 as u128;
            let z1 = F::uniform(&mut rng).into_repr().0 as u128;
            let res_should_be = (res_should_be + z1) % p;

            let s1 = s1.inner.inner.into_repr().0 as u128;
            let mut garbler_inputs = util::u128_to_bits(s1, n);
            garbler_inputs.extend_from_slice(&util::u128_to_bits(z1, n));

            let z2 = F::uniform(&mut rng).into_repr().0 as u128;
            let res_should_be = (res_should_be + z2) % p;

            let s2 = s2.inner.inner.into_repr().0 as u128;
            let mut evaluator_inputs = util::u128_to_bits(s2, n);
            evaluator_inputs.extend_from_slice(&util::u128_to_bits(z2, n));

            let (en, ev) = fancy_garbling::garble(&mut c).unwrap();
            let xs = en.encode_garbler_inputs(&garbler_inputs);
            let ys = en.encode_evaluator_inputs(&evaluator_inputs);
            let garbled_eval_results = ev.eval(&mut c, &xs, &ys).unwrap();

            let evaluated_results = c.eval_plain(&garbler_inputs, &evaluator_inputs).unwrap();
            assert!(
                assert_delta(util::u128_from_bits(&evaluated_results), res_should_be, 1),
                "Iteration {}, Pre-ReLU value is {}, value should be {}, is {}",
                i,
                n1,
                res_should_be,
                util::u128_from_bits(&evaluated_results)
            );
            assert!(
                assert_delta(
                    util::u128_from_bits(&garbled_eval_results),
                    res_should_be,
                    1
                ),
                "Iteration {}, Pre-ReLU value is {}, value should be {}, is {}",
                i,
                n1,
                res_should_be_fp,
                res_should_be_fp
            );
        }
    }

    #[test]
    pub(crate) fn test_truncated_relu() {
        let mut rng = thread_rng();
        let q = 2;
        let p = <F as PrimeField>::Params::MODULUS.0 as u128;
        let n = num_bits(p);
        let num_trunc = 3;
        let trunc_bits = num_trunc * TenBitExpParams::EXPONENT_CAPACITY;
        let Q = product(&vec![q; n]);
        println!("n={} q={} Q={}", n, q, Q);

        let mut b = CircuitBuilder::new();
        truncated_relu::<TenBitExpParams>(&mut b, 1, trunc_bits).unwrap();
        let mut c = b.finish();
        let _ = c.print_info();

        let zero = TenBitExpFP::zero();
        let one = TenBitExpFP::one();
        let six = TenBitExpFP::from(6.0);
        for i in 0..10000 {
            let (_, n1) = generate_random_number(&mut rng);
            let (s1, s2) = n1.share(&mut rng);
            let mut s1 = s1.inner;
            let mut s2 = s2.inner;

            let res_should_be_fp = if n1 <= zero {
                zero
            } else if n1 > six {
                six
            } else {
                n1
            };
            let mut res_should_be = res_should_be_fp.inner.into_repr().0 as u128;

            // Multiply client and server's inputs by one for each truncation
            for _ in 0..num_trunc {
                s1 = one * s1;
                s2 = one * s2;
            }

            // Server's randomizer
            let z1 = F::uniform(&mut rng).into_repr().0 as u128;
            res_should_be = (res_should_be + z1) % p;

            let mut garbler_inputs = util::u128_to_bits(s2.inner.into_repr().0 as u128, n);
            garbler_inputs.extend_from_slice(&util::u128_to_bits(z1, n));

            // Client's randomizer
            let z2 = F::uniform(&mut rng).into_repr().0 as u128;
            res_should_be = (res_should_be + z2) % p;

            let mut evaluator_inputs = util::u128_to_bits(s1.inner.into_repr().0 as u128, n);
            evaluator_inputs.extend_from_slice(&util::u128_to_bits(z2, n));

            let (en, ev) = fancy_garbling::garble(&mut c).unwrap();
            let xs = en.encode_garbler_inputs(&garbler_inputs);
            let ys = en.encode_evaluator_inputs(&evaluator_inputs);
            let garbled_eval_results = ev.eval(&mut c, &xs, &ys).unwrap();
            let evaluated_results = c.eval_plain(&garbler_inputs, &evaluator_inputs).unwrap();

            // Assert that inputs were less than p
            assert!(
                garbled_eval_results[0] == 1 && garbled_eval_results[1] == 1,
                "Evaluator input was greater than p"
            );
            // Assert that plaintext and GC evals are equal
            assert!(
                assert_delta(
                    util::u128_from_bits(&evaluated_results[2..]),
                    res_should_be,
                    1
                ),
                "Iteration {}, Pre-ReLU value is {}, value should be {}, is {}",
                i,
                n1,
                res_should_be,
                util::u128_from_bits(&evaluated_results[2..])
            );
            assert!(
                assert_delta(
                    util::u128_from_bits(&garbled_eval_results[2..]),
                    res_should_be,
                    1
                ),
                "Iteration {}, Pre-ReLU value is {}, value should be {}, is {}",
                i,
                n1,
                res_should_be_fp,
                res_should_be_fp
            );
        }
    }

    /// Dummy circuit which just outputs the evaluator inputs
    /// Used to check the CDS protocol
    pub(crate) fn dummy(
        b: &mut CircuitBuilder,
        layer_size: usize,
    ) -> Result<(), CircuitBuilderError> {
        let p = u128::from(<<F as PrimeField>::Params>::MODULUS.0);
        let num_bits = num_bits(p);
        let moduli = vec![2; num_bits];

        let eval_input_vec = |b: &mut CircuitBuilder| -> Vec<_> {
            (0..layer_size)
                .map(|_| Bundle::new(b.evaluator_inputs(&moduli)))
                .collect::<Vec<_>>()
        };

        let y = eval_input_vec(b);
        let r = eval_input_vec(b);

        b.output_bundles(y.as_slice())?;
        b.output_bundles(r.as_slice())?;
        Ok(())
    }

    #[test]
    pub(crate) fn test_cds() {
        let mut rng = thread_rng();
        let p = <F as PrimeField>::Params::MODULUS.0 as u128;
        let n = num_bits(p);

        let layer_size = 100;

        // Construct CDS circuit
        let mut b = CircuitBuilder::new();
        cds::<TenBitExpParams>(&mut b, layer_size).unwrap();
        let mut c = b.finish();
        let _ = c.print_info();
        let ((en, ev), out_labels) = fancy_garbling::garble_out(&mut c).unwrap();
        let deltas = en
            .get_deltas()
            .iter()
            .map(|(_, v)| v.clone())
            .collect::<Vec<_>>();

        // Mask all the output labels with a OTP
        let otp: u128 = rng.gen();
        let otp_labels = out_labels
            .into_iter()
            .map(|w| w.plus_mov(&fancy_garbling::Wire::from_block(otp.into(), 2)))
            .collect::<Vec<_>>();

        // Construct dummy circuit to check result
        let mut b_relu = CircuitBuilder::new();
        dummy(&mut b_relu, layer_size).unwrap();
        let mut c_relu = b_relu.finish();
        let (_, ev_relu) =
            fancy_garbling::garble_chain(&mut c_relu, Vec::new(), otp_labels, &deltas.as_slice())
                .unwrap();

        // Client has y and r
        let y = (0..layer_size)
            .map(|_| generate_random_number(&mut rng).1)
            .collect::<Vec<_>>();
        let r = (0..layer_size)
            .map(|_| generate_random_number(&mut rng).1)
            .collect::<Vec<_>>();

        // Server has alpha and beta
        let alpha = generate_random_number(&mut rng).1;
        let beta = generate_random_number(&mut rng).1;

        // Both parties have shares of ay, br
        let ay = y.iter().map(|e| *e * alpha).collect::<Vec<TenBitExpFP>>();
        let br = r.iter().map(|e| *e * beta).collect::<Vec<TenBitExpFP>>();

        let ay_shares = ay
            .iter()
            .map(|e| e.share(&mut rng))
            .collect::<Vec<(_, _)>>();

        let br_shares = br
            .iter()
            .map(|e| e.share(&mut rng))
            .collect::<Vec<(_, _)>>();

        // Create vector of client inputs
        let evaluator_inputs = ay_shares
            .iter()
            .map(|e| util::u128_to_bits(e.1.inner.inner.into_repr().0 as u128, n))
            .chain(
                br_shares
                    .iter()
                    .map(|e| util::u128_to_bits(e.1.inner.inner.into_repr().0 as u128, n)),
            )
            .chain(
                y.iter()
                    .map(|e| util::u128_to_bits(e.inner.into_repr().0 as u128, n)),
            )
            .chain(
                r.iter()
                    .map(|e| util::u128_to_bits(e.inner.into_repr().0 as u128, n)),
            )
            .flatten()
            .collect::<Vec<_>>();

        // Create vector of server inputs
        let mut garbler_inputs = ay_shares
            .iter()
            .map(|e| util::u128_to_bits(e.0.inner.inner.into_repr().0 as u128, n))
            .chain(
                br_shares
                    .iter()
                    .map(|e| util::u128_to_bits(e.0.inner.inner.into_repr().0 as u128, n)),
            )
            .flatten()
            .collect::<Vec<_>>();
        garbler_inputs.extend(util::u128_to_bits(alpha.inner.into_repr().0 as u128, n));
        garbler_inputs.extend(util::u128_to_bits(beta.inner.into_repr().0 as u128, n));

        // Encode inputs and evaluate circuit
        let xs = en.encode_garbler_inputs(&garbler_inputs);
        let ys = en.encode_evaluator_inputs(&evaluator_inputs);
        let masked_labels = ev.eval_labels(&mut c, &xs, &ys).unwrap();

        // Server checks bits and if true reverse OTP
        // TODO let equality_labels = masked_labels.split_off(layer_size*n*2);
        let labels = masked_labels
            .into_iter()
            .map(|w| w.plus_mov(&fancy_garbling::Wire::from_block(otp.into(), 2)))
            .collect::<Vec<_>>();

        // Convert resulting bits to wire labels as input to the dummy
        // circuit
        let dummy_result = ev_relu
            .eval(&mut c_relu, labels.as_slice(), labels.as_slice())
            .unwrap();

        // Assert all the results are correct
        y.iter()
            .chain(r.iter())
            .zip(dummy_result.as_slice().chunks(n))
            .for_each(|(e1, e2)| {
                let expected = e1.inner.into_repr().0;
                let result = util::u128_from_bits(e2) as u64;
                // println!("Expected {}, got {}", expected, result);
                assert_eq!(expected, result, "Expected {}, got {}", expected, result);
            });
    }
}
