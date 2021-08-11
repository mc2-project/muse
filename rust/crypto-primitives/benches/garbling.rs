use algebra::{fields::near_mersenne_64::F, fixed_point::*, *};
use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use crypto_primitives::additive_share::Share;
use fancy_garbling::{
    circuit::{Circuit, CircuitBuilder},
    util,
    util::RngExt,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use std::time::Duration;

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -10.0 } else { 10.0 };
    let float: f64 = rng.gen();
    let f = TenBitExpFP::truncate_float(float * mul);
    let n = TenBitExpFP::from(f);
    (f, n)
}

struct TenBitExpParams {}
impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 3;
    const EXPONENT_CAPACITY: u8 = 10;
}

type TenBitExpFP = FixedPoint<TenBitExpParams>;

fn get_inputs_cds(layer_size: usize) -> (Vec<u16>, Vec<u16>) {
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let p = <F as PrimeField>::Params::MODULUS.0 as u128;
    let n = (p.next_power_of_two() * 2).trailing_zeros() as usize;

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

    // Server has input wires for online ReLU circuit
    let zero_labels = (0..2 * layer_size * n)
        .map(|_| rng.gen_u128())
        .collect::<Vec<_>>();
    let one_labels = (0..2 * layer_size * n)
        .map(|_| rng.gen_u128())
        .collect::<Vec<_>>();

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
        .chain(
            zero_labels
                .iter()
                .map(|e| util::u128_to_bits(Into::<u128>::into(*e), 128)),
        )
        .chain(
            one_labels
                .iter()
                .map(|e| util::u128_to_bits(Into::<u128>::into(*e), 128)),
        )
        .flatten()
        .collect::<Vec<_>>();
    garbler_inputs.extend(util::u128_to_bits(alpha.inner.into_repr().0 as u128, n));
    garbler_inputs.extend(util::u128_to_bits(beta.inner.into_repr().0 as u128, n));
    (garbler_inputs, evaluator_inputs)
}

fn make_relu(n: usize) -> Circuit {
    let mut b = CircuitBuilder::new();
    crypto_primitives::gc::relu::<TenBitExpParams>(&mut b, n).unwrap();
    b.finish()
}

fn make_cds(n: usize) -> Circuit {
    let mut b = CircuitBuilder::new();
    crypto_primitives::gc::cds::<TenBitExpParams>(&mut b, n).unwrap();
    b.finish()
}

fn relu_gb(c: &mut Criterion) {
    c.bench_function_over_inputs(
        &"relu_gb",
        move |bench: &mut Bencher, &num: &&usize| {
            let mut c = make_relu(*num);
            bench.iter(|| {
                let gb = fancy_garbling::garble(&mut c).unwrap();
                criterion::black_box(gb);
            });
        },
        &[1, 10, 100usize],
    );
}

fn cds_gb(c: &mut Criterion) {
    c.bench_function_over_inputs(
        &"cds_gb",
        move |bench: &mut Bencher, &num: &&usize| {
            let mut c = make_cds(*num);
            bench.iter(|| {
                let gb = fancy_garbling::garble(&mut c).unwrap();
                criterion::black_box(gb);
            });
        },
        &[1, 100, 1000],
    );
}

fn relu_ev(c: &mut Criterion) {
    c.bench_function_over_inputs(
        &"relu_ev",
        move |bench: &mut Bencher, &num: &&usize| {
            let mut rng = rand::thread_rng();
            let mut c = make_relu(*num);
            let (en, ev) = fancy_garbling::garble(&mut c).unwrap();
            let gb_inps: Vec<_> = (0..c.num_garbler_inputs())
                .map(|i| rng.gen_u16() % c.garbler_input_mod(i))
                .collect();
            let ev_inps: Vec<_> = (0..c.num_evaluator_inputs())
                .map(|i| rng.gen_u16() % c.evaluator_input_mod(i))
                .collect();
            let xs = en.encode_garbler_inputs(&gb_inps);
            let ys = en.encode_evaluator_inputs(&ev_inps);
            bench.iter(|| {
                let ys = ev.eval(&mut c, &xs, &ys).unwrap();
                criterion::black_box(ys);
            });
        },
        &[1, 10, 100usize],
    );
}

fn cds_ev(c: &mut Criterion) {
    c.bench_function_over_inputs(
        &"cds_ev",
        move |bench: &mut Bencher, &num: &&usize| {
            let mut c = make_cds(*num);
            let (en, ev) = fancy_garbling::garble(&mut c).unwrap();

            let (gb_inps, ev_inps) = get_inputs_cds(*num);

            let xs = en.encode_garbler_inputs(&gb_inps);
            let ys = en.encode_evaluator_inputs(&ev_inps);
            bench.iter(|| {
                let ys = ev.eval(&mut c, &xs, &ys).unwrap();
                criterion::black_box(ys);
            });
        },
        &[1, 100, 1000usize],
    );
}

criterion_group! {
    name = garbling;
    config = Criterion::default().warm_up_time(Duration::from_millis(100)).sample_size(10);
    targets = relu_gb, relu_ev, cds_gb, cds_ev
}

criterion_main!(garbling);
