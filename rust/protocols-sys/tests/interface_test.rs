use algebra::{
    fields::{near_mersenne_64::F, PrimeField},
    fixed_point::{FixedPoint, FixedPointParameters},
    UniformRandom,
};
use crypto_primitives::{
    additive_share::{AuthAdditiveShare, Share},
    beavers_mul::Triple,
};
use itertools::izip;
use ndarray::s;
use neural_network::{layers::*, tensors::*, Evaluate};
use protocols_sys::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;

type AdditiveShare<P> = crypto_primitives::AdditiveShare<FixedPoint<P>>;

struct TenBitExpParams {}
impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 3;
    const EXPONENT_CAPACITY: u8 = 8;
}

type TenBitExpFP = FixedPoint<TenBitExpParams>;
type TenBitAS = AdditiveShare<TenBitExpParams>;

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -10.0 } else { 10.0 };
    let mut float: f64 = rng.gen();
    float += 1.0;
    let f = TenBitExpFP::truncate_float(float * mul);
    let n = TenBitExpFP::from(f);
    (f, n)
}

fn print_output_f64(output: &Output<TenBitExpFP>) {
    output
        .slice(s![0, .., .., ..])
        .outer_iter()
        .for_each(|out_c| {
            out_c.outer_iter().for_each(|inp_c| {
                inp_c.iter().for_each(|e| print!("{:.2}, ", f64::from(*e)));
                println!("");
            });
        });
}

fn print_output_u64(output: &Output<TenBitExpFP>) {
    output
        .slice(s![0, .., .., ..])
        .outer_iter()
        .for_each(|out_c| {
            out_c.outer_iter().for_each(|inp_c| {
                inp_c
                    .iter()
                    .for_each(|e| print!("{:.2}, ", e.inner.into_repr().0));
                println!("");
            });
        });
}

// Compares floats to 2 decimal points
fn approx_equal(f1: f64, f2: f64) -> bool {
    f64::trunc(100. * f1) == f64::trunc(100. * f2)
}

fn interface<R: Rng + rand::CryptoRng>(
    client_acg: &mut SealClientACG,
    server_acg: &mut SealServerACG,
    input_dims: (usize, usize, usize, usize),
    output_dims: (usize, usize, usize, usize),
    pt_layer: &LinearLayer<TenBitExpFP, TenBitExpFP>,
    rng: &mut R,
) -> bool {
    let mac_key_a = F::uniform(rng);
    let mac_key_b = F::uniform(rng);

    // Client preprocessing
    let mut r = Input::zeros(input_dims);
    r.iter_mut()
        .for_each(|e| *e = generate_random_number(rng).1);

    let input_ct_vec = client_acg.preprocess(&r.to_repr());

    // Server preprocessing
    let mut linear_share = Output::zeros(output_dims);
    linear_share
        .iter_mut()
        .for_each(|e| *e = generate_random_number(rng).1);

    let mut linear_mac_share = Output::zeros(output_dims);
    linear_mac_share
        .iter_mut()
        .for_each(|e| *e = generate_random_number(rng).1);

    let mut r_mac_share = Input::zeros(input_dims);
    r_mac_share
        .iter_mut()
        .for_each(|e| *e = generate_random_number(rng).1);

    server_acg.preprocess(
        &linear_share.to_repr(),
        &linear_mac_share.to_repr(),
        &r_mac_share.to_repr(),
        mac_key_a.into_repr().0,
        mac_key_b.into_repr().0,
    );

    // Server receive ciphertext and compute convolution
    let (linear_ct, linear_mac_ct, r_mac_ct) = server_acg.process(input_ct_vec);

    // Client receives ciphertexts
    client_acg.decrypt(linear_ct, linear_mac_ct, r_mac_ct);

    let mut linear: Output<TenBitAS> = Output::zeros(output_dims);
    let mut linear_mac: Output<TenBitAS> = Output::zeros(output_dims);
    let mut r_mac: Input<TenBitAS> = Input::zeros(input_dims);

    // The interface changed here after making this test so from here on
    // the code is very messy
    let mut linear_auth =
        Output::auth_share_from_parts(Output::zeros(output_dims), Output::zeros(output_dims));
    let mut r_auth = Input::zeros(input_dims);
    client_acg.postprocess::<TenBitExpParams>(&mut linear_auth, &mut r_auth);

    izip!(linear.iter_mut(), linear_mac.iter_mut(), linear_auth.iter()).for_each(|(s, m, a)| {
        *s = FixedPoint::with_num_muls(a.get_value().inner, 1).into();
        *m = FixedPoint::new(a.get_mac().inner).into();
    });

    izip!(r_mac.iter_mut(), r_auth.iter()).for_each(|(m, a)| *m = FixedPoint::new(*a).into());

    let mut success = true;

    println!(
        "MACs: {} {}",
        mac_key_a.into_repr().0,
        mac_key_b.into_repr().0
    );

    println!("\nPlaintext linear:");
    let linear_pt = pt_layer.evaluate(&r);
    print_output_f64(&linear_pt);

    println!("Linear:");
    let mut linear_result = Output::zeros(output_dims);
    linear_result
        .iter_mut()
        .zip(linear.iter().zip(linear_share.iter()))
        .zip(linear_pt.iter())
        .for_each(|((r, (s1, s2)), p)| {
            *r = FixedPoint::randomize_local_share(s1, &s2.inner).inner;
            success &= approx_equal(f64::from(*r), f64::from(*p));
        });
    print_output_f64(&linear_result);

    println!("\nPlaintext linear MAC:");
    let mut linear_mac_pt = linear_pt.clone();
    linear_mac_pt
        .iter_mut()
        .zip(linear_share.iter())
        .for_each(|(e1, e2)| {
            *e1 = TenBitExpFP::new(((*e1).inner - (*e2).inner) * mac_key_a);
        });
    print_output_u64(&linear_mac_pt);

    println!("Linear MAC:");
    let mut linear_mac_result: Output<TenBitExpFP> = Output::zeros(output_dims);
    linear_mac_result
        .iter_mut()
        .zip(linear_mac.iter().zip(linear_mac_share.iter()))
        .zip(linear_mac_pt.iter())
        .for_each(|((r, (s1, s2)), p)| {
            *r = FixedPoint::randomize_local_share(s1, &s2.inner).inner;
            success &= r.inner.into_repr().0 == p.inner.into_repr().0;
        });
    print_output_u64(&linear_mac_result);

    println!("\nPlaintext R MAC:");
    let mut r_mac_pt = r.clone();
    r_mac_pt
        .iter_mut()
        .for_each(|e| *e = TenBitExpFP::new((*e).inner * mac_key_b));
    print_output_u64(&r_mac_pt);

    println!("R MAC:");
    let mut r_mac_result: Output<TenBitExpFP> = Output::zeros(input_dims);
    r_mac_result
        .iter_mut()
        .zip(r_mac.iter().zip(r_mac_share.iter()))
        .zip(r_mac_pt.iter())
        .for_each(|((r, (s1, s2)), p)| {
            *r = FixedPoint::randomize_local_share(s1, &s2.inner).inner;
            success &= r.inner.into_repr().0 == p.inner.into_repr().0;
        });
    print_output_u64(&r_mac_result);
    success
}

#[test]
fn test_convolution() {
    use neural_network::layers::convolution::*;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    // Set the parameters for the convolution.
    let input_dims = (1, 1, 28, 28);
    let kernel_dims = (16, 1, 5, 5);
    let stride = 1;
    let padding = Padding::Valid;
    // Sample a random kernel.
    let mut kernel = Kernel::zeros(kernel_dims);
    let mut bias = Kernel::zeros((kernel_dims.0, 1, 1, 1));
    kernel
        .iter_mut()
        .for_each(|ker_i| *ker_i = generate_random_number(&mut rng).1);
    // Offline phase doesn't interact with bias so this can be 0
    bias.iter_mut()
        .for_each(|bias_i| *bias_i = TenBitExpFP::from(0.0));

    let layer_params =
        Conv2dParams::<TenBitAS, _>::new(padding, stride, kernel.clone(), bias.clone());
    let pt_layer_params =
        Conv2dParams::<TenBitExpFP, _>::new(padding, stride, kernel.clone(), bias.clone());
    let output_dims = layer_params.calculate_output_size(input_dims);
    let layer_dims = LayerDims {
        input_dims,
        output_dims,
    };
    let layer = LinearLayer::Conv2d {
        dims: layer_dims,
        params: layer_params,
    };
    let layer_info = (&layer).into();

    let pt_layer = LinearLayer::Conv2d {
        dims: layer_dims,
        params: pt_layer_params,
    };

    // Keygen
    let mut key_share = KeyShare::new();
    let (cfhe, keys_vec) = key_share.generate();
    let sfhe = key_share.receive(keys_vec);

    let input_dims = layer.input_dimensions();
    let output_dims = layer.output_dimensions();

    let mut client_acg = SealClientACG::Conv2D(client_acg::Conv2D::new(
        &cfhe,
        &layer_info,
        input_dims,
        output_dims,
    ));
    let mut server_acg =
        SealServerACG::Conv2D(server_acg::Conv2D::new(&sfhe, &layer, &kernel.to_repr()));

    assert_eq!(
        interface(
            &mut client_acg,
            &mut server_acg,
            input_dims,
            output_dims,
            &pt_layer,
            &mut rng
        ),
        true
    );
}

#[test]
fn test_fully_connected() {
    use neural_network::layers::fully_connected::*;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    // Set the parameters for the layer
    let input_dims = (1, 3, 32, 32);
    let kernel_dims = (10, 3, 32, 32);

    // Sample a random kernel.
    let mut kernel = Kernel::zeros(kernel_dims);
    let mut bias = Kernel::zeros((kernel_dims.0, 1, 1, 1));
    kernel
        .iter_mut()
        .for_each(|ker_i| *ker_i = generate_random_number(&mut rng).1);
    // Offline phase doesn't interact with bias so this can be 0
    bias.iter_mut()
        .for_each(|bias_i| *bias_i = TenBitExpFP::from(0.0));

    let layer_params = FullyConnectedParams::<TenBitAS, _>::new(kernel.clone(), bias.clone());
    let pt_layer_params = FullyConnectedParams::<TenBitExpFP, _>::new(kernel.clone(), bias.clone());
    let output_dims = layer_params.calculate_output_size(input_dims);
    let layer_dims = LayerDims {
        input_dims,
        output_dims,
    };
    let layer = LinearLayer::FullyConnected {
        dims: layer_dims,
        params: layer_params,
    };
    let layer_info = (&layer).into();

    let pt_layer = LinearLayer::FullyConnected {
        dims: layer_dims,
        params: pt_layer_params,
    };

    // Keygen
    let mut key_share = KeyShare::new();
    let (cfhe, keys_vec) = key_share.generate();
    let sfhe = key_share.receive(keys_vec);

    let input_dims = layer.input_dimensions();
    let output_dims = layer.output_dimensions();

    let mut client_acg = SealClientACG::FullyConnected(client_acg::FullyConnected::new(
        &cfhe,
        &layer_info,
        input_dims,
        output_dims,
    ));
    let mut server_acg = SealServerACG::FullyConnected(server_acg::FullyConnected::new(
        &sfhe,
        &layer,
        &kernel.to_repr(),
    ));

    assert_eq!(
        interface(
            &mut client_acg,
            &mut server_acg,
            input_dims,
            output_dims,
            &pt_layer,
            &mut rng
        ),
        true
    );
}

#[inline]
fn to_u64(x: &Vec<F>) -> Vec<u64> {
    x.iter().map(|e| e.into_repr().0).collect()
}

#[test]
fn test_rand_gen() {
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let num = 100000;

    // Keygen
    let mut key_share = KeyShare::new();
    let (cfhe, keys_vec) = key_share.generate();
    let sfhe = key_share.receive(keys_vec);
    let mac_key = F::uniform(&mut rng);

    let client_gen = SealClientGen::new(&cfhe);
    let server_gen = SealServerGen::new(&sfhe, mac_key.into_repr().0);

    let mut client_randomizers = Vec::with_capacity(num);
    let mut server_randomizers = Vec::with_capacity(num);
    let mut server_shares = Vec::with_capacity(num);
    let mut server_mac_shares = Vec::with_capacity(num);
    for _ in 0..num {
        client_randomizers.push(F::uniform(&mut rng));
        server_randomizers.push(F::uniform(&mut rng));
        server_shares.push(F::uniform(&mut rng));
        server_mac_shares.push(F::uniform(&mut rng));
    }

    let (mut client_triples, mut r_ct) =
        client_gen.rands_preprocess(to_u64(&client_randomizers).as_slice());
    let mut server_triples = server_gen.rands_preprocess(
        to_u64(&server_randomizers).as_slice(),
        to_u64(&server_shares).as_slice(),
        to_u64(&server_mac_shares).as_slice(),
    );
    let (mut r_ct, mut r_mac_ct) =
        server_gen.rands_online(&mut server_triples, r_ct.as_mut_slice());

    let (client_shares, client_mac_shares) = client_gen.rands_postprocess(
        &mut client_triples,
        r_ct.as_mut_slice(),
        r_mac_ct.as_mut_slice(),
    );

    let server_rands = server_shares
        .into_iter()
        .zip(server_mac_shares.into_iter())
        .map(|(s, m)| AuthAdditiveShare::new(s, m))
        .collect::<Vec<_>>();
    let client_rands = client_shares
        .into_iter()
        .zip(client_mac_shares.into_iter())
        .map(|(s, m)| AuthAdditiveShare::new(F::from_repr(s.into()), F::from_repr(m.into())))
        .collect::<Vec<_>>();

    server_rands
        .iter()
        .zip(client_rands.iter())
        .for_each(|(s, c)| assert!(s.combine(&c, &mac_key).is_ok()));
}

#[test]
fn test_triple_gen() {
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let num = 100000;

    // Keygen
    let mut key_share = KeyShare::new();
    let (cfhe, keys_vec) = key_share.generate();
    let sfhe = key_share.receive(keys_vec);
    let mac_key = F::uniform(&mut rng);

    let client_gen = SealClientGen::new(&cfhe);
    let server_gen = SealServerGen::new(&sfhe, mac_key.into_repr().0);

    let mut client_a_rand = Vec::with_capacity(num);
    let mut client_b_rand = Vec::with_capacity(num);

    let mut server_a_rand = Vec::with_capacity(num);
    let mut server_b_rand = Vec::with_capacity(num);
    let mut server_c_rand = Vec::with_capacity(num);
    let mut server_a_shares = Vec::with_capacity(num);
    let mut server_b_shares = Vec::with_capacity(num);
    let mut server_c_shares = Vec::with_capacity(num);
    let mut server_a_mac_shares = Vec::with_capacity(num);
    let mut server_b_mac_shares = Vec::with_capacity(num);
    let mut server_c_mac_shares = Vec::with_capacity(num);
    for i in 0..num {
        client_a_rand.push(F::uniform(&mut rng));
        client_b_rand.push(F::uniform(&mut rng));

        server_a_rand.push(F::uniform(&mut rng));
        server_b_rand.push(F::uniform(&mut rng));
        server_c_rand.push(server_a_rand[i] * server_b_rand[i]);
        server_a_shares.push(F::uniform(&mut rng));
        server_b_shares.push(F::uniform(&mut rng));
        server_c_shares.push(F::uniform(&mut rng));
        server_a_mac_shares.push(F::uniform(&mut rng));
        server_b_mac_shares.push(F::uniform(&mut rng));
        server_c_mac_shares.push(F::uniform(&mut rng));
    }

    let (mut client_triples, mut a_ct, mut b_ct) = client_gen.triples_preprocess(
        to_u64(&client_a_rand).as_slice(),
        to_u64(&client_b_rand).as_slice(),
    );

    let mut server_triples = server_gen.triples_preprocess(
        to_u64(&server_a_rand).as_slice(),
        to_u64(&server_b_rand).as_slice(),
        to_u64(&server_c_rand).as_slice(),
        to_u64(&server_a_shares).as_slice(),
        to_u64(&server_b_shares).as_slice(),
        to_u64(&server_c_shares).as_slice(),
        to_u64(&server_a_mac_shares).as_slice(),
        to_u64(&server_b_mac_shares).as_slice(),
        to_u64(&server_c_mac_shares).as_slice(),
    );

    let (mut a_ct, mut b_ct, mut c_ct, mut a_mac_ct, mut b_mac_ct, mut c_mac_ct) = server_gen
        .triples_online(
            &mut server_triples,
            a_ct.as_mut_slice(),
            b_ct.as_mut_slice(),
        );

    let (
        client_a_shares,
        client_b_shares,
        client_c_shares,
        client_a_mac_shares,
        client_b_mac_shares,
        client_c_mac_shares,
    ) = client_gen.triples_postprocess(
        &mut client_triples,
        a_ct.as_mut_slice(),
        b_ct.as_mut_slice(),
        c_ct.as_mut_slice(),
        a_mac_ct.as_mut_slice(),
        b_mac_ct.as_mut_slice(),
        c_mac_ct.as_mut_slice(),
    );

    let server_triples = izip!(
        server_a_shares,
        server_b_shares,
        server_c_shares,
        server_a_mac_shares,
        server_b_mac_shares,
        server_c_mac_shares
    )
    .map(|(a, b, c, a_m, b_m, c_m)| Triple {
        a: AuthAdditiveShare::new(a, a_m),
        b: AuthAdditiveShare::new(b, b_m),
        c: AuthAdditiveShare::new(c, c_m),
    });

    let client_triples = izip!(
        client_a_shares,
        client_b_shares,
        client_c_shares,
        client_a_mac_shares,
        client_b_mac_shares,
        client_c_mac_shares
    )
    .map(|(a, b, c, a_m, b_m, c_m)| Triple {
        a: AuthAdditiveShare::new(F::from_repr(a.into()), F::from_repr(a_m.into())),
        b: AuthAdditiveShare::new(F::from_repr(b.into()), F::from_repr(b_m.into())),
        c: AuthAdditiveShare::new(F::from_repr(c.into()), F::from_repr(c_m.into())),
    });

    izip!(server_triples, client_triples).for_each(|(s, c)| {
        let a = s.a.combine(&c.a, &mac_key);
        let b = s.b.combine(&c.b, &mac_key);
        let c = s.c.combine(&c.c, &mac_key);
        assert!(a.is_ok());
        assert!(b.is_ok());
        assert!(c.is_ok());
        assert_eq!(c.unwrap(), a.unwrap() * b.unwrap());
    });
}
