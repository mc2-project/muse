<h1 align="center">Muse</h1>

___Muse___ is a Python, C++, and Rust library for **Secure Convolutional Neural Network Inference Resilient to Malicious Clients**. 

This library was initially developed as part of the paper *"[Muse: Secure Inference Reslient to Malicious Clients][muse]"*, and is released under the MIT License and the Apache v2 License (see [License](#license)).

**WARNING:** This is an academic proof-of-concept prototype, and in particular has not received careful code review. Several components necessary for full security (but which don't affect benchmarks) are not completely implemented. Consequently, this implementation is NOT ready for production use.

## Overview

This library implements the components of a cryptographic system for efficient client-malicious inference on general convolutional neural networks as well as a model-extraction attack against semi-honest secure inference protocols based on additive secret sharing.

These constructions utilize an array of multi-party computation and machine-learning techniques, as described in the [Muse paper][muse].

## Directory structure

This repository contains several folders that implement the different building blocks of Muse. The high-level structure of the repository is as follows.

* [`python`](python): Python scripts for the model-extraction attack

* [`rust/algebra`](rust/algebra): Rust crate that provides finite fields

* [`rust/crypto-primitives`](rust/crypto-primitives): Rust crate that implements some useful cryptographic primitives

* [`rust/experiments`](rust/experiments): Rust crate for running latency and communication experiments

* [`rust/neural-network`](rust/neural-network): Rust crate that implements generic neural networks

* [`rust/protocols`](rust/protocols): Rust crate that implements cryptographic protocols

* [`rust/protocols-sys`](rust/crypto-primitives): Rust crate that provides the C++ backend for Muse's pre-processing phase and an FFI for the backend

## Build guide

The library compiles on the `nightly` toolchain of the Rust compiler. To install the latest version of Rust, first install `rustup` by following the instructions [here](https://rustup.rs/), or via your platform's package manager. Once `rustup` is installed, install the Rust toolchain by invoking:
```bash
rustup install nightly
```

Additionally, you will need to have the GCC, G++, pkg-config, OpenSSL, CMake, and Clang packages. On Ubuntu, these can be installed via:
```bash
sudo apt install pkg-config libssl-dev cmake g++ libclang-dev
```

After that, use `cargo`, the standard Rust build tool, to build the library:
```bash
git clone https://github.com/mc2-project/muse
cd muse/rust
cargo +nightly build --release
```

This library comes with unit and integration tests for each of the provided crates. Run these tests with:
```bash
cargo +nightly test
``` 

### Experiments

The rest of this README will explain how to run experiments on the various components of Muse in order to reproduce the results provided in the paper.

#### Tables 3/4 and Figure 10

##### Authenticated correlations generator (ACG)

To measure the cost of the ACG, first build the relevant binaries:
```bash
cargo +nightly build --bin acg-client --release --all-features;
cargo +nightly build --bin acg-server --release --all-features;
```

Then, execute these commands to run the experiment:
```bash
# On the server instance:
env RAYON_NUM_THREADS=2 cargo +nightly run --bin acg-server --release --all-features -- -m <0/1> 2>/dev/null > "./acg_time.txt"
# On the client instance:
env RAYON_NUM_THREADS=2 cargo +nightly run --bin acg-client --release --all-features -- -m <0/1> -i <server_ip> 2>/dev/null > "./acg_time.txt"
```
This will write out a trace of execution times and bandwidth used to `./acg_time.txt`.

Note that the `-m` flag controls which model architecture is used: MNIST (0) or MiniONN (1).

##### Garbling

To measure the cost of garbling the ReLU circuits, first build the relevant binaries:
```bash
cargo +nightly build --bin garbling-client --release --all-features;
cargo +nightly build --bin garbling-server --release --all-features;
```

Then, execute these commands to run the experiment:
```bash
# On the server instance:
env RAYON_NUM_THREADS=2 cargo +nightly run --bin garbling-server --release --all-features -- -m <0/1> 2>/dev/null > "./garbling_time.txt"
# On the client instance: 
env RAYON_NUM_THREADS=2 cargo +nightly run --bin garbling-client --release --all-features -- -m <0/1> -i <server_ip> 2>/dev/null > "./garbling_time.txt"
```
This will write out a trace of execution times and bandwidth used to `./garbling_time.txt`.

##### Triple Generation

To measure the cost of triple generation for the CDS protocol, first build the relevant binaries:
```bash
cargo +nightly build --bin triples-gen-client --release --all-features;
cargo +nightly build --bin triples-gen-server --release --all-features;
```

Then, execute these commands to run the experiment:
```bash
# On the server instance:
env RAYON_NUM_THREADS=6 cargo +nightly run --bin triples-gen-server --release --all-features -- -m <0/1> 2>/dev/null > "./triples_times.txt";
# On the client instance:
env RAYON_NUM_THREADS=6 cargo +nightly run --bin triples-gen-client --release --all-features -- -m <0/1> -i <server_ip> 2>/dev/null > "./triples_time.txt"
```
This will write out a trace to `./triples_time.txt`. Note that the results from Figure 10 can be reproduced by varying the number of threads in the `RAYON_NUM_THREADS` environment variable, and additionally including the `-n 10000000` flag.

##### Input Authentication

To measure the cost of input sharing for the CDS protocol, first build the relevant binaries:
```bash
cargo +nightly build --bin input-auth-client --release --all-features;
cargo +nightly build --bin input-auth-server --release --all-features;
```

Then, execute these commands to run the experiment:
```bash
# On the server instance:
env RAYON_NUM_THREADS=3 cargo +nightly run --bin input-auth-server --release --all-features -- -m <0/1> 2>/dev/null > "./input_auth_times.csv"
# On the client instance:
env RAYON_NUM_THREADS=3 cargo +nightly run --bin input-auth-client --release --all-features -- -m <0/1> -i <server_ip> 2>/dev/null > "./input_auth_time.txt"
```
This will write out a trace to `./input_auth_time.txt`.

##### CDS Evaluation

To measure the cost of evaluating the CDS protocol, first build the relevant binaries:
```bash
cargo +nightly build --bin cds-client --release --all-features;
cargo +nightly build --bin cds-server --release --all-features;
```

Then, execute these commands to run the experiment:
```bash
# On the server instance:
env RAYON_NUM_THREADS=2 cargo +nightly run --bin cds-server --release --all-features -- -m <0/1> 2>/dev/null > "./cds_time.csv"
# On the client instance:
env RAYON_NUM_THREADS=2 cargo +nightly run --bin cds-client --release --all-features -- -m <0/1> -i <server_ip> 2>/dev/null > "./cds_time.txt"
```
This will write out a trace of execution times to  `./cds_time.txt`.

##### Online phase

To measure the cost of the online phase, first build the relevant binaries.
(The code examples show how to do this for MNIST; for the MINIONN network, replace `mnist` with `minionn`)
```bash
cargo +nightly build --bin mnist-client --release --all-features;
cargo +nightly build --bin mnist-server --release --all-features;
```

Then, execute these commands to run the experiment:
```bash
# Start server:
env RAYON_NUM_THREADS=8 cargo +nightly run --bin mnist-server --release --all-features -- 2>/dev/null > "./mnist.txt"
# Start client:
env RAYON_NUM_THREADS=8 cargo +nightly run --bin mnist-client --release --all-features -- -i <server_ip> 2>/dev/null > "./mnist.txt"
```
This will write out a trace to `./mnist.txt`.  Note that the pre-processing phase times in this trace will be incorrect.

#### Figures 8 and 9

End-to-end experiments are currently implemented in the `end-to-end` branch (some bugs exist which are keeping this branch from being merged with `main` - these should be resolved soon).

To run these experiments, use the same commands described in the `Online phase` section above.

## License

Muse is licensed under either of the following licenses, at your discretion.

 * Apache License Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

Unless you explicitly state otherwise, any contribution submitted for inclusion in Muse by you shall be dual licensed as above (as defined in the Apache v2 License), without any additional terms or conditions.

[muse]: https://eprint.iacr.org/2021/1040

## Reference paper

[_Muse: Secure Inference Resilient to Malicious Clients_][muse]    
[Ryan Lehmkuhl](https://www.github.com/ryanleh), [Pratyush Mishra](https://www.github.com/pratyush), Akshayaram Srinivasan, and Raluca Ada Popa    
*Usenix Security Symposium 2021*

## Acknowledgements

This work was supported by:
the National Science Foundation;
and donations from Sloan Foundation, Bakar and Hellman Fellows Fund, Alibaba, Amazon Web Services, Ant Financial, Arm, Capital One, Ericsson, Facebook, Google, Intel, Microsoft, Scotiabank, Splunk and VMware

Some parts of the finite field arithmetic infrastructure in the `algebra` crate have been adapted from code in the [`algebra`](https://github.com/arkworks-rs/snark) crate.
