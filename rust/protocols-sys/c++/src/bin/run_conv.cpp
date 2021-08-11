#include <cstddef>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <memory>
#include <limits>

#include "math.h"
#include <time.h>
#include "conv2d.h"
#include "fc_layer.h"
#include "im2col.h"
#include "interface.h"

using namespace std;

bool pass = true;

/* Uses the C interface to perform convolution */
ClientShares interface_conv(ServerFHE &sfhe, ClientFHE &cfhe, Metadata data, Image image, Filters filters,
        Image r_mac_share, Image linear_share, Image linear_mac_share, u64 mac_key_a, u64 mac_key_b) {
    chrono::high_resolution_clock::time_point time_start, time_end;

    cout << "    Client Preprocessing: ";
    time_start = chrono::high_resolution_clock::now();

    ClientShares client_shares = client_conv_preprocess(&cfhe, &data, image);
    
    time_end = chrono::high_resolution_clock::now();
    auto time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "[" << time_diff.count() << " microseconds]" << endl;
    // ------------------------------------------------ 
    cout << "    Server Preprocessing: ";
    time_start = chrono::high_resolution_clock::now();

    char**** masks = server_conv_preprocess(&sfhe, &data, filters);
    ServerShares server_shares = server_conv_preprocess_shares(&sfhe, &data, linear_share,
            linear_mac_share, r_mac_share, mac_key_a, mac_key_b);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "[" << time_diff.count() << " microseconds]" << endl;
    // ------------------------------------------------ 
    cout << "    Convolution: ";
    time_start = chrono::high_resolution_clock::now();
    
    server_conv_online(&sfhe, &data, client_shares.input_ct, masks, &server_shares);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "[" << time_diff.count() << " microseconds]" << endl;
    // ------------------------------------------------ 
    client_shares.linear_ct.inner = (char*) malloc(sizeof(char)*server_shares.linear_ct.size);
    client_shares.linear_mac_ct.inner = (char*) malloc(sizeof(char)*server_shares.linear_mac_ct.size);
    client_shares.r_mac_ct.inner = (char*) malloc(sizeof(char)*server_shares.r_mac_ct.size);
    client_shares.linear_ct.size = server_shares.linear_ct.size;
    client_shares.linear_mac_ct.size = server_shares.linear_mac_ct.size;
    //delete[] shares->input_ct;
    client_shares.r_mac_ct.size = server_shares.r_mac_ct.size;
    memcpy(client_shares.linear_ct.inner, server_shares.linear_ct.inner, server_shares.linear_ct.size);
    memcpy(client_shares.linear_mac_ct.inner, server_shares.linear_mac_ct.inner, server_shares.linear_mac_ct.size);
    memcpy(client_shares.r_mac_ct.inner, server_shares.r_mac_ct.inner, server_shares.r_mac_ct.size);

    // ------------------------------------------------ 
    cout << "    Post-processing: ";
    time_start = chrono::high_resolution_clock::now();

    client_conv_decrypt(&cfhe, &data, &client_shares);
    
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "[" << time_diff.count() << " microseconds]" << endl;
    // ------------------------------------------------ 
  
    // Free C++ allocations
    server_conv_free(&data, masks, &server_shares);
    return client_shares;
}

void free_share(Image share, int chans) {
    for (int chan = 0; chan < chans; chan++) {
        delete[] share[chan];
    }
    delete[] share;
}

/* Runs plaintext and homomorphic convolution and compares result */
bool run_conv(Image image, Filters filters, int image_h, int image_w, int filter_h,
        int filter_w, int inp_chans, int out_chans, bool pad_valid, int stride_h,
        int stride_w, bool verbose) {
    chrono::high_resolution_clock::time_point time_start, time_end;
    
    // PRG for generating shares
    random_device rd;
    mt19937 engine(rd());
    uniform_int_distribution<u64> dist(0, PLAINTEXT_MODULUS);
    auto gen = [&dist, &engine](){
        return dist(engine);
    };
  
    /* --------------- KeyGen/Preprocessing -------------------- */
    SerialCT key_share;
    ClientFHE cfhe = client_keygen(&key_share);
    ServerFHE sfhe = server_keygen(key_share); 
    
    uint64_t mac_key_a = gen();
    uint64_t mac_key_b = gen();
    Metadata data = conv_metadata(cfhe.encoder, image_h, image_w, filter_h, filter_w, inp_chans, 
        out_chans, stride_h, stride_w, pad_valid);
    /* ---------------------------------------------------------- */

    // Generate server's input secret shares
    Image r_mac_share = new u64*[inp_chans];
    for (int chan = 0; chan < inp_chans; chan++) {
        Channel channel = new u64[image_h*image_w];
        for (int idx = 0; idx < image_h*image_w; idx++)
            channel[idx] = gen();
        r_mac_share[chan] = channel;
    };
    
    // Generate server's linear secret shares
    Image linear_share = new u64*[out_chans];
    Image linear_mac_share = new u64*[out_chans];
    for (int chan = 0; chan < out_chans; chan++) {
        Channel channel = new u64[data.output_h * data.output_w];
        Channel mac_channel = new u64[data.output_h * data.output_w];
        for (int idx = 0; idx < data.output_h*data.output_w; idx++) {
            channel[idx] = gen();
            mac_channel[idx] = gen();
        }
        linear_share[chan] = channel;
        linear_mac_share[chan] = mac_channel;
    };

    // Convert the raw pointer to an Eigen matrix for plaintext evaluation
    // and generate pt_r_mac_share
    EImage eimage(inp_chans);
    Image pt_r_mac_share = new u64*[data.inp_chans];
    for (int chan = 0; chan < inp_chans; chan++) {
        EChannel echannel(image_h, image_w);
        Channel mac_channel = new u64[data.image_h*data.image_w];
        for (int idx = 0; idx < image_h*image_w; idx++) {
            echannel(idx/image_w, idx%image_h) = image[chan][idx];
            // We first add the modulus so that we don't underflow the u64
            mac_channel[idx] = (PLAINTEXT_MODULUS + (__int128) image[chan][idx] * mac_key_b - r_mac_share[chan][idx]) % PLAINTEXT_MODULUS;
        }
        eimage[chan] = echannel;
        pt_r_mac_share[chan] = mac_channel;
    }

    EFilters efilters(out_chans);
    for (int o_chan = 0; o_chan < out_chans; o_chan++) {
        EImage eimage(inp_chans);
        for (int chan = 0; chan < inp_chans; chan++) {
            EChannel echannel(filter_h, filter_w);
            for (int idx = 0; idx < filter_h*filter_w; idx++) {
                echannel(idx/filter_h, idx%filter_w) = filters[o_chan][chan][idx];
            }
            eimage[chan] = echannel;
        }
        efilters[o_chan] = eimage;
    }

    cout << "Plaintext:\n";
    time_start = chrono::high_resolution_clock::now();

    EImage pt_result = im2col_conv2D(&eimage, &efilters, pad_valid, stride_h, stride_w);
    // Compute linear shares 
    Image pt_linear_share = new u64*[data.out_chans];
    Image pt_linear_mac_share = new u64*[data.out_chans];
    for (int chan = 0; chan < data.out_chans; chan++) {
        Channel channel = new u64[data.output_h * data.output_w];
        Channel mac_channel = new u64[data.output_h * data.output_w];
        for (int idx = 0; idx < data.output_h*data.output_w; idx++) {
            // We first add the modulus so that we don't underflow the u64
            channel[idx] = (PLAINTEXT_MODULUS + pt_result[chan](idx/data.output_w,idx%data.output_w) - linear_share[chan][idx]) % PLAINTEXT_MODULUS;
            mac_channel[idx] = (PLAINTEXT_MODULUS + (__int128) channel[idx] * mac_key_a - linear_mac_share[chan][idx]) % PLAINTEXT_MODULUS;
        };
        pt_linear_share[chan] = channel;
        pt_linear_mac_share[chan] = mac_channel;
    };
    
    time_end = chrono::high_resolution_clock::now();
    auto time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "Done [" << time_diff.count() << " microseconds]\n\n";

    cout << "Homomorphic:\n";
    time_start = chrono::high_resolution_clock::now();

    auto shares = interface_conv(sfhe, cfhe, data, image, filters, r_mac_share, linear_share, linear_mac_share, mac_key_a, mac_key_b);

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "Done [" << time_diff.count() << " microseconds]\n\n";

    if (verbose) {
        print(pt_linear_share, data.out_chans, data.output_h, data.output_w);
        cout << "Linear MAC: " << endl;
        print(pt_linear_mac_share, data.out_chans, data.output_h, data.output_w);
        cout << "R MAC: " << endl;
        print(pt_r_mac_share, data.inp_chans, data.image_h, data.image_w);
    }

    if (verbose) {
        print(shares.linear, data.out_chans, data.output_h, data.output_w);
        cout << "Linear MAC: " << endl;
        print(shares.linear_mac, data.out_chans, data.output_h, data.output_w);
        cout << "R MAC: " << endl;
        print(shares.r_mac, data.inp_chans, data.image_h, data.image_w);
    }
    
    // Compare linear results
    for (int i = 0; i < data.out_chans; i++) {
        for (int j = 0; j < data.output_w*data.output_h; j++) {
            if ((pt_linear_share[i][j] != shares.linear[i][j]) ||
                    (pt_linear_mac_share[i][j] != shares.linear_mac[i][j])) {
                pass = false;
            }
        }
    }
    // Compare r_mac_share
    for (int i = 0; i < data.inp_chans; i++) {
        for (int j = 0; j < data.image_h * data.image_w; j++) {
            if (pt_r_mac_share[i][j] != shares.r_mac[i][j])
                pass = false;
        }
    }
    
    // Free stuff
    free_ct(&key_share);
    client_free_keys(&cfhe);
    server_free_keys(&sfhe);
    free(shares.linear_ct.inner);
    free(shares.linear_mac_ct.inner);
    free(shares.r_mac_ct.inner);
    client_conv_free(&data, &shares);
    free_share(linear_share, data.out_chans);
    free_share(linear_mac_share, data.out_chans);
    free_share(r_mac_share, data.inp_chans);
    free_share(pt_linear_share, data.out_chans);
    free_share(pt_linear_mac_share, data.out_chans);
    free_share(pt_r_mac_share, data.inp_chans);
    return pass;
}
