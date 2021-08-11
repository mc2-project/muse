/*
 *  Example of using Delphi Offline's C interface
 *
 *  Created on: June 10, 2019
 *      Author: ryanleh
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "interface.h"
#include <time.h>
#include <string.h>

typedef uint64_t u64;

void conv(ClientFHE* cfhe, ServerFHE* sfhe, int image_h, int image_w, int filter_h, int filter_w,
    int inp_chans, int out_chans, int stride, bool pad_valid) {
    Metadata data = conv_metadata(cfhe->encoder, image_h, image_w, filter_h, filter_w, inp_chans, 
        out_chans, stride, stride, pad_valid);
   
    printf("\nClient Preprocessing: ");
    float origin = (float)clock()/CLOCKS_PER_SEC;

    u64** input = (u64**) malloc(sizeof(u64*)*data.inp_chans);
    for (int chan = 0; chan < data.inp_chans; chan++) {
        input[chan] = (u64*) malloc(sizeof(u64)*data.image_size);
        for (int idx = 0; idx < data.image_size; idx++)
            input[chan][idx] = idx;
    }

    for (int chan = 0; chan < data.inp_chans; chan++) {
        int idx = 0;
        for (int row = 0; row < data.image_h; row++) {
            printf(" [");
            int col = 0;
            for (; col < data.image_w-1; col++) {
                printf("%d, " , input[chan][row*data.output_w + col]);
            }
            printf("%d ]\n" , input[chan][row*data.output_w + col]);
        }
        printf("\n");
    }

    ClientShares client_shares = client_conv_preprocess(cfhe, &data, input);

    float endTime = (float)clock()/CLOCKS_PER_SEC;
    float timeElapsed = endTime - origin;
    printf("[%f seconds]\n", timeElapsed);


    printf("Server Preprocessing: ");
    float startTime = (float)clock()/CLOCKS_PER_SEC;

    // Server creates filter
    u64*** filters = (u64***) malloc(sizeof(u64**)*data.out_chans);
    for (int out_c = 0; out_c < data.out_chans; out_c++) {
        filters[out_c] = (u64**) malloc(sizeof(u64*)*data.inp_chans);
        for (int inp_c = 0; inp_c < data.inp_chans; inp_c++) {
            filters[out_c][inp_c] = (u64*) malloc(sizeof(u64)*data.filter_size);
            for (int idx = 0; idx < data.filter_size; idx++)
                filters[out_c][inp_c][idx] = 1;
        }
    }

    // Server creates additive secret share
    uint64_t** linear_share = (uint64_t**) malloc(sizeof(uint64_t*)*data.out_chans);
    uint64_t** linear_mac_share = (uint64_t**) malloc(sizeof(uint64_t*)*data.out_chans);
    uint64_t** r_mac_share = (uint64_t**) malloc(sizeof(uint64_t*)*data.inp_chans);
    
    for (int chan = 0; chan < data.inp_chans; chan++) {
        r_mac_share[chan] = (uint64_t*) malloc(sizeof(uint64_t)*data.image_h*data.image_w);
        for (int idx = 0; idx < data.image_h*data.image_w; idx++)
            // TODO: Adjust these for testing
            r_mac_share[chan][idx] = 5;
    }

    for (int chan = 0; chan < data.out_chans; chan++) {
        linear_share[chan] = (uint64_t*) malloc(sizeof(uint64_t)*data.output_h*data.output_w);
        linear_mac_share[chan] = (uint64_t*) malloc(sizeof(uint64_t)*data.output_h*data.output_w);
        for (int idx = 0; idx < data.output_h*data.output_w; idx++) {
            // TODO: Adjust these for testing
            linear_share[chan][idx] = 4;
            linear_mac_share[chan][idx] = 2;
        }
    }
    
    uint64_t mac_key_a = 4;
    uint64_t mac_key_b = 2;

    char**** masks = server_conv_preprocess(sfhe, &data, filters); 
    ServerShares server_shares = server_conv_preprocess_shares(sfhe, &data, linear_share,
        linear_mac_share, r_mac_share, mac_key_a, mac_key_b);

    endTime = (float)clock()/CLOCKS_PER_SEC;
    timeElapsed = endTime - startTime;
    printf("[%f seconds]\n", timeElapsed);

    printf("Convolution: ");
    startTime = (float)clock()/CLOCKS_PER_SEC;

    server_conv_online(sfhe, &data, client_shares.input_ct, masks, &server_shares);

    endTime = (float)clock()/CLOCKS_PER_SEC;
    timeElapsed = endTime - startTime;
    printf("[%f seconds]\n", timeElapsed);

    printf("Post process: ");
    startTime = (float)clock()/CLOCKS_PER_SEC;

    // This simulates the client receiving the ciphertexts 
    client_shares.linear_ct.inner = (char*) malloc(sizeof(char)*server_shares.linear_ct.size);
    client_shares.linear_mac_ct.inner = (char*) malloc(sizeof(char)*server_shares.linear_mac_ct.size);
    client_shares.r_mac_ct.inner = (char*) malloc(sizeof(char)*server_shares.r_mac_ct.size);
    client_shares.linear_ct.size = server_shares.linear_ct.size;
    client_shares.linear_mac_ct.size = server_shares.linear_mac_ct.size;
    client_shares.r_mac_ct.size = server_shares.r_mac_ct.size;
    memcpy(client_shares.linear_ct.inner, server_shares.linear_ct.inner, server_shares.linear_ct.size);
    memcpy(client_shares.linear_mac_ct.inner, server_shares.linear_mac_ct.inner, server_shares.linear_mac_ct.size);
    memcpy(client_shares.r_mac_ct.inner, server_shares.r_mac_ct.inner, server_shares.r_mac_ct.size);
    client_conv_decrypt(cfhe, &data, &client_shares);

    endTime = (float)clock()/CLOCKS_PER_SEC;
    timeElapsed = endTime - startTime;
    printf("[%f seconds]\n", timeElapsed);

    timeElapsed = endTime - origin;
    printf("Total [%f seconds]\n\n", timeElapsed);

    printf("RESULT: \n");
    for (int chan = 0; chan < data.out_chans; chan++) {
        int idx = 0;
        for (int row = 0; row < data.output_h; row++) {
            printf(" [");
            int col = 0;
            for (; col < data.output_w-1; col++) {
                printf("%d, " , client_shares.linear[chan][row*data.output_w + col]);
            }
            printf("%d ]\n" , client_shares.linear[chan][row*data.output_w + col]);
        }
        printf("\n");
    }

    printf("MAC: \n");
    for (int chan = 0; chan < data.out_chans; chan++) {
        int idx = 0;
        for (int row = 0; row < data.output_h; row++) {
            printf(" [");
            int col = 0;
            for (; col < data.output_w-1; col++) {
                printf("%d, " , client_shares.linear_mac[chan][row*data.output_w + col]);
            }
            printf("%d ]\n" , client_shares.linear_mac[chan][row*data.output_w + col]);
        }
        printf("\n");
    }

    printf("R: \n");
    for (int chan = 0; chan < data.inp_chans; chan++) {
        int idx = 0;
        for (int row = 0; row < data.image_h; row++) {
            printf(" [");
            int col = 0;
            for (; col < data.image_w-1; col++) {
                printf("%d, " , client_shares.r_mac[chan][row*data.output_w + col]);
            }
            printf("%d ]\n" , client_shares.r_mac[chan][row*data.output_w + col]);
        }
        printf("\n");
    }

    // Free filters
    for (int out_c = 0; out_c < data.out_chans; out_c++) {
        for (int inp_c = 0; inp_c < data.inp_chans; inp_c++)
            free(filters[out_c][inp_c]);
      free(filters[out_c]);
    }
    free(filters);

    // Free image
    for (int chan = 0; chan < data.inp_chans; chan++) {
        free(input[chan]);
    }
    free(input);

    // Free secret shares
    for (int chan = 0; chan < data.out_chans; chan++) {
        free(linear_share[chan]);
        free(linear_mac_share[chan]);
    }
    free(linear_share);
    free(linear_mac_share);
    
    for (int chan = 0; chan < data.inp_chans; chan++) {
        free(r_mac_share[chan]);
    }
    free(r_mac_share);

    // Free C++ allocations
    free(client_shares.linear_ct.inner);
    free(client_shares.linear_mac_ct.inner);
    free(client_shares.r_mac_ct.inner);
    client_conv_free(&data, &client_shares);
    server_conv_free(&data, masks, &server_shares);
}

void fc(ClientFHE* cfhe, ServerFHE* sfhe, int vector_len, int matrix_h) {
    Metadata data = fc_metadata(cfhe->encoder, vector_len, matrix_h);
   
    printf("\nClient Preprocessing: ");
    float origin = (float)clock()/CLOCKS_PER_SEC;

    u64* input = (u64*) malloc(sizeof(u64)*vector_len);
    for (int idx = 0; idx < vector_len; idx++)
        input[idx] = 1;

    ClientShares client_shares = client_fc_preprocess(cfhe, &data, input);

    float endTime = (float)clock()/CLOCKS_PER_SEC;
    float timeElapsed = endTime - origin;
    printf("[%f seconds]\n", timeElapsed);

    printf("Server Preprocessing: ");
    float startTime = (float)clock()/CLOCKS_PER_SEC;

    u64** matrix = (u64**) malloc(sizeof(u64*)*matrix_h);
    for (int ct = 0; ct < matrix_h; ct++) {
        matrix[ct] = (u64*) malloc(sizeof(u64)*vector_len);
        for (int idx = 0; idx < vector_len; idx++)
            matrix[ct][idx] = ct*vector_len + idx;
    }

    uint64_t* linear_share = (uint64_t*) malloc(sizeof(uint64_t)*matrix_h);
    uint64_t* linear_mac_share = (uint64_t*) malloc(sizeof(uint64_t)*matrix_h);
    uint64_t* r_mac_share = (uint64_t*) malloc(sizeof(uint64_t)*vector_len);
    for (int idx = 0; idx < matrix_h; idx++) {
        linear_share[idx] = 0;
        linear_mac_share[idx] = 0;
    }
    for (int idx = 0; idx < vector_len; idx++) {
        r_mac_share[idx] = 0;
    }

    uint64_t mac_key_a = 4;
    uint64_t mac_key_b = 2;

    char** enc_matrix = server_fc_preprocess(sfhe, &data, matrix); 
    ServerShares server_shares = server_fc_preprocess_shares(sfhe, &data, linear_share,
        linear_mac_share, r_mac_share, mac_key_a, mac_key_b);
    
    endTime = (float)clock()/CLOCKS_PER_SEC;
    timeElapsed = endTime - startTime;
    printf("[%f seconds]\n", timeElapsed);

    printf("Layer: ");
    startTime = (float)clock()/CLOCKS_PER_SEC;

    server_fc_online(sfhe, &data, client_shares.input_ct, enc_matrix, &server_shares);

    endTime = (float)clock()/CLOCKS_PER_SEC;
    timeElapsed = endTime - startTime;
    printf("[%f seconds]\n", timeElapsed);

    // This simulates the client receiving the ciphertexts 
    client_shares.linear_ct.inner = (char*) malloc(sizeof(char)*server_shares.linear_ct.size);
    client_shares.linear_mac_ct.inner = (char*) malloc(sizeof(char)*server_shares.linear_mac_ct.size);
    client_shares.r_mac_ct.inner = (char*) malloc(sizeof(char)*server_shares.r_mac_ct.size);
    client_shares.linear_ct.size = server_shares.linear_ct.size;
    client_shares.linear_mac_ct.size = server_shares.linear_mac_ct.size;
    client_shares.r_mac_ct.size = server_shares.r_mac_ct.size;
    memcpy(client_shares.linear_ct.inner, server_shares.linear_ct.inner, server_shares.linear_ct.size);
    memcpy(client_shares.linear_mac_ct.inner, server_shares.linear_mac_ct.inner, server_shares.linear_mac_ct.size);
    memcpy(client_shares.r_mac_ct.inner, server_shares.r_mac_ct.inner, server_shares.r_mac_ct.size);

    printf("Post process: ");
    startTime = (float)clock()/CLOCKS_PER_SEC;

    client_fc_decrypt(cfhe, &data, &client_shares);

    endTime = (float)clock()/CLOCKS_PER_SEC;
    timeElapsed = endTime - startTime;
    printf("[%f seconds]\n", timeElapsed);

    timeElapsed = endTime - origin;
    printf("Total [%f seconds]\n\n", timeElapsed);


    printf("Matrix: [\n");
    for (int i = 0; i < matrix_h; i++) {
        for (int j = 0; j < vector_len; j++)
            printf("%d, " , matrix[i][j]);
        printf("\n");
    }
    printf("] \n");

    printf("Input: [");
    for (int j = 0; j < vector_len; j++)
        printf("%d, " , input[j]);
    printf("] \n");

    printf("Linear: [");
    for (int idx = 0; idx < matrix_h; idx++) {
        printf("%d, " , client_shares.linear[0][idx]);
    }
    printf("] \n");

    printf("Linear MAC: [");
    for (int idx = 0; idx < matrix_h; idx++) {
        printf("%d, " , client_shares.linear_mac[0][idx]);
    }
    printf("] \n"); 

    printf("R MAC: [");
    for (int idx = 0; idx < vector_len; idx++) {
        printf("%d, " , client_shares.r_mac[0][idx]);
    }
    printf("] \n");

    // Free matrix
    for (int row = 0; row < matrix_h; row++)
        free(matrix[row]);
    free(matrix);

    // Free vector
    free(input);

    // Free secret shares
    free(client_shares.linear_ct.inner);
    free(client_shares.linear_mac_ct.inner);
    free(client_shares.r_mac_ct.inner);
    free(linear_share);
    free(linear_mac_share);
    free(r_mac_share);

    // Free C++ allocations
    client_fc_free(&client_shares);
    server_fc_free(&data, enc_matrix, &server_shares);
}

u64 reduce(unsigned __int128 x) {
    return x % PLAINTEXT_MODULUS;
}

u64 mul(u64 x, u64 y) {
    return (u64)(((unsigned __int128)x * y) % PLAINTEXT_MODULUS);
}


void beavers_triples(ClientFHE* cfhe, ServerFHE* sfhe, int num_triples) {
    printf("\nPreprocessing: ");
    float origin = (float)clock()/CLOCKS_PER_SEC;
    
    u64 modulus = PLAINTEXT_MODULUS;

    // Generate and encrypt client's shares of a and b
    u64* client_a = (u64*) malloc(sizeof(u64)*num_triples);
    u64* client_b = (u64*) malloc(sizeof(u64)*num_triples);
    for (int idx = 0; idx < num_triples; idx++) {
        // Only 20 bits so we don't need 128 bit multiplication
        client_a[idx] = rand() % modulus;
        client_b[idx] = rand() % modulus;
    }
    
    ClientTriples client_shares = client_triples_preprocess(cfhe, num_triples, client_a, client_b);
    
    // Generate and encrypt server's shares of a, b, and c
    u64* server_a_rand = (u64*) malloc(sizeof(u64)*num_triples);
    u64* server_b_rand = (u64*) malloc(sizeof(u64)*num_triples);
    u64* server_c_rand = (u64*) malloc(sizeof(u64)*num_triples);
    u64* server_a = (u64*) malloc(sizeof(u64)*num_triples);
    u64* server_b = (u64*) malloc(sizeof(u64)*num_triples);
    u64* server_c = (u64*) malloc(sizeof(u64)*num_triples);
    u64* server_a_mac = (u64*) malloc(sizeof(u64)*num_triples);
    u64* server_b_mac = (u64*) malloc(sizeof(u64)*num_triples);
    u64* server_c_mac = (u64*) malloc(sizeof(u64)*num_triples);
    for (int idx = 0; idx < num_triples; idx++) {
        server_a_rand[idx] = rand() % modulus;
        server_b_rand[idx] = rand() % modulus;
        server_c_rand[idx] = mul(server_a_rand[idx], server_b_rand[idx]);
        server_a[idx] = rand() % modulus;
        server_b[idx] = rand() % modulus;
        server_c[idx] = rand() % modulus;
        server_a_mac[idx] = rand() % modulus;
        server_b_mac[idx] = rand() % modulus;
        server_c_mac[idx] = rand() % modulus;
    }

    uint64_t mac_key = rand() % modulus;
    
    ServerTriples server_shares = server_triples_preprocess(sfhe, num_triples, server_a_rand, server_b_rand,
        server_c_rand, server_a, server_b, server_c, server_a_mac, server_b_mac, server_c_mac, mac_key);

    float endTime = (float)clock()/CLOCKS_PER_SEC;
    float timeElapsed = endTime - origin;
    printf("[%f seconds]\n", timeElapsed);

    printf("Online: ");
    float startTime = (float)clock()/CLOCKS_PER_SEC;

    server_triples_online(sfhe, client_shares.a_ct, client_shares.b_ct, &server_shares);
    
    endTime = (float)clock()/CLOCKS_PER_SEC;
    timeElapsed = endTime - startTime;
    printf("[%f seconds]\n", timeElapsed);

    printf("Post process: ");
    startTime = (float)clock()/CLOCKS_PER_SEC;
    client_triples_decrypt(cfhe, server_shares.a_ct, server_shares.b_ct, server_shares.c_ct,
            server_shares.a_mac_ct, server_shares.b_mac_ct, server_shares.c_mac_ct, &client_shares);

    endTime = (float)clock()/CLOCKS_PER_SEC;
    timeElapsed = endTime - startTime;
    printf("[%f seconds]\n", timeElapsed);

    bool correct = true;
    printf("\nMAC  : %u", mac_key);
    printf("\nA    : [");
    for (int i = 0; i < num_triples; i++) {
        printf("%llu, " , reduce(client_shares.a_share[i] + server_a[i]));
    }
    printf("] \n");

    printf("B    : [");
    for (int i = 0; i < num_triples; i++) {
        printf("%llu, " , reduce(client_shares.b_share[i] + server_b[i]));
    }
    printf("] \n");

    printf("C    : [");
    for (int i = 0; i < num_triples; i++) {
        u64 a = client_shares.a_share[i] + server_a[i];
        u64 b = client_shares.b_share[i] + server_b[i];
        u64 c = reduce(client_shares.c_share[i] + server_c[i]);
        printf("%llu, " , c);
        correct &= (c == mul(a,b));
    }
    printf("] \n");

    printf("A MAC: [");
    for (int i = 0; i < num_triples; i++) {
        u64 a = client_shares.a_share[i] + server_a[i];
        u64 a_mac = reduce(client_shares.a_mac_share[i] + server_a_mac[i]);
        printf("%llu, " , a_mac);
        correct &= (a_mac == mul(a, mac_key));
    }
    printf("] \n");

    printf("B MAC: [");
    for (int i = 0; i < num_triples; i++) {
        u64 b = client_shares.b_share[i] + server_b[i];
        u64 b_mac = reduce(client_shares.b_mac_share[i] + server_b_mac[i]);
        printf("%llu, " , b_mac);
        correct &= (b_mac == mul(b, mac_key));
    }
    printf("] \n");

    printf("C MAC: [");
    for (int i = 0; i < num_triples; i++) {
        u64 c = client_shares.c_share[i] + server_c[i];
        u64 c_mac = reduce(client_shares.c_mac_share[i] + server_c_mac[i]);
        printf("%llu, " , c_mac);
        correct &= (c_mac == mul(c, mac_key));
    }
    printf("]");

    printf("\nCorrect: %d\n", correct);

    timeElapsed = endTime - origin;
    printf("Total [%f seconds]\n\n", timeElapsed);

    free(client_a);
    free(client_b);
    free(server_a_rand);
    free(server_b_rand);
    free(server_c_rand);
    free(server_a);
    free(server_b);
    free(server_c);
    free(server_a_mac);
    free(server_b_mac);
    free(server_c_mac);
    free_ct(&client_shares.a_ct);
    free_ct(&client_shares.b_ct);
    client_triples_free(&client_shares);
    server_triples_free(&server_shares);
}

void rand_gen(ClientFHE* cfhe, ServerFHE* sfhe, int num_rand) {
    printf("\nPreprocessing: ");
    float origin = (float)clock()/CLOCKS_PER_SEC;
    
    u64 modulus = PLAINTEXT_MODULUS;

    // Generate and encrypt client's shares of a and b
    u64* client_r = (u64*) malloc(sizeof(u64)*num_rand);
    for (int idx = 0; idx < num_rand; idx++) {
        client_r[idx] = rand() % modulus;
    }
    
    ClientTriples client_shares = client_rand_preprocess(cfhe, num_rand, client_r);
    
    // Generate and encrypt server's shares of a, b, and c
    u64* server_rand = (u64*) malloc(sizeof(u64)*num_rand);
    u64* server_share = (u64*) malloc(sizeof(u64)*num_rand);
    u64* server_mac = (u64*) malloc(sizeof(u64)*num_rand);
    for (int idx = 0; idx < num_rand; idx++) {
        server_rand[idx] = rand() % modulus;
        server_share[idx] = rand() % modulus;
        server_mac[idx] = rand() % modulus;
    }

    uint64_t mac_key = rand() % modulus;
    
    ServerTriples server_shares = server_rand_preprocess(sfhe, num_rand, server_rand,
        server_share, server_mac, mac_key);

    float endTime = (float)clock()/CLOCKS_PER_SEC;
    float timeElapsed = endTime - origin;
    printf("[%f seconds]\n", timeElapsed);

    printf("Online: ");
    float startTime = (float)clock()/CLOCKS_PER_SEC;

    server_rand_online(sfhe, client_shares.a_ct, &server_shares);
    
    endTime = (float)clock()/CLOCKS_PER_SEC;
    timeElapsed = endTime - startTime;
    printf("[%f seconds]\n", timeElapsed);

    printf("Post process: ");
    startTime = (float)clock()/CLOCKS_PER_SEC;
    client_rand_decrypt(cfhe, server_shares.a_ct, server_shares.a_mac_ct, &client_shares);
    
    endTime = (float)clock()/CLOCKS_PER_SEC;
    timeElapsed = endTime - startTime;
    printf("[%f seconds]\n", timeElapsed);

    bool correct = true;
    printf("\nMAC  : %llu", mac_key);
    
    printf("\nClient R: [");
    for (int i = 0; i < num_rand; i++) {
        printf("%llu, " , client_r[i]);
    }
    printf("] \n");

    printf("Server R: [");
    for (int i = 0; i < num_rand; i++) {
        printf("%llu, " , server_rand[i]);
    }
    printf("] \n");

    printf("R: [");
    for (int i = 0; i < num_rand; i++) {
        u64 r = reduce(client_shares.a_share[i] + server_share[i]);
        printf("%llu, " , r);
        correct &= (r == (client_r[i] + server_rand[i]));
    }
    printf("] \n");

    printf("R MAC: [");
    for (int i = 0; i < num_rand; i++) {
        u64 r = reduce(client_shares.a_share[i] + server_share[i]);
        u64 r_mac = reduce(client_shares.a_mac_share[i] + server_mac[i]);
        printf("%llu, " , r_mac);
        correct &= (r_mac == mul(r, mac_key));
    }
    printf("] \n");
    
    printf("\nCorrect: %d\n", correct);

    timeElapsed = endTime - origin;
    printf("Total [%f seconds]\n\n", timeElapsed);

    free(client_r);
    free(server_rand);
    free(server_share);
    free(server_mac);
    free_ct(&client_shares.a_ct);
    client_rand_free(&client_shares);
    server_rand_free(&server_shares);
}


int main(int argc, char* argv[]) {
  SerialCT key_share;

  printf("Client Keygen: ");
  float startTime = (float)clock()/CLOCKS_PER_SEC;

  ClientFHE cfhe = client_keygen(&key_share);

  float endTime = (float)clock()/CLOCKS_PER_SEC;
  float timeElapsed = endTime - startTime;
  printf("[%f seconds]\n", timeElapsed);
 
  printf("Server Keygen: ");
  startTime = (float)clock()/CLOCKS_PER_SEC;
  
  ServerFHE sfhe = server_keygen(key_share); 

  endTime = (float)clock()/CLOCKS_PER_SEC;
  timeElapsed = endTime - startTime;
  printf("[%f seconds]\n", timeElapsed);

  //conv(&cfhe, &sfhe, 5, 5, 3, 3, 2, 2, 1, 0);
  //conv(&cfhe, &sfhe, 32, 32, 3, 3, 16, 16, 1, 0);
  //conv(&cfhe, &sfhe, 16, 16, 3, 3, 32, 32, 1, 1);
  //conv(&cfhe, &sfhe, 8, 8, 3, 3, 64, 64, 1, 1);
  
  //fc(&cfhe, &sfhe, 25, 10);
  
  //beavers_triples(&cfhe, &sfhe, 10000);
  //rand_gen(&cfhe, &sfhe, 10000);
  
  client_free_keys(&cfhe);
  free_ct(&key_share);
  server_free_keys(&sfhe);

  return 1;
}
