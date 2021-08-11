/*
 *  C interface for rust interopt
 *
 *  Created on: August 10, 2019
 *      Author: ryanleh
 */
#ifndef interface
#define interface

#include <stdint.h>
#include <stdbool.h>

//const uint64_t PLAINTEXT_MODULUS = 2061584302081;
const uint64_t PLAINTEXT_MODULUS = 17592060215297;
const uint64_t POLY_MOD_DEGREE = 8192;    
const int numThreads = 4;


/* This is ugly but very useful. */
typedef struct Metadata {
    /* Number of plaintext slots in a ciphertext */
    int32_t slot_count;
    /* Number of plaintext slots in a half ciphertext (since ciphertexts are a 
     * two column matrix) */
    int32_t pack_num;
    /* Number of Channels that can fit in a half ciphertext */
    int32_t chans_per_half;
    /* Number of input ciphertexts for convolution (this may not match what the
     * client actually transmits) */
    int32_t inp_ct;
    /* Number of output ciphertexts */
    int32_t out_ct;
    /* Image and Filters metadata */
    int32_t image_h;
    int32_t image_w;
    int32_t image_size;
    int32_t inp_chans;
    int32_t filter_h;
    int32_t filter_w;
    int32_t filter_size;
    int32_t out_chans;
    /* How many total ciphertext halves the input and output take up */
    int32_t inp_halves;
    int32_t out_halves;
    /* How many Channels are in the last output or input half */
    int32_t out_in_last;
    int32_t inp_in_last;
    /* The modulo used when deciding which output channels to pack into a mask */
    int32_t out_mod;
    /* How many permutations of ciphertexts are needed to generate all
     * int32_termediate rotation sets */
    int32_t half_perms;
    /* Whether the last output half can fit repeats */
    bool last_repeats;
    /* The number of Channels in a single repeat. This may require padding
     * since the repeat must be a power of 2 */
    int32_t repeat_chans;
    /* The number of rotations for each ciphertext half */
    int32_t half_rots;
    /* The number of rotations for the last ciphertext half */
    int32_t last_rots;
    /* Total number of convolutions needed to generate all intermediate
     * rotations sets */
    int32_t convs;
    int32_t stride_h;
    int32_t stride_w;
    int32_t output_h;
    int32_t output_w;
    int32_t pad_t;
    int32_t pad_b;
    int32_t pad_r;
    int32_t pad_l;
} Metadata;

#ifdef __cplusplus
extern "C" {
#endif
    typedef struct ClientFHE {
        void* context;
        void* encoder;
        void* encryptor;
        void* evaluator;
        void* decryptor;
    } ClientFHE;

    typedef struct ServerFHE {
        void* context;
        void* encoder;
        void* encryptor;
        void* evaluator;
        void* gal_keys;
        void* relin_keys;
        char* zero;
    } ServerFHE;

    typedef struct SerialCT {
        char* inner;
        uint64_t size;
    } SerialCT;

    typedef struct ServerShares {
        // Opaque vectors of Plaintexts  
        char** linear;
        char** linear_mac;
        char** r_mac;
        // Opaque ciphertexts to be sent to client
        SerialCT linear_ct;
        SerialCT linear_mac_ct;
        SerialCT r_mac_ct;
        // Opaque plaintexts of MAC keys
        char* mac_key_a;
        char* mac_key_b;
    } ServerShares;

    typedef struct ClientShares {
        // Opaque ciphertext sent to server
        SerialCT input_ct;
        // Opaque ciphertexts received from server
        SerialCT linear_ct;
        SerialCT linear_mac_ct;
        SerialCT r_mac_ct;
        // Decryption results
        uint64_t** linear;
        uint64_t** linear_mac;
        uint64_t** r_mac;
    } ClientShares;

    typedef struct ServerTriples {
        // Batch size
        uint32_t num;
        uint64_t vec_len;
        // Opaque vectors of Plaintexts
        char** a_rand;
        char** b_rand;
        char** c_rand;
        char** a_share;
        char** b_share;
        char** c_share;
        char** a_mac_share;
        char** b_mac_share;
        char** c_mac_share;
        // Opaque ciphertexts sent to client
        SerialCT a_ct;
        SerialCT b_ct;
        SerialCT c_ct;
        SerialCT a_mac_ct;
        SerialCT b_mac_ct;
        SerialCT c_mac_ct;
        // Opaque Plaintext of MAC key
        char* mac_key;
    } ServerTriples;

    typedef struct ClientTriples {
        // Batch size
        uint32_t num;
        uint64_t vec_len;
        // Opaque ciphertexts sent to server
        SerialCT a_ct;
        SerialCT b_ct;
        // Decryption results
        uint64_t* a_share;
        uint64_t* b_share;
        uint64_t* c_share;
        uint64_t* a_mac_share;
        uint64_t* b_mac_share;
        uint64_t* c_mac_share;
    } ClientTriples;
    
    /* Generates new keys and helpers for Client. Returns the helpers and allocates
     * a bytestream for sharing the keys with the server */
    ClientFHE client_keygen(SerialCT *key_share);

    /* Generates keys and helpers for Server given a key_share */
    ServerFHE server_keygen(SerialCT key_share); 

    /* Populates the Metadata struct for Conv layer */
    Metadata conv_metadata(void* batch_encoder, int32_t image_h, int32_t image_w,
            int32_t filter_h, int32_t filter_w, int32_t inp_chans, int32_t out_chans, int32_t stride_h,
            int32_t stride_w, bool pad_valid);

    /* Populates the Metadata struct for FC layer */
    Metadata fc_metadata(void* batch_encoder, int32_t vector_len, int32_t matrix_h);

    /* Encrypts and serializes a vector */
    SerialCT encrypt_vec(const ClientFHE* cfhe, const uint64_t* vec, uint64_t vec_size);

    /* Deserializes and decrypts a vector */
    uint64_t* decrypt_vec(const ClientFHE* cfhe, SerialCT *ct, uint64_t size);

    /* MACs the ciphertext and shares it */
    SerialCT server_mac_ct(const ServerFHE* sfhe, SerialCT *ct, const uint64_t* mac_share, uint64_t size, uint64_t mac_key);

    SerialCT client_input_auth(const ServerFHE* sfhe, SerialCT *ct, const uint64_t* shares, const uint64_t* rand, uint64_t size);
   
    /* Preprocesses and serializes input image */
    ClientShares client_conv_preprocess(const ClientFHE* cfhe, const Metadata* data, const uint64_t* const* image);
    
    /* Preprocesses convolution filters */
    char**** server_conv_preprocess(const ServerFHE* sfhe, const Metadata* data, const uint64_t* const* const* filters);

    /* Converts secret shares to opaque SEAL Plaintexts for conv computation */        
    ServerShares server_conv_preprocess_shares(const ServerFHE* sfhe, const Metadata* data,
            const uint64_t* const* linear_share, const uint64_t* const* linear_mac_share,
            const uint64_t* const* r_mac_share, const uint64_t mac_key_a, const uint64_t mac_key_b);

    /* Preprocess and serialize client input vector */
    ClientShares client_fc_preprocess(const ClientFHE* cfhe, const Metadata* data, const uint64_t* vector);

    /* Preprocess server matrix */
    char** server_fc_preprocess(const ServerFHE* sfhe, const Metadata* data, const uint64_t* const* matrix);

    /* Converts secret shares to opaque SEAL Plaintexts for fc computation */
    ServerShares server_fc_preprocess_shares(const ServerFHE* sfhe, const Metadata* data,
        const uint64_t* linear_share, const uint64_t* linear_mac_share,
        const uint64_t* r_mac_share, const uint64_t mac_key_a, const uint64_t mac_key_b);

    /* Converts secret shares to opaque SEAL Plaintexts for multiplication triples */
    ServerTriples server_triples_preprocess(const ServerFHE* sfhe, uint32_t num_triples, const uint64_t* a_rand,
    const uint64_t* b_rand, const uint64_t* c_rand, const uint64_t* a_share, const uint64_t* b_share,
    const uint64_t* c_share, const uint64_t* a_mac_share, const uint64_t* b_mac_share, const uint64_t* c_mac_share,
    const uint64_t mac_key);

    /* Converts secret shares to opaque SEAL Plaintexts for pairwise randomness */
    ServerTriples server_rand_preprocess(const ServerFHE* sfhe, uint32_t num_rand, const uint64_t* rand,
    const uint64_t* share, const uint64_t* mac_share, const uint64_t mac_key);

    /* Encrypts client inputs for multiplication triples */
    ClientTriples client_triples_preprocess(const ClientFHE* cfhe, uint32_t num_triples, const uint64_t* a_rand,
        const uint64_t* b_rand);

    /* Encrypts client inputs for pairwise randomness */
    ClientTriples client_rand_preprocess(const ClientFHE* cfhe, uint32_t num_rand, const uint64_t* rand);

    /* Computes the masking noise ciphertext for conv output */
    char* fc_preprocess_noise(const ServerFHE* sfhe, const Metadata* data, const uint64_t* secret_share);

    /* Performs convolution on the given input and stores the result in the shares struct */
    void server_conv_online(const ServerFHE* sfhe, const Metadata* data, SerialCT ciphertext,
        char**** masks, ServerShares* shares);
    
    /* Performs matrix multiplication on the given inputs */
    void server_fc_online(const ServerFHE* sfhe, const Metadata* data, SerialCT ciphertext, 
            char** matrix, ServerShares* shares);

    /* Computes the encrypted client's share of multiplication triple */
    void server_triples_online(const ServerFHE* sfhe, SerialCT client_a, SerialCT client_b, ServerTriples* shares);

    /* Computes the encrypted client's share of pairwise randomness */
    void server_rand_online(const ServerFHE* sfhe, SerialCT client_r, ServerTriples* shares);

    /* Decrypt and reshape convolution results */
    void client_conv_decrypt(const ClientFHE *cfhe, const Metadata *data, ClientShares* shares);
    
    /* Decrypts and reshapes fully-connected result */
    void client_fc_decrypt(const ClientFHE *cfhe, const Metadata *data, ClientShares *shares);

    /* Decrypts the clients multiplication triple share */
    void client_triples_decrypt(const ClientFHE *cfhe, SerialCT a, SerialCT b, SerialCT c, SerialCT a_mac,
        SerialCT b_mac, SerialCT c_mac, ClientTriples *shares);

    /* Decrypts the clients pairwise randomness share */
    void client_rand_decrypt(const ClientFHE *cfhe, SerialCT share, SerialCT mac_share, ClientTriples *shares);

    /* Free client's allocated keys */
    void client_free_keys(const ClientFHE* cfhe); 
 
    /* Free server's allocated keys */
    void server_free_keys(const ServerFHE *sfhe);
    
    /* Free a serialized ciphertext */
    void free_ct(SerialCT* ct); 

    /* Free the client's state required for a convolution */
    void client_conv_free(const Metadata* data, ClientShares* shares);
   
    /* Free the server's state required for a convolution */
    void server_conv_free(const Metadata* data, char**** masks, ServerShares* shares);
    
    /* Free the client's state required for an fc layer */
    void client_fc_free(ClientShares* shares);

    /* Free the server's state required for an fc layer */
    void server_fc_free(const Metadata* data, char** enc_matrix, ServerShares* shares);
    
    /* Free the client's state required for generating triples */
    void client_triples_free(ClientTriples* shares);

    /* Free the server's state required for generating triples */
    void server_triples_free(ServerTriples* shares);

    /* Free the client's state required for generating pairwise randomness */
    void client_rand_free(ClientTriples* shares);

    /* Free the server's state required for generating pairwise randomness */
    void server_rand_free(ServerTriples* shares);

#ifdef __cplusplus
}
#endif

#endif
