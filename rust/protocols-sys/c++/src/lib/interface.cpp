/*
 *  DelphiOffline's C interface 
 *
 *  Created on: June 10, 2019
 *      Author: ryanleh
 */
#include <iomanip>
#include <math.h>
#include "interface.h"
#include "conv2d.h"
#include "fc_layer.h"
#include "triples_gen.h"

/* Formatted printed for batched plaintext */
inline void print_batch(size_t slot_count, size_t print_size, uv64 &plain) {
    size_t row_size = slot_count / 2;
    cout << endl;

    cout << "    [";
    for (size_t i = 0; i < print_size; i++)
    {
        cout << setw(3) << plain[i] << ",";
    }
    cout << setw(3) << " ...,";
    for (size_t i = row_size - print_size; i < row_size; i++)
    {
        cout << setw(3) << plain[i] << ((i != row_size - 1) ? "," : " ]\n");
    }
    cout << "    [";
    for (size_t i = row_size; i < row_size + print_size; i++)
    {
        cout << setw(3) << plain[i] << ",";
    }
    cout << setw(3) << " ...,";
    for (size_t i = 2 * row_size - print_size; i < 2 * row_size; i++)
    {
        cout << setw(3) << plain[i] << ((i != 2 * row_size - 1) ? "," : " ]\n");
    }
    cout << endl;
}

/* Serializes an outputstream into a byte array.
 * Returns the bytearray and size */
SerialCT serialize(ostringstream &os) {
    char* serialized = new char[os.tellp()];
    string tmp_str = os.str();
    std::move(tmp_str.begin(), tmp_str.end(), serialized);
    return SerialCT { serialized, (uint64_t) os.tellp() };
}

/* Serializes a SEAL ciphertext to a byte array. 
 * returns the bytearray and size */
SerialCT serialize_ct(vector<Ciphertext> ct_vec) {
    ostringstream os;
    for (auto &ct: ct_vec)
        ct.save(os);
    return serialize(os);
}

/* Extracts a vector of Ciphertexts from provided byte stream */
void recast_opaque(SerialCT &ct, vector<Ciphertext>& destination,
        SEALContext* context) {
    istringstream is;
    is.rdbuf()->pubsetbuf(ct.inner, ct.size);
    for(int ct_idx = 0; ct_idx < destination.size(); ct_idx++) {
        destination[ct_idx].load(*context, is);
    }
}

/* Encodes a vector of u64 into SEAL Plaintext */
vector<Plaintext> encode_vec(const u64* shares, u64 num, BatchEncoder& encoder) {
    u64 slot_count = encoder.slot_count();
    int vec_size = ceil((float)num / slot_count);
    vector<Plaintext> result(vec_size);

#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int vec_idx = 0; vec_idx < vec_size; vec_idx++) {
        uv64 pod_matrix(slot_count, 0ULL);
        int limit = min(num-vec_idx*slot_count, slot_count);
        for (int i = 0; i < limit; i++) {
            pod_matrix[i] = shares[vec_idx*slot_count + i];
        }
        encoder.encode(pod_matrix, result[vec_idx]);
    }
    return result;
}

/* Encrypts and serializes a vector */
SerialCT encrypt_vec(const ClientFHE* cfhe, const u64* vec, u64 vec_size) {
    // Recast the needed fhe helpers
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
   
    // Encrypt vec
    auto pt_vec = encode_vec(vec, vec_size, *encoder);
    vector<Ciphertext> ct_vec(pt_vec.size());
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < pt_vec.size(); i++) {
        encryptor->encrypt(pt_vec[i], ct_vec[i]);
    }

    // Serialize ciphertexts
    SerialCT ct = serialize_ct(ct_vec);
    return ct;
}

/* Deserializes and decrypts a vector */
u64* decrypt_vec(const ClientFHE* cfhe, SerialCT *ct, u64 size) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(cfhe->context);

    // Recast needed fhe helpers
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);

    // Recast bytearrays to vectors of Ciphertexts and decrypt
    u64 slot_count = encoder->slot_count();
    u64 vec_size = ceil((float)size / slot_count);
    vector<Ciphertext> ct_vec(vec_size);
    recast_opaque(*ct, ct_vec, context);
    
    // Decrypt ciphertext 
    u64* share = new u64[size];
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < vec_size; i++) {
        vector<u64> pod_matrix(slot_count, 0ULL);
        Plaintext tmp;
        decryptor->decrypt(ct_vec[i], tmp);
        encoder->decode(tmp, pod_matrix);
        for (int j = 0; j < min(slot_count, size - slot_count*i); j++) {
            share[slot_count*i + j] = pod_matrix[j];
        }
    }
    return share;
}

/* MACs a ciphertext and shares it */
SerialCT server_mac_ct(const ServerFHE* sfhe, SerialCT *ct, const u64* mac_share, u64 size, u64 mac_key) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(sfhe->context);

    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);

    // Create Plaintext objects
    vector<u64> pod_matrix(encoder->slot_count(), mac_key);
    Plaintext mac_plain;
    encoder->encode(pod_matrix, mac_plain);
    auto shares = encode_vec(mac_share, size, *encoder);

    // Deserialize ciphertexts
    vector<Ciphertext> ct_vec(shares.size());
    recast_opaque(*ct, ct_vec, context);

    // MAC and randomize each ciphertext
    for (int i = 0; i < ct_vec.size(); i++) {
        evaluator->multiply_plain_inplace(ct_vec[i], mac_plain);
        evaluator->sub_plain_inplace(ct_vec[i], shares[i]);
    }
    
    // Serialize result
    SerialCT result = serialize_ct(ct_vec);
    return result;
}

SerialCT client_input_auth(const ServerFHE* sfhe, SerialCT *ct, const uint64_t* shares, const uint64_t* rand, uint64_t size) {
  // Grab shared pointer to SEALContext
  auto context = static_cast<SEALContext*>(sfhe->context);

  // Recast the needed fhe helpers
  BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
  Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);

  // Create Plaintext objects
  auto shares_pt = encode_vec(shares, size, *encoder);
  auto rand_pt = encode_vec(rand, size, *encoder);

  // Deserialize ciphertexts
  vector<Ciphertext> ct_vec(1);
  recast_opaque(*ct, ct_vec, context);

  vector<Ciphertext> result(1);
  // MAC and randomize each ciphertext
  evaluator->multiply_plain(ct_vec[0], shares_pt[0], result[0]);
  evaluator->sub_plain_inplace(result[0], rand_pt[0]);
  
  // Serialize result
  SerialCT result_ct = serialize_ct(result);
  return result_ct;
}

ClientFHE client_keygen(SerialCT *key_share) {
    //---------------Param and Key Generation---------------
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(POLY_MOD_DEGREE);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(POLY_MOD_DEGREE));
    parms.set_plain_modulus(PLAINTEXT_MODULUS);
    auto context = new SEALContext(parms);
    KeyGenerator keygen(*context);
    auto sec_key = keygen.secret_key();
    // Get serialized versions of the keys
    auto ser_pub_key = keygen.create_public_key();
    auto ser_gal_keys = keygen.create_galois_keys();
    auto ser_relin_keys = keygen.create_relin_keys();
    // Deserialize the public key since we use it when creating the local
    // objects
    PublicKey pub_key;
    stringstream pub_key_s;
    ser_pub_key.save(pub_key_s);
    pub_key.load(*context, pub_key_s);

    BatchEncoder *encoder = new BatchEncoder(*context);
    Encryptor *encryptor = new Encryptor(*context, pub_key);
    Evaluator *evaluator = new Evaluator(*context);
    Decryptor *decryptor = new Decryptor(*context, sec_key); 

    // Recast the context to void*
    void* void_context = static_cast<void*>(context);

    // Serialize params and all keys
    ostringstream os;
    parms.save(os);
    ser_pub_key.save(os);
    ser_gal_keys.save(os);
    ser_relin_keys.save(os);
    *key_share = serialize(os);
    return ClientFHE { void_context, encoder, encryptor, evaluator, decryptor };
}


ServerFHE server_keygen(SerialCT key_share) {
    istringstream is;
    is.rdbuf()->pubsetbuf(key_share.inner, key_share.size);

    // Load params
    EncryptionParameters parms;
    parms.load(is);
    auto context = new SEALContext(parms);

    // Load keys
    PublicKey pub_key;
    GaloisKeys* gal_keys = new GaloisKeys();
    RelinKeys* relin_keys = new RelinKeys();
    pub_key.load(*context, is);
    (*gal_keys).load(*context, is);
    (*relin_keys).load(*context, is);

    // Create helpers
    BatchEncoder *encoder = new BatchEncoder(*context);
    Encryptor *encryptor = new Encryptor(*context, pub_key);
    Evaluator *evaluator = new Evaluator(*context);
    
    // Recast the context to void*
    void* void_context = static_cast<void*>(context);
    
    // Generate the zero ciphertext
    vector<u64> pod_matrix(encoder->slot_count(), 0ULL);
    Plaintext tmp;
    Ciphertext* zero = new Ciphertext();
    encoder->encode(pod_matrix, tmp);
    encryptor->encrypt(tmp, *zero);

    return ServerFHE { void_context, encoder, encryptor, evaluator, gal_keys, relin_keys,
      (char*) zero };
}

Metadata conv_metadata(void* batch_encoder, int32_t image_h, int32_t image_w, int32_t filter_h, int32_t filter_w,
        int32_t inp_chans, int32_t out_chans, int32_t stride_h, int32_t stride_w, bool pad_valid) {
    int slot_count = (reinterpret_cast<BatchEncoder*>(batch_encoder))->slot_count();
    return conv_metadata(slot_count, image_h, image_w, filter_h, filter_w, inp_chans, out_chans,
        stride_h, stride_w, pad_valid);
}

Metadata fc_metadata(void* batch_encoder, int32_t vector_len, int32_t matrix_h) {
    int slot_count = (reinterpret_cast<BatchEncoder*>(batch_encoder))->slot_count();
    return fc_metadata(slot_count, vector_len, matrix_h);
}

ClientShares client_conv_preprocess(const ClientFHE* cfhe, const Metadata* data, const u64* const* image) {
    // Recast the needed fhe helpers
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    
    // Preprocess image
    auto pt = preprocess_image(*data, image);
    auto rotated_pt = filter_rotations(pt, *data);
    auto ct_rotations = HE_encrypt_rotations(rotated_pt, *data, *encryptor, *encoder);

    // Flatten rotations ciphertext
    vector<Ciphertext> ct_flat_rotations;
    for (const auto &ct: ct_rotations)
        ct_flat_rotations.insert(ct_flat_rotations.end(), ct.begin(), ct.end());

    // Serialize vector
    ClientShares shares;
    shares.input_ct = serialize_ct(ct_flat_rotations);
    return shares;
}

char**** server_conv_preprocess(const ServerFHE* sfhe, const Metadata* data,
        const u64* const* const* filters) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);

    // Preprocess filters
    auto masks_vec = HE_preprocess_filters(filters, *data, *encoder);
   
    // Recast masks to use opaque pointers for C interface
    char**** masks = new char***[masks_vec.size()];
    for (int i = 0; i < masks_vec.size(); i++) {
        masks[i] = new char**[masks_vec[0].size()];
        for (int j = 0; j < masks_vec[0].size(); j++) {
            masks[i][j] = new char*[masks_vec[0][0].size()];
            for (int z = 0; z < masks_vec[0][0].size(); z++)
                masks[i][j][z] = (char*) new Plaintext(masks_vec[i][j][z]);
        }
    }
    return masks;
}

ServerShares server_conv_preprocess_shares(const ServerFHE* sfhe, const Metadata* data,
        const u64* const* linear_share, const u64* const* linear_mac_share,
        const u64* const* r_mac_share, const u64 mac_key_a, const u64 mac_key_b) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);

    // Reshape shares
    vector<Plaintext> linear = HE_preprocess_noise(linear_share, *data, *encoder);
    vector<Plaintext> linear_mac = HE_preprocess_noise(linear_mac_share, *data, *encoder);
    vector<uv64> r_mac = preprocess_image(*data, r_mac_share);
    
    // Recast everything back to opaque C types
    auto enc_linear_share = new char*[data->out_ct];
    auto enc_linear_mac_share = new char*[data->out_ct];
    auto enc_r_mac_share = new char*[data->inp_ct];
    for (int ct_idx = 0; ct_idx < data->out_ct; ct_idx++) {
        enc_linear_share[ct_idx] = (char*) new Plaintext(linear[ct_idx]);
        enc_linear_mac_share[ct_idx] = (char*) new Plaintext(linear_mac[ct_idx]);
    }
    for (int ct_idx = 0; ct_idx < data->inp_ct; ct_idx++) {
        Plaintext tmp;
        encoder->encode(r_mac[ct_idx], tmp);
        enc_r_mac_share[ct_idx] = (char*) new Plaintext(tmp);
    }
   
    // Generate mac plaintexts
    vector<u64> mac_vec_a(encoder->slot_count(), mac_key_a);
    vector<u64> mac_vec_b(encoder->slot_count(), mac_key_b);
    Plaintext* enc_mac_a = new Plaintext();
    Plaintext* enc_mac_b = new Plaintext();
    encoder->encode(mac_vec_a, *enc_mac_a);
    encoder->encode(mac_vec_b, *enc_mac_b);

    ServerShares shares;
    shares.linear = enc_linear_share;
    shares.linear_mac = enc_linear_mac_share;
    shares.r_mac = enc_r_mac_share;
    shares.mac_key_a = (char*) enc_mac_a;
    shares.mac_key_b = (char*) enc_mac_b;
    return shares;
}


ClientShares client_fc_preprocess(const ClientFHE* cfhe, const Metadata* data, const u64* vector) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);

    // Preprocess input vector
    Plaintext enc_vec = preprocess_vec(*data, *encoder, vector);
    std::vector<Ciphertext> ct(1);
    encryptor->encrypt(enc_vec, ct[0]);

    // Convert vector to char array and flatten to a single byte array
    ClientShares shares;
    shares.input_ct = serialize_ct(ct);
    return shares;
}


char** server_fc_preprocess(const ServerFHE* sfhe, const Metadata* data, const u64* const* matrix) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);

    // Preprocess matrix
    vector<Plaintext> enc_matrix = preprocess_matrix(matrix, *data, *encoder);
   
    // Convert to opaque C types
    char** enc_matrix_c = new char*[data->inp_ct];
    for (int i = 0; i < data->inp_ct; i++) {
        enc_matrix_c[i] = (char*) new Plaintext(enc_matrix[i]);
    }
    return enc_matrix_c;
}


ServerShares server_fc_preprocess_shares(const ServerFHE* sfhe, const Metadata* data,
        const u64* linear_share, const u64* linear_mac_share,
        const u64* r_mac_share, u64 mac_key_a, u64 mac_key_b) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);

    // Reshape shares
    Plaintext linear = fc_preprocess_noise(*data, *encoder, linear_share);
    Plaintext linear_mac = fc_preprocess_noise(*data, *encoder, linear_mac_share);
    Plaintext r_mac = preprocess_vec(*data, *encoder, r_mac_share);
    
    // Recast shares to opaque pointers
    auto enc_linear_share = new char*[1];
    auto enc_linear_mac_share = new char*[1];
    auto enc_r_mac_share = new char*[1];
    enc_linear_share[0] = (char*) new Plaintext(linear);
    enc_linear_mac_share[0] = (char*) new Plaintext(linear_mac);
    enc_r_mac_share[0] = (char*) new Plaintext(r_mac);

    // Generate mac plaintexts
    vector<u64> mac_vec_a(encoder->slot_count(), mac_key_a);
    vector<u64> mac_vec_b(encoder->slot_count(), mac_key_b);
    Plaintext* enc_mac_a = new Plaintext();
    Plaintext* enc_mac_b = new Plaintext();
    encoder->encode(mac_vec_a, *enc_mac_a);
    encoder->encode(mac_vec_b, *enc_mac_b);

    ServerShares shares;
    shares.linear = enc_linear_share;
    shares.linear_mac = enc_linear_mac_share;
    shares.r_mac = enc_r_mac_share;
    shares.mac_key_a = (char*) enc_mac_a;
    shares.mac_key_b = (char*) enc_mac_b;
    return shares;
}

ServerTriples server_triples_preprocess(const ServerFHE* sfhe, uint32_t num_triples, const u64* a_rand,
    const u64* b_rand, const u64* c_rand, const u64* a_share, const u64* b_share,
    const u64* c_share, const u64* a_mac_share, const u64* b_mac_share, const u64* c_mac_share,
    const u64 mac_key) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);

    // Encode shares
    vector<Plaintext> enc_a_rand = encode_vec(a_rand, num_triples, *encoder);
    vector<Plaintext> enc_b_rand = encode_vec(b_rand, num_triples, *encoder);
    vector<Plaintext> enc_c_rand = encode_vec(c_rand, num_triples, *encoder);
    vector<Plaintext> enc_a = encode_vec(a_share, num_triples, *encoder);
    vector<Plaintext> enc_b = encode_vec(b_share, num_triples, *encoder);
    vector<Plaintext> enc_c = encode_vec(c_share, num_triples, *encoder);
    vector<Plaintext> enc_a_mac = encode_vec(a_mac_share, num_triples, *encoder);
    vector<Plaintext> enc_b_mac = encode_vec(b_mac_share, num_triples, *encoder);
    vector<Plaintext> enc_c_mac = encode_vec(c_mac_share, num_triples, *encoder);
    
    // Recast shares to opaque pointers
    u64 vec_size = enc_a.size();
    char** a_r = new char*[vec_size]; 
    char** b_r = new char*[vec_size]; 
    char** c_r = new char*[vec_size]; 
    char** a = new char*[vec_size]; 
    char** b = new char*[vec_size];
    char** c = new char*[vec_size];
    char** a_mac = new char*[vec_size];
    char** b_mac = new char*[vec_size];
    char** c_mac = new char*[vec_size];
    for (int i = 0; i < vec_size; i++) {
        a_r[i] = (char*) new Plaintext(enc_a_rand[i]);
        b_r[i] = (char*) new Plaintext(enc_b_rand[i]);
        c_r[i] = (char*) new Plaintext(enc_c_rand[i]);
        a[i] = (char*) new Plaintext(enc_a[i]);
        b[i] = (char*) new Plaintext(enc_b[i]);
        c[i] = (char*) new Plaintext(enc_c[i]);
        a_mac[i] = (char*) new Plaintext(enc_a_mac[i]);
        b_mac[i] = (char*) new Plaintext(enc_b_mac[i]);
        c_mac[i] = (char*) new Plaintext(enc_c_mac[i]);
    }

    // Generate MAC Plaintext
    vector<u64> mac_vec(encoder->slot_count(), mac_key);
    Plaintext* enc_mac = new Plaintext();
    encoder->encode(mac_vec, *enc_mac);

    ServerTriples shares;
    shares.num = num_triples;
    shares.vec_len = vec_size;
    shares.a_rand = a_r;
    shares.b_rand = b_r;
    shares.c_rand = c_r;
    shares.a_share = a;
    shares.b_share = b;
    shares.c_share = c;
    shares.a_mac_share = a_mac;
    shares.b_mac_share = b_mac;
    shares.c_mac_share = c_mac;
    shares.mac_key = (char*) enc_mac;
    return shares;
}


ClientTriples client_triples_preprocess(const ClientFHE* cfhe, uint32_t num_triples, const u64* a_rand,
        const u64* b_rand) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);

    // Encode randomizers
    vector<Plaintext> enc_a = encode_vec(a_rand, num_triples, *encoder);
    vector<Plaintext> enc_b = encode_vec(b_rand, num_triples, *encoder);

    // Encrypt randomizers
    u64 vec_size = enc_a.size();
    vector<Ciphertext> vec_a(vec_size);
    vector<Ciphertext> vec_b(vec_size);
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < vec_size; i++) {
        encryptor->encrypt(enc_a[i], vec_a[i]);
        encryptor->encrypt(enc_b[i], vec_b[i]);
    }

    // Recast to opaque pointers
    SerialCT a, b;
    a = serialize_ct(vec_a);
    b = serialize_ct(vec_b);

    ClientTriples shares;
    shares.num = num_triples;
    shares.vec_len = vec_size;
    shares.a_ct = a;
    shares.b_ct = b;
    return shares;
}


ServerTriples server_rand_preprocess(const ServerFHE* sfhe, uint32_t num_rand, const u64* rand,
    const u64* share, const u64* mac_share, const u64 mac_key) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);

    // Encode shares
    vector<Plaintext> enc_rand = encode_vec(rand, num_rand, *encoder);
    vector<Plaintext> enc_share = encode_vec(share, num_rand, *encoder);
    vector<Plaintext> enc_mac_share = encode_vec(mac_share, num_rand, *encoder);
    
    // Recast shares to opaque pointers
    u64 vec_size = enc_rand.size();
    char** r = new char*[vec_size]; 
    char** s = new char*[vec_size]; 
    char** mac_s = new char*[vec_size]; 
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < vec_size; i++) {
        r[i] = (char*) new Plaintext(enc_rand[i]);
        s[i] = (char*) new Plaintext(enc_share[i]);
        mac_s[i] = (char*) new Plaintext(enc_mac_share[i]);
    }
    
    // Generate MAC Plaintext
    vector<u64> mac_vec(encoder->slot_count(), mac_key);
    Plaintext* enc_mac = new Plaintext();
    encoder->encode(mac_vec, *enc_mac);

    ServerTriples shares;
    shares.num = num_rand;
    shares.vec_len = vec_size;
    shares.a_rand = r;
    shares.a_share = s;
    shares.a_mac_share = mac_s;
    shares.mac_key = (char*) enc_mac;
    return shares;
}


ClientTriples client_rand_preprocess(const ClientFHE* cfhe, uint32_t num_rand, const u64* rand) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);

    // Encode shares
    vector<Plaintext> enc_rand = encode_vec(rand, num_rand, *encoder);

    // Encrypt shares
    u64 vec_size = enc_rand.size();
    vector<Ciphertext> vec_rand(vec_size);
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < vec_size; i++) {
        encryptor->encrypt(enc_rand[i], vec_rand[i]);
    }

    // Recast to opaque pointers
    SerialCT r = serialize_ct(vec_rand);

    ClientTriples shares;
    shares.num = num_rand;
    shares.vec_len = vec_size;
    shares.a_ct = r;
    return shares;
}


void server_conv_online(const ServerFHE* sfhe, const Metadata* data, SerialCT ciphertext,
    char**** masks, ServerShares* shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(sfhe->context);

    // Deserialize ciphertexts
    istringstream is;
    is.rdbuf()->pubsetbuf(ciphertext.inner, ciphertext.size);
    vector<vector<Ciphertext>> ct_vec(data->inp_ct, vector<Ciphertext>(data->filter_size));
    for (int i = 0; i < data->inp_ct; i++) {
        for (int j = 0; j < ct_vec[0].size(); j++) {
            ct_vec[i][j].load(*context, is);
        } 
    }

    // Recast opaque pointers to vectors
    vector<vector<vector<Plaintext>>> masks_vec(data->convs, 
            vector<vector<Plaintext>>(data->inp_ct, 
                vector<Plaintext>(data->filter_size)));
    for (int i = 0; i < masks_vec.size(); i++) {
        for (int j = 0; j < masks_vec[0].size(); j++) {
            for (int z = 0; z < masks_vec[0][0].size(); z++)
                masks_vec[i][j][z] = *(reinterpret_cast<Plaintext*>(masks[i][j][z]));
        } 
    }
    vector<Plaintext> linear_share(data->out_ct);
    vector<Plaintext> linear_mac_share(data->out_ct);
    for (int ct_idx = 0; ct_idx < data->out_ct; ct_idx++) {
        linear_share[ct_idx] = *(reinterpret_cast<Plaintext*>(shares->linear[ct_idx]));
        linear_mac_share[ct_idx] = *(reinterpret_cast<Plaintext*>(shares->linear_mac[ct_idx]));
    }
    vector<Plaintext> r_mac_share(data->inp_ct);
    for (int ct_idx = 0; ct_idx < data->inp_ct; ct_idx++) {
        r_mac_share[ct_idx] = *(reinterpret_cast<Plaintext*>(shares->r_mac[ct_idx]));
    }

    // Recast needed fhe helpers and ciphertexts
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    GaloisKeys *gal_keys = reinterpret_cast<GaloisKeys*>(sfhe->gal_keys);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(sfhe->relin_keys);
    Ciphertext *zero = reinterpret_cast<Ciphertext*>(sfhe->zero);
    Plaintext *mac_key_a = reinterpret_cast<Plaintext*>(shares->mac_key_a);
    Plaintext *mac_key_b = reinterpret_cast<Plaintext*>(shares->mac_key_b);

    // Evaluation
    auto rotation_sets = HE_conv(masks_vec, ct_vec, *data, *evaluator, *relin_keys, *zero); 
    vector<Ciphertext> linear = HE_output_rotations(rotation_sets, *data, *evaluator, *gal_keys, *zero);

    // Secret share the result as well as the MAC of linear_share 
    vector<Ciphertext> linear_mac(data->out_ct);
    for (int ct_idx = 0; ct_idx < data->out_ct; ct_idx++) {
        // Linear share
        evaluator->sub_plain_inplace(linear[ct_idx], linear_share[ct_idx]);
        // Linear MAC share
        evaluator->multiply_plain(linear[ct_idx], *mac_key_a, linear_mac[ct_idx]);
        evaluator->sub_plain_inplace(linear_mac[ct_idx], linear_mac_share[ct_idx]);
    }

    // The ciphertext we receive from the client is preprocessed such that it
    // has data->filter_size rotations of the input image. We need to compute
    // the index of the 0th rotation of the input so indices match up (You
    // don't need to understand the calculation here)
    vector<Ciphertext> r_mac(data->inp_ct);
    int zero_idx = 0;
    int offset = (data->image_w + data->pad_l + data->pad_r - data->filter_w + 1) * data->pad_t + data->pad_l;
    if (offset != 0) {
        zero_idx = (data->filter_size - 1) / 2;
    }
    for (int ct_idx = 0; ct_idx < data->inp_ct; ct_idx++) {
        evaluator->multiply_plain(ct_vec[ct_idx][zero_idx], *mac_key_b, r_mac[ct_idx]);
        evaluator->sub_plain_inplace(r_mac[ct_idx], r_mac_share[ct_idx]);
    }

    // Serialize the resulting ciphertexts into bytearrays and store in ServerShares
    shares->linear_ct = serialize_ct(linear);
    shares->linear_mac_ct = serialize_ct(linear_mac);
    shares->r_mac_ct = serialize_ct(r_mac);
}


void server_fc_online(const ServerFHE* sfhe, const Metadata* data, SerialCT ciphertext,
    char** matrix, ServerShares* shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(sfhe->context);

    // Deserialize ciphertext
    istringstream is;
    is.rdbuf()->pubsetbuf(ciphertext.inner, ciphertext.size);
    Ciphertext input;
    input.load(*context, is);

    // Recast opaque pointers
    vector<Plaintext> matrix_vec(data->inp_ct);
    for (int i = 0; i < data->inp_ct; i++)
        matrix_vec[i] = *(reinterpret_cast<Plaintext*>(matrix[i]));
    Plaintext linear_share = *(reinterpret_cast<Plaintext*>(shares->linear[0]));
    Plaintext linear_mac_share = *(reinterpret_cast<Plaintext*>(shares->linear_mac[0]));
    Plaintext r_mac_share = *(reinterpret_cast<Plaintext*>(shares->r_mac[0]));
  
    // Recast needed fhe helpers and ciphertexts
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    GaloisKeys *gal_keys = reinterpret_cast<GaloisKeys*>(sfhe->gal_keys);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(sfhe->relin_keys);
    Ciphertext *zero = reinterpret_cast<Ciphertext*>(sfhe->zero);
    Plaintext *mac_key_a = reinterpret_cast<Plaintext*>(shares->mac_key_a);
    Plaintext *mac_key_b = reinterpret_cast<Plaintext*>(shares->mac_key_b);

    // Evaluation
    vector<Ciphertext> linear(1, fc_online(input, matrix_vec, *data, *evaluator, *gal_keys, *relin_keys, *zero));

    // Secret share the result as well as the MAC of linear_share 
    vector<Ciphertext> linear_mac(1);
    vector<Ciphertext> r_mac(1);
    // Linear share
    evaluator->sub_plain_inplace(linear[0], linear_share);
    // Linear MAC share
    evaluator->multiply_plain(linear[0], *mac_key_a, linear_mac[0]);
    evaluator->sub_plain_inplace(linear_mac[0], linear_mac_share);
    // R MAC share
    evaluator->multiply_plain(input, *mac_key_b, r_mac[0]);
    evaluator->sub_plain_inplace(r_mac[0], r_mac_share);

    // Serialize the resulting ciphertexts into bytearrays and store in ServerShares
    shares->linear_ct = serialize_ct(linear);
    shares->linear_mac_ct = serialize_ct(linear_mac);
    shares->r_mac_ct = serialize_ct(r_mac);
}


void server_triples_online(const ServerFHE* sfhe, SerialCT client_a, SerialCT client_b, ServerTriples* shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(sfhe->context);

    // Recast needed fhe helpers
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(sfhe->relin_keys);

    // Recast client ciphertexts
    vector<Ciphertext> client_a_ct(shares->vec_len);
    vector<Ciphertext> client_b_ct(shares->vec_len);
    recast_opaque(client_a, client_a_ct, context);
    recast_opaque(client_b, client_b_ct, context);

    // Recast opaque pointers
    Plaintext *mac_key = reinterpret_cast<Plaintext*>(shares->mac_key);
    vector<Plaintext> a_rand(shares->vec_len);
    vector<Plaintext> b_rand(shares->vec_len);
    vector<Plaintext> c_rand(shares->vec_len);
    vector<Plaintext> a_share(shares->vec_len);
    vector<Plaintext> b_share(shares->vec_len); 
    vector<Plaintext> c_share(shares->vec_len);
    vector<Plaintext> a_mac_share(shares->vec_len);
    vector<Plaintext> b_mac_share(shares->vec_len);
    vector<Plaintext> c_mac_share(shares->vec_len);
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < shares->vec_len; i++) {
        a_rand[i] = *(reinterpret_cast<Plaintext*>(shares->a_rand[i]));
        b_rand[i] = *(reinterpret_cast<Plaintext*>(shares->b_rand[i]));
        c_rand[i] = *(reinterpret_cast<Plaintext*>(shares->c_rand[i]));
        a_share[i] = *(reinterpret_cast<Plaintext*>(shares->a_share[i]));
        b_share[i] = *(reinterpret_cast<Plaintext*>(shares->b_share[i]));
        c_share[i] = *(reinterpret_cast<Plaintext*>(shares->c_share[i]));
        a_mac_share[i] = *(reinterpret_cast<Plaintext*>(shares->a_mac_share[i]));
        b_mac_share[i] = *(reinterpret_cast<Plaintext*>(shares->b_mac_share[i]));
        c_mac_share[i] = *(reinterpret_cast<Plaintext*>(shares->c_mac_share[i]));
    }

    // Copy client_a_ct since it gets overwritten in online evaluation
    vector<Ciphertext> a_ct(client_a_ct.size());
    vector<Ciphertext> b_ct(client_a_ct.size());
    for (int i = 0; i < client_a_ct.size(); i++) {
        a_ct[i] = Ciphertext(client_a_ct[i]);
        b_ct[i] = Ciphertext(client_b_ct[i]);
    }

    // Evaluation - share of c is now in client_a_ct
    vector<Ciphertext> c_ct(client_a_ct.size());
    triples_online(client_a_ct, client_b_ct, c_ct, a_rand, b_rand, c_rand, *evaluator, *relin_keys);

    // Create resulting ciphertexts for client
    vector<Ciphertext> a_mac_ct(shares->vec_len);
    vector<Ciphertext> b_mac_ct(shares->vec_len);
    vector<Ciphertext> c_mac_ct(shares->vec_len);

#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < shares->vec_len; i++) {
      // Add a_rand/b_rand to recover encrypted a/b values
      evaluator->add_plain_inplace(a_ct[i], a_rand[i]);
      evaluator->add_plain_inplace(b_ct[i], b_rand[i]);
      // MAC shares
      evaluator->multiply_plain(a_ct[i], *mac_key, a_mac_ct[i]);
      evaluator->multiply_plain(b_ct[i], *mac_key, b_mac_ct[i]);
      evaluator->multiply_plain(c_ct[i], *mac_key, c_mac_ct[i]);
      evaluator->sub_plain_inplace(a_mac_ct[i], a_mac_share[i]);
      evaluator->sub_plain_inplace(b_mac_ct[i], b_mac_share[i]);
      evaluator->sub_plain_inplace(c_mac_ct[i], c_mac_share[i]);
      // Value shares
      evaluator->sub_plain_inplace(a_ct[i], a_share[i]);
      evaluator->sub_plain_inplace(b_ct[i], b_share[i]);
      evaluator->sub_plain_inplace(c_ct[i], c_share[i]);
    }

    // Serialize the ciphertexts
    shares->a_ct = serialize_ct(a_ct);
    shares->b_ct = serialize_ct(b_ct);
    shares->c_ct = serialize_ct(c_ct);
    shares->a_mac_ct = serialize_ct(a_mac_ct);
    shares->b_mac_ct = serialize_ct(b_mac_ct);
    shares->c_mac_ct = serialize_ct(c_mac_ct);
}


void server_rand_online(const ServerFHE* sfhe, SerialCT client_r, ServerTriples* shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(sfhe->context);

    // Recast needed fhe helpers
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(sfhe->relin_keys);

    // Recast client ciphertexts
    vector<Ciphertext> client_r_ct(shares->vec_len);
    recast_opaque(client_r, client_r_ct, context);

    // Recast opaque pointers
    Plaintext *mac_key = reinterpret_cast<Plaintext*>(shares->mac_key);
    vector<Plaintext> r(shares->vec_len);
    vector<Plaintext> s(shares->vec_len);
    vector<Plaintext> mac_s(shares->vec_len);
    for (int i = 0; i < shares->vec_len; i++) {
        r[i] = *(reinterpret_cast<Plaintext*>(shares->a_rand[i]));
        s[i] = *(reinterpret_cast<Plaintext*>(shares->a_share[i]));
        mac_s[i] = *(reinterpret_cast<Plaintext*>(shares->a_mac_share[i]));
    }

    // Evaluation
    vector<Ciphertext> pair_r_mac(shares->vec_len);
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < shares->vec_len; i++) {
        // r_1 + r_2
        evaluator->add_plain_inplace(client_r_ct[i], r[i]);
        // MAC(r_1 + r_2)
        evaluator->multiply_plain(client_r_ct[i], *mac_key, pair_r_mac[i]);
        // Secret share values
        evaluator->sub_plain_inplace(client_r_ct[i], s[i]);
        evaluator->sub_plain_inplace(pair_r_mac[i], mac_s[i]);
    }

    // Serialize the ciphertexts
    shares->a_ct = serialize_ct(client_r_ct);
    shares->a_mac_ct = serialize_ct(pair_r_mac);
}


void client_conv_decrypt(const ClientFHE *cfhe, const Metadata *data, ClientShares *shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(cfhe->context);

    // Recast needed fhe helpers
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);

    // Recast bytearrays to vectors of Ciphertexts and decrypt
    vector<Ciphertext> linear_ct(data->out_ct);
    recast_opaque(shares->linear_ct, linear_ct, context);
    shares->linear = HE_decrypt(linear_ct, *data, *decryptor, *encoder);
    
    vector<Ciphertext> linear_mac_ct(data->out_ct);
    recast_opaque(shares->linear_mac_ct, linear_mac_ct, context);
    shares->linear_mac = HE_decrypt(linear_mac_ct, *data, *decryptor, *encoder);

    vector<Ciphertext> r_mac_ct(data->inp_ct);
    recast_opaque(shares->r_mac_ct, r_mac_ct, context);
    shares->r_mac = conv_preprocess_invert(r_mac_ct, *data, *decryptor, *encoder);
}

void client_fc_decrypt(const ClientFHE *cfhe, const Metadata *data, ClientShares *shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(cfhe->context);

    // Recast needed fhe helpers
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);

    // Recast bytearrays to vectors of Ciphertexts and decrypt
    vector<Ciphertext> linear_ct(1);
    recast_opaque(shares->linear_ct, linear_ct, context);
    shares->linear = new u64*[1];
    shares->linear[0] = fc_postprocess(linear_ct[0], *data, *encoder, *decryptor);
    
    vector<Ciphertext> linear_mac_ct(1);
    recast_opaque(shares->linear_mac_ct, linear_mac_ct, context);
    shares->linear_mac = new u64*[1];
    shares->linear_mac[0] = fc_postprocess(linear_mac_ct[0], *data, *encoder, *decryptor);
     
    vector<Ciphertext> r_mac_ct(1);
    recast_opaque(shares->r_mac_ct, r_mac_ct, context);
    shares->r_mac = new u64*[1];
    shares->r_mac[0] = fc_preprocess_invert(r_mac_ct, *data, *encoder, *decryptor);
}

/* Decrypts the clients multiplication triple share */
void client_triples_decrypt(const ClientFHE *cfhe, SerialCT a, SerialCT b, SerialCT c, SerialCT a_mac,
    SerialCT b_mac, SerialCT c_mac, ClientTriples *shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(cfhe->context);
    
    // Recast needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);

    // Recast received bytearrays to Ciphertexts
    vector<Ciphertext> a_ct(shares->vec_len);
    vector<Ciphertext> b_ct(shares->vec_len);
    vector<Ciphertext> c_ct(shares->vec_len);
    vector<Ciphertext> a_mac_ct(shares->vec_len);
    vector<Ciphertext> b_mac_ct(shares->vec_len);
    vector<Ciphertext> c_mac_ct(shares->vec_len);
    
    recast_opaque(a, a_ct, context);
    recast_opaque(b, b_ct, context);
    recast_opaque(c, c_ct, context);
    recast_opaque(a_mac, a_mac_ct, context);
    recast_opaque(b_mac, b_mac_ct, context);
    recast_opaque(c_mac, c_mac_ct, context);

    // Decrypt Ciphertexts
    shares->a_share = client_triples_postprocess(shares->num, a_ct, *encoder, *decryptor);
    shares->b_share = client_triples_postprocess(shares->num, b_ct, *encoder, *decryptor);
    shares->c_share = client_triples_postprocess(shares->num, c_ct, *encoder, *decryptor);
    shares->a_mac_share = client_triples_postprocess(shares->num, a_mac_ct, *encoder, *decryptor);
    shares->b_mac_share = client_triples_postprocess(shares->num, b_mac_ct, *encoder, *decryptor);
    shares->c_mac_share = client_triples_postprocess(shares->num, c_mac_ct, *encoder, *decryptor);
}

void client_rand_decrypt(const ClientFHE *cfhe, SerialCT share, SerialCT mac_share, ClientTriples *shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(cfhe->context);
    
    // Recast needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);

    // Recast received bytearrays to Ciphertexts
    vector<Ciphertext> s_ct(shares->vec_len);
    vector<Ciphertext> mac_ct(shares->vec_len);
    recast_opaque(share, s_ct, context);
    recast_opaque(mac_share, mac_ct, context);

    // Decrypt Ciphertexts
    shares->a_share = client_triples_postprocess(shares->num, s_ct, *encoder, *decryptor);
    shares->a_mac_share = client_triples_postprocess(shares->num, mac_ct, *encoder, *decryptor);
}


void client_free_keys(const ClientFHE* cfhe) {
    delete (BatchEncoder*) cfhe->encoder;
    delete (Encryptor*) cfhe->encryptor;
    delete (Evaluator*) cfhe->evaluator;
    delete (Decryptor*) cfhe->decryptor;

    // Delete SEALContext ptr
    auto tmp_ptr = static_cast<SEALContext*>(cfhe->context);
    delete tmp_ptr;
}

void server_free_keys(const ServerFHE *sfhe) {
    delete (BatchEncoder*) sfhe->encoder;
    delete (Encryptor*) sfhe->encryptor;
    delete (Evaluator*) sfhe->evaluator;
    delete (GaloisKeys*) sfhe->gal_keys;
    delete (RelinKeys*) sfhe->relin_keys;
    delete (Ciphertext*) sfhe->zero;

    // Delete SEALContext ptr
    auto tmp_ptr = static_cast<SEALContext*>(sfhe->context);
    delete tmp_ptr;
}

void free_ct(SerialCT *ct) {
    delete[] ct->inner;
}

void client_conv_free(const Metadata *data, ClientShares* shares) {
    // Received ciphertexts are allocated by Rust so only need to free input
    free_ct(&shares->input_ct);
    // Free shares
    for (int idx = 0; idx < data->out_chans; idx++) {
        delete[] shares->linear[idx];
        delete[] shares->linear_mac[idx];
    }
    delete[] shares->linear;
    delete[] shares->linear_mac;
    for (int idx = 0; idx < data->inp_chans; idx++) {
        delete[] shares->r_mac[idx];
    }
    delete[] shares->r_mac;
}


void server_conv_free(const Metadata* data, char**** masks, ServerShares* shares) {
    // Free masks
    for (int conv = 0; conv < data->convs; conv++) {
        for (int ct_idx = 0; ct_idx < data->inp_ct; ct_idx++) {
            for (int rot = 0; rot < data->filter_size; rot++) {
                delete (Plaintext*) masks[conv][ct_idx][rot]; 
            }
            delete[] masks[conv][ct_idx];
        }
        delete[] masks[conv];
    } 
    delete[] masks;
    // Free shares
    for (int ct = 0; ct < data->out_ct; ct++) {
        delete (Plaintext*) shares->linear[ct];
        delete (Plaintext*) shares->linear_mac[ct];
    }
    delete[] shares->linear;
    delete[] shares->linear_mac;
    
    for (int ct = 0; ct < data->inp_ct; ct++) {
        delete (Plaintext*) shares->r_mac[ct];
    }
    delete[] shares->r_mac;

    // Free ciphertexts
    free_ct(&shares->linear_ct);
    free_ct(&shares->linear_mac_ct);
    free_ct(&shares->r_mac_ct);
    // Free MAC keys 
    delete (Plaintext*) shares->mac_key_a;
    delete (Plaintext*) shares->mac_key_b;
}


void client_fc_free(ClientShares* shares) {
    // Received ciphertexts are allocated by Rust so only need to free input
    free_ct(&shares->input_ct);
    // Free shares
    delete[] shares->linear[0];
    delete[] shares->linear_mac[0];
    delete[] shares->r_mac[0];
    delete[] shares->linear;
    delete[] shares->linear_mac;
    delete[] shares->r_mac;
}


void server_fc_free(const Metadata* data, char** enc_matrix, ServerShares* shares) {
    // Free matrix
    for (int idx = 0; idx < data->inp_ct; idx++) {
        delete (Plaintext*) enc_matrix[idx];
    }
    delete[] enc_matrix;
    // Free shares
    delete (Plaintext*) shares->linear[0];
    delete (Plaintext*) shares->linear_mac[0];
    delete (Plaintext*) shares->r_mac[0];
    delete[] shares->linear;
    delete[] shares->linear_mac;
    delete[] shares->r_mac;
    // Free ciphertexts
    free_ct(&shares->linear_ct);
    free_ct(&shares->linear_mac_ct);
    free_ct(&shares->r_mac_ct);
    // Free MAC keys 
    delete (Plaintext*) shares->mac_key_a;
    delete (Plaintext*) shares->mac_key_b;
}


void client_triples_free(ClientTriples* shares) {
    // Free shares
    delete[] shares->a_share;
    delete[] shares->b_share;
    delete[] shares->c_share;
    delete[] shares->a_mac_share;
    delete[] shares->b_mac_share;
    delete[] shares->c_mac_share;
}


void server_triples_free(ServerTriples* shares) {
    // Free vectors of Plaintexts
    for (int idx = 0; idx < shares->vec_len; idx++) {
        delete (Plaintext*) shares->a_rand[idx];
        delete (Plaintext*) shares->b_rand[idx];
        delete (Plaintext*) shares->c_rand[idx];
        delete (Plaintext*) shares->a_share[idx];
        delete (Plaintext*) shares->b_share[idx];
        delete (Plaintext*) shares->c_share[idx];
        delete (Plaintext*) shares->a_mac_share[idx];
        delete (Plaintext*) shares->b_mac_share[idx];
        delete (Plaintext*) shares->c_mac_share[idx];
    }
    delete[] shares->a_rand;
    delete[] shares->b_rand;
    delete[] shares->c_rand;
    delete[] shares->a_share;
    delete[] shares->b_share;
    delete[] shares->c_share;
    delete[] shares->a_mac_share;
    delete[] shares->b_mac_share;
    delete[] shares->c_mac_share;
    // Free ciphertexts
    free_ct(&shares->a_ct);
    free_ct(&shares->b_ct);
    free_ct(&shares->c_ct);
    free_ct(&shares->a_mac_ct);
    free_ct(&shares->b_mac_ct);
    free_ct(&shares->c_mac_ct);
    // Free MAC key
    delete (Plaintext*) shares->mac_key;
}


void client_rand_free(ClientTriples* shares) {
    // Free shares
    delete[] shares->a_share;
    delete[] shares->a_mac_share;
}


void server_rand_free(ServerTriples* shares) {
    // Free vectors of Plaintexts
    for (int idx = 0; idx < shares->vec_len; idx++) {
        delete (Plaintext*) shares->a_rand[idx];
        delete (Plaintext*) shares->a_share[idx];
        delete (Plaintext*) shares->a_mac_share[idx];
    }
    delete[] shares->a_rand;
    delete[] shares->a_share;
    delete[] shares->a_mac_share;
    // Free ciphertexts
    free_ct(&shares->a_ct);
    free_ct(&shares->a_mac_ct);
    // Free MAC key
    delete (Plaintext*) shares->mac_key;
}
