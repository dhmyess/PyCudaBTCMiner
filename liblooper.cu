#include <stdint.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Macro rotasi bit SHA-256
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

// Konstanta K SHA-256
__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Fungsi LEV (Little Endian Value)
__device__ __forceinline__ uint32_t cuda_lev(uint32_t val) {
    return ((val & 0xFF000000) >> 24) |
           ((val & 0x00FF0000) >> 8)  |
           ((val & 0x0000FF00) << 8)  |
           ((val & 0x000000FF) << 24);
}

// Kernel Mining Linear (Grid Stride / Offset Based)
__global__ void find_nonce_kernel(
    const uint32_t* input_state, 
    uint32_t* result_buffer, 
    uint32_t* found_count,
    uint32_t start_nonce,
    uint32_t max_results
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = start_nonce + idx;

    // --- LOGIKA MINING ---

    // Step 1: Prepare W2
    uint32_t w[64];
    w[0] = input_state[8];
    w[1] = input_state[9];
    w[2] = input_state[10];
    w[3] = nonce; // Nonce Linear
    w[4] = 0x80000000;
    
    #pragma unroll
    for(int i=5; i<15; i++) w[i] = 0;
    w[15] = 0x280;

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROTR(w[i-15], 7) ^ ROTR(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROTR(w[i-2], 17) ^ ROTR(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    // Step 2: Compression Round 1
    uint32_t a = input_state[0]; 
    uint32_t b = input_state[1]; 
    uint32_t c = input_state[2]; 
    uint32_t d = input_state[3]; 
    uint32_t e = input_state[4]; 
    uint32_t f = input_state[5]; 
    uint32_t g = input_state[6]; 
    uint32_t h = input_state[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    w[0] = input_state[0] + a;
    w[1] = input_state[1] + b;
    w[2] = input_state[2] + c;
    w[3] = input_state[3] + d;
    w[4] = input_state[4] + e;
    w[5] = input_state[5] + f;
    w[6] = input_state[6] + g;
    w[7] = input_state[7] + h;

    // Step 3: Prepare W3
    w[8] = 0x80000000;
    #pragma unroll
    for(int i=9; i<15; i++) w[i] = 0;
    w[15] = 0x100;

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROTR(w[i-15], 7) ^ ROTR(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROTR(w[i-2], 17) ^ ROTR(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    // Step 4: Compression Round 2
    a = 0x6a09e667; b = 0xbb67ae85; c = 0x3c6ef372; d = 0xa54ff53a;
    e = 0x510e527f; f = 0x9b05688c; g = 0x1f83d9ab; h = 0x5be0cd19;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    // --- STEP 5: CHECK RESULT ---
    uint32_t diff_target1 = input_state[11];
    uint32_t diff_target2 = input_state[12];

    uint32_t h37 = (0x5be0cd19 + h);
    if (h37 != 0x00000000) return;
    uint32_t h36 = (0x1f83d9ab + g);
    if (cuda_lev(h36) > diff_target1) return;
    uint32_t h35 = (0x9b05688c + f);
    if (cuda_lev(h35) > diff_target2) return;

    // Found valid nonce!
    // Simpan ke buffer menggunakan atomic index
    uint32_t index = atomicAdd(found_count, 1);
    
    // Pastikan tidak overflow buffer
    if (index < max_results) {
        result_buffer[index] = nonce;
    }
}

extern "C" {
    // Fungsi ini akan melakukan loop penuh dari 0 s/d 0xFFFFFFFF
    int run_gpu_miner(uint32_t* input_data, uint32_t* output_buffer, int max_results) {
        uint32_t *d_input, *d_result_buffer, *d_found_count;
        uint32_t h_found_count = 0;

        // Grid Configuration
        // GPU Modern bisa menangani grid besar. 
        // 256 thread per block adalah ukuran standar yang aman.
        int threadsPerBlock = 256;
        int blocksPerGrid = 8192; // 256 * 8192 = 2,097,152 threads per batch launch
        uint32_t batch_size = threadsPerBlock * blocksPerGrid;

        // Alokasi Memory
        cudaMalloc((void**)&d_input, 13 * sizeof(uint32_t));
        cudaMalloc((void**)&d_result_buffer, max_results * sizeof(uint32_t));
        cudaMalloc((void**)&d_found_count, sizeof(uint32_t));

        // Init Data
        cudaMemcpy(d_input, input_data, 13 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemset(d_found_count, 0, sizeof(uint32_t)); // Counter dimulai dari 0

        // --- FULL LOOP 0 to 0xFFFFFFFF ---
        // Kita loop di host agar tidak kena Timeout (TDR) OS jika loop di dalam kernel terlalu lama
        for (uint64_t start = 0; start <= 0xFFFFFFFF; start += batch_size) {
            
            // Cek overflow untuk batch terakhir
            if (start > 0xFFFFFFFF) break;

            find_nonce_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_input, 
                d_result_buffer, 
                d_found_count,
                (uint32_t)start,
                (uint32_t)max_results
            );

            // Opsional: Synchronize setiap beberapa batch untuk memastikan GPU tidak freeze UI/OS
            // Tapi untuk performa maksimal, kita bisa minimize sync.
            // Kita sync di akhir loop saja jika memungkinkan, atau per batch.
            // Sync per batch lebih aman:
            cudaDeviceSynchronize();
        }
        
        // Ambil hasil akhir
        cudaMemcpy(&h_found_count, d_found_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        if (h_found_count > 0) {
            if (h_found_count > max_results) h_found_count = max_results;
            cudaMemcpy(output_buffer, d_result_buffer, h_found_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_input);
        cudaFree(d_result_buffer);
        cudaFree(d_found_count);

        return (int)h_found_count;
    }
}
