import ctypes
import os
import time
import hashlib

# Load library
lib_name = "./rr.so" if os.name == 'posix' else "./rr.dll"
if not os.path.exists(lib_name):
    print(f"Error: {lib_name} not found.")
    exit()

cuda_miner = ctypes.CDLL(os.path.abspath(lib_name))

# Konstanta batas maksimum output (harus sama/lebih kecil dari alokasi Python, tapi sinkron dengan logika C)
MAX_RESULTS = 20

# Argumen: (Pointer Input, Seed, Pointer Output Buffer, Max Results)
cuda_miner.run_gpu_miner.argtypes = [
    ctypes.POINTER(ctypes.c_uint32), 
    ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_int
]
# Return: Jumlah nonce yang ditemukan (int)
cuda_miner.run_gpu_miner.restype = ctypes.c_int

def right_rotate(value, bits):
    return ((value >> bits) | (value << (32 - bits))) & 0xFFFFFFFF

def lev(nonce):
    byte0 = (nonce & 0xFF000000) >> 24
    byte1 = (nonce & 0x00FF0000) >> 16
    byte2 = (nonce & 0x0000FF00) >> 8
    byte3 = (nonce & 0x000000FF)
    little_endian_value = (byte3 << 24) | (byte2 << 16) | (byte1 << 8) | byte0
    return little_endian_value

def double_sha256_hex(hexdata):
    b = bytes.fromhex(hexdata)
    return hashlib.sha256(hashlib.sha256(b).digest()).hexdigest()

def mining_nonce(hex_header, pool_target):
    pw = []
    for i in range(0, len(hex_header), 8):
        chunk = hex_header[i:i+8]
        value = int(chunk, 16)
        pw.append(value)

    # --- Bagian Awal (Midstate Calculation) ---
    h0 = 0x6a09e667
    h1 = 0xbb67ae85
    h2 = 0x3c6ef372
    h3 = 0xa54ff53a
    h4 = 0x510e527f
    h5 = 0x9b05688c
    h6 = 0x1f83d9ab
    h7 = 0x5be0cd19

    k = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ]
    w1c=[0]*64
    for i in range(16):
        w1c[i]=pw[i]
    for i in range(16, 64):
        s0 = right_rotate(w1c[i-15], 7) ^ right_rotate(w1c[i-15], 18) ^ (w1c[i-15] >> 3)
        s1 = right_rotate(w1c[i-2], 17) ^ right_rotate(w1c[i-2], 19) ^ (w1c[i-2] >> 10)
        w1c[i] = (w1c[i-16] + s0 + w1c[i-7] + s1) & 0xFFFFFFFF

    a, b, c, d, e, f, g, h = h0, h1, h2, h3, h4, h5, h6, h7
    for i in range(64):
        S1 = right_rotate(e, 6) ^ right_rotate(e, 11) ^ right_rotate(e, 25)
        ch = (e & f) ^ ((~e) & g)
        temp1 = (h + S1 + ch + k[i] + w1c[i]) & 0xFFFFFFFF
        S0 = right_rotate(a, 2) ^ right_rotate(a, 13) ^ right_rotate(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        temp2 = (S0 + maj) & 0xFFFFFFFF
        h = g; g = f; f = e; e = (d + temp1) & 0xFFFFFFFF
        d = c; c = b; b = a; a = (temp1 + temp2) & 0xFFFFFFFF

    h10 = (h0 + a) & 0xFFFFFFFF
    h11 = (h1 + b) & 0xFFFFFFFF
    h12 = (h2 + c) & 0xFFFFFFFF
    h13 = (h3 + d) & 0xFFFFFFFF
    h14 = (h4 + e) & 0xFFFFFFFF
    h15 = (h5 + f) & 0xFFFFFFFF
    h16 = (h6 + g) & 0xFFFFFFFF
    h17 = (h7 + h) & 0xFFFFFFFF

    if pool_target < 1:
        diff_target1 = 0xffff0000
        diff_target2 = 0xffffffff
    else:
        if pool_target < 4294967296:
            diff_target1 = int(0xffff0000 / pool_target)
            diff_target2 = 0xffffffff
        else:
            if pool_target < 18446744073709551616:
                diff_target1 = 0
                diff_target2 = int(0xffff000000000000 / pool_target)
            else:
                diff_target1 = 0
                diff_target2 = 0

    # Input ke GPU
    input_data = (ctypes.c_uint32 * 13)(
        h10, h11, h12, h13, h14, h15, h16, h17,
        pw[16], pw[17], pw[18], diff_target1, diff_target2
    )

    # Persiapkan Buffer Output
    output_buffer = (ctypes.c_uint32 * MAX_RESULTS)()

    # Gunakan waktu sekarang sebagai seed random
    seed = int(time.time_ns() & 0xffffffff)

    # Call GPU function
    count_found = cuda_miner.run_gpu_miner(input_data, seed, output_buffer, MAX_RESULTS)

    found_shares = []

    # Loop selesai (ini yang Anda maksud dengan output -1, secara teknis function return berarti selesai)
    # Jika count_found == 0, berarti tidak ada share.
    if count_found > 0:
        for i in range(count_found):
            le = output_buffer[i]
            
            # Reconstruct data
            levnonce = le
            full_header = hex_header + f"{levnonce:08x}"
            final_hash = double_sha256_hex(full_header)
            nonce = lev(le)
            
            byte_chunks = [final_hash[i:i+2] for i in range(0, len(final_hash), 2)]
            reversed_chunks = byte_chunks[::-1]
            blockhash = "".join(reversed_chunks)
            
            # Tambahkan ke list hasil
            found_shares.append((nonce, blockhash))

    return found_shares
