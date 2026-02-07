def right_rotate(value, bits):
    return ((value >> bits) | (value << (32 - bits))) & 0xFFFFFFFF
def lev(nonce):
    byte0 = (nonce & 0xFF000000) >> 24
    byte1 = (nonce & 0x00FF0000) >> 16
    byte2 = (nonce & 0x0000FF00) >> 8
    byte3 = (nonce & 0x000000FF)
    little_endian_value = (byte3 << 24) | (byte2 << 16) | (byte1 << 8) | byte0
    return little_endian_value

def fchunk(hex_header):
    pw = []
    # Loop melalui string, dengan langkah 8 karakter
    for i in range(0, len(hex_header), 8):
        # Ambil potongan 8 karakter
        chunk = hex_header[i:i+8]
        
        # Konversi ke format heksadesimal dengan awalan '0x'
        # Gunakan int(..., 16) untuk memastikan ini adalah bilangan heksa
        # lalu format kembali ke string heksa 8 digit
        value = int(chunk, 16)
        
        # Tambahkan ke list pw
        pw.append(value)

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
        h = g
        g = f
        f = e
        e = (d + temp1) & 0xFFFFFFFF
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & 0xFFFFFFFF

    h10 = (h0 + a) & 0xFFFFFFFF
    h11 = (h1 + b) & 0xFFFFFFFF
    h12 = (h2 + c) & 0xFFFFFFFF
    h13 = (h3 + d) & 0xFFFFFFFF
    h14 = (h4 + e) & 0xFFFFFFFF
    h15 = (h5 + f) & 0xFFFFFFFF
    h16 = (h6 + g) & 0xFFFFFFFF
    h17 = (h7 + h) & 0xFFFFFFFF

    return h10, h11, h12, h13, h14, h15, h16, h17, pw[16], pw[17], pw[18]
