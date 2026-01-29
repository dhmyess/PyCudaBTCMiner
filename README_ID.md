# PyCudaBTCMiner

Aplikasi penambangan Bitcoin berbasis Python dengan akselerasi GPU menggunakan teknologi NVIDIA CUDA. Meskipun menggunakan GPU terkuat sekalipun, aplikasi ini tidak dapat menyamai kecepatan perangkat ASIC yang didesain khusus untuk mining. Oleh karena itu, strategi mining yang digunakan adalah pendekatan acak (random) dengan harapan menemukan nonce yang valid lebih cepat dibandingkan metode iterasi berurutan.

## Prasyarat

1. **GPU NVIDIA dengan dukungan CUDA**
   - GPU harus kompatibel dengan CUDA toolkit
   
2. **NVIDIA CUDA Toolkit (Versi 10 atau lebih tinggi)**
   - Download dari: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Pastikan CUDA Toolkit terinstal dan tersedia di PATH sistem

3. **Python 3.x**
   - Diperlukan untuk menjalankan script mining

## Struktur File

### File CUDA (Kernel GPU)
- **liblooper.cu** - Source code kernel CUDA untuk scanning nonce secara iteratif dari 0 hingga 4.294.967.295 (0x00000000 - 0xFFFFFFFF)
- **rr.cu** - Source code kernel CUDA untuk scanning nonce secara acak

### Script Build
- **build.bat** - Script kompilasi untuk sistem operasi Windows (menghasilkan `.dll`)
- **build.sh** - Script kompilasi untuk sistem operasi Linux (menghasilkan `.so`)

### Modul Python
- **looper.py** - Modul Python sebagai interface antara `miner.py` dan kernel CUDA (`liblooper`)
- **rrnonce.py** - Modul Python sebagai interface antara `random_miner.py` dan kernel CUDA (`rr`)

### Aplikasi Mining
- **miner.py** - Aplikasi mining dengan metode **extranonce2 acak** dan **full scan nonce** (0-4294967295)
- **random_miner.py** - Aplikasi mining dengan metode **extranonce2 acak** dan **nonce acak**

### Script Testing
- **test.py** - Script testing untuk full scan dan benchmark hashrate GPU (menggunakan block 933,995)
- **test_1_block.py** - Script testing menggunakan header block pertama Bitcoin (Genesis Block)

### File Dokumentasi
- **working_script.png** - Screenshot contoh aplikasi yang berjalan
- **Readme.md** - File dokumentasi ini

## Cara Penggunaan

### 1. Kompilasi Kernel CUDA

**Untuk Windows:**
```bash
build.bat
```

**Untuk Linux:**
```bash
chmod +x build.sh
./build.sh
```

Proses kompilasi akan menghasilkan:
- **Windows:** `rr.dll` dan `liblooper.dll`
- **Linux:** `rr.so` dan `liblooper.so`

### 2. Testing Kernel dan Benchmark

Sebelum memulai mining, jalankan test untuk memastikan kernel CUDA berfungsi dengan baik:

```bash
python3 test.py
```

Script ini akan:
- Menguji fungsionalitas kernel CUDA
- Mengukur hashrate GPU Anda
- Menampilkan waktu yang dibutuhkan untuk menyelesaikan full scan (0-4294967295)

### 3. Konfigurasi Mining

Edit file `miner.py` atau `random_miner.py` sesuai kebutuhan:

```python
config = {
    "pool_address": "public-pool.io",  # Alamat mining pool
    "pool_port": 3333,                 # Port mining pool
    "user_name": "ALAMAT_BITCOIN_ANDA", # Ganti dengan alamat Bitcoin Anda
    "password": "x",                    # Password (biasanya "x" atau "password")
    "min_diff": 1,                      # Difficulty minimum (minimal 1)
    "poll_sleep": 0.05,
    "reconnect_backoff": 5.0,
}
```

**Catatan Penting:**
- Ganti `user_name` dengan alamat Bitcoin Anda sendiri
- `min_diff` minimal adalah 1 (kernel hanya menghasilkan hash jika difficulty ≥ 1)

### 4. Memulai Mining

#### Metode A: Extranonce2 Acak + Full Scan Nonce
**Direkomendasikan untuk GPU dengan hashrate > 4.294 MH/s**

```bash
python3 miner.py
```

Metode ini akan:
- Menggunakan extranonce2 secara acak
- Melakukan full scan nonce dari 0 hingga 4.294.967.295
- Cocok untuk GPU yang sangat kuat

#### Metode B: Extranonce2 Acak + Nonce Acak
**Direkomendasikan untuk semua GPU**

```bash
python3 random_miner.py
```

Metode ini akan:
- Menggunakan extranonce2 secara acak
- Melakukan sampling nonce acak (196.608.000 nonce per extranonce2)
- Lebih efisien untuk GPU dengan kecepatan sedang

### 5. Penyesuaian Sampling Nonce Acak (Opsional)

Jika Anda menggunakan `random_miner.py` dan ingin meningkatkan jumlah nonce yang dicoba per extranonce2:

1. Buka file `rr.cu`
2. Cari baris 157 dan 158:
   ```c
   int threadsPerBlock = 256;  // Thread per block
   int blocksPerGrid = 768;    // Block per grid
   ```
3. Hitung total nonce = `1000 × threadsPerBlock × blocksPerGrid`
   - Default: 1000 × 256 × 768 = 196.608.000 nonce
4. Tingkatkan nilai jika GPU Anda kuat, contoh:
   ```c
   int threadsPerBlock = 512;
   int blocksPerGrid = 1024;
   // Total: 1000 × 512 × 1024 = 524.288.000 nonce
   ```
5. Kompilasi ulang kernel dengan menjalankan `build.bat` atau `build.sh`

## Pemahaman Konsep Mining

### Difficulty Target
- Mining pool menetapkan target difficulty untuk setiap share
- Semakin tinggi difficulty, semakin sulit menemukan share yang valid
- `min_diff` dalam konfigurasi adalah difficulty minimum yang akan dicari oleh miner

### Extranonce2
- Bagian dari coinbase transaction yang dapat diubah oleh miner
- Mengubah extranonce2 menghasilkan merkle root yang berbeda
- Setiap extranonce2 memberikan 4.294.967.296 kemungkinan nonce

### Strategi Random vs Sequential
- **Sequential (ASIC):** Mencoba nonce 0, 1, 2, 3, ... secara berurutan
- **Random (PyCudaBTCMiner):** Mencoba nonce secara acak dengan harapan menemukan lebih cepat

## Monitoring dan Output

Saat mining berjalan, Anda akan melihat output seperti:
```
[+] New Job ID detected! Switching to job: 6978d6ba00000709 Pool difficulty : 1.0
  [✅] 02:16:09 Found Share!
       Job ID: 6978d6ba00000709
       EN2   : 755c8940
       Nonce : d0ee7abf
       Hash  : 000000000fb1abc2d054395e4aaec12fb8c4d2d4a962056e4c656b6708b8c6
       Diff  : 260.99 (Target from pool: 1.00)
       Hashrate : 4294.96 MH/s
```

## Troubleshooting

### Error: "liblooper.so not found" atau "rr.dll not found"
- Pastikan Anda sudah menjalankan script build (`build.bat` atau `build.sh`)
- Pastikan file `.so` atau `.dll` ada di direktori yang sama dengan script Python

### Error: "NVCC not found"
- CUDA Toolkit belum terinstal atau belum ditambahkan ke PATH
- Download dan install dari: https://developer.nvidia.com/cuda-downloads

### GPU tidak terdeteksi
- Pastikan driver NVIDIA terbaru sudah terinstal
- Jalankan `nvidia-smi` untuk mengecek status GPU

### Hashrate sangat rendah
- GPU Anda mungkin tidak cukup kuat untuk mining Bitcoin
- Coba gunakan `random_miner.py` untuk efisiensi lebih baik


## Dukungan

Jika Anda mengalami masalah:
1. Pastikan semua prasyarat terpenuhi
2. Jalankan `test.py` untuk memverifikasi kernel CUDA
3. Periksa log error untuk detail masalah
4. Buat issue di repository GitHub (jika tersedia)

---

**Catatan Developer:**
- Saya menggunakan Linux untuk development
- Script Windows (`build.bat`) belum diuji secara menyeluruh
- Jika Anda menemukan masalah di Windows, mohon informasikan

**Lisensi:** Silakan gunakan sesuai kebutuhan Anda untuk tujuan edukasi dan eksperimen. Tapi jika anda berhasil mendapatkan block rewards tolong beri donasi ke saya seikhlasnya alamat btc saya bc1qngzehzs73x2p5k7r7pa7na69ej89p40qxnrh60
