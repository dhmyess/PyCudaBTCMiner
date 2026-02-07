# PyCudaBTCMiner

A Python-based Bitcoin mining application with GPU acceleration using NVIDIA CUDA technology. Even with the most powerful GPUs, this application cannot match the speed of dedicated ASIC mining devices. Therefore, it employs a random sampling strategy, hoping to find valid nonces faster than sequential iteration methods.

## Prerequisites

1. **NVIDIA GPU with CUDA Support**
   - GPU must be compatible with CUDA toolkit
   
2. **NVIDIA CUDA Toolkit (Version 10 or higher)**
   - Download from: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Ensure CUDA Toolkit is installed and available in system PATH

3. **Python 3.x**
   - Required to run mining scripts

## File Structure

### CUDA Files (GPU Kernels)
- **liblooper.cu** - CUDA kernel source code for sequential nonce scanning from 0 to 4,294,967,295 (0x00000000 - 0xFFFFFFFF)
- **rr.cu** - CUDA kernel source code for random nonce scanning

### Build Scripts
- **build.bat** - Compilation script for Windows (generates `.dll` files)
- **build.sh** - Compilation script for Linux (generates `.so` files)

### Python Modules
- **looper.py** - Python module serving as interface between `miner.py` and CUDA kernel (`liblooper`)
- **rrnonce.py** - Python module serving as interface between `random_miner.py` and CUDA kernel (`rr`)

### Mining Applications
- **miner.py** - Mining application using **random extranonce2** and **full nonce scan** (0-4294967295)
- **random_miner.py** - Mining application using **random extranonce2** and **random nonce**

### Testing Scripts
- **test.py** - Testing script for full scan and GPU hashrate benchmark (using block 933,995)
- **test_1_block.py** - Testing script using Bitcoin's first block header (Genesis Block)

### Documentation Files
- **working_script.png** - Screenshot of the application running
- **Readme.md** - This documentation file

## Usage Instructions

### 1. Compile CUDA Kernels

**For Windows:**
```bash
build.bat
```

**For Linux:**
```bash
chmod +x build.sh
./build.sh
```

The compilation process will generate:
- **Windows:** `rr.dll` and `liblooper.dll`
- **Linux:** `rr.so` and `liblooper.so`

### 2. Testing Kernel and Benchmark

Before starting mining, run tests to ensure CUDA kernels are working properly:

```bash
python3 test.py
```

This script will:
- Test CUDA kernel functionality
- Measure your GPU's hashrate
- Display time required to complete full scan (0-4294967295)

### 3. Mining Configuration

Edit `miner.py` or `random_miner.py` according to your needs:

```python
config = {
    "pool_address": "public-pool.io",  # Mining pool address
    "pool_port": 3333,                 # Mining pool port
    "user_name": "YOUR_BITCOIN_ADDRESS", # Replace with your Bitcoin address
    "password": "x",                    # Password (usually "x" or "password")
    "min_diff": 1,                      # Minimum difficulty (minimum is 1)
    "poll_sleep": 0.05,
    "reconnect_backoff": 5.0,
}
```

**Important Notes:**
- Replace `user_name` with your own Bitcoin address
- `min_diff` minimum is 1 (kernel only outputs hash if difficulty ≥ 1)

### 4. Start Mining

#### Method A: Random Extranonce2 + Full Nonce Scan
**Recommended for GPUs with hashrate > 4,294 MH/s**

```bash
python3 miner.py
```

This method will:
- Use random extranonce2
- Perform full nonce scan from 0 to 4,294,967,295
- Suitable for very powerful GPUs

#### Method B: Random Extranonce2 + Random Nonce
**Recommended for all GPUs**

```bash
python3 random_miner.py
```

This method will:
- Use random extranonce2
- Perform random nonce sampling (196,608,000 nonces per extranonce2)
- More efficient for medium-speed GPUs

### 5. Adjusting Random Nonce Sampling (Optional)

If you're using `random_miner.py` and want to increase the number of nonces tried per extranonce2:

1. Open `random_miner.py` file
2. change batch_number in config:
config = {
    "pool_address": "127.0.0.1",
    "pool_port": 3333,
    "user_name": "bc1qngzehzs73x2p5k7r7pa7na69ej89p40qxnrh60",
    "password": "x",
    "min_diff": 1,
    "batch_number": 42,   # 1.048.576 nonce per batch x 42 = 44.040.192 sample nonce per extranonce2
    "poll_sleep": 0.05,
    "reconnect_backoff": 5.0,
    "max_extranonce2": 0xFFFFFFFF
}
3. If you want to change how many nonce launch per batch edit rr.cu`line 134, 135
   int threadsPerBlock = 256;
   int blocksPerGrid = 4096;
   nonce launch per batch = 256 * 4096 = 1.048.576
4. Recompile kernel by running `build.bat` or `build.sh`

## Understanding Mining Concepts

### Difficulty Target
- Mining pool sets target difficulty for each share
- Higher difficulty means harder to find valid shares
- `min_diff` in configuration is the minimum difficulty the miner will search for

### Extranonce2
- Part of coinbase transaction that can be modified by miner
- Changing extranonce2 produces different merkle root
- Each extranonce2 provides 4,294,967,296 possible nonces

### Random vs Sequential Strategy
- **Sequential (ASIC):** Tries nonces 0, 1, 2, 3, ... in order
- **Random (PyCudaBTCMiner):** Tries random nonces hoping to find faster

## Monitoring and Output

When mining is running, you'll see output like:
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

### Error: "liblooper.so not found" or "rr.dll not found"
- Ensure you've run the build script (`build.bat` or `build.sh`)
- Verify `.so` or `.dll` files exist in the same directory as Python scripts

### Error: "NVCC not found"
- CUDA Toolkit is not installed or not added to PATH
- Download and install from: https://developer.nvidia.com/cuda-downloads

### GPU not detected
- Ensure latest NVIDIA drivers are installed
- Run `nvidia-smi` to check GPU status

### Very low hashrate
- Your GPU may not be powerful enough for Bitcoin mining
- Try using `random_miner.py` for better efficiency

## Support

If you encounter issues:
1. Ensure all prerequisites are met
2. Run `test.py` to verify CUDA kernel
3. Check error logs for details
4. Create an issue on GitHub repository (if available)

---

**Developer Notes:**
- I use Linux for development
- Windows script (`build.bat`) has not been thoroughly tested
- If you encounter issues on Windows, please report them

**License:** Feel free to use for educational and experimental purposes. But if you succeed in getting block rewards, please donate to me, my btc address bc1qngzehzs73x2p5k7r7pa7na69ej89p40qxnrh60
