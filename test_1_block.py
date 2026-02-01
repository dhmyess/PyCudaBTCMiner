import looper
import time

# Konstanta Max Target (Difficulty 1)
max_target = 0x00000000ffff0000000000000000000000000000000000000000000000000000

hex_header = "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d"
pool_target = 1

print(f"Starting mining scan (Full Range: 0 - 0xFFFFFFFF)...")
print("Please wait...")

start = time.time()

found_shares = looper.mining_nonce(hex_header, pool_target)

elapsed = time.time() - start

# --- Tampilkan Hasil ---
if not found_shares:
    print("❌ No valid nonce found in this range.")
else:
    print(f"\n✅ Found {len(found_shares)} valid share(s)!")
    
    # Loop semua hasil yang ditemukan
    for i, (nonce, blockhash) in enumerate(found_shares):
        print(f"\n--- Share #{i+1} ---")
        print(f"Nonce       : {nonce} (0x{nonce:08x})")
        print(f"Blockhash   : {blockhash}")
        
        # Hitung difficulty share tersebut
        diff = max_target / int(blockhash, 16)
        print(f"Difficulty  : {diff:.2f}")

# --- Hitung Hashrate ---
total_hashes = 0xFFFFFFFF 
rate = total_hashes / elapsed
hashrate_mh = rate / 1000000

print("\n" + "="*40)
print(f"Execution Stats:")
print(f"Elapsed Time : {elapsed:.4f} seconds")
print(f"Hashrate     : {hashrate_mh:.2f} MH/s")
print("="*40)
print("\nThis is sample blockhash from block 1 mined on 03 Jan 2009, 18:15:05")
