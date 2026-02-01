import looper
import time

# Konstanta Max Target (Difficulty 1)
max_target = 0x00000000ffff0000000000000000000000000000000000000000000000000000

# Sample Header
# https://learnmeabitcoin.com/explorer/block/00000000000000000000132e777589264ca2ffc47319ea55f8cbf6f180a7293d
hex_header = "00800220e8f33cff4a360027dbf6a0ffec523939b329a91ae58a01000000000000000000d67d87fef5331b8055007ca44082f38c63887a019352ca517332dc516fc2f6e5d2807b69a1fc0117"
pool_target = 1

print(f"Starting mining scan (Full Range: 0 - 0xFFFFFFFF)...")
print("Please wait...")

start = time.time()

# Panggil fungsi looper (sekarang mengembalikan LIST of tuples)
# Return format: [(nonce1, hash1), (nonce2, hash2), ...] atau [] jika kosong
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
# Karena loop pasti berjalan penuh (kecuali di-interrupt), total hash adalah 2^32 - 1
total_hashes = 0xFFFFFFFF 
rate = total_hashes / elapsed
hashrate_mh = rate / 1000000

print("\n" + "="*40)
print(f"Execution Stats:")
print(f"Elapsed Time : {elapsed:.4f} seconds")
print(f"Hashrate     : {hashrate_mh:.2f} MH/s")
print("="*40)
print("\nThis is sample blockhash from block 934,235 mined on 29 Jan 2026")
