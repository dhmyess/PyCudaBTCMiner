import gminer
import time
def lev(nonce):
    byte0 = (nonce & 0xFF000000) >> 24
    byte1 = (nonce & 0x00FF0000) >> 16
    byte2 = (nonce & 0x0000FF00) >> 8
    byte3 = (nonce & 0x000000FF)
    little_endian_value = (byte3 << 24) | (byte2 << 16) | (byte1 << 8) | byte0
    return little_endian_value

max_target = 0x00000000ffff0000000000000000000000000000000000000000000000000000
# Test header
hex_header = "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d"
pool_target = 1
print("Starting mining...")
start = time.time()
nonce, blockhash = gminer.mining_nonce(hex_header, pool_target)
elapsed = time.time() - start

if nonce is None or blockhash is None :
    print("❌ No valid nonce found.")
    rate = 0xffffffff/elapsed
    Hashrate = rate/1000000
    print(f"Hashrate : {Hashrate:.2f} MH/s")
    print(f"You need {elapsed:.3f} second to complete nonce scan from 0 to {0xffffffff}")
else:
    print("✅ found!")
    print(f"Nonce       : {nonce} (0x{nonce:08x})")
    print(f"blockhash   : {blockhash}")
    diff = max_target / int(blockhash, 16)
    print(f"Difficulty  : {diff:.2f}")
    rate = lev(nonce)/elapsed
    Hashrate = rate/1000000
    print(f"Hashrate    : {Hashrate:.2f} MH/s")
    print(f"\nYou need {(0xffffffff/rate):.3f} second to complete nonce scan from 0 to {0xffffffff}")
print("\nThis is sample blockhash from block 1 mined on 03 Jan 2009, 18:15:05")
