import looper
import time
def lev(nonce):
    byte0 = (nonce & 0xFF000000) >> 24
    byte1 = (nonce & 0x00FF0000) >> 16
    byte2 = (nonce & 0x0000FF00) >> 8
    byte3 = (nonce & 0x000000FF)
    little_endian_value = (byte3 << 24) | (byte2 << 16) | (byte1 << 8) | byte0
    return little_endian_value

max_target = 0x00000000ffff0000000000000000000000000000000000000000000000000000
# Test header = version + prevhash + merkle_root + ntime + nbits
# https://learnmeabitcoin.com/explorer/block/00000000000000000000132e777589264ca2ffc47319ea55f8cbf6f180a7293d
hex_header = "00800220e8f33cff4a360027dbf6a0ffec523939b329a91ae58a01000000000000000000d67d87fef5331b8055007ca44082f38c63887a019352ca517332dc516fc2f6e5d2807b69a1fc0117"
pool_target = 1
print("Starting mining...")
start = time.time()
nonce, blockhash = looper.mining_nonce(hex_header, pool_target)
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
print("\nThis is sample blockhash from block 934,235 mined on 29 Jan 2026, 15:46:26")
