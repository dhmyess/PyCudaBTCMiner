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
# Test header
hex_header = "0000003ed928bbba0d4904b547fd0a12f7985f00bd8af5c618bb0100000000000000000059833d60fa5d3b072a13ba1c15e78053c5dc31f36aaf09e5925c70a9de86348763ed7869a1fc0117"
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
print("\nThis is sample blockhash from block 933,995 mined on 27 Jan 2026, 16:52:51")
