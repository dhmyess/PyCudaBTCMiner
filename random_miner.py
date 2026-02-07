import socket
import json
import time
import hashlib
import sys
import traceback
import random
from datetime import datetime
from rrnonce import mining_nonce, lev

# -------------------------------
# CONFIG
# -------------------------------
config = {
    "pool_address": "127.0.0.1",
    "pool_port": 3333,
    "user_name": "bc1qngzehzs73x2p5k7r7pa7na69ej89p40qxnrh60",
    "password": "x",
    "min_diff": 1,
    "batch_number": 42,
    "poll_sleep": 0.05,
    "reconnect_backoff": 5.0,
    "max_extranonce2": 0xFFFFFFFF
}

# -------------------------------
# Stratum client
# -------------------------------
class StratumClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = b""
        self.extranonce1 = None
        self.extranonce2_size = 0
        self.connected = False
        self.difficulty = 100000.0
        self._connect()

    def _connect(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(20.0) # Timeout lebih panjang karena full loop mungkin lama
        self.sock.connect((self.host, self.port))
        self.buffer = b""
        self.connected = True

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None
        self.connected = False

    def send(self, method, params, _id=1):
        payload = json.dumps({"id": _id, "method": method, "params": params})
        try:
            self.sock.sendall((payload + "\n").encode())
        except Exception:
            self.connected = False

    def recv_blocking_line(self):
        while b"\n" not in self.buffer:
            try:
                data = self.sock.recv(4096)
                if not data:
                    raise Exception("Disconnected")
                self.buffer += data
            except socket.timeout:
                pass # Continue waiting
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line.decode())

    def poll_message(self):
        msgs = []
        if not self.connected:
            return msgs
        while True:
            try:
                self.sock.settimeout(0.0)
                data = self.sock.recv(4096)
                if not data:
                    self.connected = False
                    break
                self.buffer += data
            except BlockingIOError:
                break
            except socket.timeout:
                break
            except Exception:
                break
        
        self.sock.settimeout(20.0)

        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            try:
                if line.strip():
                    msgs.append(json.loads(line.decode()))
            except Exception:
                pass
        return msgs

    def login(self, user, password):
        self.send("mining.subscribe", [])
        try:
            r = self.recv_blocking_line()
            res = r.get("result")
            self.extranonce1 = res[1]
            self.extranonce2_size = res[2]
        except Exception:
            self.extranonce1 = ""
            self.extranonce2_size = 4

        self.send("mining.authorize", [user, password])
        self.recv_blocking_line()
        return True

# -------------------------------
# helpers
# -------------------------------
def rev_hex(hexstr):
    return "".join([hexstr[i:i+2] for i in range(0, len(hexstr), 2)][::-1])

def rev_8B(hexstr):
    out = []
    for i in range(0, len(hexstr), 8):
        out.append(rev_hex(hexstr[i:i+8]))
    return "".join(out)

def double_sha256_hex(hexdata):
    b = bytes.fromhex(hexdata)
    return hashlib.sha256(hashlib.sha256(b).digest()).hexdigest()

def build_block_header_from_job(job, extranonce1, extranonce2):
    job_id = job[0]
    prevhash = rev_8B(job[1])
    coinb1 = job[2]
    coinb2 = job[3]
    merkle_branch = job[4]
    version = rev_hex(job[5])
    nbits = rev_hex(job[6])
    ntime = rev_hex(job[7])
    coinbase = coinb1 + extranonce1 + extranonce2 + coinb2
    root = double_sha256_hex(coinbase)
    for m in merkle_branch:
        root = double_sha256_hex(root + m)
    clean_job = job[8]
    header = version + prevhash + root + ntime + nbits
    return job_id, header

# -------------------------------
# Main mining loop
# -------------------------------
def mine_loop():
    backoff = config["reconnect_backoff"]

    while True:
        client = None
        try:
            client = StratumClient(config["pool_address"], config["pool_port"])
            client.login(config["user_name"], config["password"])
            print("[+] Auth OK")
            print(f"[+] Initial difficulty: {client.difficulty:.2f}")
            current_job = None
            current_job_id = None
            extranonce2_int = random.randint(0,0xffffffff)
            need_new_job = True 
            en2_counter = 0
            target_miner = client.difficulty
            start_time = time.time()
            while client.connected:
                # 1. Poll messages
                for msg in client.poll_message():
                    if msg.get("method") == "mining.notify":
                        new_job = msg["params"]
                        new_job_id = new_job[0]
                        new_clean_job = new_job[8]

                        if new_job_id != current_job_id or new_clean_job:
                            stop_time = time.time()
                            elapse_time = stop_time - start_time
                            rate = ((en2_counter*(256*4096*config["batch_number"]))/elapse_time)/1000000
                            print(f"  [i] {en2_counter} extranonce2 has been tried in {elapse_time:.2f} hashrate = {rate:.2f} MH/s")
                            print(f"[+] New Job detected! ID: {new_job_id}")
                            current_job = new_job
                            current_job_id = new_job_id
                            
                            # Reset loop logic for new job
                            start_time = time.time()
                            en2_counter = 0
                            extranonce2_int = random.randint(0,0xffffffff)
                            need_new_job = False
                    
                    elif msg.get("method") == "mining.set_difficulty":
                        new_difficulty = float(msg["params"][0])
                        if new_difficulty < target_miner:
                            if new_difficulty < config["min_diff"]:
                                target_miner = config["min_diff"]
                            else:
                                target_miner = new_difficulty
                        if new_difficulty > target_miner:
                            target_miner = new_difficulty
                        if new_difficulty != client.difficulty:
                            client.difficulty = new_difficulty
                            print(f"[+] Pool difficulty updated: {new_difficulty:.6f}")

                # 2. Wait for job
                if need_new_job and current_job is None:
                    time.sleep(config["poll_sleep"])
                    continue

                # 3. Mining Setup
                extranonce2 = f"{extranonce2_int:0{client.extranonce2_size*2}x}"
                job_id, header_hex = build_block_header_from_job(
                    current_job,
                    client.extranonce1,
                    extranonce2
                )
                # 4. Mining
                found_shares_list = mining_nonce(header_hex, target_miner,config["batch_number"])

                stop_time = time.time()
                elapse_time = stop_time - start_time

                # 5. Submit
                if found_shares_list:
                    for (nonce, blockhash) in found_shares_list:
                        found_difficulty = 0x00000000ffff0000000000000000000000000000000000000000000000000000 / int(blockhash, 16)
                        now = datetime.now()
                        current_time_string = now.strftime("%H:%M:%S")
                        
                        print(f"  [✅] {current_time_string} Found Share!")
                        print(f"       Job ID       : {current_job[0]}")
                        print(f"       extranonce2  : {extranonce2}")
                        print(f"       Nonce        : {nonce:08x}")
                        print(f"       Hash         : {blockhash}")
                        print(f"       Diff         : {found_difficulty:.2f}")

                        params = [
                            config["user_name"],
                            current_job[0],
                            extranonce2,
                            current_job[7],
                            f"{nonce:08x}"
                        ]
                        client.send("mining.submit", params)
                else:
                    pass

                # 6. Ganti Extranonce2
                en2_counter += 1
                extranonce2_int = random.randint(0,0xffffffff)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            if client:
                client.close()
            return
        except Exception as e:
            print("[!] Exception:", e)
            traceback.print_exc()
            if client:
                client.close()
            print(f"[!] Reconnecting in {backoff}s...")
            time.sleep(backoff)

if __name__ == "__main__":
    mine_loop()
