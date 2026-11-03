# ai generated script manager (monitors for freezes and crashes to automatically restart the program)
import hashlib
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque

SCRIPT = sys.argv[1]
TRIGGER = "WARNING:gym_donkeycar.envs.donkey_sim:waiting for sim to start.."
TRIGGER_LIMIT = 10
BUFFER_MAXLEN = 40
RESTART_DELAY = 600
EXIT_DELAY = 5
IMAGE_PATH = "training_logs/obs.jpeg"
FREEZE_LIMIT = 25 * 60        # restart if image unchanged this long (seconds)
IMAGE_CHECK_INTERVAL = 60     # how often to check the image (seconds)


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def kill_sim():
    subprocess.run(
        ["killall", "-9", "donkey_sim"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def grab_image_hash():
    # hashes the latest obs image; a missing file counts as "unchanged"
    if not os.path.exists(IMAGE_PATH):
        return "missing"
    with open(IMAGE_PATH, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def reader(stream, q):
    # reads output in a thread so the main loop never blocks on it
    try:
        for line in stream:
            q.put(line)
    except (ValueError, OSError):
        pass
    q.put(None)


while True:
    print(f"launching script: {SCRIPT}", flush=True)
    buffer = deque(maxlen=BUFFER_MAXLEN)
    trigger_count = 0

    process = subprocess.Popen(
        [sys.executable, "-u", "-m", SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    q = queue.Queue()
    threading.Thread(target=reader, args=(process.stdout, q), daemon=True).start()

    last_hash = None
    last_change = time.monotonic()
    next_check = time.monotonic()

    restart = False
    try:
        while True:
            # --- handle output (waits at most 1s so the image check keeps running) ---
            try:
                line = q.get(timeout=1)
            except queue.Empty:
                line = ""
            else:
                if line is None:  # script exited
                    break
                line = line.rstrip("\n")
                print(line, flush=True)
                buffer.append(line)

                if TRIGGER in line:
                    trigger_count += 1
                    if trigger_count >= TRIGGER_LIMIT:
                        print(f"\ntrigger seen {trigger_count} times in a row, last logs:")
                        print("\n".join(buffer))
                        print("--- killing script and sim ---\n", flush=True)
                        restart = True
                        break
                elif line.strip():
                    trigger_count = 0

            # --- frozen image check ---
            now = time.monotonic()
            if now >= next_check:
                next_check = now + IMAGE_CHECK_INTERVAL
                try:
                    h = grab_image_hash()
                except Exception as e:
                    print(f"image check failed: {e}", flush=True)
                    h = None

                if h is not None:
                    if h != last_hash:
                        last_hash = h
                        last_change = now
                    elif now - last_change >= FREEZE_LIMIT:
                        mins = int((now - last_change) // 60)
                        print(f"\nimage unchanged for {mins} minutes, last logs:")
                        print("\n".join(buffer))
                        print("--- killing script and sim ---\n", flush=True)
                        restart = True
                        break
    except KeyboardInterrupt:
        stop(process)
        kill_sim()
        sys.exit(0)
    finally:
        stop(process)
        process.stdout.close()

    if restart:
        kill_sim()
        print(f"waiting {RESTART_DELAY // 60} minutes before relaunch...", flush=True)
        time.sleep(RESTART_DELAY)
    else:
        print(f"script exited with code {process.returncode}", flush=True)
        kill_sim()
        time.sleep(EXIT_DELAY)