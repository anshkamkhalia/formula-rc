# ai
import queue
import subprocess
import sys
import threading
import time
from collections import deque

SCRIPT = "src.data.sim_evaluation"
TRIGGER = "WARNING:gym_donkeycar.envs.donkey_sim:waiting for sim to start.."
TRIGGER_LIMIT = 10
BUFFER_MAXLEN = 40
RESTART_DELAY = 1200
EXIT_DELAY = 5
FREEZE_LIMIT = 10 * 60        # restart if the script prints nothing this long (seconds)


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

    last_output = time.monotonic()

    restart = False
    try:
        while True:
            try:
                line = q.get(timeout=1)
            except queue.Empty:
                # --- frozen check: episodes are capped at max_steps, so long silence means the sim hung ---
                silent = time.monotonic() - last_output
                if silent >= FREEZE_LIMIT:
                    print(f"\nno output for {int(silent // 60)} minutes, last logs:")
                    print("\n".join(buffer))
                    print("--- killing script and sim ---\n", flush=True)
                    restart = True
                    break
                continue

            if line is None:  # script exited
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                break

            last_output = time.monotonic()
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
    elif process.returncode == 0:
        print("eval finished", flush=True)
        break
    else:
        print(f"script exited with code {process.returncode}", flush=True)
        kill_sim()
        time.sleep(EXIT_DELAY)