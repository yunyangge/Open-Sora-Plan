import subprocess
import wandb
import re
import time
import threading


def monitor_npu_power():
    result = subprocess.run(["npu-smi", "info"], stdout=subprocess.PIPE, text=True)
    avg_power = 0

    for line in result.stdout.splitlines():
        if line.startswith('|'):
            parts = line.split('|')
            if '910' in parts[1]:
                match = re.search(r'\d+\.\d+', parts[3])
                avg_power += float(match.group())

    avg_power /= 8
    return avg_power

def wandb_log_npu_power():
    if wandb.run is None:
        raise NotImplementedError("wandb is not initialized")

    def log():
        while True:
            avg_power = monitor_npu_power()
            wandb.log({'npu_power': avg_power})
            time.sleep(5)

    log_thread = threading.Thread(target=log)
    log_thread.daemon = True 
    log_thread.start()