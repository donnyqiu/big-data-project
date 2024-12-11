import subprocess
import sys

def launch_processes(script_path, world_size):
    # Start the parameter server (rank 0)
    ps_process = subprocess.Popen(
        [sys.executable, script_path, "--rank=0", f"--world_size={world_size}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Start workers
    worker_processes = []
    for rank in range(1, world_size):
        worker_process = subprocess.Popen(
            [sys.executable, script_path, "--rank={}".format(rank), f"--world_size={world_size}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        worker_processes.append(worker_process)

    # Wait for all processes to finish and print their output
    try:
        for p in worker_processes + [ps_process]:
            stdout, stderr = p.communicate()
            if stdout:
                print(stdout.decode())
            if stderr:
                print(stderr.decode(), file=sys.stderr)
    finally:
        # Ensure all processes are terminated when the script exits
        for p in worker_processes + [ps_process]:
            p.terminate()

if __name__ == "__main__":
    launch_processes("parameter_server.py", 8)