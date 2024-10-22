import os
import sys
import subprocess

def main():
    if len(sys.argv) < 3:
        print("Usage: python dist_train.py <config> <gpus> [additional args...]")
        sys.exit(1)

    config = sys.argv[1]
    gpus = sys.argv[2]
    additional_args = sys.argv[3:]

    nnodes = os.getenv('NNODES', '1')
    node_rank = os.getenv('NODE_RANK', '0')
    port = os.getenv('PORT', '29501')
    master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')

    pythonpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.environ['PYTHONPATH'] = f"{pythonpath}:{os.getenv('PYTHONPATH', '')}"

    command = [
        sys.executable, '-m', 'torch.distributed.launch',
        '--nnodes', nnodes,
        '--node_rank', node_rank,
        '--master_addr', master_addr,
        '--nproc_per_node', gpus,
        '--master_port', port,
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'train.py')),
        config,
        '--seed', '0',
        '--launcher', 'pytorch'
    ] + additional_args

    print("Running command:", ' '.join(command))
    print("Environment variables:")
    print(f"  NNODES={nnodes}")
    print(f"  NODE_RANK={node_rank}")
    print(f"  PORT={port}")
    print(f"  MASTER_ADDR={master_addr}")
    print(f"  PYTHONPATH={os.environ['PYTHONPATH']}")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        sys.exit(e.returncode)

if __name__ == '__main__':
    main()