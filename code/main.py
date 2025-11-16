import os
import argparse
import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "true"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc_per_node", type=int, required=False)
    parser.add_argument("--nnodes", type=int, required=False)
    parser.add_argument("--rdzv_id", type=int, required=False)
    parser.add_argument("--rdzv_endpoint", type=str, required=False)
    parser.add_argument('--sh_script', action='store_true',
                        help='call main.py via sh script instead of slurm')
    parser.add_argument("--config_file", nargs='+')

    args, unknown_args = parser.parse_known_args()
    config_args = ' '.join(args.config_file)
    nproc_per_node = args.nproc_per_node
    nnodes = args.nnodes
    rdzv_id = args.rdzv_id
    rdzv_endpoint = args.rdzv_endpoint

    if args.sh_script:
        print('Shell script used.')
        run_yaml = f"../TORCHRUN run.py --config_file {config_args} {' '.join(unknown_args)}"

    else:
        run_yaml = (f"srun torchrun --nproc_per_node={nproc_per_node} --nnodes={nnodes} --rdzv_id={rdzv_id} "
                        f"--rdzv_backend=c10d --rdzv_endpoint={rdzv_endpoint} run.py "
                        f"--config_file {config_args} {' '.join(unknown_args)}")

    os.system(run_yaml)
