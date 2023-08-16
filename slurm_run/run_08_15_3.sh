#!/bin/bash
#SBATCH -c 24                # Number of cores (-c)
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_requeue  # Partition to submit to
#SBATCH --mem=40G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --account lichtman_lab
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1

# run code
cd ../from_core
source ~/venv1/bin/activate
python3 train_unet_mb_pytorch.py ../data/test_08_16/ims_masks_train.h5 ../data/test_08_16/models1/ -val_dataset_h5 ../data/test_08_16/ims_masks_val.h5