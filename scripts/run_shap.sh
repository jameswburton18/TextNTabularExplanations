#!/bin/bash
# Instructing SLURM to locate and assign X number of nodes with Y number of
# cores in each node. X,Y are integers.
#SBATCH -N 1
#SBATCH -c 4

# Governs the run time limit and resource limit for the job. See QOS tables for combinations
# Available QOS: [debug, short, long-high-prio, long-low-prio, long-cpu]
#SBATCH -p res-gpu-small
#SBATCH --qos short #long-high-prio
#SBATCH -t 02-00:00
# -x shows which ones to ignore
# #SBATCH -x gpu[0-8] #[7,8,10,11,12]
# #SBATCH --gres=gpu:ampere:1 #--gres gpu for normal, --gres=gpu:ampere:1 for whole 80gb card
#SBATCH -x gpu[0-6,10-12] #[0-6,7,8,10,11,12]
#SBATCH --gres gpu #--gres gpu for normal, --gres=gpu:ampere:1 for whole 80gb card

# Job name appears in the squeue output, output is the output filename 
#SBATCH -o logs/%x-%A.out

# Pick how much memory to allocate per CPU core.
#SBATCH --mem 8G

source ./env/bin/activate

# Commands to be run:
python --version
nvidia-smi
hostname
echo "Node id: $SLURM_NODEID"

mem=`nvidia-smi --query-gpu=memory.total --format=csv | tail -n 1 | awk '{print $1}'`
echo "$mem Mb available"

date '+%c'
echo "--ds_type $1 --text_model_code $2"
# python combine_LIME.py --dataset fraud --model joint
python src/run_shap.py --ds_type $1 --text_model_code $2
