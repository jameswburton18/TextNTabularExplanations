#!/bin/bash
for idx in 0
do
    for model_name in imdb_genre_${idx}
    do
        # sbatch --job-name=$model_name scripts/run_exps_office_pc.sh $model_name
        # sbatch --job-name=$model_name scripts/run_exps.sh $model_name test
        sbatch --job-name=$model_name scripts/run_exps.sh $model_name validation
        sleep 4 # so that wandb runs don't get assigned the same number
    done
done