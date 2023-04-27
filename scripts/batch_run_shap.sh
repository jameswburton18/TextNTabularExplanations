#!/bin/bash
# for idx in 0 # {7..10}
# do
#     for cfg in airbnb_${idx}
#     do
#         sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
#         # sbatch --job-name=$cfg scripts/train.sh $cfg
#         sleep 4 # so that wandb runs don't get assigned the same number
#     done
# done

# imdb_genre", "prod_sent", "fake", "kick", "jigsaw", "wine"
for ds_type in imdb_genre prod_sent fake kick jigsaw wine
do
    # sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
    sbatch --job-name=${ds_type}_shap scripts/run_shap.sh $ds_type
    sleep 4 # so that wandb runs don't get assigned the same number
done
