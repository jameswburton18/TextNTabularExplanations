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
for code in disbert #drob
do
    for ds_type in jigsaw wine imdb_genre prod_sent fake kick 
    do
        # sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
        sbatch --job-name=${code}_${ds_type}_shap scripts/run_shap.sh $ds_type $code
        sleep 4 # so that wandb runs don't get assigned the same number
    done
done
