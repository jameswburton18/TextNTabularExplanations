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

for idx in 30 39 #{1..4}
do
    for cfg in  fake_${idx} kick_${idx} imdb_genre_${idx} jigsaw_${idx} wine_${idx} prod_sent_${idx} 
    do
        # sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
        sbatch --job-name=$cfg scripts/train.sh $cfg
        sleep 4 # so that wandb runs don't get assigned the same number
    done
done