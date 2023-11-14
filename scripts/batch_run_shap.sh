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
# for code in deberta disbert drob  bert #
# do
#     for ds_type in airbnb  #channel # salary #kick jigsaw fake wine imdb_genre prod_sent 
#     do
#         # sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
#         sbatch --job-name=${code}_${ds_type}_shap scripts/run_shap.sh $ds_type $code
#         sleep 4 # so that wandb runs don't get assigned the same number
#     done
# done

for code in bert disbert drob deberta #
do
    for ds_type in airbnb channel salary kick jigsaw fake wine imdb_genre prod_sent 
    do
        # sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
        sbatch --job-name=${code}_${ds_type}_shap scripts/run_shap.sh $ds_type $code
        sleep 4 # so that wandb runs don't get assigned the same number
    done
done

# for code in disbert #
# do
#     for ds_type in airbnb  #channel # salary #kick jigsaw fake wine imdb_genre prod_sent 
#     do
#         # sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
#         sbatch --dependency=afterok:322531,322532 --job-name=${code}_${ds_type}_shap scripts/run_shap.sh $ds_type $code
#         sleep 4 # so that wandb runs don't get assigned the same number
#     done
# done

# for code in bert #
# do
#     for ds_type in airbnb  #channel # salary #kick jigsaw fake wine imdb_genre prod_sent 
#     do
#         # sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
#         sbatch --dependency=afterok:322533,322534 --job-name=${code}_${ds_type}_shap scripts/run_shap.sh $ds_type $code
#         sleep 4 # so that wandb runs don't get assigned the same number
#     done
# done

# for code in drob #
# do
#     for ds_type in airbnb  #channel # salary #kick jigsaw fake wine imdb_genre prod_sent 
#     do
#         # sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
#         sbatch --dependency=afterok:322536,322537 --job-name=${code}_${ds_type}_shap scripts/run_shap.sh $ds_type $code
#         sleep 4 # so that wandb runs don't get assigned the same number
#     done
# done

# for code in deberta #
# do
#     for ds_type in airbnb  #channel # salary #kick jigsaw fake wine imdb_genre prod_sent 
#     do
#         # sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
#         sbatch --dependency=afterok:322538,322540 --job-name=${code}_${ds_type}_shap scripts/run_shap.sh $ds_type $code
#         sleep 4 # so that wandb runs don't get assigned the same number
#     done
# done