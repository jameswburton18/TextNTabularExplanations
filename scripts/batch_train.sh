for idx in 0 1 # {7..10}
do
    for cfg in airbnb_${idx}
    do
        sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
        sleep 4 # so that wandb runs don't get assigned the same number
    done
done