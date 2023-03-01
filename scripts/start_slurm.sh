#Enabling Slurm Status
sudo systemctl enable slurmctld.service
sudo systemctl enable slurmd.service
#Enabling Stopping Status
sudo systemctl stop slurmctld.service
sudo systemctl stop slurmd.service
#Starting Slurm Services
sudo systemctl start slurmctld.service
sudo systemctl start slurmd.service
#Checking Slurm Status
sudo systemctl status slurmctld.service
sudo systemctl status slurmd.service
