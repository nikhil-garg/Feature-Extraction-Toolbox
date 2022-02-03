#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --time=23:00:00 # DD-HH:MM:SS
#SBATCH --mem-per-cpu=4GB
#SBATCH --array=1-24
#SBATCH --job-name=ecog_encoding

echo "Moving files"
cp -r $HOME/Feature-Extraction-Toolbox $SLURM_TMPDIR/Feature-Extraction-Toolbox
cd $SLURM_TMPDIR/Feature-Extraction-Toolbox

echo "Starting application"
mkdir -p "$HOME/ecog_results_features/"

if $HOME/env/bin/python baseline_exploration_parallel.py --run $SLURM_ARRAY_TASK_ID ; then
    echo "Copying results"
    mv "accuracy_log_$SLURM_ARRAY_TASK_ID.csv" "$HOME/ecog_results_features/"
fi

wait