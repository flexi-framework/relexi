module purge
module load gcc/8.3.0
module load nvidia-hpc-sdk/22.2
module load cmake/3.15.4
module load hdf5/1.12.0
module load anaconda3/2023.03
conda activate /gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2

export PYTHONPATH=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2/lib/python3.9/site-packages
export CUDNN_LIBRARY=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2/lib
export LD_LIBRARY_PATH=$CUDNN_LIBRARY:$LD_LIBRARY_PATH:/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2/lib/python3.9/site-packages/torch/lib
export NO_CHECKS=1
export SMARTREDIS_DIR=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2/smartredis/install
export PATH=$SMARTREDIS_DIR/bin:$PATH
export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH
export SSDB="127.0.0.1:6379"
