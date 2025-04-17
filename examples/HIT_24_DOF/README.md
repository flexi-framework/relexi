# Turbulence Modeling for Homogeneous Isotropic Turbulence

This tutorial demonstrates how Relexi can be used to learn turbulence models for homogeneous isotropic turbulence.
It is meant to serve as a starting point for users to modify for their own applications.
The tutorial is based on our work in

[Kurz, Marius, Philipp Offenhäuser, and Andrea Beck. "Deep reinforcement learning for turbulence modeling in large eddy simulations." International Journal of Heat and Fluid Flow 99 (2023): 109094.](https://doi.org/10.1016/j.ijheatfluidflow.2022.109094)

The accepted manuscript is also available without paywall at [arXiv](https://doi.org/10.48550/arXiv.2206.11038). 
The training and evaluation setup is identical and the provided trained checkpoints and models are the same as in the original work.
For in-depthe discussion of the results and the problem setup, please have a look at the original publication.

The following sections provide a brief overview of the problem setup, how to run the pretrained model to reproduce the results in the paper, and how to train the model from scratch.

## Problem Description

The physical setup is a forced homogeneous isotropic turbulence flow with a Reynolds number of $Re_{\lambda} \approx 180$.
The flow is simulated in a periodic domain with a side length of $\mathbf{x} \in [2,\pi]$ with periodic boundary conditions.
The linear forcing injects energy into the flow that counteracts the viscous dissipation to obtain a statistically steady flow.

A DNS simulation serves as baseline and provides the target statistics and suitable intermediate filtered flow states that can be used as initial condition for the training runs.
The goal for the training is to recover the mean energy spectrum of the DNS simulation.
For this, the model receives the element-wise velocity field $u_i$ as input and predicts for each DG element the model constant of Smagorinsky's model $C_s$.
This model constant is then used to calculate the eddy viscosity $\nu_t$ according to Smagorinsky as

$\nu_t = (C_s \Delta)^2 \sqrt{2S_{ij}S_{ij}}, \qquad \text{with} \quad S_{ij} = \frac{1}{2} \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right)$

The simulation environment is implemented using [FLEXI](https://numericsresearchgroup.org/flexi_index.html), which is a compressible flow solver based on the discontinuous Galerkin method.
The corresponding simulation-specific files are provided in the `simulation_files` folder.
This includes most importantly
- the mesh file `CART_HEX_PERIODIC_004_mesh.h5`,
- the state/checkpoints files `run_f200_N5_4Elems_State_*.h5` that are used to restart the training runs from,
- the `parameter_flexi.ini` file, which entails the simulation parameters,
- the file `DNS_spectrum_stats_t2_to_t10.csv` containing the target statistics of the DNS simulation used as training target.
The [FLEXI userguide](https://numericsresearchgroup.org/flexi_userguide.pdf) provides more details on the individual parameters and how to adapt the simulation setup.

## Running the pretrained model

First, make sure that `relexi` is installed correctly on the system using the installation instructions given in the main `README.md` file.
The checkpoint for the pretrained model is provided in the `trained_model` folder.
The `prm_eval.yaml` file is already set up to run the pretrained model for $t=20$, while providing the flow states and the statistics of the simulation as output files.
To evaluate the pretrained model, i.e. running a simulation without performing training or optimization steps, simply run
```bash
relexi --only-eval prm_eval.yaml
```
The FLEXI simulation will produce a range of state files `*State*.h5` that contain the flow solution at different points in time and a CSV file that contains more information on the turbulent statistics of the flow.
For convenience, FLEXI also directly converts those into `*.vtu` files that can be opened directly in ParaView or similar programs.
For more details on how to post-process the results, have a look at the [FLEXI userguide](https://numericsresearchgroup.org/flexi_userguide.pdf).

## Training the model

The following paragraphs show you how to run the training starting from a serial run on a local machine up to an HPC cluster.

### Training on a local machine

Again, make sure that `relexi` is installed correctly on the system using the installation instructions given in the main `README.md` file.
The training can then be started on a local machine using
```bash
relexi prm.yaml
```
where `prm.yaml` is the parameter file provided in this repository for local training.

The following selection of parameters is of particular interest to adapt the training process:

| Parameter                    | Description                                           |
| ---------------------------- | ----------------------------------------------------- |
| `run_name`                   | Name of run; determines the name of folder in `logs/` |
| `executable_path`            | Path to the executable                                |
| `train_num_iterations`       | Number of training iterations to run                  |
| `num_parallel_environments`  | Number of parallel environments for training          |

### Accelerate training using MPI

If MPI is available on the machine and FLEXI is compiled with MPI (see the [FLEXI userguide](https://numericsresearchgroup.org/flexi_userguide.pdf)),the training can be accelerated by using multiple processors per environment.
For this, the following parameters have to be adapted:

| Parameter                    | Description                                  |
| ---------------------------- | -------------------------------------------- |
| `num_procs_per_environment`  | Number of processors used per environment    |
| `env_launcher`               | To use MPI, it should be set to `mpirun`     |

### Training on a cluster

For running the training on a distributed cluster, two different schedulers are currently supported: `SLURM` and `PBS`.
The scripts `jobfile.slurm` and `jobfile.pbs` are given as exemplary jobfiles to submit the job to the scheduler using either
```bash
qsub jobfile.pbs      # for PBS
sbatch jobfile.slurm  # for SLURM
```
Please be aware that both scripts have to be adapted to the specific cluster setup regarding core counts, queue names, and other parameters.
If only a single node is available in the job, Relexi runs in `local` mode, i.e. the same way as it would run on a local machine.
If running in `distributed` mode (i.e. on multiple nodes of the system), the first node is always reserved to run relexi itself, while the other nodes are used to run the FLEXI instances.
Most importantly, ensure to change the launcher to `env_launcher: srun` in the `prm_train.yaml` file to run the training with SLURM.
Other than that, `prm_train.yaml` entails the exact training setup of the reference publication such that the reported results can be obtained by submitting this job on the HPC system.
