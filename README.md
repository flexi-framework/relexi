![logo](https://numericsresearchgroup.org/images/icons/relexi.svg "RELEXI")

[![license](https://img.shields.io/github/license/flexi-framework/relexi.svg?maxAge=2592000 "GPL-3.0 License")](LICENSE.md)
[![doi](https://img.shields.io/badge/DOI-10.1016/j.simpa.2022.100422-blue "DOI")](https://doi.org/10.1016/j.simpa.2022.100422)

# About Relexi

Relexi is a Reinforcement Learning (RL) framework developed for the high-order HPC flow solver [FLEXI][flexi].
However, Relexi is developed with modularity in mind and allows to used with other HPC solvers as well.
Relexi builds upon TensorFlow and its RL extension TF-Agents.
For the efficient communication, data handling and the managment of the simulations runs on HPC systems, Relexi uses the SmartSim package with its SmartRedis communication clients.
For details on its scaling behavior, suitability for HPC and for use cases, please see
* [Kurz, M., Offenhäuser, P., Viola, D., Resch, M., & Beck, A. (2022). Relexi—A scalable open source reinforcement learning framework for high-performance computing. Software Impacts, 14, 100422.](https://www.sciencedirect.com/science/article/pii/S2665963822001063)
* [Kurz, M., Offenhäuser, P., Viola, D., Shcherbakov, O., Resch, M., & Beck, A. (2022). Deep Reinforcement Learning for Computational Fluid Dynamics on HPC Systems. Journal of Computational Science, 65, 101884.](https://www.sciencedirect.com/science/article/pii/S1877750322002435)
* [Kurz, M., Offenhäuser, P., & Beck, A. (2023). Deep reinforcement learning for turbulence modeling in large eddy simulations. International Journal of Heat and Fluid Flow, 99, 109094.](https://arxiv.org/pdf/2206.11038)

This is a scientific project.
If you use Relexi or find it helpful, please cite the project using a suitable reference from the list above referring to either the general Relexi project, its HPC aspects or its application for scientific modeling tasks, respectively.

# Installation

The following quick start details a standard installation of the Relexi framework.

### Dependencies

Relexi has a variety of dependencies.
The main dependencies of Relexi are listed in the following with their supported version.

| Package          | Version       | Note     | 
|:-----------------|--------------:|:---------|
| Python           |     ≥3.8      |          |
| TensorFlow       |     ≥2.9      |          |
| TF-Agents        |     ≥0.13     |          |
| SmartSim         |     ≥0.3.2    |          |
| SmartRedis       |      0.2.0    |          |
| Cmake            |     ≥3.0      |          |
| Make             |     ≥4.0      |          |
| gcc-fortran      |     ≥9.4      | GCC-10 not supported! (GCC-11 and higher is fine) |
| gcc              |     ≥9.4      |          |
| gcc-c++          |     ≥9.4      |          |

Please be ware that The major dependencies (SmartSim, TensorFlow, FLEXI) might have a more expansive dependency tree, for which we refer the user to the corresponding documentation for details.

### Prerequisites
* Open a terminal
* Change into the directory where you want to install Relexi and its dependecies
* For convienience, save the current directory with
    ```
    ROOTDIR=$(pwd)
    ```

* It is highly recommended to use some form of virtual environment for the installation. You can use any tool you like, we use `virtualenv` which can be installed with 
    ```
    python3 -m pip install virtualenv
    ```

* Then create and activate a new environment with
    ```
    python3 -m virtualenv env_relexi
    source env_relexi/bin/activate
    ```

* Then install the necessary dependecies
    ```
    python3 -m pip install smartredis cmake tensorflow tf-agents pyyaml matplotlib
    ```

### Install SmartSim
The installation commands were changed in SmartSim version `0.4.0`. So use the following commands depending on the version you want to use.

#### SmartSim 0.3.2
* Now, install SmartSim in version `0.3.2`. For this, first the package has to be installed via pip and then we can install it using the smart tool provided by SmartSim.
    ```
    pip install smartsim==0.3.2
    smart --clobber
    smart --clean
    smart --no_tf --no_pt -v
    SMARTSIM_DIR=$(smart --site)
    export PATH=$PATH:$SMARTSIM_DIR/_core/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SMARTSIM_DIR}/_core/lib
    ```

#### SmartSim >= 0.4.0
* Now, install SmartSim in version you like. The following commands install the latest version. The individual tools of the `smart` command line tool are now not longer prefixed by a double dash.
    ```
    pip install smartsim
    smart clobber
    smart clean
    smart build --no_tf --no_pt -v
    SMARTSIM_DIR=$(smart site)
    export PATH=$PATH:$SMARTSIM_DIR/_core/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SMARTSIM_DIR}/_core/lib
    ```

### Install SmartRedis
* Go back into the main directory
    ```
    cd $ROOTDIR
    ```

* Then, we install the SmartRedis clients for C/C++ and Fortran. For this we clone its repository and build version `0.2.0`
    ```
    git clone https://github.com/CrayLabs/SmartRedis.git
    cd SmartRedis
    git checkout v0.2.0
    make lib -j
    ```

* Export the build directory, so FLEXI finds the installation to link against, in order to build with support for SmartRedis.
    ```
    export SMARTREDIS_DIR=$(pwd)
    ```

### Install FLEXI
* Clone the required version of FLEXI from GitHub and build it with the standard compile flags
    ```
    cd $ROOTDIR
    git clone https://github.com/flexi-framework/flexi-extensions.git
    cd flexi-extensions
    git checkout smartsim
    mkdir -p build && cd  build
    cmake .. -DLIBS_BUILD_HDF5=ON -DLIBS_USE_MPI=OFF -DLIBS_USE_SMARTREDIS=ON -DLIBS_USE_FFTW=ON -DPOSTI=OFF -DFLEXI_TESTCASE=hit -DFLEXI_NODETYPE=GAUSS-LOBATTO -DFLEXI_SPLIT_DG=ON -DFLEXI_EDDYVISCOSITY=ON
    make -j
    ```
* This compiles FLEXI without MPI and thus in its serial version. To enable MPI or to change the configuration of FLEXI, please see the [official documentation][userguide] of FLEXI.

### Install Relexi
* Finally, we can clone the Relexi repository.
    ```
    cd $ROOTDIR
    git clone https://github.com/flexi-framework/relexi.git
    ```

# Running the Code
* Relexi comes with some example setups to test if the code runs. Enter the directory of the first test case with
    ```
    cd relexi/examples/HIT_24_DOF/
    ```
* Open the ``prm.yaml`` file in a text editor of your choice. If you have installed the ``flexi`` binary not in the default path, adapt the path of the executable under ``library_path`` accordingly.
* Then you can start the training process with
    ```
    python3 ../../src/train.py prm.yaml
    ```
* You may also set the number of parallel environments by setting ``num_parallel_environments`` according to your local hardware resources.
* You can change the number of processors used for each FLEXI environment by setting ``num_procs_per_environment`` to the appropriate value. Please be aware that for using FLEXI in parallel, i.e. with more than 1 CPU core per environment, it has to be compiled with MPI. Please refer to the [FLEXI documentation](https://www.flexi-project.org/doc/userguide/userguide.pdf) for details.

# Results
* To visualize the results, Relexi uses the TensorBoard suite. After running the code, Relexi should create a directory ``logs``, where the model, training checkpoints and the training metrics are saved. Open it with
    ```
    tensorboard --logdir logs/
    ```
* Tensorboard then provides a URL that can be opened in the Browser.
* If the training is performed on a remote server, the port where TensorBoard sends its data has to be redirected to your local machine. If you use `ssh` to connect to the server, you can redirect the standard TensorBoard port (6006) with
    ```
    ssh -L 6006:127.0.0.1:6006 your_remote_server
    ```

[nrg]:       https://numericsresearchgroup.org/index.html
[flexi]:     https://numericsresearchgroup.org/flexi_index.html
[userguide]: https://numericsresearchgroup.org/userguide/userguide.pdf
