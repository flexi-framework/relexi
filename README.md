[![logo](https://numericsresearchgroup.org/images/icons/relexi.svg "RELEXI")][relexi_github]

[![license](https://img.shields.io/github/license/flexi-framework/relexi.svg?maxAge=2592000 "GPL-3.0 License")](LICENSE.md)
[![doi](https://img.shields.io/badge/DOI-10.1016/j.simpa.2022.100422-blue "DOI")](https://doi.org/10.1016/j.simpa.2022.100422)
[![docsbuild](https://github.com/flexi-framework/relexi/actions/workflows/docs.yml/badge.svg "Documentation")][relexi_docs]
[![pypi](https://img.shields.io/pypi/v/relexi "PyPI")](https://pypi.org/project/relexi/)
[![github](https://img.shields.io/badge/View%20on%20GitHub-%23121011.svg?logo=github&logoColor=white "GitHub")][relexi_github]

# About Relexi
Relexi is a Reinforcement Learning (RL) framework developed for the high-order HPC flow solver [FLEXI][flexi].
However, Relexi is developed with modularity in mind and allows to used with other HPC solvers as well.
Relexi builds upon TensorFlow and its RL extension TF-Agents.
For the efficient communication, data handling and the managment of the simulations runs on HPC systems, Relexi uses the SmartSim package with its SmartRedis communication clients.
For details on its scaling behavior, suitability for HPC and for use cases, please see
* [Kurz, M., Offenhäuser, P., Viola, D., Resch, M., & Beck, A. (2022). Relexi—A scalable open source reinforcement learning framework for high-performance computing. Software Impacts, 14, 100422.](https://doi.org/10.1016/j.simpa.2022.100422)
* [Kurz, M., Offenhäuser, P., Viola, D., Shcherbakov, O., Resch, M., & Beck, A. (2022). Deep Reinforcement Learning for Computational Fluid Dynamics on HPC Systems. Journal of Computational Science, 65, 101884.](https://doi.org/10.1016/j.jocs.2022.101884)
* [Kurz, M., Offenhäuser, P., & Beck, A. (2023). Deep reinforcement learning for turbulence modeling in large eddy simulations. International Journal of Heat and Fluid Flow, 99, 109094.](https://doi.org/10.1016/j.ijheatfluidflow.2022.109094)
* [Beck, A., & Kurz, M. (2023). Toward discretization-consistent closure schemes for large eddy simulation using reinforcement learning. Physics of Fluids, 35(12), 125122.](https://doi.org/10.1063/5.0176223)

This is a scientific project.
If you use Relexi or find it helpful, please cite the project using a suitable reference from the list above referring to either the general Relexi project, its HPC aspects or its application for scientific modeling tasks, respectively.

# Installation
The following quick start guide covers a standard installation of the Relexi framework.

### Dependencies
Relexi has a variety of dependencies.
The main dependencies of Relexi are listed in the following with their supported version.

| Package          | Version         | Note                                    |
|:-----------------|----------------:|:----------------------------------------|
| Python           |          ≥3.9   |                                         |
| CMake            |          ≥3.0   |                                         |
| Make             |          ≥4.0   |                                         |
| gcc-fortran      |          ≥9.4   | gcc 10 not supported! (gcc ≥11 is fine) |
| gcc              |          ≥9.4   |                                         |
| gcc-c++          |          ≥9.4   |                                         |

Be aware that the major dependencies (SmartSim, TensorFlow, FLEXI) might have a more expansive dependency tree, for which we refer the user to the corresponding documentations for details.

### Prerequisites
Open a terminal and change into the directory where you want to install Relexi and its dependecies.
It is highly recommended to use some form of virtual environment for the installation.
You can use create and activate a new environment using `virtualenv` via
```bash
pip install virtualenv
virtualenv env_relexi
source env_relexi/bin/activate
```

### Install Relexi
Relexi can be installed most simply with `pip` via
```bash
pip install relexi
```

Alternatively, it can be installed as editable package from the repository with
```bash
git clone https://github.com/flexi-framework/relexi.git
pip install -e relexi/
```

### Build SmartSim
After installing the `smartsim` package via pip, its database backend has to be installed with its dependencies via the `smart` commandline tool:
```bash
smart clobber && smart clean
smart build --no_pt
```

### Install FLEXI
Clone the required version of FLEXI from GitHub and build it with the standard compile flags
```bash
git clone --branch smartsim --depth 1 https://github.com/flexi-framework/flexi-extensions.git
cd flexi-extensions
mkdir -p build && cd build
cmake .. -DLIBS_BUILD_HDF5=ON -DLIBS_USE_MPI=OFF -DLIBS_BUILD_SMARTREDIS=ON -DLIBS_USE_SMARTREDIS=ON -DLIBS_USE_FFTW=ON -DPOSTI=OFF -DFLEXI_TESTCASE=hit -DFLEXI_NODETYPE=GAUSS-LOBATTO -DFLEXI_SPLIT_DG=ON -DFLEXI_EDDYVISCOSITY=ON
make -j
cd ../../
```
Note that in this configuration FLEXI tries to install all its dependencies automatically, which can require several minutes.
If HDF5 is available on the system, the compile time can be reduced significantly by switching off the corresponding flag in the CMake configuration.
Moreover, this configuration compiles FLEXI without MPI and thus in its serial version.
To enable MPI or to change the configuration of FLEXI, please see its [official documentation][userguide] for more details.

# Running the Code
Relexi comes with some example setups to verify that it is correctly installed.
Enter the directory of the first test case and run Relexi using its CLI.
```bash
cd relexi/examples/HIT_24_DOF/
relexi prm.yaml
```
The file ``prm.yaml`` contains the configuration for the reinforcement learning training.
It can be adapted using the text editor of your choice.
If you have installed the ``flexi`` binary not in the default path, adapt the path of the executable under ``executabl_path`` accordingly.
You may also set the number of parallel environments by setting ``num_parallel_environments`` according to your local hardware resources.
The number of processors used for each FLEXI environment can also be changed by setting ``num_procs_per_environment`` to the appropriate value.
Be aware that for using FLEXI in parallel, i.e. with more than 1 CPU core per environment, it has to be compiled with MPI.
Please refer to the [FLEXI documentation][userguide] for details.

# Results
To visualize the results, Relexi uses the TensorBoard suite.
After running the code, Relexi should create a directory ``logs``, where the model, training checkpoints and the training metrics are saved.
Open it with
```bash
tensorboard --logdir logs/
```
Tensorboard then provides a URL that can be opened in the Browser.
If the training is performed on a remote server, the port where TensorBoard sends its data has to be redirected to your local machine. If you use `ssh` to connect to the server, you can redirect the standard TensorBoard port (6006) with
```bash
ssh -L 6006:127.0.0.1:6006 your_remote_server
```

# Documentation
The documentation of Relexi can be found [here][relexi_docs].
It is built with the [`pdoc`](https://github.com/mitmproxy/pdoc) package, which is included in the [`requirements.txt`](requirements.txt) and thus is already installed with Relexi.
To build the documentation yourself, execute
```bash
cd docs
bash build_docs.sh
```
Open the resulting `index.html` with your browser.

# Testing
A suite of unit tests is implemented for Relexi using the [`pytest`](https://docs.pytest.org/) testing environment. To run the tests, simply execute in the root directory
```bash
pytest
```

[nrg]:           https://numericsresearchgroup.org/index.html
[flexi]:         https://numericsresearchgroup.org/flexi_index.html
[userguide]:     https://numericsresearchgroup.org/userguide/userguide.pdf
[relexi_docs]:   https://flexi-framework.github.io/relexi/relexi.html
[relexi_github]: https://github.com/flexi-framework/relexi
