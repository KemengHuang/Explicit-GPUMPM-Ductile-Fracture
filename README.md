# Explicit-GPUMPM-Ductile-Fracture-HKM


DESCRIPTION
===========

This project is the source code of the paper: [A Novel Plastic Phase-Field Method for Ductile Fracture with GPU Optimization](https://doi.org/10.1111/cgf.14130)

This is an optimized GPU MPM framework designed for fracture simulation, featuring a clean codebase without unnecessary contents. This framework greatly aids in the development of GPU MPM applications.


Source code contributor: [Kemeng Huang](https://kemenghuang.github.io)


**Note: this software is released under the MPLv2.0 license. For commercial use, please email authors for negotiation.**


## BibTex 

Please cite the following paper if it helps. 

```
@article{ZhaoAndHuang,
  author       = {Zipeng Zhao and
                  Kemeng Huang and
                  Chen Li and
                  Changbo Wang and
                  Hong Qin},
  title        = {A Novel Plastic Phase-Field Method for Ductile Fracture with {GPU}
                  Optimization},
  journal      = {Comput. Graph. Forum},
  volume       = {39},
  number       = {7},
  pages        = {105--117},
  year         = {2020},
  url          = {https://doi.org/10.1111/cgf.14130}
}
```


Requirements
============

Hardware requirements: Nvidia GPUs

Support platforms: Windows, Linux 

## Dependencies

| Name                                   | Version | Usage                                               | Import         |
| -------------------------------------- | ------- | --------------------------------------------------- | -------------- |
| cuda                                   | >=11.0  | GPU programming                                     | system install |
| eigen3                                 | 3.4.0   | matrix calculation                                  | package        |
| freeglut                               | 3.4.0   | visualization                                       | package        |
| glew                                   | 2.2.0#3 | visualization                                       | package        |

### linux

We use CMake to build the project.

```bash
sudo apt install libglew-dev freeglut3-dev libeigen3-dev
```


### Windows
We use [vcpkg](https://github.com/microsoft/vcpkg) to manage the libraries we need and use CMake to build the project. The simplest way to let CMake detect vcpkg is to set the system environment variable `CMAKE_TOOLCHAIN_FILE` to `(YOUR_VCPKG_PARENT_FOLDER)/vcpkg/scripts/buildsystems/vcpkg.cmake`

```shell
vcpkg install eigen3 freeglut glew
```





EXTERNAL CREDITS
================

This work refers to the code from the following open-source repositories:

zpc https://github.com/KemengHuang/zpc

GPUMPM https://github.com/kuiwuchn/GPUMPM

