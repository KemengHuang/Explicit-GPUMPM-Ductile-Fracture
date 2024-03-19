# Explicit-GPUMPM-Ductile-Fracture-HKM


DESCRIPTION
===========

This project is the source code of the paper: [A Novel Plastic Phase-Field Method for Ductile Fracture with GPU Optimization](https://doi.org/10.1111/cgf.14130)

This is an optimized GPU MPM framework designed for fracture simulation, featuring a clean codebase without unnecessary contents. This framework greatly aids in the development of GPU MPM applications, as it can be compiled within 10 seconds. In terms of performance, it stands as one of the most efficient single GPU MPM frameworks available.


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

### linux
*CUDA, glew, freeglut


`sudo apt install libglew-dev freeglut3-dev


Once the dependencies are installed, proceed to create a CMakeLists.txt file to facilitate building the project.

### Windows
Please ensure that CUDA and Visual Studio are installed on your system. Modify the CUDA version in the xxx.vcxproj file to match the version you have installed, and then simply open the project using Visual Studio.





EXTERNAL CREDITS
================

This work utilizes the following code, which have been included here for convenience:
Copyrights are retained by the original authors.

zpc https://github.com/KemengHuang/zpc

GPUMPM https://github.com/kuiwuchn/GPUMPM

