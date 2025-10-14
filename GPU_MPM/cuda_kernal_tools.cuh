//
// cuda_kernel_tools.cuh
// GPUMPM_HKM
//
// created by Kemeng Huang on 2020/05/23
// Copyright (c) 2020 Kemeng Huang. All rights reserved.
//


#ifndef __CUDA_KERNAL_TOOLS_CUH_
#define __CUDA_KERNAL_TOOLS_CUH_


void Allocate_Prefix_Sum_RecursiveMem_Int(int num);

void Free_Prefix_Sum_RecursiveMem_Int();

void In_Prefix_Sum_Recursive_Int(int offsetI, const int& blockNum, int* inputArray, const int& Number, const  int& offset);

void Ex_Prefix_Sum_Int(int* inputArray, int* outputArray, const int& Number);

void In_Prefix_Sum_Int(int* inputArray, int* outputArray, const int& Number);



#endif