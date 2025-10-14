#include"SPGridMask.h"
#include<iostream>

unsigned long long Bit_Spread(unsigned long long mask, unsigned long long uldata)
{
    
    unsigned long long result = 0;

    if (0x0000000000000001UL & mask) result |= uldata & 0x0000000000000001UL; else uldata <<= 1;
    if (0x0000000000000002UL & mask) result |= uldata & 0x0000000000000002UL; else uldata <<= 1;
    if (0x0000000000000004UL & mask) result |= uldata & 0x0000000000000004UL; else uldata <<= 1;
    if (0x0000000000000008UL & mask) result |= uldata & 0x0000000000000008UL; else uldata <<= 1;
    if (0x0000000000000010UL & mask) result |= uldata & 0x0000000000000010UL; else uldata <<= 1;
    if (0x0000000000000020UL & mask) result |= uldata & 0x0000000000000020UL; else uldata <<= 1;
    if (0x0000000000000040UL & mask) result |= uldata & 0x0000000000000040UL; else uldata <<= 1;
    if (0x0000000000000080UL & mask) result |= uldata & 0x0000000000000080UL; else uldata <<= 1;
    if (0x0000000000000100UL & mask) result |= uldata & 0x0000000000000100UL; else uldata <<= 1;
    if (0x0000000000000200UL & mask) result |= uldata & 0x0000000000000200UL; else uldata <<= 1;
    if (0x0000000000000400UL & mask) result |= uldata & 0x0000000000000400UL; else uldata <<= 1;
    if (0x0000000000000800UL & mask) result |= uldata & 0x0000000000000800UL; else uldata <<= 1;
    if (0x0000000000001000UL & mask) result |= uldata & 0x0000000000001000UL; else uldata <<= 1;
    if (0x0000000000002000UL & mask) result |= uldata & 0x0000000000002000UL; else uldata <<= 1;
    if (0x0000000000004000UL & mask) result |= uldata & 0x0000000000004000UL; else uldata <<= 1;
    if (0x0000000000008000UL & mask) result |= uldata & 0x0000000000008000UL; else uldata <<= 1;
    if (0x0000000000010000UL & mask) result |= uldata & 0x0000000000010000UL; else uldata <<= 1;
    if (0x0000000000020000UL & mask) result |= uldata & 0x0000000000020000UL; else uldata <<= 1;
    if (0x0000000000040000UL & mask) result |= uldata & 0x0000000000040000UL; else uldata <<= 1;
    if (0x0000000000080000UL & mask) result |= uldata & 0x0000000000080000UL; else uldata <<= 1;
    if (0x0000000000100000UL & mask) result |= uldata & 0x0000000000100000UL; else uldata <<= 1;
    if (0x0000000000200000UL & mask) result |= uldata & 0x0000000000200000UL; else uldata <<= 1;
    if (0x0000000000400000UL & mask) result |= uldata & 0x0000000000400000UL; else uldata <<= 1;
    if (0x0000000000800000UL & mask) result |= uldata & 0x0000000000800000UL; else uldata <<= 1;
    if (0x0000000001000000UL & mask) result |= uldata & 0x0000000001000000UL; else uldata <<= 1;
    if (0x0000000002000000UL & mask) result |= uldata & 0x0000000002000000UL; else uldata <<= 1;
    if (0x0000000004000000UL & mask) result |= uldata & 0x0000000004000000UL; else uldata <<= 1;
    if (0x0000000008000000UL & mask) result |= uldata & 0x0000000008000000UL; else uldata <<= 1;
    if (0x0000000010000000UL & mask) result |= uldata & 0x0000000010000000UL; else uldata <<= 1;
    if (0x0000000020000000UL & mask) result |= uldata & 0x0000000020000000UL; else uldata <<= 1;
    if (0x0000000040000000UL & mask) result |= uldata & 0x0000000040000000UL; else uldata <<= 1;
    if (0x0000000080000000UL & mask) result |= uldata & 0x0000000080000000UL; else uldata <<= 1;
    if (0x0000000100000000UL & mask) result |= uldata & 0x0000000100000000UL; else uldata <<= 1;
    if (0x0000000200000000UL & mask) result |= uldata & 0x0000000200000000UL; else uldata <<= 1;
    if (0x0000000400000000UL & mask) result |= uldata & 0x0000000400000000UL; else uldata <<= 1;
    if (0x0000000800000000UL & mask) result |= uldata & 0x0000000800000000UL; else uldata <<= 1;
    if (0x0000001000000000UL & mask) result |= uldata & 0x0000001000000000UL; else uldata <<= 1;
    if (0x0000002000000000UL & mask) result |= uldata & 0x0000002000000000UL; else uldata <<= 1;
    if (0x0000004000000000UL & mask) result |= uldata & 0x0000004000000000UL; else uldata <<= 1;
    if (0x0000008000000000UL & mask) result |= uldata & 0x0000008000000000UL; else uldata <<= 1;
    if (0x0000010000000000UL & mask) result |= uldata & 0x0000010000000000UL; else uldata <<= 1;
    if (0x0000020000000000UL & mask) result |= uldata & 0x0000020000000000UL; else uldata <<= 1;
    if (0x0000040000000000UL & mask) result |= uldata & 0x0000040000000000UL; else uldata <<= 1;
    if (0x0000080000000000UL & mask) result |= uldata & 0x0000080000000000UL; else uldata <<= 1;
    if (0x0000100000000000UL & mask) result |= uldata & 0x0000100000000000UL; else uldata <<= 1;
    if (0x0000200000000000UL & mask) result |= uldata & 0x0000200000000000UL; else uldata <<= 1;
    if (0x0000400000000000UL & mask) result |= uldata & 0x0000400000000000UL; else uldata <<= 1;
    if (0x0000800000000000UL & mask) result |= uldata & 0x0000800000000000UL; else uldata <<= 1;
    if (0x0001000000000000UL & mask) result |= uldata & 0x0001000000000000UL; else uldata <<= 1;
    if (0x0002000000000000UL & mask) result |= uldata & 0x0002000000000000UL; else uldata <<= 1;
    if (0x0004000000000000UL & mask) result |= uldata & 0x0004000000000000UL; else uldata <<= 1;
    if (0x0008000000000000UL & mask) result |= uldata & 0x0008000000000000UL; else uldata <<= 1;
    if (0x0010000000000000UL & mask) result |= uldata & 0x0010000000000000UL; else uldata <<= 1;
    if (0x0020000000000000UL & mask) result |= uldata & 0x0020000000000000UL; else uldata <<= 1;
    if (0x0040000000000000UL & mask) result |= uldata & 0x0040000000000000UL; else uldata <<= 1;
    if (0x0080000000000000UL & mask) result |= uldata & 0x0080000000000000UL; else uldata <<= 1;
    if (0x0100000000000000UL & mask) result |= uldata & 0x0100000000000000UL; else uldata <<= 1;
    if (0x0200000000000000UL & mask) result |= uldata & 0x0200000000000000UL; else uldata <<= 1;
    if (0x0400000000000000UL & mask) result |= uldata & 0x0400000000000000UL; else uldata <<= 1;
    if (0x0800000000000000UL & mask) result |= uldata & 0x0800000000000000UL; else uldata <<= 1;
    if (0x1000000000000000UL & mask) result |= uldata & 0x1000000000000000UL; else uldata <<= 1;
    if (0x2000000000000000UL & mask) result |= uldata & 0x2000000000000000UL; else uldata <<= 1;
    if (0x4000000000000000UL & mask) result |= uldata & 0x4000000000000000UL; else uldata <<= 1;
    if (0x8000000000000000UL & mask) result |= uldata & 0x8000000000000000UL; else uldata <<= 1;

    return result;
}


unsigned long long SPGridMask::Linear_Offset(unsigned long long* mask, const int i, const int j, const int k)
{
	return Bit_Spread(mask[0], i) | Bit_Spread(mask[1], j) | Bit_Spread(mask[2], k);
}



void SPGridMask::Cal_elemMask() {
	unsigned int block_bits = 12 - struct_size;
	unsigned int block_zbits = block_bits / 3 + (block_bits % 3 > 0);
	unsigned int block_ybits = block_bits / 3 + (block_bits % 3 > 1);
	unsigned int block_xbits = block_bits / 3;

	elementMask_z = (((unsigned long long)1 << block_zbits) - 1) << element_size;
	elementMask_y = (((unsigned long long)1 << block_ybits) - 1) << (element_size + block_zbits);
	elementMask_x = (((unsigned long long)1 << block_xbits) - 1) << (element_size + block_zbits + block_ybits);
}

unsigned int SPGridMask::getElement_per_Block() {
	unsigned int block_bits = 12 - struct_size;
	return 1u << block_bits;
}

void SPGridMask::Cal_pageMask() {
	pageMask_z = (unsigned long long)(0x9249249249249249UL << (3 - struct_size % 3)) & 0xfffffffffffff000UL;
	pageMask_y = (unsigned long long)(0x2492492492492492UL << (3 - struct_size % 3)) & 0xfffffffffffff000UL;
	pageMask_x = (unsigned long long)(0x4924924924924924UL << (3 - struct_size % 3)) & 0xfffffffffffff000UL;
}

void SPGridMask::getMasks(unsigned long long* mk_x, unsigned long long* mk_y, unsigned long long* mk_z) {
	*mk_x = mask_x;
	*mk_y = mask_y;
	*mk_z = mask_z;
}

void SPGridMask::getMasks(unsigned long long* mk_a) {
	mk_a[0] = mask_x;
	mk_a[1] = mask_y;
	mk_a[2] = mask_z;
}

void SPGridMask::Cal_Masks() {
	Cal_elemMask();
	Cal_pageMask();
	mask_x = pageMask_x | elementMask_x;
	mask_y = pageMask_y | elementMask_y;
	mask_z = pageMask_z | elementMask_z;
}

