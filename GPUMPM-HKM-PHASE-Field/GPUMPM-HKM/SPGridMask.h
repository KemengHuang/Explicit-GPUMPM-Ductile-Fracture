#pragma once
class SPGridMask {
public:
	SPGridMask(unsigned long long strc_s, unsigned long long elem_s) {
		struct_size = strc_s;
		element_size = elem_s;
	}
	~SPGridMask() {};
	static unsigned long long Linear_Offset(unsigned long long* mask, const int i, const int j, const int k);
	void getMasks(unsigned long long* mk_x, unsigned long long* mk_y, unsigned long long* mk_z);
	void getMasks(unsigned long long *mk_a);
	void Cal_Masks();
	unsigned int getElement_per_Block();
private:
	void Cal_elemMask();
	void Cal_pageMask();
	unsigned long long struct_size;
	unsigned long long element_size;
	unsigned long long elementMask_x;
	unsigned long long elementMask_y;
	unsigned long long elementMask_z;
	unsigned long long pageMask_x;
	unsigned long long pageMask_y;
	unsigned long long pageMask_z;
	unsigned long long mask_x;
	unsigned long long mask_y;
	unsigned long long mask_z;

};

