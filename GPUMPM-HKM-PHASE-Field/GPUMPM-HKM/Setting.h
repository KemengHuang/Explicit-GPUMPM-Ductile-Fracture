#pragma once


using T = float;
constexpr const T rateSize = 1.f;
constexpr const unsigned int N = 128 * rateSize;
constexpr const T dx = 1.f / N;
constexpr const T DX = dx;// *rateSize;
constexpr const T one_over_dx = N;
constexpr const T D_inverse = 4.f * one_over_dx * one_over_dx;
constexpr const unsigned int space_page_num = (N / 4 + 2) * (N / 4 + 2) * (N / 4 + 2);
constexpr int Dim = 3;
constexpr int pretype_threshold = 2500000;

constexpr const T MEMORY_SCALE = 0.1f;
// 0: explicit 1: implicit
#define MPM_SIM_TYPE 0

//0��apic_conflict_free  1: apic 2: mls
#define TRANSFER_SCHEME 1

// 0: 7M cube 1: two dragons collide
#define GEOMETRY_TYPE 1

// the amount of grid being used

constexpr const int typem = 0;

#define PARA_GAMA 0.01f
#define PARA_P 0.0001f

struct vector3T {
    T x;
    T y;
    T z;
};
struct CH_STRUCT {
    unsigned flags;
    T ch0;
    T ch1;
    T ch2;
    T ch3;
    T ch4;
    T ch5;
    T ch6;
    T ch7;
    T ch8;
    T ch9;
    T ch10;
    T ch11;
    T ch12;
    T ch13;
    T ch14;
};



