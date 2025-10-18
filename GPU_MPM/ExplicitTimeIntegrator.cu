#include"TimeIntegrator.cuh"
#include"cuda_tools.h"
//#include <cuda_runtime.h>
#include"P2GKernal.cuh"
#include"G2PKernal.cuh"
#include<iostream>
#include<fstream>
#include "QRSVD.hpp"



__global__ void calcIndex(
    const int numCell, const T one_over_dx, const int* d_cell_first_particles_indices,
    const T** d_sorted_positions, int** smallest_nodes) {
    int cellid = blockDim.x * blockIdx.x + threadIdx.x;
    if (cellid >= numCell) return;
    smallest_nodes[0][cellid] = (int)((d_sorted_positions[0][d_cell_first_particles_indices[cellid]]) * one_over_dx + 0.5f) - 1;
    smallest_nodes[1][cellid] = (int)((d_sorted_positions[1][d_cell_first_particles_indices[cellid]]) * one_over_dx + 0.5f) - 1;
    smallest_nodes[2][cellid] = (int)((d_sorted_positions[2][d_cell_first_particles_indices[cellid]]) * one_over_dx + 0.5f) - 1;
}

__global__ void computeContributionFixedCorotated(const int numParticle, const T* d_F, const T lambda, const T mu, const T volume, T* d_contribution) {
    int parid = blockDim.x * blockIdx.x + threadIdx.x;
    if (parid >= numParticle) return;

    double F[9];
    F[0] = d_F[parid + 0 * numParticle]; F[1] = d_F[parid + 1 * numParticle]; F[2] = d_F[parid + 2 * numParticle];
    F[3] = d_F[parid + 3 * numParticle]; F[4] = d_F[parid + 4 * numParticle]; F[5] = d_F[parid + 5 * numParticle];
    F[6] = d_F[parid + 6 * numParticle]; F[7] = d_F[parid + 7 * numParticle]; F[8] = d_F[parid + 8 * numParticle];

    double U[9]; double S[3]; double V[9];
    __GEIGEN__::math::msvd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);

    //
    T J = S[0] * S[1] * S[2]; T scaled_mu = 2.f * mu; T scaled_lambda = lambda * (J - 1.f);
    T P_hat[3]; P_hat[0] = scaled_mu * (S[0] - 1.f) + scaled_lambda * (S[1] * S[2]); P_hat[1] = scaled_mu * (S[1] - 1.f) + scaled_lambda * (S[0] * S[2]); P_hat[2] = scaled_mu * (S[2] - 1.f) + scaled_lambda * (S[0] * S[1]);
    // 

    T P[9];
    P[0] = P_hat[0] * U[0] * V[0] + P_hat[1] * U[3] * V[3] + P_hat[2] * U[6] * V[6]; P[1] = P_hat[0] * U[1] * V[0] + P_hat[1] * U[4] * V[3] + P_hat[2] * U[7] * V[6]; P[2] = P_hat[0] * U[2] * V[0] + P_hat[1] * U[5] * V[3] + P_hat[2] * U[8] * V[6];
    P[3] = P_hat[0] * U[0] * V[1] + P_hat[1] * U[3] * V[4] + P_hat[2] * U[6] * V[7]; P[4] = P_hat[0] * U[1] * V[1] + P_hat[1] * U[4] * V[4] + P_hat[2] * U[7] * V[7]; P[5] = P_hat[0] * U[2] * V[1] + P_hat[1] * U[5] * V[4] + P_hat[2] * U[8] * V[7];
    P[6] = P_hat[0] * U[0] * V[2] + P_hat[1] * U[3] * V[5] + P_hat[2] * U[6] * V[8]; P[7] = P_hat[0] * U[1] * V[2] + P_hat[1] * U[4] * V[5] + P_hat[2] * U[7] * V[8]; P[8] = P_hat[0] * U[2] * V[2] + P_hat[1] * U[5] * V[5] + P_hat[2] * U[8] * V[8];

    d_contribution[parid + 0 * numParticle] = (P[0] * F[0] + P[3] * F[3] + P[6] * F[6]) * volume; d_contribution[parid + 1 * numParticle] = (P[1] * F[0] + P[4] * F[3] + P[7] * F[6]) * volume; d_contribution[parid + 2 * numParticle] = (P[2] * F[0] + P[5] * F[3] + P[8] * F[6]) * volume;
    d_contribution[parid + 3 * numParticle] = (P[0] * F[1] + P[3] * F[4] + P[6] * F[7]) * volume; d_contribution[parid + 4 * numParticle] = (P[1] * F[1] + P[4] * F[4] + P[7] * F[7]) * volume; d_contribution[parid + 5 * numParticle] = (P[2] * F[1] + P[5] * F[4] + P[8] * F[7]) * volume;
    d_contribution[parid + 6 * numParticle] = (P[0] * F[2] + P[3] * F[5] + P[6] * F[8]) * volume; d_contribution[parid + 7 * numParticle] = (P[1] * F[2] + P[4] * F[5] + P[7] * F[8]) * volume; d_contribution[parid + 8 * numParticle] = (P[2] * F[2] + P[5] * F[5] + P[8] * F[8]) * volume;
}

__global__ void computePhase_field_vol(const int numParticle, const T* d_F, T vol0, T* particle_vol) {
    int parid = blockDim.x * blockIdx.x + threadIdx.x;
    if (parid >= numParticle) return;

    T F[9];
    F[0] = d_F[parid + 0 * numParticle]; F[1] = d_F[parid + 1 * numParticle]; F[2] = d_F[parid + 2 * numParticle];
    F[3] = d_F[parid + 3 * numParticle]; F[4] = d_F[parid + 4 * numParticle]; F[5] = d_F[parid + 5 * numParticle];
    F[6] = d_F[parid + 6 * numParticle]; F[7] = d_F[parid + 7 * numParticle]; F[8] = d_F[parid + 8 * numParticle];

    T determinant = F[0] * F[4] * F[8] + F[3] * F[7] * F[2] + F[6] * F[1] * F[5] - F[2] * F[4] * F[6] - F[1] * F[3] * F[8] - F[0] * F[5] * F[7];

    particle_vol[parid] = vol0 * (determinant);
    
}

__device__ T R(const T& alpha, const T& yield_stress, const T& harden) {
    //TODO: input some para.
    //T harden = 3.f;
    return sqrt(2.f / 3.f) * yield_stress - (sqrt(2.f / 3.f) * yield_stress - yield_stress) / exp(harden * alpha);
}


__device__ void Inverse(double* matIn, double* mat) {

    for (int i = 0; i < 9; i++) mat[i] = matIn[i];

    int swapR[3], swapC[3];
    int pivot[3] = { 0 };
    for (int i = 0; i < 3; ++i) {
        int pr, pc;
        double maxValue = 0;
        for (int j = 0; j < 3; ++j) {
            if (pivot[j] != 1) {
                for (int k = 0; k < 3; ++k) {
                    if (pivot[k] == 0) {
                        if (mat[j * 3 + k] > maxValue) {
                            maxValue = mat[j * 3 + k];
                            pr = j;
                            pc = k;
                        }
                    }
                }
            }
        }
        if (pr != pc) {
            double pv;
            for (int j = 0; j < 3; ++j) {
                pv = mat[pr * 3 + j];
                mat[pr * 3 + j] = mat[pc * 3 + j];
                mat[pc * 3 + j] = pv;
            }
        }
        swapC[i] = pc;
        swapR[i] = pr;
        ++pivot[i];
        double inv = 1.f / mat[pc * 3 + pc];
        mat[pc * 3 + pc] = 1;
        for (int j = 0; j < 3; ++j) {
            mat[pc * 3 + j] *= inv;
        }
        for (int j = 0; j < 3; ++j) {
            if (j != pc) {
                double powerRatio = mat[j * 3 + pc];
                mat[j * 3 + pc] = 0.f;
                for (int k = 0; k < 3; ++k) {
                    mat[j * 3 + k] -= mat[pc * 3 + k] * powerRatio;
                }
            }
        }
    }
    for (int i = 0; i < 3; ++i) {
        if (swapR[i] != swapC[i]) {
            double pv;
            for (int j = 0; j < 3; ++j) {
                pv = mat[j * 3 + swapC[i]];
                mat[j * 3 + swapC[i]] = mat[j * 3 + swapR[i]];
                mat[j * 3 + swapR[i]] = pv;
            }
        }
    }
}


__global__ void applyVonMises(T* p_g, T* p_alpha, const T lambda, const T mu, T dx, T* FP, const int numParticle, T* d_F, T* d_maxPsi, const T* d_phase, const T kappa, const T volume, T* d_contribution) {
    int parid = blockDim.x * blockIdx.x + threadIdx.x;
    if (parid >= numParticle) return;
    T g = p_g[parid];
    T alpha = p_alpha[parid];

    double F[9];
    F[0] = d_F[parid + 0 * numParticle]; F[1] = d_F[parid + 1 * numParticle]; F[2] = d_F[parid + 2 * numParticle];
    F[3] = d_F[parid + 3 * numParticle]; F[4] = d_F[parid + 4 * numParticle]; F[5] = d_F[parid + 5 * numParticle];
    F[6] = d_F[parid + 6 * numParticle]; F[7] = d_F[parid + 7 * numParticle]; F[8] = d_F[parid + 8 * numParticle];

    double U[9]; double S[3]; double V[9];
    __GEIGEN__::math::msvd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);

    double C[9];
    C[0] = 2 * mu + lambda; C[1] = lambda;          C[2] = lambda;
    C[3] = lambda;          C[4] = 2 * mu + lambda; C[5] = lambda;
    C[6] = lambda;          C[7] = lambda;          C[8] = 2 * mu + lambda;

    T eps[3];
    if ((S[0]) > 0.f) {
        eps[0] = log((S[0]));
    }
    else {
        eps[0] = 0;
    }
    if ((S[1]) > 0.f) {
        eps[1] = log((S[1]));
    }
    else {
        eps[1] = 0;
    }
    if ((S[2]) > 0.f) {
        eps[2] = log((S[2]));
    }
    else {
        eps[2] = 0;
    }

    T tau_tr[3];
    tau_tr[0] = C[0] * eps[0] + C[1] * eps[1] + C[2] * eps[2];
    tau_tr[1] = C[3] * eps[0] + C[4] * eps[1] + C[5] * eps[2];
    tau_tr[2] = C[6] * eps[0] + C[7] * eps[1] + C[8] * eps[2];

    T vefy = tau_tr[0] * tau_tr[0] + tau_tr[1] * tau_tr[1] + tau_tr[2] * tau_tr[2];
    T tau_tr_F;
    tau_tr_F = sqrt(vefy);

    T tau_tr_sum = tau_tr[0] + tau_tr[1] + tau_tr[2];

    T tau_dev_tr[3];
    T divider = (tau_tr_sum / 3.f);
    tau_dev_tr[0] = g * (tau_tr[0] - divider);
    tau_dev_tr[1] = g * (tau_tr[1] - divider);
    tau_dev_tr[2] = g * (tau_tr[2] - divider);

    T yield_stress = 800;
    T harden = 3.f;
    T parameter1 = 2.f / 3.f;

    T y = tau_tr_F - g * R(alpha, yield_stress, harden) * sqrt(parameter1);

    double alpha_c = 1;
    if (true && y >= (T)1e-12)
    {
        T tau_tr_trace = tau_tr[0] + tau_tr[1] + tau_tr[2];
        T dlambda_init = 0.001;
        T dlambda = 0.f;

        if (dlambda < 0) {
            dlambda = 0;
        }

        //if (abs(tau_tr_sum) < 1e-4)
        //{
        //    dlambda = 100.f;
        //}
        //else
        {
            T bss = (parameter1 * g * tau_tr_sum);
            if (bss > 0.f) {
                dlambda = y / bss;
            }
        }

        if (dlambda > 100.f)
        {
            dlambda = 100.f;
        }

        if (alpha < alpha_c) {
            alpha += sqrt(parameter1) * dlambda;
            p_alpha[parid] = alpha;
        }
        if (alpha > alpha_c) {
            p_alpha[parid] = alpha_c;
        }

        T tau_normal = tau_dev_tr[0] * tau_dev_tr[0] + tau_dev_tr[1] * tau_dev_tr[1] + tau_dev_tr[2] * tau_dev_tr[2];

        T tau_dev_tr_norm;
     
        tau_dev_tr_norm = sqrt(tau_normal);

        //if (abs(tau_dev_tr_norm) < 1e-2)
        //{
        //    tau_dev_tr_norm = 0.01f;
        //}

        T tau_dev_nn[3];
        T taudiv = (1.f / tau_dev_tr_norm) * parameter1 * g * tau_tr_sum * dlambda;
        tau_dev_nn[0] = tau_dev_tr[0] - tau_dev_tr[0] * taudiv;
        tau_dev_nn[1] = tau_dev_tr[1] - tau_dev_tr[1] * taudiv;
        tau_dev_nn[2] = tau_dev_tr[2] - tau_dev_tr[2] * taudiv;

        T tau_nn[3];
        T parameter2 = (1.0 / 3.0) * tau_tr_sum;
        tau_nn[0] = tau_dev_nn[0] + parameter2;
        tau_nn[1] = tau_dev_nn[1] + parameter2;
        tau_nn[2] = tau_dev_nn[2] + parameter2;

        double D[9];
        Inverse(C, D);

        T eps_nn[3];
        eps_nn[0] = D[0] * tau_nn[0] + D[1] * tau_nn[1] + D[2] * tau_nn[2];
        eps_nn[1] = D[3] * tau_nn[0] + D[4] * tau_nn[1] + D[5] * tau_nn[2];
        eps_nn[2] = D[6] * tau_nn[0] + D[7] * tau_nn[1] + D[8] * tau_nn[2];


        T verify = eps_nn[0] * eps_nn[0] + eps_nn[1] * eps_nn[1] + eps_nn[2] * eps_nn[2];

        T normS = sqrt(verify);


        for (int i = 0; i < 3; i++) {
            if (eps_nn[i] > 3) {
                eps_nn[i] = eps_nn[i] * (3.f / normS);
            }
        }

        S[0] = exp(eps_nn[0]); S[1] = exp(eps_nn[1]); S[2] = exp(eps_nn[2]);


        F[0] = S[0] * U[0] * V[0] + S[1] * U[3] * V[3] + S[2] * U[6] * V[6]; F[1] = S[0] * U[1] * V[0] + S[1] * U[4] * V[3] + S[2] * U[7] * V[6]; F[2] = S[0] * U[2] * V[0] + S[1] * U[5] * V[3] + S[2] * U[8] * V[6];
        F[3] = S[0] * U[0] * V[1] + S[1] * U[3] * V[4] + S[2] * U[6] * V[7]; F[4] = S[0] * U[1] * V[1] + S[1] * U[4] * V[4] + S[2] * U[7] * V[7]; F[5] = S[0] * U[2] * V[1] + S[1] * U[5] * V[4] + S[2] * U[8] * V[7];
        F[6] = S[0] * U[0] * V[2] + S[1] * U[3] * V[5] + S[2] * U[6] * V[8]; F[7] = S[0] * U[1] * V[2] + S[1] * U[4] * V[5] + S[2] * U[7] * V[8]; F[8] = S[0] * U[2] * V[2] + S[1] * U[5] * V[5] + S[2] * U[8] * V[8];

    }

    d_F[parid + 0 * numParticle] = F[0]; d_F[parid + 1 * numParticle] = F[1]; d_F[parid + 2 * numParticle] = F[2];
    d_F[parid + 3 * numParticle] = F[3]; d_F[parid + 4 * numParticle] = F[4]; d_F[parid + 5 * numParticle] = F[5];
    d_F[parid + 6 * numParticle] = F[6]; d_F[parid + 7 * numParticle] = F[7]; d_F[parid + 8 * numParticle] = F[8];

    T be_bar[3];
    T J = S[0] * S[1] * S[2];
    T J_bar = pow(J, -parameter1);

    be_bar[0] = J_bar * S[0] * S[0]; /**/ be_bar[1] = J_bar * S[1] * S[1]; /**/ be_bar[2] = J_bar * S[2] * S[2];
    T be_bar_sum = be_bar[0] + be_bar[1] + be_bar[2];
    T ln_J;

    if (J > 0.f) {
        ln_J = log((J));
    }
    else {
        ln_J = 0.f;
    }

    // 1. update Psi
    T Psi_vol = (kappa * 0.5) * (((J * J - 1.f) * 0.5f) - ln_J);
    T Psi_dev = (mu * 0.5) * (be_bar[0] + be_bar[1] + be_bar[2] - 3.f);

    T Psi_pos = 0.f;
    T Psi_neg = 0.f;

    if (J < 1.f) {
        Psi_pos = Psi_dev;
        Psi_neg = Psi_vol;
    }
    else
    {
        Psi_pos = Psi_dev + Psi_vol;
    }
    if (Psi_pos > d_maxPsi[parid] && Psi_pos < (T)1e6)
    {
        d_maxPsi[parid] = Psi_pos;
    }
    T G_c_0 = 3.f;
    T p = alpha / 2;
    T G_C;
    double ps = 0.01;
    if (p < ps)
    {
        G_C = G_c_0;
    }
    else
    {
        double residual_c = 0.1;
        G_C = G_c_0 * (((1.f - residual_c) * exp(ps - p)) + residual_c);
    }

    float theta = 0.0001f;
    T L0 = dx * 0.5;
    //float beta = 15;

    FP[parid] = 4 * L0 * (1 - theta) * d_maxPsi[parid] * (1.f / G_C) + 1.f;

    T tau_vol[3];
    parameter1 = (kappa * 0.5) * (J * J - 1.f);
    tau_vol[0] = parameter1; tau_vol[1] = parameter1; tau_vol[2] = parameter1;

    T tau_dev[3];
    parameter1 = (1.0 / 3.0) * be_bar_sum;
    tau_dev[0] = mu * (be_bar[0] - parameter1); tau_dev[1] = mu * (be_bar[1] - parameter1); tau_dev[2] = mu * (be_bar[2] - parameter1);

    T P_hat[3];


    g = d_phase[parid] * d_phase[parid] * (1 - theta) + theta;
    p_g[parid] = g;
    if (J < 1.f)
    {
        P_hat[0] = g * tau_dev[0] + tau_vol[0]; P_hat[1] = g * tau_dev[1] + tau_vol[1]; P_hat[2] = g * tau_dev[2] + tau_vol[2];
    }
    else
    {
        P_hat[0] = g * (tau_dev[0] + tau_vol[0]); P_hat[1] = g * (tau_dev[1] + tau_vol[1]); P_hat[2] = g * (tau_dev[2] + tau_vol[2]);
    }

    T P[9];
    P[0] = P_hat[0] * U[0] * V[0] + P_hat[1] * U[3] * V[3] + P_hat[2] * U[6] * V[6]; P[1] = P_hat[0] * U[1] * V[0] + P_hat[1] * U[4] * V[3] + P_hat[2] * U[7] * V[6]; P[2] = P_hat[0] * U[2] * V[0] + P_hat[1] * U[5] * V[3] + P_hat[2] * U[8] * V[6];
    P[3] = P_hat[0] * U[0] * V[1] + P_hat[1] * U[3] * V[4] + P_hat[2] * U[6] * V[7]; P[4] = P_hat[0] * U[1] * V[1] + P_hat[1] * U[4] * V[4] + P_hat[2] * U[7] * V[7]; P[5] = P_hat[0] * U[2] * V[1] + P_hat[1] * U[5] * V[4] + P_hat[2] * U[8] * V[7];
    P[6] = P_hat[0] * U[0] * V[2] + P_hat[1] * U[3] * V[5] + P_hat[2] * U[6] * V[8]; P[7] = P_hat[0] * U[1] * V[2] + P_hat[1] * U[4] * V[5] + P_hat[2] * U[7] * V[8]; P[8] = P_hat[0] * U[2] * V[2] + P_hat[1] * U[5] * V[5] + P_hat[2] * U[8] * V[8];

    d_contribution[parid + 0 * numParticle] = (P[0] * F[0] + P[3] * F[3] + P[6] * F[6]) * volume; d_contribution[parid + 1 * numParticle] = (P[1] * F[0] + P[4] * F[3] + P[7] * F[6]) * volume; d_contribution[parid + 2 * numParticle] = (P[2] * F[0] + P[5] * F[3] + P[8] * F[6]) * volume;
    d_contribution[parid + 3 * numParticle] = (P[0] * F[1] + P[3] * F[4] + P[6] * F[7]) * volume; d_contribution[parid + 4 * numParticle] = (P[1] * F[1] + P[4] * F[4] + P[7] * F[7]) * volume; d_contribution[parid + 5 * numParticle] = (P[2] * F[1] + P[5] * F[4] + P[8] * F[7]) * volume;
    d_contribution[parid + 6 * numParticle] = (P[0] * F[2] + P[3] * F[5] + P[6] * F[8]) * volume; d_contribution[parid + 7 * numParticle] = (P[1] * F[2] + P[4] * F[5] + P[7] * F[8]) * volume; d_contribution[parid + 8 * numParticle] = (P[2] * F[2] + P[5] * F[5] + P[8] * F[8]) * volume;
}

__global__ void computeContributionNeoHookean(T dx, T* FP, const int numParticle, T* d_F, T* d_maxPsi, const T* d_phase, const T kappa, const T mu, const T volume, T* d_contribution) {
    int parid = blockDim.x * blockIdx.x + threadIdx.x;
    if (parid >= numParticle) return;

    double F[9];
    F[0] = d_F[parid + 0 * numParticle]; F[1] = d_F[parid + 1 * numParticle]; F[2] = d_F[parid + 2 * numParticle];
    F[3] = d_F[parid + 3 * numParticle]; F[4] = d_F[parid + 4 * numParticle]; F[5] = d_F[parid + 5 * numParticle];
    F[6] = d_F[parid + 6 * numParticle]; F[7] = d_F[parid + 7 * numParticle]; F[8] = d_F[parid + 8 * numParticle];

    double U[9]; double S[3]; double V[9];

    __GEIGEN__::math::msvd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);

    //if (S[0] < 0.0001f)S[0] = 0.0001f;
    //if (S[1] < 0.0001f)S[1] = 0.0001f;
    //if (S[2] < 0.0001f)S[2] = 0.0001f;

    //
    T be_bar[3];
    T J = S[0] * S[1] * S[2];
    T J_bar = __powf(J, -2.f / 3.f);

    be_bar[0] = J_bar * S[0] * S[0]; /**/ be_bar[1] = J_bar * S[1] * S[1]; /**/ be_bar[2] = J_bar * S[2] * S[2];
    T be_bar_sum = be_bar[0] + be_bar[1] + be_bar[2];

    T ln_J = log(J);

    // 1. update Psi
    T Psi_vol = kappa / 2.f * ((J - 1.f) / 2.f - ln_J);
    T Psi_dev = mu / 2.f * (be_bar[0] + be_bar[1] + be_bar[2] - 3.f);

    T Psi_pos = 0.f;
    T Psi_neg = 0.f;

    if (J < 1.f) {
        Psi_pos = Psi_dev;
    }
    else
    {
        Psi_pos = Psi_dev + Psi_vol;
    }
    if (Psi_pos > d_maxPsi[parid] && Psi_pos < (T)1e6)
    {
        d_maxPsi[parid] = Psi_pos;
    }
    T G_C = 3;
    T L0 = 0.5*dx;
	FP[parid] = 4 * L0 * (1.f - 0.001f) * d_maxPsi[parid] * 1.f / G_C + 1.f;
    // 2. compute tau
    T tau_vol[3];
    tau_vol[0] = kappa / 2.f * (J * J - 1.f); tau_vol[1] = kappa / 2.f * (J * J - 1.f); tau_vol[2] = kappa / 2.f * (J * J - 1.f);

    T tau_dev[3];
    tau_dev[0] = mu * (be_bar[0] - 1.f / 3.f * be_bar_sum); tau_dev[1] = mu * (be_bar[1] - 1.f / 3.f * be_bar_sum); tau_dev[2] = mu * (be_bar[2] - 1.f / 3.f * be_bar_sum);

    T P_hat[3];
    T g =  d_phase[parid] * d_phase[parid] * (1.f - 0.0001f) + 0.0001f;
    if (J < 1.f)
    {
        P_hat[0] = g * tau_dev[0] + tau_vol[0]; P_hat[1] = g * tau_dev[1] + tau_vol[1]; P_hat[2] = g * tau_dev[2] + tau_vol[2];
    }
    else
    {
        P_hat[0] = g * (tau_dev[0] + tau_vol[0]); P_hat[1] = g * (tau_dev[1] + tau_vol[1]); P_hat[2] = g * (tau_dev[2] + tau_vol[2]);
    }

    T P[9];
    P[0] = P_hat[0] * U[0] * V[0] + P_hat[1] * U[3] * V[3] + P_hat[2] * U[6] * V[6]; P[1] = P_hat[0] * U[1] * V[0] + P_hat[1] * U[4] * V[3] + P_hat[2] * U[7] * V[6]; P[2] = P_hat[0] * U[2] * V[0] + P_hat[1] * U[5] * V[3] + P_hat[2] * U[8] * V[6];
    P[3] = P_hat[0] * U[0] * V[1] + P_hat[1] * U[3] * V[4] + P_hat[2] * U[6] * V[7]; P[4] = P_hat[0] * U[1] * V[1] + P_hat[1] * U[4] * V[4] + P_hat[2] * U[7] * V[7]; P[5] = P_hat[0] * U[2] * V[1] + P_hat[1] * U[5] * V[4] + P_hat[2] * U[8] * V[7];
    P[6] = P_hat[0] * U[0] * V[2] + P_hat[1] * U[3] * V[5] + P_hat[2] * U[6] * V[8]; P[7] = P_hat[0] * U[1] * V[2] + P_hat[1] * U[4] * V[5] + P_hat[2] * U[7] * V[8]; P[8] = P_hat[0] * U[2] * V[2] + P_hat[1] * U[5] * V[5] + P_hat[2] * U[8] * V[8];

    d_contribution[parid + 0 * numParticle] = (P[0] * F[0] + P[3] * F[3] + P[6] * F[6]) * volume; d_contribution[parid + 1 * numParticle] = (P[1] * F[0] + P[4] * F[3] + P[7] * F[6]) * volume; d_contribution[parid + 2 * numParticle] = (P[2] * F[0] + P[5] * F[3] + P[8] * F[6]) * volume;
    d_contribution[parid + 3 * numParticle] = (P[0] * F[1] + P[3] * F[4] + P[6] * F[7]) * volume; d_contribution[parid + 4 * numParticle] = (P[1] * F[1] + P[4] * F[4] + P[7] * F[7]) * volume; d_contribution[parid + 5 * numParticle] = (P[2] * F[1] + P[5] * F[4] + P[8] * F[7]) * volume;
    d_contribution[parid + 6 * numParticle] = (P[0] * F[2] + P[3] * F[5] + P[6] * F[8]) * volume; d_contribution[parid + 7 * numParticle] = (P[1] * F[2] + P[4] * F[5] + P[7] * F[8]) * volume; d_contribution[parid + 8 * numParticle] = (P[2] * F[2] + P[5] * F[5] + P[8] * F[8]) * volume;
}
//__global__ void updateVelocity(const T dt, T** d_channels) {
//#if TRANSFER_SCHEME != 2
//    int idx = blockIdx.x;
//    int cellid = threadIdx.x;
//    T mass = *((T*)((unsigned long long)d_channels[0] + idx * MEMOFFSET) + cellid);
//    if (mass != 0.f)
//    {
//        mass = dt / mass;
//        for (int i = 0; i < Dim; i++) {
//            *((T*)((unsigned long long)d_channels[i + 1] + idx * MEMOFFSET) + cellid) += *((T*)((unsigned long long)d_channels[i + 4] + idx * MEMOFFSET) + cellid) * mass;
//        }
//    }
//#endif
//}


__device__ bool atomicMaxf2(T* address, T val) {
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	if (*address >= val) return false;
	do {
		assumed = old;
		old = atomicCAS(address_as_i, assumed,
			__float_as_int(fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return true;
}

__global__ void calcMaxVel2(const int numParticle, const T* input, T* _maxVelSquared) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= numParticle) return;

	T vel_squared = input[idx];
	vel_squared *= vel_squared;

	atomicMaxf2(_maxVelSquared, vel_squared);
}

__global__
void mysumFsquare(T* mem1, T* mem2, int numbers) {
	//int tid = threadIdx.x;
	int idof = blockIdx.x * blockDim.x;
	int idx = threadIdx.x + idof;
	extern __shared__ T tep[];
	if (idx >= numbers) return;
    T temp = mem1[idx];
	temp *= temp;
	int warpTid = threadIdx.x % 32;
	int warpId = (threadIdx.x >> 5);
	int warpNum;
	if (blockIdx.x == gridDim.x-1) {
		warpNum = ((numbers - idof + 31) >> 5);
	}
	else {
		warpNum = ((blockDim.x) >> 5);
	}

	T temp2;
	for (int i = 1; i < 32; i = (i << 1)) {
		//temp += __shfl_down_sync(__activemask(), temp, i);
		temp2 = __shfl_down_sync(__activemask(), temp, i);
		temp = temp > temp2 ? temp : temp2;
	}
	if (warpTid == 0) {
		tep[warpId] = temp;
	}
	__syncthreads();
	if (threadIdx.x >= warpNum) return;
	if (warpNum > 1) {
		temp = tep[threadIdx.x];
		//T temp2;
		for (int i = 1; i < warpNum; i = (i << 1)) {
			//temp += __shfl_down_sync(__activemask(), temp, i);
			temp2 = __shfl_down_sync(__activemask(), temp, i);
			temp = temp > temp2 ? temp : temp2;
		}
	}
	if (threadIdx.x == 0) {
		mem2[blockIdx.x] = temp;
	}
}

__global__
void mysumFm(T* mem, int numbers) {
	//int tid = threadIdx.x;
	int idof = blockIdx.x * blockDim.x;
	int idx = threadIdx.x + idof;
	extern __shared__ T tep[];
	if (idx >= numbers) return;
    T temp = mem[idx];
	int warpTid = threadIdx.x % 32;
	int warpId = (threadIdx.x >> 5);
	int warpNum;

	if (blockIdx.x == gridDim.x-1) {
		warpNum = ((numbers - idof + 31) >> 5);
	}
	else {
		warpNum = ((blockDim.x) >> 5);
	}
	T temp2;
	for (int i = 1; i < 32; i = (i << 1)) {
		//temp += __shfl_down_sync(__activemask(), temp, i);
		temp2 = __shfl_down_sync(__activemask(), temp, i);
		temp = temp > temp2 ? temp : temp2;
	}
	if (warpTid == 0) {
		tep[warpId] = temp;
	}
	__syncthreads();
	if (threadIdx.x >= warpNum) return;
	if (warpNum > 1) {

		temp = tep[threadIdx.x];
		for (int i = 1; i < warpNum; i = (i << 1)) {
			//temp += __shfl_down_sync(__activemask(), temp, i);
			temp2 = __shfl_down_sync(__activemask(), temp, i);
			temp = temp > temp2 ? temp : temp2;
		}

	}
	if (threadIdx.x == 0) {
		mem[blockIdx.x] = temp;
	}
}

__global__
void myDotSum(T* A, T* B, T* C, int numbers) {
	//int tid = threadIdx.x;
	int idof = blockIdx.x * blockDim.x;
	int idx = threadIdx.x + idof;
	extern __shared__ T tep[];
	if (idx >= numbers) return;
    T temp = A[idx] * B[idx];
	int warpTid = threadIdx.x % 32;
	int warpId = (threadIdx.x >> 5);
	int warpNum;

	if (blockIdx.x == gridDim.x-1) {
		warpNum = ((numbers - idof + 31) >> 5);
	}
	else {
		warpNum = ((blockDim.x) >> 5);
	}
	for (int i = 1; i < 32; i = (i << 1)) {
		temp += __shfl_down_sync(__activemask(), temp, i);
	}
	if (warpTid == 0) {
		tep[warpId] = temp;
	}
	__syncthreads();
	if (threadIdx.x >= warpNum) return;
	if (warpNum > 1) {
		temp = tep[threadIdx.x];
		for (int i = 1; i < warpNum; i = (i << 1)) {
			temp += __shfl_down_sync(__activemask(), temp, i);
		}

	}
	if (threadIdx.x == 0) {
		C[blockIdx.x] = temp;
	}
}

__global__ void transGrid_to_vector_kernal(T** d_channels, T* grid_r, int id) {
    int idx = blockIdx.x;

    int cellid = (threadIdx.x);
    T tr = *((T*)((unsigned long long)d_channels[9] + idx * MEMOFFSET) + cellid);
    if (tr > 0.f)
    {
        T a = *((T*)((unsigned long long)d_channels[id] + idx * MEMOFFSET) + cellid);
        grid_r[idx * 64 + cellid] = a;
    }
}

__global__ void cal_grid_r_vad_kernal(T** d_channel, T rate, T* in, T* out) {
	int idx = blockIdx.x;
	int cellid = (threadIdx.x);
	T tr = *((T*)((unsigned long long)d_channel[9] + idx * MEMOFFSET) + cellid);
	if (tr > 0.f)
		out[idx * 64 + cellid] = out[idx * 64 + cellid] * rate + in[idx * 64 + cellid];

}

__global__ void update_grid_phase(T** d_channel, T* x) {
    int idx = blockIdx.x;

    int cellid = (threadIdx.x);
	T tr = *((T*)((unsigned long long)d_channel[9] + idx * MEMOFFSET) + cellid);
	if (tr > 0.f)
    {
        T g_c = *((T*)((unsigned long long)d_channel[7] + idx * MEMOFFSET) + cellid);
        *((T*)((unsigned long long)d_channel[7] + idx * MEMOFFSET) + cellid) = x[idx * 64 + cellid] - g_c;
    }
}

__global__ void cal_grid_l_vad_kernal(T** d_channel, T rate, T* in, T* out) {
	int idx = blockIdx.x;

	int cellid = (threadIdx.x);
	T tr = *((T*)((unsigned long long)d_channel[9] + idx * MEMOFFSET) + cellid);
	if (tr > 0.f)
		out[idx * 64 + cellid] += in[idx * 64 + cellid] * rate;

}

__global__ void cal_grid_l_sub2Zero_kernal(T** d_channel, T rate, T* in, T* out) {
	int idx = blockIdx.x;

	int cellid = (threadIdx.x);
	T tr = *((T*)((unsigned long long)d_channel[9] + idx * MEMOFFSET) + cellid);
	if (tr > 0.f) {
		T a = out[idx * 64 + cellid] - in[idx * 64 + cellid] * rate;
		out[idx * 64 + cellid] = a;// > 0.f ? a : 0.f;
	}

}

__global__
void setArray(T* d_array, int numbers, T val) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= numbers) return;
	d_array[idx] = val;
}

__global__ void precondition_subd(T** d_channels, T* grid_r, T* grid_mr) {
    int idx = blockIdx.x;

    int cellid = (threadIdx.x);
    T tag = *((T*)((unsigned long long)d_channels[9] + idx * MEMOFFSET) + cellid);
    if (tag > 0.f)
    {
        T mr = *((T*)((unsigned long long)d_channels[11] + idx * MEMOFFSET) + cellid);
        *((T*)((unsigned long long)d_channels[11] + idx * MEMOFFSET) + cellid) = 0.f;
		/*if ((mr < (T)1e-36) && (mr >= 0.f)) {
			mr = (T)1e-36;
		}
		else if ((mr > (T)-1e-36) && (mr < 0.f)) {
			mr = -(T)1e-36;
		}*/

		if (abs(mr) < (T)1e-30) {
			//grid_mr[64 * idx + cellid] = 0.f;
			return;
		}
        grid_mr[64 * idx + cellid] = (grid_r[64 * idx + cellid]/mr);
		/*if (abs(grid_mr[64 * idx + cellid]) > 1e15) {
			grid_mr[64 * idx + cellid] = 0;
		}*/
    }
}


__global__ void InnerProduct(T* d_implicit_x, T* d_implicit_y, T* _innerProduct, int numbers) {
	int idof = blockIdx.x * blockDim.x;
	int idx = threadIdx.x + idof;
	if (idx >= numbers) return;

	float sum = d_implicit_x[idx] * d_implicit_y[idx];
	for (int offset = 32 / 2; offset > 0; offset /= 2)
		sum += __shfl_down_sync(__activemask(), sum, offset);
	if (threadIdx.x % 32 == 0)    atomicAdd(_innerProduct, sum);
}

void getNormSquare_b(T* mem1, T* mem2, T* NormSize, int numbers) {

    int threads = 128;
    //int dataSize = threads * 2;
    int shasize = sizeof(T) * (threads>>5);
    int blocks = (numbers + threads - 1) / (threads);
    mysumFsquare << <blocks, threads, shasize >> > (mem1, mem2, numbers);
    numbers = blocks;
    blocks = (numbers + threads - 1) / (threads);
    while (numbers > 1) {
        mysumFm << <blocks, threads, shasize >> > (mem2, numbers);
        numbers = blocks;
        blocks = (numbers + threads - 1) / (threads);

    }
    cudaMemcpy(NormSize, mem2, sizeof(T), cudaMemcpyDeviceToHost);
    *NormSize = sqrt(*NormSize);
}

void getDotProduct(T* A, T* B, T* C, T* result, int numbers) {

    int threads = 128;
    //int dataSize = threads * 2;
    int shasize = sizeof(T) * (threads>>5);
    int blocks = (numbers + threads - 1) / (threads);
    myDotSum << <blocks, threads, shasize >> > (A, B, C, numbers);
    numbers = blocks;
    blocks = (numbers + threads - 1) / (threads);
    while (numbers > 1) {
        mysumFm << <blocks, threads, shasize >> > (C, numbers);
        numbers = blocks;
        blocks = (numbers + threads - 1) / (threads);

    }
    cudaMemcpy(result, C, sizeof(T), cudaMemcpyDeviceToHost);
}




void transferVolP2G(
    const T dt,
    Model& model,
    std::unique_ptr<SPGrid>& grid,
    std::unique_ptr<DomainTransformer>& trans)
{

    const unsigned int blockNum = (unsigned int)trans->_numVirtualPage;
    const unsigned int threadNum = 512;

    auto& particlePtr = model.getParticlesPtr();
    auto& materialPtr = model.getMaterialPtr();
    auto material = (ElasticMaterial*)materialPtr.get();


    volP2G_APIC << <blockNum, threadNum >> > (particlePtr->_numParticle, (const int*)trans->d_targetPage, (const int*)trans->d_virtualPageOffset,
        particlePtr->d_smallestNodeIndex,
        trans->d_page2particle, trans->d_particle2cell, particlePtr->d_indices, particlePtr->d_indexTrans, particlePtr->d_orderedPos, particlePtr->d_vol,
        grid->d_channels, trans->d_adjPage, dt, material->parabolic_M);


}



void preCondition_CG(
    const T dt,
    Model& model,
    std::unique_ptr<SPGrid>& grid,
    std::unique_ptr<DomainTransformer>& trans)
{

    const unsigned int blockNum = (unsigned int)trans->_numVirtualPage;
    const unsigned int threadNum = 512;

    auto& particlePtr = model.getParticlesPtr();
    auto& materialPtr = model.getMaterialPtr();
    auto material = (ElasticMaterial*)materialPtr.get();


    //T* pv = new T[particlePtr->_numParticle];
    //cudaMemcpy(pv, particlePtr->d_vol, sizeof(T) * particlePtr->_numParticle, cudaMemcpyDeviceToHost);
    //for (int i = 0; i < particlePtr->_numParticle; i++) {
    //    std::cout << pv[i] << "    ";
    //}
    //delete[] pv;

    preConditionP2G_APIC << <blockNum, threadNum >> > (particlePtr->_numParticle, (const int*)trans->d_targetPage, (const int*)trans->d_virtualPageOffset,
        particlePtr->d_smallestNodeIndex,
        trans->d_page2particle, trans->d_particle2cell, particlePtr->d_indices, particlePtr->d_indexTrans, particlePtr->d_orderedPos, particlePtr->d_vol, particlePtr->d_FP,
        grid->d_channels, trans->d_adjPage, dt, material->parabolic_M);


    precondition_subd << < trans->_numTotalPage, 64 >> > (grid->d_channels, grid->d_grid_r, grid->d_grid_mr);

}

void Ax_CG(
    const T dt,
    Model& model,
    std::unique_ptr<SPGrid>& grid,
    std::unique_ptr<DomainTransformer>& trans)
{

    const unsigned int blockNum = (unsigned int)trans->_numVirtualPage;
    const unsigned int threadNum = 512;

    auto& particlePtr = model.getParticlesPtr();
    auto& materialPtr = model.getMaterialPtr();
    auto material = (ElasticMaterial*)materialPtr.get();


    AxG2P_APIC << <blockNum, threadNum >> > (particlePtr->_numParticle, 0.5*DX, (const int*)trans->d_targetPage, (const int*)trans->d_virtualPageOffset,
        particlePtr->d_smallestNodeIndex,
        trans->d_page2particle, trans->d_particle2cell, particlePtr->d_indices, particlePtr->d_indexTrans, particlePtr->d_orderedPos, particlePtr->d_vol, particlePtr->d_orderedCol,
        grid->d_grid_s, trans->d_adjPage);

    

    AxP2G_APIC << <blockNum, threadNum >> > (particlePtr->_numParticle, (const int*)trans->d_targetPage, (const int*)trans->d_virtualPageOffset,
        particlePtr->d_smallestNodeIndex,
        trans->d_page2particle, trans->d_particle2cell, particlePtr->d_indices, particlePtr->d_indexTrans, particlePtr->d_orderedPos, particlePtr->d_orderedCol, particlePtr->d_vol,
        grid->d_grid_q, particlePtr->d_FP, grid->d_grid_s, trans->d_adjPage, dt, material->parabolic_M);


    //precondition_subd << < trans->_numTotalPage, 64 >> > (grid->d_channels, grid->d_grid_r, grid->d_grid_mr);

}


ExplicitTimeIntegrator::ExplicitTimeIntegrator(int transferScheme, int numParticle, T* dMemTrunk) :
    MPMTimeIntegrator(transferScheme, numParticle, dMemTrunk) {}


void ExplicitTimeIntegrator::computeForceCoefficient(Model& model) {
    auto& particlePtr = model.getParticlesPtr();
    auto& materialPtr = model.getMaterialPtr();
    auto material = (ElasticMaterial*)materialPtr.get();
    const unsigned int numthread = 256;
    const unsigned int numblock = (_numParticle + numthread - 1) / numthread;



#if (Energy_Model == 0)
    computeContributionFixedCorotated << <numblock, numthread>> > (particlePtr->_numParticle, particlePtr->d_F,
        material->_lambda, material->_mu, material->_volume, d_contribution);
#elif (Energy_Model == 1)
    computeContributionNeoHookean << <numblock, numthread >> > (DX, particlePtr->d_FP, particlePtr->_numParticle, particlePtr->d_F, particlePtr->d_maxPsi, particlePtr->d_phase_C,
        material->_kappa, material->_mu, material->_volume, d_contribution);
#elif (Energy_Model == 2)
    applyVonMises << <numblock, numthread >> > (particlePtr->d_g, particlePtr->d_alpha, material->_lambda, material->_mu, DX, particlePtr->d_FP, particlePtr->_numParticle, particlePtr->d_F, particlePtr->d_maxPsi, particlePtr->d_phase_C,
        material->_kappa, material->_volume, d_contribution);
#endif
}

void ExplicitTimeIntegrator::computeParticlePhase_vol(Model& model) {
    auto& particlePtr = model.getParticlesPtr();
    auto& materialPtr = model.getMaterialPtr();
    auto material = (ElasticMaterial*)materialPtr.get();
    const unsigned int numthread = 256;
    const unsigned int numblock = (_numParticle + numthread - 1) / numthread;
    computePhase_field_vol << <numblock, numthread >> > (_numParticle, particlePtr->d_F, material->_volume, particlePtr->d_vol);
   
}

//void ExplicitTimeIntegrator::updateGridVelocity(const T dt, std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans) {
//    updateVelocity << <trans->_numTotalPage, 64 >> > (dt, grid->d_channels);
//}
T totalT = 0.f;
void ExplicitTimeIntegrator::integrate(int type, float* simulationTime, float* preprocessTime, const T dt, Model& model, std::unique_ptr<SPGrid>& grid, std::unique_ptr<DomainTransformer>& trans) {
    
    auto& particles = model.getParticlesPtr();
	cudaEvent_t start, end0,end1,end2;
	(cudaEventCreate(&start));
	(cudaEventCreate(&end0));
	(cudaEventCreate(&end1));
	(cudaEventCreate(&end2));
	float time1,time2,time3;

	cudaEventRecord(start);

    trans->rebuild();

	cudaEventRecord(end0);
    computeCellIndex(trans, particles);
    computeForceCoefficient(model);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    transferP2G(dt, particles, grid, trans);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    undateGrid(dt, grid, trans);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    bool update_phase_field = true;

#if (Energy_Model == 0)
    update_phase_field = false;
#endif

    if(update_phase_field)
    {
        //////////////////////////////////////////////////
        /////////////////CG_SOLVER////////////////////////
        //////////////////////////////////////////////////
		T cg_dt = dt;
        computeParticlePhase_vol(model);
        transferVolP2G(cg_dt, model, grid, trans);//rhs
        int numbers = trans->_numTotalPage * 64;
		int threadNumA = 256;
		int blockNumA = (numbers + threadNumA - 1) / threadNumA;

		setArray << <blockNumA, threadNumA >> > (grid->d_grid_r, numbers, 0.f);
		setArray << <blockNumA, threadNumA >> > (grid->d_grid_x, numbers, 0.f);
        int maxIterationTimes = 100;
        T tolerance = 1e-6f;
        T convergence_norm = 0;
        transGrid_to_vector_kernal << < trans->_numTotalPage, 64 >> > (grid->d_channels, grid->d_grid_r, 10);
        getNormSquare_b(grid->d_grid_r, grid->d_grid_temp, &convergence_norm, numbers);
      
		setArray << <blockNumA, threadNumA >> > (grid->d_grid_mr, numbers, 0.f);
        preCondition_CG(cg_dt, model, grid, trans);

        CUDA_SAFE_CALL(cudaMemcpy(grid->d_grid_s, grid->d_grid_mr, sizeof(T) * numbers, cudaMemcpyDeviceToDevice));

        T rho_old;
        getDotProduct(grid->d_grid_r, grid->d_grid_mr, grid->d_grid_temp, &rho_old, numbers);
        for (int iterations = 0; iterations < maxIterationTimes; iterations++) {    
			setArray << <blockNumA, threadNumA >> > (grid->d_grid_q, numbers, 0.f);
            Ax_CG(cg_dt, model, grid, trans);
            T s_dot_q;
            getDotProduct(grid->d_grid_s, grid->d_grid_q, grid->d_grid_temp, &s_dot_q, numbers);
            T alpha = rho_old / s_dot_q;

            cal_grid_l_vad_kernal << < trans->_numTotalPage, 64 >> > (grid->d_channels, alpha, grid->d_grid_s, grid->d_grid_x);
            cal_grid_l_sub2Zero_kernal << < trans->_numTotalPage, 64 >> > (grid->d_channels, alpha, grid->d_grid_q, grid->d_grid_r);
            
            getNormSquare_b(grid->d_grid_r, grid->d_grid_temp, &convergence_norm, numbers);


            if (convergence_norm < tolerance) {          
                break;
            }
            preCondition_CG(cg_dt, model, grid, trans);

            T rho;

            getDotProduct(grid->d_grid_r, grid->d_grid_mr, grid->d_grid_temp, &rho, numbers);


			if (rho_old == 0.f) {			
				setArray << <blockNumA, threadNumA >> > (grid->d_grid_x, numbers, 1e20);
				break;
			}
			T rate = rho / rho_old;
            rho_old = rho;
            cal_grid_r_vad_kernal << < trans->_numTotalPage, 64 >> > (grid->d_channels, rate, grid->d_grid_mr, grid->d_grid_s);

        }
        update_grid_phase << < trans->_numTotalPage, 64 >> > (grid->d_channels, grid->d_grid_x);
        //PullGrid(grid, trans, totalT);
        resolveCollision(grid, trans);
        totalT += dt;
    }
    else {
        resolveCollision(grid, trans);
    }
    
    transferG2P(dt, particles, grid, trans);
	cudaEventRecord(end1);
    particles->reorder();
	cudaEventRecord(end2);
	cudaDeviceSynchronize();

	(cudaEventElapsedTime(&time1, start, end0));
	(cudaEventElapsedTime(&time2, end0, end1));
	(cudaEventElapsedTime(&time3, end1, end2));
	*simulationTime = time2;
	*preprocessTime = time1;
	*preprocessTime += time3;


	(cudaEventDestroy(start));
	(cudaEventDestroy(end0));
	(cudaEventDestroy(end1));
	(cudaEventDestroy(end2));
}