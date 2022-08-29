// This file is auto-generated. Do not edit!

#include "OpenclKernels.hpp"

const std::string OpenclKernels::qr_decomposition_str = R"( 
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

void atomic_add(__local double *val, double delta)
{
    union{
        double f;
        ulong i;
    } old;
    union {
        double f;
        ulong i;
    } new;

    do{
        old.f = *val;
        new.f = old.f + delta;
    } while(atom_cmpxchg((volatile __local ulong *)val, old.i, new.i) != old.i);
}

unsigned int dense_block_ind(const unsigned int sqrp,
                             const unsigned int nbcols,
                             const unsigned int bs,
                             const unsigned int br,
                             const unsigned int bc,
                             const unsigned int r,
                             const unsigned int c)
{
    return sqrp + nbcols * br * bs * bs + (r * nbcols + bc) * bs + c;
}

__kernel void sp2dense(const unsigned int bs,
                       const unsigned int tsm,     // target submatrix
                       __global const int *nbrows, // nbrows for submatrix
                       __global const int *nbcols, // nbcols for submatrix
                       __global const int *srp,    // submatrix rowPointers
                       __global const int *srpp,   // pointers for submatrix rowPointers
                       __global const int *sci,    // submatrix colIndices
                       __global const int *scip,   // pointers for submatrix colIndices
                       __global const int *svl,    // submatrix valsLocations
                       __global const int *svlp,   // pointers to submatrix valsLocations
                       __global const int *sqrp,   // pointers to the dense submatrices
                       __global const double *vals,
                       __global double *QR)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
    const unsigned int num_rows_per_warp = warpsize / bs / bs;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int c = (lane / bs) % bs;
    const unsigned int r = lane % bs;
    unsigned int br = lane / bs / bs;

    if(lane < num_active_threads){
        while(br < nbrows[tsm]){
            for(unsigned int ptr = srp[srpp[tsm] + br]; ptr < srp[srpp[tsm] + br + 1]; ptr++){
                QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, sci[scip[tsm] + ptr], r, c)] = vals[svl[svlp[tsm] + ptr] * bs * bs + r * bs + c];
            }

            br += num_rows_per_warp;
        }
    }
}

__kernel void set_qx0(const unsigned int bs,
                      const unsigned int tsm,
                      __global const int *rhsp,
                      __global const int *ibid,
                      __global double *qx0)
{
    const unsigned int idx_t = get_local_id(0);

    if(idx_t < bs){
        qx0[rhsp[tsm] + ibid[tsm] * bs * bs + idx_t * bs + idx_t] = 1.0;
    }
}

__kernel void coldotp(const unsigned int bs,
                      const unsigned int tsm,
                      const unsigned int coll,
                      const unsigned int colr,
                      const unsigned int row_offset,
                      __global const int *nbrows,
                      __global const int *nbcols,
                      __global const int *sqrp,
                      __global const double *QR,
                      __local double *sum)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;

    sum[lane] = 0.0;

    for(unsigned int r = lane + coll + row_offset; r < nbrows[tsm] * bs; r += warpsize){
        sum[lane] += QR[sqrp[tsm] + r * nbcols[tsm] * bs + coll] * QR[sqrp[tsm] + r * nbcols[tsm] * bs + colr];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int stride = warpsize / 2; stride > 0; stride /= 2){
        if(lane < stride){
            sum[lane] += sum[lane + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void sub_scale_col(const unsigned int bs,
                            const unsigned int tsm,
                            const unsigned int coll,
                            const unsigned int colr,
                            const unsigned int row_offset,
                            const double factor,
                            __global const int *nbrows,
                            __global const int *nbcols,
                            __global const int *sqrp,
                            __global double *QR)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;

    for(unsigned int r = lane + colr + row_offset; r < nbrows[tsm] * bs; r += warpsize){
        QR[sqrp[tsm] + r * nbcols[tsm] * bs + coll] -= factor * QR[sqrp[tsm] + r * nbcols[tsm] * bs + colr];
    }
}

__kernel void scale_col(const unsigned int bs,
                        const unsigned int tsm,
                        const unsigned int col,
                        const unsigned int row_offset,
                        const double factor,
                        __global const int *nbrows,
                        __global const int *nbcols,
                        __global const int *sqrp,
                        __global double *QR)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;

    for(unsigned int r = lane + col + row_offset; r < nbrows[tsm] * bs; r += warpsize){
        QR[sqrp[tsm] + r * nbcols[tsm] * bs + col] /= factor;
    }
}

__kernel void tile_house(const unsigned int bs,
                         const unsigned int tile,
                         const unsigned int tsm,
                         __global const int *nbrows,
                         __global const int *nbcols,
                         __global const int *sqrp,
                         __global double *QR,
                         __local double *sum,
                         __local double *T)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;

    double v0, beta, mu;

    for(unsigned int col = tile * bs; col < (tile + 1) * bs; col++){
        coldotp(bs, tsm, col, col, 1, nbrows, nbcols, sqrp, QR, sum);

        double alpha = QR[sqrp[tsm] + col * nbcols[tsm] * bs + col];
        double sigma = sum[0];

        if(sigma == 0){
            v0 = 1.0;
            beta = alpha >= 0 ? 0.0 : -2.0;
        }
        else{
            mu = sqrt(alpha * alpha + sigma);
            v0 = alpha <= 0 ? alpha - mu : -sigma / (alpha + mu);
            beta = 2 * (v0 * v0) / (sigma + v0 * v0);
        }

        for(unsigned int i = 0; i < bs - (col - tile * bs); i++){
            unsigned int _col = (tile + 1) * bs - i - 1;

            coldotp(bs, tsm, col, _col, 1, nbrows, nbcols, sqrp, QR, sum);
            alpha = QR[sqrp[tsm] + col * nbcols[tsm] * bs + _col];
            double s = alpha + sum[0] / v0;
            QR[sqrp[tsm] + col * nbcols[tsm] * bs + _col] -= beta * s;

            if(_col > col){
                sub_scale_col(bs, tsm, col, _col, 1, beta * s / v0, nbrows, nbcols, sqrp, QR);
            }
        }

        scale_col(bs, tsm, col, 1, v0, nbrows, nbcols, sqrp, QR);
    }

    if(lane < bs * bs){
        T[lane] = 0.0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int i = 0; i < bs; i++){
        coldotp(bs, tsm, tile * bs + i, tile * bs + i, 1, nbrows, nbcols, sqrp, QR, sum);
        T[i * bs + i] = (1 + sum[0]) / 2;

        for(unsigned int j = i + 1; j < bs; j++){
            coldotp(bs, tsm, tile * bs + i, tile * bs + j, j + 1, nbrows, nbcols, sqrp, QR, sum);
            T[i * bs + j] = QR[sqrp[tsm] + (tile * bs + j) * nbcols[tsm] * bs + (tile * bs + i)] + sum[0];
        }
    }
}

__kernel void block_coldotp_transp(const unsigned int bs,
                                   const unsigned int tsm,
                                   const unsigned int bc,
                                   const unsigned int tile,
                                   __global const int *nbrows,
                                   __global const int *nbcols,
                                   __global const int *sqrp,
                                   __global const double *QR,
                                   __local double *W)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
    const unsigned int num_cols_per_warp = warpsize / bs / bs;
    const unsigned int num_rows_per_warp = warpsize / bs / bs;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    W[lane] = 0.0;

    for(unsigned int _bc = 0; _bc < num_cols_per_warp && bc + _bc < nbcols[tsm]; _bc++){
        if(lane < num_active_threads){
            for(unsigned int br = tile + lane / bs / bs; br < nbrows[tsm]; br += num_rows_per_warp){
                double temp = 0.0;

                for(unsigned int k = 0; k < bs; k++){
                    if(br == tile){
                        if(k == 0){
                            temp += QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, bc + _bc, i, j)];
                        }
                        else if(k > i){
                            temp += QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, tile, k, i)] * \
                                QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, bc + _bc, k, j)];
                        }
                    }
                    else{
                        temp += QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, tile, k, i)] * \
                            QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, bc + _bc, k, j)];
                    }
                }

                atomic_add(W + _bc * bs * bs + i * bs + j, temp);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void block_coldotp_transp_qx(const unsigned int bs,
                                      const unsigned int tsm,
                                      const unsigned int tile,
                                      __global const int *nbrows,
                                      __global const int *nbcols,
                                      __global const int *sqrp,
                                      __global const int *rhsp,
                                      __global const double *QR,
                                      __global const double *qx,
                                      __local double *W)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
    const unsigned int num_rows_per_warp = warpsize / bs / bs;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    W[lane] = 0.0;

    if(lane < num_active_threads){
        for(unsigned int br = tile + lane / bs / bs; br < nbrows[tsm]; br += num_rows_per_warp){
            double temp = 0.0;

            for(unsigned int k = 0; k < bs; k++){
                if(br == tile){
                    if(k == 0){
                        temp += qx[rhsp[tsm] + br * bs * bs + i * bs + j];
                    }
                    else if(k > i){
                        temp += QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, tile, k, i)] * \
                            qx[rhsp[tsm] + br * bs * bs + k * bs + j];
                    }
                }
                else{
                    temp += QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, tile, k, i)] * \
                        qx[rhsp[tsm] + br * bs * bs + k * bs + j];
                }
            }

            atomic_add(W + i * bs + j, temp);
        }
    }
}

__kernel void block_col_trsolve(const unsigned int bs,
                                __local const double *T,
                                __local double *W)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    if(lane < num_active_threads){
        unsigned int blk = lane / bs / bs;

        W[blk * bs * bs + i * bs + j] /= T[i * bs + i];

        for(unsigned int k = 0; k < bs - 1; k++){
            if(i > k){
                W[blk * bs * bs + i * bs + j] -= T[k * bs + i] * W[blk * bs * bs + k * bs + j] / T[i * bs + i];
            }
        }
    }
}

__kernel void block_col_mult_sub(const unsigned int bs,
                                 const unsigned int tsm,
                                 const unsigned int bc,
                                 const unsigned int tile,
                                 __global const int *nbrows, 
                                 __global const int *nbcols, 
                                 __global const int *sqrp, 
                                 __global double *QR,
                                 __local const double *W)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
    const unsigned int num_cols_per_warp = warpsize / bs / bs;
    const unsigned int num_rows_per_warp = warpsize / bs / bs;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    for(unsigned int _bc = 0; _bc < num_cols_per_warp && bc + _bc < nbcols[tsm]; _bc++){
        if(lane < num_active_threads){
            for(unsigned int br = tile + lane / bs / bs; br < nbrows[tsm]; br += num_rows_per_warp){
                for(unsigned int k = 0; k < bs; k++){
                    if(br == tile){
                        if(k == 0){
                            QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, bc + _bc, i, j)] -= W[_bc * bs * bs + i * bs + j];
                        }
                        else if(k < i){
                            QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, bc + _bc, i, j)] -= QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, tile, i, k)] * \
                                W[_bc * bs * bs + k * bs + j];
                        }
                    }
                    else{
                        QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, bc + _bc, i, j)] -= QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, tile, i, k)] * \
                            W[_bc * bs * bs + k * bs + j];
                    }
                }
            }
        }
    }
}

__kernel void block_col_mult_sub_qx(const unsigned int bs,
                                    const unsigned int tsm,
                                    const unsigned int tile,
                                    __global const int *nbrows,
                                    __global const int *nbcols,
                                    __global const int *sqrp,
                                    __global const int *rhsp,
                                    __global const double *QR,
                                    __global double *qx,
                                    __local double *W)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
    const unsigned int num_rows_per_warp = warpsize / bs / bs;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    if(lane < num_active_threads){
        for(unsigned int br = tile + lane / bs / bs; br < nbrows[tsm]; br += num_rows_per_warp){
            for(unsigned int k = 0; k < bs; k++){
                if(br == tile){
                    if(k == 0){
                        qx[rhsp[tsm] + br * bs * bs + i * bs + j] -= W[i * bs + j];
                    }
                    else if(k < i){
                        qx[rhsp[tsm] + br * bs * bs + i * bs + j] -= QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, tile, i, k)] * W[k * bs + j];
                    }
                }
                else{
                    qx[rhsp[tsm] + br * bs * bs + i * bs + j] -= QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, tile, i, k)] * W[k * bs + j];
                }
            }
        }
    }
}

__kernel void update_tr(const unsigned int bs,
                        const unsigned int tsm,
                        const unsigned int tile,
                        __global const int *nbrows,
                        __global const int *nbcols,
                        __global const int *sqrp,
                        __global double *QR,
                        __local const double *T,
                        __local double *W)
{
    const unsigned int warpsize = 32;
    const unsigned int num_cols_per_warp = warpsize / bs / bs;

    for(unsigned int bc = tile + 1; bc < nbcols[tsm]; bc += num_cols_per_warp){
        block_coldotp_transp(bs, tsm, bc, tile, nbrows, nbcols, sqrp, QR, W);
        block_col_trsolve(bs, T, W);
        block_col_mult_sub(bs, tsm, bc, tile, nbrows, nbcols, sqrp, QR, W);
    }
}

__kernel void update_qx(const unsigned int bs,
                        const unsigned int tsm,
                        const unsigned int tile,
                        __global const int *nbrows,
                        __global const int *nbcols,
                        __global const int *sqrp,
                        __global const int *rhsp,
                        __global const double *QR,
                        __global double *qx,
                        __local const double *T,
                        __local double *W)
{
    block_coldotp_transp_qx(bs, tsm, tile, nbrows, nbcols, sqrp, rhsp, QR, qx, W);
    block_col_trsolve(bs, T, W);
    block_col_mult_sub_qx(bs, tsm, tile, nbrows, nbcols, sqrp, rhsp, QR, qx, W);
}

__kernel void qr_decomposition(const unsigned int bs,
                               const unsigned int Nb,
                               const unsigned int tile,
                               __global const int *nbrows, // nbrows for submatrix
                               __global const int *nbcols, // nbcols for submatrix
                               __global const int *srp,    // submatrix rowPointers
                               __global const int *srpp,   // pointers to submatrix rowPointers
                               __global const int *sci,    // submatrix colIndices
                               __global const int *scip,   // pointers to submatrix colIndices
                               __global const int *svl,    // submatrix valsLocations
                               __global const int *svlp,   // pointers to submatrix valsLocations
                               __global const int *sqrp,   // pointers to the dense submatrices
                               __global const int *rhsp,   // pointers to LSQ systems' right-hand-sides
                               __global const int *ibid,   // identity block index for target LSQ system
                               __global const double *vals,
                               __global double *QR,
                               __global double *qx,
                               __local double *aux)
{
    const unsigned int warpsize = 32;
    const unsigned int bsize = get_local_size(0);
    const unsigned int idx_b = get_global_id(0) / bsize;
    const unsigned int idx_t = get_local_id(0);
    unsigned int idx = idx_b * bsize + idx_t;
    const unsigned int NUM_THREADS = get_global_size(0);
    const unsigned int num_warps_in_grid = NUM_THREADS / warpsize;
    unsigned int tsm = idx / warpsize; // target subsystem (eq. to column)

    __local double T[9];
    
    while(tsm < Nb){
        if(tile == 0){
            sp2dense(bs, tsm, nbrows, nbcols, srp, srpp, sci, scip, svl, svlp, sqrp, vals, QR);
            set_qx0(bs, tsm, rhsp, ibid, qx);
        }
        else if(tile < nbcols[tsm]){
            tile_house(bs, tile, tsm, nbrows, nbcols, sqrp, QR, aux, T);
            update_tr(bs, tsm, tile, nbrows, nbcols, sqrp, QR, T, aux);
            update_qx(bs, tsm, tile, nbrows, nbcols, sqrp, rhsp, QR, qx, T, aux);
        }
        
        tsm += num_warps_in_grid;
    }
})"; 

const std::string OpenclKernels::solve_str = R"( 
__kernel void up_trsolve(const unsigned int bs,
                         const unsigned int tsm,
                         const unsigned int br,
                         __global const int *nbcols,
                         __global const int *sqrp,
                         __global const double *QR,
                         __global const double *B,
                         __global double *X)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    X[i * bs + j] = B[i * bs + j] / QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, br, i, i)];

    for(unsigned int k = 1; k < bs; k++){
        if(i < k){
            X[i * bs + j] -= QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, br, i, k)] * \
                             X[k * bs + j] / QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, br, i, i)];
        }
    }
}

__kernel void up_trsolve_mat(const unsigned int bs,
                             const unsigned int tsm,
                             const unsigned int br,
                             const unsigned int bc,
                             __global const int *nbcols,
                             __global const int *sqrp,
                             __global double *QR)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, bc, i, j)] /= QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, br, i, i)];

    for(unsigned int k = 1; k < bs; k++){
        if(i < k){
            QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, bc, i, j)] -= QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, br, i, k)] * \
                QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, bc, k, j)] / QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, br, i, i)];
        }
    }
}

__kernel void block_mult_sub(const unsigned int bs,
                             const unsigned int tsm,
                             const unsigned int br,
                             const unsigned int bc,
                             __global const int *nbcols,
                             __global const int *sqrp,
                             __global double *QR,
                             __global const double *B,
                             __global double *C)
{
    const unsigned int warpsize = 32;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int lane = idx_t % warpsize;
    const unsigned int i = (lane / bs) % bs;
    const unsigned int j = lane % bs;

    for(unsigned int k = 0; k < bs; k++){
        C[i * bs + j] -= QR[dense_block_ind(sqrp[tsm], nbcols[tsm], bs, br, bc, i, k)] * B[k * bs + j];
    }
}

__kernel void solve(const unsigned int bs,
                    const unsigned int Nb,
                    __global const int *nbcols,
                    __global const int *sqrp,
                    __global const int *rhsp,
                    __global const int *spaip,
                    __global double *spaiv,
                    __global double *QR,
                    __global const double *b)
{

    const unsigned int warpsize = 32;
    const unsigned int bsize = get_local_size(0);
    const unsigned int idx_b = get_global_id(0) / bsize;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int NUM_THREADS = get_global_size(0);
    const unsigned int num_warps_in_grid = NUM_THREADS / warpsize;
    const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
    const unsigned int num_rows_per_warp = warpsize / bs / bs;
    const unsigned int lane = idx_t % warpsize;
    unsigned int idx = idx_b * bsize + idx_t;
    unsigned int tsm = idx / warpsize; // target subsystem (eq. to column)

    while(tsm < Nb){
        if(lane < num_active_threads){
            for(unsigned int br = lane / bs / bs; br < nbcols[tsm]; br += num_rows_per_warp){
                up_trsolve(bs, tsm, br, nbcols, sqrp, QR, b + rhsp[tsm] + br * bs * bs, spaiv + (spaip[tsm] + br) * bs * bs);
            }

            for(unsigned int _br = 1; _br < nbcols[tsm]; _br++){
                for(unsigned int br = lane / bs / bs; br < nbcols[tsm] - _br; br += num_rows_per_warp){
                    up_trsolve_mat(bs, tsm, br, nbcols[tsm] - _br, nbcols, sqrp, QR);
                    block_mult_sub(bs, tsm, br, nbcols[tsm] - _br, nbcols, sqrp, QR,
                                   spaiv + (spaip[tsm] + nbcols[tsm] - _br) * bs * bs, spaiv + (spaip[tsm] + br) * bs * bs);
                }
            }
        }

        tsm += num_warps_in_grid;
    }
})"; 

