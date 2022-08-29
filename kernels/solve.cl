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
}