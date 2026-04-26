export const codePairs = {
  signature: {
    left: String.raw`template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {`,
    right: String.raw`template <
    const int BM, const int BN, const int BK, const int TM, const int TN
    class ATiler, class BTiler, class CTiler,
    class AStride, class ASmemLayout, class AThreadLayout,
    class BStride, class BSmemLayout, class BThreadLayout,
    class CStride, class CThreadLayout
>
__global__ void sgemm_2d_block_tiling_cute(
    int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C,
    AStride A_strides, BStride B_strides, CStride C_strides,
    ATiler A_tiler, BTiler B_tiler, CTiler C_tiler,
    ASmemLayout A_shared_layout, BSmemLayout B_shared_layout,
    AThreadLayout A_thread_layout, BThreadLayout B_thread_layout, CThreadLayout C_thread_layout
) {`,
  },
  preamble: {
    left: String.raw`const uint cRow = blockIdx.y;
const uint cCol = blockIdx.x;

// BN/TN are the number of threads to span a column
const int threadCol = threadIdx.x % (BN / TN);
const int threadRow = threadIdx.x / (BN / TN);

// allocate space for the current blocktile in smem
__shared__ float As[BM * BK];
__shared__ float Bs[BK * BN];

// Move blocktile to beginning of A's row and B's column
A += cRow * BM * K;
B += cCol * BN;
C += cRow * BM * N + cCol * BN;`,
    right: String.raw`// create Tensors from data + Layout
Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), A_strides);
Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(K, N), B_strides);
Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), C_strides);

// shared tiles
__shared__ float A_smem[cosize_v<ASmemLayout>];
__shared__ float B_smem[cosize_v<BSmemLayout>];
Tensor sA = make_tensor(make_smem_ptr(A_smem), A_shared_layout);
Tensor sB = make_tensor(make_smem_ptr(B_smem), B_shared_layout);

// global tiles
Tensor gA = local_tile(mA, A_tiler, make_coord(blockIdx.y, _)); // (BM, BK, k)
Tensor gB = local_tile(mB, B_tiler, make_coord(_, blockIdx.x)); // (BK, BN, k)
Tensor gC = local_tile(mC, C_tiler, make_coord(blockIdx.y, blockIdx.x)); // (BM, BN)`,
  },
  threadWork: {
    left: String.raw`// calculating the indices that this thread will load into SMEM
const uint innerRowA = threadIdx.x / BK;
const uint innerColA = threadIdx.x % BK;
// calculates the number of rows of As that are being loaded in a single step
// by a single block
const uint strideA = numThreadsBlocktile / BK;
const uint innerRowB = threadIdx.x / BN;
const uint innerColB = threadIdx.x % BN;
// for both As and Bs we want each load to span the full column-width, for
// better GMEM coalescing (as opposed to spanning full row-width and iterating
// across columns)
const uint strideB = numThreadsBlocktile / BN;

// allocate thread-local cache for results in registerfile
float threadResults[TM * TN] = {0.0};
// register caches for As and Bs
float regM[TM] = {0.0};
float regN[TN] = {0.0};`,
    right: String.raw`// part of gA each thread loads to sA
Tensor gA_to_r = local_partition(gA, A_thread_layout, threadIdx.x);
Tensor sA_to_w = local_partition(sA, A_thread_layout, threadIdx.x);

// part of gB each thread loads to sB
Tensor gB_to_r = local_partition(gB, B_thread_layout, threadIdx.x);
Tensor sB_to_w = local_partition(sB, B_thread_layout, threadIdx.x);

// part of sA, sB each thread reads for computation
auto thread_row_C = threadIdx.x / (BN / TN);
auto thread_col_C = threadIdx.x % (BN / TN);
auto A_col_shape = make_shape(TM, 1);
auto B_row_shape = make_shape(1, TN);
Tensor sA_to_r = local_tile(sA, A_col_shape, make_coord(thread_row_C, _)); // (TM, 1, BK)
Tensor sB_to_r = local_tile(sB, B_row_shape, make_coord(_, thread_col_C)); // (1, TN, BK)

// part of gC each thread writes results
Tensor gC_to_w = local_tile(gC, shape(C_thread_layout), make_coord(thread_row_C, thread_col_C)); // (TM, TN)

// rmem
Tensor thread_results = make_tensor_like(gC_to_w);
clear(thread_results);
Tensor tmp_A = make_tensor_like<float>(make_layout(make_shape(TM)));
Tensor tmp_B = make_tensor_like<float>(make_layout(make_shape(TN)));`,
  },
  hotLoop: {
    left: String.raw`// outer-most loop over block tiles
for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
        As[(innerRowA + loadOffset) * BK + innerColA] =
            A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
        Bs[(innerRowB + loadOffset) * BN + innerColB] =
            B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // block into registers
        for (uint i = 0; i < TM; ++i) {
            regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
        }
        for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
        }
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                threadResults[resIdxM * TN + resIdxN] +=
                regM[resIdxM] * regN[resIdxN];
            }
        }
    }
    __syncthreads();
}`,
    right: String.raw`auto max_tile_idx = shape<2>(gA);
for (int tile_idx = 0; tile_idx < max_tile_idx; tile_idx++) {
    // load tiles
    Tensor gA_tile = gA_to_r(_, _, tile_idx);
    CUTE_UNROLL
    for (int i = 0; i < size(gA_tile); i++) {
        sA_to_w(i) = gA_tile(i);
    }
    Tensor gB_tile = gB_to_r(_, _, tile_idx);
    CUTE_UNROLL
    for (int i = 0; i < size(gB_tile); i++) {
        sB_to_w(i) = gB_tile(i);
    }

    __syncthreads();

    // compute partial results for this tile
    CUTE_UNROLL
    for (int dot_idx = 0; dot_idx < shape<2>(sA_to_r); dot_idx++) {
        // load row/col from smem to rmem
        Tensor sA_col = sA_to_r(_, _, dot_idx);
        CUTE_UNROLL
        for (int i = 0; i < size(tmp_A); i++) {
            tmp_A(i) = sA_col(i);
        }
        Tensor sB_row = sB_to_r(_, _, dot_idx);
        CUTE_UNROLL
        for (int i = 0; i < size(tmp_B); i++) {
            tmp_B(i) = sB_row(i);
        }

        // outer product
        CUTE_UNROLL
        for (int i = 0; i < shape<0>(thread_results); i++) {
            CUTE_UNROLL
            for (int j = 0; j < shape<1>(thread_results); j++) {
                thread_results(i, j) += tmp_A(i) * tmp_B(j);
            }
        }
    }
    __syncthreads();
}`,
  },
  writeback: {
    left: String.raw`// write out the results
for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
        C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
            alpha * threadResults[resIdxM * TN + resIdxN] +
            beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
}`,
    right: String.raw`// write results back to gmem
CUTE_UNROLL
for (int i = 0; i < shape<0>(thread_results); i++) {
    CUTE_UNROLL
    for (int j = 0; j < shape<1>(thread_results); j++) {
        gC_to_w(i, j) = alpha * thread_results(i, j) + beta * gC_to_w(i, j);
    }
}`,
  },
};
