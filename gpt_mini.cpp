#include "gpt_mini.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <immintrin.h>

// Force aggressive optimization for this file
// #pragma GCC optimize("O3,unroll-loops")

// Helper for aligned allocation
static inline void* alloc_aligned(size_t size) {
    void* ptr;
    if (posix_memalign(&ptr, 64, size)) return nullptr;
    return ptr;
}

static inline void free_aligned(void* ptr) {
    free(ptr);
}

using std::max;
using std::min;
using std::unique_ptr;
using std::vector;

namespace {

int sample_from(const float* p, int size) {
    int best_idx = 0;
    float best_val = p[0];
    for (int i = 1; i < size; i++) {
        if (p[i] > best_val) {
            best_val = p[i];
            best_idx = i;
        }
    }
    return best_idx;
}

float* softmax(const float* x, int size) {
    float maxv = *std::max_element(x, x + size);
    float sum = 0;
    float* y = new float[size];
    for (int i = 0; i < size; i++) {
        y[i] = std::exp(x[i] - maxv);
        sum += y[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < size; i++) y[i] *= inv;
    return y;
}

void transpose(const float* __restrict__ M, int rows, int cols, float* __restrict__ T) {
    const int BLOCK = 32;
    for (int i = 0; i < rows; i += BLOCK) {
        for (int j = 0; j < cols; j += BLOCK) {
            int i_lim = min(rows, i + BLOCK);
            int j_lim = min(cols, j + BLOCK);
            for (int ii = i; ii < i_lim; ++ii) {
                for (int jj = j; jj < j_lim; ++jj) {
                    T[jj * rows + ii] = M[ii * cols + jj];
                }
            }
        }
    }
}

float* transpose(const float* M, int rows, int cols) {
    float* T = new float[cols * rows];
    transpose(M, rows, cols, T);
    return T;
}

// ==================== Optimized Matrix Multiplication ====================

// Micro-kernel: 16x16 using AVX-512 with 8-unrolling
// Uses all 32 ZMM registers effectively
__attribute__((always_inline, hot))
inline void kernel_16x16(int k, const float* __restrict__ packedA, 
                         const float* __restrict__ packedB, 
                         float* __restrict__ C, int ldc) {
    __m512 c0  = _mm512_loadu_ps(C + 0*ldc);
    __m512 c1  = _mm512_loadu_ps(C + 1*ldc);
    __m512 c2  = _mm512_loadu_ps(C + 2*ldc);
    __m512 c3  = _mm512_loadu_ps(C + 3*ldc);
    __m512 c4  = _mm512_loadu_ps(C + 4*ldc);
    __m512 c5  = _mm512_loadu_ps(C + 5*ldc);
    __m512 c6  = _mm512_loadu_ps(C + 6*ldc);
    __m512 c7  = _mm512_loadu_ps(C + 7*ldc);
    __m512 c8  = _mm512_loadu_ps(C + 8*ldc);
    __m512 c9  = _mm512_loadu_ps(C + 9*ldc);
    __m512 c10 = _mm512_loadu_ps(C + 10*ldc);
    __m512 c11 = _mm512_loadu_ps(C + 11*ldc);
    __m512 c12 = _mm512_loadu_ps(C + 12*ldc);
    __m512 c13 = _mm512_loadu_ps(C + 13*ldc);
    __m512 c14 = _mm512_loadu_ps(C + 14*ldc);
    __m512 c15 = _mm512_loadu_ps(C + 15*ldc);

    const float* b_ptr = packedB;
    const float* a_ptr = packedA;

    int p = 0;
    for (; p <= k - 8; p += 8) {
        // Prefetch next block
        _mm_prefetch((const char*)(b_ptr + 256), _MM_HINT_T0);
        _mm_prefetch((const char*)(a_ptr + 256), _MM_HINT_T0);
        
        __m512 b0 = _mm512_load_ps(b_ptr);
        __m512 b1 = _mm512_load_ps(b_ptr + 16);
        __m512 b2 = _mm512_load_ps(b_ptr + 32);
        __m512 b3 = _mm512_load_ps(b_ptr + 48);
        __m512 b4 = _mm512_load_ps(b_ptr + 64);
        __m512 b5 = _mm512_load_ps(b_ptr + 80);
        __m512 b6 = _mm512_load_ps(b_ptr + 96);
        __m512 b7 = _mm512_load_ps(b_ptr + 112);

        // Process all 16 rows with 8 k-iterations
        #define PROCESS_ROW(row) \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row]), b0, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 16]), b1, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 32]), b2, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 48]), b3, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 64]), b4, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 80]), b5, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 96]), b6, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 112]), b7, c##row);

        PROCESS_ROW(0)
        PROCESS_ROW(1)
        PROCESS_ROW(2)
        PROCESS_ROW(3)
        PROCESS_ROW(4)
        PROCESS_ROW(5)
        PROCESS_ROW(6)
        PROCESS_ROW(7)
        PROCESS_ROW(8)
        PROCESS_ROW(9)
        PROCESS_ROW(10)
        PROCESS_ROW(11)
        PROCESS_ROW(12)
        PROCESS_ROW(13)
        PROCESS_ROW(14)
        PROCESS_ROW(15)
        
        #undef PROCESS_ROW

        b_ptr += 128;
        a_ptr += 128;
    }
    
    for (; p < k; ++p) {
        __m512 b = _mm512_load_ps(b_ptr);
        b_ptr += 16;
        
        c0  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[0]), b, c0);
        c1  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[1]), b, c1);
        c2  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[2]), b, c2);
        c3  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[3]), b, c3);
        c4  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[4]), b, c4);
        c5  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[5]), b, c5);
        c6  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[6]), b, c6);
        c7  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[7]), b, c7);
        c8  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[8]), b, c8);
        c9  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[9]), b, c9);
        c10 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[10]), b, c10);
        c11 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[11]), b, c11);
        c12 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[12]), b, c12);
        c13 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[13]), b, c13);
        c14 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[14]), b, c14);
        c15 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[15]), b, c15);
        a_ptr += 16;
    }

    _mm512_storeu_ps(C + 0*ldc, c0);
    _mm512_storeu_ps(C + 1*ldc, c1);
    _mm512_storeu_ps(C + 2*ldc, c2);
    _mm512_storeu_ps(C + 3*ldc, c3);
    _mm512_storeu_ps(C + 4*ldc, c4);
    _mm512_storeu_ps(C + 5*ldc, c5);
    _mm512_storeu_ps(C + 6*ldc, c6);
    _mm512_storeu_ps(C + 7*ldc, c7);
    _mm512_storeu_ps(C + 8*ldc, c8);
    _mm512_storeu_ps(C + 9*ldc, c9);
    _mm512_storeu_ps(C + 10*ldc, c10);
    _mm512_storeu_ps(C + 11*ldc, c11);
    _mm512_storeu_ps(C + 12*ldc, c12);
    _mm512_storeu_ps(C + 13*ldc, c13);
    _mm512_storeu_ps(C + 14*ldc, c14);
    _mm512_storeu_ps(C + 15*ldc, c15);
}

__attribute__((always_inline, hot))
inline void kernel_16x16_masked(int k, const float* __restrict__ packedA, 
                                const float* __restrict__ packedB, 
                                float* __restrict__ C, int ldc, __mmask16 mask) {
    __m512 c0  = _mm512_maskz_loadu_ps(mask, C + 0*ldc);
    __m512 c1  = _mm512_maskz_loadu_ps(mask, C + 1*ldc);
    __m512 c2  = _mm512_maskz_loadu_ps(mask, C + 2*ldc);
    __m512 c3  = _mm512_maskz_loadu_ps(mask, C + 3*ldc);
    __m512 c4  = _mm512_maskz_loadu_ps(mask, C + 4*ldc);
    __m512 c5  = _mm512_maskz_loadu_ps(mask, C + 5*ldc);
    __m512 c6  = _mm512_maskz_loadu_ps(mask, C + 6*ldc);
    __m512 c7  = _mm512_maskz_loadu_ps(mask, C + 7*ldc);
    __m512 c8  = _mm512_maskz_loadu_ps(mask, C + 8*ldc);
    __m512 c9  = _mm512_maskz_loadu_ps(mask, C + 9*ldc);
    __m512 c10 = _mm512_maskz_loadu_ps(mask, C + 10*ldc);
    __m512 c11 = _mm512_maskz_loadu_ps(mask, C + 11*ldc);
    __m512 c12 = _mm512_maskz_loadu_ps(mask, C + 12*ldc);
    __m512 c13 = _mm512_maskz_loadu_ps(mask, C + 13*ldc);
    __m512 c14 = _mm512_maskz_loadu_ps(mask, C + 14*ldc);
    __m512 c15 = _mm512_maskz_loadu_ps(mask, C + 15*ldc);

    const float* b_ptr = packedB;
    const float* a_ptr = packedA;

    int p = 0;
    for (; p <= k - 8; p += 8) {
        __m512 b0 = _mm512_load_ps(b_ptr);
        __m512 b1 = _mm512_load_ps(b_ptr + 16);
        __m512 b2 = _mm512_load_ps(b_ptr + 32);
        __m512 b3 = _mm512_load_ps(b_ptr + 48);
        __m512 b4 = _mm512_load_ps(b_ptr + 64);
        __m512 b5 = _mm512_load_ps(b_ptr + 80);
        __m512 b6 = _mm512_load_ps(b_ptr + 96);
        __m512 b7 = _mm512_load_ps(b_ptr + 112);

        #define PROCESS_ROW(row) \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row]), b0, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 16]), b1, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 32]), b2, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 48]), b3, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 64]), b4, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 80]), b5, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 96]), b6, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 112]), b7, c##row);

        PROCESS_ROW(0) PROCESS_ROW(1) PROCESS_ROW(2) PROCESS_ROW(3)
        PROCESS_ROW(4) PROCESS_ROW(5) PROCESS_ROW(6) PROCESS_ROW(7)
        PROCESS_ROW(8) PROCESS_ROW(9) PROCESS_ROW(10) PROCESS_ROW(11)
        PROCESS_ROW(12) PROCESS_ROW(13) PROCESS_ROW(14) PROCESS_ROW(15)
        
        #undef PROCESS_ROW

        b_ptr += 128;
        a_ptr += 128;
    }
    
    for (; p < k; ++p) {
        __m512 b = _mm512_load_ps(b_ptr);
        b_ptr += 16;
        
        c0  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[0]), b, c0);
        c1  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[1]), b, c1);
        c2  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[2]), b, c2);
        c3  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[3]), b, c3);
        c4  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[4]), b, c4);
        c5  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[5]), b, c5);
        c6  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[6]), b, c6);
        c7  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[7]), b, c7);
        c8  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[8]), b, c8);
        c9  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[9]), b, c9);
        c10 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[10]), b, c10);
        c11 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[11]), b, c11);
        c12 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[12]), b, c12);
        c13 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[13]), b, c13);
        c14 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[14]), b, c14);
        c15 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[15]), b, c15);
        a_ptr += 16;
    }

    _mm512_mask_storeu_ps(C + 0*ldc, mask, c0);
    _mm512_mask_storeu_ps(C + 1*ldc, mask, c1);
    _mm512_mask_storeu_ps(C + 2*ldc, mask, c2);
    _mm512_mask_storeu_ps(C + 3*ldc, mask, c3);
    _mm512_mask_storeu_ps(C + 4*ldc, mask, c4);
    _mm512_mask_storeu_ps(C + 5*ldc, mask, c5);
    _mm512_mask_storeu_ps(C + 6*ldc, mask, c6);
    _mm512_mask_storeu_ps(C + 7*ldc, mask, c7);
    _mm512_mask_storeu_ps(C + 8*ldc, mask, c8);
    _mm512_mask_storeu_ps(C + 9*ldc, mask, c9);
    _mm512_mask_storeu_ps(C + 10*ldc, mask, c10);
    _mm512_mask_storeu_ps(C + 11*ldc, mask, c11);
    _mm512_mask_storeu_ps(C + 12*ldc, mask, c12);
    _mm512_mask_storeu_ps(C + 13*ldc, mask, c13);
    _mm512_mask_storeu_ps(C + 14*ldc, mask, c14);
    _mm512_mask_storeu_ps(C + 15*ldc, mask, c15);
}

__attribute__((always_inline))
inline void pack_A(int k, const float* A, int lda, int i0, int i_max, int p0, int p_max, float* packed) {
    (void)k;
    
    int indices[16];
    for(int i=0; i<16; ++i) indices[i] = i * lda;
    __m512i vindex = _mm512_loadu_si512(indices);

    if (i0 + 16 <= i_max) {
        for (int p = p0; p < p_max; ++p) {
            __m512 a = _mm512_i32gather_ps(vindex, &A[i0 * lda + p], 4);
            _mm512_store_ps(packed, a);
            packed += 16;
        }
    } else {
        __mmask16 mask = (1 << (i_max - i0)) - 1;
        for (int p = p0; p < p_max; ++p) {
            __m512 a = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, vindex, &A[i0 * lda + p], 4);
            _mm512_store_ps(packed, a);
            packed += 16;
        }
    }
}

__attribute__((always_inline))
inline void pack_B(int k, const float* B, int ldb, int p0, int p_max, int j0, int j_max, int nr, float* packed) {
    (void)k;
    if (nr == 16 && j0 + 16 <= j_max) {
        for (int p = p0; p < p_max; ++p) {
            __m512 b = _mm512_loadu_ps(&B[p * ldb + j0]);
            _mm512_store_ps(packed, b);
            packed += 16;
        }
    } else {
        for (int p = p0; p < p_max; ++p) {
            for (int j = 0; j < nr; ++j) {
                if (j0 + j < j_max) {
                    *packed++ = B[p * ldb + (j0 + j)];
                } else {
                    *packed++ = 0.0f;
                }
            }
        }
    }
}

__attribute__((always_inline))
inline float dot_product(const float* A, const float* B, int k) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    
    int p = 0;
    for (; p <= k - 64; p += 64) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(A + p), _mm512_loadu_ps(B + p), sum0);
        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(A + p + 16), _mm512_loadu_ps(B + p + 16), sum1);
        sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(A + p + 32), _mm512_loadu_ps(B + p + 32), sum2);
        sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(A + p + 48), _mm512_loadu_ps(B + p + 48), sum3);
    }
    
    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    
    for (; p <= k - 16; p += 16) {
        sum = _mm512_fmadd_ps(_mm512_loadu_ps(A + p), _mm512_loadu_ps(B + p), sum);
    }
    
    float res = _mm512_reduce_add_ps(sum);
    for (; p < k; ++p) {
        res += A[p] * B[p];
    }
    return res;
}

__attribute__((always_inline))
inline void scale_vector(const float* src, float scale, int n, float* dst) {
    __m512 s = _mm512_set1_ps(scale);
    int j = 0;
    for (; j <= n - 64; j += 64) {
        _mm512_storeu_ps(dst + j, _mm512_mul_ps(_mm512_loadu_ps(src + j), s));
        _mm512_storeu_ps(dst + j + 16, _mm512_mul_ps(_mm512_loadu_ps(src + j + 16), s));
        _mm512_storeu_ps(dst + j + 32, _mm512_mul_ps(_mm512_loadu_ps(src + j + 32), s));
        _mm512_storeu_ps(dst + j + 48, _mm512_mul_ps(_mm512_loadu_ps(src + j + 48), s));
    }
    for (; j <= n - 16; j += 16) {
        _mm512_storeu_ps(dst + j, _mm512_mul_ps(_mm512_loadu_ps(src + j), s));
    }
    for (; j < n; ++j) {
        dst[j] = src[j] * scale;
    }
}

// Block sizes tuned for Intel Xeon Gold 5218
static constexpr int MR = 16;
static constexpr int NR = 16;
static constexpr int MC = 256;
static constexpr int KC = 256;
static constexpr int NC = 256;

float* matrix_matrix_multiply(const float* A, int m, int k, const float* B, int n) {
    if (m == 1 && n == 1) {
        float* C = new float[1];
        C[0] = dot_product(A, B, k);
        return C;
    }

    if (m == 1 && k == 1) {
        float* C = new float[n];
        scale_vector(B, A[0], n, C);
        return C;
    }

    int m_padded = (m + MR - 1) & ~(MR - 1);
    float* C = new float[m_padded * n];
    std::fill(C, C + m_padded * n, 0.0f);

    static float* packedA = nullptr;
    static float* packedB = nullptr;
    static size_t packedA_size = 0;
    static size_t packedB_size = 0;
    
    size_t needed_A = (size_t)MC * KC;
    size_t needed_B = (size_t)KC * NC;
    
    if (packedA_size < needed_A) {
        if (packedA) free_aligned(packedA);
        packedA = (float*)alloc_aligned(needed_A * sizeof(float));
        packedA_size = needed_A;
    }
    if (packedB_size < needed_B) {
        if (packedB) free_aligned(packedB);
        packedB = (float*)alloc_aligned(needed_B * sizeof(float));
        packedB_size = needed_B;
    }

    for (int p0 = 0; p0 < k; p0 += KC) {
        int p_lim = min(k, p0 + KC);
        
        for (int i0 = 0; i0 < m_padded; i0 += MC) {
            int i_lim = min(m_padded, i0 + MC);
            
            for (int i = i0; i < i_lim; i += MR) {
                pack_A(k, A, k, i, m, p0, p_lim, &packedA[(i - i0) * (p_lim - p0)]);
            }
        }
        
        for (int j0 = 0; j0 < n; j0 += NC) {
            int j_lim = min(n, j0 + NC);

            float* b_pack_ptr = packedB;
            for (int j = j0; j < j_lim; j += NR) {
                int current_nr = min(NR, j_lim - j);
                pack_B(k, B, n, p0, p_lim, j, j + current_nr, 16, b_pack_ptr);
                b_pack_ptr += (p_lim - p0) * 16;
            }

            const float* p0_a_ptr = packedA;
            for (int i0 = 0; i0 < m_padded; i0 += MC) {
                int i_lim = min(m_padded, i0 + MC);
                
                const float* current_a_block = p0_a_ptr;
                size_t block_size = (size_t)(i_lim - i0) * (p_lim - p0);
                p0_a_ptr += block_size;

                float* current_b_ptr = packedB;
                for (int j = j0; j < j_lim; j += NR) {
                    int current_nr = min(NR, j_lim - j);
                    
                    if (current_nr == 16) {
                        for (int i = i0; i < i_lim; i += MR) {
                            kernel_16x16(p_lim - p0, &current_a_block[(i - i0) * (p_lim - p0)], current_b_ptr, &C[i * n + j], n);
                        }
                    } else {
                        __mmask16 mask = (1 << current_nr) - 1;
                        for (int i = i0; i < i_lim; i += MR) {
                            kernel_16x16_masked(p_lim - p0, &current_a_block[(i - i0) * (p_lim - p0)], current_b_ptr, &C[i * n + j], n, mask);
                        }
                    }
                    current_b_ptr += (p_lim - p0) * 16;
                }
            }
        }
    }
    
    return C;
}

__attribute__((always_inline, hot))
inline void kernel_16x1(int k, const float* packedA, const float* B, float* C) {
    __m512 c0 = _mm512_loadu_ps(C);
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    __m512 c3 = _mm512_setzero_ps();
    
    const float* a_ptr = packedA;
    const float* b_ptr = B;

    int p = 0;
    for (; p <= k - 4; p += 4) {
        __m512 b0 = _mm512_set1_ps(b_ptr[0]);
        __m512 a0 = _mm512_load_ps(a_ptr);
        c0 = _mm512_fmadd_ps(a0, b0, c0);

        __m512 b1 = _mm512_set1_ps(b_ptr[1]);
        __m512 a1 = _mm512_load_ps(a_ptr + 16);
        c1 = _mm512_fmadd_ps(a1, b1, c1);

        __m512 b2 = _mm512_set1_ps(b_ptr[2]);
        __m512 a2 = _mm512_load_ps(a_ptr + 32);
        c2 = _mm512_fmadd_ps(a2, b2, c2);

        __m512 b3 = _mm512_set1_ps(b_ptr[3]);
        __m512 a3 = _mm512_load_ps(a_ptr + 48);
        c3 = _mm512_fmadd_ps(a3, b3, c3);

        a_ptr += 64;
        b_ptr += 4;
    }
    
    __m512 c = _mm512_add_ps(_mm512_add_ps(c0, c1), _mm512_add_ps(c2, c3));

    for (; p < k; ++p) {
        __m512 b = _mm512_set1_ps(*b_ptr++);
        __m512 a = _mm512_load_ps(a_ptr);
        a_ptr += 16;
        c = _mm512_fmadd_ps(a, b, c);
    }
    
    _mm512_storeu_ps(C, c);
}

void matrix_matrix_multiply_prepacked(const float* packedA, int m, int k, const float* B, int n, float* C) {
    int m_padded = (m + MR - 1) & ~(MR - 1);
    std::fill(C, C + m_padded * n, 0.0f);

    static float* packedB = nullptr;
    static size_t packedB_size = 0;
    
    size_t needed_B = (size_t)KC * NC;
    if (packedB_size < needed_B) {
        if (packedB) free_aligned(packedB);
        packedB = (float*)alloc_aligned(needed_B * sizeof(float));
        packedB_size = needed_B;
    }

    if (n == 1) {
        const float* a_ptr = packedA;
        for (int p0 = 0; p0 < k; p0 += KC) {
            int p_lim = min(k, p0 + KC);
            for (int i0 = 0; i0 < m_padded; i0 += MC) {
                int i_lim = min(m_padded, i0 + MC);
                
                const float* current_a_block = a_ptr;
                size_t block_size = (size_t)(i_lim - i0) * (p_lim - p0);
                a_ptr += block_size;

                for (int i = i0; i < i_lim; i += MR) {
                    kernel_16x1(p_lim - p0, &current_a_block[(i - i0) * (p_lim - p0)], &B[p0], &C[i]);
                }
            }
        }
        return;
    }

    const float* a_ptr = packedA;

    for (int p0 = 0; p0 < k; p0 += KC) {
        int p_lim = min(k, p0 + KC);
        
        for (int j0 = 0; j0 < n; j0 += NC) {
            int j_lim = min(n, j0 + NC);

            float* b_pack_ptr = packedB;
            for (int j = j0; j < j_lim; j += NR) {
                int current_nr = min(NR, j_lim - j);
                pack_B(k, B, n, p0, p_lim, j, j + current_nr, 16, b_pack_ptr);
                b_pack_ptr += (p_lim - p0) * 16;
            }

            const float* p0_a_ptr = a_ptr;
            for (int i0 = 0; i0 < m_padded; i0 += MC) {
                int i_lim = min(m_padded, i0 + MC);
                
                const float* current_a_block = p0_a_ptr;
                size_t block_size = (size_t)(i_lim - i0) * (p_lim - p0);
                p0_a_ptr += block_size;

                float* current_b_ptr = packedB;
                for (int j = j0; j < j_lim; j += NR) {
                    int current_nr = min(NR, j_lim - j);
                    
                    if (current_nr == 16) {
                        for (int i = i0; i < i_lim; i += MR) {
                            kernel_16x16(p_lim - p0, &current_a_block[(i - i0) * (p_lim - p0)], current_b_ptr, &C[i * n + j], n);
                        }
                    } else {
                        __mmask16 mask = (1 << current_nr) - 1;
                        for (int i = i0; i < i_lim; i += MR) {
                            kernel_16x16_masked(p_lim - p0, &current_a_block[(i - i0) * (p_lim - p0)], current_b_ptr, &C[i * n + j], n, mask);
                        }
                    }
                    current_b_ptr += (p_lim - p0) * 16;
                }
            }
        }
        
        for (int i0 = 0; i0 < m_padded; i0 += MC) {
             int i_lim = min(m_padded, i0 + MC);
             a_ptr += (size_t)(i_lim - i0) * (p_lim - p0);
        }
    }
}

float* pack_matrix_A(int m, int k, const float* A) {
    int m_padded = (m + MR - 1) & ~(MR - 1);

    size_t total_size = 0;
    for (int p0 = 0; p0 < k; p0 += KC) {
        int p_lim = min(k, p0 + KC);
        for (int i0 = 0; i0 < m_padded; i0 += MC) {
            int i_lim = min(m_padded, i0 + MC);
            total_size += (size_t)(i_lim - i0) * (p_lim - p0);
        }
    }
    
    float* packed = (float*)alloc_aligned(total_size * sizeof(float));
    float* ptr = packed;
    
    for (int p0 = 0; p0 < k; p0 += KC) {
        int p_lim = min(k, p0 + KC);
        for (int i0 = 0; i0 < m_padded; i0 += MC) {
            int i_lim = min(m_padded, i0 + MC);
            
            for (int i = i0; i < i_lim; i += MR) {
                pack_A(k, A, k, i, m, p0, p_lim, ptr);
                ptr += MR * (p_lim - p0);
            }
        }
    }
    return packed;
}

// ==================== Neural Network Layers ====================

float* relu(const float* input, int size) {
    float* output = new float[size];
    __m512 zero = _mm512_setzero_ps();
    int i = 0;
    for (; i <= size - 64; i += 64) {
        _mm512_storeu_ps(output + i, _mm512_max_ps(zero, _mm512_loadu_ps(input + i)));
        _mm512_storeu_ps(output + i + 16, _mm512_max_ps(zero, _mm512_loadu_ps(input + i + 16)));
        _mm512_storeu_ps(output + i + 32, _mm512_max_ps(zero, _mm512_loadu_ps(input + i + 32)));
        _mm512_storeu_ps(output + i + 48, _mm512_max_ps(zero, _mm512_loadu_ps(input + i + 48)));
    }
    for (; i <= size - 16; i += 16) {
        _mm512_storeu_ps(output + i, _mm512_max_ps(zero, _mm512_loadu_ps(input + i)));
    }
    for (; i < size; i++) output[i] = max(0.0f, input[i]);
    return output;
}

struct Linear {
    int in_dim;
    int out_dim;
    float* W;
    float* packedW;

    mutable std::vector<float> workspace;

    Linear(int in_dim, int out_dim, float* weights)
        : in_dim(in_dim), out_dim(out_dim), W(weights) {
        packedW = pack_matrix_A(out_dim, in_dim, W);
    }

    ~Linear() { 
        delete[] W; 
        free_aligned(packedW);
    }

    float* forward(const float* x, int batch_size) const {
        if (batch_size == 1) {
            float* y = new float[out_dim];
            matrix_matrix_multiply_prepacked(packedW, out_dim, in_dim, x, 1, y);
            return y;
        }

        int m_padded = (out_dim + 15) & ~15;
        size_t required = (size_t)in_dim * batch_size + (size_t)m_padded * batch_size;
        if (workspace.size() < required) workspace.resize(required);
        
        float* x_transposed = workspace.data();
        float* y = workspace.data() + in_dim * batch_size;

        transpose(x, batch_size, in_dim, x_transposed);
        matrix_matrix_multiply_prepacked(packedW, out_dim, in_dim, x_transposed, batch_size, y);
        
        return transpose(y, out_dim, batch_size);
    }
};

struct LayerNorm {
    int dim;
    float* gamma;

    explicit LayerNorm(int dim) : dim(dim), gamma(new float[dim]) {
        for (int i = 0; i < dim; i++) gamma[i] = 1.0f;
    }

    ~LayerNorm() { delete[] gamma; }

    float* forward(const float* x) const {
        __m512 vsum = _mm512_setzero_ps();
        int i = 0;
        for (; i <= dim - 16; i += 16) {
            vsum = _mm512_add_ps(vsum, _mm512_loadu_ps(x + i));
        }
        float mean = _mm512_reduce_add_ps(vsum);
        for (; i < dim; i++) mean += x[i];
        mean /= dim;

        __m512 vmean = _mm512_set1_ps(mean);
        __m512 vvar = _mm512_setzero_ps();
        i = 0;
        for (; i <= dim - 16; i += 16) {
            __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(x + i), vmean);
            vvar = _mm512_fmadd_ps(diff, diff, vvar);
        }
        float var = _mm512_reduce_add_ps(vvar);
        for (; i < dim; i++) {
            float diff = x[i] - mean;
            var += diff * diff;
        }
        var /= dim;
        
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        __m512 vinv_std = _mm512_set1_ps(inv_std);

        float* y = new float[dim];
        i = 0;
        for (; i <= dim - 16; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 vg = _mm512_loadu_ps(gamma + i);
            __m512 normalized = _mm512_mul_ps(_mm512_sub_ps(vx, vmean), vinv_std);
            _mm512_storeu_ps(y + i, _mm512_mul_ps(vg, normalized));
        }
        for (; i < dim; i++) {
            y[i] = gamma[i] * (x[i] - mean) * inv_std;
        }
        return y;
    }
};

struct FeedForward {
    Linear fc1;
    Linear fc2;

    FeedForward(int d_model, int d_ff, float* fc1_weights, float* fc2_weights)
        : fc1(d_model, d_ff, fc1_weights), fc2(d_ff, d_model, fc2_weights) {}

    float* forward(const float* x, int batch_size) {
        float* h = fc1.forward(x, batch_size);
        float* r = relu(h, batch_size * fc1.out_dim);
        delete[] h;
        float* o = fc2.forward(r, batch_size);
        delete[] r;
        return o;
    }
};

struct SelfAttention {
    int d_model;
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear o_proj;

    SelfAttention(int d_model, [[maybe_unused]] int n_head_unused, float* Wq_weights, float* Wk_weights,
                  float* Wv_weights, float* Wo_weights)
        : d_model(d_model),
          q_proj(d_model, d_model, transpose(Wq_weights, d_model, d_model)),
          k_proj(d_model, d_model, transpose(Wk_weights, d_model, d_model)),
          v_proj(d_model, d_model, transpose(Wv_weights, d_model, d_model)),
          o_proj(d_model, d_model, transpose(Wo_weights, d_model, d_model)) {
        delete[] Wq_weights;
        delete[] Wk_weights;
        delete[] Wv_weights;
        delete[] Wo_weights;
    }

    float* forward(const float* x, int T) {
        float* Q = q_proj.forward(x, T);
        float* K = k_proj.forward(x, T);
        float* V = v_proj.forward(x, T);

        float* KT = transpose(K, T, d_model);
        float* scores = matrix_matrix_multiply(Q, T, d_model, KT, T);
        float scale = 1.0f / std::sqrt(static_cast<float>(d_model));
        for (int i = 0; i < T * T; i++) scores[i] *= scale;
        delete[] KT;

        float* attn = new float[T * T];
        for (int i = 0; i < T; i++) {
            float* row_softmax = softmax(scores + i * T, i + 1);
            for (int j = 0; j <= i; ++j) {
                attn[i * T + j] = row_softmax[j];
            }
            delete[] row_softmax;
            for (int j = i + 1; j < T; ++j) {
                attn[i * T + j] = 0.0f;
            }
        }
        delete[] scores;

        float* out = matrix_matrix_multiply(attn, T, T, V, d_model);
        delete[] attn;
        delete[] Q;
        delete[] K;
        delete[] V;

        float* out_proj = o_proj.forward(out, T);

        delete[] out;
        return out_proj;
    }
};

struct TransformerBlock {
    SelfAttention attn;
    FeedForward ffn;
    LayerNorm ln1;
    LayerNorm ln2;

    TransformerBlock(int d_model, int n_head, int d_ff, float* Wq_weights, float* Wk_weights,
                     float* Wv_weights, float* Wo_weights, float* fc1_weights, float* fc2_weights)
        : attn(d_model, n_head, Wq_weights, Wk_weights, Wv_weights, Wo_weights),
          ffn(d_model, d_ff, fc1_weights, fc2_weights),
          ln1(d_model),
          ln2(d_model) {}

    float* forward(const float* x, int T, int d_model) {
        float* x_norm = new float[T * d_model];
        for (int t = 0; t < T; t++) {
            float* norm = ln1.forward(x + t * d_model);
            for (int i = 0; i < d_model; i++) x_norm[t * d_model + i] = norm[i];
            delete[] norm;
        }

        float* attn_out = attn.forward(x_norm, T);
        delete[] x_norm;

        float* out = new float[T * d_model];
        int size = T * d_model;
        int i = 0;
        for (; i <= size - 16; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 va = _mm512_loadu_ps(attn_out + i);
            _mm512_storeu_ps(out + i, _mm512_add_ps(vx, va));
        }
        for (; i < size; i++) out[i] = x[i] + attn_out[i];
        delete[] attn_out;

        float* y_norm = new float[T * d_model];
        for (int t = 0; t < T; t++) {
            float* norm = ln2.forward(out + t * d_model);
            for (int i = 0; i < d_model; i++) y_norm[t * d_model + i] = norm[i];
            delete[] norm;
        }
        
        float* f = ffn.forward(y_norm, T);
        delete[] y_norm;
        
        float* final_out = new float[T * d_model];
        i = 0;
        for (; i <= size - 16; i += 16) {
            __m512 vo = _mm512_loadu_ps(out + i);
            __m512 vf = _mm512_loadu_ps(f + i);
            _mm512_storeu_ps(final_out + i, _mm512_add_ps(vo, vf));
        }
        for (; i < size; i++) final_out[i] = out[i] + f[i];
        
        delete[] f;
        delete[] out;
        return final_out;
    }
};

}  // namespace

struct GPTMini::Impl {
    int vocab_size;
    int d_model;
    int n_head;
    int d_ff;
    int n_layer;
    vector<unique_ptr<TransformerBlock>> blocks;
    float* embed_W;
    Linear lm_head;
    bool dump_enabled = false;
    std::string dump_dir;

    Impl(int vocab, int d_model, int n_head, int d_ff, int n_layer, float* embed_weights,
         float* lm_head_weights, const vector<GPTMini::BlockWeights>& block_weights)
        : vocab_size(vocab),
          d_model(d_model),
          n_head(n_head),
          d_ff(d_ff),
          n_layer(n_layer),
          blocks(),
          embed_W(embed_weights),
          lm_head(d_model, vocab, lm_head_weights) {
        blocks.reserve(n_layer);
        for (int i = 0; i < n_layer; i++) {
            const auto& bw = block_weights[i];
            blocks.emplace_back(std::make_unique<TransformerBlock>(d_model, n_head, d_ff, bw.Wq,
                                                                   bw.Wk, bw.Wv, bw.Wo, bw.fc1,
                                                                   bw.fc2));
        }
    }

    ~Impl() {
        delete[] embed_W;
    }

    void enable_layer_dumping(const std::string& directory) {
        utils::enable_layer_dumping(dump_enabled, dump_dir, directory);
    }

    void dump_layer_output(const float* data, int rows, int cols) {
        utils::dump_layer_output(dump_enabled, dump_dir, data, rows, cols);
    }

    float* embed_tokens(const vector<int>& tokens) {
        int T = tokens.size();
        float* x = new float[T * d_model];
        for (int t = 0; t < T; t++) {
            int id = tokens[t];
            for (int i = 0; i < d_model; i++) x[t * d_model + i] = embed_W[id * d_model + i];
        }
        return x;
    }

    int generate_next(const vector<int>& context) {
        int T = context.size();
        float* x = embed_tokens(context);
        for (int layer = 0; layer < n_layer; ++layer) {
            float* new_x = blocks[layer]->forward(x, T, d_model);
            delete[] x;
            x = new_x;
        }
        dump_layer_output(x, T, d_model);
        float* logits = lm_head.forward(x + (T - 1) * d_model, 1);
        float* probs = softmax(logits, vocab_size);
        int next = sample_from(probs, vocab_size);
        
        delete[] x;
        delete[] logits;
        delete[] probs;
        return next;
    }
};

GPTMini::GPTMini(int vocab, int d_model, int n_head, int d_ff, int n_layer,
                 float* embed_weights, float* lm_head_weights,
                 const vector<BlockWeights>& block_weights)
    : impl(new Impl(vocab, d_model, n_head, d_ff, n_layer, embed_weights, lm_head_weights,
                    block_weights)) {}

GPTMini::~GPTMini() { delete impl; }

int GPTMini::generate_next(const vector<int>& context) { return impl->generate_next(context); }

void GPTMini::enable_layer_dumping(const std::string& directory) {
    impl->enable_layer_dumping(directory);
}
