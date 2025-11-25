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

// Helper for aligned allocation
void* alloc_aligned(size_t size) {
    void* ptr;
    if (posix_memalign(&ptr, 64, size)) return nullptr;
    return ptr;
}

void free_aligned(void* ptr) {
    free(ptr);
}

using std::max;
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
    for (int i = 0; i < size; i++) y[i] /= sum;
    return y;
}

void transpose(const float* M, int rows, int cols, float* T) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            T[j * rows + i] = M[i * cols + j];
        }
    }
}

float* transpose(const float* M, int rows, int cols) {
    float* T = new float[cols * rows];
    transpose(M, rows, cols, T);
    return T;
}

// === Need to Optimize === //

void kernel_16x16(int k, const float* packedA, const float* packedB, float* C, int ldc) {
    __m512 c[16];
    
    // Load C rows (contiguous)
    for (int i = 0; i < 16; ++i) c[i] = _mm512_loadu_ps(C + i * ldc);

    const float* b_ptr = packedB;
    const float* a_ptr = packedA;

    int p = 0;
    for (; p <= k - 8; p += 8) {
        // Load B rows (contiguous)
        __m512 b0 = _mm512_load_ps(b_ptr);
        __m512 b1 = _mm512_load_ps(b_ptr + 16);
        __m512 b2 = _mm512_load_ps(b_ptr + 32);
        __m512 b3 = _mm512_load_ps(b_ptr + 48);
        __m512 b4 = _mm512_load_ps(b_ptr + 64);
        __m512 b5 = _mm512_load_ps(b_ptr + 80);
        __m512 b6 = _mm512_load_ps(b_ptr + 96);
        __m512 b7 = _mm512_load_ps(b_ptr + 112);
        b_ptr += 128;
        
        #pragma GCC unroll 16
        for (int i = 0; i < 16; ++i) {
            // Broadcast A[i, p...p+7]
            __m512 a0 = _mm512_set1_ps(a_ptr[i]);
            __m512 a1 = _mm512_set1_ps(a_ptr[i + 16]);
            __m512 a2 = _mm512_set1_ps(a_ptr[i + 32]);
            __m512 a3 = _mm512_set1_ps(a_ptr[i + 48]);
            __m512 a4 = _mm512_set1_ps(a_ptr[i + 64]);
            __m512 a5 = _mm512_set1_ps(a_ptr[i + 80]);
            __m512 a6 = _mm512_set1_ps(a_ptr[i + 96]);
            __m512 a7 = _mm512_set1_ps(a_ptr[i + 112]);
            
            c[i] = _mm512_fmadd_ps(a0, b0, c[i]);
            c[i] = _mm512_fmadd_ps(a1, b1, c[i]);
            c[i] = _mm512_fmadd_ps(a2, b2, c[i]);
            c[i] = _mm512_fmadd_ps(a3, b3, c[i]);
            c[i] = _mm512_fmadd_ps(a4, b4, c[i]);
            c[i] = _mm512_fmadd_ps(a5, b5, c[i]);
            c[i] = _mm512_fmadd_ps(a6, b6, c[i]);
            c[i] = _mm512_fmadd_ps(a7, b7, c[i]);
        }
        a_ptr += 128;
    }
    
    // Cleanup loop
    for (; p < k; ++p) {
        __m512 b = _mm512_load_ps(b_ptr);
        b_ptr += 16;
        
        #pragma GCC unroll 16
        for (int i = 0; i < 16; ++i) {
            __m512 a = _mm512_set1_ps(a_ptr[i]);
            c[i] = _mm512_fmadd_ps(a, b, c[i]);
        }
        a_ptr += 16;
    }

    // Store C rows (contiguous)
    for (int i = 0; i < 16; ++i) _mm512_storeu_ps(C + i * ldc, c[i]);
}

void kernel_16x16_masked(int k, const float* packedA, const float* packedB, float* C, int ldc, __mmask16 mask) {
    __m512 c[16];
    
    // Load C rows (contiguous)
    for (int i = 0; i < 16; ++i) c[i] = _mm512_maskz_loadu_ps(mask, C + i * ldc);

    const float* b_ptr = packedB;
    const float* a_ptr = packedA;

    int p = 0;
    for (; p <= k - 8; p += 8) {
        // Load B rows (contiguous)
        __m512 b0 = _mm512_load_ps(b_ptr);
        __m512 b1 = _mm512_load_ps(b_ptr + 16);
        __m512 b2 = _mm512_load_ps(b_ptr + 32);
        __m512 b3 = _mm512_load_ps(b_ptr + 48);
        __m512 b4 = _mm512_load_ps(b_ptr + 64);
        __m512 b5 = _mm512_load_ps(b_ptr + 80);
        __m512 b6 = _mm512_load_ps(b_ptr + 96);
        __m512 b7 = _mm512_load_ps(b_ptr + 112);
        b_ptr += 128;
        
        #pragma GCC unroll 16
        for (int i = 0; i < 16; ++i) {
            // Broadcast A[i, p...p+7]
            __m512 a0 = _mm512_set1_ps(a_ptr[i]);
            __m512 a1 = _mm512_set1_ps(a_ptr[i + 16]);
            __m512 a2 = _mm512_set1_ps(a_ptr[i + 32]);
            __m512 a3 = _mm512_set1_ps(a_ptr[i + 48]);
            __m512 a4 = _mm512_set1_ps(a_ptr[i + 64]);
            __m512 a5 = _mm512_set1_ps(a_ptr[i + 80]);
            __m512 a6 = _mm512_set1_ps(a_ptr[i + 96]);
            __m512 a7 = _mm512_set1_ps(a_ptr[i + 112]);
            
            c[i] = _mm512_fmadd_ps(a0, b0, c[i]);
            c[i] = _mm512_fmadd_ps(a1, b1, c[i]);
            c[i] = _mm512_fmadd_ps(a2, b2, c[i]);
            c[i] = _mm512_fmadd_ps(a3, b3, c[i]);
            c[i] = _mm512_fmadd_ps(a4, b4, c[i]);
            c[i] = _mm512_fmadd_ps(a5, b5, c[i]);
            c[i] = _mm512_fmadd_ps(a6, b6, c[i]);
            c[i] = _mm512_fmadd_ps(a7, b7, c[i]);
        }
        a_ptr += 128;
    }
    
    // Cleanup loop
    for (; p < k; ++p) {
        __m512 b = _mm512_load_ps(b_ptr);
        b_ptr += 16;
        
        #pragma GCC unroll 16
        for (int i = 0; i < 16; ++i) {
            __m512 a = _mm512_set1_ps(a_ptr[i]);
            c[i] = _mm512_fmadd_ps(a, b, c[i]);
        }
        a_ptr += 16;
    }

    // Store C rows (contiguous)
    for (int i = 0; i < 16; ++i) _mm512_mask_storeu_ps(C + i * ldc, mask, c[i]);
}

void pack_A(int k, const float* A, int lda, int i0, int i_max, int p0, int p_max, float* packed) {
    (void)k;
    int mr = 16;
    
    // Prepare gather indices
    int indices[16];
    for(int i=0; i<16; ++i) indices[i] = i * lda;
    __m512i vindex = _mm512_loadu_si512(indices);

    if (i0 + 16 <= i_max) {
        // Fast path: unmasked gather
        for (int p = p0; p < p_max; ++p) {
            __m512 a = _mm512_i32gather_ps(vindex, &A[i0 * lda + p], 4);
            _mm512_store_ps(packed, a);
            packed += 16;
        }
    } else {
        // Slow path: masked gather
        __mmask16 mask = (1 << (i_max - i0)) - 1;
        for (int p = p0; p < p_max; ++p) {
            __m512 a = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, vindex, &A[i0 * lda + p], 4);
            _mm512_store_ps(packed, a);
            packed += 16;
        }
    }
}

void pack_B(int k, const float* B, int ldb, int p0, int p_max, int j0, int j_max, int nr, float* packed) {
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

float* matrix_matrix_multiply(const float* A, int m, int k, const float* B, int n) {
    const int MR = 16;
    const int NR = 16;
    
    int m_padded = (m + MR - 1) & ~(MR - 1);
    float* C = new float[m_padded * n];
    std::fill(C, C + m_padded * n, 0.0f);

    const int MC = 256;
    const int KC = 256;
    const int NC = 144;

    static float* packedA = (float*)alloc_aligned(MC * KC * sizeof(float));
    static float* packedB = (float*)alloc_aligned(KC * NC * sizeof(float));

    for (int p0 = 0; p0 < k; p0 += KC) {
        int p_lim = std::min(k, p0 + KC);
        
        for (int i0 = 0; i0 < m_padded; i0 += MC) {
            int i_lim = std::min(m_padded, i0 + MC);
            
            for (int i = i0; i < i_lim; i += MR) {
                pack_A(k, A, k, i, m, p0, p_lim, &packedA[(i - i0) * (p_lim - p0)]); // Pass m, not i_lim
            }

            for (int j0 = 0; j0 < n; j0 += NC) {
                int j_lim = std::min(n, j0 + NC);

                int j = j0;
                for (; j <= j_lim - NR; j += NR) {
                    pack_B(k, B, n, p0, p_lim, j, j + NR, NR, packedB);
                    
                    for (int i = i0; i < i_lim; i += MR) {
                        kernel_16x16(p_lim - p0, &packedA[(i - i0) * (p_lim - p0)], packedB, &C[i * n + j], n);
                    }
                }
                
                if (j < j_lim) {
                    int remain = j_lim - j;
                    pack_B(k, B, n, p0, p_lim, j, j + remain, 16, packedB); // Pad to 16
                    __mmask16 mask = (1 << remain) - 1;
                    
                    for (int i = i0; i < i_lim; i += MR) {
                        kernel_16x16_masked(p_lim - p0, &packedA[(i - i0) * (p_lim - p0)], packedB, &C[i * n + j], n, mask);
                    }
                }
            }
        }
    }
    
    return C;
}

void kernel_16x1(int k, const float* packedA, const float* B, float* C) {
    __m512 c = _mm512_loadu_ps(C);
    
    const float* a_ptr = packedA;
    const float* b_ptr = B;

    for (int p = 0; p < k; ++p) {
        __m512 b = _mm512_set1_ps(*b_ptr++);
        __m512 a = _mm512_load_ps(a_ptr);
        a_ptr += 16;
        c = _mm512_fmadd_ps(a, b, c);
    }
    
    _mm512_storeu_ps(C, c);
}

void matrix_matrix_multiply_prepacked(const float* packedA, int m, int k, const float* B, int n, float* C) {
    const int MR = 16;
    const int NR = 16;
    
    int m_padded = (m + MR - 1) & ~(MR - 1);
    // C must be pre-allocated with size m_padded * n
    std::fill(C, C + m_padded * n, 0.0f);

    const int MC = 256;
    const int KC = 256;
    const int NC = 144;

    if (n == 1) {
        // Optimized path for GEMV (n=1)
        for (int p0 = 0; p0 < k; p0 += KC) {
            int p_lim = std::min(k, p0 + KC);
            
            for (int i0 = 0; i0 < m_padded; i0 += MC) {
                int i_lim = std::min(m_padded, i0 + MC);
                
                const float* current_a_block = packedA + ((size_t)i0 / MC * (k / KC) + p0 / KC) * (MC * KC); 
                // Wait, packedA layout is flat?
                // pack_matrix_A packs in blocks: p0 loop, then i0 loop.
                // Let's check pack_matrix_A layout.
                // It iterates p0, then i0.
                // So blocks are ordered by (p0, i0).
                // Block size: (i_lim - i0) * (p_lim - p0).
                // We need to track pointer carefully.
            }
        }
        
        // Re-implementing loop to match pack_matrix_A order
        const float* a_ptr = packedA;
        for (int p0 = 0; p0 < k; p0 += KC) {
            int p_lim = std::min(k, p0 + KC);
            for (int i0 = 0; i0 < m_padded; i0 += MC) {
                int i_lim = std::min(m_padded, i0 + MC);
                
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

    static float* packedB = (float*)alloc_aligned(KC * NC * sizeof(float));

    const float* a_ptr = packedA;

    for (int p0 = 0; p0 < k; p0 += KC) {
        int p_lim = std::min(k, p0 + KC);
        
        for (int i0 = 0; i0 < m_padded; i0 += MC) {
            int i_lim = std::min(m_padded, i0 + MC);
            
            const float* current_a_block = a_ptr;
            size_t block_size = (size_t)(i_lim - i0) * (p_lim - p0);
            a_ptr += block_size;

            for (int j0 = 0; j0 < n; j0 += NC) {
                int j_lim = std::min(n, j0 + NC);

                int j = j0;
                for (; j <= j_lim - NR; j += NR) {
                    pack_B(k, B, n, p0, p_lim, j, j + NR, NR, packedB);
                    
                    for (int i = i0; i < i_lim; i += MR) {
                        kernel_16x16(p_lim - p0, &current_a_block[(i - i0) * (p_lim - p0)], packedB, &C[i * n + j], n);
                    }
                }
                
                if (j < j_lim) {
                    int remain = j_lim - j;
                    pack_B(k, B, n, p0, p_lim, j, j + remain, 16, packedB); // Pad to 16
                    __mmask16 mask = (1 << remain) - 1;
                    
                    for (int i = i0; i < i_lim; i += MR) {
                        kernel_16x16_masked(p_lim - p0, &current_a_block[(i - i0) * (p_lim - p0)], packedB, &C[i * n + j], n, mask);
                    }
                }
            }
        }
    }
}

float* pack_matrix_A(int m, int k, const float* A) {
    const int MC = 256;
    const int KC = 256;
    const int MR = 16;
    
    int m_padded = (m + MR - 1) & ~(MR - 1);

    size_t total_size = 0;
    for (int p0 = 0; p0 < k; p0 += KC) {
        int p_lim = std::min(k, p0 + KC);
        for (int i0 = 0; i0 < m_padded; i0 += MC) {
            int i_lim = std::min(m_padded, i0 + MC);
            total_size += (size_t)(i_lim - i0) * (p_lim - p0);
        }
    }
    
    float* packed = (float*)alloc_aligned(total_size * sizeof(float));
    float* ptr = packed;
    
    for (int p0 = 0; p0 < k; p0 += KC) {
        int p_lim = std::min(k, p0 + KC);
        for (int i0 = 0; i0 < m_padded; i0 += MC) {
            int i_lim = std::min(m_padded, i0 + MC);
            
            for (int i = i0; i < i_lim; i += MR) {
                pack_A(k, A, k, i, m, p0, p_lim, ptr);
                ptr += MR * (p_lim - p0);
            }
        }
    }
    return packed;
}

// ======================= //

float* relu(const float* input, int size) {
    float* output = new float[size];
    for (int i = 0; i < size; i++) output[i] = max(0.0f, input[i]);
    return output;
}

struct Linear {
    int in_dim;
    int out_dim;
    float* W;
    float* packedW; // Pre-packed weights

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
            // Optimization for generation phase:
            // x (1 x in) and x^T (in x 1) have the same memory layout.
            // y (out x 1) and y^T (1 x out) have the same memory layout.
            // We can skip transposes and intermediate buffers.
            float* y = new float[out_dim];
            matrix_matrix_multiply_prepacked(packedW, out_dim, in_dim, x, 1, y);
            return y;
        }

        // Calculate required size: in_dim * batch_size + out_dim * batch_size
        // We need space for x_transposed (in * batch) and y (out * batch)
        // Note: m_padded might be slightly larger than out_dim, but we can just alloc enough.
        int m_padded = (out_dim + 15) & ~15;
        size_t required = (size_t)in_dim * batch_size + (size_t)m_padded * batch_size;
        if (workspace.size() < required) workspace.resize(required);
        
        float* x_transposed = workspace.data();
        float* y = workspace.data() + in_dim * batch_size;

        transpose(x, batch_size, in_dim, x_transposed);
        
        // Use pre-packed weights
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
        float mean = 0, var = 0;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= dim;
        for (int i = 0; i < dim; i++) var += (x[i] - mean) * (x[i] - mean);
        var /= dim;
        float stdv = std::sqrt(var + 1e-5f);
        float* y = new float[dim];
        for (int i = 0; i < dim; i++) y[i] = gamma[i] * (x[i] - mean) / stdv;
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
        // Linear takes ownership of the transposed weights.
        // We must delete the original weights passed in.
        delete[] Wq_weights;
        delete[] Wk_weights;
        delete[] Wv_weights;
        delete[] Wo_weights;
    }

    // Linear destructors will handle cleanup of their weights

    float* forward(const float* x, int T) {
        float* Q = q_proj.forward(x, T);
        float* K = k_proj.forward(x, T);
        float* V = v_proj.forward(x, T);

        float* KT = transpose(K, T, d_model);
        float* scores = matrix_matrix_multiply(Q, T, d_model, KT, T);
        for (int i = 0; i < T * T; i++) scores[i] /= std::sqrt(static_cast<float>(d_model));
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
        for (int i = 0; i < T * d_model; i++) out[i] = x[i] + attn_out[i];
        delete[] attn_out;

        float* final_out = new float[T * d_model];
        float* y_norm = new float[T * d_model];
        for (int t = 0; t < T; t++) {
            float* norm = ln2.forward(out + t * d_model);
            for (int i = 0; i < d_model; i++) y_norm[t * d_model + i] = norm[i];
            delete[] norm;
        }
        
        float* f = ffn.forward(y_norm, T);
        delete[] y_norm;
        for (int i = 0; i < T * d_model; i++) final_out[i] = out[i] + f[i];
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
          embed_W(embed_weights), // Changed initialization
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
