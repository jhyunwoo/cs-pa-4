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
    if (posix_memalign(&ptr, 32, size)) return nullptr;
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

float* transpose(const float* M, int rows, int cols) {
    float* T = new float[cols * rows];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            T[j * rows + i] = M[i * cols + j];
        }
    }
    return T;
}

// === Need to Optimize === //

void kernel_16x16(int k, const float* packedA, const float* packedB, float* C, int ldc) {
    __m512 c[16];
    
    // Load C rows (contiguous)
    for (int i = 0; i < 16; ++i) {
        c[i] = _mm512_loadu_ps(C + i * ldc);
    }

    const float* b_ptr = packedB;
    const float* a_ptr = packedA;

    for (int p = 0; p < k; ++p) {
        // Load B row (contiguous)
        __m512 b = _mm512_load_ps(b_ptr);
        b_ptr += 16;

        #pragma GCC unroll 16
        for (int i = 0; i < 16; ++i) {
            // Broadcast A[i, p]
            __m512 a = _mm512_set1_ps(a_ptr[i]);
            c[i] = _mm512_fmadd_ps(a, b, c[i]);
        }
        a_ptr += 16;
    }

    // Store C rows (contiguous)
    for (int i = 0; i < 16; ++i) {
        _mm512_storeu_ps(C + i * ldc, c[i]);
    }
}

void kernel_16x1(int k, const float* packedA, const float* packedB, float* C, int ldc) {
    if (ldc == 1) {
        __m512 c0 = _mm512_loadu_ps(C);
        for (int p = 0; p < k; ++p) {
            __m512 a0 = _mm512_load_ps(&packedA[p * 16]);
            __m512 b = _mm512_set1_ps(packedB[p]);
            c0 = _mm512_fmadd_ps(a0, b, c0);
        }
        _mm512_storeu_ps(C, c0);
        return;
    }

    // Prepare indices for gather/scatter
    int indices[16];
    for(int i=0; i<16; ++i) indices[i] = i * ldc;
    __m512i vindex = _mm512_loadu_si512(indices);

    // Load C (16 rows, 1 col)
    // C[0,0], C[1,0]... stride ldc
    // We can use the same gather with &C[0]
    __m512 c0 = _mm512_i32gather_ps(vindex, C, 4);

    for (int p = 0; p < k; ++p) {
        __m512 a0 = _mm512_load_ps(&packedA[p * 16]);
        __m512 b = _mm512_set1_ps(packedB[p]);
        c0 = _mm512_fmadd_ps(a0, b, c0);
    }

    _mm512_i32scatter_ps(C, vindex, c0, 4);
}

void pack_A(int k, const float* A, int lda, int i0, int i_max, int p0, int p_max, float* packed) {
    (void)k;
    int mr = 16;
    for (int p = p0; p < p_max; ++p) {
        for (int i = 0; i < mr; ++i) {
            if (i0 + i < i_max) {
                *packed++ = A[(i0 + i) * lda + p];
            } else {
                *packed++ = 0.0f;
            }
        }
    }
}

void pack_B(int k, const float* B, int ldb, int p0, int p_max, int j0, int j_max, int nr, float* packed) {
    (void)k;
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
                
                for (; j < j_lim; ++j) {
                    pack_B(k, B, n, p0, p_lim, j, j + 1, 1, packedB);
                    
                    for (int i = i0; i < i_lim; i += MR) {
                        kernel_16x1(p_lim - p0, &packedA[(i - i0) * (p_lim - p0)], packedB, &C[i * n + j], n);
                    }
                }
            }
        }
    }
    
    return C;
}

float* matrix_matrix_multiply_prepacked(const float* packedA, int m, int k, const float* B, int n) {
    const int MR = 16;
    const int NR = 16;
    
    int m_padded = (m + MR - 1) & ~(MR - 1);
    float* C = new float[m_padded * n];
    std::fill(C, C + m_padded * n, 0.0f);

    const int MC = 256;
    const int KC = 256;
    const int NC = 144;

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
                
                for (; j < j_lim; ++j) {
                    pack_B(k, B, n, p0, p_lim, j, j + 1, 1, packedB);
                    
                    for (int i = i0; i < i_lim; i += MR) {
                        kernel_16x1(p_lim - p0, &current_a_block[(i - i0) * (p_lim - p0)], packedB, &C[i * n + j], n);
                    }
                }
            }
        }
    }
    
    return C;
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

    Linear(int in_dim, int out_dim, float* weights)
        : in_dim(in_dim), out_dim(out_dim), W(weights) {
        packedW = pack_matrix_A(out_dim, in_dim, W);
    }

    ~Linear() { 
        delete[] W; 
        free_aligned(packedW);
    }

    float* forward(const float* x, int batch_size) const {
        float* x_transposed = transpose(x, batch_size, in_dim);
        // Use pre-packed weights
        float* y = matrix_matrix_multiply_prepacked(packedW, out_dim, in_dim, x_transposed, batch_size);
        delete[] x_transposed;
        float* y_transposed = transpose(y, out_dim, batch_size);
        delete[] y;
        return y_transposed;
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
    float* Wq;
    float* Wk;
    float* Wv;
    float* Wo;

    SelfAttention(int d_model, [[maybe_unused]] int n_head_unused, float* Wq_weights, float* Wk_weights,
                  float* Wv_weights, float* Wo_weights)
        : d_model(d_model),
          Wq(Wq_weights),
          Wk(Wk_weights),
          Wv(Wv_weights),
          Wo(Wo_weights) {}

    ~SelfAttention() {
        delete[] Wq;
        delete[] Wk;
        delete[] Wv;
    }

    float* forward(const float* x, int T) {
        float* Q = matrix_matrix_multiply(x, T, d_model, Wq, d_model);
        float* K = matrix_matrix_multiply(x, T, d_model, Wk, d_model);
        float* V = matrix_matrix_multiply(x, T, d_model, Wv, d_model);

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

        float* out_proj = matrix_matrix_multiply(out, T, d_model, Wo, d_model);

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