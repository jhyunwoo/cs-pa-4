#pragma once

#include <string>
#include <vector>

struct GPTMini {
    struct BlockWeights {
        float* Wq;
        float* Wk;
        float* Wv;
        float* Wo;
        float* fc1;
        float* fc2;
    };

    GPTMini(int vocab, int d_model, int n_head, int d_ff, int n_layer,
            float* embed_weights, float* lm_head_weights,
            const std::vector<BlockWeights>& block_weights);
    ~GPTMini();

    int generate_next(const std::vector<int>& context);
    void enable_layer_dumping(const std::string& directory);

private:
    struct Impl;
    Impl* impl;
};