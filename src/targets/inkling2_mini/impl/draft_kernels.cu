#include "targets/inkling2_mini/impl/draft_kernels.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ninfer::targets::inkling2_mini::detail {
namespace {

constexpr int H = 2304;
constexpr int R = 256;
constexpr int D = 128;
constexpr int Q_HEADS = 24;
constexpr int KV_HEADS = 8;
constexpr int BLOCK = 8;

__device__ float bf16(uint16_t value) {
    return __bfloat162float(__ushort_as_bfloat16(value));
}

__global__ void prepare_embedding_kernel(const uint16_t* anchor, const uint16_t* mask,
                                         uint16_t* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H * BLOCK) { return; }
    const int token = i / H;
    out[i] = token == 0 ? anchor[i] : mask[i - token * H];
}

__global__ void mix_rmsnorm_kernel(const uint16_t* f0, const uint16_t* f1,
                                   const uint16_t* f2, const uint16_t* f3,
                                   const float* alpha, const uint16_t* norm,
                                   int token_begin, uint16_t* out) {
    const int token = blockIdx.x;
    const int lane = threadIdx.x;
    const uint16_t* features[4] = {f0, f1, f2, f3};
    const int offset = (token_begin + token) * H;
    float square_sum = 0.0F;
    for (int h = lane; h < H; h += 32) {
        float mixed = 0.0F;
        for (int tap = 0; tap < 4; ++tap) {
            mixed += alpha[tap] * bf16(features[tap][offset + h]);
        }
        square_sum += mixed * mixed;
    }
    #pragma unroll
    for (int delta = 16; delta; delta /= 2) {
        square_sum += __shfl_down_sync(0xffffffffU, square_sum, delta);
    }
    const float inverse_rms = rsqrtf(__shfl_sync(0xffffffffU, square_sum, 0) / H + 1.0e-6F);
    for (int h = lane; h < H; h += 32) {
        float mixed = 0.0F;
        for (int tap = 0; tap < 4; ++tap) {
            mixed += alpha[tap] * bf16(features[tap][offset + h]);
        }
        out[token * H + h] = __bfloat16_as_ushort(
            __float2bfloat16_rn(mixed * inverse_rms * bf16(norm[h])));
    }
}

__global__ void rope_kernel(uint16_t* q, uint16_t* k, const int* positions, int tokens,
                            bool has_query) {
    const int q_pairs = has_query ? tokens * Q_HEADS * (D / 2) : 0;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const bool query = index < q_pairs;
    const int pair_index = query ? index : index - q_pairs;
    const int heads = query ? Q_HEADS : KV_HEADS;
    const int pair = pair_index % (D / 2);
    const int head = (pair_index / (D / 2)) % heads;
    const int token = pair_index / ((D / 2) * heads);
    if (token >= tokens) { return; }
    uint16_t* base = (query ? q : k) + (token * heads + head) * D;
    const float position = static_cast<float>(positions[token]);
    const float frequency = powf(1000000.0F, -2.0F * pair / D);
    const float angle = position * frequency;
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float first = bf16(base[pair]);
    const float second = bf16(base[pair + D / 2]);
    base[pair] = __bfloat16_as_ushort(__float2bfloat16_rn(first * cosine - second * sine));
    base[pair + D / 2] = __bfloat16_as_ushort(
        __float2bfloat16_rn(first * sine + second * cosine));
}

__global__ void attention_kernel(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                                 const uint16_t* context_k, const uint16_t* context_v,
                                 int context_length, int context_capacity, bool local,
                                 float* scores, int score_stride, uint16_t* out) {
    const int head = blockIdx.x % Q_HEADS;
    const int token = blockIdx.x / Q_HEADS;
    const int kv_head = head / (Q_HEADS / KV_HEADS);
    const int lane = threadIdx.x;
    const int dim = lane * 4;
    const uint16_t* query = q + (token * Q_HEADS + head) * D + dim;
    float query_values[4];
    #pragma unroll
    for (int part = 0; part < 4; ++part) { query_values[part] = bf16(query[part]); }
    const int context_begin = local ? max(0, context_length - 1024) : 0;
    const int context_count = context_length - context_begin;
    const int keys = context_count + token + 1;
    float* score_row = scores + (token * Q_HEADS + head) * score_stride;
    float maximum = -INFINITY;
    for (int key = 0; key < keys; ++key) {
        const uint16_t* key_data;
        if (key < context_count) {
            key_data = context_k + (context_begin + key) * context_capacity + kv_head * D + dim;
        } else {
            key_data = k + ((key - context_count) * KV_HEADS + kv_head) * D + dim;
        }
        float dot = 0.0F;
        #pragma unroll
        for (int part = 0; part < 4; ++part) { dot += query_values[part] * bf16(key_data[part]); }
        #pragma unroll
        for (int delta = 16; delta; delta /= 2) {
            dot += __shfl_down_sync(0xffffffffU, dot, delta);
        }
        if (lane == 0) {
            const float score = dot * 0.08838834764831845F;
            score_row[key] = score;
            maximum = fmaxf(maximum, score);
        }
    }
    maximum = __shfl_sync(0xffffffffU, maximum, 0);
    float denominator = 0.0F;
    float accumulators[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (int key = 0; key < keys; ++key) {
        const float weight = expf(score_row[key] - maximum);
        denominator += weight;
        const uint16_t* value_data;
        if (key < context_count) {
            value_data = context_v + (context_begin + key) * context_capacity + kv_head * D + dim;
        } else {
            value_data = v + ((key - context_count) * KV_HEADS + kv_head) * D + dim;
        }
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            accumulators[part] += weight * bf16(value_data[part]);
        }
    }
    uint16_t* output = out + (token * Q_HEADS + head) * D + dim;
    #pragma unroll
    for (int part = 0; part < 4; ++part) {
        output[part] = __bfloat16_as_ushort(__float2bfloat16_rn(accumulators[part] / denominator));
    }
}

__global__ void rnn_inputs_kernel(const uint16_t* state, const int* previous_token,
                                  const uint16_t* markov, const uint16_t* hidden,
                                  const uint16_t* f_rnn, uint16_t* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * R + 2 * H) { return; }
    if (i < R) {
        out[i] = state[i];
    } else if (i < 2 * R) {
        const int r = i - R;
        out[i] = markov[previous_token[0] * R + r];
    } else if (i < 2 * R + H) {
        out[i] = hidden[i - 2 * R];
    } else {
        out[i] = f_rnn[i - 2 * R - H];
    }
}

__device__ float round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ void rnn_gates_kernel(const uint16_t* projection, uint16_t* state, uint16_t* latent) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) { return; }
    const float gate = round_bf16(1.0F / (1.0F + expf(-bf16(projection[r]))));
    const float kept = round_bf16(gate * bf16(state[r]));
    const float candidate = round_bf16(tanhf(bf16(projection[R + r])));
    const float blend = round_bf16(round_bf16(1.0F - gate) * candidate);
    state[r] = __bfloat16_as_ushort(__float2bfloat16_rn(kept + blend));
    latent[r] = __bfloat16_as_ushort(__float2bfloat16_rn(tanhf(bf16(projection[2 * R + r]))));
}

__global__ void fold_kernel(const uint16_t* lm_head, const uint16_t* hidden_out,
                            const uint16_t* hidden_bias, uint16_t* folded_weight,
                            uint16_t* folded_bias, int vocab) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= vocab * R) { return; }
    const int token = index / R;
    const int rank = index % R;
    const uint16_t* head = lm_head + token * H;
    float sum = 0.0F;
    float bias = 0.0F;
    for (int h = 0; h < H; ++h) {
        const float value = bf16(head[h]);
        sum += value * bf16(hidden_out[h * R + rank]);
        if (rank == 0) { bias += value * bf16(hidden_bias[h]); }
    }
    folded_weight[index] = __bfloat16_as_ushort(__float2bfloat16_rn(sum));
    if (rank == 0) { folded_bias[token] = __bfloat16_as_ushort(__float2bfloat16_rn(bias)); }
}

int div_up_int(int value, int divisor) { return (value + divisor - 1) / divisor; }

} // namespace

void draft_prepare_embedding(const Tensor& anchor, const Tensor& mask, Tensor& out,
                             cudaStream_t stream) {
    prepare_embedding_kernel<<<div_up_int(H * BLOCK, 256), 256, 0, stream>>>(
        static_cast<const uint16_t*>(anchor.data), static_cast<const uint16_t*>(mask.data),
        static_cast<uint16_t*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void draft_mix_rmsnorm(const std::array<Tensor, 4>& features, int token_begin, int tokens,
                       const Tensor& alpha_row, const Tensor& norm, Tensor& out,
                       cudaStream_t stream) {
    mix_rmsnorm_kernel<<<tokens, 32, 0, stream>>>(
        static_cast<const uint16_t*>(features[0].data), static_cast<const uint16_t*>(features[1].data),
        static_cast<const uint16_t*>(features[2].data), static_cast<const uint16_t*>(features[3].data),
        static_cast<const float*>(alpha_row.data), static_cast<const uint16_t*>(norm.data),
        token_begin, static_cast<uint16_t*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void draft_rope(Tensor& q, Tensor& k, const Tensor& positions, cudaStream_t stream) {
    const int tokens = q.ne[2];
    const int pairs = tokens * (Q_HEADS + KV_HEADS) * (D / 2);
    rope_kernel<<<div_up_int(pairs, 256), 256, 0, stream>>>(
        static_cast<uint16_t*>(q.data), static_cast<uint16_t*>(k.data),
        static_cast<const int*>(positions.data), tokens, true);
    CUDA_CHECK(cudaGetLastError());
}

void draft_rope_context_k(Tensor& k, const Tensor& positions, cudaStream_t stream) {
    const int tokens = k.ne[2];
    const int pairs = tokens * KV_HEADS * (D / 2);
    rope_kernel<<<div_up_int(pairs, 256), 256, 0, stream>>>(nullptr,
        static_cast<uint16_t*>(k.data), static_cast<const int*>(positions.data), tokens, false);
    CUDA_CHECK(cudaGetLastError());
}

void draft_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                     const Tensor& context_k, const Tensor& context_v,
                     int context_length, bool local, Tensor& scores, Tensor& out,
                     cudaStream_t stream) {
    attention_kernel<<<BLOCK * Q_HEADS, 32, 0, stream>>>(
        static_cast<const uint16_t*>(q.data), static_cast<const uint16_t*>(k.data),
        static_cast<const uint16_t*>(v.data), static_cast<const uint16_t*>(context_k.data),
        static_cast<const uint16_t*>(context_v.data), context_length, context_k.ne[0], local,
        static_cast<float*>(scores.data), scores.ne[0], static_cast<uint16_t*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void draft_rnn_inputs(const Tensor& state, const Tensor& previous_token,
                      const Weight& markov_embedding, const Tensor& hidden,
                      const Tensor& f_rnn, Tensor& out, cudaStream_t stream) {
    rnn_inputs_kernel<<<div_up_int(2 * R + 2 * H, 256), 256, 0, stream>>>(
        static_cast<const uint16_t*>(state.data), static_cast<const int*>(previous_token.data),
        static_cast<const uint16_t*>(markov_embedding.qdata), static_cast<const uint16_t*>(hidden.data),
        static_cast<const uint16_t*>(f_rnn.data), static_cast<uint16_t*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void draft_rnn_gates(const Tensor& projection, Tensor& state, Tensor& latent,
                     cudaStream_t stream) {
    rnn_gates_kernel<<<1, R, 0, stream>>>(static_cast<const uint16_t*>(projection.data),
        static_cast<uint16_t*>(state.data), static_cast<uint16_t*>(latent.data));
    CUDA_CHECK(cudaGetLastError());
}

void draft_fold_lm_head(const Weight& lm_head, const Weight& hidden_out,
                        const Tensor& hidden_bias, Weight& folded_weight, Tensor& folded_bias,
                        cudaStream_t stream) {
    const int vocab = lm_head.n;
    fold_kernel<<<div_up_int(vocab * R, 256), 256, 0, stream>>>(
        static_cast<const uint16_t*>(lm_head.qdata), static_cast<const uint16_t*>(hidden_out.qdata),
        static_cast<const uint16_t*>(hidden_bias.data),
        static_cast<uint16_t*>(const_cast<void*>(folded_weight.qdata)),
        static_cast<uint16_t*>(folded_bias.data), vocab);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::inkling2_mini::detail
