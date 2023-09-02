#pragma once
#include <ncnn/net.h>
#include <ncnn/layer_type.h>
#include <vector>
#include <iostream>

#define DEBUG_TIME 1

#if DEBUG_TIME
#include <chrono>
#endif

inline void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int z=0; z<m.d; z++)
        {
            for (int y=0; y<m.h; y++)
            {
                for (int x=0; x<10; x++)
                {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");
    }
}

namespace rwkv {

enum float_mode_t {
    fp32 = 1,
    fp16,
    bf16,
};

struct model_args_t {
    enum float_mode_t float_mode;
    int vocab_size;
    int layer_num;
    int embd_num;
    int ctx_len;
    int rescale_layer;
    char model_bin_path[64];
    char model_param_path[64];
    char emb_weights_path[64];
};

struct runtime_args_t {
    float temp;
    float top_p;
    float alpha_presence;
    float alpha_frequency;
    float penalty_decay;
    int end_of_text;
    int end_of_line;

    int chat_len_short;
    int chat_len_long;
    int free_gen_len;
};

static inline ncnn::Mat convert_fp32_to_fp16(ncnn::Mat &m, const ncnn::Option& opt) {
    ncnn::Layer* cast = ncnn::create_layer(ncnn::LayerType::Cast);
    ncnn::ParamDict pd;
    ncnn::Mat out;
    pd.set(0, 1);
    pd.set(1, 2);
    cast->load_param(pd);
    cast->create_pipeline(opt);
    cast->forward(m, out, opt);
    cast->destroy_pipeline(opt);

    delete cast;
    return out.clone();
}

class RWKV {
public:
    ncnn::Mat state;
#if DEBUG_TIME
    std::vector<int64_t> time_data;
#endif

    RWKV(model_args_t *args);

    int load_model_files();

    inline ncnn::Mat forward(int token) {
    #if DEBUG_TIME
        auto start = std::chrono::system_clock::now();
    #endif
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", emb_weights[token]);
        ex.input("in1", state);

        ncnn::Mat out;
        ex.extract("out0", out);
        ex.extract("out1", state);
    #if DEBUG_TIME
        auto forward_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - start);
        time_data.push_back(forward_time.count());
    #endif
        return out;
    }

    inline ncnn::Mat forward(std::vector<int> tokens) {
        ncnn::Mat out;
        for(int i = 0; i < tokens.size(); i++) {
            if(i == tokens.size() - 1)
                out = forward(tokens[i]);
            else
                forward(tokens[i]);
        }
        return out;
    }

    int sample_logits(ncnn::Mat logits, float temp = 1.0, float top_p = 0.85, float top_k = 0);

private:
    ncnn::Net net;
    model_args_t *args;
    ncnn::Layer *softmax;
    ncnn::Layer *cumsum;

    std::vector<ncnn::Mat> emb_weights;
};

#define DECLARE_WEIGHT(name) \
    ncnn::Mat name; \
    ncnn::Mat name##_shape;

#define LOAD_WEIGHT_PARAM(to, id) \
    to##_shape = pd.get(id, ncnn::Mat());

#define LOAD_WEIGHT_DATA(to, args...) \
    ptr = to##_shape; \
    to = mb.load(args, 1); \
    if(model_args->float_mode == fp16) { \
        to = convert_fp32_to_fp16(to, ncnn::Option());\
    } \
    if(to.empty()) \
        return -100;

class RWKV_Time_Mixing : public ncnn::Layer {
public:
    model_args_t *model_args;

    RWKV_Time_Mixing(model_args_t *args);

    virtual int load_param(const ncnn::ParamDict& pd);

    virtual int load_model(const ncnn::ModelBin& mb);

    virtual int forward_inplace(std::vector<ncnn::Mat>& bottom_top_blobs, const ncnn::Option& opt) const;

    virtual int create_pipeline(const ncnn::Option& opt);

    virtual int destroy_pipeline(const ncnn::Option& opt);

private:
    // weights
    DECLARE_WEIGHT(time_mix_k)
    DECLARE_WEIGHT(time_mix_v)
    DECLARE_WEIGHT(time_mix_r)

    DECLARE_WEIGHT(time_first)
    DECLARE_WEIGHT(time_decay)
    DECLARE_WEIGHT(rw)
    DECLARE_WEIGHT(kw)
    DECLARE_WEIGHT(vw)
    DECLARE_WEIGHT(ow)

    ncnn::Mat _time_mix_k;
    ncnn::Mat _time_mix_v;
    ncnn::Mat _time_mix_r;

    ncnn::Mat mix(ncnn::Mat in0, ncnn::Mat in1, ncnn::Mat param, ncnn::Mat _param, const ncnn::Option& opt) const;

    ncnn::Layer *sigmoid;
    ncnn::Layer *matmul;
    ncnn::Layer *add;
    ncnn::Layer *sub;
    ncnn::Layer *mul;
    ncnn::Layer *div_op;
    ncnn::Layer *one_sub;
    ncnn::Layer *max;
    ncnn::Layer *exp;
    ncnn::Layer *layernorm;

    int layer_num;
};

class RWKV_Channel_Mixing : public ncnn::Layer {
public:
    model_args_t *model_args;

    RWKV_Channel_Mixing(model_args_t *args);

    virtual int load_param(const ncnn::ParamDict& pd);

    virtual int load_model(const ncnn::ModelBin& mb);

    virtual int forward_inplace(std::vector<ncnn::Mat>& bottom_top_blobs, const ncnn::Option& opt) const;

    virtual int create_pipeline(const ncnn::Option& opt);

    virtual int destroy_pipeline(const ncnn::Option& opt);

private:
    // weights
    DECLARE_WEIGHT(time_mix_k)
    DECLARE_WEIGHT(time_mix_r)
    DECLARE_WEIGHT(rw)
    DECLARE_WEIGHT(kw)
    DECLARE_WEIGHT(vw)

    ncnn::Mat _time_mix_k;
    ncnn::Mat _time_mix_r;

    ncnn::Mat mix(ncnn::Mat in0, ncnn::Mat in1, ncnn::Mat param, ncnn::Mat _param, const ncnn::Option& opt) const;

    ncnn::Layer *sigmoid;
    ncnn::Layer *relu;
    ncnn::Layer *matmul;
    ncnn::Layer *add;
    ncnn::Layer *mul;
    ncnn::Layer *one_sub;
    ncnn::Layer *max;
    ncnn::Layer *square;
    ncnn::Layer *exp;
    ncnn::Layer *layernorm;

    int layer_num;
};

class RWKV_Decoder : public ncnn::Layer {
public:
    model_args_t *model_args;

    RWKV_Decoder(model_args_t *args) {
        model_args = args;
        one_blob_only = false;
        support_inplace = true;
        matmul = 0;
        layernorm = ncnn::create_layer(ncnn::LayerType::LayerNorm);
    }

    virtual int load_param(const ncnn::ParamDict& pd) {
        ncnn::Mat ln_weight_shape;
        LOAD_WEIGHT_PARAM(ln_weight, 10);
        int *ptr = ln_weight_shape;
        ncnn::ParamDict ln_pd;
        ln_pd.set(0, ptr[0]);
        layernorm->load_param(ln_pd);

        LOAD_WEIGHT_PARAM(ow, 12);
        return 0;
    }

    virtual int load_model(const ncnn::ModelBin& mb) {
        int *ptr;
        layernorm->load_model(mb);
        LOAD_WEIGHT_DATA(ow, ptr[1], ptr[0]);

        return 0;
    }

    virtual int forward_inplace(std::vector<ncnn::Mat>& bottom_top_blobs, const ncnn::Option& opt) const {
        ncnn::Mat& x = bottom_top_blobs[0];
        layernorm->forward_inplace(x, opt);
        std::vector<ncnn::Mat> bottom_blobs(2);
        bottom_blobs[0] = ow;
        bottom_blobs[1] = x;
        std::vector<ncnn::Mat> top_blobs(1);
        matmul->forward(bottom_blobs, bottom_top_blobs, opt);
        return 0;
    }

    virtual int create_pipeline(const ncnn::Option& opt) {
        matmul = ncnn::create_layer(ncnn::LayerType::MatMul);
        matmul->create_pipeline(opt);

        layernorm->create_pipeline(opt);

        return 0;
    }

    virtual int destroy_pipeline(const ncnn::Option& opt) {
        matmul->destroy_pipeline(opt);
        layernorm->destroy_pipeline(opt);
        return 0;
    }

private:
    DECLARE_WEIGHT(ow)
    ncnn::Layer *matmul;
    ncnn::Layer *layernorm;
};

class RWKV_Encoder : public ncnn::Layer {
public:
    model_args_t *model_args;

    RWKV_Encoder(model_args_t *args) {
        model_args = args;
        one_blob_only = true;
        support_inplace = true;
        layernorm = ncnn::create_layer(ncnn::LayerType::LayerNorm);
    }

    virtual int load_param(const ncnn::ParamDict& pd) {
        ncnn::Mat ln_weight_shape;
        LOAD_WEIGHT_PARAM(ln_weight, 10);
        int *ptr = ln_weight_shape;
        ncnn::ParamDict ln_pd;
        ln_pd.set(0, ptr[0]);
        layernorm->load_param(ln_pd);
        return 0;
    }

    virtual int load_model(const ncnn::ModelBin& mb) {
        layernorm->load_model(mb);
        return 0;
    }

    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const {
        layernorm->forward_inplace(bottom_top_blob, opt);
        return 0;
    }

    virtual int create_pipeline(const ncnn::Option& opt) {
        layernorm->create_pipeline(opt);
        return 0;
    }

    virtual int destroy_pipeline(const ncnn::Option& opt) {
        layernorm->destroy_pipeline(opt);
        return 0;
    }

private:
    ncnn::Layer *layernorm;
};

}