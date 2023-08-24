#pragma once
#include <ncnn/net.h>

namespace rwkv {

enum float_mode_t {
    fp32,
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
};

class RWKV {
public:

private:
    ncnn::Net ncnn;

};

#define DECLARE_WEIGHT(name) \
    ncnn::Mat name; \
    ncnn::Mat name##_shape;

class RWKV_Time_Mixing : public ncnn::Layer {
public:
    RWKV_Time_Mixing();

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

    ncnn::Mat mix(ncnn::Mat in1, ncnn::Mat in2, ncnn::Mat param, const ncnn::Option& opt) const;

    ncnn::Layer *sigmoid;
    ncnn::Layer *matmul;
    ncnn::Layer *add;
    ncnn::Layer *sub;
    ncnn::Layer *mul;
    ncnn::Layer *div_op;
    ncnn::Layer *max;
    ncnn::Layer *exp;
};

class RWKV_Channel_Mixing : public ncnn::Layer {
public:
    RWKV_Channel_Mixing();

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

    ncnn::Mat mix(ncnn::Mat in1, ncnn::Mat in2, ncnn::Mat param, const ncnn::Option& opt) const;

    ncnn::Layer *sigmoid;
    ncnn::Layer *relu;
    ncnn::Layer *matmul;
    ncnn::Layer *add;
    ncnn::Layer *sub;
    ncnn::Layer *mul;
    ncnn::Layer *div_op;
    ncnn::Layer *max;
    ncnn::Layer *square;
    ncnn::Layer *exp;
};

}