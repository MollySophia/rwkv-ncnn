#include "rwkv_v4neo.h"
#include <layer_type.h>

using namespace rwkv;

#define PARAM(id) params.get(id, ncnn::Mat())

#define F_PIPELINE(op, in0, in1, out) \
    { \
        std::vector<ncnn::Mat> bottom_blobs(2); \
        bottom_blobs[0] = in0; \
        bottom_blobs[1] = in1; \
        std::vector<ncnn::Mat> top_blobs(1); \
        op->forward(bottom_blobs, top_blobs, opt); \
        out = top_blobs[0]; \
    }

RWKV_Channel_Mixing::RWKV_Channel_Mixing(model_args_t *args) {
    model_args = args;
    one_blob_only = false;
    support_inplace = true;
    sigmoid = 0;
    relu = 0;
    matmul = 0;
    add = 0;
    mul = 0;
    one_sub = 0;
    max = 0;
    square = 0;
    exp = 0;
    layernorm = ncnn::create_layer(ncnn::LayerType::LayerNorm);
}

int RWKV_Channel_Mixing::load_param(const ncnn::ParamDict& pd) {
    ncnn::Mat ln_weight_shape;
    LOAD_WEIGHT_PARAM(ln_weight, 10);
    int *ptr = ln_weight_shape;
    ncnn::ParamDict ln_pd;
    ln_pd.set(0, ptr[0]);
    layernorm->load_param(ln_pd);

    LOAD_WEIGHT_PARAM(time_mix_k, 12);
    LOAD_WEIGHT_PARAM(time_mix_r, 13);
    LOAD_WEIGHT_PARAM(rw, 14);
    LOAD_WEIGHT_PARAM(kw, 15);
    LOAD_WEIGHT_PARAM(vw, 16);
    layer_num = pd.get(1, 0);

    return 0;
}

int RWKV_Channel_Mixing::load_model(const ncnn::ModelBin& mb) {
    int *ptr;
    layernorm->load_model(mb);
    LOAD_WEIGHT_DATA(time_mix_k, ptr[0]);
    LOAD_WEIGHT_DATA(time_mix_r, ptr[0]);
    LOAD_WEIGHT_DATA(rw, ptr[1], ptr[0]);
    LOAD_WEIGHT_DATA(kw, ptr[1], ptr[0]);
    LOAD_WEIGHT_DATA(vw, ptr[1], ptr[0]);

    return 0;
}

int RWKV_Channel_Mixing::create_pipeline(const ncnn::Option& opt) {
    #define CREATE(layer, type) \
        layer = ncnn::create_layer(ncnn::LayerType::type); \
        layer->create_pipeline(opt);

    #define CREATE_BINARY_UNARY(layer, op, layertype) \
        { \
            layer = ncnn::create_layer(ncnn::LayerType::layertype); \
            ncnn::ParamDict pd; \
            pd.set(0, op); \
            layer->load_param(pd); \
            layer->create_pipeline(opt); \
        }
        

    CREATE(sigmoid, Sigmoid);
    CREATE(relu, ReLU);
    CREATE(matmul, MatMul);
    CREATE_BINARY_UNARY(add, 0, BinaryOp);
    CREATE_BINARY_UNARY(mul, 2, BinaryOp);
    CREATE_BINARY_UNARY(max, 4, BinaryOp);
    CREATE_BINARY_UNARY(square, 4, UnaryOp);
    CREATE_BINARY_UNARY(exp, 7, UnaryOp);

    {
        one_sub = ncnn::create_layer(ncnn::LayerType::BinaryOp);
        ncnn::ParamDict pd;
        pd.set(0, 7);
        pd.set(1, 1);
        pd.set(2, 1.0f);
        one_sub->load_param(pd);
        one_sub->create_pipeline(opt);
    }

    layernorm->create_pipeline(opt);

    _time_mix_k = time_mix_k.clone();
    _time_mix_r = time_mix_r.clone();
    one_sub->forward_inplace(_time_mix_k, opt);
    one_sub->forward_inplace(_time_mix_r, opt);

    #undef CREATE
    #undef CREATE_BINARY_UNARY
    return 0;
}

int RWKV_Channel_Mixing::destroy_pipeline(const ncnn::Option& opt) {
    #define DESTROY(layer) \
        layer->destroy_pipeline(opt); \
        delete layer; \
        layer = 0;

    DESTROY(sigmoid);
    DESTROY(relu);
    DESTROY(matmul);
    DESTROY(add);
    DESTROY(mul);
    DESTROY(max);
    DESTROY(square);
    DESTROY(exp);
    DESTROY(one_sub);
    DESTROY(layernorm);

    #undef DESTROY
    return 0;
}

int RWKV_Channel_Mixing::forward_inplace(std::vector<ncnn::Mat>& bottom_top_blobs, const ncnn::Option& opt) const {
    ncnn::Mat& x = bottom_top_blobs[0];

    layernorm->forward_inplace(x, opt);

    ncnn::Mat state = bottom_top_blobs[1].row_range(5 * layer_num, 1);
    state.dims = 1;

    ncnn::Mat xk = mix(x, state, time_mix_k, _time_mix_k, opt);
    ncnn::Mat xr = mix(x, state, time_mix_r, _time_mix_r, opt);

    void *ptr = bottom_top_blobs[1].row(5 * layer_num);
    memcpy(ptr, x, sizeof(float) * x.w);

    F_PIPELINE(matmul, rw, xr, xr);
    sigmoid->forward_inplace(xr, opt);
    F_PIPELINE(matmul, kw, xk, xk);
    relu->forward_inplace(xk, opt);
    square->forward_inplace(xk, opt);
    F_PIPELINE(matmul, vw, xk, xk);
    F_PIPELINE(mul, xr, xk, x);
    return 0;
}

ncnn::Mat RWKV_Channel_Mixing::mix(ncnn::Mat in0, ncnn::Mat in1, ncnn::Mat param, ncnn::Mat _param, const ncnn::Option& opt) const {
    ncnn::Mat tmp1, tmp2, out;
    F_PIPELINE(mul, in0, param, tmp1);
    F_PIPELINE(mul, in1, _param, tmp2);
    F_PIPELINE(add, tmp1, tmp2, out);    

    return out;
}