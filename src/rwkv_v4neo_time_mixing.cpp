#include "rwkv_v4neo.h"
#include <layer_type.h>
#include <chrono>

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

RWKV_Time_Mixing::RWKV_Time_Mixing(model_args_t *args) {
    model_args = args;
    one_blob_only = false;
    support_inplace = true;
    sigmoid = 0;
    matmul = 0;
    add = 0;
    sub = 0;
    mul = 0;
    one_sub = 0;
    div_op = 0;
    max = 0;
    exp = 0;
    layernorm = ncnn::create_layer(ncnn::LayerType::LayerNorm);
}

int RWKV_Time_Mixing::load_param(const ncnn::ParamDict& pd) {
    ncnn::Mat ln_weight_shape;
    LOAD_WEIGHT_PARAM(ln_weight, 10);
    int *ptr = ln_weight_shape;
    ncnn::ParamDict ln_pd;
    ln_pd.set(0, ptr[0]);
    layernorm->load_param(ln_pd);

    LOAD_WEIGHT_PARAM(time_mix_k, 12);
    LOAD_WEIGHT_PARAM(time_mix_v, 13);
    LOAD_WEIGHT_PARAM(time_mix_r, 14);
    LOAD_WEIGHT_PARAM(rw, 15);
    LOAD_WEIGHT_PARAM(kw, 16);
    LOAD_WEIGHT_PARAM(vw, 17);
    LOAD_WEIGHT_PARAM(time_first, 18);
    LOAD_WEIGHT_PARAM(time_decay, 19);
    LOAD_WEIGHT_PARAM(ow, 20);
    layer_num = pd.get(1, 0);

    #undef LOAD
    return 0;
}

int RWKV_Time_Mixing::load_model(const ncnn::ModelBin& mb) {
    int *ptr;
    layernorm->load_model(mb);
    LOAD_WEIGHT_DATA(time_mix_k, ptr[0]);
    LOAD_WEIGHT_DATA(time_mix_v, ptr[0]);
    LOAD_WEIGHT_DATA(time_mix_r, ptr[0]);
    LOAD_WEIGHT_DATA(rw, ptr[1], ptr[0]);
    LOAD_WEIGHT_DATA(kw, ptr[1], ptr[0]);
    LOAD_WEIGHT_DATA(vw, ptr[1], ptr[0]);
    LOAD_WEIGHT_DATA(time_first, ptr[0]);
    LOAD_WEIGHT_DATA(time_decay, ptr[0]);
    LOAD_WEIGHT_DATA(ow, ptr[1], ptr[0]);

    return 0;
}

int RWKV_Time_Mixing::create_pipeline(const ncnn::Option& opt) {
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
    CREATE(matmul, MatMul);
    CREATE_BINARY_UNARY(add, 0, BinaryOp);
    CREATE_BINARY_UNARY(sub, 1, BinaryOp);
    CREATE_BINARY_UNARY(mul, 2, BinaryOp);
    CREATE_BINARY_UNARY(div_op, 3, BinaryOp);
    CREATE_BINARY_UNARY(max, 4, BinaryOp);
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
    _time_mix_v = time_mix_v.clone();
    _time_mix_r = time_mix_r.clone();
    one_sub->forward_inplace(_time_mix_k, opt);
    one_sub->forward_inplace(_time_mix_v, opt);
    one_sub->forward_inplace(_time_mix_r, opt);

    #undef CREATE
    #undef CREATE_BINARY_UNARY
    return 0;
}

int RWKV_Time_Mixing::destroy_pipeline(const ncnn::Option& opt) {
    #define DESTROY(layer) \
        layer->destroy_pipeline(opt); \
        delete layer; \
        layer = 0;

    DESTROY(sigmoid);
    DESTROY(matmul);
    DESTROY(add);
    DESTROY(sub);
    DESTROY(mul);
    DESTROY(div_op);
    DESTROY(max);
    DESTROY(exp);
    DESTROY(one_sub);
    DESTROY(layernorm);

    #undef DESTROY
    return 0;
}

int RWKV_Time_Mixing::forward_inplace(std::vector<ncnn::Mat>& bottom_top_blobs, const ncnn::Option& opt) const {
    ncnn::Mat& x = bottom_top_blobs[0];

    layernorm->forward_inplace(x, opt);

    ncnn::Mat state = ncnn::Mat(bottom_top_blobs[1].row_range(5 * layer_num + 1, 1));
    ncnn::Mat state_a = ncnn::Mat(bottom_top_blobs[1].row_range(5 * layer_num + 2, 1));
    ncnn::Mat state_b = ncnn::Mat(bottom_top_blobs[1].row_range(5 * layer_num + 3, 1));
    ncnn::Mat state_p = ncnn::Mat(bottom_top_blobs[1].row_range(5 * layer_num + 4, 1));

    state.dims = 1;
    state_a.dims = 1;
    state_b.dims = 1;
    state_p.dims = 1;
    
    ncnn::Mat xk = mix(x, state, time_mix_k, _time_mix_k, opt);
    ncnn::Mat xv = mix(x, state, time_mix_v, _time_mix_v, opt);
    ncnn::Mat xr = mix(x, state, time_mix_r, _time_mix_r, opt);

    void *ptr = bottom_top_blobs[1].row(5 * layer_num + 1);
    memcpy(ptr, x, sizeof(float) * x.w);

    F_PIPELINE(matmul, rw, xr, xr);
    sigmoid->forward_inplace(xr, opt);
    F_PIPELINE(matmul, kw, xk, xk);

    F_PIPELINE(matmul, vw, xv, xv);

    ncnn::Mat ww, e1, e2, tmp1, tmp2;

    F_PIPELINE(add, time_first, xk, ww);
    F_PIPELINE(max, state_p, ww, tmp1);
    F_PIPELINE(sub, state_p, tmp1, e1);
    exp->forward_inplace(e1, opt);
    F_PIPELINE(sub, ww, tmp1, e2);
    exp->forward_inplace(e2, opt);
    F_PIPELINE(mul, e1, state_a, tmp1);
    F_PIPELINE(mul, e2, xv, tmp2);
    F_PIPELINE(add, tmp1, tmp2, tmp2);
    F_PIPELINE(mul, e1, state_b, tmp1);
    F_PIPELINE(add, tmp1, e2, tmp1);

    F_PIPELINE(div_op, tmp2, tmp1, tmp1);
    F_PIPELINE(mul, xr, tmp1, tmp2);

    F_PIPELINE(matmul, ow, tmp2, x);

    F_PIPELINE(add, state_p, time_decay, ww);
    F_PIPELINE(max, ww, xk, state_p);
    F_PIPELINE(sub, ww, state_p, e1);
    exp->forward_inplace(e1, opt);
    F_PIPELINE(sub, xk, state_p, e2);
    exp->forward_inplace(e2, opt);
    F_PIPELINE(mul, e1, state_a, tmp1);
    F_PIPELINE(mul, e2, xv, tmp2);
    F_PIPELINE(add, tmp1, tmp2, state_a);
    F_PIPELINE(mul, e1, state_b, tmp1);
    F_PIPELINE(add, tmp1, e2, state_b);

    ptr = bottom_top_blobs[1].row(5 * layer_num + 2);
    memcpy(ptr, state_a, sizeof(float) * state_a.w);
    ptr = bottom_top_blobs[1].row(5 * layer_num + 3);
    memcpy(ptr, state_b, sizeof(float) * state_a.w);
    ptr = bottom_top_blobs[1].row(5 * layer_num + 4);
    memcpy(ptr, state_p, sizeof(float) * state_a.w);

    return 0;
}

ncnn::Mat RWKV_Time_Mixing::mix(ncnn::Mat in0, ncnn::Mat in1, ncnn::Mat param, ncnn::Mat _param, const ncnn::Option& opt) const {
    ncnn::Mat tmp1, tmp2, out;
    F_PIPELINE(mul, in0, param, tmp1);
    F_PIPELINE(mul, in1, _param, tmp2);
    F_PIPELINE(add, tmp1, tmp2, out);    

    return out;
}