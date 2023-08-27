#include "rwkv_v4neo.h"
#include <ncnn/layer_type.h>

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

RWKV_Time_Mixing::RWKV_Time_Mixing() {
    one_blob_only = false;
    support_inplace = true;
    sigmoid = 0;
    matmul = 0;
    max = 0;
    exp = 0;
}

int RWKV_Time_Mixing::load_param(const ncnn::ParamDict& pd) {
    #define LOAD(to, id) \
        to##_shape = pd.get(id, ncnn::Mat());
    
    LOAD(time_mix_k, 10);
    LOAD(time_mix_v, 11);
    LOAD(time_mix_r, 12);
    LOAD(rw, 13);
    LOAD(kw, 14);
    LOAD(vw, 15);
    LOAD(time_first, 16);
    LOAD(time_decay, 17);
    LOAD(ow, 18);

    #undef LOAD
    return 0;
}

int RWKV_Time_Mixing::load_model(const ncnn::ModelBin& mb) {
    #define LOAD(to, args...) \
        ptr = to##_shape; \
        to = mb.load(args, 1); \
        if(to.empty()) \
            return -100;

    int *ptr;
    
    LOAD(time_mix_k, ptr[0]);
    LOAD(time_mix_v, ptr[0]);
    LOAD(time_mix_r, ptr[0]);
    LOAD(rw, ptr[1], ptr[0]);
    LOAD(kw, ptr[1], ptr[0]);
    LOAD(vw, ptr[1], ptr[0]);
    LOAD(time_first, ptr[0]);
    LOAD(time_decay, ptr[0]);
    LOAD(ow, ptr[1], ptr[0]);

    #undef LOAD
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
    CREATE_BINARY_UNARY(max, 4, BinaryOp);
    CREATE_BINARY_UNARY(exp, 7, UnaryOp);

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
    DESTROY(max);
    DESTROY(exp);

    #undef DESTROY
    return 0;
}

int RWKV_Time_Mixing::forward_inplace(std::vector<ncnn::Mat>& bottom_top_blobs, const ncnn::Option& opt) const {

    ncnn::Mat& x = bottom_top_blobs[0];
    ncnn::Mat& state = bottom_top_blobs[1];
    ncnn::Mat& state_a = bottom_top_blobs[2];
    ncnn::Mat& state_b = bottom_top_blobs[3];
    ncnn::Mat& state_p = bottom_top_blobs[4];

    ncnn::Mat xk = mix(x, state, time_mix_k, opt);
    ncnn::Mat xv = mix(x, state, time_mix_v, opt);
    ncnn::Mat xr = mix(x, state, time_mix_r, opt);

    ncnn::Mat r;
    F_PIPELINE(matmul, rw, xr, r);
    sigmoid->forward_inplace(r, opt);
    ncnn::Mat kk;
    F_PIPELINE(matmul, kw, xk, kk);

    ncnn::Mat vv;
    F_PIPELINE(matmul, vw, xv, vv);

    ncnn::Mat ww, a, b, p, e1, e2, tmp1, tmp2;

    ww = add(time_first, kk, opt);
    F_PIPELINE(max, state_p, ww, p);
    e1 = sub(state_p, p, opt);
    exp->forward_inplace(e1, opt);
    e2 = sub(ww, p, opt);
    exp->forward_inplace(e2, opt);
    tmp1 = multiply(e1, state_a, opt);
    tmp2 = multiply(e2, vv, opt);
    a = add(tmp1, tmp2, opt);
    tmp1 = multiply(e1, state_b, opt);
    b = add(tmp1, e2, opt);

    ww = add(state_p, time_decay, opt);
    F_PIPELINE(max, ww, kk, p);
    e1 = sub(ww, p, opt);
    exp->forward_inplace(e1, opt);
    e2 = sub(kk, p, opt);
    exp->forward_inplace(e2, opt);
    tmp1 = multiply(e1, state_a, opt);
    tmp2 = multiply(e2, vv, opt);
    state_a = add(tmp1, tmp2, opt);
    tmp1 = multiply(e1, state_b, opt);
    state_b = add(tmp1, e2, opt);
    state_p.clone_from(p);

    tmp1 = divide(a, b, opt);
    tmp2 = multiply(r, tmp1, opt);

    ncnn::Mat out;
    F_PIPELINE(matmul, ow, tmp2, out);
    
    bottom_top_blobs[0] = state_p;
    bottom_top_blobs[1] = state_b;
    bottom_top_blobs[2] = state_a;
    bottom_top_blobs[3] = out;

    return 0;
}