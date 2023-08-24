#include "rwkv_v4neo.h"
#include <ncnn/layer_type.h>

using namespace rwkv;

#define PARAM(id) params.get(id, ncnn::Mat())

RWKV_Time_Mixing::RWKV_Time_Mixing() {
    one_blob_only = false;
    support_inplace = true;
    sigmoid = 0;
    matmul = 0;
    add = 0;
    sub = 0;
    mul = 0;
    div_op = 0;
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
    LOAD(rw, ptr[0], ptr[1]);
    LOAD(kw, ptr[0], ptr[1]);
    LOAD(vw, ptr[0], ptr[1]);
    LOAD(time_first, ptr[0]);
    LOAD(time_decay, ptr[0]);
    LOAD(ow, ptr[0], ptr[1]);

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
            add->load_param(pd); \
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

    #undef DESTROY
    return 0;
}

int RWKV_Time_Mixing::forward_inplace(std::vector<ncnn::Mat>& bottom_top_blobs, const ncnn::Option& opt) const {
    #define F_PIPELINE(op, in0, in1, out) \
        { \
            std::vector<ncnn::Mat> bottom_blobs(2); \
            bottom_blobs[0] = in0; \
            bottom_blobs[1] = in1; \
            std::vector<ncnn::Mat> top_blobs(1); \
            op->forward(bottom_blobs, top_blobs, opt); \
            out = top_blobs[0]; \
        }

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

    F_PIPELINE(add, time_first, kk, ww);
    F_PIPELINE(max, state_p, ww, p);
    F_PIPELINE(sub, state_p, p, e1);
    exp->forward_inplace(e1, opt);
    F_PIPELINE(sub, ww, p, e2);
    exp->forward_inplace(e2, opt);
    F_PIPELINE(mul, e1, state_a, tmp1);
    F_PIPELINE(mul, e2, vv, tmp2);
    F_PIPELINE(add, tmp1, tmp2, a);
    F_PIPELINE(mul, e1, state_b, tmp1);
    F_PIPELINE(add, tmp1, e2, b);

    F_PIPELINE(add, state_p, time_decay, ww);
    F_PIPELINE(max, ww, kk, p);
    F_PIPELINE(sub, ww, p, e1);
    exp->forward_inplace(e1, opt);
    F_PIPELINE(sub, kk, p, e2);
    exp->forward_inplace(e2, opt);
    F_PIPELINE(mul, e1, state_a, tmp1);
    F_PIPELINE(mul, e2, vv, tmp2);
    F_PIPELINE(add, tmp1, tmp2, state_a);
    F_PIPELINE(mul, e1, state_b, tmp1);
    F_PIPELINE(add, tmp1, e2, state_b);
    state_p.clone_from(p);

    F_PIPELINE(div_op, a, b, tmp1);
    F_PIPELINE(mul, r, tmp1, tmp2);
    F_PIPELINE(matmul, ow, tmp2, x);

    #undef F_PIPELINE
    return 0;
}

ncnn::Mat RWKV_Time_Mixing::mix(ncnn::Mat in1, ncnn::Mat in2, ncnn::Mat param, const ncnn::Option& opt) const {
    // in1 & in2 & param all in the same shape & dim=1
    int w = in1.w;
    ncnn::Mat out = ncnn::Mat(w);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < w; i++)
        out[i] = (const float)in1[i] * (const float)param[i] + (const float)in2[i] * (1 - (const float)param[i]);

    return out;
}
