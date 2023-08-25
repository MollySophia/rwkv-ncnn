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

RWKV_Channel_Mixing::RWKV_Channel_Mixing() {
    one_blob_only = false;
    support_inplace = true;
    sigmoid = 0;
    relu = 0;
    matmul = 0;
    add = 0;
    sub = 0;
    mul = 0;
    div_op = 0;
    max = 0;
    square = 0;
    exp = 0;
}

int RWKV_Channel_Mixing::load_param(const ncnn::ParamDict& pd) {
    #define LOAD(to, id) \
        to##_shape = pd.get(id, ncnn::Mat()) 
    
    LOAD(time_mix_k, 10);
    LOAD(time_mix_r, 11);
    LOAD(rw, 12);
    LOAD(kw, 13);
    LOAD(vw, 14);

    #undef LOAD
    return 0;
}

int RWKV_Channel_Mixing::load_model(const ncnn::ModelBin& mb) {
    #define LOAD(to, args...) \
        ptr = to##_shape; \
        to = mb.load(args, 1); \
        if(to.empty()) \
            return -100;

    int *ptr;
    
    LOAD(time_mix_k, ptr[0]);
    LOAD(time_mix_r, ptr[0]);
    LOAD(rw, ptr[0], ptr[1]);
    LOAD(kw, ptr[0], ptr[1]);
    LOAD(vw, ptr[0], ptr[1]);

    #undef LOAD
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
            add->load_param(pd); \
            layer->create_pipeline(opt); \
        }
        

    CREATE(sigmoid, Sigmoid);
    CREATE(relu, ReLU);
    CREATE(matmul, MatMul);
    CREATE_BINARY_UNARY(add, 0, BinaryOp);
    CREATE_BINARY_UNARY(sub, 1, BinaryOp);
    CREATE_BINARY_UNARY(mul, 2, BinaryOp);
    CREATE_BINARY_UNARY(div_op, 3, BinaryOp);
    CREATE_BINARY_UNARY(max, 4, BinaryOp);
    CREATE_BINARY_UNARY(square, 4, UnaryOp);
    CREATE_BINARY_UNARY(exp, 7, UnaryOp);

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
    DESTROY(sub);
    DESTROY(mul);
    DESTROY(div_op);
    DESTROY(max);
    DESTROY(square);
    DESTROY(exp);

    #undef DESTROY
    return 0;
}

int RWKV_Channel_Mixing::forward_inplace(std::vector<ncnn::Mat>& bottom_top_blobs, const ncnn::Option& opt) const {
    ncnn::Mat& x = bottom_top_blobs[0];
    ncnn::Mat& state = bottom_top_blobs[1];

    ncnn::Mat xk = mix(x, state, time_mix_k, opt);
    ncnn::Mat xr = mix(x, state, time_mix_r, opt);

    ncnn::Mat r;
    F_PIPELINE(matmul, rw, xr, r);
    sigmoid->forward_inplace(r, opt);
    
    ncnn::Mat k;
    F_PIPELINE(matmul, kw, xk, k);
    relu->forward_inplace(k, opt);
    square->forward_inplace(k, opt);

    ncnn::Mat kv;
    F_PIPELINE(matmul, vw, k, kv);
    F_PIPELINE(mul, r, kv, x);

    return 0;
}

ncnn::Mat RWKV_Channel_Mixing::mix(ncnn::Mat in0, ncnn::Mat in1, ncnn::Mat param, const ncnn::Option& opt) const {
    ncnn::Mat out, tmp0, tmp1, one_sub_param = ncnn::Mat(param).clone();
    F_PIPELINE(mul, in0, param, tmp0);
    
    const int channels = one_sub_param.c;
    const int size = one_sub_param.w * one_sub_param.h * one_sub_param.d;
    
    #pragma omp parallel for num_threads(opt.num_threads)
    for(int q = 0; q < channels; q++) {
        float *ptr = one_sub_param.channel(q);
        for(int i = 0; i < size; i++) {
            ptr[i] = 1 - ptr[i];
        }
    }

    F_PIPELINE(mul, in1, one_sub_param, tmp1);
    F_PIPELINE(add, tmp0, tmp1, out);

    return out;
}
