#pragma once
#include <ncnn/net.h>

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
    char model_bin_path[64];
    char model_param_path[64];
    char emb_weights_path[64];
};

class RWKV {
public:
    ncnn::Mat state;

    RWKV(model_args_t *args);

    int load_model_files();

    inline ncnn::Mat forward(int token) {
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", emb_weights[token]);
        ex.input("in1", state);

        ncnn::Mat out;
        ex.extract("out0", out);
        return out;
    }

private:
    ncnn::Net net;
    model_args_t *args;

    std::vector<ncnn::Mat> emb_weights;
};

inline ncnn::Mat mix(ncnn::Mat in0, ncnn::Mat in1, ncnn::Mat param, const ncnn::Option& opt) {
    const int channels = param.c;
    const int size = param.w * param.h * param.d;
    ncnn::Mat out = ncnn::Mat(size);

    #pragma omp parallel for num_threads(opt.num_threads)
    for(int q = 0; q < channels; q++) {
        float *ptr0 = in0.channel(q);
        float *ptr1 = in1.channel(q);
        float *ptr2 = param.channel(q);
        float *ptr3 = out.channel(q);
        for(int i = 0; i < size; i++)
            ptr3[i] = ptr0[i] * ptr2[i] + ptr1[i] * (1 - ptr2[i]);
    }

    return out;
}

inline ncnn::Mat multiply(ncnn::Mat in0, ncnn::Mat in1, const ncnn::Option& opt) {
    // we only do multiply on mats of the same size
    if((in0.w * in0.h * in0.d) != (in1.w * in1.h * in1.d))
        return ncnn::Mat();

    const int channels = in0.c;
    const int size = in0.w * in0.h * in0.d;
    ncnn::Mat out = ncnn::Mat(size);

    #pragma omp parallel for num_threads(opt.num_threads)
    for(int q = 0; q < channels; q++) {
        float *ptr0 = in0.channel(q);
        float *ptr1 = in1.channel(q);
        float *ptr2 = out.channel(q);
        for(int i = 0; i < size; i++)
            ptr2[i] = ptr0[i] * ptr1[i];
    }

    return out;
}

inline ncnn::Mat add(ncnn::Mat in0, ncnn::Mat in1, const ncnn::Option& opt) {
    if((in0.w * in0.h * in0.d) != (in1.w * in1.h * in1.d))
        return ncnn::Mat();

    const int channels = in0.c;
    const int size = in0.w * in0.h * in0.d;
    ncnn::Mat out = ncnn::Mat(size);

    #pragma omp parallel for num_threads(opt.num_threads)
    for(int q = 0; q < channels; q++) {
        float *ptr0 = in0.channel(q);
        float *ptr1 = in1.channel(q);
        float *ptr2 = out.channel(q);
        for(int i = 0; i < size; i++)
            ptr2[i] = ptr0[i] + ptr1[i];
    }

    return out;
}

inline ncnn::Mat sub(ncnn::Mat in0, ncnn::Mat in1, const ncnn::Option& opt) {
    if((in0.w * in0.h * in0.d) != (in1.w * in1.h * in1.d))
        return ncnn::Mat();

    const int channels = in0.c;
    const int size = in0.w * in0.h * in0.d;
    ncnn::Mat out = ncnn::Mat(size);

    #pragma omp parallel for num_threads(opt.num_threads)
    for(int q = 0; q < channels; q++) {
        float *ptr0 = in0.channel(q);
        float *ptr1 = in1.channel(q);
        float *ptr2 = out.channel(q);
        for(int i = 0; i < size; i++)
            ptr2[i] = ptr0[i] - ptr1[i];
    }

    return out;
}

inline ncnn::Mat divide(ncnn::Mat in0, ncnn::Mat in1, const ncnn::Option& opt) {
    if((in0.w * in0.h * in0.d) != (in1.w * in1.h * in1.d)) {
        return ncnn::Mat();
    }

    const int channels = in0.c;
    const int size = in0.w * in0.h * in0.d;
    ncnn::Mat out = ncnn::Mat(size);

    #pragma omp parallel for num_threads(opt.num_threads)
    for(int q = 0; q < channels; q++) {
        float *ptr0 = in0.channel(q);
        float *ptr1 = in1.channel(q);
        float *ptr2 = out.channel(q);
        for(int i = 0; i < size; i++)
            ptr2[i] = ptr0[i] / ptr1[i];
    }

    return out;
}

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

    ncnn::Layer *sigmoid;
    ncnn::Layer *matmul;
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

    ncnn::Layer *sigmoid;
    ncnn::Layer *relu;
    ncnn::Layer *matmul;
    ncnn::Layer *max;
    ncnn::Layer *square;
    ncnn::Layer *exp;
};

}