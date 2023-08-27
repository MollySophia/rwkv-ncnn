#include <iostream>
#include <cstdio>
#include <ncnn/net.h>
#include "rwkv_v4neo.h"

::ncnn::Layer* RWKV_Time_Mixing_layer_creator(void* /*userdata*/)
{
    return new rwkv::RWKV_Time_Mixing;
}

::ncnn::Layer* RWKV_Channel_Mixing_layer_creator(void* /*userdata*/)
{
    return new rwkv::RWKV_Channel_Mixing;
}

using namespace std;

rwkv::model_args_t model_args = {
    .float_mode = rwkv::fp32,
    .vocab_size = 50277,
    .layer_num = 32,
    .embd_num = 2560,
    .ctx_len = 1024,
};

int main(int argc, char **argv) {
    ncnn::Net net;
    net.register_custom_layer("rwkv.rwkv_v4neo.RWKV_Time_Mixing", RWKV_Time_Mixing_layer_creator);
    net.register_custom_layer("rwkv.rwkv_v4neo.RWKV_Channel_Mixing", RWKV_Channel_Mixing_layer_creator);

    cout << "Loading model" << endl;

    net.load_param("../output/model.ncnn.param");
    net.load_model("../output/model.ncnn.bin");

    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;

    // net.opt.lightmode = false;

    float in0_array[2560];
    FILE *emb_weight_fp = fopen("../output/emb_weight.bin", "rb");
    int size_read = fread(in0_array, sizeof(in0_array), 1, emb_weight_fp);

    ncnn::Mat in0 = ncnn::Mat(2560, (void*)in0_array).reshape(2560).clone();
    ncnn::Mat in1 = ncnn::Mat(2560, 160);
    in1.fill(0.0f);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in0);
    ex.input("in1", in1);

    ncnn::Mat out;
    ex.extract("out0", out);

    pretty_print(out);
    return 0;
}