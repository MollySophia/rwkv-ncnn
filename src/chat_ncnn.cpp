#include <iostream>
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

    net.load_param("../output/model.ncnn.param");
    net.load_model("../output/model.ncnn.bin");

    return 0;
}