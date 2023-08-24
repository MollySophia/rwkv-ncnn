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

static void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int z=0; z<m.d; z++)
        {
            for (int y=0; y<m.h; y++)
            {
                for (int x=0; x<m.w; x++)
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

int main(int argc, char **argv) {
    ncnn::Net net;
    net.register_custom_layer("rwkv.rwkv_v4neo.RWKV_Time_Mixing", RWKV_Time_Mixing_layer_creator);
    net.register_custom_layer("rwkv.rwkv_v4neo.RWKV_Channel_Mixing", RWKV_Channel_Mixing_layer_creator);

    cout << "Loading model" << endl;

    net.load_param("../output/model.ncnn.param");
    net.load_model("../output/model.ncnn.bin");


    ncnn::Mat in0 = ncnn::Mat(2560);
    ncnn::Mat in1 = ncnn::Mat(160, 2560);
    
    // pretty_print(in0);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in0);
    ex.input("in1", in1);

    ncnn::Mat out;
    ex.extract("out0", out);

    // pretty_print(out);
    return 0;
}