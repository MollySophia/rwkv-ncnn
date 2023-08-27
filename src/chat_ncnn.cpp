#include <iostream>
#include <cstdio>
#include <ncnn/net.h>
#include "rwkv_v4neo.h"

using namespace std;

rwkv::model_args_t model_args = {
    .float_mode = rwkv::fp32,
    .vocab_size = 50277,
    .layer_num = 32,
    .embd_num = 2560,
    .ctx_len = 1024,
    .model_bin_path = "../output/model.ncnn.bin",
    .model_param_path = "../output/model.ncnn.param",
    .emb_weights_path = "../output/emb_weight.bin",
};

rwkv::RWKV RWKV(&model_args);

int main(int argc, char **argv) {
    RWKV.load_model_files();
    ncnn::Mat output = RWKV.forward(0);
    pretty_print(output);
    return 0;
}