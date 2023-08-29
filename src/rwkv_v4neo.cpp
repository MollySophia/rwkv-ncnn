#include "rwkv_v4neo.h"
#include <ncnn/net.h>

::ncnn::Layer* RWKV_Time_Mixing_layer_creator(void* /*userdata*/)
{
    return new rwkv::RWKV_Time_Mixing;
}

::ncnn::Layer* RWKV_Channel_Mixing_layer_creator(void* /*userdata*/)
{
    return new rwkv::RWKV_Channel_Mixing;
}

::ncnn::Layer* RWKV_Decoder_layer_creator(void* /*userdata*/)
{
    return new rwkv::RWKV_Decoder;
}

using namespace rwkv;

RWKV::RWKV(model_args_t *args) {
    if(!args)
        return;
    this->args = args;
    state = ncnn::Mat(args->embd_num, 5 * args->layer_num);
    state.fill(0.0f);

    net.register_custom_layer("rwkv.rwkv_v4neo.RWKV_Time_Mixing", RWKV_Time_Mixing_layer_creator);
    net.register_custom_layer("rwkv.rwkv_v4neo.RWKV_Channel_Mixing", RWKV_Channel_Mixing_layer_creator);
    net.register_custom_layer("rwkv.rwkv_v4neo.RWKV_Decoder", RWKV_Decoder_layer_creator);
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_vulkan_compute = false;
}

int RWKV::load_model_files() {
    printf("Loading model files...\n");

    net.load_param(args->model_param_path);
    net.load_model(args->model_bin_path);

    FILE *emb_weight_fp = fopen(args->emb_weights_path, "rb");
    if(!emb_weight_fp) {
        printf("fopen failed: %s\n", args->emb_weights_path);
        return -1;
    }

    float *array = new float[args->embd_num];
    if(!array)
        return -1;

    int i = 0;
    
    for(i = 0; i < args->vocab_size; i++) {
        int size_read = fread(array, sizeof(float) * args->embd_num, 1, emb_weight_fp);
        if(!size_read)
            break;
        
        emb_weights.push_back(ncnn::Mat(args->embd_num, (void*)array).reshape(args->embd_num).clone());
    }

    delete[] array;
    fclose(emb_weight_fp);

    if(i != args->vocab_size) {
        printf("Failed to load emb_weight.bin!\n");
        return -1;
    }

    return 0;
}