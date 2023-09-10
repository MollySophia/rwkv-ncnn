#include "rwkv_v4neo.h"
#include <ncnn/net.h>
#include <ncnn/layer_type.h>
#include <zip.h>
#include <algorithm>

::ncnn::Layer* RWKV_Time_Mixing_layer_creator(void* userdata)
{
    return new rwkv::RWKV_Time_Mixing((rwkv::model_args_t*)userdata);
}

::ncnn::Layer* RWKV_Channel_Mixing_layer_creator(void* userdata)
{
    return new rwkv::RWKV_Channel_Mixing((rwkv::model_args_t*)userdata);
}

::ncnn::Layer* RWKV_Decoder_layer_creator(void* userdata)
{
    return new rwkv::RWKV_Decoder((rwkv::model_args_t*)userdata);
}

::ncnn::Layer* RWKV_Encoder_layer_creator(void* userdata)
{
    return new rwkv::RWKV_Encoder((rwkv::model_args_t*)userdata);
}

using namespace rwkv;

RWKV::RWKV(model_args_t *args) {
    if(!args)
        return;
    this->args = args;
    state = ncnn::Mat(args->embd_num, 5 * args->layer_num);
    state.fill(0.0f);

    // net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_arithmetic = true;
    // net.opt.use_fp16_storage = false;
    net.opt.use_bf16_storage = true;

    for(int i = 0; i < args->layer_num; i++) {
        float *ptr = state.row(5 * i + 4);
        for(int j = 0; j < args->embd_num; j++)
            ptr[j] = -1e30;
    }

    softmax = ncnn::create_layer(ncnn::LayerType::Softmax);
    softmax->create_pipeline(net.opt);
    cumsum = ncnn::create_layer(ncnn::LayerType::CumulativeSum);
    cumsum->create_pipeline(net.opt);

    net.register_custom_layer("rwkv.rwkv_v4neo.RWKV_Time_Mixing", RWKV_Time_Mixing_layer_creator, 0, (void*)args);
    net.register_custom_layer("rwkv.rwkv_v4neo.RWKV_Channel_Mixing", RWKV_Channel_Mixing_layer_creator, 0, (void*)args);
    net.register_custom_layer("rwkv.rwkv_v4neo.RWKV_Decoder", RWKV_Decoder_layer_creator, 0, (void*)args);
    net.register_custom_layer("rwkv.rwkv_v4neo.RWKV_Encoder", RWKV_Encoder_layer_creator, 0, (void*)args);
}

int RWKV::load_model_pack(const char *filename) {
    int err = 0;
    zip_t *zip = zip_open(filename, ZIP_RDONLY, &err);
    int files_total = zip_get_num_entries(zip, 0);
    
    char *param_buffer;
    unsigned char *model_buffer;

    struct zip_stat zstat;
    long long sum = 0, len = 0;

    for(int i = 0; i < files_total; i++) {
        if(zip_stat_index(zip, i, 0, &zstat) == 0) {
            std::cout << "Loading: " << zstat.name << std::endl;
            sum = 0;
            if(!strcmp(zstat.name, "emb_weight.bin")) {
                zip_file *zf = zip_fopen_index(zip, i, 0);
                float *array = new float[args->embd_num];
                if(!array)
                    return -1;

                int i = 0;
                
                for(i = 0; i < args->vocab_size; i++) {
                    int size_read = zip_fread(zf, array, sizeof(float) * args->embd_num);
                    if(size_read < 0)
                        break;
                    
                    emb_weights.push_back(ncnn::Mat(args->embd_num, (void*)array).reshape(args->embd_num).clone());
                }
                zip_fclose(zf);
                delete[] array;
            }
            else if(!strcmp(zstat.name, "model.ncnn.param")) {
                zip_file *zf = zip_fopen_index(zip, i, 0);
                param_buffer = new char[zstat.size];
                while(sum != zstat.size) {
                    len = zip_fread(zf, param_buffer, zstat.size);
                    if(len < 0) {
                        std::cout << "failed to read" <<  zstat.name << std::endl;
                        continue;
                    }
                    sum += len;
                }
                zip_fclose(zf);
            }
            else if(!strcmp(zstat.name, "model.ncnn.bin")) {
                zip_file *zf = zip_fopen_index(zip, i, 0);
                model_buffer = new unsigned char[zstat.size];
                while(sum != zstat.size) {
                    len = zip_fread(zf, model_buffer, zstat.size);
                    if(len < 0) {
                        std::cout << "failed to read" <<  zstat.name << std::endl;
                        continue;
                    }
                    sum += len;
                }
                zip_fclose(zf);
            }
            else if(!strcmp(zstat.name, "parameters.txt")) {
                zip_file *zf = zip_fopen_index(zip, i, 0);
                char *buffer = new char[zstat.size];
                len = zip_fread(zf, buffer, zstat.size);
                if(len < 0) {
                    std::cout << "failed to read" <<  zstat.name << std::endl;
                    continue;
                }
                sscanf(buffer, "%d,%d,%d", &args->vocab_size, &args->layer_num, &args->embd_num);
                delete[] buffer;
                zip_fclose(zf);
            }
            else continue;
        }
    }

    if(!model_buffer || !param_buffer) {
        std::cout << "Failed in loading files" << std::endl;
        return 1;
    }

    net.load_param_mem(param_buffer);
    net.load_model(model_buffer);
    delete[] param_buffer;

    return err;
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

int RWKV::sample_logits(ncnn::Mat logits, float temp, float top_p, float top_k) {
    softmax->forward_inplace(logits, net.opt);
    ncnn::Mat sorted_probs = ncnn::Mat(logits).clone();
    float *ptr = sorted_probs;
    int size = sorted_probs.w;
    std::sort(ptr, ptr + size, std::greater<float>());
    ncnn::Mat cumulative_probs = ncnn::Mat(sorted_probs).clone();
    cumsum->forward_inplace(cumulative_probs, net.opt);
    float cutoff = 0;
    int index;
    ptr = cumulative_probs;
    for(index = 0; index < size; index++)
        if(ptr[index] > top_p) {
            break;
        }

    ptr = sorted_probs;
    cutoff = ptr[index];

    ptr = logits;
    for(int i = 0; i < size; i++)
        if(ptr[i] < cutoff)
            ptr[i] = 0;
    
    float sum = 0;
    for(int i = 0; i < size; i++)
        sum += ptr[i];
    for(int i = 0; i < size; i++)
        ptr[i] /= sum;

    int out = 0;
    float max = 0;
    for(int i = 0; i < size; i++) {
        if(ptr[i] > max) {
            max = ptr[i];
            out = i;
        }
    }
    return out;
}