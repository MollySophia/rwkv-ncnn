#include <iostream>
#include <cstdio>
#include <string>
#include <ncnn/net.h>
#include "rwkv_v4neo.h"
#include "rwkv_tokenizer.h"

using namespace std;

string init_prompt = "The following is a coherent verbose detailed conversation between a Chinese girl named Alice and her friend Bob. "
"Alice is very intelligent, creative and friendly. "
"Alice likes to tell Bob a lot about herself and her opinions. "
"Alice usually gives Bob kind, helpful and informative advices."
"\n"
"Bob: lhc\n\n"

"Alice: LHC是指大型强子对撞机（Large Hadron Collider），是世界最大最强的粒子加速器，由欧洲核子中心（CERN）在瑞士日内瓦地下建造。LHC的原理是加速质子（氢离子）并让它们相撞，让科学家研究基本粒子和它们之间的相互作用，并在2012年证实了希格斯玻色子的存在。\n\n"

"Bob: 企鹅会飞吗\n\n"

"Alice: 企鹅是不会飞的。企鹅的翅膀短而扁平，更像是游泳时的一对桨。企鹅的身体结构和羽毛密度也更适合在水中游泳，而不是飞行。\n\n"
;

rwkv::model_args_t model_args = {
    .float_mode = rwkv::fp32,
    .vocab_size = 65536,
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
    rwkv::TRIE_Tokenizer tokenizer("../rwkv_vocab_v20230424.bin");
    cout << "Running prompt" << endl;
    ncnn::Mat out = RWKV.forward(tokenizer.Encode(init_prompt));
    // ncnn::Mat out = RWKV.forward(0);
    pretty_print(out);
    return 0;
}