#include <iostream>
#include <cstdio>
#include <string>
#include <map>
#include <net.h>
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
    .layer_num = 24,
    .embd_num = 2048,
    .ctx_len = 1024,
    .model_bin_path = "../output/model.ncnn.bin",
    .model_param_path = "../output/model.ncnn.param",
    .emb_weights_path = "../output/emb_weight.bin",
    .parameters_path = "../output/parameters.txt",
};

rwkv::runtime_args_t runtime_args = {
    .temp = 1.2,
    .top_p = 0.5,
    .alpha_presence = 0.4,
    .alpha_frequency = 0.4,
    .penalty_decay = 0.996,
    .end_of_text = 0,
    .end_of_line = 11,

    .chat_len_short = 40,
    .chat_len_long = 150,
    .free_gen_len = 256,
};

int main(int argc, char **argv) {
    if(argc != 6) {
        cout << "Usage: chat_rwkv_ncnn [model.bin] [model.param] [emb_weight.bin] [vocab.bin] [parameters.txt]" << endl;
        exit(1);
    }
    cout.setf(ios::unitbuf);
    strcpy(model_args.model_bin_path, argv[1]);
    strcpy(model_args.model_param_path, argv[2]);
    strcpy(model_args.emb_weights_path, argv[3]);
    strcpy(model_args.parameters_path, argv[5]);

    FILE *parameters = fopen(model_args.parameters_path, "r");
    if(!parameters) {
        printf("fopen failed: %s\n", model_args.parameters_path);
        return -1;
    }
    char tmp[5] = {0, 0, 0, 0, 0};
    fscanf(parameters, "%d,%d,%d,%s", &model_args.vocab_size, &model_args.layer_num, &model_args.embd_num, tmp);
    fclose(parameters);

    rwkv::RWKV RWKV(&model_args);

    RWKV.load_model_files();
    rwkv::TRIE_Tokenizer tokenizer(argv[4]);
    map<int, float> occurences;
    vector<int> model_tokens;
    // cout << "Running prompt" << endl;
    // ncnn::Mat out = RWKV.forward(tokenizer.Encode(init_prompt));

    while (true) {
        cout << "Bob: ";
        string input;
        getline(cin, input);
        cout << "Alice:";
        ncnn::Mat out = RWKV.forward(tokenizer.Encode(
            "Bob: " + input + "\n\nAlice:"
        ));
        model_tokens.clear();
        for(int i = 0; i < runtime_args.free_gen_len + 100; i++) {
            for(auto &i : occurences) {
                out[i.first] -= (runtime_args.alpha_frequency * i.second + runtime_args.alpha_presence);
            }
            int output = RWKV.sample_logits(out, runtime_args.temp, runtime_args.top_p);
            model_tokens.push_back(output);
            
            for(auto &i : occurences) {
                i.second *= runtime_args.penalty_decay;
            }
            
            if(occurences.find(output) != occurences.end())
                occurences[output] += 1;
            else
                occurences.insert(pair<int, float>(output, 1));

            out = RWKV.forward(output);
            out[runtime_args.end_of_text] = -999999999.0;
            auto output_str = tokenizer.Decode(output);
            cout << output_str;
            output_str = tokenizer.Decode(model_tokens);
            if(output_str.find("\n\n") != output_str.npos)
                break;
        }

    #if DEBUG_TIME
        float time_total = 0;
        int time_count = 0;
        for(auto i : RWKV.time_data) {
            time_count++;
            time_total += i;
        }

        time_total /= time_count;
        cout << "Avg token generation time: " << time_total << " ms" << endl;
        cout << "Avg tokes per second: " << 1000.0 / time_total << endl << endl;
        RWKV.time_data.clear();
    #endif
    }
    return 0;
}