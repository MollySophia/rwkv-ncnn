from rwkv.rwkv_v4neo import *
import os, types, sys
import torch
import numpy as np
import gc
import zipfile

if len(sys.argv) != 4:
    print("Usage: python convert_model.py [pth file] [output path] [fp16/fp32]")
    exit(1)

args = types.SimpleNamespace()
args.FLOAT_MODE = "fp32" # always fp32 when tracing the torch model
if not sys.argv[3] in ["fp32", "fp16", "bf16"]:
    print("Invalid argument")
    exit(1)

args.vocab_size = 65536

args.MODEL_NAME = sys.argv[1][:-4]
args.n_layer = 24
args.n_embd = 2048
args.ctx_len = 1024

output_path = sys.argv[2]

print(f'loading... {args.MODEL_NAME}')
model = RWKV(args)

current_state = torch.ones(args.n_layer * 5, args.n_embd, device="cpu")
in0 = torch.ones(args.n_embd, device="cpu")

traced_model = torch.jit.trace(model, (in0, current_state))
if not os.path.exists(output_path):
    os.mkdir(output_path)
traced_model.save(os.path.join(output_path, "model.pt"))
print(f'TorchScript IR saved to {output_path}/model.pt')

emb_weight = model.w.emb.weight.numpy().tofile(os.path.join(output_path, "emb_weight.bin"))
print(f"emb_weight saved to {output_path}/emb_weight.bin")

del model
del traced_model
del emb_weight

gc.collect()

if not os.path.exists("./pnnx"):
    print("Get pnnx first!")
    exit()

print("Running pnnx...")

use_fp16 = 1
if sys.argv[3] == "fp32":
    use_fp16 = 0

os.system(f"./pnnx {output_path}/model.pt fp16={use_fp16} inputshape=[{args.n_embd}],[{args.n_layer * 5},{args.n_embd}] moduleop=rwkv.rwkv_v4neo.RWKV_Channel_Mixing,rwkv.rwkv_v4neo.RWKV_Time_Mixing,rwkv.rwkv_v4neo.RWKV_Decoder,rwkv.rwkv_v4neo.RWKV_Encoder")

print("Running model param post processing...")
with open(f"{output_path}/model.ncnn.param", "r") as f:
    lines = f.readlines()

state = 'in1'
state_index = -1
time_mixing_layer_n = 0
channel_mixing_layer_n = 0

for i in lines:
    i_list = i.split()
    if i_list[0] == 'Split':
        if i_list[4] == 'in1':
            lines.remove(i)
    elif i_list[0] == 'rwkv.rwkv_v4neo.RWKV_Time_Mixing' or i_list[0] == 'rwkv.rwkv_v4neo.RWKV_Channel_Mixing':
        i_list[3] = '2'
        i_list[5] = state
        state_index += 1
        state = 'state' + str(state_index)
        i_list.insert(7, state)
        if i_list[0] == 'rwkv.rwkv_v4neo.RWKV_Time_Mixing':
            i_list.append('1=' + str(time_mixing_layer_n))
            time_mixing_layer_n += 1
        else:
            i_list.append('1=' + str(channel_mixing_layer_n))
            channel_mixing_layer_n += 1
        lines[lines.index(i)] = ' '.join(i_list) + '\n'

for i in lines:
    i_list = i.split()
    if lines.index(i) == 1:
        i_list[0] = str(int(i_list[0]) - 1)
        i_list[1] = str(int(i_list[1]) - 1)
        lines[1] = ' '.join(i_list) + '\n'
    elif i_list[0] == 'rwkv.rwkv_v4neo.RWKV_Channel_Mixing':
        if i_list[7] == 'state' + str(state_index):
            i_list[7] = 'out1'
            lines[lines.index(i)] = ' '.join(i_list) + '\n'

with open(f"{output_path}/model.ncnn.param", "w") as f:
    f.writelines(lines)

with open('./output/parameters.txt', 'w') as f:
    f.write(f'{args.vocab_size},{args.n_layer},{args.n_embd},{sys.argv[3]}')
