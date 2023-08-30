from rwkv.rwkv_v4neo import *
import os, types
import torch
import numpy as np
import gc

args = types.SimpleNamespace()
args.FLOAT_MODE = "fp32" # fp32 // fp16 // bf16
args.vocab_size = 65536
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0

args.MODEL_NAME = '/home/molly/Downloads/RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096'
args.n_layer = 32
args.n_embd = 2560
args.ctx_len = 1024

print(f'loading... {args.MODEL_NAME}.pth')
model = RWKV(args)

current_state = torch.ones(args.n_layer * 5, args.n_embd, device="cpu")
in0 = torch.ones(args.n_embd, device="cpu")

traced_model = torch.jit.trace(model, (in0, current_state))
if not os.path.exists("./output"):
    os.mkdir("./output")
traced_model.save('./output/model.pt')
print("TorchScript IR saved to ./output/model.pt")

emb_weight = model.w.emb.weight.numpy().tofile("./output/emb_weight.bin")
print("emb_weight saved to ./output/emb_weight.bin")

del model
del traced_model
del emb_weight

gc.collect()

if not (os.path.exists("pnnx") or os.path.exists("pnnx.exe")):
    print("Get pnnx first!")
    exit()

print("Running pnnx...")
os.system(f"pnnx ./output/model.pt inputshape=[{args.n_embd}],[{args.n_layer * 5},{args.n_embd}] moduleop=rwkv.rwkv_v4neo.RWKV_Channel_Mixing,rwkv.rwkv_v4neo.RWKV_Time_Mixing,rwkv.rwkv_v4neo.RWKV_Decoder")

print("Running model param post processing...")
with open("./output/model.ncnn.param", "r") as f:
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
            i_list.append('19=' + str(time_mixing_layer_n))
            time_mixing_layer_n += 1
        else:
            i_list.append('15=' + str(channel_mixing_layer_n))
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

with open("./output/model.ncnn.param", "w") as f:
    f.writelines(lines)
