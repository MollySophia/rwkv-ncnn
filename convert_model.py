from rwkv.rwkv_v4neo import *
import os, types
import torch
import numpy as np

args = types.SimpleNamespace()
args.FLOAT_MODE = "fp32" # fp32 // fp16 // bf16
args.vocab_size = 50277
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0

args.MODEL_NAME = '/home/molly/Downloads/RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096'
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

emb_weight = model.emb_weight.numpy().tofile("./output/emb_weight.bin")
print("emb_weight saved to ./output/emb_weight.bin")

if not os.path.exists("./pnnx"):
    print("Get pnnx first!")
    exit()

print("Running pnnx...")
os.system(f"./pnnx ./output/model.pt inputshape=[{args.n_embd}],[{args.n_layer * 5},{args.n_embd}]")