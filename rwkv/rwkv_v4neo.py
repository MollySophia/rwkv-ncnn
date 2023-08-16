########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types
import torch
import math, os, gc
from torch.nn import functional as F
import torch.nn as nn
from typing import List, Dict

# try torch jit --> faster for fp32, slower for fp16 (why?)
# if os.environ["RWKV_JIT_ON"] == "1":
#     MyModule = torch.jit.ScriptModule
#     MyFunction = torch.jit.script_method

# RWKV_HEAD_QK_DIM = 0
# print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM} RWKV_JIT_ON {os.environ["RWKV_JIT_ON"]}\n')

DEBUG_TIME = False   # True False - show trained time-coeffs

RWKV_RESCALE_LAYER = 6 # set x=x/2 every X layer

############################################################################################################

class RWKV_RNN(torch.nn.Module):
    def __init__(self, args, w):
        super().__init__()

        self.args = args
        self.FLOAT_MODE = args.FLOAT_MODE
        self.w = w

    def LN(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    # state[] 0=ffn_xx 1=att_xx 2=att_aa 3=att_bb 4=att_pp

    def FF(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        if self.FLOAT_MODE == "bf16":
            xk = x * time_mix_k + state[5*i+0].type(torch.bfloat16) * (1 - time_mix_k)
            xr = x * time_mix_r + state[5*i+0].type(torch.bfloat16) * (1 - time_mix_r)
            state[5*i+0] = x.float()
        elif self.FLOAT_MODE == "fp16":
            xk = x * time_mix_k + state[5*i+0].half() * (1 - time_mix_k)
            xr = x * time_mix_r + state[5*i+0].half() * (1 - time_mix_r)
            state[5*i+0] = x.float()            
        else:
            xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
            xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
            state[5*i+0] = x

        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))
        kv = vw @ k

        return r * kv

    def SA(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        if self.FLOAT_MODE == "bf16":
            xk = x * time_mix_k + state[5*i+1].type(torch.bfloat16) * (1 - time_mix_k)
            xv = x * time_mix_v + state[5*i+1].type(torch.bfloat16) * (1 - time_mix_v)
            xr = x * time_mix_r + state[5*i+1].type(torch.bfloat16) * (1 - time_mix_r)
            state[5*i+1] = x.float()
        elif self.FLOAT_MODE == "fp16":
            xk = x * time_mix_k + state[5*i+1].half() * (1 - time_mix_k)
            xv = x * time_mix_v + state[5*i+1].half() * (1 - time_mix_v)
            xr = x * time_mix_r + state[5*i+1].half() * (1 - time_mix_r)
            state[5*i+1] = x.float()            
        else:
            xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
            xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
            xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
            state[5*i+1] = x

        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv

        if '16' in self.FLOAT_MODE:
            kk = k.float()
            vv = v.float()
        else:
            kk = k
            vv = v
        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + kk
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * vv
        b = e1 * bb + e2
        ww = pp + time_decay
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kk - p)
        state[5*i+2] = e1 * aa + e2 * vv
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p
        if self.FLOAT_MODE == "bf16":
            wkv = (a / b).type(torch.bfloat16)
        elif self.FLOAT_MODE == "fp16":
            wkv = (a / b).half()
        else:
            wkv = a / b
        
        return ow @ (r * wkv)

    def forward(self, x, state, preprocess_only = False):
        with torch.no_grad():
            w = self.w
            args = self.args

            if state == None:
                state = torch.zeros(args.n_layer * 5, args.n_embd, device=self.RUN_DEVICE)
                for i in range(args.n_layer):
                    state[5*i+4] -= 1e30

            for i in range(args.n_layer):
                ww = w.blocks[i].att
                x = x + self.SA(self.LN(x, w.blocks[i].ln1), state, i, 
                    ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)
                
                ww = w.blocks[i].ffn
                x = x + self.FF(self.LN(x, w.blocks[i].ln2), state, i, 
                    ww.time_mix_k, ww.time_mix_r, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight)
                
                if (i+1) % RWKV_RESCALE_LAYER == 0:
                    x = x / 2
            
            return x.float(), state
        
class Encoder(torch.nn.Module):
    def __init__(self, args, emb, ln_weight, ln_bias):
        super().__init__()
        self.args = args
        self.emb = emb
        self.ln_weight = ln_weight
        self.ln_bias = ln_bias

    def forward(self, token_embd):
        return F.layer_norm(token_embd, (self.args.n_embd,), weight=self.ln_weight, bias=self.ln_bias)
    
class Decoder(torch.nn.Module):
    def __init__(self, args, head_weight, ln_weight, ln_bias):
        super().__init__()
        self.args = args
        self.head_weight = head_weight
        self.ln_weight = ln_weight
        self.ln_bias = ln_bias

    def forward(self, x):
        x = F.layer_norm(x, (self.args.n_embd,), weight=self.ln_weight, bias=self.ln_bias)
        return self.head_weight @ x
    
class RWKV(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.FLOAT_MODE = args.FLOAT_MODE

        with torch.no_grad():
            w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
            # refine weights and send to correct device
            keys = list(w.keys())
            for x in keys:
                block_id = 0
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                if 'att.output.weight' in x:
                    w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))
                if 'ffn.value.weight' in x:
                    w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))
                                
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                if '.time_decay' in x:
                    w[x] = w[x].float()
                    w[x] = -torch.exp(w[x])
                elif '.time_first' in x:
                    w[x] = w[x].float()
                else:
                    if self.FLOAT_MODE == "fp32":
                        w[x] = w[x].float()
                    elif self.FLOAT_MODE == "bf16":
                        w[x] = w[x].bfloat16()
                    elif self.FLOAT_MODE == "fp16":
                        w[x] = w[x].half()

                w[x].requires_grad = False

        # store weights in self.w
        keys = list(w.keys())
        mixer_w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = mixer_w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        self.encoder = Encoder(args, mixer_w.emb.weight, mixer_w.blocks[0].ln0.weight, mixer_w.blocks[0].ln0.bias)
        self.encoder.eval()
        self.decoder = Decoder(args, mixer_w.head.weight, mixer_w.ln_out.weight, mixer_w.ln_out.bias)
        self.decoder.eval()
        self.mixer = RWKV_RNN(args, mixer_w)
        self.mixer.eval()
        self.emb_weight = mixer_w.emb.weight
        self.eval()
        gc.collect()

    def forward(self, token_embd, state):
        with torch.no_grad():
            x, state = self.mixer.forward(self.encoder.forward(token_embd), state)
            return self.decoder.forward(x), state