import types
import torch
from torch.nn import functional as F
import torch.nn as nn

DEBUG_TIME = False   # True False - show trained time-coeffs

RWKV_RESCALE_LAYER = 6 # set x=x/2 every X layer

############################################################################################################

class RWKV_Channel_Mixing(torch.nn.Module):
    def forward(self, input, state, layer_num, time_mix_k, time_mix_r, kw, vw, rw, ln_weight, ln_bias, FLOAT_MODE, n_embd):
        x = F.layer_norm(input, (n_embd,), weight=ln_weight, bias=ln_bias)
        i = int(layer_num.item())
        if FLOAT_MODE == "bf16":
            xk = x * time_mix_k + state[5*i+0].type(torch.bfloat16) * (1 - time_mix_k)
            xr = x * time_mix_r + state[5*i+0].type(torch.bfloat16) * (1 - time_mix_r)
            state[5*i+0] = x.float()
        elif FLOAT_MODE == "fp16":
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

        return state, r * kv

class RWKV_Time_Mixing(torch.nn.Module):
    def forward(self, input, state, layer_num, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow, ln_weight, ln_bias, FLOAT_MODE, n_embd):
        x = F.layer_norm(input, (n_embd,), weight=ln_weight, bias=ln_bias)
        i = int(layer_num.item())
        if FLOAT_MODE == "bf16":
            xk = x * time_mix_k + state[5*i+1].type(torch.bfloat16) * (1 - time_mix_k)
            xv = x * time_mix_v + state[5*i+1].type(torch.bfloat16) * (1 - time_mix_v)
            xr = x * time_mix_r + state[5*i+1].type(torch.bfloat16) * (1 - time_mix_r)
            state[5*i+1] = x.float()
        elif FLOAT_MODE == "fp16":
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
        kk = kw @ xk
        vv = vw @ xv

        if '16' in FLOAT_MODE:
            kk = kk.float()
            vv = vv.float()

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
        if FLOAT_MODE == "bf16":
            wkv = (a / b).type(torch.bfloat16)
        elif FLOAT_MODE == "fp16":
            wkv = (a / b).half()
        else:
            wkv = a / b
        
        return state, ow @ (r * wkv)
        
class RWKV_Decoder(torch.nn.Module):
    def forward(self, x, head_weight, ln_weight, ln_bias, n_embd):
        return head_weight @ (F.layer_norm(x.float(), (n_embd,), weight=ln_weight, bias=ln_bias))

class RWKV_Encoder(torch.nn.Module):
    def forward(self, token_embd, ln_weight, ln_bias, n_embd):
        return F.layer_norm(token_embd, (n_embd,), weight=ln_weight, bias=ln_bias)
    
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
        self.w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = self.w
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

        self.encoder = RWKV_Encoder()
        self.encoder.eval()
        self.decoder = RWKV_Decoder()
        self.channel_mixing = RWKV_Channel_Mixing()
        self.time_mixing = RWKV_Time_Mixing()
        self.current_layer = torch.zeros(1)
        self.eval()

    def forward(self, token_embd, state):
        with torch.no_grad():
            x = self.encoder(token_embd, self.w.blocks[0].ln0.weight, self.w.blocks[0].ln0.bias, self.args.n_embd)
            w = self.w

            for i in range(self.args.n_layer):
                self.current_layer[0] = i
                ww = w.blocks[i].att
                state, output = self.time_mixing(x,
                    state, self.current_layer,
                    ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight,
                    w.blocks[i].ln1.weight, w.blocks[i].ln1.bias, self.FLOAT_MODE, self.args.n_embd)
                x += output
                
                ww = w.blocks[i].ffn
                state, output = self.channel_mixing(x, state, self.current_layer,
                    ww.time_mix_k, ww.time_mix_r, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight,
                    w.blocks[i].ln2.weight, w.blocks[i].ln2.bias, self.FLOAT_MODE, self.args.n_embd)
                x += output
                
                if (i+1) % RWKV_RESCALE_LAYER == 0:
                    x = x / 2

            return self.decoder(x, w.head.weight, w.ln_out.weight, w.ln_out.bias, self.args.n_embd), state