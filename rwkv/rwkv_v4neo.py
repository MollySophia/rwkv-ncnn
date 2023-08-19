import types
import torch
from torch.nn import functional as F
import torch.nn as nn

DEBUG_TIME = False   # True False - show trained time-coeffs

RWKV_RESCALE_LAYER = 6 # set x=x/2 every X layer

############################################################################################################

class RWKV_Channel_Mixing(torch.nn.Module):
    def forward(self, x, state, time_mix_k, time_mix_r, kw, vw, rw):
        xk = x * time_mix_k + state * (1 - time_mix_k)
        xr = x * time_mix_r + state * (1 - time_mix_r)
        state = x

        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))
        kv = vw @ k

        return r * kv, state

class RWKV_Time_Mixing(torch.nn.Module):
    def forward(self, x, state, state_a, state_b, state_p, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xk = x * time_mix_k + state * (1 - time_mix_k)
        xv = x * time_mix_v + state * (1 - time_mix_v)
        xr = x * time_mix_r + state * (1 - time_mix_r)
        state = x

        r = torch.sigmoid(rw @ xr)
        kk = kw @ xk
        vv = vw @ xv
        aa = state_a
        bb = state_b
        pp = state_p
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
        state_a = e1 * aa + e2 * vv
        state_b = e1 * bb + e2
        state_p = p
        wkv = a / b
        
        return ow @ (r * wkv), state, state_a, state_b, state_p
        
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

        self.encoder = Encoder(args, self.w.emb.weight, self.w.blocks[0].ln0.weight, self.w.blocks[0].ln0.bias)
        self.encoder.eval()
        self.decoder = Decoder(args, self.w.head.weight, self.w.ln_out.weight, self.w.ln_out.bias)
        self.decoder.eval()
        self.channel_mixing = RWKV_Channel_Mixing()
        self.time_mixing = RWKV_Time_Mixing()
        self.eval()

    def LN(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    def forward(self, token_embd, state):
        with torch.no_grad():
            x = self.encoder.forward(token_embd)
            w = self.w

            for i in range(self.args.n_layer):
                ww = w.blocks[i].att
                t, state[5*i+1], state[5*i+2], state[5*i+3], state[5*i+4] = self.time_mixing(self.LN(x, w.blocks[i].ln1), 
                    state[5*i+1], state[5*i+2], state[5*i+3], state[5*i+4],
                    ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)
                x += t
                
                ww = w.blocks[i].ffn
                t, state[5*i+0],  = self.channel_mixing(self.LN(x, w.blocks[i].ln2), state[5*i+0], 
                    ww.time_mix_k, ww.time_mix_r, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight)
                x += t
                
                if (i+1) % RWKV_RESCALE_LAYER == 0:
                    x = x / 2

            return self.decoder.forward(x.float()), state