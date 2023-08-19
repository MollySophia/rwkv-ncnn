import numpy as np
import ncnn
from ncnn.utils.functional import sigmoid

class RWKV_Time_Mixing_NCNN_Layer(ncnn.Layer):
    time_mixing_layers = []

    def ReLU(x):
        return x * (x > 0)

    def __init__(self):
        ncnn.Layer.__init__(self)
        self.one_blob_only = True
        self.time_mixing_layers.append(self)

    def forward(self, bottom_blobs, top_blobs, opt):
        x = np.array(bottom_blobs[0])
        state = np.array(bottom_blobs[1])
        time_mix_k = np.array(bottom_blobs[2])
        time_mix_r = np.array(bottom_blobs[3])
        kw = np.array(bottom_blobs[4])
        vw = np.array(bottom_blobs[5])
        rw = np.array(bottom_blobs[6])

        xk = x * time_mix_k + state * (1 - time_mix_k)
        xr = x * time_mix_r + state * (1 - time_mix_r)
        state = x
        r = sigmoid(np.multiply(rw, xr))
        k = np.square(self.ReLU(np.multiply(kw, xk)))
        kv = np.multiply(vw, k)

        x = r * kv
        top_blobs[0].clone_from(ncnn.Mat(x))
        top_blobs[1].clone_from(ncnn.Mat(state))

        return 0
    
def RWKV_Time_Mixing_layer_creator():
    return RWKV_Time_Mixing_NCNN_Layer()

def RWKV_Time_Mixing_layer_destroyer(layer):
    for i in range(len(RWKV_Time_Mixing_NCNN_Layer.time_mixing_layers)):
        if RWKV_Time_Mixing_NCNN_Layer.time_mixing_layers[i] == layer:
            del RWKV_Time_Mixing_NCNN_Layer.time_mixing_layers[i]
            break


class RWKV_Channel_Mixing_NCNN_Layer(ncnn.Layer):
    channel_mixing_layers = []

    def __init__(self):
        ncnn.Layer.__init__(self)
        self.one_blob_only = True
        self.channel_mixing_layers.append(self)

    def forward(self, bottom_blobs, top_blobs, opt):
        x = np.array(bottom_blobs[0])
        state = np.array(bottom_blobs[1])
        state_a = np.array(bottom_blobs[2])
        state_b = np.array(bottom_blobs[3])
        state_p = np.array(bottom_blobs[4])
        time_mix_k = np.array(bottom_blobs[5])
        time_mix_v = np.array(bottom_blobs[6])
        time_mix_r = np.array(bottom_blobs[7])
        time_first = np.array(bottom_blobs[8])
        time_decay = np.array(bottom_blobs[9])
        kw = np.array(bottom_blobs[10])
        vw = np.array(bottom_blobs[11])
        rw = np.array(bottom_blobs[12])
        ow = np.array(bottom_blobs[13])

        xk = x * time_mix_k + state * (1 - time_mix_k)
        xv = x * time_mix_v + state * (1 - time_mix_v)
        xr = x * time_mix_r + state * (1 - time_mix_r)
        state = x
        r = sigmoid(np.multiply(rw, xr))
        kk = np.multiply(kw, xk)
        vv = np.multiply(vw, xv)
        ww = time_first + kk
        p = np.maximum(state_p, ww)
        e1 = np.exp(state_p - p)
        e2 = np.exp(ww - p)
        a = e1 * state_a + e2 * vv
        b = e1 * state_b + e2

        ww = state_p + time_decay
        p = np.maximum(ww, kk)
        e1 = np.exp(ww - p)
        e2 = np.exp(kk - p)
        state_a = e1 * state_a + e2 * vv
        state_b = e1 * state_b + e2
        state_p = p
        wkv = a / b

        x = np.multiply(ow, (r * wkv))

        top_blobs[0].clone_from(ncnn.Mat(x))
        top_blobs[1].clone_from(ncnn.Mat(state))
        top_blobs[2].clone_from(ncnn.Mat(state_a))
        top_blobs[3].clone_from(ncnn.Mat(state_b))
        top_blobs[4].clone_from(ncnn.Mat(state_p))

        return 0
    
def RWKV_Channel_Mixing_layer_creator():
    return RWKV_Channel_Mixing_NCNN_Layer()

def RWKV_Channel_Mixing_layer_destroyer(layer):
    for i in range(len(RWKV_Channel_Mixing_NCNN_Layer.channel_mixing_layers)):
        if RWKV_Channel_Mixing_NCNN_Layer.channel_mixing_layers[i] == layer:
            del RWKV_Channel_Mixing_NCNN_Layer.channel_mixing_layers[i]
            break