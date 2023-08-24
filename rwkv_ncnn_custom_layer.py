import numpy as np
import ncnn
from ncnn.utils.functional import sigmoid
import struct

# ncnn element type
# 0 = auto
# 1 = float32
# 2 = float16
# 3 = int8
data_type = 1

def ftoi(input : float):
    return struct.unpack('I', struct.pack('f', input))[0]

class RWKV_Channel_Mixing_NCNN_Layer(ncnn.Layer):
    channel_mixing_layers = []

    def ReLU(x):
        return x * (x > 0)

    def __init__(self):
        ncnn.Layer.__init__(self)
        self.one_blob_only = True
        self.channel_mixing_layers.append(self)

    def forward_inplace(self, bottom_top_blobs, opt):
        x = np.array(bottom_top_blobs[0])
        state = np.array(bottom_top_blobs[1])

        xk = x * self.time_mix_k + state * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + state * (1 - self.time_mix_r)
        state = x
        r = sigmoid(self.rw @ xr)
        k = np.square(self.ReLU(self.kw @ xk))
        kv = self.vw @ k

        x = r * kv
        print("Channel_Mixing")

        return 0
    
    def load_param(self, pd):
        self.time_mix_k_shape = pd.get(10, ncnn.Mat())
        self.time_mix_r_shape = pd.get(11, ncnn.Mat())
        self.rw_shape = pd.get(12, ncnn.Mat())
        self.kw_shape = pd.get(13, ncnn.Mat())
        self.vw_shape = pd.get(14, ncnn.Mat())
        return 0

    def load_model(self, mb):
        self.time_mix_k = np.array(mb.load(ftoi(self.time_mix_k_shape[0]), data_type))
        self.time_mix_r = np.array(mb.load(ftoi(self.time_mix_r_shape[0]), data_type))
        self.rw = np.array(mb.load(ftoi(self.rw_shape[0]) * ftoi(self.rw_shape[1]), data_type).reshape(ftoi(self.rw_shape[0]), ftoi(self.rw_shape[1])))
        self.kw = np.array(mb.load(ftoi(self.kw_shape[0]) * ftoi(self.kw_shape[1]), data_type).reshape(ftoi(self.kw_shape[0]), ftoi(self.kw_shape[1])))
        self.vw = np.array(mb.load(ftoi(self.vw_shape[0]) * ftoi(self.vw_shape[1]), data_type).reshape(ftoi(self.vw_shape[0]), ftoi(self.vw_shape[1])))
        return 0

def RWKV_Channel_Mixing_layer_creator():
    return RWKV_Channel_Mixing_NCNN_Layer()

def RWKV_Channel_Mixing_layer_destroyer(layer):
    for i in range(len(RWKV_Channel_Mixing_NCNN_Layer.channel_mixing_layers)):
        if RWKV_Channel_Mixing_NCNN_Layer.channel_mixing_layers[i] == layer:
            del RWKV_Channel_Mixing_NCNN_Layer.channel_mixing_layers[i]
            break

class RWKV_Time_Mixing_NCNN_Layer(ncnn.Layer):
    time_mixing_layers = []

    def __init__(self):
        ncnn.Layer.__init__(self)
        self.one_blob_only = True
        self.time_mixing_layers.append(self)

    def forward_inplace(self, bottom_top_blobs, opt):
        x = np.array(bottom_top_blobs[0])
        state = np.array(bottom_top_blobs[1])
        state_a = np.array(bottom_top_blobs[2])
        state_b = np.array(bottom_top_blobs[3])
        state_p = np.array(bottom_top_blobs[4])

        xk = x * self.time_mix_k + state * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + state * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + state * (1 - self.time_mix_r)
        state = x
        r = sigmoid(self.rw @ xr)
        kk = self.kw @ xk
        vv = self.vw @ xv
        ww = self.time_first + kk
        p = np.maximum(state_p, ww)
        e1 = np.exp(state_p - p)
        e2 = np.exp(ww - p)
        a = e1 * state_a + e2 * vv
        b = e1 * state_b + e2

        ww = state_p + self.time_decay
        p = np.maximum(ww, kk)
        e1 = np.exp(ww - p)
        e2 = np.exp(kk - p)
        state_a = e1 * state_a + e2 * vv
        state_b = e1 * state_b + e2
        state_p = p
        wkv = a / b

        x = self.ow @ (r * wkv)

        print("Time_Mixing")

        return 0
    
    def load_param(self, pd):
        self.time_mix_k_shape = pd.get(10, ncnn.Mat())
        self.time_mix_v_shape = pd.get(11, ncnn.Mat())
        self.time_mix_r_shape = pd.get(12, ncnn.Mat())
        self.rw_shape = pd.get(13, ncnn.Mat())
        self.kw_shape = pd.get(14, ncnn.Mat())
        self.vw_shape = pd.get(15, ncnn.Mat())
        self.time_first_shape = pd.get(16, ncnn.Mat())
        self.time_decay_shape = pd.get(17, ncnn.Mat())
        self.ow_shape = pd.get(18, ncnn.Mat())
        return 0

    def load_model(self, mb):
        self.time_mix_k = np.array(mb.load(ftoi(self.time_mix_k_shape[0]), data_type))
        self.time_mix_v = np.array(mb.load(ftoi(self.time_mix_v_shape[0]), data_type))
        self.time_mix_r = np.array(mb.load(ftoi(self.time_mix_r_shape[0]), data_type))
        self.rw = np.array(mb.load(ftoi(self.rw_shape[0]) * ftoi(self.rw_shape[1]), data_type).reshape(ftoi(self.rw_shape[0]), ftoi(self.rw_shape[1])))
        self.kw = np.array(mb.load(ftoi(self.kw_shape[0]) * ftoi(self.kw_shape[1]), data_type).reshape(ftoi(self.kw_shape[0]), ftoi(self.kw_shape[1])))
        self.vw = np.array(mb.load(ftoi(self.vw_shape[0]) * ftoi(self.vw_shape[1]), data_type).reshape(ftoi(self.vw_shape[0]), ftoi(self.vw_shape[1])))
        self.time_first = np.array(mb.load(ftoi(self.time_first_shape[0]), data_type))
        self.time_decay = np.array(mb.load(ftoi(self.time_decay_shape[0]), data_type))
        self.ow = np.array(mb.load(ftoi(self.ow_shape[0]) * ftoi(self.ow_shape[1]), data_type).reshape(ftoi(self.ow_shape[0]), ftoi(self.ow_shape[1])))
        return 0

def RWKV_Time_Mixing_layer_creator():
    return RWKV_Time_Mixing_NCNN_Layer()

def RWKV_Time_Mixing_layer_destroyer(layer):
    for i in range(len(RWKV_Time_Mixing_NCNN_Layer.time_mixing_layers)):
        if RWKV_Time_Mixing_NCNN_Layer.time_mixing_layers[i] == layer:
            del RWKV_Time_Mixing_NCNN_Layer.time_mixing_layers[i]
            break