import numpy as np
import ncnn
from ncnn.utils.functional import sigmoid

# ncnn element type
# 0 = auto
# 1 = float32
# 2 = float16
# 3 = int8
data_type = 1

class RWKV_Channel_Mixing_NCNN_Layer(ncnn.Layer):
    channel_mixing_layers = []

    def ReLU(x):
        return x * (x > 0)

    def __init__(self):
        ncnn.Layer.__init__(self)
        self.one_blob_only = True
        self.channel_mixing_layers.append(self)

    def forward(self, bottom_blobs, top_blobs, opt):
        x = np.array(bottom_blobs[0])
        state = np.array(bottom_blobs[1])

        xk = x * self.time_mix_k + state * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + state * (1 - self.time_mix_r)
        state = x
        r = sigmoid(np.multiply(self.rw, xr))
        k = np.square(self.ReLU(np.multiply(self.kw, xk)))
        kv = np.multiply(self.vw, k)

        x = r * kv
        top_blobs[0].clone_from(ncnn.Mat(x))
        top_blobs[1].clone_from(ncnn.Mat(state))

        return 0
    
    def load_param(self, pd):
        self.time_mix_k_shape = pd.get(10, ncnn.Mat())
        self.time_mix_r_shape = pd.get(11, ncnn.Mat())
        self.rw_shape = pd.get(12, ncnn.Mat())
        self.kw_shape = pd.get(13, ncnn.Mat())
        self.vw_shape = pd.get(14, ncnn.Mat())

    def load_model(self, mb):
        self.time_mix_k = mb.load(self.time_mix_k_shape[0], data_type)
        self.time_mix_r = mb.load(self.time_mix_r_shape[0], data_type)
        self.rw = mb.load(self.rw_shape[0], self.rw_shape[1], data_type)
        self.kw = mb.load(self.kw_shape[0], self.kw_shape[1], data_type)
        self.vw = mb.load(self.vw_shape[0], self.vw_shape[1], data_type)

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

    def forward(self, bottom_blobs, top_blobs, opt):
        x = np.array(bottom_blobs[0])
        state = np.array(bottom_blobs[1])
        state_a = np.array(bottom_blobs[2])
        state_b = np.array(bottom_blobs[3])
        state_p = np.array(bottom_blobs[4])

        xk = x * self.time_mix_k + state * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + state * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + state * (1 - self.time_mix_r)
        state = x
        r = sigmoid(np.multiply(self.rw, xr))
        kk = np.multiply(self.kw, xk)
        vv = np.multiply(self.vw, xv)
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

        x = np.multiply(self.ow, (r * wkv))

        top_blobs[0].clone_from(ncnn.Mat(x))
        top_blobs[1].clone_from(ncnn.Mat(state))
        top_blobs[2].clone_from(ncnn.Mat(state_a))
        top_blobs[3].clone_from(ncnn.Mat(state_b))
        top_blobs[4].clone_from(ncnn.Mat(state_p))

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

    def load_model(self, mb):
        print("2 load_model called!")
        self.time_mix_k = mb.load(self.time_mix_k_shape[0], data_type)
        self.time_mix_v = mb.load(self.time_mix_v_shape[0], data_type)
        self.time_mix_r = mb.load(self.time_mix_r_shape[0], data_type)
        self.rw = mb.load(self.rw_shape[0], self.rw_shape[1], data_type)
        self.kw = mb.load(self.kw_shape[0], self.kw_shape[1], data_type)
        self.vw = mb.load(self.vw_shape[0], self.vw_shape[1], data_type)
        self.time_first = mb.load(self.time_first_shape[0], data_type)
        self.time_decay = mb.load(self.time_decay_shape[0], data_type)
        self.ow = mb.load(self.ow_shape[0], self.ow_shape[1], data_type)

def RWKV_Time_Mixing_layer_creator():
    return RWKV_Time_Mixing_NCNN_Layer()

def RWKV_Time_Mixing_layer_destroyer(layer):
    for i in range(len(RWKV_Time_Mixing_NCNN_Layer.time_mixing_layers)):
        if RWKV_Time_Mixing_NCNN_Layer.time_mixing_layers[i] == layer:
            del RWKV_Time_Mixing_NCNN_Layer.time_mixing_layers[i]
            break