import torch
import sys, os
import types

if len(sys.argv) != 4:
    print("Usage: python convert_model.py [pth file] [output path] [fp16/fp32]")
    exit(1)

rescale_layer_num = 6
n_layer = 0
pth_path = sys.argv[1]
output_path = sys.argv[2]
float_mode = sys.argv[3]
weights = types.SimpleNamespace()

w = torch.load(pth_path, map_location='cpu')
keys = list(w.keys())
for x in keys:
    block_id = 0
    if 'blocks.' in x:
        block_id = int(x.split('.')[1])
        if block_id > n_layer:
            n_layer = block_id
    if 'att.output.weight' in x:
        w[x] = w[x] / (2 ** int(block_id // rescale_layer_num))
    if 'ffn.value.weight' in x:
        w[x] = w[x] / (2 ** int(block_id // rescale_layer_num))
                    
    if '.time_' in x:
        w[x] = w[x].squeeze()
    if '.time_decay' in x:
        w[x] = w[x].float()
        w[x] = -torch.exp(w[x])
    elif '.time_first' in x:
        w[x] = w[x].float()
    else:
        if float_mode == "fp32":
            w[x] = w[x].float()
        elif float_mode == "bf16":
            w[x] = w[x].bfloat16()
        elif float_mode == "fp16":
            w[x] = w[x].half()
    w[x].requires_grad = False

# store weights in weights
keys = list(w.keys())
for x in keys:
    xx = x.split('.')
    here = weights
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

del w

n_layer = n_layer + 1
vocab_size = weights.emb.weight.shape[0]
n_embd = weights.emb.weight.shape[1]

num_ncnn_layers = n_layer * 6 + 4 + int(n_layer / rescale_layer_num)
num_ncnn_blobs = n_layer * 10 + 4 + int(n_layer / rescale_layer_num)

param_lines = ["7767517",
                f"{num_ncnn_layers} {num_ncnn_blobs}",
                "Input in0 0 1 in0",
                "Input in1 0 1 in1",
                f"rwkv.rwkv_v4neo.RWKV_Encoder encoder 1 1 in0 1 -23310=1,{n_embd} -23311=1,{n_embd}"]

out0 = 1
out1 = "in1"
rescale_id = 0
t = weights.blocks[0].ffn.key.weight.shape[0]
time_mixing_params = f"-23310=1,{n_embd} -23311=1,{n_embd} -23312=1,{n_embd} -23313=1,{n_embd} -23314=1,{n_embd} -23315=2,{n_embd},{n_embd} -23316=2,{n_embd},{n_embd} -23317=2,{n_embd},{n_embd} -23318=1,{n_embd} -23319=1,{n_embd} -23320=2,{n_embd},{n_embd}"
channel_mixing_params = f"-23310=1,{n_embd} -23311=1,{n_embd} -23312=1,{n_embd} -23313=1,{n_embd} -23314=2,{n_embd},{n_embd} -23315=2,{t},{n_embd} -23316=2,{n_embd},{t}"
decoder_params = f"-23310=1,{n_embd} -23311=1,{n_embd} -23312=2,{vocab_size},{n_embd}"

print(f"Generating {output_path}/model.ncnn.param")

for layer_id in range(1, n_layer + 1):
    i = out0
    # Split
    param_lines.append(f"Split rwkv_split_{2 * layer_id - 1} 1 2 {i} {i + 3} {i + 2}")
    # Time_Mixing
    param_lines.append(f"rwkv.rwkv_v4neo.RWKV_Time_Mixing rwkv_time_mixing_{layer_id} 2 2 {i + 3} {out1} {i + 4} {i + 6} {time_mixing_params} 1={layer_id - 1}")
    # Add
    param_lines.append(f"BinaryOp rwkv_add_{2 * layer_id - 1} 2 1 {i + 2} {i + 4} {i + 5} 0=0")

    # Split
    param_lines.append(f"Split rwkv_split_{2 * layer_id} 1 2 {i + 5} {i + 7} {i + 8}")
    # Channel_Mixing
    if layer_id == n_layer:
        param_lines.append(f"rwkv.rwkv_v4neo.RWKV_Channel_Mixing rwkv_channel_mixing_{layer_id} 2 2 {i + 7} {i + 6} {i + 9} out1 {channel_mixing_params} 1={layer_id - 1}")
    else:
        param_lines.append(f"rwkv.rwkv_v4neo.RWKV_Channel_Mixing rwkv_channel_mixing_{layer_id} 2 2 {i + 7} {i + 6} {i + 9} {i + 11} {channel_mixing_params} 1={layer_id - 1}")
    # Add
    param_lines.append(f"BinaryOp rwkv_add_{2 * layer_id} 2 1 {i + 8} {i + 9} {i + 10} 0=0")

    out0 = i + 10
    out1 = i + 11

    if layer_id == n_layer:
        param_lines.append(f"rwkv.rwkv_v4neo.RWKV_Decoder decoder 1 1 {out0} out0 {decoder_params}")
    elif layer_id % rescale_layer_num == 0:
        param_lines.append(f"BinaryOp rwkv_rescale_{rescale_id} 1 1 {out0} {out1 + 1} 0=3 1=1 2=2.000000e+00")
        rescale_id = rescale_id + 1
        out0 = out1 + 1

with open(os.path.join(output_path, "model.ncnn.param"), "w") as f:
    f.writelines(s + '\n' for s in param_lines)

def write_weights(file, tensor, float_mode = "fp32"):
    if float_mode == "fp32":
        file.write(tensor.numpy().tobytes())

    # elif float_mode == "fp16":

    # elif float_mode == "bf16":


print(f"Generating {output_path}/model.ncnn.bin")
with open(os.path.join(output_path, "model.ncnn.bin"), "wb") as f:
    # encoder
    write_weights(f, weights.blocks[0].ln0.weight)
    write_weights(f, weights.blocks[0].ln0.bias)
    for i in range(n_layer):
        ww = weights.blocks[i].att
        write_weights(f, weights.blocks[i].ln1.weight)
        write_weights(f, weights.blocks[i].ln1.bias)
        write_weights(f, ww.time_mix_k, float_mode)
        write_weights(f, ww.time_mix_v, float_mode)
        write_weights(f, ww.time_mix_r, float_mode)
        write_weights(f, ww.receptance.weight, float_mode)
        write_weights(f, ww.key.weight, float_mode)
        write_weights(f, ww.value.weight, float_mode)
        write_weights(f, ww.time_first, float_mode)
        write_weights(f, ww.time_decay, float_mode)
        write_weights(f, ww.output.weight, float_mode)

        ww = weights.blocks[i].ffn
        write_weights(f, weights.blocks[i].ln2.weight)
        write_weights(f, weights.blocks[i].ln2.bias)
        write_weights(f, ww.time_mix_k, float_mode)
        write_weights(f, ww.time_mix_r, float_mode)
        write_weights(f, ww.receptance.weight, float_mode)
        write_weights(f, ww.key.weight, float_mode)
        write_weights(f, ww.value.weight, float_mode)

    # decoder
    write_weights(f, weights.ln_out.weight)
    write_weights(f, weights.ln_out.bias)
    write_weights(f, weights.head.weight)

print(f"Generating {output_path}/emb_weight.bin")
weights.emb.weight.numpy().tofile(os.path.join(output_path, "emb_weight.bin"))

print(f"Generating {output_path}/parameters.txt")
with open(os.path.join(output_path, "parameters.txt"), 'w') as f:
    f.write(f'{vocab_size},{n_layer},{n_embd},{float_mode}')