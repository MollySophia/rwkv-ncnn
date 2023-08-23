# rwkv-ncnn

## ***NOTICE***
- This repo is currently at super early stage. Expect things like Segmentation Fault or insanely low inferring speed.
- Optimizations are on the way

## Convert model file
- ~~Get the latest pnnx binary and put in this folder (or use 'get_pnnx.sh' on linux)~~ Build pnnx at the latest commit and put pnnx binary in this folder(since moduleop requires some fixes in pnnx after the latest pnnx release)
- Get the RWKV-4-Raven-7B model file from [https://huggingface.co/BlinkDL/rwkv-4-raven/blob/main/RWKV-4-Raven-7B-v12-Eng49%25-Chn49%25-Jpn1%25-Other1%25-20230530-ctx8192.pth](https://huggingface.co/BlinkDL/rwkv-4-raven/blob/main/RWKV-4-Raven-7B-v12-Eng49%25-Chn49%25-Jpn1%25-Other1%25-20230530-ctx8192.pth)
- Modify and run ```python convert_model.py```

## Run the model on NCNN
- Convert the model
- run ```python chat_ncnn.py```
- Wait for it

## Acknowledgements
// TODO