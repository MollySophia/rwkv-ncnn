# rwkv-ncnn

## ***NOTICE***
- This repo is currently at an early stage. Expect things like relatively low inferring speed.
- Optimizations are on the way

## Convert model file
- ~~Get the latest pnnx binary and put in this folder (or use 'get_pnnx.sh' on linux)~~ Build pnnx at the latest commit and put pnnx binary in this folder(since moduleop requires some fixes in pnnx after the latest pnnx release)
- Get the RWKV-4-World-CHNtuned-3B model file from [https://huggingface.co/BlinkDL/rwkv-4-world/blob/main/RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096.pth](https://huggingface.co/BlinkDL/rwkv-4-world/blob/main/RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096.pth)
- Modify and run ```python convert_model.py```

## Build how-to

```
$ mkdir build && cd build
$ cmake ..
$ make
```

## Run the model on NCNN
- Convert the model
- run the built binary
- Wait for it

## Example output
```
$ ./chat_rwkv_ncnn 
Loading model files...
User: Hello! How are you today?
Assisstant: I'm doing well, thank you. How can I assist you today?
User: Write a poem for me.
Assisstant: Sure, I'd be happy to. What kind of poem would you like?
User: Write a poem about flowers
Assisstant: Flowers are a beautiful and delicate symbol of love and beauty. They symbolize the fleeting nature of life and the fleeting nature of happiness. They also symbolize the fleeting nature of time and the inevitability of death.
```

```
$ ./chat_rwkv_ncnn
Loading model files...
User: 你好
Assisstant: 你好，有什么我可以帮助你的吗？
User: 企鹅会飞吗
Assisstant: 不，企鹅不会飞。它是一种鸟类，不会飞行。
```

## Acknowledgements
// TODO