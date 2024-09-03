# rwkv-ncnn

## ***NOTICE***
- This repo is currently at an early stage. Expect things like relatively low inferring speed.
～～- Optimizations are on the way～～

## Convert model file
- Get the RWKV-4-World-CHNtuned-3B model file (or other rwkv models) from [https://huggingface.co/BlinkDL/rwkv-4-world/blob/main/RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096.pth](https://huggingface.co/BlinkDL/rwkv-4-world/blob/main/RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096.pth)
- Run ```python convert_model.py [pth file] [output path] [fp32/fp16]```
- *NOTE:fp16 not implemented yet*

## Build how-to

### Build on Linux
```
$ mkdir build && cd build
$ cmake ..
$ make
```
### Build with Android NDK
```
$ mkdir build && cd build
$ cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_NDK=/opt/android-ndk -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -GNinja ..
$ ninja
```

## Run the model on NCNN
- Convert the model
- run the built binary ```./chat_rwkv_ncnn [model.bin] [model.param] [emb_weight.bin] [vocab.bin] [parameters.txt]```

## Example output
```
$ ./chat_rwkv_ncnn ../output/model.ncnn.bin ../output/model.ncnn.param ../output/emb_weight.bin ../rwkv_vocab_v20230424.bin ../output/parameters.txt
Loading model files...
User: Hello! How are you today?
Assisstant: I'm doing well, thank you. How can I assist you today?
User: Write a poem for me.
Assisstant: Sure, I'd be happy to. What kind of poem would you like?
User: Write a poem about flowers
Assisstant: Flowers are a beautiful and delicate symbol of love and beauty. They symbolize the fleeting nature of life and the fleeting nature of happiness. They also symbolize the fleeting nature of time and the inevitability of death.
```

```
$ ./chat_rwkv_ncnn ../output/model.ncnn.bin ../output/model.ncnn.param ../output/emb_weight.bin ../rwkv_vocab_v20230424.bin ../output/parameters.txt
Loading model files...
User: 你好
Assisstant: 你好，有什么我可以帮助你的吗？
User: 企鹅会飞吗
Assisstant: 不，企鹅不会飞。它是一种鸟类，不会飞行。
```

## Acknowledgements
// TODO
