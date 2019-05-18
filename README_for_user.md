# MMDNN 部分使用帮助

## mmdnn 安装

```bash
pip install mmdnn
```

## mmdnn 所需要的环境支持[部分]（2019/5/19）

| 包名称      | 版本号             |
| ----------- | :----------------- |
| python      | 3.6.2（旧版本）    |
| numpy       | 1.16.2（旧版本）   |
| pytorch-cpu | 0.4.0（旧版本）    |
| tensorflow  | 1.13.1（当前最新） |

## 模型转换步骤

|序号| 对应指令 | 必须输入（可能有其他） | 输出 |
|---| ------- | --- | ---- |
| 1 | mmtoir | 拥有的其他模型参数文件 | \_1\_.pb;  \_1\_.npy;  \_1\_.json; |
| 2 | mmtocode | _1\_.pb;  \_1\_.npy; | \_2\_.py; |
| 3 | mmtomodel | \_2\_.py;  \_1\_.npy; | 需求模型的文件夹 |

其中，第一步指令使用参数从 ***所拥有模型的使用说明书***  获得；第二、三步指令使用的参数从 ***目标模型的使用说明书***  获得。

列表如下：
- [TensorFlow](mmdnn/conversion/tensorflow/README.md) (不管怎么从什么转什么模型，都推荐先阅读该手册) 
- [PyTorch](mmdnn/conversion/pytorch/README.md) 
- [Keras](mmdnn/conversion/keras/README.md) 
- [Caffe](mmdnn/conversion/caffe/README.md) 
- [Microsoft Cognitive Toolkit (CNTK)](mmdnn/conversion/cntk/README.md) 
- [CoreML](mmdnn/conversion/coreml/README.md) 
- [MXNet](mmdnn/conversion/mxnet/README.md) 
- [ONNX](mmdnn/conversion/onnx/README.md) (只能作为目标模型)
- [DarkNet](mmdnn/conversion/darknet/README.md) (只能作为来源模型)

## Android 适配
### 从转化出的TensorFlow模型向TensorFlow Lite 的转化

```python
import tensorflow as tf

# saved_model_dir即为上一部分提到的：需求模型的文件夹
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

### [Android配置方法](<https://github.com/Microsoft/MMdnn/wiki/Deploy-your-TensorFlow-Lite-Model-in-Android>) 

### [其他模型转向TensorFlow Lite的方法](<https://www.tensorflow.org/lite/guide/get_started#2_convert_the_model_format>) 