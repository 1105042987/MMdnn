# PyTorch README

Currently, we have already implemented both the the PyTorch -> IR part and the IR -> PyTorch part.

Models                   | Caffe | CoreML | CNTK | Keras | MXNet | PyTorch | TensorFlow| Onnx
:-----------------------:|:-----:|:------:|:----:|:-----:|:-----:|:-------:|:------:|:------:|
Vgg16                    |   √   |   √    |      |   √   |   √   |    √    | √       | √
Inception_v3             |   √   |   √    |      |   √   |   √   |    √    | √       | √
ResNet 50                |   √   |   √    |      |   √   |   √   |    √    | √       | √
MobileNet V1             |   √   |   √    |      |   √   |   √   |    √    | √       | √
Tiny-yolo                |       |   √    |      |   √   |   √   |    √    | √       | √

**√** - Correctness tested

**o** - Some difference after conversion

**space** - not tested


The PyTorch parser is modified from branch [pytorch](https://github.com/Microsoft/MMdnn/tree/pytorch) , using jit CppOP to build the graph.

Any contribution is welcome.

## Extract PyTorch pre-trained models

You can refer [PyTorch model extractor](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/examples/pytorch/extractor.py) to extract your pytorch models.

```bash
$ mmdownload -f pytorch -h
Support frameworks: ['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']

$ mmdownload -f pytorch -n resnet101 -o ./
Downloading: "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth" to /home/ruzhang/.torch/models/resnet101-5d3b4d8f.pth
███████████████████| 102502400/102502400 [00:06<00:00, 15858546.50it/s]
PyTorch pretrained model is saved as [./imagenet_resnet101.pth].

```

### Convert Pytorch pre-trained models to IR
You can convert the whole pytorch model to IR structure. Please remember for the generality, we now only take the whole model `pth`, not just the state dict. To be more specific, it is save using `torch.save()` and `torch.load()` can load the whole model.

```bash
$ mmtoir -f pytorch -d resnet101 --inputShape 3,224,224 -n imagenet_resnet101.pth
```

Please bear in mind that always add `--inputShape` argparse. This thing is different from other framework because pytorch is a dynamic framework.

Then you will get
```
IR network structure is saved as [resnet101.json].
IR network structure is saved as [resnet101.pb].
IR weights are saved as [resnet101.npy].
```

### 从非官方模型转化到IR

使用这个工具的时候遇到了一个严重的问题：

因为pytorch一直存在一个bug，这个bug不是pytorch本身的问题，而是pytorch所调用的一个获取相对路径包的问题，这导致导入和加载整个模型（即torch.load()方法）时会因为路径问题无法成功，所以pytorch的工作小组建议用户使用加载模型参数的方式进行。

```python
import torch
import torch.nn as nn

# 自定义模型
class MyModel(nn.Module):
    def __init__(self,args):
        pass
    def forward(self,args):
        pass

model_object = MyModel()
# 【这里略去】
# 用任意方式去得到参数，如1.进行训练获得参数，2.读取保存的参数，3.空参数用转化后的模型训练

# 全模型的保存与载入
torch.save(model_object, 'model.pth')
#-------------------------------------------------------------------
model = torch.load('model.pth')

# 仅参数的保存与载入（官方推荐）
torch.save(model_object.state_dict(), 'params.pth')
#--------------------------------------------------------------------
model_object.load_state_dict(torch.load('params.pth'))
```



但是这个工具在内部使用的方法就是 torch.load('model\_dir.pth')。不过我们发现，对于pytorch自行实现过的网络，便可以通过该方式直接加载模型而不会报错。利用这种方式，可以解决该问题。

- 找到torchvision包的安装位置（如果使用Anaconda进行包管理，则地址如下）

  - **%Anaconda安装地址% / envs / [环境名] / Lib / site-packages / torchvision/** 
  - base环境：**%Anaconda安装地址% / Lib / site-packages / torchvision/** 

- 在 **./models/** 文件夹中加入自定义的模型结构，命名为"[name].py"（[name]意为自己决定），并用\_\_all\_\_属性注明import内容，如下所示：

  -  ```python
     import torch.nn as nn
     __all__ = ["MyModel",]
     class MyModel(nn.Module):
         def __init__(self,args):
             # code you need to write
             pass
         def forward(self,args):
             # code you need to write
             pass
     ```

- 在 **./models/\_\_init\_\_.py** 中加入一行新的代码：

  - ```python
    from .[name] import *
    ```

- 到此为止便已经可以使用mmdnn工具转化我们自定义的MyModel模型了，但是依然存在问题，那就是我们保存模型的代码不能再使用之前的方式，而是要替换成如下操作：

  - ```python
    from torchvision.models import MyModel
    model_object = MyModel()
    # 任意手段获取参数
    torch.save(model_object, 'model.pth')
    ```

- **特别强调：**只有用上述办法保存的 **model.pth** 文件才能被用于转化。如果在该文件夹内再次出现MyModel的定义，而不使用import方式获得的话，模型加载时依旧会出现找不到模型的问题。

### Convert models from IR to PyTorch code snippet and weights

You can use following bash command to convert the IR architecture file [*inception_v3.pb*] and weights file [*inception_v3.npy*] to Caffe Python code file[*pytorch_inception_v3.py*] and IR weights file suit for caffe model[*pytorch_inception_v3.npy*]

> Note: We need to transform the IR weights to PyTorch suitable weights. Use argument *-dw* to specify the output weight file name.

```bash
$ mmtocode -f pytorch -n inception_v3.pb --IRWeightPath inception_v3.npy --dstModelPath pytorch_inception_v3.py -dw pytorch_inception_v3.npy

Parse file [inception_v3.pb] with binary format successfully.
Target network code snippet is saved as [pytorch_inception_v3.py].
Target weights are saved as [pytorch_inception_v3.npy].
```

### Generate PyTorch model from code snippet file and weight file

You can use following bash command to generate PyTorch model file [*pytorch_inception_v3.pth*] from python code [*pytorch_inception_v3.py*] and weights file [*pytorch_inception_v3.npy*] for further usage.

```bash
$ mmtomodel -f pytorch -in pytorch_inception_v3.py -iw pytorch_inception_v3.npy -o pytorch_inception_v3.pth

PyTorch model file is saved as [pytorch_inception_v3.pth], generated by [pytorch_inception_v3.py] and [pytorch_inception_v3.npy]. Notice that you may need [pytorch_inception_v3.py] to load the model back.

```

## Example

Detail scripts of *Tensorflow slim resnet_v1_101 model* to *PyTorch* conversion are in [issue 22](https://github.com/Microsoft/MMdnn/issues/22). You can refer it to implement your conversion.

## Develop version

Ubuntu 16.04 with

- PyTorch 0.4.0

@ 2018/04/25

## Links

- [pytorch to keras converter](https://github.com/nerox8664/pytorch2keras)

## Limitation

- The main dataflow in a pytorch network is converted from NHWC(channel last) to NCHW(channel first) format, but some operators (like Concat) with axis may not transform correctly. You may need to correct it manually.

- Currently, no RNN-related operations supported

## FAQ

- There are two types models saved in PyTorch. One is including architecture and weights, which is supported in the MMdnn now. The other  one is only including the weights, which is not supported now.

```python
only_weight_file = "./alexnet-owt-4df8aa71.pth"      # Download from the model zoo
architecture_weight_file = "imagenet_alexnet.pth"    # Download using mmdownload()

m = torch.load(only_weight_file)                    # <class 'collections.OrderedDict'>
m_1 = torch.load(architecture_weight_file)          # <class 'torchvision.models.alexnet.AlexNet'> supported!

```
- When you get the error "AttributeError: 'collections.OrderedDict' object has no attribute 'state_dict'" , it's because you use the model only include weights part. You need to save a new model with archietecture

```python
torch.save(model, filename)
```

- How to load the converted PyTorch model ?

```python

import torch
import imp
import numpy as np
MainModel = imp.load_source('MainModel', "tf_pytorch_vgg19.py")

the_model = torch.load("tf_pytorch_vgg19.pth")
the_model.eval()

x = np.random.random([224,224,3])
x = np.transpose(x, (2, 0, 1))
x = np.expand_dims(x, 0).copy()
data = torch.from_numpy(x)
data = torch.autograd.Variable(data, requires_grad = False).float()

predict = the_model(data)


```


