# cvae_mnist (Final proj. for Big data modeling) 

### A simple conditional variaional auto-encoder for Mnist 
--------------------------------------------------------------
1. Basic configurations can be performed by adapting the mnist.yaml file.
2. Running mode can be switched through terminal.
~~~ Bash
python main.py [--train] [--test]
~~~
3. The network is built with fully connected layers, which has been proved to be ineffective on other datasets such as (Cifar10).
4. Pytorch 1.0 is required.


### 这是一个简单的条件变分自编码器实现
--------------------------------------------------------------
1. 大多数模型设置，数据集设置，还有读取模型，保存模型路径等设置均可以在mnist.yaml文件中进行修改；
2. 模型在怎样的模式下运行可以在终端进行修改；
~~~ Bash
python main.py [--train] [--test]
~~~
3. 网络就用最简单的全连接实现的，效果居然比简单的CNN堆叠要好一些。当然我没有深究，有可能是写代码时候哪里没注意少加了什么。同样的套路放在Cifar10数据集上loss就是降不下去，目前还在探索中。不过应付大数据建模作业应该是足够了[滑稽]。
4. 感觉只要是正常的pytorch应该就能跑，没写什么比较复杂的用法。
