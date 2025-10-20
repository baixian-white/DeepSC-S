# -*- coding: utf-8 -*-                           # 指定源码文件的字符编码为 UTF-8，保证中文注释不乱码
"""
Created on Sat Mar  5 13:36:35 2022               # 文件创建日期（信息性注释）
@author: Zhenzi Weng                                # 作者信息
"""
from __future__ import absolute_import              # 兼容 Python2/3 的导入行为：避免相对导入的歧义
from __future__ import division                     # 使用真正的除法（Python2 中 1/2 = 0 的问题在此修正）
from __future__ import print_function               # 使用 Python3 的 print 函数语法

import tensorflow as tf                             # 导入 TensorFlow 主包
from tensorflow.keras.layers import (               # 从 tf.keras.layers 导入常用层
    Conv2D,                                         # 2D 卷积层
    Conv2DTranspose,                                # 2D 反卷积（转置卷积）层
    GlobalAveragePooling2D,                         # 全局平均池化 2D（SE 模块的 squeeze）
    Dense,                                          # 全连接层
    Concatenate,                                    # 张量在通道维拼接
    BatchNormalization                              # 批归一化
)
from speech_processing import (                     # 从本项目的 speech_processing.py 导入信号处理函数
    enframe,                                        # 按帧切分一维波形 -> [num_frame, frame_length]
    deframe,                                        # 帧序列重组回一维波形
    wav_norm,                                       # 归一化波形，返回归一化后的信号及其均值/方差
    wav_denorm                                      # 用记录的均值/方差执行反归一化
)

def conv_bn_layer(inputs, filters, strides, name):  # 封装：Conv2D + BN，不带激活
    conv = Conv2D(filters=filters,                  # 卷积核个数（输出通道数）
                  kernel_size=(5, 5),               # 卷积核大小 5x5，这个 5×5 卷积核是为了让模型在时频域上一次看更大的上下文，提取到更有意义的语音语义特征，同时兼顾计算效率和表达能力。
                  strides=strides,                  # 步幅（下采样比例）
                  padding="same",                   # SAME 填充，保持空间尺寸按步幅缩放，输出尺寸 = 输入尺寸 ÷ stride （向上取整），如果原本的特征图是6乘6，如果stride是1，那么输出还是6乘6，因为还要补两圈0
                  use_bias=False,                   # 结合 BN，bias 可省，BN = batch Normalization，实际就是对每一层的输出，先减均值、再除方差，让它变成“均值 0、方差 1”的标准分布，再加上可学习的缩放和偏移。防止梯度爆炸的
                  name="{}_conv".format(name)       # 层名，便于可视化和权重管理
                 )(inputs)                          # 以函数式 API 调用层
    conv_bn = BatchNormalization(                   # 紧接 BN，稳定训练
        name="{}_bn".format(name)
    )(conv)
    return conv_bn                                  # 返回没有激活的特征图（外部再接 ReLU），一般来说，如果使用了BN，通常的顺序是conv->BN->ReLU

def convtrans_bn_layer(inputs, filters, strides, name): # 封装：Conv2DTranspose + BN（上采样）
    convtrans = Conv2DTranspose(                    # 反卷积：用于空间尺度放大
        filters=filters,
        kernel_size=(5, 5),
        strides=strides,                            # 步幅 >1 时实现上采样
        padding="same",
        use_bias=False,
        name="{}_convtrans".format(name)
    )(inputs)
    convtrans_bn = BatchNormalization(              # 配合 BN
        name="{}_bn".format(name)
    )(convtrans)
    return convtrans_bn

###################  SE-ResNet  ###################  # 下面是 SE-ResNeXt 风格的残差模块参数
depth = 128                                         # 每个分支的卷积输出通道数（transform 层 filters）
cardinality = 4                                     # 分组/分支数（split 成 4 路）
reduction_ratio = 4                                 # SE 模块的压缩比 r（通道注意力的降维比例）

def global_average_pooling(inputs, name):           # Squeeze：对 HxW 做全局平均 -> [B, C]
    pooling_output = GlobalAveragePooling2D(        # 等价对空间维度求平均
        name="{}_squeeze".format(name)
    )(inputs)
    return pooling_output

def transform_layer(inputs, filters, strides, name):# ResNeXt 里的变换层：Conv+BN+ReLU
    conv_bn = conv_bn_layer(inputs=inputs,
                            filters=filters,
                            strides=strides,
                            name=name)
    transform_output = tf.nn.relu(conv_bn)          # 非线性激活
    return transform_output

def split_layer(inputs, filters, strides, name):    # 将输入复制成多路分支，每路做 transform，再拼接
    layers_split = list()                           # 用 Python list 收集每个分支输出
    for i in range(cardinality):                    # 建立 cardinality 条分支
        splits = transform_layer(inputs=inputs,     # 每条分支共享相同的超参
                                 filters=filters,
                                 strides=strides,
                                 name="{}_transform{}".format(name, i))
        layers_split.append(splits)                 # 收集
    split_output = Concatenate(axis=-1)(layers_split) # 通道维拼接（类似 grouped conv 的等效实现）
    return split_output

def transition_layer(inputs, out_dim, name):        # 过渡层：1x1 卷积调整通道数 + BN
    transition_output = Conv2D(filters=out_dim,     # 输出通道调整到 out_dim
                               kernel_size=(1, 1),  # 1x1 卷积仅做通道映射
                               strides=(1, 1),
                               padding="same",
                               use_bias=False,
                               name="{}_conv".format(name)
                              )(inputs)
    transition_output = BatchNormalization(         # BN 稳定分布
        name="{}_bn".format(name)
    )(transition_output)
    return transition_output

def SE_layer(SE_input, out_dim, reduction_ratio, name): # SE 注意力：通道重标定
    squeeze = global_average_pooling(SE_input, name=name) #对每个通道求平均值（把 H×W 压缩成 1 个数） [B, H, W, C]-》[B, C]
    excitation = Dense(units=out_dim/reduction_ratio,     # 降维（C -> C/r），注意这里使用的是浮点除法
                       use_bias=False,
                       name="{}_dense1".format(name)
                      )(squeeze)
    excitation = tf.nn.relu(excitation)                   # ReLU
    excitation = Dense(units=out_dim,                     # 升维（C/r -> C）
                       use_bias=False,
                       name="{}_dense2".format(name)
                      )(excitation)
    excitation = tf.keras.activations.sigmoid(excitation) # Sigmoid 到 [0,1] 权重
    SE_output = tf.reshape(excitation, [-1, 1, 1, out_dim]) # 形状变成 [B, 1, 1, C] 以便逐通道缩放
    return SE_output

def SEResNet(inputs, out_dim, name):               # 完整 SE-ResNet block：Split -> 1x1 -> SE -> 残差加和
    split_output = split_layer(inputs,              # 多分支特征抽取
                               filters=depth,
                               strides=(1, 1),
                               name="{}_split".format(name))
    transition_output = transition_layer(split_output,    # 1x1 调整通道到 out_dim
                                         out_dim=out_dim,
                                         name="{}_transition".format(name))
    SE_output = SE_layer(transition_output,                # 计算通道权重
                         out_dim=out_dim,
                         reduction_ratio=reduction_ratio,
                         name="{}_SE".format(name))
    SEResNet_output = tf.math.add(                        # 残差连接：输入 + (SE权重 * 变换后特征)
        inputs,
        tf.math.multiply(SE_output, transition_output)
    )
    return SEResNet_output

###################  model function  ###################  # 下面是四个功能模块：语义编/解码器 + 信道编/解 + 信道模型

# semantic encoder（语义编码器）
class Sem_Enc(object):  
    def __init__(self, frame_length, stride_length, args): # 初始化时传入帧长、帧移、以及 args 超参
        self.num_frame = args.num_frame                    # 帧数（例如 128）
        self.frame_length = frame_length                   # 每帧采样点数（例如 128）
        self.stride_length = stride_length                 # 帧移采样点数（例如 128）
        self.sem_enc_outdims = args.sem_enc_outdims        # 每层/模块的通道配置（列表）

    def __call__(self, _input):                            # 以函数形式调用模块（_input 为一维波形 [B, T]）
        # preprocessing _intput
        _input, batch_mean, batch_var = wav_norm(_input)   # 归一化波形，同时记录均值和方差用于反归一化
        _input = enframe(_input,                           # 划窗成帧：[B, num_frame, frame_length]
                          self.num_frame,
                          self.frame_length,
                          self.stride_length)
        _input = tf.expand_dims(_input, axis=-1)           # 升维为 “单通道图像”：[B, F, L, 1]

        ######################   semantic encoder   ######################
        _output = conv_bn_layer(_input,                    # 第1个卷积块（带下采样）
                                filters=self.sem_enc_outdims[0],
                                strides=(2, 2),
                                name="sem_enc_cnn1")
        _output = tf.nn.relu(_output)                      # ReLU
        _output = conv_bn_layer(_output,                   # 第2个卷积块（带下采样）
                                filters=self.sem_enc_outdims[1],
                                strides=(2, 2),
                                name="sem_enc_cnn2")
        _output = tf.nn.relu(_output)                      # ReLU
        for module_count, outdim in enumerate(self.sem_enc_outdims[2:]): # 若干个 SE-ResNet 模块堆叠
            module_id = module_count + 1                   # 从 1 开始计数，便于命名
            _output = SEResNet(_output,                    # SE-ResNet，通道数 outdim
                               out_dim=outdim,
                               name="sem_enc_module{}".format(module_id))
            _output = tf.nn.relu(_output)                  # 模块后接 ReLU
        return _output, batch_mean, batch_var              # 返回语义特征图，以及归一化统计量

# channel encoder（信道编码器）
class Chan_Enc(object):  
    def __init__(self, frame_length, args):
        self.num_frame = args.num_frame                    # 保留帧数信息（本实现未直接用到）
        self.frame_length = frame_length                   # 帧长（未直接用到）
        self.chan_enc_filters = args.chan_enc_filters      # 信道编码器卷积的通道列表

    def __call__(self, _intput):                           # 输入为语义编码器输出的特征图
        ######################   chanel encoder   ######################         
        _output = conv_bn_layer(_intput,                   # 一层 Conv+BN，strides=1 不改变空间尺寸
                                filters=self.chan_enc_filters[0],
                                strides=(1, 1),
                                name="chan_enc_cnn1")
        return _output                                     # 不加激活，这里按作者实现返回 BN 后特征

# channel model（信道模型）
class Chan_Model(object):  
    """Define MIMO channel model."""                      # 注释：定义（简化）MIMO 信道模型（Rayleigh + AWGN）
    def __init__(self, name):
        self.name = name                                   # 保留一个名字（未被进一步使用）

    def __call__(self, _input, std):                       # 输入特征、噪声标准差 std（控制 SNR）
        _input = tf.transpose(_input, perm=[0, 3, 1, 2])   # 调整维度 [B, H, W, C] -> [B, C, H, W] 便于后续 reshape

        batch_size = tf.shape(_input)[0]                   # 动态 batch 大小
        _shape = _input.get_shape().as_list()              # 静态形状（列表），形如 [None, C, H, W]
        assert (_shape[2]*_shape[3]) % 2 == 0, "number of transmitted symbols must be an integer."# 断言 H*W 是偶数，便于两两分组成 (I,Q)

        # reshape layer and normalize the average power of each dim in x into 0.5
        x = tf.reshape(_input,                             # 将空间维合并为 “符号” 维，并把最后维度分成 2（实部/虚部）
                        [batch_size, _shape[1], _shape[2]*_shape[3]//2, 2])#所以reshape之后就是[B, C, (H*W)//2, 2]
        x_norm = tf.math.sqrt(_shape[2]*_shape[3]//2 / 2.0) * tf.math.l2_normalize( # L2 归一化 + 缩放，使每维平均功率≈0.5
            x, axis=2
        )

        x_real = x_norm[:, :, :, 0]                        # 取第 0 通道作为实部
        x_imag = x_norm[:, :, :, 1]                        # 取第 1 通道作为虚部
        x_complex = tf.dtypes.complex(real=x_real,         # 组合为复数张量 x (B, C, Nsym)
                                      imag=x_imag)

        # channel h
        h = tf.random.normal(shape=[batch_size, _shape[1], 1, 2], dtype=tf.float32) # 生成复高斯增益（实部/虚部分开）
        h = (tf.math.sqrt(1./2.) + tf.math.sqrt(1./2.)*h) / tf.math.sqrt(2.)        # 标准化（均值/方差调节）
        h_real = h[:, :, :, 0]                              # 实部
        h_imag = h[:, :, :, 1]                              # 虚部
        h_complex = tf.dtypes.complex(real=h_real,          # 合成为复数 h
                                      imag=h_imag)

        # noise n
        n = tf.random.normal(shape=tf.shape(x),             # 生成与 x 同形状的高斯噪声（实/虚两路）
                              mean=0.0,
                              stddev=std,
                              dtype=tf.float32)
        n_real = n[:, :, :, 0]                              # 实部
        n_imag = n[:, :, :, 1]                              # 虚部
        n_complex = tf.dtypes.complex(real=n_real,          # 合成复数噪声 n
                                      imag=n_imag)

        # receive y
        y_complex = tf.math.multiply(h_complex, x_complex) + n_complex  # 接收端：y = h * x + n

        # estimate x_hat with perfect CSI
        x_hat_complex = tf.math.divide(y_complex, h_complex)            # 理想信道估计（已知 h）：x_hat = y / h

        # convert complex to real
        x_hat_real = tf.expand_dims(tf.math.real(x_hat_complex), axis=-1) # 取实部，恢复到最后维为 1
        x_hat_imag = tf.expand_dims(tf.math.imag(x_hat_complex), axis=-1) # 取虚部
        x_hat = tf.concat([x_hat_real, x_hat_imag], -1)                   # 拼接回 [B, C, Nsym, 2]

        _output = tf.reshape(x_hat, shape=tf.shape(_input))               # 还原为 [B, C, H, W]
        _output = tf.transpose(_output, perm=[0, 2, 3, 1])                # 转回 [B, H, W, C] 供后续卷积处理
        return _output

# channel decoder（信道解码器）
class Chan_Dec(object):  
    def __init__(self, frame_length, args):
        self.num_frame = args.num_frame                    # 保留帧数信息（未直接用到）
        self.frame_length = frame_length                   # 帧长（未直接用到）
        self.chan_dec_filters = args.chan_dec_filters      # 信道解码器通道配置

    def __call__(self, _input):                            # 输入为信道输出特征图
        ######################   channel decoder   ######################  
        _output = conv_bn_layer(_input,                    # 一层 Conv+BN，strides=1
                                filters=self.chan_dec_filters[0],
                                strides=(1, 1),
                                name="chan_dec_cnn1")
        _output = tf.nn.relu(_output)                      # ReLU
        return _output

# semantic decoder（语义解码器）
class Sem_Dec(object):  
    def __init__(self, frame_length, stride_length, args):
        self.num_frame = args.num_frame                    # 帧数
        self.frame_length=frame_length                     # 帧长
        self.stride_length = stride_length                 # 帧移
        self.sem_dec_outdims = args.sem_dec_outdims        # 语义解码器各阶段通道配置

    def __call__(self, _input, batch_mean, batch_var):     # 输入特征 + 训练时记录的均值/方差（用于反归一化）
        ######################   semantic decoder   ######################
        for module_count, outdim in enumerate(self.sem_dec_outdims[:-2]): # 若干个 SE-ResNet 模块（对称于编码器）
            module_id = module_count + 1
            _input = SEResNet(_input,
                              out_dim=outdim,
                              name="sem_dec_module{}".format(module_id))
            _input = tf.nn.relu(_input)
        _output = convtrans_bn_layer(_input,               # 反卷积上采样 1
                                     filters=self.sem_dec_outdims[-2],
                                     strides=(2, 2),
                                     name="sem_dec_cnn1")
        _output = tf.nn.relu(_output)
        _output = convtrans_bn_layer(_output,              # 反卷积上采样 2（空间维恢复到帧级分辨率）
                                     filters=self.sem_dec_outdims[-1],
                                     strides=(2, 2),
                                     name="sem_dec_cnn2")
        _output = tf.nn.relu(_output)

        # last layer
        _output = Conv2D(filters=1,                        # 最后一层 1x1 卷积，压到单通道
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding="same",
                         use_bias=False,
                         name="sem_dec_cnn3")(_output)
        _output = tf.squeeze(_output, axis=-1)             # 去掉最后的通道维 -> [B, F, L]

        # processing _output
        _output = deframe(_output,                         # 按帧重组回一维波形（与 enframe 逆操作）
                          self.num_frame,
                          self.frame_length,
                          self.stride_length)
        _output = wav_denorm(_output,                      # 用 batch_mean/batch_var 反归一化回原幅度尺度
                             batch_mean,
                             batch_var)
        return _output

###################  defined models  ###################  # 将上面模块包成 Keras Model，便于 main.py 组网与训练

# semantic encoder
def sem_enc_model(frame_length, stride_length, args):
    wav_size = args.num_frame*stride_length +  frame_length - stride_length                # #计算输入波形长度（窗口长度）= F*stride + frame - stride
    _input = tf.keras.layers.Input(name="wav_input",       # 定义一维波形输入 [T]
                                   shape=(wav_size,),
                                   dtype=tf.float32)
    sem_enc = Sem_Enc(frame_length,                        # 实例化编码器
                      stride_length,
                      args)
    _output, batch_mean, batch_var = sem_enc(_input)       # 前向得到特征与统计量
    model = tf.keras.models.Model(inputs=_input,           # 封装成 Keras 模型
                                  outputs=[_output,        # 输出三个张量：语义特征、均值、方差
                                           batch_mean,
                                           batch_var],
                                  name="Semantic_Encoder")
    return model

# channel encoder
def chan_enc_model(frame_length, args):
    _input = tf.keras.layers.Input(                        # 信道编码器输入是 4D 特征图
        name="chan_enc_input",
        shape=(args.num_frame//4,                          # 空间维 H：编码阶段下采样了 2 次 -> /4
               frame_length//4,                            # 空间维 W：同理 /4
               args.sem_enc_outdims[-1]),                  # 通道维：与语义编码器最后一层通道一致
        dtype=tf.float32
    )
    chan_enc = Chan_Enc(frame_length, args)                # 实例化
    _output = chan_enc(_input)                             # 前向
    model = tf.keras.models.Model(inputs=_input,           # 包装成模型
                                  outputs=_output,
                                  name="Channel_Encoder")
    return model

# channel decoder
def chan_dec_model(frame_length, args):
    _input = tf.keras.layers.Input(                        # 信道解码器输入形状
        name="chan_dec_input",
        shape=(args.num_frame//4,                          # 与信道编码器输出尺寸一致
               frame_length//4,
               args.chan_enc_filters[-1]),                 # 通道数取信道编码器最后的 filters
        dtype=tf.float32
    )
    chan_dec = Chan_Dec(frame_length, args)                # 实例化
    _output = chan_dec(_input)                             # 前向
    model = tf.keras.models.Model(inputs=_input,           # 包装
                                  outputs=_output,
                                  name="Channel_Decoder")
    return model

# semantic decoder
def sem_dec_model(frame_length, stride_length, args):
    _intput = tf.keras.layers.Input(                       # 语义解码器输入（注意原代码变量名拼写 intput）
        name="sem_dec_intput",
        shape=(args.num_frame//4,
               frame_length//4,
               args.chan_dec_filters[-1]),                 # 与信道解码器输出通道匹配
        dtype=tf.float32
    )
    batch_mean = tf.keras.layers.Input(                    # 额外输入：归一化时记录的 batch 均值
        name="batch_mean",
        shape=(1,),
        dtype=tf.float32
    )
    batch_var = tf.keras.layers.Input(                     # 额外输入：归一化时记录的 batch 方差
        name="batch_var",
        shape=(1,),
        dtype=tf.float32
    )
    sem_dec = Sem_Dec(frame_length,                        # 实例化语义解码器
                      stride_length,
                      args)
    _output = sem_dec(_intput,                             # 前向：需要传入均值与方差以便反归一化
                      batch_mean,
                      batch_var)
    model = tf.keras.models.Model(inputs=[_intput,         # Keras 模型：三个输入
                                          batch_mean,
                                          batch_var],
                                  outputs=_output,         # 输出为重建波形
                                  name="Semantic_Decoder")
    return model
