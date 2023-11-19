import os
import json
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 划分训练集验证集
def divide(feature_sequence_3,type_int,val_rate):
    # 打乱数据集
    index = [i for i in range(feature_sequence_3.shape[0])]
    np.random.shuffle(index)
    feature_sequence_3 = feature_sequence_3[index]
    type_int = type_int[index]

    # 划分数据集
    train_data = []
    valid_data = []
    train_label = []
    valid_label = []
    valid_data_num = int(feature_sequence_3.shape[0] * val_rate)
    train_data_num = feature_sequence_3.shape[0] - valid_data_num
    for i in range(len(feature_sequence_3)):
        if i < train_data_num:
            train_data.append(feature_sequence_3[i])
            train_label.append(type_int[i])
        else:
            valid_data.append(feature_sequence_3[i])
            valid_label.append(type_int[i])
    train_data = np.array(train_data)
    valid_data = np.array(valid_data)
    train_label = np.array(train_label)
    valid_label = np.array(valid_label)
    return train_data,valid_data,train_label,valid_label


def generate_ds(feature_sequence_3,
                type_int,
                # train_im_height: int = 160,
                # train_im_width: int = 160,
                # val_im_height: int = 160,
                # val_im_width: int = 160,
                train_im_height: int = 224,
                train_im_width: int = 224,
                val_im_height: int = 224,
                val_im_width: int = 224,
                batch_size: int = 8,
                val_rate: float = 0.1,
                cache_data: bool = False):
    """
    读取划分数据集，并生成训练集和验证集的迭代器
    :param feature_sequence_3: 数据根目录
    :param train_im_height: 训练输入网络图像的高度
    :param train_im_width:  训练输入网络图像的宽度
    :param val_im_height: 验证输入网络图像的高度
    :param val_im_width:  验证输入网络图像的宽度
    :param batch_size: 训练使用的batch size
    :param val_rate:  将数据按给定比例划分到验证集
    :param cache_data: 是否缓存数据
    :return:
    """
    assert train_im_height is not None
    assert train_im_width is not None
    if val_im_width is None:
        val_im_width = train_im_width
    if val_im_height is None:
        val_im_height = train_im_height

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # 把输入矩阵变成固定大小并进行相关处理
    def process_train_info(train_data, label):
        image = train_data
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.int32)
        image = tf.image.resize_with_crop_or_pad(image, train_im_height, train_im_width)
        # image = tf.image.random_flip_left_right(image) #将图片左右反转
        # image = (image / 255. - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] #标准化图片
        return image, label

    def process_val_info(train_data, label):
        image = train_data
        # image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.int32)
        image = tf.image.resize_with_crop_or_pad(image, train_im_height, train_im_width) #改为统一的224格式，填充
        # image = (image / 255. - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] #将图片进行归一化
        return image, label

    # Configure dataset for performance
    def configure_for_performance(ds,
                                  shuffle_size: int,
                                  shuffle: bool = False,
                                  cache: bool = False):
        if cache:
            ds = ds.cache()  # 读取数据后缓存至内存
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_size)  # 打乱数据顺序
        ds = ds.batch(batch_size)                      # 指定batch size
        ds = ds.prefetch(buffer_size=AUTOTUNE)         # 在训练的同时提前准备下一个step的数据
        return ds

    # 划分训练集和测试集
    train_data,valid_data,train_label,valid_label = divide(feature_sequence_3,type_int,val_rate)
    total_train = len(train_label)
    total_val = len(valid_label)
    print('按比例划分后 train_data：',train_data.shape)
    print('按比例划分后 valid_data：',valid_data.shape)

    #处理数据集
    train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(train_data),
                                                   tf.constant(train_label)))
    print('train_ds111',train_ds)  # <TensorSliceDataset shapes: ((129, 129, 3), ()), types: (tf.float64, tf.int64)>
    #total_train = len(train_img_path)

    # Use Dataset.map to create a dataset of image, label pairs
    train_ds = train_ds.map(process_train_info, num_parallel_calls=AUTOTUNE)
    print('train_ds222',train_ds)  # <ParallelMapDataset shapes: ((224, 224, 3), ()), types: (tf.float32, tf.int32)>
    train_ds = configure_for_performance(train_ds, total_train, shuffle=True, cache=cache_data)
    print('train_ds333',train_ds)  # <PrefetchDataset shapes: ((None, 224, 224, 3), (None,)), types: (tf.float32, tf.int32)>

    val_ds = tf.data.Dataset.from_tensor_slices((tf.constant(valid_data),
                                                 tf.constant(valid_label)))
    # total_val = len(val_img_path)
    # Use Dataset.map to create a dataset of image, label pairs
    val_ds = val_ds.map(process_val_info, num_parallel_calls=AUTOTUNE)
    val_ds = configure_for_performance(val_ds, total_val, cache=False)
    print('val_ds',val_ds)
    return train_ds, val_ds
