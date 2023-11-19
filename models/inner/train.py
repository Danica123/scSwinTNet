import os
import re
import datetime
import sys
import math

import tensorflow as tf
from tqdm import tqdm

from model import swin_tiny_patch4_window7_160 as create_model
# from model import swin_tiny_patch4_window7_224 as create_model
# from model import swin_small_patch4_window7_224 as create_model
# from model import swin_base_patch4_window7_224 as create_model
# from model import swin_base_patch4_window7_224_in22k as create_model
# from model import swin_large_patch4_window7_224_in22k as create_model

# from model import swin_base_patch4_window12_384 as create_model
# from model import swin_base_patch4_window12_384_in22k as create_model
# from model import swin_large_patch4_window12_384_in22k as create_model

from utils import generate_ds

from read import type_num
from read import feature_sequence_3
from read import type_int

from numpy.random import seed
import random
import numpy as np
my_seed = 42  # 42，123,66,
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)
# tf.config.experimental_run_functions_eagerly(True)

assert tf.version.VERSION >= "2.4.0", "version of tf must greater/equal than 2.4.0"

#tensorboard可视化
tensorboard = tf.keras.callbacks.TensorBoard(histogram_freq=1)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print('测试GPU是否可用：', tf.test.is_gpu_available())
gpus = tf.config.experimental.list_physical_devices('GPU')
print('gpus', gpus)
# # 此处设置GPU序号
# my_idx = 0
# # 对需要进行限制的GPU进行设置
# tf.config.experimental.set_virtual_device_configuration(gpus[my_idx ],[tf.config.experimental.VirtualDeviceConfiguration()])


def main():

    #输入图片大小img_size
    # img_size = 224
    # batch_size = 16 #过小，会导致模型损失波动大，难以收敛，过大时，模型前期由于梯度的平均，导致收敛速度过慢，调大点8,32,16，
    # epochs = 100
    # num_classes = type_num
    # freeze_layers = False
    # initial_lr = 0.0001 #0.0001,
    # weight_decay = 1e-5

    epochs = 150
    batch_size = 128 #过大，显卡内存不够
    num_classes = type_num
    freeze_layers = False
    initial_lr = 0.0003 #学习率太小，需要提升学习率以加快收敛;过小，会训练缓慢
    weight_decay = 1e-5

    print('epoch大小:', epochs)


    # 每次运行后都做一记录在logs文件夹，按照运行时间建立文件夹
    log_dir = "./In_tiny_logs/human_Ascending_colon2026tiny_epoch50_160.ckpt/"
    # log_dir = "./logs/Mouse_Pancreas1354_swin_tiny_patch4_window7_224_epoch150/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    # data generator with data augmentation
    train_ds, val_ds = generate_ds(feature_sequence_3,
                                   type_int,
                                   train_im_width=160,
                                   train_im_height=160,
                                   batch_size=batch_size,
                                   val_rate=0.2)
    print('train_ds', train_ds)
    print('val_ds', val_ds)

    # create model
    model = create_model(num_classes=num_classes)
    model.build((1, 160, 160, 3))


    # freeze bottom layers
    if freeze_layers:
        for layer in model.layers:
            if "head" not in layer.name:
                layer.trainable = False
            else:
                print("training {}".format(layer.name))

    model.summary()

    # custom learning rate curve
    def scheduler(now_epoch):
        end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
        rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
        new_lr = rate * initial_lr

        # writing lr into tensorboard
        with train_writer.as_default():
            tf.summary.scalar('learning rate', data=new_lr, step=epoch)

        return new_lr

    # using keras low level api for training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    #     from_logits=True)  # 改loss: tf.keras.losses.CategoricalCrossentropy()
    # optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr,
    #                                     momentum=0.9)  # optimizer=tf.keras.optimizers.SGD(lr=0.005)  momentum：加速相关方向的梯度下降并抑制振荡。默认为 0，即普通梯度下降。

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            output = model(train_images, training=True)
            # cross entropy loss
            ce_loss = loss_object(train_labels, output)

            # l2 loss
            matcher = re.compile(".*(bias|gamma|beta).*")
            l2loss = weight_decay * tf.add_n([
                tf.nn.l2_loss(v)
                for v in model.trainable_variables
                if not matcher.match(v.name)
            ])

            loss = ce_loss + l2loss

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(ce_loss)
        train_accuracy(train_labels, output)

    @tf.function
    def val_step(val_images, val_labels):
        output = model(val_images, training=False)
        loss = loss_object(val_labels, output)

        val_loss(loss)
        val_accuracy(val_labels, output)

    best_val_acc = 0.
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # train
        train_bar = tqdm(train_ds, file=sys.stdout)
        for images, labels in train_bar:
            train_step(images, labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())

        # update learning rate
        optimizer.learning_rate = scheduler(epoch)
        # validate
        val_bar = tqdm(val_ds, file=sys.stdout)
        for images, labels in val_bar:
            val_step(images, labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())
        # writing training loss and acc
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), epoch)

        # writing validation loss and acc
        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), epoch)
            tf.summary.scalar("accuracy", val_accuracy.result(), epoch)

        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            # save_name = "./save_weightd/Human_Blood_model_3223_swin_tiny_patch4_window7_224_epoch100.ckpt"
            #save_name = "./save_weights/Human_Blood_model_3223_swin_small_patch4_window7_224.ckpt"#这个不太行，训练集acc最高70.5
            #save_name = "./save_weights/Human_Blood_model_3223_swin_base_patch4_window7_224.ckpt"#训练集acc85.9
            # save_name = "./save_weights/Human_Blood_model_3223_swin_base_patch4_window12_384.ckpt"#训练集acc91.02811
            #save_name = "./save_weights/Human_Blood_model_3223_swin_base_patch4_window7_224_in22k.ckpt"# 40个epoch训练集acc97.4，应该是还没有到达best，可以增加epoch试试看
            # save_name = "./save_weights/Human_Blood_model_3223_swin_base_patch4_window7_224_in22k_epoch100.ckpt"# 100个epoch训练集acc99.85,testacc0.45,过拟合
            # save_name = "./save_weights/Human_Blood_model_3223_swin_base_patch4_window12_384_in22k.ckpt"  #训练集acc0.844，没到best
            # save_name = "./save_weights/Human_Blood_model_3223_swin_large_patch4_window7_224_in22k.ckpt"#训练集acc0.861，没到best，可以增加epoch试试
            # save_name = "./save_weights/Human_Blood_model_3223_swin_large_patch4_window12_384_in22k.ckpt"#训练集acc0.8529072
            # save_name = "./base224_in22k_weights/Human_Brain251_swin_base_patch4_window7_224_in22k_epoch50.ckpt"#
            # save_name = "./tiny_weights/Human_Lung6338_swin_tiny_patch4_window7_224_epoch10.ckpt"  #
            # save_name = "./tiny_weights/Human_Lung6338_swin_tiny_patch4_window7_224_epoch50.ckpt"  #
            # save_name = "./tiny_weights/Human_Lung6338AdamBN_swin_tiny_patch4_window7_224_epoch50"  #
            # save_name = "./tiny_weights/Human_Lung6338AdamBNFLT_swin_tiny_patch4_window7_224_epoch50.ckpt"
            # save_name = "./tiny_weights/Human_Colorectum94AdamBNFLT_swin_tiny_patch4_window7_224_epoch50.ckpt"
            # save_name = "./tiny_weights/Human_Spleen9887seed123AdamBNFLT_swin_tiny_patch4_window7_224_epoch50.ckpt"




            # save_name = "./save_weights/Human_Kidney_model_5675_swin_tiny_patch4_window7_224_epoch100.ckpt"
            # save_name = "./save_weights/Human_Lung_model_6338_swin_tiny_patch4_window7_224_epoch100.ckpt"
            # save_name = "./save_weights/Human_Pancreas_model_51_swin_tiny_patch4_window7_224_epoch100.ckpt"
            # save_name = "./save_weights/Human_Spleen_model_9887_swin_tiny_patch4_window7_224_epoch60.ckpt"
            # save_name = "./save_weights/Human_Liver_model_5105_swin_tiny_patch4_window7_224_epoch200.ckpt"#过拟合
            # save_name = "./save_weights/Human_Brain_model_251_swin_tiny_patch4_window7_224_epoch100.ckpt"
            # save_name = "./save_weights/Human_Colorectum_model_94_swin_tiny_patch4_window7_224_epoch100.ckpt"
            # save_name = "./save_weights/Human_Fetal_kidney_model_540_swin_tiny_patch4_window7_224_epoch100.ckpt"

            #Mouse
            # save_name = "./save_weights/Mouse_Testis2584_swin_tiny_patch4_window7_224_epoch100.ckpt"
            # save_name = "./save_weights/Mouse_Brain2502_swin_tiny_patch4_window7_224_epoch100.ckpt"
            # save_name = "./save_weights/Mouse_Kidney1435_swin_tiny_patch4_window7_224_epoch500.ckpt"
            # save_name = "./save_weights/Mouse_Bone_marrow467_swin_tiny_patch4_window7_224_epoch150.ckpt"
            # save_name = "./save_weights/Mouse_Blood768_swin_tiny_patch4_window7_224_epoch150.ckpt"#0,0,0
            # save_name = "./save_weights/Mouse_Lung6340_swin_tiny_patch4_window7_224_epoch150.ckpt"
            # save_name = "./save_weights/Mouse_Brain2502_swin_tiny_patch4_window7_224_epoch150.ckpt"
            # save_name = "./new_tiny_weights/Mouse_Testis2584newAdamBNFLT_epoch50.ckpt"

            #newHuman
            save_name = "./In_tiny_weights/human_Ascending_colon2026tiny_epoch50_160.ckpt"

            model.save_weights(save_name, save_format="tf")
            print('best_val_acc', best_val_acc)


if __name__ == '__main__':
    main()
