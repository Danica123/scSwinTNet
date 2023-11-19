import os
import json
import glob
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,matthews_corrcoef
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

#导入对应的模型
# from model import swin_tiny_patch4_window7_160 as create_model
from model import swin_tiny_patch4_window7_224 as create_model
# from model import swin_small_patch4_window7_224 as create_model
# from model import swin_base_patch4_window7_224 as create_model
# from model import swin_base_patch4_window12_384 as create_model
# from model import swin_base_patch4_window7_224_in22k as create_model#epoch40accuracy,f1,mcc： 0.8399007136208502 0.5828295339724767 0.716586037869663
# from model import swin_base_patch4_window12_384_in22k as create_model
# from model import swin_large_patch4_window7_224_in22k as create_model
# from model import swin_large_patch4_window12_384_in22k as create_model

from dataread import type_num
from dataread import test_feature_sequence_3
from dataread import test_label_int

def main():
    num_classes = type_num
    im_height = im_width = 224
    # im_height = im_width = 160
    image = test_feature_sequence_3
    image = tf.cast(image, tf.float32)

    # 改为统一的224格式，填充
    image = tf.image.resize_with_crop_or_pad(image, im_width, im_height)

    # create model
    model = create_model(num_classes=num_classes, has_logits=False)
    # model.build([1, im_height, im_width, 3])
    model.build((1, 224, 224, 3))
    # model.build((1, 224, 224, 3))
    # 载入模型参数
    # weights_path = './save_weightd/Human_Blood_model_3223_swin_tiny_patch4_window7_224_epoch100.ckpt'
    # weights_path = './save_weights/Human_Blood_model_3223_swin_small_patch4_window7_224.ckpt'
    # weights_path = './save_weights/Human_Blood_model_3223_swin_base_patch4_window7_224.ckpt'
    # weights_path = './save_weights/Human_Blood_model_3223_swin_base_patch4_window12_384.ckpt'
    # weights_path = './save_weights/Human_Blood_model_3223_swin_base_patch4_window7_224_in22k_epoch100.ckpt'
    # weights_path = './save_weights/Human_Blood_model_3223_swin_base_patch4_window12_384_in22k.ckpt'
    # weights_path = './save_weights/Human_Blood_model_3223_swin_large_patch4_window7_224_in22k.ckpt'
    # weights_path = './save_weights/Human_Blood_model_3223_swin_large_patch4_window12_384_in22k.ckpt'

    # weights_path = './save_weightd/Human_Blood_model_3223_swin_tiny_patch4_window7_224_epoch100.ckpt'
    # weights_path = './save_weights/Human_Kidney_model_5675_swin_tiny_patch4_window7_224_epoch100.ckpt'
    # weights_path = './save_weights/Human_Lung_model_6338_swin_tiny_patch4_window7_224_epoch100.ckpt'
    # weights_path ='./save_weights/Human_Pancreas_model_51_swin_tiny_patch4_window7_224_epoch100.ckpt'
    # weights_path = './save_weights/Human_Spleen_model_9887_swin_tiny_patch4_window7_224_epoch100.ckpt'
    # weights_path = './save_weights/Human_Spleen_model_9887_swin_tiny_patch4_window7_224_epoch60.ckpt'
    #tiny
    # weights_path = './tiny_weights/Human_Lung6338_swin_tiny_patch4_window7_224_epoch50.ckpt'
    # weights_path = './tiny_weights/Human_Lung_swin_tiny_patch4_window7_224_epoch50.ckpt'
    # weights_path = 'tiny_weights/Human_Lung6338AdamBN_swin_tiny_patch4_window7_224_epoch50'
    # weights_path = 'tiny_weights/Mouse_Pancreas1354newAdamBNFLT_swin_tiny_patch4_window7_224_epoch50.ckpt'
    # weights_path = './new_tiny_weights/Mouse_Testis2584newAdamBNFLT_epoch50.ckpt'

    #newHuman
    weights_path = './new_tiny_weights/mouse_Kidney7926_epoch100_224.ckpt'



    assert len(glob.glob(weights_path+"*")), "cannot find {}".format(weights_path)
    model.load_weights(weights_path)

    predict = model.predict(image, batch_size= 1)  # 64，1
    print('predict', predict.shape)
    print(predict[0])

    # predict转为class
    predict_max = []
    for i in range(predict.shape[0]):
        temp = []
        temp = np.argmax(predict[i])
        temp = temp
        predict_max.append(temp)
    predict_max = np.array(predict_max)
    print('predict_max', predict_max.shape)
    print('predict_max', predict_max)

    #predict_max写入csv
    with open('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/visual/7926y_pred.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(predict_max)

    # 计算预测到“不确定”类型的概率
    nan = 0
    for i in range(len(predict_max)):
        if predict_max[i] == type_num - 1:
            print('预测为不确定的类别', predict_max[i])
            nan = nan + 1
    print('预测到不确定类型的cell数', nan)
    nan_score = nan / len(predict_max)
    print('“不确定”类型的概率', nan_score)

    # 计算各个评价指标的值
    data_pred = predict_max
    data_true = test_label_int
    f1 = f1_score(data_true, data_pred, average='macro', zero_division=1)  # 越1越好
    precision = precision_score(data_true, data_pred, average='macro', zero_division=1)  # 越1
    recall = recall_score(data_true, data_pred, average='macro', zero_division=1)  # 越1
    accuracy = accuracy_score(data_true, data_pred)
    mcc = matthews_corrcoef(data_true, data_pred)
    print('accuracy,f1,mcc,precision,recall：', accuracy, f1, mcc, precision, recall)


if __name__ == '__main__':
    main()
