import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

########################################################################
truelabel = []
file = open('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/visual/6338y_true_str.csv')#读取真实标签
lable_lines = file.readlines()
lable_line_0 = lable_lines[0].strip('\n').split(',')
file.close()
#转int
#lable_line_0 = [int(i) for i in lable_line_0[1:]]
for i in range(1,len(lable_line_0)):
    truelabel.append(str(lable_line_0[i]))
#print(label)#[5, 2, 5, 2, 5....]
print(len(truelabel)) #5729
print(truelabel)
###########################################
predictlabel = []
file = open('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/visual/6338y_pred_str.csv')#读取预测到的标签
lable_lines = file.readlines()
lable_line_0 = lable_lines[0].strip('\n').split(',')
file.close()
#转int
#lable_line_0 = [int(i) for i in lable_line_0[1:]]
for i in range(1, len(lable_line_0)):
    predictlabel.append(str(lable_line_0[i]))
#print(label)#[5, 2, 5, 2, 5....]
print(len(predictlabel)) #5729
print(predictlabel)

################################################
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

plt.subplots(figsize=(15,20))#调整画布大小
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)#相对画布的位置


y_true = truelabel # ['0','1','2','3','4'] # 类似的格式
y_pred = predictlabel# ['2','2','3','1','4'] # 类似的格式


# 对上面进行赋值
C = confusion_matrix(y_true, y_pred, labels=['Basal cell', 'T cell', 'AT2 cell', 'Transformed epithelium', 'Macrophage', 'Endothelium', 'Fibroblast'])  # 可将'1'等替换成自己的类别，如'cat'。
plt.matshow(C, cmap=plt.cm.YlOrBr) # 根据最下面的图按自己需求更改颜色

for i in range(len(C)):
    for j in range(len(C)):
        plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

plt.tick_params(labelsize=10) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。
labels = ['Basal cell', 'T cell', 'AT2 cell', 'Transformed epithelium', 'Macrophage', 'Endothelium', 'Fibroblast']
plt.title('Confusion Matrix')
plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 14}) # 设置字体大小。
plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 14})
plt.xticks(range(0, 7), labels, rotation=90) # 将x轴或y轴坐标，刻度 替换为文字/字符
plt.yticks(range(0, 7), labels)


# plt.ylim(len(labels) - 0.5, -0.5)
fig = plt.gcf()
fig.savefig('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/visual/6338confution.svg',dpi=64,bbox_inches='tight')
cb=plt.colorbar(shrink=0.6, location='bottom',pad = 0.08)#颜色条大小、位置、相对距离
cb.ax.tick_params(labelsize=8)#颜色条字体大小
plt.show()







