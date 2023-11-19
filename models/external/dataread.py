# 合并几个data
# 选择相同的feature,type map,输出traindata,testdata csv文件
import numpy as np
import pandas as pd
import math
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 存文件名
def human_cell_atlas_name():
    name = []
    # Blood
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood2156_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood2156_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood2719_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood2719_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood5296_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood5296_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood7160_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood7160_celltype.csv')

    # Brain
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Cerebellum7324_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Cerebellum7324_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Brain251_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Brain251_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Brain1834_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Brain1834_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Brain2892_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Brain2892_celltype.csv')


    # Colorectum#结直肠
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Duodenum4681_data.csv')#十二指肠
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Duodenum4681_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Ileum3367_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Ileum3367_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_JeJunum5549_data.csv')#空场
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_JeJunum5549_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Rectum5718_data.csv')#直肠
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Rectum5718_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Sigmoid_colon3281_data.csv')#乙状结肠
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Sigmoid_colon3281_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Transverse_colon5765_data.csv')#横结肠
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Transverse_colon5765_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Transverse_colon11229_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Transverse_colon11229_celltype.csv')

    # Fetal kidney
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney3057_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney3057_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney4734_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney4734_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney4939_data.csv') # 没用？
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney4939_celltype.csv') # 没用？
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney9932_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney9932_celltype.csv')

    # Kidney肾脏
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Kidney3849_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Kidney3849_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Kidney9153_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Kidney9153_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Kidney9966_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Kidney9966_celltype.csv')

    # Liver
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Liver1811_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Liver1811_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Liver4377_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Liver4377_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Liver4384_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Liver4384_celltype.csv')

    # # Lung
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Lung6022_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Lung6022_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Lung8426_data.csv') # 没用？
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Lung8426_celltype.csv') # 没用？
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Lung9603_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Lung9603_celltype.csv')

    # Pancreas
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Pancreas9727_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Pancreas9727_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas51_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas51_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas180_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas180_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas349_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas349_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas1507_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas1507_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas1841_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas1841_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas2227_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas2227_celltype.csv')

    # Placenta
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Placenta9595_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Placenta9595_celltype.csv')

    # Spleen
    name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Spleen15806_data.csv')
    name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Spleen15806_celltype.csv')

    # tissue实验没有用到的：
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Adipose1372_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Adipose1372_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Adrenal_gland8114_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Adrenal_gland8114_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Artery9652_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Artery9652_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Ascending_colon2026_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Ascending_colon2026_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Bladder1267_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Bladder1267_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Bladder2750_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Bladder2750_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Bone_marrow2261_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Bone_marrow2261_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Bone_marrow6443_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Bone_marrow6443_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Cervix8096_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Cervix8096_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Chorionic_villus9898_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Chorionic_villus9898_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood2150_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood2150_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood4444_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood4444_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood5607_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood5607_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood11297_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood11297_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Epityphlon4486_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Epityphlon4486_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Esophagus2696_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Esophagus2696_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Esophagus8668_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Esophagus8668_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fallopian_tube6556_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fallopian_tube6556_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Female_gonad2710_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Female_gonad2710_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Female_gonad4231_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Female_gonad4231_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_adrenal_gland9875_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_adrenal_gland9875_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain1705_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain1705_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain2904_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain2904_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain3920_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain3920_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain5096_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain5096_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_calvaria15129_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_calvaria15129_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_eye1880_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_eye1880_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_heart2678_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_heart2678_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_heart5319_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_heart5319_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine1338_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine1338_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine1448_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine1448_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine4059_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine4059_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine6931_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine6931_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine9740_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine9740_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_liver17929_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_liver17929_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_Lung4526_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_Lung4526_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_Lung5121_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_Lung5121_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_male_gonad3358_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_male_gonad3358_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_male_gonad9853_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_male_gonad9853_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_muscle18345_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_muscle18345_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_pancreas2830_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_pancreas2830_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_pancreas6939_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_pancreas6939_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_pancreas8977_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_pancreas8977_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_rib1432_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_rib1432_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_rib4560_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_rib4560_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_skin1697_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_skin1697_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_skin5294_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_skin5294_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_spinal_cord5916_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_spinal_cord5916_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_stomach1322_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_stomach1322_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_stomach6631_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_stomach6631_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_thymus2068_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_thymus2068_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_thymus2448_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_thymus2448_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Gall_bladder8905_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Gall_bladder8905_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Heart1308_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Heart1308_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Heart1478_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Heart1478_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Muscle7775_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Muscle7775_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Neonatal_adrenal_gland5863_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Neonatal_adrenal_gland5863_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Omentum1354_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Omentum1354_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Omentum1487_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Omentum1487_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Omentum9971_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Omentum9971_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Pleura9996_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Pleura9996_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Prostat2445_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Prostat2445_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Stomach1879_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Stomach1879_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Stomach4669_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Stomach4669_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Stomach8005_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Stomach8005_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Temporal_lobe9544_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Temporal_lobe9544_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Thyroid6319_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Thyroid6319_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Thyroid6328_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Thyroid6328_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Trachea9949_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Trachea9949_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Ureter2390_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Ureter2390_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Ureter7694_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_cell_atlas/human_Ureter7694_celltype.csv')
    return name
def mouse_cell_atlas_name():
    name = []
    # Blood外周血
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood135_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood135_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood283_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood283_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood352_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood352_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood658_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood658_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood2466_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood2466_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood3201_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood3201_celltype.csv')

    # Bone marrow骨髓
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow510_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow510_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow5298_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow5298_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow8166_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow8166_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow13019_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow13019_celltype.csv')

    # Brain
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Brain753_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Brain753_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Brain3285_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Brain3285_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain2502_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain2502_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain2545_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain2545_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain2695_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain2695_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain3005_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain3005_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain4397_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain4397_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain19431_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain19431_celltype.csv')

    # Fetal brain
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_brain4369_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_brain4369_celltype.csv')

    # Intestine肠道
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Small_intestine1575_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Small_intestine1575_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Small_intestine1671_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Small_intestine1671_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Small_intestine3438_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Small_intestine3438_celltype.csv')

    # Kidney肾脏
    name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Kidney4682_data.csv')
    name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Kidney4682_celltype.csv')


    # Liver肝脏
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Liver261_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Liver261_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Liver4424_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Liver4424_celltype.csv')

    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver3729_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver3729_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver4122_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver4122_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver7761_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver7761_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver18000_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver18000_celltype.csv')

    # Lung肺部
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Lung1414_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Lung1414_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Lung2512_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Lung2512_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Lung3014_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Lung3014_celltype.csv')


    # Mammary gland乳腺
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland648_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland648_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland1059_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland1059_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland1311_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland1311_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland1592_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland1592_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland2081_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland2081_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland3510_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland3510_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland4909_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland4909_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland6633_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland6633_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland6905_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland6905_celltype.csv')

    # Pancreas胰腺
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Pancreas3610_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Pancreas3610_celltype.csv')


    # Spleen
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Spleen1970_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Spleen1970_celltype.csv')

    # Testis睾丸
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Testis2216_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Testis2216_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Testis11789_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Testis11789_celltype.csv')
    #
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis199_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis199_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis296_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis296_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis299_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis299_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis300_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis300_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis398_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis398_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis1662_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis1662_celltype.csv')
    # # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis2584_data.csv')
    # # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis2584_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis4233_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis4233_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis4239_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis4239_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis6598_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis6598_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis8792_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis8792_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis9923_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis9923_celltype.csv')

    # tissue实验没用上
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bladder2746_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bladder2746_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow_mesenchyme7365_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow_mesenchyme7365_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Embryonic_mesenchyme2771_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Embryonic_mesenchyme2771_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_intestine6076_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_intestine6076_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_liver2699_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_liver2699_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_lung6453_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_lung6453_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_stomach6192_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_stomach6192_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Muscle1102_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Muscle1102_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_calvaria3617_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_calvaria3617_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_calvaria4347_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_calvaria4347_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_heart3948_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_heart3948_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_muscle829_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_muscle829_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_muscle4044_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_muscle4044_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_pancreas4571_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_pancreas4571_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_rib1217_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_rib1217_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_rib1963_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_rib1963_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_rib3082_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_rib3082_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_skin3392_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_skin3392_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Ovary1931_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Ovary1931_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Ovary2432_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Ovary2432_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Placenta1873_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Placenta1873_celltype.csv'
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Placenta2473_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Placenta2473_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Prostate1031_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Prostate1031_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Prostate1474_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Prostate1474_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Stomach2389_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Stomach2389_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Testis2216_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Testis2216_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Testis11789_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Testis11789_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Thymus4289_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Thymus4289_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Uterus1704_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Uterus1704_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Uterus2035_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Uterus2035_celltype.csv')
    return name
def celltype_map_name():
    name = []
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Human_Blood.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Human_Brain.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Human_Colorectum.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Human_Fetal kidney.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Human_Kidney.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Human_Liver.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Human_Lung.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Human_Pancreas.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Human_Placenta.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Human_Spleen.csv')

    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Mouse_Blood.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Mouse_Bone_marrow.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Mouse_Brain.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Mouse_Fetal_brain.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Mouse_Intestine.csv')
    name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Mouse_Kidney.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Mouse_Liver.csv')#过拟合
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Mouse_Lung.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Mouse_Mammary_gland.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Mouse_Pancreas.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Mouse_Spleen.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/celltype_map/Mouse_Testis.csv')
    return name
def human_test_name():
    name = []
    # Blood
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Blood2469_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Blood2469_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Blood9649_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Blood9649_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Blood3223_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Blood3223_celltype.csv')

    # Brain
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Brain251_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Brain251_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Brain1834_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Brain1834_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Brain2892_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Brain2892_celltype.csv')

    # Colorectum
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Colorectum94_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Colorectum94_celltype.csv')

    # Fetal kidney
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Fetal_kidney540_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Fetal_kidney540_celltype.csv')

    # Kidney
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Kidney5675_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Kidney5675_celltype.csv')

    # Liver
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Liver298_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Liver298_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Liver3502_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Liver3502_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Liver5105_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Liver5105_celltype.csv')

    # Lung
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Lung2064_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Lung2064_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Lung6338_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Lung6338_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Lung9566_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Lung9566_celltype.csv')

    # Pancreas
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas51_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas51_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas180_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas180_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas349_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas349_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas1507_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas1507_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas1841_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas1841_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas2227_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Pancreas2227_celltype.csv')

    # Placenta
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Placenta615_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Placenta615_celltype.csv')

    # Spleen
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Spleen9887_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Spleen9887_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Spleen11081_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Spleen11081_celltype.csv')
    name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Spleen14848_data.csv')
    name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Spleen14848_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Spleen16286_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Spleen16286_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Spleen18513_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/human_test_data/human_Spleen18513_celltype.csv')
    return name
def mouse_test_name():
    name = []
    # Blood
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Blood768_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Blood768_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Blood1109_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Blood1109_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Blood1223_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Blood1223_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Blood1610_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Blood1610_celltype.csv')

    # Bone marrow
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Bone_marrow47_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Bone_marrow47_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Bone_marrow467_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Bone_marrow467_celltype.csv')

    # Brain
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain2502_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain2502_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain2545_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain2545_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain2695_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain2695_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain3005_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain3005_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain4397_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain4397_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain19431_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Brain19431_celltype.csv')

    # Fetal brain
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Fetal_brain369_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Fetal_brain369_celltype.csv')

    # Intestine
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Intestine28_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Intestine28_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Intestine192_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Intestine192_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Intestine260_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Intestine260_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Intestine529_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Intestine529_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Intestine1449_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Intestine1449_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Intestine3260_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Intestine3260_celltype.csv')

    # Kidney
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Kidney203_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Kidney203_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Kidney1435_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Kidney1435_celltype.csv')
    name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Kidney7926_data.csv')
    name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Kidney7926_celltype.csv')

    # Liver
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver3729_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver3729_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver4122_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver4122_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver7761_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver7761_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver18000_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Liver18000_celltype.csv')

    # Lung
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Lung707_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Lung707_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Lung769_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Lung769_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Lung1920_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Lung1920_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Lung6340_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Lung6340_celltype.csv')

    # Mammary gland
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Mammary_gland133_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Mammary_gland133_celltype.csv')

    # Pancreas
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Pancreas108_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Pancreas108_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Pancreas131_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Pancreas131_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Pancreas207_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Pancreas207_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Pancreas1354_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Pancreas1354_celltype.csv')

    # Spleen
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Spleen1081_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Spleen1081_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Spleen1433_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Spleen1433_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Spleen1759_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Spleen1759_celltype.csv')

    # Testis
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis199_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis199_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis296_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis296_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis299_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis299_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis300_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis300_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis398_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis398_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis1662_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis1662_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis2584_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis2584_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis4233_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis4233_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis4239_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis4239_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis6598_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis6598_celltype.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis8792_data.csv')
    # name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis8792_celltype.csv')
    ## name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis9923_data.csv')
    ## name.append('/home/2023/23dhh/scSwinTNet/dataset/mouse_test_data/mouse_Testis9923_celltype.csv')
    return name


# name = human_cell_atlas_name()
name = mouse_cell_atlas_name()
# test_name = human_test_name()
test_name = mouse_test_name()

# 读特征
def readdata(filename):
    print('读文件',filename) #du
    data = pd.read_csv(filename, header=None, low_memory=False)
    data = np.array(data)

    data = data[1:]
    genename = []
    for i in range(len(data)):
        genename.append(data[i][0])
    genename = np.array(genename)

    newdata = []
    for i in range(len(data)):
        newdata.append(data[i][1:])
    newdata= np.array(newdata).transpose()
    newdata = newdata.astype(np.float)
    return newdata,genename

# 读标签
def readlabel(filename):
    print('读标签',filename)
    type = pd.read_csv(filename)
    type = type['Cell_type']
    type  = np.array(type)
    return type

# 读type_map
def readtypemap(filename):
    print('读文件', filename)
    data = pd.read_csv(filename, header=None, low_memory=False)
    test_type = data[1]
    train_type = data[2]
    total_type = data[3]
    test_type = np.array(test_type)[1:]
    train_type = np.array(train_type)[1:]
    total_type = np.array(total_type)[1:]
    print('输入对应celltypemap',len(train_type))
    return test_type,train_type,total_type

# 读label中type个数
def typenum(type):
    # print('type',type.shape)
    type_short = []
    for i in type:
        if i not in type_short:
            type_short.append(i)
    type_short = np.array(type_short)
    # print('type_short',type_short) # 细胞类型分别名字
    type_num = type_short.shape[0]
    return type_num,type_short

test_type, train_type, total_type = readtypemap(celltype_map_name()[0])
type_num, type_short = typenum(total_type)
print('type_short:', type_short)
print('存在 celltype', type_num)

# 加上预测为训练集不存在cell类型的预测值
type_num = type_num + 1
print('输入模型的celltype',type_num)

# 读train数据feature,label
feature_list = []
label_list = []
genename_list = []
for i in range(len(name)):
    if i%2 == 0:
        feature,genename = readdata(name[i])
        feature_list.append(feature)
        genename_list.append(genename)
    else:
        label = readlabel(name[i])
        label_list.append(label)
print('输入train文件个数:',len(feature_list))

#判断细胞标签最大最小
def max1(a):
    m=a[0]
    n=a[0]
    for x in a:
        if x>m:
            m=x
        elif x<n:
            n=x
    return m,n,len(a)

# 删掉不要的train数据类型
def deletecelltype(feature_list,label_list):
    new_label_list = []
    new_label_int_list = []
    new_feature_list = []
    for i in range(len(label_list)):
        print('第%d个文件'%(i),feature_list[i].shape)
        new_label = []
        new_label_int = []
        new_feature = []
        for j in range(label_list[i].shape[0]):
            temp = 0
            for z in range(len(train_type)):
                if label_list[i][j] == train_type[z]:
                    temp = 1
            if temp == 1:
                for q in range(len(train_type)):
                    if label_list[i][j] == train_type[q]:
                        label_list[i][j] = total_type[q]
                new_label.append(label_list[i][j])
                new_feature.append(feature_list[i][j])
        # 把label变成数字格式
        for t in range(len(new_label)):
            for j in range(len(type_short)):
                if new_label[t] == type_short[j]:
                    new_label_int.append(j)
                    break
        # print('第%d个文件处理后label个数'%(i),np.array(new_label).shape)
        # print('第%d个文件处理后label_int个数'%(i),np.array(new_label_int).shape)
        print('第%d个文件删除cell后'%(i),np.array(new_feature).shape)

        new_label_list.append(new_label)
        new_label_int_list.append(new_label_int)
        new_feature_list.append(new_feature)
    return new_label_list,new_label_int_list,new_feature_list
label_list,label_int_list,feature_list = deletecelltype(feature_list,label_list)
# 合并train的label
train_label_int = []
for n in range(len(label_int_list)):
    for i in range(len(label_int_list[n])):
        train_label_int.append(label_int_list[n][i])
train_label_int = np.array(train_label_int)
print('训练集细胞类型从几开始：', max1(train_label_int))#类型，0，细胞个数
# 读test数据feature,label
test_feature,test_genename = readdata(test_name[0])
test_label = readlabel(test_name[1])
print('输入test文件',test_feature.shape)


# 删掉不要的test数据类型
def deletecelltype_test(feature,label):
    new_label = []
    new_label_int = []
    new_feature = []
    # print('处理前label个数',label.shape)
    for j in range(label.shape[0]):
        temp = 0
        for z in range(len(test_type)):
            if label[j] == test_type[z]:
                temp = 1
        if temp == 1:
            for q in range(len(test_type)):
                if label[j] == test_type[q]:
                    label[j] = total_type[q]
            new_label.append(label[j])
            new_feature.append(feature[j])
    # 把label变成数字格式
    for t in range(len(new_label)):
        for j in range(len(type_short)):
            if new_label[t] == type_short[j]:
                new_label_int.append(j)
                break
    print('测试集细胞类型从几开始：', max1(new_label_int))
    # print('处理后label个数', np.array(new_label).shape)
    # print('处理后label_int个数', np.array(new_label_int).shape)
    print('删除cell后feature个数', np.array(new_feature).shape)
    return new_label, new_label_int, new_feature
test_label,test_label_int,test_feature = deletecelltype_test(test_feature,test_label)


# 取相同的genenname
def samefeature(test_genename,genename_list):
    genename_same = []
    genename_total = []
    for n in range(len(genename_list)):
        for i in range(len(genename_list[n])):
            genename_total.append(genename_list[n][i])
    for j in range(len(test_genename)):
        genename_total.append(test_genename[j])
    genename_total = np.array(genename_total)
    genename_total = sorted(genename_total)
    # print(genename_total)

    genename_short = np.unique(genename_total)
    # print(genename_short)
    genenum = []

    for i in genename_short:
        temp = 0
        for j in range(len(genename_total)):
            if i == genename_total[j]:
                temp = temp + 1
        genenum.append(temp)

    for i in range(len(genenum)):
        if genenum[i] == len(genename_list)+1:
            genename_same.append(genename_short[i])
    genename_same = np.array(genename_same)
    print('feature_same', genename_same.shape)
    return genename_same
feature_same = samefeature(test_genename,genename_list)

# 删除test不要的特征
def deletefeature(feature,feature_same,feature_name):
    new_feature = []
    feature = np.transpose(feature)
    for i in feature_same:
        for j in range(len(feature_name)):
            if i == feature_name[j]:
                new_feature.append(feature[j])
    new_feature = np.transpose(new_feature)
    new_feature = np.array(new_feature)
    return new_feature
test_feature = deletefeature(test_feature,feature_same,test_genename)
print('test_feature', test_feature.shape)

# 删除train不要的特征并合并为一个文件
def deletefeature_train():
    train_feature = []
    for n in range(len(feature_list)):
        temp = deletefeature(feature_list[n],feature_same,genename_list[n])
        for i in range(len(temp)):
            train_feature.append(temp[i])
    train_feature = np.array(train_feature)
    return train_feature
train_feature = deletefeature_train()
print('train_feature',train_feature.shape)


# # 把特征转为224×224的三通道正方形
# def changefeature(feature,genename):
#     # 设置转换为正方形后的边长
#     new_size = 224
#     sequence_length = math.sqrt(genename.shape[0])
#     sequence_length = int(sequence_length) + 1
#
#     if sequence_length < new_size:
#         # Padding if the original size is smaller
#         pad_size = new_size - sequence_length
#         pad_width = ((0, pad_size), (0, pad_size))
#         feature = np.pad(feature, pad_width, mode='constant', constant_values=0)
#     elif sequence_length > new_size:
#         # Cropping if the original size is larger
#         crop_start = (sequence_length - new_size) // 2
#         feature = feature[crop_start:crop_start + new_size, crop_start:crop_start + new_size]
#
#     # Reshape to 224x224x1
#     # feature = feature.reshape((new_size, new_size, 1))
#
#     # Create three identical channels with the same data
#     feature_sequence = np.stack([feature] * 3, axis=-1)
#
#     return feature_sequence, new_size
#
# test_feature_sequence,sequence_length = changefeature(test_feature,feature_same)
# print('test_feature_sequence',test_feature_sequence.shape)
# train_feature_sequence,sequence_length = changefeature(train_feature,feature_same)
# print('train_feature_sequence',train_feature_sequence.shape)
# # print('sequence_length',sequence_length)
# print('test_label_int',len(test_label_int))
# # # print('test_label_int',test_label_int)
# print('train_label_int',len(train_label_int))


# 把特征转为正方形
def changefeature(feature,genename):
    # 计算出转换为正方形后的边长
    sequence_length = math.sqrt(genename.shape[0])
    sequence_length = int(sequence_length) + 1
    feature_sequence = []
    for i in range(1, len(feature) + 1):
        temp = list(feature[i - 1])
        temp = temp + [0] * ((sequence_length * sequence_length) - len(feature[i - 1]))
        temp = np.array(temp)
        temp = temp.reshape((sequence_length, sequence_length, 1))
        feature_sequence.append(temp)
    feature_sequence = np.array(feature_sequence)
    return feature_sequence,sequence_length
test_feature_sequence,sequence_length = changefeature(test_feature,feature_same)
print('test_feature_sequence',test_feature_sequence.shape)
train_feature_sequence,sequence_length = changefeature(train_feature,feature_same)
print('train_feature_sequence',train_feature_sequence.shape)
# print('sequence_length',sequence_length)

# 转为3个通道的格式
def changefeature_3(feature_sequence):
    feature_sequence_3 = []
    for i in range(len(feature_sequence)):
        temp3 = []
        for j in range(len(feature_sequence[i])):
            temp2 = []
            for z in range(len(feature_sequence[i][j])):
                temp1 = []
                temp1.append(feature_sequence[i][j][z][0])
                temp1.append(feature_sequence[i][j][z][0])
                temp1.append(feature_sequence[i][j][z][0])
                temp2.append(temp1)
            temp3.append(temp2)
        feature_sequence_3.append(temp3)
    feature_sequence_3 = np.array(feature_sequence_3)
    return feature_sequence_3
test_feature_sequence_3 = changefeature_3(test_feature_sequence)
train_feature_sequence_3 = changefeature_3(train_feature_sequence)
print('test_feature_sequence_3',test_feature_sequence_3.shape)
print('train_feature_sequence_3',train_feature_sequence_3.shape)

print('test_label_int',len(test_label_int))
# print('test_label_int',test_label_int)
print('train_label_int',len(train_label_int))
