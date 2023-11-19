# 合并几个data
# 选择相同的feature,type map,输出traindata,testdata csv文件
import numpy as np
import pandas as pd
import math
import tensorflow as tf

# 存文件名
def human_cell_atlas_name():
    name = []
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Adipose1372_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Adipose1372_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Adrenal_gland8114_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Adrenal_gland8114_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Artery9652_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Artery9652_celltype.csv')

    name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Ascending_colon2026_data.csv')
    name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Ascending_colon2026_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Bladder1267_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Bladder1267_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Bladder2750_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Bladder2750_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Bone_marrow2261_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Bone_marrow2261_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Bone_marrow6443_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Bone_marrow6443_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Cerebellum7324_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Cerebellum7324_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Cervix8096_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Cervix8096_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Duodenum4681_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Duodenum4681_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Epityphlon4486_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Epityphlon4486_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Esophagus2696_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Esophagus2696_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Esophagus8668_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Esophagus8668_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fallopian_tube6556_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fallopian_tube6556_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Gall_bladder8905_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Gall_bladder8905_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Heart1308_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Heart1308_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Heart1478_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Heart1478_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Ileum3367_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Ileum3367_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_JeJunum5549_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_JeJunum5549_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Kidney3849_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Kidney3849_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Kidney9153_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Kidney9153_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Kidney9966_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Kidney9966_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Liver1811_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Liver1811_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Liver4377_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Liver4377_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Liver4384_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Liver4384_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Lung6022_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Lung6022_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Lung8426_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Lung8426_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Lung9603_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Lung9603_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Muscle7775_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Muscle7775_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Omentum1354_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Omentum1354_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Omentum1487_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Omentum1487_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Omentum9971_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Omentum9971_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Pancreas9727_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Pancreas9727_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood2156_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood2156_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood2719_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood2719_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood5296_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood5296_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood7160_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Peripheral_blood7160_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Pleura9996_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Pleura9996_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Prostat2445_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Prostat2445_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Rectum5718_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Rectum5718_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Sigmoid_colon3281_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Sigmoid_colon3281_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Spleen15806_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Spleen15806_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Stomach1879_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Stomach1879_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Stomach4669_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Stomach4669_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Stomach8005_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Stomach8005_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Temporal_lobe9544_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Temporal_lobe9544_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Thyroid6319_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Thyroid6319_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Thyroid6328_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Thyroid6328_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Trachea9949_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Trachea9949_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Transverse_colon5765_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Transverse_colon5765_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Transverse_colon11229_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Transverse_colon11229_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Ureter2390_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Ureter2390_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Ureter7694_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Ureter7694_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Chorionic_villus9898_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Chorionic_villus9898_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood2150_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood2150_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood4444_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood4444_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood5607_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood5607_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood11297_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Cord_blood11297_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_adrenal_gland9875_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_adrenal_gland9875_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain1705_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain1705_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain2904_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain2904_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain3920_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain3920_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain5096_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_brain5096_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_calvaria15129_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_calvaria15129_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_eye1880_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_eye1880_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Female_gonad2710_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Female_gonad2710_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Female_gonad4231_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Female_gonad4231_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_heart2678_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_heart2678_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_heart5319_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_heart5319_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine1338_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine1338_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine1448_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine1448_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine4059_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine4059_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine6931_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine6931_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine9740_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_intestine9740_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney3057_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney3057_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney4734_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney4734_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney4939_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney4939_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney9932_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_kidney9932_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_liver17929_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_liver17929_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_Lung4526_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_Lung4526_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_Lung5121_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_Lung5121_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_male_gonad3358_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_male_gonad3358_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_male_gonad9853_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_male_gonad9853_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_muscle18345_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_muscle18345_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_pancreas2830_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_pancreas2830_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_pancreas6939_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_pancreas6939_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_pancreas8977_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_pancreas8977_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_rib1432_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_rib1432_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_rib4560_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_rib4560_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_skin1697_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_skin1697_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_skin5294_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_skin5294_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_spinal_cord5916_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_spinal_cord5916_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_stomach1322_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_stomach1322_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_stomach6631_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_stomach6631_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_thymus2068_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_thymus2068_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_thymus2448_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Fetal_thymus2448_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Neonatal_adrenal_gland5863_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Neonatal_adrenal_gland5863_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Placenta9595_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/human_cell_atlas/human_Placenta9595_celltype.csv')

    return name

def mouse_cell_atlas_name():
    name = []
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bladder2746_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bladder2746_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow510_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow510_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow5298_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow5298_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow8166_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow8166_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow13019_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow13019_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Brain753_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Brain753_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Brain3285_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Brain3285_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Embryonic_mesenchyme2771_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Embryonic_mesenchyme2771_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_brain4369_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_brain4369_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_intestine6076_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_intestine6076_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_liver2699_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_liver2699_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_lung6453_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_lung6453_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_stomach6192_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Fetal_stomach6192_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Kidney4682_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Kidney4682_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Liver261_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Liver261_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Liver4424_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Liver4424_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Lung1414_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Lung1414_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Lung2512_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Lung2512_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Lung3014_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Lung3014_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland648_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland648_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland1059_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland1059_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland1311_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland1311_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland1592_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland1592_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland2081_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland2081_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland3510_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland3510_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland4909_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland4909_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland6633_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland6633_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland6905_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Mammary_gland6905_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow_mesenchyme7365_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Bone_marrow_mesenchyme7365_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Muscle1102_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Muscle1102_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_calvaria3617_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_calvaria3617_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_calvaria4347_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_calvaria4347_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_heart3948_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_heart3948_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_muscle829_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_muscle829_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_muscle4044_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_muscle4044_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_pancreas4571_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_pancreas4571_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_rib1217_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_rib1217_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_rib1963_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_rib1963_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_rib3082_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_rib3082_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_skin3392_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Neonatal_skin3392_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Ovary1931_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Ovary1931_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Ovary2432_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Ovary2432_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Pancreas3610_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Pancreas3610_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood135_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood135_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood283_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood283_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood352_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood352_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood658_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood658_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood2466_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood2466_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood3201_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Peripheral_blood3201_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Placenta1873_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Placenta1873_celltype.csv'
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Placenta2473_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Placenta2473_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Prostate1031_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Prostate1031_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Prostate1474_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Prostate1474_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Small_intestine1575_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Small_intestine1575_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Small_intestine1671_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Small_intestine1671_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Small_intestine3438_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Small_intestine3438_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Spleen1970_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Spleen1970_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Stomach2389_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Stomach2389_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Testis2216_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Testis2216_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Testis11789_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Testis11789_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Thymus4289_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Thymus4289_celltype.csv')

    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Uterus1704_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Uterus1704_celltype.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Uterus2035_data.csv')
    # name.append('/home/aita/4444/Daihh/scSwinTNet/dataset/mouse_cell_atlas/mouse_Uterus2035_celltype.csv')
    return name

name = human_cell_atlas_name()
# name = mouse_cell_atlas_name()

# 读特征文件
def readdata():

    print('读文件',name[0]) #du
    data = pd.read_csv(name[0], header=None, low_memory=False)
    data = np.array(data)

    cellname = np.array(data[0])
    cellname = cellname[1:]
    print('cellname',cellname)
    print(cellname.shape)

    data = data[1:]
    genename = []
    for i in range(len(data)):
        genename.append(data[i][0])
    genename = np.array(genename)
    print('genename',genename)
    print(genename.shape)

    newdata = []
    for i in range(len(data)):
        newdata.append(data[i][1:])
    newdata= np.array(newdata).transpose()
    newdata = newdata.astype(np.float)
    # newdata = newdata[:len(newdata) - 1]
    print('newdata',newdata)
    print(newdata.shape)
    return newdata, cellname, genename
feature,cellname,genename = readdata()
print("feature",feature.shape)

# 读标签文件/读每个细胞类型
def readtype():
    print('读标签',name[1])
    type = pd.read_csv(name[1])
    type = type['Cell_type']
    type  = np.array(type)
    print('type',type)
    print('type',type.shape)
    return type
type_all = readtype()
# type_small = pd.read_csv(name[2])
# type_small = np.array(type_small)
# type_small = np.squeeze(type_small)
# type_small = np.array(type_small)
# print("type_all.shape",type_all.shape)
# print("type_small.shape",type_small.shape,type_small)


# 读label中type的名字，计算一共多少细胞类型type_num
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
type_num,type_short = typenum(type_all)
print('feature',feature.shape)
print("type_num,type_short",type_num,type_short)

def changetype(type,type_num,type_short):
    # 把标签转换为数字格式
    int_type = []
    for i in range(len(type)):
        for j in range(len(type_short)):
            if type[i] == type_short[j]:
                int_type.append(j)
    int_type = np.array(int_type)
    return int_type
type_int = changetype(type_all, type_num, type_short)
print('type_int',type_int.shape)

# 把特征转为正方形
def changefeature(feature = feature,genename = genename):
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
feature_sequence,sequence_length = changefeature()
print('feature_sequence',feature_sequence.shape)
print('sequence_length',sequence_length)

# 转为3个通道的格式
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


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('输入 feature_sequence_3',feature_sequence_3.shape)
# print(feature_sequence_3[0])
# print('PCA降维 feature_pca', feature_pca.shape)
print('输入类型个数',type_num)
# print(type_int)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
