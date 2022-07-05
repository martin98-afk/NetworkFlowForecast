# NetworkFlowForecast

# 无线网业务预测

## 0. Prerequisites

```bash
# 安装所需工具包
pip install -r ./requirements.txt
```

## 1. Data

目前全量数据只提取出：宿迁，常州，徐州，淮安，作为样例数据。

## 2. Data Preprocess

```bash
# 提取全省各市数据，并进行缺失值填充
python NaExtractAndFill.py
```

## 3. Data Preprocess

```bash
# 分解时间特征，提取节假日信息
# 目前只获取到2026年的信息，如果预测时间超过2026年需要手动修改程序的时间
python GetDateTable.py
```

## 4. Model Structure

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
previous info (InputLayer)      [(None, 60, 13)]     0
__________________________________________________________________________________________________
tf.__operators__.getitem (Slici (None, 60, 1)        0           previous info[0][0]
__________________________________________________________________________________________________
time_distributed (TimeDistribut (None, 60, 2)        4           tf.__operators__.getitem[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 60, 15)       0           previous info[0][0]              
time_distributed[0][0]
__________________________________________________________________________________________________
predict info (InputLayer)       [(None, 150, 7)]     0
__________________________________________________________________________________________________
layer_normalization (LayerNorma (None, 60, 15)       30          concatenate[0][0]
__________________________________________________________________________________________________
transformer_block_2 (Transforme (None, 150, 7)       1018        predict info[0][0]
__________________________________________________________________________________________________
transformer_block (TransformerB (None, 60, 15)       2090        layer_normalization[0][0]
__________________________________________________________________________________________________
tf.math.multiply_10 (TFOpLambda (None, 150, 7)       0           transformer_block_2[0][0]
__________________________________________________________________________________________________
tf.math.multiply_11 (TFOpLambda (None, 150, 7)       0           predict info[0][0]
__________________________________________________________________________________________________
tf.math.multiply (TFOpLambda)   (None, 60, 15)       0           transformer_block[0][0]
__________________________________________________________________________________________________
tf.math.multiply_1 (TFOpLambda) (None, 60, 15)       0           layer_normalization[0][0]
__________________________________________________________________________________________________
tf.__operators__.add_5 (TFOpLam (None, 150, 7)       0           tf.math.multiply_10[0][0]        
tf.math.multiply_11[0][0]
__________________________________________________________________________________________________
tf.__operators__.add (TFOpLambd (None, 60, 15)       0           tf.math.multiply[0][0]           
tf.math.multiply_1[0][0]
__________________________________________________________________________________________________
transformer_block_3 (Transforme (None, 150, 7)       1018        tf.__operators__.add_5[0][0]
__________________________________________________________________________________________________
transformer_block_1 (Transforme (None, 60, 15)       2090        tf.__operators__.add[0][0]
__________________________________________________________________________________________________
tf.math.multiply_18 (TFOpLambda (None, 150, 7)       0           transformer_block_3[0][0]
__________________________________________________________________________________________________
tf.math.multiply_19 (TFOpLambda (None, 150, 7)       0           tf.__operators__.add_5[0][0]
__________________________________________________________________________________________________
tf.math.multiply_6 (TFOpLambda) (None, 60, 15)       0           transformer_block_1[0][0]
__________________________________________________________________________________________________
tf.math.multiply_7 (TFOpLambda) (None, 60, 15)       0           tf.__operators__.add[0][0]
__________________________________________________________________________________________________
tf.__operators__.add_9 (TFOpLam (None, 150, 7)       0           tf.math.multiply_18[0][0]        
tf.math.multiply_19[0][0]
__________________________________________________________________________________________________
tf.__operators__.add_3 (TFOpLam (None, 60, 15)       0           tf.math.multiply_6[0][0]         
tf.math.multiply_7[0][0]
__________________________________________________________________________________________________
multi_head_attention_4 (MultiHe (None, 150, 7)       759         tf.__operators__.add_9[0][0]     
tf.__operators__.add_3[0][0]     
tf.__operators__.add_3[0][0]
__________________________________________________________________________________________________
tf.__operators__.add_11 (TFOpLa (None, 150, 7)       0           tf.__operators__.add_9[0][0]     
multi_head_attention_4[0][0]
__________________________________________________________________________________________________
layer_normalization_9 (LayerNor (None, 150, 7)       14          tf.__operators__.add_11[0][0]
__________________________________________________________________________________________________
feed_forward_4 (FeedForward)    (None, 150, 7)       501         layer_normalization_9[0][0]
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 150, 1)       8           feed_forward_4[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 150)          0           dense_10[0][0]
==================================================================================================
Total params: 7,532
Trainable params: 7,532
Non-trainable params: 0
__________________________________________________________________________________________________
```

## 5. StartTraining

```bash
# mode : na_test_5g_00_00, na_test_5g_01_10, na_test_5g_10_20, na_test_5g_20_30
# result : predict, validate 
# console_output : True, False
python TrainAndPredict.py --mode na_test_5g_00_00 --result predict --console_output True
```


## 6. Result

###程序输出结果存储地址：

TrainAndPredict.py ------>  ./result/out_TrainAndPredict.log

NaExtractAndFill.py ------>  ./result/out_NaExtractAndFill.log

###预测结果输出地址： 

result_region_transformer_Inp60_Out150_ + 城市名

###预测文件名：

所选缺失值模式_stationID_预测模式.csv

样例： na_test_5g_00_00_264499_79876_5G_13_predict.csv

![avatar](./figures/val_TrainBeforeSum.png)