#!/usr/bin/env python3

import subprocess, sys, os, time


NR_THREAD = 5

## beta机器运行
# train_file_name='hotel_train_20170813.libsvm_2017-08-23.csv'
# test_file_name='hotel_test_20170813.libsvm_2017-08-23.csv'

# train_file_name='toutiao_hotel_behavior_train_feature_20170822.txt.csv'
# test_file_name='toutiao_hotel_behavior_test_feature_20170822.txt.csv'

train_file_name='train_toutiao_std_feature_20170822.libsvm_2017-08-25.csv'
test_file_name='test_toutiao_std_feature_20170822.libsvm_2017-08-25.csv'

## 本地运行
# train_file_name='hotel_train_20170813_libsvm_2017-08-23_1w.csv'
# test_file_name='hotel_test_20170813_libsvm_2017-08-23_1w.csv'

train_file_head=train_file_name.split(".")[0]
test_file_head=test_file_name.split(".")[0]


start = time.time()

'''----------------------------------------------------------------------------------------------'''
'''--------------------------------GBDT模型--------------------------------------------'''
'''----------------------------------------------------------------------------------------------'''

#1. 计算正例 和负例 样本个数 及正例样本占比
cmd = './utils/count.py {tr_file_name} > toutiao_hotel_fc.trva.t10.txt'.format(nr_thread=NR_THREAD,tr_file_name=train_file_name,tr_file_head=train_file_head,te_file_head=test_file_head,te_file_name=test_file_name)
subprocess.call(cmd, shell=True)

#2.1 数字特征和分类特征分离并行化处理 （训练数据）
cmd = 'converters/parallelizer-a.py -s {nr_thread} converters/pre-a.py {tr_file_name} {tr_file_head}.gbdt.dense {tr_file_head}.gbdt.sparse'.format(nr_thread=NR_THREAD,tr_file_name=train_file_name,tr_file_head=train_file_head,te_file_head=test_file_head,te_file_name=test_file_name)
subprocess.call(cmd, shell=True) 

#2.2 数字特征和分类 特征分离及并行化处理 （测试数据）
cmd = 'converters/parallelizer-a.py -s {nr_thread} converters/pre-a.py {te_file_name} {te_file_head}.gbdt.dense {te_file_head}.gbdt.sparse'.format(nr_thread=NR_THREAD,tr_file_name=train_file_name,tr_file_head=train_file_head,te_file_head=test_file_head,te_file_name=test_file_name)
subprocess.call(cmd, shell=True) 


#3. gbdt模型 生成新的数据特征
cmd = './gbdt -t 30 -s {nr_thread} {te_file_head}.gbdt.dense {te_file_head}.gbdt.sparse {tr_file_head}.gbdt.dense {tr_file_head}.gbdt.sparse {te_file_head}.gbdt.out {tr_file_head}.gbdt.out'.format(nr_thread=NR_THREAD,tr_file_name=train_file_name,tr_file_head=train_file_head,te_file_head=test_file_head,te_file_name=test_file_name)
subprocess.call(cmd, shell=True)

'''----------------------------------------------------------------------------------------------'''
'''--------------------------------FFM模型--------------------------------------------'''
'''----------------------------------------------------------------------------------------------'''

# cmd = 'rm -f te.gbdt.dense te.gbdt.sparse tr.gbdt.dense tr.gbdt.sparse'
# subprocess.call(cmd, shell=True)

#5.1 生成ffm的输入特征 (线性模型)
cmd = 'converters/parallelizer-b.py -s {nr_thread} converters/pre-b.py {tr_file_name} {tr_file_head}.gbdt.out {tr_file_head}.ffm'.format(nr_thread=NR_THREAD,tr_file_name=train_file_name,tr_file_head=train_file_head,te_file_head=test_file_head,te_file_name=test_file_name)
subprocess.call(cmd, shell=True) 

#5.2 测试数据并行预处理 （线性模型）
cmd = 'converters/parallelizer-b.py -s {nr_thread} converters/pre-b.py {te_file_name} {te_file_head}.gbdt.out {te_file_head}.ffm'.format(nr_thread=NR_THREAD,tr_file_name=train_file_name,tr_file_head=train_file_head,te_file_head=test_file_head,te_file_name=test_file_name)
subprocess.call(cmd, shell=True) 

# cmd = 'rm -f te.gbdt.out tr.gbdt.out'
# subprocess.call(cmd, shell=True)

#6. 训练线性分类器(??输入数据的类型)
cmd = './ffm-train -k 5 -t 18 -s {nr_thread} -p {te_file_head}.ffm {tr_file_head}.ffm model'.format(nr_thread=NR_THREAD,tr_file_name=train_file_name,tr_file_head=train_file_head,te_file_head=test_file_head,te_file_name=test_file_name)
subprocess.call(cmd, shell=True)

#7. 线性模型预测
cmd = './ffm-predict {te_file_head}.ffm model {te_file_head}.out'.format(nr_thread=NR_THREAD,tr_file_name=train_file_name,tr_file_head=train_file_head,te_file_head=test_file_head,te_file_name=test_file_name)
subprocess.call(cmd, shell=True)

#8. 预测结果指标计算
cmd = './utils/calibrate.py {te_file_head}.out {te_file_head}.out.cal'.format(nr_thread=NR_THREAD,tr_file_name=train_file_name,tr_file_head=train_file_head,te_file_head=test_file_head,te_file_name=test_file_name)
subprocess.call(cmd, shell=True)

#9. 生成提交数据
cmd = './utils/make_submission.py {te_file_head}.out.cal {te_file_head}submission.csv'.format(nr_thread=NR_THREAD,tr_file_name=train_file_name,tr_file_head=train_file_head,te_file_head=test_file_head,te_file_name=test_file_name)
#subprocess.call(cmd, shell=True)

print('time used = {0:.0f}'.format(time.time()-start))
