#!/usr/bin/env python3

import subprocess, sys, os, time

NR_THREAD = 1

start = time.time()

'''----------------------------------------------------------------------------------------------'''
'''--------------------------------GBDT模型--------------------------------------------'''
'''----------------------------------------------------------------------------------------------'''

#1. 计算正例 和负例 样本个数 及正例样本占比
cmd = './utils/count.py toutiao_hotel_tr.csv > toutiao_hotel_fc.trva.t10.txt'
subprocess.call(cmd, shell=True)

#2.1 数字特征和分类特征分离并行化处理 （训练数据）
cmd = 'converters/parallelizer-a.py -s {nr_thread} converters/pre-a.py toutiao_hotel_tr.csv toutiao_hotel_tr.gbdt.dense toutiao_hotel_tr.gbdt.sparse'.format(nr_thread=NR_THREAD)
subprocess.call(cmd, shell=True) 

#2.2 数字特征和分类 特征分离及并行化处理 （测试数据）
cmd = 'converters/parallelizer-a.py -s {nr_thread} converters/pre-a.py toutiao_hotel_te.csv toutiao_hotel_te.gbdt.dense toutiao_hotel_te.gbdt.sparse'.format(nr_thread=NR_THREAD)
subprocess.call(cmd, shell=True) 


#3. gbdt模型 生成新的数据特征
cmd = './gbdt -t 30 -s {nr_thread} toutiao_hotel_te.gbdt.dense toutiao_hotel_te.gbdt.sparse toutiao_hotel_tr.gbdt.dense toutiao_hotel_tr.gbdt.sparse toutiao_hotel_te.gbdt.out toutiao_hotel_tr.gbdt.out'.format(nr_thread=NR_THREAD)
subprocess.call(cmd, shell=True)

'''----------------------------------------------------------------------------------------------'''
'''--------------------------------LR模型--------------------------------------------'''
'''----------------------------------------------------------------------------------------------'''

# cmd = 'rm -f te.gbdt.dense te.gbdt.sparse tr.gbdt.dense tr.gbdt.sparse'
# subprocess.call(cmd, shell=True)

#5.1 生成ffm的输入特征 (线性模型)
cmd = 'converters/parallelizer-b.py -s {nr_thread} converters/pre-b.py toutiao_hotel_tr.csv toutiao_hotel_tr.gbdt.out toutiao_hotel_tr.ffm'.format(nr_thread=NR_THREAD)
subprocess.call(cmd, shell=True) 

#5.2 测试数据并行预处理 （线性模型）
cmd = 'converters/parallelizer-b.py -s {nr_thread} converters/pre-b.py toutiao_hotel_te.csv toutiao_hotel_te.gbdt.out toutiao_hotel_te.ffm'.format(nr_thread=NR_THREAD)
subprocess.call(cmd, shell=True) 

# cmd = 'rm -f te.gbdt.out tr.gbdt.out'
# subprocess.call(cmd, shell=True)

#6. 训练线性分类器(??输入数据的类型)
cmd = './ffm-train -k 4 -t 18 -s {nr_thread} -p toutiao_hotel_te.ffm toutiao_hotel_tr.ffm model'.format(nr_thread=NR_THREAD)
subprocess.call(cmd, shell=True)

#7. 线性模型预测
cmd = './ffm-predict toutiao_hotel_te.ffm model toutiao_hotel_te.out'.format(nr_thread=NR_THREAD)
subprocess.call(cmd, shell=True)

#8. 预测结果指标计算
cmd = './utils/calibrate.py toutiao_hotel_te.out toutiao_hotel_te.out.cal'.format(nr_thread=NR_THREAD)
subprocess.call(cmd, shell=True)

#9. 生成提交数据
cmd = './utils/make_submission.py toutiao_hotel_te.out.cal toutiao_hotel_submission.csv'.format(nr_thread=NR_THREAD)
#subprocess.call(cmd, shell=True)

print('time used = {0:.0f}'.format(time.time()-start))
