#!/usr/bin/env python3

import argparse, csv, sys

from converters.common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('dense_path', type=str)
parser.add_argument('sparse_path', type=str)
args = vars(parser.parse_args())

#These features are dense enough (they appear in the dataset more than 4 million times), so we include them in GBDT
target_cat_feats = ['C9-a73ee510', 'C22-', 'C17-e5ba7672', 'C26-', 'C23-32c7478e', 'C6-7e0ccccf', 'C14-b28479f6', 'C19-21ddcdc9', 'C14-07d13a8f', 'C10-3b08e48b', 'C6-fbad5c96', 'C23-3a171ecb', 'C20-b1252a9d', 'C20-5840adea', 'C6-fe6b92e5', 'C20-a458ea53', 'C14-1adce6ef', 'C25-001f3601', 'C22-ad3062eb', 'C17-07c540c4', 'C6-', 'C23-423fab69', 'C17-d4bb7bd8', 'C2-38a947a1', 'C25-e8b83407', 'C9-7cc72ec2']


target_cat_feats = ['C2-[OPPO A33m]','C2-[HUAWEI MLA-AL10]','C2-[vivo Y51A]','C3-[武汉]','C3-[西安]','C2-[HUAWEI NXT-AL10]','C2-[vivo X9]','C2-[vivo Y51]','C2-[vivo V3Max A]','C2-[A31]','C2-[vivo Y67]','C3-[石家庄]','C3-[天津]','C2-[OPPO A59m]','C2-[HM NOTE 1LTE]','C2-[GN5001S]','C3-[苏州]','C2-[2014813]','C3-[东莞]','C3-[深圳]','C2-[OPPO R9s]','C2-[HUAWEI TAG-AL00]','C2-[OPPO A57]','C3-[广州]','C2-[OPPO R9m]','C3-[成都]','C2-[HM NOTE 1S]','C2-[OPPO A59s]','C3-[重庆]','C2-[vivo Y66]','C3-[上海]','C2-[HM 2A]','C2-[OPPO A33]','C2-[Redmi Note 2]','C3-[北京]','C2-[OPPO A37m]','C2-[Redmi Note 3]']


with open(args['dense_path'], 'w') as f_d, open(args['sparse_path'], 'w') as f_s:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        for j in range(1, 29):
            val = row['I{0}'.format(j)]
            if val == '':
                val = -10 
            feats.append('{0}'.format(val))
        f_d.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
        
        cat_feats = set()
        for j in range(1, 3):
            field = 'C{0}'.format(j)
            key = field + '-' + row[field]
            cat_feats.add(key)

        feats = []
        for j, feat in enumerate(target_cat_feats, start=1):
            if feat in cat_feats:
                feats.append(str(j))
        f_s.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
