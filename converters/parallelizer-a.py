#!/usr/bin/env python3

import argparse, sys

from common import *

def parse_args():
    
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='nr_thread', default=12, type=int)
    parser.add_argument('cvt_path')
    parser.add_argument('src_path')
    parser.add_argument('dst1_path')
    parser.add_argument('dst2_path')
    args = vars(parser.parse_args())

    return args

def main():
    
    args = parse_args()

    nr_thread = args['nr_thread']
    
    split(args['src_path'], nr_thread, True)

    # 将输入特征中的数字和分类特征分开存储到不同的文件， （数字特征和分类特征再分割成小文件供并行处理）
    parallel_convert(args['cvt_path'], [args['src_path'], args['dst1_path'], args['dst2_path']], nr_thread)

    # 将分开的小文件合并到一起
    cat(args['dst1_path'], nr_thread)

    cat(args['dst2_path'], nr_thread)

    delete(args['src_path'], nr_thread)

    delete(args['dst1_path'], nr_thread)

    delete(args['dst2_path'], nr_thread)

main()
