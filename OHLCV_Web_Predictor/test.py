# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:18:44 2025

@author: Administrator
"""

import akshare as ak
import pandas as pd

def get_all_stock_info():
    """
    获取沪深京A股所有股票的基本信息
    """
    # 获取基础代码和名称
    base_info = ak.stock_info_a_code_name()
    
    # 获取行业分类信息（这里使用东方财富的行业分类）
    industry_info = ak.stock_board_industry_name_em()
    
    print("A股基础信息:")
    print(f"股票总数: {len(base_info)}")
    print("\n前10只股票:")
    print(base_info.head(10))
    
    print("\n行业分类:")
    print(industry_info.head())
    
    return base_info, industry_info

# 执行获取
# base_df, industry_df = get_all_stock_info()
industry_info = ak.stock_board_industry_name_em()
stock_board_concept_name_em_df = ak.stock_board_concept_name_em()
individual_info = ak.stock_individual_info_em(symbol='002583')

