# !/usr/bin/env python
"""
Date: 2025/3/10 18:00
Desc: 通用帮助函数
"""

import math
from typing import List, Dict

import pandas as pd
import requests
import time
from akshare.utils.tqdm import get_tqdm

headers = {
    # 用户代理标识 - 使用最新版 Chrome
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    
    # 接受的内容类型
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    
    # 接受的语言
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    
    # 接受的编码
    'Accept-Encoding': 'gzip, deflate, br',
    
    # 缓存控制
    'Cache-Control': 'no-cache',
    
    # 连接类型
    'Connection': 'keep-alive',
    
    # 安全请求相关头部
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    
    # 升级不安全请求
    'Upgrade-Insecure-Requests': '1',
    
    # 页面模式 (IE 兼容性设置)
    'Sec-Ch-Ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"'
}
def fetch_paginated_data(url: str, base_params: Dict, timeout: int = 15):
    """
    东方财富-分页获取数据并合并结果
    https://quote.eastmoney.com/f1.html?newcode=0.000001
    :param url: 股票代码
    :type url: str
    :param base_params: 基础请求参数
    :type base_params: dict
    :param timeout: 请求超时时间
    :type timeout: str
    :return: 合并后的数据
    :rtype: pandas.DataFrame
    """
    # 复制参数以避免修改原始参数
    params = base_params.copy()
    # 获取第一页数据，用于确定分页信息
    r = requests.get(url, params=params, headers = headers, timeout = timeout)
    data_json = r.json()
    # 计算分页信息
    per_page_num = len(data_json["data"]["diff"])
    total_page = math.ceil(data_json["data"]["total"] / per_page_num)
    # 存储所有页面数据
    temp_list = []
    # 添加第一页数据
    temp_list.append(pd.DataFrame(data_json["data"]["diff"]))
    # 获取进度条
    tqdm = get_tqdm()
    # 获取剩余页面数据
    for page in tqdm(range(2, total_page + 1), leave=False):
        params.update({"pn": page})
        r = requests.get(url, params=params, headers = headers, timeout = timeout)
        data_json = r.json()
        inner_temp_df = pd.DataFrame(data_json["data"]["diff"])
        temp_list.append(inner_temp_df)
    # 合并所有数据
    temp_df = pd.concat(temp_list, ignore_index=True)
    temp_df["f3"] = pd.to_numeric(temp_df["f3"], errors="coerce")
    temp_df.sort_values(by=["f3"], ascending=False, inplace=True, ignore_index=True)
    temp_df.reset_index(inplace=True)
    temp_df["index"] = temp_df["index"].astype(int) + 1
    return temp_df


def set_df_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    设置 pandas.DataFrame 为空的情况
    :param df: 需要设置命名的数据框
    :type df: pandas.DataFrame
    :param cols: 字段的列表
    :type cols: list
    :return: 重新设置后的数据
    :rtype: pandas.DataFrame
    """
    if df.shape == (0, 0):
        return pd.DataFrame(data=[], columns=cols)
    else:
        df.columns = cols
        return df
