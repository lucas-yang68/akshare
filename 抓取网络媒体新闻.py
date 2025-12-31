# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 12:59:41 2025

@author: Administrator
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time  # 可选，用于延时避免频繁请求

def 抓取华尔街见闻红色焦点信息()-> pd.DataFrame:
    '''
    url:https://wallstreetcn.com/live/global 
    页面里的 红色项，每条保留时间，内容，链接
    '''
    # URL
    url = 'https://wallstreetcn.com/live/global'
    
    # 请求头，模拟浏览器避免反爬
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # 发送请求
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        response.encoding = 'utf-8'  # 处理中文编码
        
        # 解析 HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找直播项容器（根据页面常见结构调整，如 'live-item' 或 'message-item'）
        items = soup.find_all('div', class_='live-item') or soup.find_all('li', class_='live-message') or soup.find_all('div', {'data-role': 'live-item'})
        
        results = []
        
        for item in items:
            # 提取时间（常见类名：time, timestamp, date）
            time_elem = item.find('span', class_='time') or item.find('div', class_='time') or item.find('time')
            time_text = time_elem.text.strip() if time_elem else '未知时间'
            
            # 提取内容（优先红色元素）
            content_elem = None
            # 优先找红色：style含 red 或类含 red/important
            red_candidates = item.find_all(style=lambda s: s and 'red' in s.lower()) or item.find_all(class_=lambda c: c and ('red' in str(c).lower() or 'important' in str(c).lower()))
            if red_candidates:
                content_elem = red_candidates[0].find('a') or red_candidates[0]
            else:
                # 否则取整个项的内容链接
                content_elem = item.find('a') or item.find('span', class_='content') or item
            
            content_text = content_elem.text.strip() if content_elem else '无内容'
            
            # 提取链接
            link = ''
            if content_elem and hasattr(content_elem, 'get'):
                link = content_elem.get('href', '')
                if link and not link.startswith('http'):
                    link = 'https://wallstreetcn.com' + link  # 补全相对链接
            
            # 只保留有内容的条目
            if content_text:
                results.append({
                    '时间': time_text,
                    '内容': content_text,
                    '链接': link
                })
        
        # 输出结果（取前 10 条，避免太多）
        print(f"抓取到 {len(results)} 条红色/直播项：")
        for idx, res in enumerate(results[:10], 1):
            print(f"{idx}. 时间: {res['时间']}")
            print(f"    内容: {res['内容']}")
            print(f"    链接: {res['链接']}\n")
        # 3. 转为 DataFrame
        df = pd.DataFrame(results, columns=["时间", "内容", "链接"])

        # 按时间降序（最新的在最上面）
        if not df.empty and "时间" in df.columns:
        # 把时间转成可排序的格式（如果有年月日的话更好，这里简单处理）
          df = df.sort_values(by="时间", ascending=False).reset_index(drop=True)
        return df      
        # 可选：保存到文件
        # import json
        # with open('live_red_items.json', 'w', encoding='utf-8') as f:
        #     json.dump(results, f, ensure_ascii=False, indent=2)
    except requests.exceptions.RequestException as e:
        print(f"请求失败：{e}")
        print("建议：检查网络，或用 VPN；如果 JS 渲染，用 Selenium。")
    except Exception as e:
        print(f"解析错误：{e}")
        print("页面结构可能变化，检查源代码调整选择器。")
        
        
# ================== 执行 ==================
if __name__ == "__main__":
    print("正在抓取华尔街见闻全球直播红色重要资讯...")
    df = 抓取华尔街见闻红色焦点信息()

    if df.empty:
        print("没有抓到红色条目，可能是页面结构变化或网络问题")
    else:
        print(f"成功抓取 {len(df)} 条红色资讯")
        print(df.head(10))        # 显示前10条
        print("\n完整 DataFrame 已生成，可直接使用 df 变量")