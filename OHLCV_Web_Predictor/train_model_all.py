# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 21:10:04 2025

@author: Administrator
"""
# train_model.py
import pandas as pd
import numpy as np
import joblib
import ta  # 技术指标库: pip install ta
import akshare as ak
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import os
from datetime import datetime

DATA_DIR = 'data_cache'
MODEL_DIR = 'models_cache'

def get_stock_data(code, start, end):
    cache_file = f"{DATA_DIR}/{code}_{start}_{end}.csv"
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)

    # 使用 akshare 获取日线
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start.replace('-',''), end_date=end.replace('-',''),adjust='qfq')
    if df.empty:
        raise ValueError("无数据，请检查股票代码或日期")
    
    df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start) & (df['date'] <= end)].copy()
    df.to_csv(cache_file, index=False)
    return df

def compute_features(df):
    df = df.copy()
    df['body'] = abs(df['close'] - df['open'])
    df['upper'] = df['high'] - df[['open','close']].max(axis=1)
    df['lower'] = df[['open','close']].min(axis=1) - df['low']
    df['body_dir'] = np.sign(df['close'] - df['open'])
    df['amplitude'] = (df['high'] - df['low']) / df['open']
    df['close_pos'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0,1e-6)
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vr'] = df['volume'] / df['vol_ma20'].replace(0,1e-6)
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    
    # 放量突破 & 缩量阴跌（布尔 → 转为 0/1）
    df['surge_break'] = ((df['close'] > df['high'].shift(1)) & (df['volume'] > 1.5 * df['vol_ma5'])).astype(int)
    df['shrink_fall'] = ((df['close'] < df['low'].shift(1)) & (df['volume'] < 0.7 * df['vol_ma5'])).astype(int)
    
    # 形态识别
    df['hammer'] = (
        (df['lower'] > 2 * df['body']) & 
        (df['upper'] < 0.3 * df['body']) &
        (df['close_pos'] < 0.4)
    ).astype(int)
    
    df['hanging_man'] = (
        (df['upper'] > 2 * df['body']) & 
        (df['lower'] < 0.3 * df['body']) &
        (df['close_pos'] > 0.6)
    ).astype(int)
    
    df['doji'] = (df['body'] < 0.001 * (df['high'] - df['low'])).astype(int)
    
    # 吞没形态（需前一根K线）
    df['prev_body'] = abs(df['close'].shift(1) - df['open'].shift(1))
    df['engulfing'] = (
        (df['body'] > df['prev_body'].shift(1)) &
        (df['body_dir'] == -df['body_dir'].shift(1)) &
        (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))  # 看涨吞没示例
    ).astype(int)
    
    # 辅助指标：RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # 量价背离（简化版）
    df['price_down'] = df['close'] < df['close'].shift(1)
    df['vol_up'] = df['volume'] > df['volume'].shift(1)
    df['bull_div'] = (df['price_down'] & df['vol_up'] & (df['rsi'] < 30)).astype(int)
    df['bear_div'] = (~df['price_down'] & ~df['vol_up'] & (df['rsi'] > 70)).astype(int)
    
    # 量能系数
    df['energy_coeff'] = df['volume'] * df['body'] / (df['high'] - df['low']).replace(0, 1e-6)
    
    # ================================================================
    # 3. 最终特征列表（18维）
    # ================================================================
    features = [
        'body', 'upper', 'lower', 'body_dir', 'amplitude', 'close_pos',
        'vr', 'surge_break', 'shrink_fall',
        'hammer', 'hanging_man', 'doji', 'engulfing',
        'rsi', 'bull_div', 'bear_div', 'energy_coeff'
    ]
  
    # 补上第18维：量价相关性（滑动2周期）
    df['vol_price_corr'] = df[['open', 'close']].apply(lambda x: x.corr(df['volume'].iloc[-2:]))  # 简化
    # 更稳健写法：
    corr_list = []
    for i in range(len(df)):
        if i >= 1:
            price = [df['open'].iloc[i-1], df['close'].iloc[i]]
            vol = [df['volume'].iloc[i-1], df['volume'].iloc[i]]
            corr = np.corrcoef(price, vol)[0,1] if len(price)==2 else 0
        else:
            corr = 0
        corr_list.append(corr)
    df['vol_price_corr'] = corr_list
    
    # 最终加入第18维
    features.append('vol_price_corr')
 
    return df[features].fillna(0)

def train_and_predict(code, start, end, eval_date):
    df = get_stock_data(code, start, end)
    df = df[df['date'] < eval_date].copy()  # 确保不包含 eval_date
    
    if len(df) < 100:
        raise ValueError("数据不足100条，无法训练")
    
    # === 划分训练/测试（时间顺序）===
    split_idx = int(len(df) * 0.8)
    train_raw = df.iloc[:split_idx].copy()
    test_raw  = df.iloc[split_idx:].copy()
    
    # === 训练集：生成 target 并删除最后一行 ===
    train_raw['target'] = (train_raw['close'].shift(-1) > train_raw['close'] * 1.005).astype(int)
    train_df = train_raw.iloc[:-1].copy()  # 删除最后一行（无未来）
    
    # === 测试集：同样生成 target 并删除最后一行 ===
    test_raw['target'] = (test_raw['close'].shift(-1) > test_raw['close'] * 1.005).astype(int)
    test_df = test_raw.iloc[:-1].copy()    # 必须删最后一行！
    
    # === 特征 + 训练 ===
    X_train = compute_features(train_df)
    y_train = train_df['target']
    X_test  = compute_features(test_df)
    y_test  = test_df['target']  # 现在有定义了！
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train_s, y_train)
    print(f'scaler  {X_train_s}')
    # === 评估 ===
    prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = (prob > 0.6).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, prob) if len(y_test.unique()) > 1 else 0.5
    
    # === 预测 eval_date 前一天 ===
    eval_target_date = pd.to_datetime(eval_date) - pd.Timedelta(days=1)
    eval_row = df[df['date'] == eval_target_date.strftime('%Y-%m-%d')]
    if eval_row.empty:
        raise ValueError(f"评估日期 {eval_date} 前一天无数据")
    
    X_new = compute_features(eval_row)
    X_new_s = scaler.transform(X_new)
    prob_up = float(model.predict_proba(X_new_s)[0, 1])
    
    return {
        'prob_up': prob_up,
        'train_size': len(train_df),
        'test_size': len(test_df),
        'accuracy': round(acc, 4),
        'auc': round(auc, 4)
    }

train_and_predict('300766', start='2024-01-01', end='2025-11-12' ,eval_date='2025-11-13')