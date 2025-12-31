# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 21:06:47 2025

@author: Administrator
"""

# app.py
from flask import Flask, render_template, request, jsonify
import os
from train_model import train_and_predict
from datetime import datetime

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/ai')
def ai():
    return render_template('AI.txt')
@app.route('/shares')
def share1():
    return render_template('shares.txt')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        code = data['code'].upper()
        start = data['start']
        end = data['end']
        eval_date = data['eval_date']

        for d in [start, end, eval_date]:
            datetime.strptime(d, '%Y-%m-%d')

        result = train_and_predict(code, start, end, eval_date)
        
        return jsonify({
            'success': True,
            'probability': round(result['prob_up'] * 100, 2),
            'message': f"{eval_date} 后上涨概率: {result['prob_up']:.1%}",
            'model_info': {
                'train_size': result['train_size'],
                'test_size': result['test_size'],
                'accuracy': round(result['accuracy'] * 100, 2),
                'auc': round(result['auc'], 3)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/test_json', methods=['GET'])
def test_json():
    # 模拟 numpy float32
    import numpy as np
    prob = np.float32(0.724)
    prob = float(prob)  # 修复
    
    return jsonify({
        'prob': prob,
        'percent': float(prob * 100)
    })

if __name__ == '__main__':
    os.makedirs('data_cache', exist_ok=True)
    os.makedirs('models_cache', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)