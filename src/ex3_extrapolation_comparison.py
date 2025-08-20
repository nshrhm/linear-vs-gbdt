#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機械学習モデルの比較実験 - 外挿（Extrapolation）タスクでの性能比較
線形モデルとGDBTモデルの外挿性能の根本的な違いを評価
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings

# 特定の警告を抑制
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='LightGBM binary classifier with TreeExplainer')

# scikit-learnからモデルとメトリクスをインポート
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# GDBTモデルをインポート
import xgboost as xgb
import lightgbm as lgb

def main():
    print("=== 機械学習モデル比較実験 (外挿タスク) ===")
    print("5回実行の中央値による統計的に信頼性の高い評価を実施します。")
    
    # 実験設定
    n_experiments = 5
    all_results = []
    all_predictions = {}
    
    # ===================================================================
    # 複数回実験の実行
    # ===================================================================
    for experiment_id in range(n_experiments):
        print(f"\n{'='*20} 実験 {experiment_id + 1}/{n_experiments} {'='*20}")
        
        # ===================================================================
        # 1. 人工データの生成 (Extrapolation-Task Data)
        # ===================================================================
        print(f"\n1. 外挿タスク用データの生成中... (実験 {experiment_id + 1})")
        print("単純な線形データを生成し、訓練範囲外での予測性能を評価")
        
        # 実験ごとに異なるランダムシードを使用
        current_seed = 42 + experiment_id
        np.random.seed(current_seed)
        
        # 単一特徴量の線形データを作成
        X = np.linspace(0, 100, 500).reshape(-1, 1)
        y = 2 * X.flatten() + 30 + np.random.normal(0, 10, 500)
        
        # データを訓練用（内挿範囲）とテスト用（外挿範囲）に分割
        train_mask = X.flatten() <= 50
        test_mask = X.flatten() > 50
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # LightGBMの警告を回避するため、DataFrameに変換
        feature_names = ['feature_0']
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        X_df = pd.DataFrame(X, columns=feature_names)
        
        print(f"データ生成完了。")
        print(f"訓練データ範囲: X <= 50, テストデータ範囲: X > 50")
        print(f"訓練データ: {X_train.shape}, テストデータ: {X_test.shape}")
        
        # ===================================================================
        # 2. モデルの定義
        # ===================================================================
        print("\n2. モデルの定義...")
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=current_seed),
            "Lasso": Lasso(random_state=current_seed),
            "XGBoost": xgb.XGBRegressor(random_state=current_seed),
            "LightGBM": lgb.LGBMRegressor(random_state=current_seed, verbosity=-1)
        }
        
        # ===================================================================
        # 3. 実験の実行と評価
        # ===================================================================
        print("\n3. 実験の実行と評価...")
        experiment_results = []
        predictions = {}  # 可視化用の予測結果を保存
        
        for name, model in models.items():
            print(f"--- {name} の評価を開始 ---")
            
            # --- 学習時間の計測 ---
            start_train_time = time.time()
            if name in ["Linear Regression", "Ridge", "Lasso"]:
                # 線形モデルはnumpy配列を使用
                model.fit(X_train, y_train)
            else:
                # GDBTモデルはDataFrameを使用
                model.fit(X_train_df, y_train)
            train_time = time.time() - start_train_time
            
            # --- 推論時間の計測 ---
            start_pred_time = time.time()
            if name in ["Linear Regression", "Ridge", "Lasso"]:
                y_pred = model.predict(X_test)
                predictions[name] = model.predict(X)  # 全範囲での予測
            else:
                y_pred = model.predict(X_test_df)
                predictions[name] = model.predict(X_df)  # 全範囲での予測
            pred_time = time.time() - start_pred_time
            
            # --- 予測精度の評価 ---
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)  # GDBTでは大きな負の値になることが予想される
            
            experiment_results.append({
                "Model": name,
                "Train Time (s)": train_time,
                "Inference Time (s)": pred_time,
                "RMSE": rmse,
                "MAE": mae,
                "R2 Score": r2,
            })
            print(f"--- {name} の評価が完了 ---")
        
        # 実験結果を全体の結果に追加
        all_results.extend(experiment_results)
        if experiment_id == 0:  # 最初の実験の予測結果のみ可視化用に保存
            all_predictions = predictions
    
    # ===================================================================
    # 4. 統計的評価（5回実験の中央値計算）
    # ===================================================================
    print(f"\n4. 統計的評価（{n_experiments}回実験の中央値計算）...")
    
    # 全実験結果をDataFrameに変換
    all_results_df = pd.DataFrame(all_results)
    
    # モデルごとに中央値を計算
    median_results = []
    model_names = all_results_df['Model'].unique()
    
    for model_name in model_names:
        model_data = all_results_df[all_results_df['Model'] == model_name]
        median_result = {
            "Model": model_name,
            "Train Time (s)": model_data["Train Time (s)"].median(),
            "Inference Time (s)": model_data["Inference Time (s)"].median(),
            "RMSE": model_data["RMSE"].median(),
            "MAE": model_data["MAE"].median(),
            "R2 Score": model_data["R2 Score"].median(),
        }
        median_results.append(median_result)
        
        print(f"{model_name}: RMSE中央値={median_result['RMSE']:.4f}, R2中央値={median_result['R2 Score']:.4f}")
    
    # ===================================================================
    # 5. 結果の保存と可視化
    # ===================================================================
    print("\n5. 結果の保存と可視化...")
    
    # 中央値結果をDataFrameに変換
    results_df = pd.DataFrame(median_results).sort_values(by="RMSE").reset_index(drop=True)
    
    # CSVファイルとして保存（中央値結果）
    results_df.to_csv("ex3_extrapolation_results.csv", index=False, encoding='utf-8')
    print("中央値結果をex3_extrapolation_results.csvに保存しました。")
    
    # 全実験結果も保存
    all_results_df.to_csv("ex3_extrapolation_all_results.csv", index=False, encoding='utf-8')
    print("全実験結果をex3_extrapolation_all_results.csvに保存しました。")
    
    # 結果の表示
    print("\n--- 中央値実験結果 (外挿性能) ---")
    print(results_df.to_string(index=False))
    
    # --- 結果の可視化 ---
    plt.style.use('default')  # seaborn-v0_8-whitegridが利用できない場合があるため
    plt.figure(figsize=(12, 8))
    
    # 元データのプロット
    plt.scatter(X_train, y_train, label='Training Data', color='blue', alpha=0.6, s=50)
    plt.scatter(X_test, y_test, label='Extrapolation Data', color='green', alpha=0.6, s=50)
    
    # 各モデルの予測結果をプロット
    # 線形モデルの代表としてLinear Regressionをプロット
    plt.plot(X, all_predictions['Linear Regression'], color='red', linestyle='-', linewidth=2, 
             label='Linear Regression Prediction')
    # GDBTモデルの代表としてLightGBMをプロット
    plt.plot(X, all_predictions['LightGBM'], color='purple', linestyle='--', linewidth=2, 
             label='LightGBM Prediction')
    
    # 訓練データの最大値を線で示す
    max_train_y = y_train.max()
    plt.axhline(y=max_train_y, color='orange', linestyle=':', linewidth=2, 
                label=f'Max value in Training Data ({max_train_y:.2f})')
    plt.axvline(x=50, color='gray', linestyle=':', linewidth=1.5, label='Train/Test Boundary')
    
    plt.title('Model Behavior Comparison on Extrapolation Task', fontsize=18)
    plt.xlabel('Feature (X)', fontsize=14)
    plt.ylabel('Target (y)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 画像を保存
    plt.savefig("ex3_extrapolation_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig("ex3_extrapolation_comparison.pdf", bbox_inches='tight')
    print("グラフをex3_extrapolation_comparison.png/.pdfに保存しました。")
    
    plt.show()
    
    # 性能比較の可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Extrapolation Performance Comparison', fontsize=16)
    
    # RMSE比較
    sns.barplot(x='RMSE', y='Model', data=results_df.sort_values('RMSE', ascending=True), 
                ax=axes[0], palette='Reds', hue='Model', legend=False)
    axes[0].set_title('RMSE (Lower is Better)')
    
    # MAE比較
    sns.barplot(x='MAE', y='Model', data=results_df.sort_values('MAE', ascending=True), 
                ax=axes[1], palette='Oranges', hue='Model', legend=False)
    axes[1].set_title('MAE (Lower is Better)')
    
    # R2 Score比較
    sns.barplot(x='R2 Score', y='Model', data=results_df.sort_values('R2 Score', ascending=False), 
                ax=axes[2], palette='Blues', hue='Model', legend=False)
    axes[2].set_title('R2 Score (Higher is Better)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 性能比較グラフを保存
    plt.savefig("ex3_performance_metrics.png", dpi=300, bbox_inches='tight')
    plt.savefig("ex3_performance_metrics.pdf", bbox_inches='tight')
    print("性能比較グラフをex3_performance_metrics.png/.pdfに保存しました。")
    
    plt.show()
    
    # ===================================================================
    # 6. 考察の保存
    # ===================================================================
    print("\n6. 考察の保存...")
    
    consideration = """
--- 考察 ---
外挿タスクでは、モデルの根本的な特性が明確に現れます。

1. 線形モデル: 訓練データで捉えた線形トレンドを訓練範囲外にも外挿します。
   そのため、トレンドが継続する限り良好な予測を提供できます。

2. GDBTモデル: 決定木ベースのモデルは、訓練データの最大値（または最小値）を
   超える値を予測することができません。グラフで見られるように、GDBTモデルの
   予測は訓練データの最大値周辺でプラトー（平坦）になります。これにより、
   RMSEやMAEが大幅に悪化し、R2スコアは大きな負の値となります。

結論として、時系列の将来トレンド予測など、訓練データ範囲外の値を予測する
必要があるタスクでは、線形モデルが根本的により適しています。
このようなタスクでGDBTを使用する場合は、事前にトレンド成分を除去するなどの
特徴量エンジニアリングが不可欠です。

実験設定:
- 単純な線形データ: y = 2*X + 30 + noise
- 訓練範囲: X <= 50
- テスト範囲: X > 50 (外挿領域)
- 線形モデルは外挿を適切に処理
- GDBTモデルは訓練データの最大値でプラトーを形成
"""
    
    with open("ex3_extrapolation_consideration.txt", "w", encoding='utf-8') as f:
        f.write(consideration)
    
    print("考察をex3_extrapolation_consideration.txtに保存しました。")
    print("\n=== 実験完了 ===")

if __name__ == "__main__":
    main()
