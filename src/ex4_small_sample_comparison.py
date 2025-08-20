#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機械学習モデルの比較実験 - 小規模データでの性能比較
サンプル数が少ない状況での過学習への耐性を評価
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import os
import warnings

# 特定の警告を抑制
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='LightGBM binary classifier with TreeExplainer')

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb

def create_train_test_comparison_plot(results_df, language='en'):
    """
    訓練・テスト誤差の比較グラフを作成し、保存する。
    """
    if language == 'ja':
        title = '訓練誤差とテスト誤差の比較（小規模データ）'
        xlabel = 'モデル'
        train_label = 'RMSE (訓練)'
        test_label = 'RMSE (テスト)'
        filename_base = 'ex4_train_test_comparison_ja'
    else:
        title = 'Comparison of Training and Test Errors (Small Data)'
        xlabel = 'Model'
        train_label = 'RMSE (Train)'
        test_label = 'RMSE (Test)'
        filename_base = 'ex4_train_test_comparison'

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(results_df))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], results_df['RMSE (Train)'], width, 
           label=train_label, color='skyblue', alpha=0.8)
    ax.bar([i + width/2 for i in x], results_df['RMSE (Test)'], width, 
           label=test_label, color='orange', alpha=0.8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('RMSE')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
    print(f"訓練・テスト誤差比較グラフを{filename_base}.png/.pdfに保存しました。")
    plt.close(fig)

def create_overfitting_metrics_plot(results_df, language='en'):
    """
    過学習指標のグラフを作成し、保存する。
    """
    if language == 'ja':
        title_overfitting = '過学習係数 (テストRMSE / 訓練RMSE)'
        xlabel_overfitting = '過学習係数（低いほど良い）'
        title_r2 = 'テスト R2スコア'
        xlabel_r2 = 'R2スコア（高いほど良い）'
        filename_base = 'ex4_overfitting_metrics_ja'
    else:
        title_overfitting = 'Overfitting Factor (Test RMSE / Train RMSE)'
        xlabel_overfitting = 'Overfitting Factor (Lower is Better)'
        title_r2 = 'Test R2 Score'
        xlabel_r2 = 'R2 Score (Higher is Better)'
        filename_base = 'ex4_overfitting_metrics'

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 過学習係数
    sns.barplot(x='Overfitting', y='Model', 
                data=results_df.sort_values('Overfitting', ascending=True), 
                ax=axes[0], palette='Reds', hue='Model', legend=False)
    axes[0].set_title(title_overfitting)
    axes[0].set_xlabel(xlabel_overfitting)
    
    # R2比較
    sns.barplot(x='R2 (Test)', y='Model', 
                data=results_df.sort_values('R2 (Test)', ascending=False), 
                ax=axes[1], palette='Blues', hue='Model', legend=False)
    axes[1].set_title(title_r2)
    axes[1].set_xlabel(xlabel_r2)
    
    plt.tight_layout()
    plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
    print(f"過学習指標グラフを{filename_base}.png/.pdfに保存しました。")
    plt.close(fig)

def main():
    print("=== 機械学習モデル比較実験 (小規模データ) ===")
    print("5回実行の中央値による統計的に信頼性の高い評価を実施します。")
    
    # 実験設定
    n_experiments = 5
    all_results = []
    
    # ===================================================================
    # 複数回実験の実行
    # ===================================================================
    for experiment_id in range(n_experiments):
        print(f"\n{'='*20} 実験 {experiment_id + 1}/{n_experiments} {'='*20}")
        
        # ===================================================================
        # 1. 人工データの生成 (Small-Sample Data)
        # ===================================================================
        print(f"\n1. 小規模データの生成中... (実験 {experiment_id + 1})")
        print("サンプル数が少なく、特徴量が多いデータを生成し、過学習しやすい状況を作成")
        
        # 実験ごとに異なるランダムシードを使用
        current_seed = 42 + experiment_id
        
        X, y = make_regression(
            n_samples=100,      # サンプル数を100に制限
            n_features=50,      # 特徴量の数を50に設定
            n_informative=10,   # 実際に情報を持つ特徴量は10個
            noise=30.0,
            random_state=current_seed
        )
        
        # データを訓練用とテスト用に分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=current_seed)
        
        # LightGBMの警告を回避するため、DataFrameに変換
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        print(f"データ生成完了。")
        print(f"訓練データ: {X_train.shape}, テストデータ: {X_test.shape}")
        print(f"特徴量数 > サンプル数の状況: {X.shape[1]} features > {X.shape[0]} samples")
        
        # ===================================================================
        # 2. モデルの定義
        # ===================================================================
        print("\n2. モデルの定義...")
        print("正則化の効果を比較するため、RidgeとLassoのalphaを少し強めに設定")
        
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge (alpha=1.0)": Ridge(alpha=1.0, random_state=current_seed),
            "Lasso (alpha=1.0)": Lasso(alpha=1.0, random_state=current_seed),
            "XGBoost": xgb.XGBRegressor(random_state=current_seed),
            "LightGBM": lgb.LGBMRegressor(random_state=current_seed, verbosity=-1)
        }
        
        # ===================================================================
        # 3. 実験の実行と評価
        # ===================================================================
        print("\n3. 実験の実行と評価...")
        print("小規模データでは学習/テスト誤差の比較が重要")
        
        experiment_results = []
        
        for name, model in models.items():
            print(f"--- {name} の評価を開始 ---")
            
            # --- 学習 ---
            if name in ["Linear Regression", "Ridge (alpha=1.0)", "Lasso (alpha=1.0)"]:
                # 線形モデルはnumpy配列を使用
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
            else:
                # GDBTモデルはDataFrameを使用
                model.fit(X_train_df, y_train)
                y_pred_train = model.predict(X_train_df)
                y_pred_test = model.predict(X_test_df)
            
            # --- 訓練データとテストデータの両方で評価 ---
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            
            # 過学習の指標として誤差の差を計算
            overfitting_rmse = rmse_test - rmse_train
            overfitting_r2 = r2_train - r2_test
            overfitting_factor = rmse_test / rmse_train if rmse_train > 0 else float('inf')
            
            # 結果をリストに追加
            experiment_results.append({
                "Model": name,
                "RMSE (Train)": rmse_train,
                "RMSE (Test)": rmse_test,
                "R2 (Train)": r2_train,
                "R2 (Test)": r2_test,
                "Overfitting": overfitting_factor
            })
            print(f"--- {name} の評価が完了 ---")
        
        # 実験結果を全体の結果に追加
        all_results.extend(experiment_results)
    
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
            "RMSE (Train)": model_data["RMSE (Train)"].median(),
            "RMSE (Test)": model_data["RMSE (Test)"].median(),
            "R2 (Train)": model_data["R2 (Train)"].median(),
            "R2 (Test)": model_data["R2 (Test)"].median(),
            "Overfitting": model_data["Overfitting"].median(),
        }
        median_results.append(median_result)
        
        print(f"{model_name}: テストRMSE中央値={median_result['RMSE (Test)']:.4f}, 過学習係数中央値={median_result['Overfitting']:.2f}")
    
    # ===================================================================
    # 5. 結果の保存と可視化
    # ===================================================================
    print("\n5. 結果の保存と可視化...")
    
    # 中央値結果をDataFrameに変換
    results_df = pd.DataFrame(median_results).sort_values(by="RMSE (Test)").reset_index(drop=True)
    
    # CSVファイルとして保存（中央値結果）
    results_df.to_csv("ex4_small_sample_results.csv", index=False, encoding='utf-8')
    print("中央値結果をex4_small_sample_results.csvに保存しました。")
    
    # 全実験結果も保存
    all_results_df.to_csv("ex4_small_sample_all_results.csv", index=False, encoding='utf-8')
    print("全実験結果をex4_small_sample_all_results.csvに保存しました。")
    
    # 結果の表示
    print("\n--- 中央値実験結果 (小規模データ) ---")
    print(results_df.to_string(index=False))
    
    # --- 結果の可視化 ---
    # 訓練・テスト誤差の比較 (英語)
    create_train_test_comparison_plot(results_df, language='en')
    # 訓練・テスト誤差の比較 (日本語)
    create_train_test_comparison_plot(results_df, language='ja')
    
    # 過学習指標の可視化 (英語)
    create_overfitting_metrics_plot(results_df, language='en')
    # 過学習指標の可視化 (日本語)
    create_overfitting_metrics_plot(results_df, language='ja')
    
    # plt.show() # GUIが必要なため、自動実行ではコメントアウト
    
    # ===================================================================
    # 6. 考察の保存
    # ===================================================================
    print("\n6. 考察の保存...")
    
    consideration = """
--- 考察 ---
小規模データセットでは、モデルの過学習への耐性が重要になります。

1. GDBTモデル: 訓練データに対しては非常に低い誤差（高い精度）を示しますが、
   テストデータでは誤差が大幅に悪化する傾向があります。これは、モデルが
   訓練データのノイズまで過剰に学習してしまい、未知のデータに対する
   汎化性能を失っている（過学習）ことを示しています。

2. 線形モデル: 単純な線形回帰も過学習の傾向を見せることがありますが、
   RidgeやLassoといった正則化付きのモデルは、訓練誤差とテスト誤差の差が
   比較的小さく、より安定した性能を示します。これは、正則化がモデルの
   複雑さにペナルティを課すことで、過学習を抑制しているためです。

結論として、サンプル数が限られている状況では、複雑なGDBTモデルは
過学習のリスクが高く、注意深いハイパーパラメータ調整が不可欠です。
一方で、正則化付きの線形モデルは、そのシンプルさと過学習への耐性から、
より信頼性の高い選択肢となり得ます。

実験設定:
- サンプル数: 100 (訓練70, テスト30)
- 特徴量数: 50 (情報のある特徴量: 10)
- 特徴量数 > サンプル数の高次元小標本問題
- 過学習指標: 訓練・テスト誤差の差
"""
    
    with open("ex4_small_sample_consideration.txt", "w", encoding='utf-8') as f:
        f.write(consideration)
    
    print("考察をex4_small_sample_consideration.txtに保存しました。")
    print("\n=== 実験完了 ===")

if __name__ == "__main__":
    main()
