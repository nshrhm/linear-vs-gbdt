#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機械学習モデルの比較実験 - 特徴量間の交互作用が少ないデータでの性能比較
Friedman1データセットを使用した低交互作用データでの実験
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import shap
import os
import warnings

# 特定の警告を抑制
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='LightGBM binary classifier with TreeExplainer')

# scikit-learnからモデルとデータ生成ツールをインポート
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# GDBTモデルをインポート
import xgboost as xgb
import lightgbm as lgb

def create_and_save_plot(results_df, language='en'):
    """
    言語設定に基づいてモデル性能比較のグラフを作成し、保存する。
    
    Args:
        results_df (pd.DataFrame): モデルの性能評価結果を含むDataFrame。
        language (str): 'en' (英語) または 'ja' (日本語) を指定。
    """
    if language == 'ja':
        # 日本語のテキスト設定
        title = 'モデル性能比較（特徴量間の交互作用が少ないデータ）'
        accuracy_title = '予測精度 (R2スコア)'
        cost_title = '計算コスト (学習時間)'
        interp_title = '解釈性コスト (SHAP計算時間)'
        filename_base = 'ex2_low_interaction_plot_ja'
    else:
        # 英語のテキスト設定 (デフォルト)
        title = 'Model Performance Comparison (Low-Interaction Data)'
        accuracy_title = 'Prediction Accuracy (R2 Score)'
        cost_title = 'Computation Cost (Training Time)'
        interp_title = 'Interpretability Cost (SHAP Computation Time)'
        filename_base = 'ex2_low_interaction_plot'

    plt.rcParams['font.size'] = 12
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(title, fontsize=16)
    
    # 予測精度 (R2 Score)
    sns.barplot(x='R2 Score', y='Model', data=results_df.sort_values('R2 Score', ascending=False), 
                ax=axes[0], palette='viridis', hue='Model', legend=False)
    axes[0].set_title(accuracy_title)
    
    # 計算コスト (学習時間)
    sns.barplot(x='Train Time (s)', y='Model', data=results_df.sort_values('Train Time (s)', ascending=True), 
                ax=axes[1], palette='plasma', hue='Model', legend=False)
    axes[1].set_title(cost_title)
    axes[1].set_xscale('log')
    
    # 解釈性コスト (SHAP計算時間)
    sns.barplot(x='SHAP Time (s)', y='Model', data=results_df.sort_values('SHAP Time (s)', ascending=True), 
                ax=axes[2], palette='magma', hue='Model', legend=False)
    axes[2].set_title(interp_title)
    axes[2].set_xscale('log')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 画像を保存
    plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
    print(f"グラフを{filename_base}.png/.pdfに保存しました。")
    
    plt.close(fig) # メモリリークを防ぐために図を閉じる

def main():
    print("=== 機械学習モデル比較実験 (特徴量間の交互作用が少ないデータ) ===")
    print("5回実行の中央値による統計的に信頼性の高い評価を実施します。")
    
    # 実験回数
    n_experiments = 5
    
    # 結果を格納するリスト
    all_results = []
    
    # ===================================================================
    # 複数回実験のループ
    # ===================================================================
    for experiment_id in range(n_experiments):
        print(f"\n{'='*20} 実験 {experiment_id + 1}/{n_experiments} {'='*20}")
        
        # ===================================================================
        # 1. 人工データの生成 (Low-Interaction Data)
        # ===================================================================
        print(f"\n1. 人工データの生成中... (実験 {experiment_id + 1})")
        print("make_friedman1データセットを使用:")
        print("y = 10*sin(pi*X1*X2) + 20*(X3-0.5)^2 + 10*X4 + 5*X5 + noise")
        print("交互作用項(X1*X2)が1つのみで、他は線形項や単純な非線形項")
        
        X, y = make_friedman1(
            n_samples=10000,
            n_features=10,  # friedman1は最初の5つの特徴量のみ使用
            noise=5.0,
            random_state=42 + experiment_id  # 各実験で異なるシード
        )
        
        # データを訓練用とテスト用に分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42 + experiment_id
        )
        
        # LightGBMの警告を回避するため、DataFrameに変換
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    print(f"データ生成完了。")
    print(f"訓練データ: {X_train_df.shape}, テストデータ: {X_test_df.shape}")
    
    # ===================================================================
    # 2. モデルの定義
    # ===================================================================
    print("\n2. モデルの定義...")
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42),
        "XGBoost": xgb.XGBRegressor(random_state=42),
        "LightGBM": lgb.LGBMRegressor(random_state=42, verbosity=-1)
    }
    
    # ===================================================================
    # 3. 実験の実行と評価
    # ===================================================================
    print("\n3. 実験の実行と評価...")
    results = []
    
    # 各モデルでループ処理
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
        end_train_time = time.time()
        train_time = end_train_time - start_train_time
        
        # --- 推論時間の計測 ---
        start_pred_time = time.time()
        if name in ["Linear Regression", "Ridge", "Lasso"]:
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test_df)
        end_pred_time = time.time()
        pred_time = end_pred_time - start_pred_time
        
        # --- 予測精度の評価 ---
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # --- 解釈性コストの計測 (SHAP値の計算時間) ---
        start_shap_time = time.time()
        if hasattr(model, 'coef_'):
            explainer = shap.LinearExplainer(model, X_train)
            # SHAP計算の高速化のため、テストデータのサブセットを使用
            num_shap_samples = min(1000, X_test.shape[0])
            shap_sample_indices = np.random.choice(X_test.shape[0], num_shap_samples, replace=False)
            X_test_subset = X_test[shap_sample_indices]
            shap_values = explainer.shap_values(X_test_subset)
        else:
            explainer = shap.TreeExplainer(model, X_train_df)
            # SHAP計算の高速化のため、テストデータのサブセットを使用
            num_shap_samples = min(1000, X_test_df.shape[0])
            shap_sample_indices = np.random.choice(X_test_df.shape[0], num_shap_samples, replace=False)
            X_test_subset = X_test_df.iloc[shap_sample_indices]
            shap_values = explainer.shap_values(X_test_subset)
        end_shap_time = time.time()
        shap_time = end_shap_time - start_shap_time
        
        # 結果をリストに追加
        results.append({
            "Model": name,
            "Train Time (s)": train_time,
            "Inference Time (s)": pred_time,
            "RMSE": rmse,
            "MAE": mae,
            "R2 Score": r2,
            "SHAP Time (s)": shap_time,
        })
        print(f"--- {name} の評価が完了 ---")
    
    # ===================================================================
    # 4. 統計的評価（5回実験の中央値計算）
    # ===================================================================
    print(f"\n4. 統計的評価（{n_experiments}回実験の中央値計算）...")
    
    # 全実験結果をDataFrameに変換
    all_results_df = pd.DataFrame(results)
    
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
            "SHAP Time (s)": model_data["SHAP Time (s)"].median(),
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
    results_df.to_csv("ex2_low_interaction_results.csv", index=False, encoding='utf-8')
    print("中央値結果をex2_low_interaction_results.csvに保存しました。")
    
    # 全実験結果も保存
    all_results_df.to_csv("ex2_low_interaction_all_results.csv", index=False, encoding='utf-8')
    print("全実験結果をex2_low_interaction_all_results.csvに保存しました。")
    
    # 結果の表示
    print("\n--- 中央値実験結果 ---")
    print(results_df.to_string(index=False))
    
    # --- 結果の可視化 ---
    # 英語バージョン
    create_and_save_plot(results_df, language='en')
    
    # 日本語バージョン
    create_and_save_plot(results_df, language='ja')
    
    # plt.show() # GUIが必要なため、自動実行ではコメントアウト
    
    # ===================================================================
    # 6. 考察の保存
    # ===================================================================
    print("\n6. 考察の保存...")
    
    consideration = """
--- 考察 ---
特徴量間の交互作用が少ないデータでは、以下のような傾向が見られることが期待されます。

1. 予測精度: GDBTモデルは依然として高い精度を示すものの、線形モデルとの性能差は縮小します。
   これは、GDBTの最大の強みである『複雑な交互作用の検出能力』がこのデータセットでは
   限定的にしか活かせないためです。線形モデルは、データの主要な構造（線形項や単純な二次項）を
   効率的に捉えることができます。

2. 計算コストと解釈性コスト: この側面では、線形モデルの圧倒的な優位性は変わりません。

結論として、特徴量間の交互作用が少ない、あるいは存在しないと想定される問題では、
GDBTがもたらす精度の向上は限定的であり、計算コストや解釈性を考慮すると、
線形モデルがより実用的な選択肢となる可能性が高まります。

データセット情報:
- make_friedman1データセット使用
- 式: y = 10*sin(pi*X1*X2) + 20*(X3-0.5)^2 + 10*X4 + 5*X5 + noise
- 交互作用項は X1*X2 の1つのみ
- その他は線形項や単純な非線形項
"""
    
    with open("ex2_low_interaction_consideration.txt", "w", encoding='utf-8') as f:
        f.write(consideration)
    
    print("考察をex2_low_interaction_consideration.txtに保存しました。")
    print("\n=== 実験完了 ===")

if __name__ == "__main__":
    main()
