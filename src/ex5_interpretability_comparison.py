#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機械学習モデルの比較実験 - 実データでの解釈性比較
UCIドイツ信用データセットを使用した分類タスクでの解釈性評価
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

# データ取得と前処理
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# モデル
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# 評価指標
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

def create_performance_plot(results_df, language='en'):
    """
    分類性能比較のグラフを作成し、保存する。
    """
    if language == 'ja':
        title = '分類性能の比較'
        accuracy_title = '正解率（高いほど良い）'
        f1_title = 'F1スコア（高いほど良い）'
        roc_auc_title = 'ROC AUC（高いほど良い）'
        filename_base = 'ex5_classification_performance_ja'
    else:
        title = 'Classification Performance Comparison'
        accuracy_title = 'Accuracy (Higher is Better)'
        f1_title = 'F1 Score (Higher is Better)'
        roc_auc_title = 'ROC AUC (Higher is Better)'
        filename_base = 'ex5_classification_performance'

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)
    
    # Accuracy比較
    sns.barplot(x='Accuracy', y='Model', data=results_df.sort_values('Accuracy', ascending=False), 
                ax=axes[0], palette='Blues', hue='Model', legend=False)
    axes[0].set_title(accuracy_title)
    
    # F1 Score比較
    sns.barplot(x='F1 Score', y='Model', data=results_df.sort_values('F1 Score', ascending=False), 
                ax=axes[1], palette='Greens', hue='Model', legend=False)
    axes[1].set_title(f1_title)
    
    # ROC AUC比較
    sns.barplot(x='ROC AUC', y='Model', data=results_df.sort_values('ROC AUC', ascending=False), 
                ax=axes[2], palette='Purples', hue='Model', legend=False)
    axes[2].set_title(roc_auc_title)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
    print(f"性能比較グラフを{filename_base}.png/.pdfに保存しました。")
    plt.close(fig)

def create_coefficients_plot(coef_df, language='en'):
    """
    ロジスティック回帰係数のグラフを作成し、保存する。
    """
    if language == 'ja':
        title = 'ロジスティック回帰の係数（上位/下位10）'
        xlabel = '係数（値が大きいほどリスク増）'
        ylabel = '特徴量'
        filename_base = 'ex5_logistic_coefficients_ja'
    else:
        title = 'Logistic Regression Coefficients (Top/Bottom 10)'
        xlabel = 'Coefficient (Larger values increase risk)'
        ylabel = 'Feature'
        filename_base = 'ex5_logistic_coefficients'

    fig = plt.figure(figsize=(12, 10))
    top_bottom_features = pd.concat([coef_df.head(10), coef_df.tail(10)])
    sns.barplot(x='Coefficient', y='Feature', data=top_bottom_features, hue='Feature', palette='coolwarm', legend=False)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
    print(f"ロジスティック回帰係数グラフを{filename_base}.png/.pdfに保存しました。")
    plt.close(fig)

def create_shap_summary_plot(shap_values, X_sample, language='en'):
    """
    SHAPサマリープロットを作成し、保存する。
    """
    if language == 'ja':
        title = 'LightGBMのSHAPサマリープロット'
        filename_base = 'ex5_shap_summary_ja'
    else:
        title = 'SHAP Summary Plot for LightGBM'
        filename_base = 'ex5_shap_summary'

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
    print(f"SHAPサマリープロットを{filename_base}.png/.pdfに保存しました。")
    plt.close(fig)

def main():
    print("=== 機械学習モデル比較実験 (実データでの解釈性比較) ===")
    print("5回実行の中央値による統計的に信頼性の高い評価を実施します。")
    
    # 実験設定
    n_experiments = 5
    all_results = []
    
    # ===================================================================
    # 1. データの読み込みと前処理（全実験共通）
    # ===================================================================
    print("\n1. データの読み込みと前処理...")
    print("UCIリポジトリからStatlog (German Credit Data) データセットを取得")
    
    try:
        # UCIリポジトリからStatlog (German Credit Data) データセットを取得
        german_credit = fetch_ucirepo(id=144)
        
        # データフレームに変換
        X = german_credit.data.features
        y = german_credit.data.targets
        # ターゲット変数を0（Good）と1（Bad）に変換
        y = y.replace({'class': {1: 0, 2: 1}})['class']
        
        print(f"データセット取得完了: {X.shape}")
        
    except Exception as e:
        print(f"UCIデータセットの取得に失敗しました: {e}")
        print("代替として人工データを生成します...")
        
        # 代替として人工データを生成
        from sklearn.datasets import make_classification
        X_array, y_array = make_classification(
            n_samples=1000, 
            n_features=20, 
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        
        # カテゴリカル変数を含むデータフレームを作成
        feature_names = [f'num_feature_{i}' for i in range(10)] + [f'cat_feature_{i}' for i in range(10)]
        X = pd.DataFrame(X_array, columns=feature_names)
        
        # 一部の特徴量をカテゴリカルに変換
        for i in range(10, 20):
            X.iloc[:, i] = pd.cut(X.iloc[:, i], bins=3, labels=['low', 'medium', 'high'])
        
        y = pd.Series(y_array)
        print(f"人工データ生成完了: {X.shape}")
    
    # 特徴量の型を特定
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    
    print(f"数値特徴量: {len(numerical_features)}, カテゴリカル特徴量: {len(categorical_features)}")
    
    # 前処理パイプラインの作成
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # ===================================================================
    # 複数回実験の実行
    # ===================================================================
    for experiment_id in range(n_experiments):
        print(f"\n{'='*20} 実験 {experiment_id + 1}/{n_experiments} {'='*20}")
        
        # 実験ごとに異なるランダムシードでデータ分割
        current_seed = 42 + experiment_id
        
        # データを訓練用とテスト用に分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=current_seed, stratify=y)
        
        print(f"データ読み込みと分割完了。")
        print(f"訓練データ: {X_train.shape}, テストデータ: {X_test.shape}")
        
        # ===================================================================
        # 2. モデルの定義
        # ===================================================================
        print("\n2. モデルの定義...")
        print("分類タスク用のモデルを定義（解釈性比較が目的）")
        
        models = {
            "Logistic Regression (L2)": LogisticRegression(penalty='l2', solver='liblinear', random_state=current_seed),
            "Logistic Regression (L1)": LogisticRegression(penalty='l1', solver='liblinear', random_state=current_seed),
            "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=current_seed),
            "LightGBM": lgb.LGBMClassifier(random_state=current_seed, verbosity=-1)
        }
        
        # パイプラインを各モデルに適用
        for name, model in models.items():
            models[name] = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('classifier', model)])
        
        # ===================================================================
        # 3. 実験の実行と評価
        # ===================================================================
        print("\n3. 実験の実行と評価...")
        
        experiment_results = []
        for name, pipeline in models.items():
            print(f"--- {name} の評価を開始 ---")
            
            # --- 学習時間の計測 ---
            start_train_time = time.time()
            pipeline.fit(X_train, y_train)
            train_time = time.time() - start_train_time
            
            # --- 推論時間の計測 ---
            start_pred_time = time.time()
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]  # ROC AUC計算用
            pred_time = time.time() - start_pred_time
            
            # --- 予測精度の評価 ---
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            
            experiment_results.append({
                "Model": name,
                "Train Time (s)": train_time,
                "Inference Time (s)": pred_time,
                "Accuracy": accuracy,
                "F1 Score": f1,
                "ROC AUC": roc_auc,
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
            "Train Time (s)": model_data["Train Time (s)"].median(),
            "Inference Time (s)": model_data["Inference Time (s)"].median(),
            "Accuracy": model_data["Accuracy"].median(),
            "F1 Score": model_data["F1 Score"].median(),
            "ROC AUC": model_data["ROC AUC"].median(),
        }
        median_results.append(median_result)
        
        print(f"{model_name}: ROC AUC中央値={median_result['ROC AUC']:.4f}, 学習時間中央値={median_result['Train Time (s)']:.4f}s")
    
    # ===================================================================
    # 5. 結果の保存と可視化
    # ===================================================================
    print("\n5. 結果の保存と可視化...")
    
    # 中央値結果をDataFrameに変換
    results_df = pd.DataFrame(median_results).sort_values(by="ROC AUC", ascending=False).reset_index(drop=True)
    
    # CSVファイルとして保存（中央値結果）
    results_df.to_csv("ex5_interpretability_results.csv", index=False, encoding='utf-8')
    print("中央値結果をex5_interpretability_results.csvに保存しました。")
    
    # 全実験結果も保存
    all_results_df.to_csv("ex5_interpretability_all_results.csv", index=False, encoding='utf-8')
    print("全実験結果をex5_interpretability_all_results.csvに保存しました。")
    
    # 結果の表示
    print("\n--- 中央値実験結果 (信用スコアリング) ---")
    print(results_df.to_string(index=False))
    
    # 性能比較の可視化 (英語)
    create_performance_plot(results_df, language='en')
    # 性能比較の可視化 (日本語)
    create_performance_plot(results_df, language='ja')
    
    # ===================================================================
    # 5. 解釈性の比較 (この実験の核心)
    # ===================================================================
    print("\n5. 解釈性の比較...")
    print("線形モデルとGDBTモデルの解釈性の根本的な違いを比較")
    
    # --- 5.1. 線形モデルの解釈 (ロジスティック回帰の係数) ---
    print("\n5.1. 線形モデル（ロジスティック回帰）の解釈...")
    
    # パイプラインから学習済みモデルと前処理情報を取得
    log_reg_model = models['Logistic Regression (L2)'].named_steps['classifier']
    preprocessor_fitted = models['Logistic Regression (L2)'].named_steps['preprocessor']
    
    # ワンホットエンコード後の特徴量名を取得
    try:
        ohe_feature_names = preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names = np.concatenate([numerical_features, ohe_feature_names])
    except:
        # 後方互換性のための処理
        feature_names = list(numerical_features) + [f'cat_{i}' for i in range(len(log_reg_model.coef_[0]) - len(numerical_features))]
    
    # 係数をデータフレームに格納
    coef_df = pd.DataFrame({
        'Feature': feature_names[:len(log_reg_model.coef_[0])],
        'Coefficient': log_reg_model.coef_[0]
    }).sort_values('Coefficient', ascending=False)
    
    # 係数の可視化 (英語)
    create_coefficients_plot(coef_df, language='en')
    # 係数の可視化 (日本語)
    create_coefficients_plot(coef_df, language='ja')
    
    # 係数データの保存
    coef_df.to_csv("ex5_logistic_coefficients.csv", index=False, encoding='utf-8')
    print("ロジスティック回帰係数をex5_logistic_coefficients.csvに保存しました。")
    
    # --- 5.2. GDBTモデルの解釈 (LightGBM + SHAP) ---
    print("\n5.2. GDBTモデル（LightGBM + SHAP）の解釈...")
    
    # パイプラインから学習済みモデルと前処理済みデータを取得
    lgbm_model = models['LightGBM'].named_steps['classifier']
    preprocessor_fitted_lgbm = models['LightGBM'].named_steps['preprocessor']
    X_test_transformed = preprocessor_fitted_lgbm.transform(X_test)
    
    # SHAPはDataFrameを要求する場合があるため変換
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names[:X_test_transformed.shape[1]])
    
    print("SHAP値の計算中... (少し時間がかかります)")
    try:
        # SHAP explainerを作成
        explainer = shap.TreeExplainer(lgbm_model)
        
        # 計算時間短縮のため、サンプル数を制限
        sample_size = min(100, len(X_test_transformed_df))
        X_sample = X_test_transformed_df.sample(n=sample_size, random_state=42)
        
        # SHAP値を計算
        shap_values = explainer.shap_values(X_sample)
        
        # バイナリ分類の場合、shap_valuesは配列または配列のリストの場合がある
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 正のクラス（リスクあり）のSHAP値
        
        print("SHAP値の計算完了。")
        
        # SHAPサマリープロットの表示 (英語)
        create_shap_summary_plot(shap_values, X_sample, language='en')
        # SHAPサマリープロットの表示 (日本語)
        create_shap_summary_plot(shap_values, X_sample, language='ja')
        
        # SHAP値の重要度を計算
        feature_importance = np.abs(shap_values).mean(axis=0)
        shap_importance_df = pd.DataFrame({
            'Feature': X_sample.columns,
            'SHAP_Importance': feature_importance
        }).sort_values('SHAP_Importance', ascending=False)
        
        # SHAP重要度を保存
        shap_importance_df.to_csv("ex5_shap_importance.csv", index=False, encoding='utf-8')
        print("SHAP重要度をex5_shap_importance.csvに保存しました。")
        
    except Exception as e:
        print(f"SHAP計算でエラーが発生しました: {e}")
        print("特徴量重要度で代替します...")
        
        # 代替として特徴量重要度を使用
        feature_importance = lgbm_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(feature_importance)],
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
        plt.title('LightGBM Feature Importance (Top 15)', fontsize=16)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig("ex5_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.savefig("ex5_feature_importance.pdf", bbox_inches='tight')
        print("特徴量重要度グラフをex5_feature_importance.png/.pdfに保存しました。")
        plt.close(fig) # メモリリークを防ぐために図を閉じる
        
        importance_df.to_csv("ex5_feature_importance.csv", index=False, encoding='utf-8')
        print("特徴量重要度をex5_feature_importance.csvに保存しました。")
    
    # ===================================================================
    # 6. 考察の保存
    # ===================================================================
    print("\n6. 考察の保存...")
    
    consideration = """--- 考察 ---
実データを用いた分類タスクでの解釈性比較実験の結果：

【線形モデル（ロジスティック回帰）の解釈性】
1. 直接的な解釈: 係数は各特徴量が結果に与える影響の大きさと方向を直接示します。
2. 単位あたりの影響: 「特徴量Xが1単位増加すると、ログオッズがβだけ変化する」
   という明確な数値的関係を提供します。
3. グローバルな説明: モデル全体の動作を一つの式で表現できます。
4. 規制対応: 金融業界などの規制が厳しい分野で求められる「説明可能性」を満たします。

【GDBTモデル（LightGBM + SHAP）の解釈性】
1. 事後的な説明: SHAPは学習後にモデルの予測を分析する手法です。
2. 個別予測の説明: 各予測に対する特徴量の貢献度を視覚的に示します。
3. 複雑な相互作用: 特徴量間の複雑な相互作用も捉えることができます。
4. 局所的な説明: 個々のサンプルに対する説明は詳細ですが、
   モデル全体の動作原理は依然として「ブラックボックス」です。

【解釈性の根本的な違い】
- 線形モデル: 「なぜそう予測するのか」をモデル構造自体が説明
- GDBTモデル: 「どのような予測をしたのか」を事後的に分析

【実用的な示唆】
1. 規制の厳しい分野（金融、医療等）: 線形モデルの透明性が重要
2. 高精度が最優先の分野: GDBTモデル + SHAP の組み合わせが有効
3. ビジネス理解重視: 線形モデルの直接的な解釈性が価値を持つ

実験データ: UCIドイツ信用データセット（代替：人工分類データ）
評価指標: Accuracy, F1 Score, ROC AUC
解釈性手法: 係数分析（線形）、SHAP値分析（GDBT）
"""
    
    with open("ex5_interpretability_consideration.txt", "w", encoding='utf-8') as f:
        f.write(consideration)
    
    print("考察をex5_interpretability_consideration.txtに保存しました。")
    print("\n=== 実験完了 ===")

if __name__ == "__main__":
    main()
