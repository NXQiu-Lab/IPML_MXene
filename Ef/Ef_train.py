"""
Machine Learning Pipeline for Materials Property Prediction
Companion code for paper: Formation Energy Prediction via Feature Selection and Ensemble Learning.

This script implements:
1. Data Preprocessing & 3-Sigma Outlier Filtering
2. Pearson Correlation Analysis & Collinearity Feature Reduction
3. Benchmark Evaluation across 8 ML Models (5-Fold CV & Train/Test Split)
"""

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =====================================================
# 1. 特征筛选与预处理模块 (Feature Engineering)
# =====================================================
def preprocess_and_select_features(data_path, target_col, corr_threshold=0.8):
    """
    数据预处理与特征筛选逻辑：
    - 去除无用标识列
    - 3-Sigma 目标值异常检测与剔除
    - 基于 Pearson 相关系数过滤高共线性特征（保留与目标相关性更高者）
    - 数据标准化 (StandardScaler)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[Error] Dataset file not found at: {data_path}")

    # 读取数据，过滤前两列索引/标识
    data = pd.read_csv(data_path).iloc[:, 2:]
    numeric_cols = data.columns.difference([target_col])

    # 1.1 3-Sigma 异常值过滤
    target = data[target_col].copy()
    target = target.replace([np.inf, -np.inf], np.nan).fillna(target.mean())
    mu, sigma = target.mean(), target.std(ddof=0)
    
    outlier_mask = np.abs((target - mu) / sigma) > 3
    print(f"[Preprocess] 3-Sigma outlier filtering: Removed {outlier_mask.sum()} samples ({outlier_mask.mean():.2%}).")
    data_clean = data.loc[~outlier_mask].reset_index(drop=True)

    # 1.2 Pearson 相关系数特征筛选
    num_cols_for_corr = numeric_cols.tolist() + [target_col]
    corr_matrix = data_clean[num_cols_for_corr].corr(method='pearson')
    feature_corr_matrix = corr_matrix.drop(index=target_col, columns=target_col)

    to_remove = set()
    for i, col1 in enumerate(feature_corr_matrix.columns):
        for j, col2 in enumerate(feature_corr_matrix.columns):
            if j <= i:
                continue
            if abs(feature_corr_matrix.loc[col1, col2]) > corr_threshold:
                # 保留与目标列相关性更强的特征，删除另一个
                corr1 = abs(corr_matrix.loc[col1, target_col])
                corr2 = abs(corr_matrix.loc[col2, target_col])
                to_remove.add(col2 if corr1 >= corr2 else col1)

    selected_features = [col for col in numeric_cols if col not in to_remove]
    print(f"[Feature Selection] Removed {len(to_remove)} redundant features (threshold = {corr_threshold}).")
    print(f"[Feature Selection] Final retained feature count: {len(selected_features)}")

    # 1.3 数据标准化
    X_full = data_clean[selected_features]
    y_full = data_clean[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    return X_scaled, y_full, selected_features


# =====================================================
# 2. 模型定义配置 (Model Configuration)
# =====================================================
def get_model_pipeline(input_dim, seed=100):
    """定义论文评估的 8 种机器学习模型极其超参数"""
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=seed
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.11024461847067128,
            max_depth=13, min_samples_split=14, min_samples_leaf=18,
            subsample=0.608789769443629, random_state=seed
        ),
        'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=seed),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'KernelRidge': KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1),
        'PLS': PLSRegression(n_components=min(2, input_dim)),
        'Ridge': Ridge(alpha=1.0, random_state=seed),
        'MLP': MLPRegressor(
            hidden_layer_sizes=(80, 40), activation='relu', solver='adam', alpha=0.01,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
            learning_rate='adaptive', learning_rate_init=0.001, batch_size='auto',
            max_iter=1000, random_state=seed
        ),
    }

    abbreviations = {
        'RandomForest': 'RFR', 'GradientBoosting': 'GBR', 'AdaBoost': 'ABR',
        'KNN': 'KNN', 'KernelRidge': 'KRR', 'Ridge': 'Ridge', 'PLS': 'PLS', 'MLP': 'MLP'
    }
    return models, abbreviations


# =====================================================
# 3. 评估与实验流程 (Experiment Execution)
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Paper Replication Pipeline for Material Property Prediction")
    parser.add_argument('--data_path', type=str, default='feature1_unique_FIG_SHAP.csv', help='Path to dataset CSV')
    parser.add_argument('--target_col', type=str, default='Formation_Energy_eV_per_atom', help='Target column name')
    parser.add_argument('--threshold', type=float, default=0.8, help='Pearson correlation threshold')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility')
    args = parser.parse_args()

    # --- 1. 运行特征筛选 ---
    print("=" * 80)
    print(" 1. DATA PREPROCESSING & FEATURE SELECTION ")
    print("=" * 80)
    X_processed, y_full, selected_features = preprocess_and_select_features(
        args.data_path, args.target_col, args.threshold
    )

    models, abbreviations = get_model_pipeline(input_dim=X_processed.shape[1], seed=args.seed)

    # --- 2. 5折交叉验证 ---
    print("\n" + "=" * 80)
    print(" 2. 5-FOLD CROSS-VALIDATION BENCHMARK ")
    print("=" * 80)

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    scoring = {
        'r2': 'r2',
        'neg_mse': 'neg_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error'
    }

    cv_results = {}
    for name, model in models.items():
        try:
            res = cross_validate(model, X_processed, y_full, cv=kf, scoring=scoring)
            r2_s, rmse_s, mae_s = res['test_r2'], np.sqrt(-res['test_neg_mse']), -res['test_neg_mae']
            cv_results[name] = {
                'r2_mean': r2_s.mean(), 'r2_std': r2_s.std(),
                'rmse_mean': rmse_s.mean(), 'rmse_std': rmse_s.std(),
                'mae_mean': mae_s.mean(), 'mae_std': mae_s.std()
            }
        except Exception as e:
            print(f"[Warning] Model {name} failed in CV: {e}")

    # 打印适合作为论文 Table 输出的 CV 汇总格式
    print(f"{'Model':<8} | {'R² (Mean ± STD)':<20} | {'RMSE (Mean ± STD)':<20} | {'MAE (Mean ± STD)':<20}")
    print("-" * 80)
    for name, m in cv_results.items():
        print(f"{abbreviations[name]:<8} | {m['r2_mean']:.4f} ± {m['r2_std']:.4f}     | "
              f"{m['rmse_mean']:.4f} ± {m['rmse_std']:.4f}   | "
              f"{m['mae_mean']:.4f} ± {m['mae_std']:.4f}")

    # --- 3. 80/20 Train/Test Split 划分评估 ---
    print("\n" + "=" * 80)
    print(" 3. SINGLE TRAIN/TEST SPLIT (80/20) EVALUATION ")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_full, test_size=0.2, random_state=args.seed
    )
    single_split_results = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_tr_pred, y_te_pred = model.predict(X_train), model.predict(X_test)
            single_split_results[name] = {
                'r2_train': r2_score(y_train, y_tr_pred),
                'r2_test': r2_score(y_test, y_te_pred),
                'rmse_train': np.sqrt(mean_squared_error(y_train, y_tr_pred)),
                'rmse_test': np.sqrt(mean_squared_error(y_test, y_te_pred)),
                'mae_train': mean_absolute_error(y_train, y_tr_pred),
                'mae_test': mean_absolute_error(y_test, y_te_pred)
            }
        except Exception as e:
            print(f"[Warning] Model {name} failed in Train/Test split: {e}")

    # 打印单次划分评估格式
    print(f"{'Model':<8} | {'Train R²':<10} {'Test R²':<10} | {'Train RMSE':<11} {'Test RMSE':<10} | {'Train MAE':<10} {'Test MAE':<10}")
    print("-" * 85)
    for name, res in single_split_results.items():
        print(f"{abbreviations[name]:<8} | {res['r2_train']:.4f}    {res['r2_test']:.4f}    | "
              f"{res['rmse_train']:.4f}     {res['rmse_test']:.4f}    | "
              f"{res['mae_train']:.4f}    {res['mae_test']:.4f}")


if __name__ == '__main__':
    main()
