"""
Official Code Implementation for Paper Benchmark:
"Convex Hull Energy Prediction via Hybrid Feature Selection and Machine Learning"

Pipeline Workflow:
1. Target 3-Sigma Outlier Filtering
2. Train/Test Split (80/20) to prevent data leakage
3. Pearson Correlation Filtering (on Train Set only)
4. RFECV Feature Selection with Random Forest (on Train Set only)
5. 5-Fold Cross Validation & Train/Test Evaluation across 8 Models
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


# =====================================================
# 1. 数据预处理与两阶段特征筛选 (Feature Engineering)
# =====================================================
def feature_selection_pipeline(data_path, target_col, pearson_threshold=0.95, seed=1000):
    """
    两阶段特征筛选流程 (严格防止数据泄漏):
    1. 剔除目标列 3-Sigma 异常值
    2. 随机划分 80% 训练集与 20% 测试集
    3. 基于训练集的 Pearson 相关系数剔除高度冗余特征
    4. 基于训练集的 RFECV (递归特征消除交叉验证) 确定最优特征子集
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[Error] Dataset file not found at: {data_path}")

    # 读取数据，忽略前两列编号/标识列
    data = pd.read_csv(data_path).iloc[:, 2:]

    # --- 1.1 3-Sigma 异常值过滤 ---
    target = data[target_col].copy()
    target = target.replace([np.inf, -np.inf], np.nan).fillna(target.mean())
    mu, sigma = target.mean(), target.std(ddof=0)
    
    outlier_mask = np.abs((target - mu) / sigma) > 3
    print(f"[Preprocess] 3-Sigma outlier filtering: Removed {outlier_mask.sum()} samples ({outlier_mask.mean():.2%}).")
    data_clean = data.loc[~outlier_mask].reset_index(drop=True)

    # --- 1.2 划分训练集与测试集 (防止后续特征选择发生泄漏) ---
    X = data_clean.drop(columns=[target_col])
    y = data_clean[target_col].values

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # --- 1.3 Pearson 相关系数筛选 (仅在训练集上计算) ---
    train_df = X_train_df.copy()
    train_df[target_col] = y_train
    numeric_cols = X_train_df.columns.tolist()

    corr_matrix = train_df.corr(method='pearson')
    feature_corr_matrix = corr_matrix.drop(index=target_col, columns=target_col)

    to_remove = set()
    for i, col1 in enumerate(feature_corr_matrix.columns):
        for j, col2 in enumerate(feature_corr_matrix.columns):
            if j <= i:
                continue
            if abs(feature_corr_matrix.loc[col1, col2]) > pearson_threshold:
                corr1 = abs(corr_matrix.loc[col1, target_col])
                corr2 = abs(corr_matrix.loc[col2, target_col])
                to_remove.add(col2 if corr1 >= corr2 else col1)

    selected_pearson = [col for col in numeric_cols if col not in to_remove]
    print(f"[Stage 1 - Pearson Filter] Retained {len(selected_pearson)} features (threshold = {pearson_threshold}).")

    # --- 1.4 RFECV 递归特征消除 (仅在训练集上训练) ---
    scaler_rfe = StandardScaler()
    X_train_pearson = X_train_df[selected_pearson]
    X_train_scaled = scaler_rfe.fit_transform(X_train_pearson)

    rfe_estimator = RandomForestRegressor(n_estimators=100, random_state=100, n_jobs=1)
    rfecv = RFECV(
        estimator=rfe_estimator,
        step=1,
        cv=KFold(n_splits=10, shuffle=True, random_state=100),
        scoring='r2',
        min_features_to_select=3,
        n_jobs=1
    )
    rfecv.fit(X_train_scaled, y_train)

    selected_rfe = [selected_pearson[i] for i in range(len(selected_pearson)) if rfecv.support_[i]]
    print(f"[Stage 2 - RFECV Selection] Optimal feature count selected by RFECV: {rfecv.n_features_}")
    print(f"[Selected Features]: {selected_rfe}\n")

    return X, y, X_train_df[selected_rfe], X_test_df[selected_rfe], y_train, y_test, selected_rfe


# =====================================================
# 2. 模型 Pipeline 定义
# =====================================================
def get_model_pipelines(n_features, seed=100):
    """定义 8 个包含 Standard Scaler 的 Scikit-Learn Pipeline 模型"""
    models = {
        'RandomForest': Pipeline([
            ('model', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=seed, n_jobs=1))
        ]),
        'GradientBoosting': Pipeline([
            ('model', GradientBoostingRegressor(
                n_estimators=330, learning_rate=0.1305966, max_depth=3,
                min_samples_split=8, min_samples_leaf=5, random_state=seed
            ))
        ]),
        'AdaBoost': Pipeline([
            ('model', AdaBoostRegressor(n_estimators=50, random_state=seed))
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsRegressor(n_neighbors=5))
        ]),
        'KernelRidge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1))
        ]),
        'PLS': Pipeline([
            ('scaler', StandardScaler()),
            ('model', PLSRegression(n_components=min(2, n_features)))
        ]),
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ]),
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(200, 20, 140), activation='relu', solver='lbfgs',
                alpha=0.0840744, learning_rate_init=0.000118, max_iter=5000, random_state=seed
            ))
        ]),
    }

    abbreviations = {
        'RandomForest': 'RFR', 'GradientBoosting': 'GBR', 'AdaBoost': 'ABR',
        'KNN': 'KNN', 'KernelRidge': 'KRR', 'PLS': 'PLS', 'Ridge': 'Ridge', 'MLP': 'MLP'
    }
    return models, abbreviations


# =====================================================
# 3. 主实验运行逻辑
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Paper Replication Pipeline for Convex Hull Energy Prediction")
    parser.add_argument('--data_path', type=str, default='New_hull_feature_processed11.csv', help='Path to dataset CSV')
    parser.add_argument('--target_col', type=str, default='E_above_hull_eV_per_atom', help='Target column name')
    parser.add_argument('--threshold', type=float, default=0.95, help='Pearson correlation threshold')
    parser.add_argument('--seed', type=int, default=1000, help='Random seed for reproducibility')
    args = parser.parse_args()

    # --- 1. 执行数据预处理与特征筛选 ---
    print("=" * 80)
    print(" 1. DATA PREPROCESSING & HYBRID FEATURE SELECTION ")
    print("=" * 80)
    X, y, X_train_final, X_test_final, y_train, y_test, selected_features = feature_selection_pipeline(
        args.data_path, args.target_col, args.threshold, args.seed
    )

    models, abbrevs = get_model_pipelines(n_features=len(selected_features), seed=100)

    # --- 2. 5折交叉验证评估 ---
    print("=" * 80)
    print(" 2. 5-FOLD CROSS-VALIDATION BENCHMARK ")
    print("=" * 80)

    kf = KFold(n_splits=5, shuffle=True, random_state=100)
    scoring = {'r2': 'r2', 'neg_mse': 'neg_mean_squared_error', 'neg_mae': 'neg_mean_absolute_error'}

    X_cv = X[selected_features]
    cv_results = {}

    for name, model in models.items():
        try:
            res = cross_validate(model, X_cv, y, cv=kf, scoring=scoring, n_jobs=1)
            r2_s, rmse_s, mae_s = res['test_r2'], np.sqrt(-res['test_neg_mse']), -res['test_neg_mae']

            cv_results[name] = {
                'r2_mean': r2_s.mean(), 'r2_std': r2_s.std(),
                'rmse_mean': rmse_s.mean(), 'rmse_std': rmse_s.std(),
                'mae_mean': mae_s.mean(), 'mae_std': mae_s.std()
            }
        except Exception as e:
            print(f"[Warning] Model {name} failed in CV: {e}")

    # 打印适合作为论文 Table 输出的 5-Fold CV 格式
    print(f"{'Model':<8} | {'R² (Mean ± STD)':<20} | {'RMSE (Mean ± STD)':<20} | {'MAE (Mean ± STD)':<20}")
    print("-" * 80)
    for name, m in cv_results.items():
        print(f"{abbrevs[name]:<8} | {m['r2_mean']:.4f} ± {m['r2_std']:.4f}     | "
              f"{m['rmse_mean']:.4f} ± {m['rmse_std']:.4f}   | "
              f"{m['mae_mean']:.4f} ± {m['mae_std']:.4f}")

    # --- 3. 单次 Holdout (80/20 Train/Test) 划分评估 ---
    print("\n" + "=" * 80)
    print(" 3. HOLDOUT TRAIN/TEST (80/20) EVALUATION ")
    print("=" * 80)

    single_split_results = {}
    for name, model in models.items():
        try:
            model.fit(X_train_final, y_train)
            y_train_pred = np.ravel(model.predict(X_train_final))
            y_test_pred = np.ravel(model.predict(X_test_final))

            single_split_results[name] = {
                'r2_train': r2_score(y_train, y_train_pred),
                'r2_test': r2_score(y_test, y_test_pred),
                'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'mae_train': mean_absolute_error(y_train, y_train_pred),
                'mae_test': mean_absolute_error(y_test, y_test_pred)
            }
        except Exception as e:
            print(f"[Warning] Model {name} failed in holdout split: {e}")


if __name__ == '__main__':
    main()