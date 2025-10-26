import pandas as pd
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def perform_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Performs core Exploratory Data Analysis and returns a structured dictionary."""
    results = {}
    
    # 1. Nulls and Shape
    results['shape'] = f"{df.shape[0]} rows, {df.shape[1]} columns"
    
    null_summary = df.isnull().sum()
    results['null_report'] = {
        col: {"count": count, "percent": count/len(df)*100}
        for col, count in null_summary.items() if count > 0
    }
    
    # 2. Descriptive Stats
    numeric_df = df.select_dtypes(include=np.number)
    results['descriptive_stats'] = numeric_df.describe().to_dict()
    
    # 3. Correlation Matrix (Top 5 absolute correlation pairs)
    corr_matrix = numeric_df.corr().abs().unstack()
    corr_pairs = corr_matrix.sort_values(ascending=False)
    
    corr_pairs = corr_pairs[corr_pairs != 1.0]
    
    unique_pairs = []
    seen = set()
    for (feat1, feat2), corr in corr_pairs.items():
        if feat1 != feat2 and tuple(sorted((feat1, feat2))) not in seen:
            unique_pairs.append({'feat1': feat1, 'feat2': feat2, 'corr': corr})
            seen.add(tuple(sorted((feat1, feat2))))
        if len(unique_pairs) >= 5:
            break
            
    results['top_correlations'] = unique_pairs
        
    return results

def get_feature_importance(df: pd.DataFrame, target_col: str) -> List[Dict[str, Any]]:
    """Calculates feature importance using a Random Forest model."""
    if target_col not in df.columns: return []

    df_model = df.copy().select_dtypes(include=np.number).dropna()
    if target_col not in df_model.columns or df_model.shape[0] < 2: return []

    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    if X.empty: return []

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    importance_list = sorted(
        [{'feature': feat, 'importance': imp} for feat, imp in zip(X.columns, model.feature_importances_)],
        key=lambda x: x['importance'],
        reverse=True
    )[:5]
    
    return importance_list
