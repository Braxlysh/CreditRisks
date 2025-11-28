from typing import Any, Dict, List, Sequence

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
ID_COL = "id"
OTHER = "__OTHER__"


def build_features_from_df(
    df: pd.DataFrame,
    ohe_cols: Sequence[str],
    levels: Dict[str, set],
    feature_cols: Sequence[str] | None = None,
    id_col: str = ID_COL,
    other_label: str = OTHER,
) -> pd.DataFrame:
    rn = df.groupby(id_col).size().rename("rn_count").to_frame()
    parts = [rn]

    for c in ohe_cols:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            s = s.astype("Int8")

        s = s.astype(str)
        keep = levels[c]
        s = s.where(s.isin(keep), other=other_label)

        tbl = (
            df.assign(__lvl=s)
              .groupby([id_col, "__lvl"])
              .size()
              .unstack(fill_value=0)
        )
        tbl.columns = [f"{c}__{lvl}" for lvl in tbl.columns]
        parts.append(tbl)

    feats = parts[0]
    for p in parts[1:]:
        feats = feats.join(p, how="left")
    feats = feats.fillna(0)

    feats = feats.reset_index().rename(columns={id_col: "id"})

    if feature_cols is not None:
        feature_cols = list(feature_cols)
        for col in feature_cols:
            if col not in feats.columns:
                feats[col] = 0
        extra_cols = [c for c in feats.columns if c not in (["id"] + feature_cols)]
        if extra_cols:
            feats = feats.drop(columns=extra_cols)
        feats = feats[["id"] + feature_cols]

    return feats


class LgbmPipeline:
    def __init__(
        self,
        model: LGBMClassifier,
        ohe_cols,
        levels,
        feature_cols,
    ):
        self.model = model
        self.ohe_cols = ohe_cols
        self.levels = levels
        self.feature_cols = feature_cols

    @classmethod
    def from_files(cls, model_path: str, artifacts_path: str) -> "LgbmPipeline":
        model: LGBMClassifier = joblib.load(model_path)
        artifacts: Dict[str, Any] = joblib.load(artifacts_path)

        return cls(
            model=model,
            ohe_cols=artifacts["ohe_cols"],
            levels=artifacts["levels"],
            feature_cols=artifacts["feature_cols"],
        )

    # ---- предикт для одной строки JSON ----
    def predict_one(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        df_raw = pd.DataFrame([payload])

        feats = build_features_from_df(
            df=df_raw,
            ohe_cols=self.ohe_cols,
            levels=self.levels,
            feature_cols=self.feature_cols,
        )

        X_new = feats[self.feature_cols]
        proba = self.model.predict_proba(X_new)[0, 1]

        return {
            "id": int(feats["id"].iloc[0]),
            "proba": float(proba),
        }

    # ---- предикт для списка JSON (batch) ----
    def predict_batch(self, payloads: List[Dict[str, Any]]) -> pd.DataFrame:
        df_raw = pd.DataFrame(payloads)

        feats = build_features_from_df(
            df=df_raw,
            ohe_cols=self.ohe_cols,
            levels=self.levels,
            feature_cols=self.feature_cols,
        )

        X_new = feats[self.feature_cols]
        proba = self.model.predict_proba(X_new)[:, 1]

        result = feats[["id"]].copy()
        result["proba"] = proba
        return result


# Пример использования 
# df = pd.read_parquet("test_dataset\test_unprepeated_data.parquet", engine="pyarrow") # сырые данные
# test = pd.read_csv("test_dataset\\test_predictions.csv") # таргет для сырых данных

# pipe = LgbmPipeline.from_files(
#     model_path="models/lgbm_final2.pkl", # модель
#     artifacts_path="models/feature_artifacts.pkl", # артефакты для подготовки сырых данных
# )

# pred_df = pipe.predict_batch(df.to_dict(orient="records"))
# df1 = (test.merge(pred_df, on="id", how="inner", validate="one_to_one"))
# y_true = df1["y_true"]
# y_pred = df1["proba"]

# auc = roc_auc_score(y_true, y_pred)
# print("ROC AUC:", auc)