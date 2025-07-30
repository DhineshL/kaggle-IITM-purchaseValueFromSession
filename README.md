# Engage 2 - Value from Clicks to Conversions

**Kaggle Competition:** [Engage 2 - Value from Clicks to Conversions](https://www.kaggle.com/competitions/engage-2-value-from-clicks-to-conversions/data)

---

## 📊 Feature Engineering & EDA

- Cleaned the data and removed duplicate rows.
- Replaced placeholder values like `'not available in demo dataset'` and `'(not set)'` with `NaN`.
- Dropped high-cardinality categorical columns that did not contribute significantly to the target.
- Extracted new features from:
  - `geoNetwork.country` by grouping similar countries.
  - `date` and `visitStartTime` to derive time-based features like hour, day of week, etc.
- Removed columns with a single unique value (zero variance).
- Dropped identifier columns like `sessionId` and `fullVisitorId` to avoid data leakage.

---

## 🧹 Preprocessing

- Imputed missing **numerical** values using the **mean**.
- Imputed missing **categorical** values using the **most frequent** value.
- Applied **Target Encoding** to all categorical columns to numerically represent them based on the target distribution.

---

## 🤖 Model Training

- Trained and evaluated multiple models:
  - **Linear models**: Ridge, Lasso
  - **Tree-based models**: Random Forest, LightGBM, XGBoost
- XGBoost performed best on validation/test metrics and was selected as the final model.

---

## 🚀 Final Model: XGBoost

```python
from xgboost import XGBRegressor

best_model = XGBRegressor(
    max_depth=15,
    max_leaves=None,
    min_child_weight=None,
    missing=np.nan,
    monotone_constraints=None,
    multi_strategy=None,
    n_estimators=100,
    n_jobs=-1
)
```
| Parameter                   | Description                                                                                                                     |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `max_depth=15`              | Maximum depth of each decision tree. Higher depth allows the model to learn more complex patterns, but may lead to overfitting. |
| `max_leaves=None`           | Automatically determines the maximum number of leaves per tree (no manual cap set).                                             |
| `min_child_weight=None`     | Uses the default value to control overfitting by requiring a minimum sum of instance weight (hessian) in each child node.       |
| `missing=np.nan`            | Specifies the placeholder value for missing entries during training.                                                            |
| `monotone_constraints=None` | No constraints on the direction of feature influence on predictions.                                                            |
| `multi_strategy=None`       | Not applicable for regression tasks; relevant only for multi-class problems.                                                    |
| `n_estimators=100`          | Number of trees (boosting rounds) to build. More estimators can lead to better performance but slower training.                 |
| `n_jobs=-1`                 | Enables parallel processing using all available CPU cores for faster model training.                                            |

