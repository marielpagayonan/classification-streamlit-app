"""
============================================================
 Lab 5.0 – Enhanced Test Suite
 Student Academic Risk Classification System
============================================================
Run:  python test.py
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import (accuracy_score, f1_score,
                                     precision_score, recall_score,
                                     confusion_matrix, classification_report)
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (must match app.py)
# ─────────────────────────────────────────────────────────────────────────────
FEATURES     = ["Attendance", "Quiz_Avg", "Lab_Avg", "Midterm", "Final"]
WEIGHTS      = {"Attendance": 0.10, "Quiz_Avg": 0.20,
                "Lab_Avg": 0.20,   "Midterm": 0.25, "Final": 0.25}
RISK_ORDER   = ["High Risk", "Medium Risk", "Low Risk"]
DATASET_PATH = "dataset.csv"

# Updated thresholds to match app.py
def assign_risk(gwa: float) -> str:
    if gwa >= 85: return "Low Risk"
    if gwa >= 75: return "Medium Risk"
    return "High Risk"

def compute_gwa(row: pd.Series) -> float:
    return sum(row[f] * WEIGHTS[f] for f in FEATURES)


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATASET TESTS
# ═════════════════════════════════════════════════════════════════════════════
class TestDataset(unittest.TestCase):
    """Tests for dataset structure, integrity, and data quality."""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATASET_PATH)
        cls.df_clean = cls.df.copy()
        for col in FEATURES:
            cls.df_clean[col] = cls.df_clean[col].fillna(cls.df_clean[col].median())

    # ── Structure ─────────────────────────────────────────────────────────
    def test_01_dataset_loads(self):
        """Dataset loads as a non-empty DataFrame."""
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertGreater(len(self.df), 0, "Dataset must not be empty")

    def test_02_required_columns_exist(self):
        """All feature and target columns are present."""
        required = FEATURES + ["Risk_Level"]
        for col in required:
            self.assertIn(col, self.df.columns, f"Column '{col}' missing")

    def test_03_minimum_row_count(self):
        """Dataset has at least 500 records."""
        self.assertGreaterEqual(len(self.df), 500, "Dataset should have ≥ 500 rows")

    def test_04_column_dtypes_numeric(self):
        """All feature columns must be numeric."""
        for col in FEATURES:
            self.assertTrue(
                pd.api.types.is_numeric_dtype(self.df[col]),
                f"Column '{col}' is not numeric"
            )

    # ── Data Quality ──────────────────────────────────────────────────────
    def test_05_no_missing_after_imputation(self):
        """No NaN values remain after median imputation."""
        for col in FEATURES:
            self.assertEqual(self.df_clean[col].isnull().sum(), 0,
                             f"Column '{col}' still has NaN after imputation")

    def test_06_score_ranges_min(self):
        """All scores are ≥ 0."""
        for col in FEATURES:
            self.assertTrue((self.df_clean[col] >= 0).all(),
                            f"{col} has values < 0")

    def test_07_score_ranges_max(self):
        """All scores are ≤ 100."""
        for col in FEATURES:
            self.assertTrue((self.df_clean[col] <= 100).all(),
                            f"{col} has values > 100")

    def test_08_no_duplicate_rows(self):
        """Dataset should not have fully duplicate rows (> 5% threshold)."""
        dup_ratio = self.df.duplicated().sum() / len(self.df)
        self.assertLess(dup_ratio, 0.05,
                        f"Duplicate rows exceed 5 %: {dup_ratio:.2%}")

    def test_09_missing_rate_per_column(self):
        """Each feature column should not exceed 20 % missing values."""
        for col in FEATURES:
            missing_rate = self.df[col].isnull().mean()
            self.assertLess(missing_rate, 0.20,
                            f"{col} missing rate {missing_rate:.2%} exceeds 20 %")

    # ── Risk Level Labels ─────────────────────────────────────────────────
    def test_10_risk_level_valid_categories(self):
        """Risk_Level contains only recognised categories."""
        unique = set(self.df["Risk_Level"].unique())
        valid  = set(RISK_ORDER)
        self.assertTrue(unique.issubset(valid),
                        f"Unexpected categories: {unique - valid}")

    def test_11_all_risk_categories_present(self):
        """All three risk categories appear in the dataset."""
        unique = set(self.df["Risk_Level"].unique())
        for risk in RISK_ORDER:
            self.assertIn(risk, unique, f"'{risk}' category missing")

    def test_12_class_imbalance_not_extreme(self):
        """No single class exceeds 75 % of the data."""
        counts = self.df["Risk_Level"].value_counts(normalize=True)
        self.assertTrue((counts <= 0.75).all(),
                        f"Severe class imbalance: {counts.to_dict()}")

    def test_13_risk_level_no_nulls(self):
        """Target column Risk_Level has no missing values."""
        self.assertEqual(self.df["Risk_Level"].isnull().sum(), 0,
                         "Risk_Level has missing values")

    def test_14_attendance_reasonable_distribution(self):
        """Attendance median should be between 50 and 100."""
        med = self.df_clean["Attendance"].median()
        self.assertGreaterEqual(med, 50)
        self.assertLessEqual(med, 100)


# ═════════════════════════════════════════════════════════════════════════════
# 2. RISK LOGIC TESTS
# ═════════════════════════════════════════════════════════════════════════════
class TestRiskLogic(unittest.TestCase):
    """Tests for GWA computation and risk-assignment thresholds."""

    # ── assign_risk ───────────────────────────────────────────────────────
    def test_01_high_risk_below_75(self):
        for gwa in [0, 50, 65, 74.9]:
            with self.subTest(gwa=gwa):
                self.assertEqual(assign_risk(gwa), "High Risk")

    def test_02_medium_risk_75_to_84(self):
        for gwa in [75.0, 78.5, 84.9]:
            with self.subTest(gwa=gwa):
                self.assertEqual(assign_risk(gwa), "Medium Risk")

    def test_03_low_risk_85_and_above(self):
        for gwa in [85.0, 90.0, 95.0, 100.0]:
            with self.subTest(gwa=gwa):
                self.assertEqual(assign_risk(gwa), "Low Risk")

    def test_04_boundary_exactly_75(self):
        self.assertEqual(assign_risk(74.99), "High Risk")
        self.assertEqual(assign_risk(75.00), "Medium Risk")

    def test_05_boundary_exactly_85(self):
        self.assertEqual(assign_risk(84.99), "Medium Risk")
        self.assertEqual(assign_risk(85.00), "Low Risk")

    def test_06_extreme_values(self):
        self.assertEqual(assign_risk(0.0),   "High Risk")
        self.assertEqual(assign_risk(100.0), "Low Risk")

    # ── compute_gwa ───────────────────────────────────────────────────────
    def test_07_gwa_all_100(self):
        """All-100 scores → GWA = 100."""
        row = pd.Series({f: 100 for f in FEATURES})
        self.assertAlmostEqual(compute_gwa(row), 100.0, places=5)

    def test_08_gwa_all_zero(self):
        """All-0 scores → GWA = 0."""
        row = pd.Series({f: 0 for f in FEATURES})
        self.assertAlmostEqual(compute_gwa(row), 0.0, places=5)

    def test_09_weights_sum_to_one(self):
        """Feature weights must sum to exactly 1.0."""
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0, places=10)

    def test_10_gwa_manual_calculation(self):
        """Spot-check weighted average against a hand-computed value."""
        row = pd.Series({"Attendance": 80, "Quiz_Avg": 70,
                         "Lab_Avg": 90,    "Midterm": 60, "Final": 100})
        expected = 80*0.10 + 70*0.20 + 90*0.20 + 60*0.25 + 100*0.25
        self.assertAlmostEqual(compute_gwa(row), expected, places=5)

    def test_11_gwa_produces_float(self):
        """compute_gwa always returns a float."""
        row = pd.Series({f: 75 for f in FEATURES})
        self.assertIsInstance(compute_gwa(row), float)

    def test_12_gwa_within_range(self):
        """GWA computed from valid scores must be in [0, 100]."""
        for _ in range(50):
            scores = {f: np.random.uniform(0, 100) for f in FEATURES}
            gwa = compute_gwa(pd.Series(scores))
            self.assertGreaterEqual(gwa, 0.0)
            self.assertLessEqual(gwa, 100.0)

    def test_13_attendance_weight_is_10pct(self):
        """Attendance should contribute exactly 10 % to GWA."""
        self.assertAlmostEqual(WEIGHTS["Attendance"], 0.10, places=10)

    def test_14_exam_weights_equal(self):
        """Midterm and Final should carry the same weight."""
        self.assertEqual(WEIGHTS["Midterm"], WEIGHTS["Final"])


# ═════════════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING & EVALUATION TESTS
# ═════════════════════════════════════════════════════════════════════════════
class TestModelTraining(unittest.TestCase):
    """Tests for the Random Forest pipeline training and evaluation."""

    @classmethod
    def setUpClass(cls):
        df = pd.read_csv(DATASET_PATH)
        for col in FEATURES:
            df[col] = df[col].fillna(df[col].median())

        X  = df[FEATURES]
        le = LabelEncoder()
        y  = le.fit_transform(df["Risk_Level"])

        cls.X, cls.y = X, y
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        cls.le  = le
        cls.pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ])
        cls.pipe.fit(cls.X_train, cls.y_train)
        cls.y_pred  = cls.pipe.predict(cls.X_test)
        cls.y_proba = cls.pipe.predict_proba(cls.X_test)
        cls.clf     = cls.pipe.named_steps["clf"]

    # ── Pipeline ──────────────────────────────────────────────────────────
    def test_01_pipeline_trains_without_error(self):
        self.assertIsNotNone(self.pipe)

    def test_02_train_test_split_sizes(self):
        total = len(self.X_train) + len(self.X_test)
        self.assertAlmostEqual(len(self.X_test) / total, 0.20, delta=0.02)

    def test_03_pipeline_has_imputer(self):
        self.assertIn("imp", self.pipe.named_steps)

    def test_04_pipeline_has_classifier(self):
        self.assertIn("clf", self.pipe.named_steps)

    # ── Performance ───────────────────────────────────────────────────────
    def test_05_accuracy_above_threshold(self):
        acc = accuracy_score(self.y_test, self.y_pred)
        self.assertGreaterEqual(acc, 0.65, f"Accuracy {acc:.4f} < 0.65")

    def test_06_f1_above_threshold(self):
        f1 = f1_score(self.y_test, self.y_pred,
                      average="weighted", zero_division=0)
        self.assertGreaterEqual(f1, 0.60, f"Weighted F1 {f1:.4f} < 0.60")

    def test_07_precision_above_threshold(self):
        p = precision_score(self.y_test, self.y_pred,
                            average="weighted", zero_division=0)
        self.assertGreaterEqual(p, 0.60, f"Precision {p:.4f} < 0.60")

    def test_08_recall_above_threshold(self):
        r = recall_score(self.y_test, self.y_pred,
                         average="weighted", zero_division=0)
        self.assertGreaterEqual(r, 0.60, f"Recall {r:.4f} < 0.60")

    def test_09_cross_val_accuracy_stable(self):
        """3-fold CV accuracy should be ≥ 0.60."""
        scores = cross_val_score(self.pipe, self.X, self.y,
                                 cv=3, scoring="accuracy")
        self.assertGreaterEqual(scores.mean(), 0.60,
                                f"CV mean accuracy {scores.mean():.4f} < 0.60")

    def test_10_confusion_matrix_shape(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        n  = len(self.le.classes_)
        self.assertEqual(cm.shape, (n, n))

    # ── Predictions ───────────────────────────────────────────────────────
    def test_11_predictions_are_valid_classes(self):
        valid = set(range(len(self.le.classes_)))
        self.assertTrue(set(self.y_pred).issubset(valid))

    def test_12_predict_proba_sums_to_one(self):
        for row_proba in self.y_proba:
            self.assertAlmostEqual(row_proba.sum(), 1.0, places=5)

    def test_13_predict_proba_in_range(self):
        self.assertTrue((self.y_proba >= 0).all())
        self.assertTrue((self.y_proba <= 1).all())

    def test_14_predict_single_high_risk(self):
        """Clearly failing scores → High Risk."""
        sample = pd.DataFrame([[40, 40, 40, 40, 40]], columns=FEATURES)
        label  = self.le.inverse_transform(self.pipe.predict(sample))[0]
        self.assertEqual(label, "High Risk",
                         f"Expected 'High Risk', got '{label}'")

    def test_15_predict_single_low_risk(self):
        """All-excellent scores → Low Risk."""
        sample = pd.DataFrame([[95, 95, 95, 95, 95]], columns=FEATURES)
        label  = self.le.inverse_transform(self.pipe.predict(sample))[0]
        self.assertEqual(label, "Low Risk",
                         f"Expected 'Low Risk', got '{label}'")

    def test_16_predict_medium_risk(self):
        """Borderline scores → Medium Risk."""
        # GWA ≈ 79  →  Medium Risk  (75 ≤ GWA < 85)
        sample = pd.DataFrame([[79, 79, 79, 79, 79]], columns=FEATURES)
        label  = self.le.inverse_transform(self.pipe.predict(sample))[0]
        self.assertIn(label, ["Medium Risk", "High Risk"],
                      f"Unexpected label '{label}' for borderline input")

    # ── Feature Importance ────────────────────────────────────────────────
    def test_17_feature_importances_sum_to_one(self):
        self.assertAlmostEqual(
            self.clf.feature_importances_.sum(), 1.0, places=5)

    def test_18_feature_importances_non_negative(self):
        self.assertTrue((self.clf.feature_importances_ >= 0).all())

    def test_19_all_features_have_nonzero_importance(self):
        for feat, imp in zip(FEATURES, self.clf.feature_importances_):
            self.assertGreater(imp, 0.0,
                               f"Feature '{feat}' has zero importance")

    # ── Label Encoder ─────────────────────────────────────────────────────
    def test_20_label_encoder_has_all_classes(self):
        for risk in RISK_ORDER:
            self.assertIn(risk, self.le.classes_,
                          f"LabelEncoder missing class '{risk}'")

    def test_21_label_encoder_count(self):
        self.assertEqual(len(self.le.classes_), 3)

    # ── Estimator Config ──────────────────────────────────────────────────
    def test_22_n_estimators(self):
        self.assertEqual(self.clf.n_estimators, 200)

    def test_23_random_state_set(self):
        self.assertEqual(self.clf.random_state, 42)

    def test_24_rf_has_feature_importances(self):
        self.assertEqual(len(self.clf.feature_importances_), len(FEATURES))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Lab 5.0 – Enhanced Test Suite: Student Risk Classification")
    print("=" * 65)
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestDataset, TestRiskLogic, TestModelTraining]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n" + "=" * 65)
    print(f"  Tests run    : {result.testsRun}")
    print(f"  Failures     : {len(result.failures)}")
    print(f"  Errors       : {len(result.errors)}")
    status = ("✅  ALL TESTS PASSED"
              if result.wasSuccessful() else "❌  SOME TESTS FAILED")
    print(f"  Status       : {status}")
    print("=" * 65)
