from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import auc, roc_curve

from a1_inference.clip4retrofit_ovr_metrics import (
    compute_cosine_similarity_scores,
    compute_ovr_metrics,
    label_sets_to_multihot,
)


class OVRMetricsTest(unittest.TestCase):
    def test_single_label_ovr_matches_sklearn_per_class(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        y_score = np.array(
            [
                [0.90, 0.05, 0.05],
                [0.70, 0.20, 0.10],
                [0.10, 0.80, 0.10],
                [0.20, 0.60, 0.20],
                [0.10, 0.20, 0.70],
                [0.05, 0.15, 0.80],
                [0.55, 0.35, 0.10],
                [0.25, 0.50, 0.25],
                [0.20, 0.25, 0.55],
            ]
        )

        result = compute_ovr_metrics(
            y_true,
            y_score,
            class_names=["zero", "one", "two"],
        )

        expected_aurocs = []
        for class_index, class_result in enumerate(result.per_class):
            expected_binary = (y_true == class_index).astype(np.uint8)
            expected_fpr, expected_tpr, _ = roc_curve(
                expected_binary,
                y_score[:, class_index],
                pos_label=1,
                drop_intermediate=True,
            )
            expected_auc = auc(expected_fpr, expected_tpr)
            expected_aurocs.append(expected_auc)
            np.testing.assert_allclose(class_result.fpr, expected_fpr)
            np.testing.assert_allclose(class_result.tpr, expected_tpr)
            self.assertAlmostEqual(class_result.auroc, expected_auc)

        self.assertAlmostEqual(result.macro_auroc, float(np.mean(expected_aurocs)))

    def test_multilabel_semantic_mask_targets(self) -> None:
        labels_per_image = [
            {"bridge", "building"},
            {"building"},
            {"bicycle"},
            {"bridge", "bicycle"},
            {"building", "bicycle"},
            {"bridge"},
        ]
        class_labels = ["bridge", "building", "bicycle"]
        y_true = label_sets_to_multihot(labels_per_image, class_labels)
        y_score = np.array(
            [
                [0.9, 0.8, 0.1],
                [0.1, 0.9, 0.2],
                [0.2, 0.1, 0.9],
                [0.8, 0.2, 0.8],
                [0.1, 0.8, 0.7],
                [0.7, 0.1, 0.2],
            ]
        )

        result = compute_ovr_metrics(
            y_true,
            y_score,
            class_names=class_labels,
            class_labels=class_labels,
        )

        self.assertEqual(result.n_classes, 3)
        self.assertEqual([item.positive_count for item in result.per_class], [3, 3, 3])
        self.assertAlmostEqual(result.macro_auroc, 1.0)
        self.assertAlmostEqual(result.macro_fpr_at_target_tpr, 0.0)

    def test_fpr95_uses_first_attainable_tpr_point(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array(
            [
                [0.9, 0.1],
                [0.7, 0.3],
                [0.4, 0.6],
                [0.8, 0.2],
                [0.3, 0.7],
                [0.1, 0.9],
            ]
        )
        result = compute_ovr_metrics(y_true, y_score, target_tpr=0.95)

        for class_result in result.per_class:
            first_index = int(np.flatnonzero(class_result.tpr >= 0.95)[0])
            self.assertEqual(
                class_result.fpr_at_target_tpr,
                class_result.fpr[first_index],
            )
            self.assertEqual(
                class_result.threshold_at_target_tpr,
                class_result.thresholds[first_index],
            )
        self.assertAlmostEqual(
            result.macro_fpr_at_target_tpr,
            float(
                np.mean(
                    [item.fpr_at_target_tpr for item in result.per_class]
                )
            ),
        )

    def test_cosine_similarity_scores(self) -> None:
        image_embeddings = np.array([[1.0, 0.0], [1.0, 1.0]])
        class_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        scores = compute_cosine_similarity_scores(
            image_embeddings,
            class_embeddings,
        )
        expected = np.array(
            [
                [1.0, 0.0],
                [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            ]
        )
        np.testing.assert_allclose(scores, expected)

    def test_undefined_class_raises(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.8, 0.1],
            ]
        )
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            compute_ovr_metrics(y_true, y_score)


if __name__ == "__main__":
    unittest.main()
