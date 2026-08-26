from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier


class TorchRandomForest(nn.Module):
    def __init__(
        self,
        *,
        features: Tensor,
        thresholds: Tensor,
        left_children: Tensor,
        right_children: Tensor,
        probabilities: Tensor,
        max_depth: int,
    ) -> None:
        super().__init__()
        self.register_buffer("features", features.to(dtype=torch.int64))
        self.register_buffer("thresholds", thresholds.to(dtype=torch.float64))
        self.register_buffer("left_children", left_children.to(dtype=torch.int64))
        self.register_buffer("right_children", right_children.to(dtype=torch.int64))
        self.register_buffer("probabilities", probabilities.to(dtype=torch.float64))
        self.max_depth = max_depth

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs.to(dtype=torch.float64)
        if inputs.ndim != 2:
            raise ValueError("inputs must have shape [batch, features]")
        batch_size = inputs.shape[0]
        tree_count = self.features.shape[0]
        tree_indices = torch.arange(tree_count, device=inputs.device).expand(batch_size, -1)
        nodes = torch.zeros((batch_size, tree_count), dtype=torch.int64, device=inputs.device)
        for _ in range(self.max_depth):
            split_features = self.features[tree_indices, nodes]
            leaves = split_features < 0
            safe_features = torch.clamp(split_features, min=0)
            values = torch.gather(inputs, 1, safe_features)
            thresholds = self.thresholds[tree_indices, nodes]
            next_nodes = torch.where(
                values <= thresholds,
                self.left_children[tree_indices, nodes],
                self.right_children[tree_indices, nodes],
            )
            nodes = torch.where(leaves, nodes, next_nodes)
        return self.probabilities[tree_indices, nodes].mean(dim=1)

    def to_payload(self) -> dict[str, Any]:
        return {
            "features": self.features.cpu(),
            "thresholds": self.thresholds.cpu(),
            "left_children": self.left_children.cpu(),
            "right_children": self.right_children.cpu(),
            "probabilities": self.probabilities.cpu(),
            "max_depth": self.max_depth,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TorchRandomForest:
        return cls(
            features=payload["features"],
            thresholds=payload["thresholds"],
            left_children=payload["left_children"],
            right_children=payload["right_children"],
            probabilities=payload["probabilities"],
            max_depth=int(payload["max_depth"]),
        )


def from_sklearn(model: RandomForestClassifier) -> TorchRandomForest:
    tree_count = len(model.estimators_)
    class_count = len(model.classes_)
    max_nodes = max(estimator.tree_.node_count for estimator in model.estimators_)
    max_depth = max(estimator.tree_.max_depth for estimator in model.estimators_) + 1

    features = np.full((tree_count, max_nodes), -2, dtype=np.int64)
    thresholds = np.zeros((tree_count, max_nodes), dtype=np.float64)
    left_children = np.zeros((tree_count, max_nodes), dtype=np.int64)
    right_children = np.zeros((tree_count, max_nodes), dtype=np.int64)
    probabilities = np.zeros((tree_count, max_nodes, class_count), dtype=np.float64)

    for tree_index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        count = tree.node_count
        features[tree_index, :count] = tree.feature
        thresholds[tree_index, :count] = tree.threshold
        left_children[tree_index, :count] = tree.children_left
        right_children[tree_index, :count] = tree.children_right
        values = tree.value[:, 0, :]
        totals = values.sum(axis=1, keepdims=True)
        probabilities[tree_index, :count] = np.divide(
            values,
            totals,
            out=np.zeros_like(values),
            where=totals != 0,
        )

    return TorchRandomForest(
        features=torch.from_numpy(features),
        thresholds=torch.from_numpy(thresholds),
        left_children=torch.from_numpy(left_children),
        right_children=torch.from_numpy(right_children),
        probabilities=torch.from_numpy(probabilities),
        max_depth=max_depth,
    )
