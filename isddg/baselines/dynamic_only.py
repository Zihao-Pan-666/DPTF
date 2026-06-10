from __future__ import annotations

from isddg.models.backbone import FeatureBERT4Rec


def build_dynamic_only_model(role_features, cfg):
    m = cfg["model"]
    return FeatureBERT4Rec(
        item_features=role_features,
        hidden_dim=m["hidden_dim"],
        max_len=cfg["data"]["max_len"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        dropout=m["dropout"],
    )
