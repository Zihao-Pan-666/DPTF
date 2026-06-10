from pathlib import Path
import torch
from isddg.models.backbone import FeatureBERT4Rec
from isddg.models.isddg import ISDDGModel


def test_forward_shapes():
    item_features = torch.randn(11, 8)
    role = torch.softmax(torch.randn(11, 4), dim=-1)
    role[0] = 0
    backbone = FeatureBERT4Rec(item_features, hidden_dim=16, max_len=5, role_features=role, role_alpha=0.2)
    proto_k = torch.randn(6, 16)
    proto_v = torch.softmax(torch.randn(6, 4), dim=-1)
    model = ISDDGModel(backbone, role, proto_k, proto_v)
    hist = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 5, 6, 7]])
    cand = torch.tensor([[4, 5, 6], [7, 8, 9]])
    out = model.score(hist, cand)
    assert out.shape == (2, 3)


if __name__ == "__main__":
    test_forward_shapes()
    print("smoke test passed")
