import os
import csv
from typing import Dict
import torch
from torch_geometric.explain import Explainer, GNNExplainer


class _ExplainWrapper(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, return_type: str = "log_probs"):
        super().__init__()
        self.base = base
        self.return_type = return_type
    def forward(self, x_in, edge_index_in):
        out = self.base(x_in, edge_index_in)
        if isinstance(out, tuple):
            out = out[0]
        if self.return_type == "log_probs":
            return torch.log_softmax(out, dim=-1)
        return out


def explain_and_save(
    models: Dict[str, torch.nn.Module],
    data,
    device: torch.device,
    outputs_root: str = "./outputs",
    return_type: str = "log_probs",
    epochs: int = 200,
    max_nodes: int = None,
) -> None:
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    num_nodes = x.size(0)
    if max_nodes is not None:
        num_nodes = min(num_nodes, max_nodes)

    for method, model in models.items():
        wrapped = _ExplainWrapper(model, return_type=return_type).to(device).eval()
        for i in range(num_nodes):
            explainer = Explainer(
                model=wrapped,
                algorithm=GNNExplainer(epochs=epochs).to(device),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode='multiclass_classification',
                    task_level='node',
                    return_type=return_type,
                ),
            )
            explanation = explainer(x, edge_index, index=i)

            # Ensure output dirs
            out_dir = os.path.join(outputs_root, method)
            os.makedirs(out_dir, exist_ok=True)

            # Visualizations
            feat_png = os.path.join(out_dir, f"feature_importance_{i}.png")
            explanation.visualize_feature_importance(feat_png, top_k=10)

            subgraph_pdf = os.path.join(out_dir, f"subgraph_{i}.pdf")
            explanation.visualize_graph(subgraph_pdf)

            # CSV
            node_importance = explanation.node_mask.tolist()
            edge_importance = explanation.edge_mask.tolist()

            sorted_node = sorted(enumerate(node_importance), key=lambda x: x[1], reverse=True)
            node_csv = os.path.join(out_dir, f"node_importance_{i}.csv")
            with open(node_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "importance"])
                writer.writerows(sorted_node)

            sorted_edge = sorted(enumerate(edge_importance), key=lambda x: x[1], reverse=True)
            edge_csv = os.path.join(out_dir, f"edge_importance_{i}.csv")
            with open(edge_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "importance"])
                writer.writerows(sorted_edge)