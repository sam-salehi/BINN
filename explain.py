


import bruno.nn.modules as modules
from bruno.nn.modules import Encoder
import torch
import numpy as np
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCNConv
import csv
from bruno.data import AnnDataToGraphData
from bruno.data import PrepareAnnData, get_map


def write_tensor_to_file(file, tensor):
    """
    Writes a 2D PyTorch tensor to a file object in CSV format.
    
    Parameters:
        file (file object): The file object to write to (must be opened in write mode).
        tensor (torch.Tensor): A 2D PyTorch tensor to write to the file.
    """
    if len(tensor.shape) != 2:
        raise ValueError("The tensor must be 2D.")

    for row in tensor:
        line = ",".join(map(str, row.tolist()))
        file.write(line + "\n")



gcn = torch.load("data/GCN.pt",weights_only=False)
gat = torch.load("data/GAT.pt",weights_only=False)
torch.manual_seed(123)
np.random.seed(123)



gcn.args.simple = True
gat.args.simple = True

obs_vars = ['Purity', 'Ploidy', 'Tumor.Coverage', 'Normal.Coverage', 'Mutation.burden', 'Fraction.genome.altered', 'Mutation_count']
obs_vars.append("Response")

prepareData = PrepareAnnData(data = all, obs_vars=obs_vars, map = get_map(reactome_net, n_levels=3))
adata, map = prepareData.anndata()
demo = AnnDataToGraphData("data",
                            transform=RandomNodeSplit(split="random", num_train_per_class = 200, num_val=200, num_test=314), 
                            group = 'response',
                            adata = adata,
                            knn = 3)
data = demo.data

models = [gcn, gat]
for model in models:
    for i in range(914):
        explainer = Explainer(
            model=gat,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
            ),
        )
        node_index = i
        explanation = explainer(data.x, data.edge_index, index=node_index)
        print(f'Generated explanations in {explanation.available_explanations}')

        path = './outputs/{}/feature_importance_{}.png'.format(model.args.method, i)
        explanation.visualize_feature_importance(path, top_k=10)
        print(f"Feature importance plot has been saved to '{path}'")

        path = './outputs/{}/subgraph_{}.pdf'.format(model.args.method, i)
        explanation.visualize_graph(path)
        print(f"Subgraph visualization plot has been saved to '{path}'")

        # Save node and edge importance to CSV
        node_importance = explanation.node_mask.tolist()  # Convert to list
        edge_importance = explanation.edge_mask.tolist()  # Convert to list

        # Sort node importance in descending order
        sorted_node_importance = sorted(
            enumerate(node_importance), key=lambda x: x[1], reverse=True
        )
        with open('./outputs/{}/node_importance_{}.csv'.format(model.args.method, i), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'importance'])
            writer.writerows(sorted_node_importance)

        # Sort edge importance in descending order
        sorted_edge_importance = sorted(
            enumerate(edge_importance), key=lambda x: x[1], reverse=True
        )
        with open('./outputs/{}/edge_importance_{}.csv'.format(model.args.method, i), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'importance'])
            writer.writerows(sorted_edge_importance)

