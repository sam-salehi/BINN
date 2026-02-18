"""
Main entry point for training ANN, GCN, and GAT models using scvi-tools BaseModelClass.
This script trains all three models and compares their performance on the test set.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from data_utils import load_and_merge_tables, build_reactome_network, prepare_graph_data
from nn import Hyperparameters
from training_scvi import train_graph_model


def evaluate_model(model, graph_data, mask):
    """Evaluate model accuracy on a given mask."""
    predictions = model.predict(mask=mask)
    pred_classes = predictions.argmax(dim=1)
    true_labels = graph_data.y[mask]
    correct = (pred_classes == true_labels).sum().item()
    total = mask.sum().item()
    accuracy = correct / total
    return accuracy

def _plot_confusion_matrix(model, graph_data, mask, title: str) -> None:
    """Plot confusion matrix on the given mask."""
    preds = model.predict(mask=mask).argmax(dim=1).detach().cpu().numpy()
    y_true = graph_data.y[mask].detach().cpu().numpy()
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

def _plot_pca_latent(model, graph_data, title: str) -> None:
    """PCA of latent representations from model.get_latent_representation()."""
    latent = model.get_latent_representation()
    X = latent.detach().cpu().numpy()
    y = graph_data.y.detach().cpu().numpy()
    X2 = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(5, 4))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap="tab10", s=10, alpha=0.9, linewidths=0)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

def _plot_last_layer_weights(model, title: str) -> None:
    """Plot heatmap of final linear layer weights if present."""
    import torch.nn as nn
    enc = model.module.encoder
    last_linear = None
    # enc.layers is nn.Sequential of blocks; find the last nn.Linear encountered
    for block in enc.layers:
        for layer in block:
            if isinstance(layer, nn.Linear):
                last_linear = layer
    if last_linear is None:
        print(f"[warn] No final Linear layer found for {title}, skipping weight plot.")
        return
    w = last_linear.weight.detach().cpu().numpy()
    plt.figure(figsize=(6, 4))
    sns.heatmap(w, cmap="coolwarm", center=0, cbar=True)
    plt.title(f"{title} - Last layer weights")
    plt.xlabel("Input features")
    plt.ylabel("Classes")
    plt.tight_layout()


def main() -> None:
    """Main function to train and compare ANN, GCN, and GAT models."""
    # Data directory setup
    data_dir = "train_data"
    reactome_dir = os.path.join(data_dir, "reactome")
    
    # Define clinical variables and observation variables
    clin_vars = [
        'Purity', 
        'Ploidy', 
        'Tumor.Coverage', 
        'Normal.Coverage', 
        'Mutation.burden', 
        'Fraction.genome.altered', 
        'Mutation_count'
    ]
    obs_vars = clin_vars.copy()
    obs_vars.append('response')
    
    # Load and prepare data
    print("Loading and preparing data...")
    merged = load_and_merge_tables(data_dir)
    reactome_net = build_reactome_network(reactome_dir)
    data, adata, map_df = prepare_graph_data(merged, obs_vars, reactome_net, data_dir)
    
    print(f"Graph data: {data}")
    print(f"Number of nodes: {data.x.shape[0]}")
    print(f"Number of features: {data.x.shape[1]}")
    print(f"Number of edges: {data.edge_index.shape[1]}")
    print(f"Number of classes: {len(data.y.unique())}")
    print(f"Pathway mapping shape: {map_df.shape}")
    
    config = Hyperparameters(
        num_node_features=data.num_node_features,
        num_classes=int(len(data.y.unique())),
        lr=1e-3,
        w_decay=1e-5,
        epochs=400,
        patience=200,
        bias=False,
        heads=1,
        save_dir="./train_data/weights",
    )
    print(f"Using device: {config.device}")

    results = {}
    for name in ["ANN", "GCN", "GAT"]:
        print("\n" + "="*60)
        print(f"Training {name} model...")
        print("="*60)
        results[name] = train_graph_model(name, data, map_df, config, adata=adata)
    
    # Compare performances on test set
    print("\n" + "="*60)
    print("Model Performance Comparison (Test Set)")
    print("="*60)
    
    if hasattr(data, 'test_mask'):
        for model_name, model in results.items():
            accuracy = evaluate_model(model, data, data.test_mask)
            print(f"{model_name}: Test Accuracy = {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Find best model
        accuracies = {name: evaluate_model(model, data, data.test_mask) 
                      for name, model in results.items()}
        best_model = max(accuracies, key=accuracies.get)
        print(f"\nBest performing model: {best_model} with {accuracies[best_model]*100:.2f}% accuracy")
    else:
        print("Warning: No test_mask found in data. Cannot evaluate on test set.")
    
  
    print("\n" + "="*60)
    print("Interpretability visualizations")
    print("="*60)
    if hasattr(data, 'test_mask'):
        test_mask = data.test_mask
        for model_name, model in results.items():
            print(f"\n[{model_name}] Confusion matrix (test)")
            _plot_confusion_matrix(model, data, test_mask, title=f"{model_name} - Confusion matrix (test)")
            plt.show()

    for model_name, model in results.items():
        print(f"[{model_name}] PCA of latent representation")
        _plot_pca_latent(model, data, title=f"{model_name} - PCA (latent)")
        plt.show()

    for model_name, model in results.items():
        print(f"[{model_name}] Final layer weights")
        _plot_last_layer_weights(model, title=model_name)
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
