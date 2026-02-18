import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, graph_data, mask, title: str) -> None:
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

def plot_pca_latent(model, graph_data, title: str) -> None:
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

def plot_last_layer_weights(model, title: str) -> None:
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



# TODO: add further plotting functionalities: [Sankey, Subnetwork]