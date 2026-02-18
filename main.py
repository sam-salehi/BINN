"""
Main entry point for training ANN, GCN, and GAT models using scvi-tools BaseModelClass.
This script trains all three models and compares their performance on the test set.
"""
import os
import matplotlib.pyplot as plt

from data_utils import load_and_merge_tables, build_reactome_network, prepare_graph_data
from learn import Hyperparameters
from training_scvi import train_graph_model
from learn import plot_confusion_matrix, plot_pca_latent, plot_last_layer_weights


def evaluate_model(model, graph_data, mask):
    """Evaluate model accuracy on a given mask."""
    predictions = model.predict(mask=mask)
    pred_classes = predictions.argmax(dim=1)
    true_labels = graph_data.y[mask]
    correct = (pred_classes == true_labels).sum().item()
    total = mask.sum().item()
    accuracy = correct / total
    return accuracy

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
            plot_confusion_matrix(model, data, test_mask, title=f"{model_name} - Confusion matrix (test)")
            plt.show()

    for model_name, model in results.items():
        print(f"[{model_name}] PCA of latent representation")
        plot_pca_latent(model, data, title=f"{model_name} - PCA (latent)")
        plt.show()

    for model_name, model in results.items():
        print(f"[{model_name}] Final layer weights")
        plot_last_layer_weights(model, title=model_name)
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
