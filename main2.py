"""
Main entry point for training ANN, GCN, and GAT models using scvi-tools BaseModelClass.
This script trains all three models and compares their performance on the test set.
"""

import os
import torch
from data_utils import load_and_merge_tables, build_reactome_network, prepare_graph_data
from training_scvi import train_ann_model, train_gcn_model, train_gat_model


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
    data_dir = "data"
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
    
    # Setup device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Training parameters
    epochs = 400
    patience = 200
    lr = 1e-3
    weight_decay = 1e-5
    
    num_node_features = data.num_node_features
    num_classes = int(len(data.y.unique()))
    
    # Dictionary to store results
    results = {}
    
    # Train ANN model
    print("\n" + "="*60)
    print("Training ANN model...")
    print("="*60)
    ann_model = train_ann_model(
        graph_data=data,
        map_df=map_df,
        num_node_features=num_node_features,
        num_classes=num_classes,
        device=device,
        epochs=epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        bias=False,
        save_path="./data/weights/ANN_scvi.pt",
        adata=adata,
    )
    results['ANN'] = ann_model
    
    # Train GCN model
    print("\n" + "="*60)
    print("Training GCN model...")
    print("="*60)
    gcn_model = train_gcn_model(
        graph_data=data,
        map_df=map_df,
        num_node_features=num_node_features,
        num_classes=num_classes,
        device=device,
        epochs=epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        bias=False,
        save_path="./data/weights/GCN_scvi.pt",
        adata=adata,
    )
    results['GCN'] = gcn_model
    
    # Train GAT model
    print("\n" + "="*60)
    print("Training GAT model...")
    print("="*60)
    gat_model = train_gat_model(
        graph_data=data,
        map_df=map_df,
        num_node_features=num_node_features,
        num_classes=num_classes,
        device=device,
        epochs=epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        bias=False,
        heads=1,
        save_path="./data/weights/GAT_scvi.pt",
        adata=adata,
    )
    results['GAT'] = gat_model
    
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
    
    print("\nDone!")


if __name__ == "__main__":
    main()
