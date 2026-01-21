"""
Main entry point for training ANN models using scvi-tools BaseModelClass.
This script demonstrates how to use the new PyTorch Lightning-based training.
"""

import os
import torch
import time
from data_utils import load_and_merge_tables, build_reactome_network, prepare_graph_data
from training_scvi import train_ann_model


def main() -> None:
    """Main function to train ANN model with scvi-tools integration."""
    # Data directory setup
    data_dir = "data"
    reactome_dir = os.path.join(data_dir, "reactome")
    outputs_root = "./outputs"
    
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
    
    # Train the ANN model
    start = time.perf_counter()
    
    model = train_ann_model(
        graph_data=data,
        map_df=map_df,
        num_node_features=data.num_node_features,
        num_classes=int(len(data.y.unique())),
        device=device,
        epochs=epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        bias=False,
        save_path="./data/weights/ANN_scvi.pt",
        adata=adata,  # Pass adata for scvi-tools BaseModelClass
    )
    
    end = time.perf_counter()
    print(f"\nTraining completed in {end - start:.2f} seconds")
    
    # Example: Get predictions on test set
    if hasattr(data, 'test_mask'):
        print("\nGetting predictions on test set...")
        predictions = model.predict(mask=data.test_mask)
        print(f"Predictions shape: {predictions.shape}")
    
    # Example: Get latent representations
    print("\nGetting latent representations...")
    latent = model.get_latent_representation()
    print(f"Latent representation shape: {latent.shape}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

