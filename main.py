
import os
import torch
from data_utils import load_and_merge_tables, build_reactome_network, prepare_graph_data
from training import train_models
from explain_runner import explain_and_save
from bruno.nn.modules import Encoder
from bruno.learn import Hyperparameters
import pandas as pd 
from encoder import ANN, GAT, GCN
import time 

from learn import TrainModel as MyTrainModel

  

def main() -> None:
    data_dir = "data"
    reactome_dir = os.path.join(data_dir, "reactome")
    outputs_root = "./outputs"

    clin_vars = ['Purity', 'Ploidy', 'Tumor.Coverage', 'Normal.Coverage', 'Mutation.burden', 'Fraction.genome.altered', 'Mutation_count']
    obs_vars = clin_vars.copy()
    obs_vars.append('response')

    merged = load_and_merge_tables(data_dir)
    reactome_net = build_reactome_network(reactome_dir)
    data, adata, map_df = prepare_graph_data(merged, obs_vars, reactome_net, data_dir)


    print(map_df)



    # s = map_df.apply(lambda x: pd.factorize(x)[0])
    # print(s)

    # quit()
    # print(reactome_net.info())
    # print(data)
    # print(adata)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # args = Hyperparameters()
    # args.epochs = 2000
    # args.num_node_features = data.num_node_features
    # args.num_classes = int(len(data.y.unique()))
    # args.cuda = (device.type == "cuda")
    # args.device = device
    # args.method = "GATConv"
    # model = GAT(map_df,args=args,bias=False)
    # train_ann = MyTrainModel(model=model,graph=data,args=args)
    # train_ann.learn()
    # train_ann.plot_loss()
    # # print(train_ann.metrics())
    # # train_ann.plot_loss()
    # quit()

    start = time.perf_counter()

    models, trainers= train_models(data, map_df, device=device, epochs=400, patience=200, save_dir="./data/weights") 

    end = time.perf_counter()
    print(f"Execution time: {end - start:.6f} seconds")


    # import matplotlib.pyplot as plt
    # for m in ["ANN","GCNConv", "GATConv"]:
    #     trainer = trainers[m]
    #     trainer.metrics()
    #     plt.show()
    #     trainer.plot_loss()
    #     plt.show()
    #     trainer.plot_pca()
    #     plt.show()

    #     # trainer.plot_tsne() FIXME: segfaults.

    #     trainer.plot_weights_of_last_layer()
    #     # trainer.plot_subnetwork()
    #     plt.show()


    # print("Generating explanation")

    # explain_and_save(models, data, device=device, outputs_root=outputs_root, return_type="log_probs", epochs=200, max_nodes=None)

if __name__ == "__main__":
    main()


