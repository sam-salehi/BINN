import os
import pandas as pd
from typing import Tuple, List

from data import ReactomeNetwork, PrepareAnnData, AnnDataToGraphData, get_map
from torch_geometric.transforms import RandomNodeSplit


class _SafeFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return _SafeFrame
    def __getitem__(self, key):
        if isinstance(key, set):
            key = list(key)
        return super().__getitem__(key)


def load_and_merge_tables(data_dir: str) -> pd.DataFrame: 
    cna_path = os.path.join(data_dir, "P1000_data_CNA_paper.csv")
    response_path = os.path.join(data_dir, "response_paper.csv")
    clinical_path = os.path.join(data_dir, "prad_p1000_clinical_final.txt")
    
    cna = pd.read_csv(cna_path).set_index("Unnamed: 0")
    cna["id"] = cna.index
    response = pd.read_csv(response_path)
    clinical = pd.read_csv(clinical_path, delimiter="\t")
    clinical["id"] = clinical["comp_id"]

    merged = cna.merge(response, how="inner", on="id").merge(clinical, how="inner", on="id")
    return _SafeFrame(merged)


def build_reactome_network(reactome_dir: str) -> ReactomeNetwork:
    relations_file_name = "ReactomePathwaysRelation.txt"
    pathway_names = "ReactomePathways.txt"
    pathway_genes = "ReactomePathways.gmt"
    reactome_net = ReactomeNetwork(
        reactome_dir,
        relations_file_name,
        pathway_names,
        pathway_genes
    )
    return reactome_net


def prepare_graph_data(
    merged_df: pd.DataFrame,
    obs_vars: List[str],
    reactome_net: ReactomeNetwork,
    data_dir: str,
):
    
    # print("Map resukt")
    ad_prep = PrepareAnnData(data=merged_df, obs_vars=obs_vars, map=get_map(reactome_net, n_levels=3))
    adata, map_df = ad_prep.anndata()
    demo = AnnDataToGraphData(
        data_dir,
        transform=RandomNodeSplit(split="random", num_train_per_class=200, num_val=200, num_test=314),
        group="response",
        adata=adata,
        knn=3,
    )
    data = demo.data
    return data, adata, map_df