

import os
import tempfile
from collections.abc import Sequence

import numpy as np
import scvi
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalObsField,
    LayerField,
    ObsmField,
    NumericalObsField,
)
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.module import VAE




from encoder import ANN





# in init. Set module attribute with model
# add model summary string as representatioln for model
# run self.init_params 


class Model(BaseModelClass):
    def __init__(self,adata,map,args):
        super().__init__(adata)
        self.module = ANN(map,args) # makes sense for most part. 
        self._model_summary_string = "An Artificial Neural Network"

        self._init_parms = self._get_init_params(locals())
 

    def train(self):
        pass 
    

    @classmethod
    def setup_anndata(
        cls,
        adata,
        obs_keys=None,
        label_key="response",
        node_feature_key="node_features",
        adjacency_key="adjacency",
        layer=None,
    ):
        """Register AnnData fields for this GNN/ANN model."""

        fields = []

        # Main data matrix
        fields.append(LayerField(REGISTRY_KEYS.X_KEY, layer=layer))

        # Labels
        fields.append(
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, key=label_key)
        )

        # Clinical numerical covariates
        if obs_keys is not None:
            for k in obs_keys:
                if k != label_key:
                    fields.append(NumericalObsField(k, key=k))

        # Node features (obsm)
        fields.append(
            ObsmField("node_features", key=node_feature_key)
        )

        # Adjacency matrix (obsp)
        fields.append(
            ObsmField("adjacency", key=adjacency_key)
        )

        # Build and register the manager
        adata_manager = AnnDataManager(fields=fields)
        adata_manager.register_fields(adata)
        cls.register_manager(adata_manager)

        return adata_manager
