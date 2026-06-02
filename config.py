# wisdom/config.py
from dataclasses import dataclass, field
from typing import Dict, Optional, List

@dataclass
class AttributionConfig:
    methods: List[str] = field(default_factory=lambda: ["lrp"])  # e.g. ["lrp","ldl","lig"]
    device: str = "cuda:0"
    exclude_last_layer: bool = True
    fixed_weights: Optional[Dict[str, float]] = None  # optional voting override

@dataclass
class ClusteringConfig:
    method: str = "KMeans"         # KMeans suits DeepImportance
    params: Dict = field(default_factory=lambda: {"n_clusters": 2, "random_state": 42})
    use_silhouette: bool = False   # IDC-faithful per-neuron k selection (suitable if KMeans used)
    bo_search: bool = False        # Hook up BO

@dataclass
class WisdomConfig:
    top_m_neurons: int = 10
    test_all_classes: bool = True
    cache_path: Optional[str] = None
    
@dataclass
class WisdomTrainConfig:
    methods: List[str] = field(default_factory=lambda: ["lrp", "ldl", "lig"])
    device: str = "cuda:0"
    voting_weights: Optional[List[float]] = None
    voting_mode: str = "fine-grained"  # "fine-grained" | "coarse"
    pruning_augmentations: Optional[List[Dict]] = None
    out_csv: Optional[str] = None
