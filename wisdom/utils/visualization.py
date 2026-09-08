# wisdom/utils/visualization.py
from pathlib import Path

from captum.attr import visualization as viz
import numpy as np
import torch
import pandas as pd

def viz_attr(img, attr, dataset_name, model_name, with_original=False):
    import matplotlib.pyplot as plt

    attr_np = attr.cpu().detach().numpy()
    img_np  = img.cpu().detach().numpy()
    
    if with_original:
        fig, ax = viz.visualize_image_attr_multiple(np.transpose(attr_np, (1, 2, 0)),
                                        np.transpose(img_np, (1, 2, 0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        show_colorbar=True,
                                        outlier_perc=2,
                                        use_pyplot=False)
    else:
        fig, ax = viz.visualize_image_attr(np.transpose(attr_np, (1, 2, 0)),
                                        np.transpose(img_np, (1, 2, 0)),
                                        "heat_map",
                                        "positive",
                                        show_colorbar=True,
                                        outlier_perc=2,
                                        use_pyplot=False)

    fig.tight_layout()
    fig.savefig(f"{dataset_name}_{model_name}.png",
                dpi=300,
                bbox_inches="tight")
    plt.close(fig)

def viz_topk_neurons_score(csv_file, top_k=10, output_path=None):
    import matplotlib.pyplot as plt

    destination = Path(output_path or f"top_{top_k}_neuron_scores.pdf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(csv_file).nlargest(top_k, "Score")
    figure, axis = plt.subplots(figsize=(10, 6))
    for layer_name, group in frame.groupby("LayerName"):
        axis.scatter(group["NeuronIndex"], group["Score"], label=layer_name, alpha=0.7, s=70)
    axis.set_xlabel("Neuron Index")
    axis.set_ylabel("Importance Score")
    axis.set_title(f"Top-{top_k} Neuron Scores Across All Layers")
    axis.legend()
    figure.savefig(destination, format="pdf", dpi=1200, bbox_inches="tight")
    plt.close(figure)
    return str(destination)
