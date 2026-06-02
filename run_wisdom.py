# run_wisdom.py
import argparse
import os
import time
import json
import logging

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST

from wisdom.core.wisdom import WisdomIDC
from wisdom.core.wisdom import WisdomConfig, ClusteringConfig
from wisdom.core.wisdom_train import ConsensusWisdom, WisdomTrainConfig

from wisdom.utils.io_cache import read_layer_scores_csv
from wisdom.utils.common import get_trainable_modules_main, get_model
from wisdom.utils.visulization import viz_topk_neurons_score

"""
Running example:
python run_wisdom.py \ 
    --impl wisdom \
    --model-name lenet \
    --dataset cifar10 \
    --data-path /path/to/datasets \
    --device cuda:0 \
    --top-m-neurons 10 \
    --batch-size 64 \
    --end2end \
    --all-class \
    --csv-file ./saved_files/pre_csv/lenet_cifar10.csv \
    --model-path ./models_info/saved_models/lenet_CIFAR10_whole.pth

python run_wisdom.py --impl wisdom --model-name lenet --dataset cifar10 --data-path /data/shenghao/dataset --device cuda:0 --top-m-neurons 10 --batch-size 64 --end2end --all-class --csv-file ./saved_files/pre_csv/lenet_cifar10.csv --model-path ./models_info/saved_models/lenet_CIFAR10_whole.pth

"""

def get_model(load_model_path='./models_info/saved_models/lenet_CIFAR10_whole.pth'):
    module_name = []
    module = []
    model = torch.load(load_model_path, weights_only=False)
    
    # Alternatively, to get all submodule names (including nested ones)
    for name, layer in model.named_modules():
        module_name.append(name)
        module.append(layer)

    return model, module_name, module

def get_trainable_modules_main(model, prefix=''):
    
    trainable_module = []
    trainable_module_name = []
    
    def get_trainable_modules(model, prefix=''):
        for name, layer in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)) and any(p.requires_grad for p in layer.parameters()):
                trainable_module_name.append(full_name)
                trainable_module.append(layer)
            get_trainable_modules(layer, full_name)
    get_trainable_modules(model)
    return trainable_module, trainable_module_name

# Small configuration
def configure_logging(level='info', enable_logging=False):
    if not enable_logging:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        return logging.getLogger(__name__)
    log_level = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "crit": logging.CRITICAL,
        }.get(level.lower(), logging.INFO)
        
    start_ms = int(time.time() * 1000)
    timestamp = time.strftime("%Y%m%d‑%H%M%S", time.localtime(start_ms / 1000))
    logfile = f"debugmode-{timestamp}.log"

    logger = logging.getLogger("Wisdom")
    logger.setLevel(log_level)
        
    handler = logging.FileHandler(logfile)
    formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# Decide which testing mode is active
def select_testing_mode(args) -> dict:
    # Return a dictionary with boolean values for each mode
    testing_mode =  {
        'end2end': bool(args.end2end),
        'all_class': bool(args.all_class),
        'class_iters': bool(args.class_iters)
    }
    
    # Build list of active modes with alternative descriptions for False cases
    mode_descriptions = []
    if testing_mode['end2end']:
        mode_descriptions.append('End2End-Testing')
    else:
        mode_descriptions.append('Single-Layer-Testing')
        
    if testing_mode['all_class']:
        mode_descriptions.append('All-Class-Testing')
    else:
        mode_descriptions.append('Class-Wise-Testing')
        
    if testing_mode['class_iters']:
        mode_descriptions.append('Iterating-All-Class: On')
    else:
        mode_descriptions.append('Iterating-All-Class: Off')
        
    return testing_mode, mode_descriptions

#------------
# Loading datasets
#------------
#  Load the CIFAR-10 dataset
def load_CIFAR(batch_size=32, root='./datasets', shuffle=True):

    transform = transforms.Compose([
         transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader


#  Load the MNIST dataset
def load_MNIST(batch_size=32, root='./datasets', channel_first=False, train_all=False):
    # transform_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    
    transform_list = [
        transforms.Resize(32),  # Upscale from 28x28 to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]

    if channel_first:
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))  # If you want 3 channels
    transform = transforms.Compose(transform_list)

    train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)
    
    if train_all:
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def wisdom_end2end(args, model, final_layer, trainloader, testloader, logger):
    results_dict = {}
    if os.path.exists(args.csv_file):
        layer_scores = read_layer_scores_csv(args.csv_file)
        viz_topk_neurons_score(args.csv_file, top_k=args.top_m_neurons)
        logger.info("Layer scores loaded from %s", args.csv_file)
    else:
        logger.info("Training ConsensusWisdom to get layer scores...")
        trainer = ConsensusWisdom(model, device=args.device)
        cfg = WisdomTrainConfig(
            methods=["lrp","ldl","lig"], # 3 options as examples here
            device=args.device,
            voting_mode="fine-grained",  # or "coarse"
            out_csv="wisdom_layer_scores_demo.csv",
        )
        
        layer_scores, out_csv = trainer.fit(
            train_loader=trainloader,
            cfg=cfg,
            top_m_neurons=10,
            final_layer=final_layer,        # or "fc" / your last trainable layer name to exclude
            prune_mode="mask",       # "mask" (reversible) or "weights" (restored per batch)
        )

    cluster = ClusteringConfig(method="KMeans", params={"random_state": 42, "n_clusters": args.n_clusters}, use_silhouette=args.use_silhouette, k_max=10)
    cfg = WisdomConfig(top_m_neurons=args.top_m_neurons, test_all_classes=args.all_class, cache_path=".wisdom_cache")
    idc = WisdomIDC(model, impl=args.impl, cfg=cfg, cluster=cluster)
    selected = idc.select_top_neurons(layer_scores, exclude_last=final_layer)
    idc.fit_clusters(trainloader, selected, device=args.device)
    coverage_rate, total_combination, max_coverage = idc.coverage(testloader, selected, device=args.device)
    logger.info("Attribution Method: %s", "WISDOM")
    logger.info("Total coverage combinations: %d", total_combination)
    logger.info("Max Coverage (the best we can achieve): %.6f%%", max_coverage * 100)
    logger.info("[WISDOM] Coverage Rate: %.6f%%", coverage_rate * 100)

    results_dict = {
        'Model': args.model_name,
        'Dataset': args.dataset,
        'TopK Neurons': args.top_m_neurons,
        'Attribution Method': args.impl,
        'End2End Testing': args.end2end,
        'All-Class Testing': args.all_class,
        'Class-Iters Testing': args.class_iters,
        'Total Combination': total_combination,
        'Max Coverage': max_coverage,
        'Coverage Rate': coverage_rate
    }
    json_filename = f"coverage_results_{args.model_name}_{args.dataset}.json"
    with open(json_filename, 'w') as f:
        json.dump(results_dict, f, indent=4)


def run(args):
    if args.dataset == 'mnist':
        trainloader, testloader = load_MNIST(batch_size=args.batch_size, root=args.data_path)
    elif args.dataset == 'cifar10':
        trainloader, testloader = load_CIFAR(batch_size=args.batch_size, root=args.data_path)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    logger = configure_logging()
    
    model, module_name, module = get_model(args.model_path)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)
    final_layer = trainable_module_name[-1]
    testing_mode, mode_descriptions = select_testing_mode(args)
    logger.info("Model: %s, Dataset: %s, Topk: %s, Testing Mode: [%s]", args.model_name, args.dataset, args.top_m_neurons, ', '.join(mode_descriptions))
    model.eval()
    wisdom_end2end(args, model, final_layer, trainloader, testloader, logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("WISDOM end2end coverage demo")
    parser.add_argument("--impl", choices=["deepidc","wisdom"], default="deepidc",
                    help="deepidc=KMeans+Silhouette | wisdom=pluggable (default MeanShift)")
    parser.add_argument("--model-name", default="resnet18", help="resnet18 | vgg16 (extend as needed)")
    parser.add_argument("--model-path", default=None, help="path to state_dict .pth (optional)")
    parser.add_argument("--dataset", choices=["cifar10","mnist","imagenet"], required=True)
    parser.add_argument("--data-path", required=True, help="dataset root folder")

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument('--use-silhouette', action='store_true', help='Whether to use silhouette score for clustering.')
    parser.add_argument('--n-clusters', type=int, default=2, help='Number of clusters to use for KMeans.')
    parser.add_argument('--top-m-neurons', type=int, default=5, help='Number of top neurons to select.')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training.')

    parser.add_argument('--all-class', action='store_true', help='Attributions collected for all the classes. When activated, it will equal to batch testing.')
    parser.add_argument('--class-iters', action='store_true', help='Only valided when doing class-wise testing. If set, the model will be tested for each class separately.')
    parser.add_argument('--end2end', action='store_true', help='End to end testing for the whole model.')

    parser.add_argument('--logging', action="store_true", help="Whether to log the training process")
    parser.add_argument('--log-path', type=str, default='./logs/TestLog', help='Path (and name) to save the log file.')
    parser.add_argument('--csv-file', type=str, default='demo_layer_scores.csv', help='The file to save the layer scores.')
    args = parser.parse_args()
    run(args)