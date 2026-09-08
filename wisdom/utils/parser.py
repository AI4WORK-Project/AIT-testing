# wisdom/utils/parser.py

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lenet', help='Model to use for training.')
    parser.add_argument('--saved-model', type=str, default='/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth', help='Saved model name.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'imagenet', 'synthetic'], help='The dataset to use for training and testing.')
    parser.add_argument('--data-path', type=str, default='./datasets/', help='Path to the data directory.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training.')
    parser.add_argument('--random-prune', action='store_true', help='Randomly prune the neurons.')
    parser.add_argument('--use-silhouette', action='store_true', help='Whether to use silhouette score for clustering.')
    parser.add_argument('--n-clusters', type=int, default=2, help='Number of clusters to use for KMeans.')
    parser.add_argument('--top-m-neurons', type=int, default=5, help='Number of top neurons to select.')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training.')

    # Testing Mode arguments
    parser.add_argument('--test-image', type=str, default='1', help='Test image name. For the single image testing. (against with the `all-class`).')
    parser.add_argument('--all-class', action='store_true', help='Attributions collected for all the classes. When activated, it will equal to batch testing.')
    parser.add_argument('--class-iters', action='store_true', help='Only valided when doing class-wise testing. If set, the model will be tested for each class separately.')
    parser.add_argument('--idc-test-all', action='store_true', help='Using all the test images for the Coverage testing. Other wise will only sample some images from the test set.')
    parser.add_argument('--num-samples', type=int, default=0, help='Sampling number for the test images (against with the `idc-test-all`).')
    parser.add_argument('--attr', type=str, default='lc', choices=['lc', 'la', 'ii', 'lgxa', 'lgc', 'ldl', 'ldls', 'lgs', 'lig', 'lfa', 'lrp', 'random', 'wisdom'],  help='The attribution method to use.')
    parser.add_argument('--layer-index', type=int, default=1, help='Get the layer index for the model, should start with 1')

    
    # General arguments
    # parser.add_argument('--vis-attributions', action='store_true', help='Visualize the attributions.')
    # parser.add_argument('--viz', action='store_true', help='Visualize the input and its relevance.')
    parser.add_argument('--logging', action="store_true", help="Whether to log the training process")
    parser.add_argument('--log-path', type=str, default='./logs/TestLog', help='Path (and name) to save the log file.')
    parser.add_argument('--inordered-dataset', action='store_true', help='Whether the dataset is ordered.')
    parser.add_argument('--csv-file', type=str, default='demo_layer_scores.csv', help='The file to save the layer scores.')

    args = parser.parse_args()
    # print(args)
    
    return args