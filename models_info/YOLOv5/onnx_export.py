# Exports a pytorch *.pt model to *.onnx format
# Example usage (run from ./yolov5 directory):
# $ export PYTHONPATH="$PWD" && python models/onnx_export.py --weights ./weights/yolov5s.pt --img 640 --batch 1

import argparse
import os
import time
from pathlib import Path

import onnx
from .common import *

import torch

def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip()
    msg = weights + ' missing, try downloading from https://drive.google.com/drive/folders/1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J'

    r = 1
    if len(weights) > 0 and not os.path.isfile(weights):
        d = {'yolov3-spp.pt': '1mM67oNw4fZoIOL1c8M3hHmj66d8e-ni_',  # yolov3-spp.yaml
             'yolov5s.pt': '1R5T6rIyy3lLwgFXNms8whc-387H0tMQO',  # yolov5s.yaml
             'yolov5m.pt': '1vobuEExpWQVpXExsJ2w-Mbf3HJjWkQJr',  # yolov5m.yaml
             'yolov5l.pt': '1hrlqD1Wdei7UT4OgT785BEk1JwnSvNEV',  # yolov5l.yaml
             'yolov5x.pt': '1mM8aZJlWTxOg7BZJvNUMrTnA2AbeCVzS',  # yolov5x.yaml
             }

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)

        # Error check
        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.system('rm ' + weights)  # remove partial downloads
            raise Exception(msg)

def gdrive_download(id='1HaXkef9z6y5l4vUnCYgdmEAj61c6bfWO', name='coco.zip'):
    # https://gist.github.com/tanaikech/f0f2d122e05bf5f971611258c22c110f
    # Downloads a file from Google Drive, accepting presented query
    # from utils.google_utils import *; gdrive_download()
    t = time.time()

    print('Downloading https://drive.google.com/uc?export=download&id=%s as %s... ' % (id, name), end='')
    os.remove(name) if os.path.exists(name) else None  # remove existing
    os.remove('cookie') if os.path.exists('cookie') else None

    # Attempt file download
    os.system("curl -c ./cookie -s -L \"https://drive.google.com/uc?export=download&id=%s\" > /dev/null" % id)
    if os.path.exists('cookie'):  # large file
        s = "curl -Lb ./cookie \"https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=%s\" -o %s" % (
            id, name)
    else:  # small file
        s = "curl -s -L -o %s 'https://drive.google.com/uc?export=download&id=%s'" % (name, id)
    r = os.system(s)  # execute, capture return values
    os.remove('cookie') if os.path.exists('cookie') else None

    # Error check
    if r != 0:
        os.remove(name) if os.path.exists(name) else None  # remove partial
        print('Download error ')  # raise Exception('Download error')
        return r

    # Unzip if archive
    if name.endswith('.zip'):
        print('unzipping... ', end='')
        os.system('unzip -q %s' % name)  # unzip
        os.remove(name)  # remove zip to free space

    print('Done (%.1fs)' % (time.time() - t))
    return r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/yolov5s.pt', help='weights path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    print(opt)

    # Parameters
    f = opt.weights.replace('.pt', '.onnx')  # onnx filename
    img = torch.zeros((opt.batch_size, 3, opt.img_size, opt.img_size))  # image size, (1, 3, 320, 192) iDetection

    # Load pytorch model
    attempt_download(opt.weights)
    model = torch.load(opt.weights)['model']
    model.eval()
    # model.fuse()

    # Export to onnx
    model.model[-1].export = True  # set Detect() layer export=True
    torch.onnx.export(model, img, f, verbose=False, opset_version=11)

    # Check onnx model
    model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model)  # check onnx model
    print(onnx.helper.printable_graph(model.graph))  # print a human readable representation of the graph
    print('Export complete. ONNX model saved to %s\nView with https://github.com/lutzroeder/netron' % f)