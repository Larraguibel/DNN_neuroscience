import torch
import torchvision

import argparse


if "__main__" == __name__:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store", help="Model to use")
    parser.add_argument("--dataset", type=int, help="", default=10)
    results = parser.parse_args()


    model = results.model
    dataset = results.dataset

