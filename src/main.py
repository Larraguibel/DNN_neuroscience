import torch
import torchvision
import wandb
import random
from utils import train, test, load_model, load_data, initialize_model,Transform_image
import argparse



if "__main__" == __name__:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store", help="Model to use")
    parser.add_argument("--dataset", type=int, help="", default=10)
    parser.add_argument("--epochs", type=int, help="", default=300)
    results = parser.parse_args()


    model = results.model
    dataset = results.dataset
    epochs = results.epochs
    train_batch_size = 64
    test_batch_size = 1000

    wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": model,
    "dataset": dataset,
    "epochs": epochs,
        }
    )
    train_loader, test_loader = load_data('data', train_batch_size, test_batch_size,transform1, transform2) 

    model, optimizer, criteron, scheduler = initialize_model(model)                          

    for epoch in epochs:
        train(epoch, train_loader, model, optimizer)

    test(epoch, best_loss, best_epoch, test_loader, model, test_name)

    
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()