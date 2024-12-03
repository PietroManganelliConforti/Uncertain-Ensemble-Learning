import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torchvision import models
from models import model_dict
import os
import matplotlib.pyplot as plt
from loaders import get_train_and_test_loader
import argparse


def train(net, trainloader, valloader, criterion, optimizer, device, epochs=20, save_path=None):

     #TODO Early stopping

    train_metrics = {"running_loss": [],
                        "top1_accuracy": [],
                        "avg_loss": [],
                        "val_running_loss": [],
                        "val_top1_accuracy": [],
                        "val_avg_loss": []}
    

    for epoch in range(epochs):  
        correct_top1 = 0
        running_loss = 0.0  # Reset per epoca
        correct_top1_val = 0
        running_loss_val = 0.0

        net.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            correct_top1 += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            

        net.eval()
        for inputs, labels in valloader:
        
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_top1_val += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss_val += loss.item()

        train_metrics["val_top1_accuracy"].append(100 * correct_top1_val / len(valloader.dataset))
        train_metrics["val_running_loss"].append(running_loss_val / len(valloader))

        train_metrics["top1_accuracy"].append(100 * correct_top1 / len(trainloader.dataset)) 
        train_metrics["running_loss"].append(running_loss / len(trainloader))
        

        print(f'Epoch {epoch + 1}, Avg Loss: {running_loss / len(trainloader)}, Top-1 Accuracy: {100 * correct_top1 / len(trainloader.dataset)}')
        print(f'Validation Avg Loss: {running_loss_val / len(valloader)}, Validation Top-1 Accuracy: {100 * correct_top1_val / len(valloader.dataset)}')

    
        if save_path is not None and (epoch%10==0 or epoch==epochs-1):

            _, ax = plt.subplots(2, 1, figsize=(10, 10))
            ax[0].plot(train_metrics["running_loss"])
            ax[0].plot(train_metrics["val_running_loss"])
            ax[0].legend(["Training Loss", "Validation Loss"])
            ax[0].set_title("Loss")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")
            
            ax[1].plot(train_metrics["top1_accuracy"])
            ax[1].plot(train_metrics["val_top1_accuracy"])
            ax[1].legend(["Training Top-1 Accuracy", "Validation Top-1 Accuracy"])
            ax[1].set_title("Top-1 Accuracy")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Accuracy")
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "training_metrics.png"))
            plt.close()

    return train_metrics


def test(net, testloader, criterion, device):

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            # Calcolo della perdita per il batch
            test_loss_batch = criterion(outputs, labels)
            test_loss += test_loss_batch.item()

            # Calcolo Top-1 (massima probabilità)
            _, predicted = torch.max(outputs, 1)
            correct_top1 += (predicted == labels).sum().item()

            # Calcolo Top-5
            _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_top5 += (top5_pred == labels.view(-1, 1)).sum().item()

            total += labels.size(0)

    # Calcolo delle accuratezze finali
    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total
    avg_loss = test_loss / len(testloader)

    print(f'Accuracy of the network on the {total} test images from {len(testloader)} batches: \n'+
          f'Top-1 = {top1_accuracy}%, Top-5 = {top5_accuracy}%, Loss: {avg_loss}')
 
    return {"top1_accuracy": top1_accuracy, "top5_accuracy": top5_accuracy, "avg_loss": avg_loss}


def get_parser():

    parser = argparse.ArgumentParser(description='Train a model on a dataset')

    parser.add_argument('--model', type=str, default="resnet18", help='Model name')
    parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset name')
    parser.add_argument('--data_folder', type=str, default='./work/project/data', help='Path to dataset folder')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model weights')
    parser.add_argument('--save_path_root', type=str, default='work/project/save/', help='Path to save model and logs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save model and logs')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use (cpu or cuda:0)')
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained model')

    return parser


if __name__ == "__main__":

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    parser = get_parser()
    args = parser.parse_args()

    # Setup CUDA
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Model configuration
    model_name = args.model
    dataset_name = args.dataset
    dataset_path = args.data_folder
    batch_size = args.batch_size
    num_workers = args.num_workers
    save_path_root = args.save_path_root
    lr = args.lr
    epochs = args.epochs
    pretrained_flag = args.pretrained


    try:
        trainloader, testloader, n_cls = get_train_and_test_loader(dataset_name, 
                                                               data_folder=dataset_path, 
                                                               batch_size=batch_size, 
                                                               num_workers=num_workers)

        logger.info(f"{dataset_name} - Trainloader length: {len(trainloader)}, Testloader length: {len(testloader)}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        exit(1)


    try:
        net = model_dict[model_name](num_classes=n_cls, pretrained=pretrained_flag).to(device)
        assert net is not None, "Model not found"
        logger.info(f"Model {model_name} initialized with {n_cls} output classes.")
    except Exception as e:
        logger.error(f"Error initializing model {model_name}: {e}")
        exit(1)


    # Training phase
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    #save path for model and logs
    save_path = os.path.join(save_path_root, dataset_name, model_name+"_"+str(lr)+"_"+str(epochs))
    if pretrained_flag:
        save_path = save_path + "_pretrained"
    os.makedirs(save_path, exist_ok=True)

    logger.info("Starting training...")

    try:
        train_metrics = train(net, trainloader, testloader, criterion, optimizer, device, epochs=epochs, save_path=save_path)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        exit(1)

    # Testing phase
    logger.info("Starting testing...")


    try:
        test_metrics = test(net, testloader, criterion, device)
        logger.info(f"Test metrics: {test_metrics}")
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        exit(1)


    # Saving model and logs
    torch.save(net.state_dict(), os.path.join(save_path, 'state_dict.pth'))
    logger.info(f"Model weights saved to {save_path}/state_dict.pth")
    #save test metrics and then training metrics

    with open(os.path.join(save_path, "test_metrics.txt"), "w") as f:
        f.write(str(test_metrics))
        f.write("\n")
        f.write(str(train_metrics))
        f.write("\n")
        #write args
        f.write(str(args))
        logger.info(f"Test metrics and training metrics saved to {save_path}/test_metrics.txt")


    logger.info("Training and testing completed.")

    



