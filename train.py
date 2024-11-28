import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torchvision import models
from models import model_dict
import os

from loaders import get_train_test_loader

def train(net, trainloader, criterion, optimizer,device, epochs=20):

    for epoch in range(epochs):  

        for inputs, labels  in trainloader:

            running_loss = 0.0

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Stampa della loss
            running_loss += loss.item()

        # print della loss

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')



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
 
        

if __name__ == "__main__":


    #setup cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("Device:", device)

    dataset_name = "cifar10"
    dataset_path = './work/project/data'
    batch_size = 128
    num_workers = 8

    trainloader, testloader, n_cls = get_train_test_loader(dataset_name, 
                                                           data_folder=dataset_path, 
                                                           batch_size=batch_size, 
                                                           num_workers=num_workers)

    print(dataset_name," - Trainloader lenght: ", len(trainloader), "Testloader lenght: ", len(testloader))

    # import net

    model_name = "resnet18"
    net = model_dict[model_name](num_classes=n_cls).to(device)

    assert net is not None, "Model not found"

    lr = 0.0001
    epochs = 100

    criterion = nn.CrossEntropyLoss()  #
    optimizer = optim.Adam(net.parameters(), lr=lr)  # Adam optimizer 

    print("Start training")

    train(net, trainloader, criterion, optimizer, device, epochs=epochs)

    test(net, testloader, criterion, device)

    #join paths

    save_path = os.path.join('work/project/save/',dataset_name,model_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(net.state_dict(), save_path+'/state_dict.pth')




