import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
from main import test
from loaders import get_train_test_loader
import os
from models import model_dict
from attacks import fgsm_attack, pgd_attack



def save_images(adv_images, orig_images, save_image_path):

    # Salva la prima immagine avversariale
    print("Saving the first adversarial image")

    combined_images = torch.cat((orig_images, adv_images), dim=0)  # Combina immagini originali e avversariali

    grid = torchvision.utils.make_grid(combined_images, nrow=testloader.batch_size)

    torchvision.utils.save_image(grid.to('cpu'), save_image_path)

    print(f"Adversarial image saved at {save_image_path}")




def test_with_adversarial(net, testloader, device, epsilon, criterion, save_path=None):
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    adv_loss = 0

    for data in testloader:

        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        # Applica l'attacco FGSM
        adv_images = fgsm_attack(images, labels, net, epsilon, criterion)

        if save_path is not None:

            if not os.path.exists(os.path.dirname(save_path)): os.makedirs(save_path)

            save_image_path = os.path.join(save_path, "adv_fgsm_image_eps_" + str(epsilon) + ".png")
            
            save_images(adv_images, images, save_image_path)

            save_path = None
        
        # Ottieni le predizioni
        outputs = net(adv_images)
        adv_loss += criterion(outputs, labels).item()

        # Calcolo Top-1
        _, predicted = torch.max(outputs, 1)
        correct_top1 += (predicted == labels).sum().item()

        # Calcolo Top-5
        _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
        correct_top5 += (top5_pred == labels.view(-1, 1)).sum().item()

        total += labels.size(0)

    # Calcola le metriche finali
    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total
    avg_loss = adv_loss / len(testloader)

    print(f'Accuracy of the network on adversarial images (epsilon={epsilon}): \n Top-1 = {top1_accuracy}%, Top-5 = {top5_accuracy}%, Loss: {avg_loss}')


def test_with_pgd(net, testloader, device, epsilon, alpha, num_iter, criterion, save_path=None):
                  
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    adv_loss = 0

    for data in testloader:

        images, labels = data
        images, labels = images.to(device), labels.to(device)

        adv_images = pgd_attack(images, labels, net, epsilon, alpha, num_iter, criterion)

        if save_path is not None:

            if not os.path.exists(os.path.dirname(save_path)): os.makedirs(save_path)

            save_image_path = os.path.join(save_path, "adv_pgd_image_eps_" + str(epsilon) + "_alpha_" + str(alpha) + "_num_iter_" + str(num_iter) + ".png")
            
            save_images(adv_images, images, save_image_path)

            save_path = None
        
        # Ottieni le predizioni
        outputs = net(adv_images)
        adv_loss += criterion(outputs, labels).item()

        # Calcolo Top-1
        _, predicted = torch.max(outputs, 1)
        correct_top1 += (predicted == labels).sum().item()

        # Calcolo Top-5
        _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
        correct_top5 += (top5_pred == labels.view(-1, 1)).sum().item()

        total += labels.size(0)

    # Calcola le metriche finali
    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total
    avg_loss = adv_loss / len(testloader)

    print(f'Accuracy of the network on adversarial images (PGD, epsilon={epsilon}, alpha={alpha}, num_iter={num_iter}):\n Top-1 = {top1_accuracy}%, Top-5 = {top5_accuracy}%, Loss: {avg_loss}')



if __name__ == "__main__":

    #setup cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("Device:", device)

    dataset_name = "cifar10"
    dataset_path = './work/project/data'
    batch_size = 32
    num_workers = 8

    _, testloader, n_cls = get_train_test_loader(dataset_name, 
                                                           data_folder=dataset_path, 
                                                           batch_size=batch_size, 
                                                           num_workers=num_workers)

    print(dataset_name," - Testloader lenght: ", len(testloader))


    # import net

    model_name = "resnet18"
    net = model_dict["resnet18"](num_classes=n_cls).to(device)

    model_path = 'work/project/save/'+dataset_name+'/'+model_name+'/state_dict.pth'

    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval().to(device)

    print("Loaded model from ", model_path)

    criterion = nn.CrossEntropyLoss()

    # Esegui il test normale
    print("Testing with normal examples")
    test(net, testloader, criterion, device)


    save_path= "work/project/adv_results/" + dataset_name + "/" + model_name + "/"

    # Esegui l'attacco PGD
    epsilon = 0.1  
    alpha = 0.01  
    num_iter = 10  
    print("Testing with adversarial examples (PGD), epsilon=", epsilon, "alpha=", alpha, "num_iter=", num_iter)
    test_with_pgd(net, testloader, device, epsilon, alpha, num_iter, criterion, save_path=save_path)

    # Esegui l'attacco PGD
    epsilon = 0.1  
    alpha = 0.3  
    num_iter = 10  
    print("Testing with adversarial examples (PGD), epsilon=", epsilon, "alpha=", alpha, "num_iter=", num_iter)
    test_with_pgd(net, testloader, device, epsilon, alpha, num_iter, criterion, save_path=save_path)

    # Esegui l'attacco PGD
    epsilon = 0.1  
    alpha = 0.2  
    num_iter = 100 
    print("Testing with adversarial examples (PGD), epsilon=", epsilon, "alpha=", alpha, "num_iter=", num_iter)
    test_with_pgd(net, testloader, device, epsilon, alpha, num_iter, criterion, save_path=save_path)

    # Esegui l'attacco FGSM
    epsilon = 0.1  
    print("Testing with adversarial examples, epsilon=", epsilon)
    test_with_adversarial(net, testloader, device, epsilon, criterion, save_path=save_path)

    epsilon = 0.3  
    print("Testing with adversarial examples, epsilon=", epsilon)
    test_with_adversarial(net, testloader, device, epsilon, criterion)

    epsilon = 0.5  
    print("Testing with adversarial examples, epsilon=", epsilon)
    test_with_adversarial(net, testloader, device, epsilon, criterion, save_path=save_path)




    # epsilon = 0.01  
    # alpha = 0.01  
    # num_iter = 50  
    # print("Testing with adversarial examples (PGD), epsilon=", epsilon, "alpha=", alpha, "num_iter=", num_iter)

