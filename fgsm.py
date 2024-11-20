import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from main import test
from loaders import get_train_test_loader
import os
from models import model_dict

def fgsm_attack(image, label, model, epsilon, criterion):


    # Imposta il modello in modalità valutazione
    model.eval()
    
    # Abilita il calcolo del gradiente
    image.requires_grad = True

    
    # Esegui il forward pass
    output = model(image)  
    #get cam from model
    
    # Calcola la perdita
    loss = criterion(output, label)
    
    # Azzerare i gradienti precedenti
    model.zero_grad()
    
    # Calcola il gradiente della perdita rispetto all'immagine
    loss.backward()
    
    # Prendi il segno del gradiente
    grad = image.grad.data
    
    # Crea la nuova immagine modificata
    perturbed_image = image + epsilon * grad.sign()
    
    # Assicurati che i valori dell'immagine siano compresi tra 0 e 1
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image



def test_with_adversarial(net, testloader, device, epsilon, criterion, save_first=False):
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    saved_flag = False
    adv_loss = 0

    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        # Applica l'attacco FGSM
        adv_images = fgsm_attack(images, labels, net, epsilon, criterion)

        if save_first and not saved_flag:
            # Salva la prima immagine avversariale
            print("Saving the first adversarial image")

            save_path = "work/project/saved_fig/"
            image_name = "adv_image_eps_" + str(epsilon) + ".png"

            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(save_path)

            combined_images = torch.cat((images, adv_images), dim=0)  # Combina immagini originali e avversariali
            grid = torchvision.utils.make_grid(combined_images, nrow=testloader.batch_size)
            torchvision.utils.save_image(grid.to('cpu'), os.path.join(save_path, image_name))
            saved_flag = True
        
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

    print(f'Accuracy of the network on adversarial images (epsilon={epsilon}): Top-1 = {top1_accuracy}%, Top-5 = {top5_accuracy}%, Loss: {avg_loss}')



if __name__ == "__main__":

    #setup cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("Device:", device)

    dataset_name = "cifar10"
    dataset_path = './work/project/data'
    batch_size = 128
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

    # Esegui l'attacco FGSM
    epsilon = 0.1  # Modifica questo valore per aumentare o diminuire la forza dell'attacco
    print("Testing with adversarial examples, epsilon=", epsilon)
    test_with_adversarial(net, testloader, device, epsilon, criterion, save_first=True)

    epsilon = 0.3  # Modifica questo valore per aumentare o diminuire la forza dell'attacco
    print("Testing with adversarial examples, epsilon=", epsilon)
    test_with_adversarial(net, testloader, device, epsilon, criterion)

    epsilon = 0.5  # Modifica questo valore per aumentare o diminuire la forza dell'attacco
    print("Testing with adversarial examples, epsilon=", epsilon)
    test_with_adversarial(net, testloader, device, epsilon, criterion)






