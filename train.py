
#My functions

#To get arguments from the CMD
def get_input_arg():
    
    parser = argparse.ArgumentParser(description='Image Classifier')

    parser.add_argument("image_dir", required=True, type=str, help="path to folder of images")
    parser.add_argument('--arch',action='store_true', type = str, default='densenet121', help=" CNN model architecture")
    parser.add_argument('--learning_rate', action='store_true', type=int, default=0.001, help="Learning rate of the model")
    parser.add_argument('--save_dir', action='store_true',type=str, default='chekpoint.pth', help="To save the directory")
    parser.add_argument('--epochs', action='store_true',type=str, default=20, help="Number of epochs")
    parser.add_argument('--hidden_units', action='store_true',type=int, default=512, help="Hidden layer features")
    parser.add_argument('--gpu', action='store_true',type=str, default='cuda', help="Set to gpu mode")
    
    return parser.parse_args()

#To load the and the datalaoder
def data_loader(image_path):
    
    train_dir = image_path
    test_dir = image_path
        
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
        
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transform)
        
    trainloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    testloader = DataLoader(test_dataset, batch_size = 32)
        
    return trainloader, testloder
    
#My training loop function and print the result

def train_funct(epochs=30, hidden_units=512, lr_rate=0.001,  model='resNet50', gpu='cpu', data_dir, save_dir=False):
    #Defining and transfering the model into the gpu or cpu
    
    device = gpu
    model = resNet50
    model.to(device)
    
    #Freeze the parameters of the model and set new parameters
    for param in model.parameters():
        param.requires_grad = False
    classifer = Classifier(hidden_units=hidden_units)
    model.classifier = my_classifier()
    
    #Defining the  Criterion and the optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr_rate)
    
    #Loading the datasets and defining loaders using dataloader function
    trainloader, validloader = data_loader(image_dir)
    
    epochs = epochs

    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0 

            model.eval()
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)

                    output = model(images)
                    test_loss += criterion(output, labels)

                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor))

            model.train()

            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Trainning Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Test Accuracy: {:.3f}.. ".format(accuracy/len(testloader)))


#To save directory
def saved_dir(saved_dir, model):
    
    checkpoint = {'state_dict': model.state_dict(),
                 'opitmizer': optimizer.state_dict()}
    torch.save(chekpoint, saved_dir)
    
    return saved_dir


#Main function
def main():
    in_arg = get_input_arg()
    epochs = in_arg.epochs
    lr_rate = in_arg.learning_rate
    data_dir = in_arg.image_dir
    model_arch = in_arg.arch
    save_dir  = in_arg.save_dir
    hidden_units = in_arg.hidden_units
    gpu = in_arg.gpu
    
    
    if (epochs and lr_rate and hidden_units):
        train_func(epochs, hidden_units, lr_rate, data_dir)
    if gpu:
        train_func(data_dir, gpu)
    if save_dir:
        saved_dir(save_dir, model)
    train_func(data_dir)
    
if __name__ == "__main__":
    main()