import argparse

import torch.nn as nn;
import download_mnist as dm;
import torch;

from cnn import CNN_Classifier;
from lstm import LSTM_Classifier;

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Use the GPU if available

#CNN MODEL:
def run_cnn_model(epochs, patience, learning_rate, batch_size, tuning = False):
    train_data, validation_data, test_data = dm.get_MNIST_loaders(train_batch_size=batch_size)
    
    model = CNN_Classifier().to(device)

    if tuning:
        print("[CNN Tuning]")
    if not tuning:
        print(f"CNN Model: max_epochs={epochs}, early_stop_threshold={patience}, batch_size={batch_size}, lr={learning_rate}")

    test_patience = 0

    #TRAIN PHASE
    for epoch in range(epochs): 
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        #Train
        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device) #Move data to the same device

            optimizer.zero_grad() #Reset gradients to zero
            outputs = model(images) #Classify images
            loss = loss_function(outputs, labels) #Calculate loss
            loss.backward() #Back propagation (Fills gradients)
            optimizer.step() #Adjust weights by its new gradient

        model.eval() # Switch to eval mode (no model updates)
        
        #Validate
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for images, labels in validation_data:
                images, labels = images.to(device), labels.to(device) #Move data to the same device
                outputs = model(images) #Classify validation images
                val_loss += loss_function(outputs, labels).item() #Accumulate total loss
                _, predicted = torch.max(outputs, 1) #Parse through rows of outputs, extract highest confidence predictions
                total += labels.size(0) #Total labels
                correct += (predicted == labels).sum().item() #Tally correctly predicted outputs
        if not tuning:
            print(f" Epoch: {epoch+1}\n  Loss: {val_loss:.3f}\n  Accuracy: {100*correct/total:.2f}")

        #Early stopping
        if (epoch==0):
            #First epoch becomes the "best" by default
            best_loss = val_loss
            best_epoch = epoch
        elif(val_loss <= best_loss):
            test_patience = 0 #If new epoch is a new best, reset patience counter
            best_loss = val_loss #Save the new best
            best_correct = correct #Save the best accuracy
            best_total = total
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_cnn_state.pth') #Save the best state of the model
        else:
            test_patience += 1 #Add one for a non-improving epoch
            #Stop early after too many non-improving epochs
            if (test_patience >= patience):
                if(epoch != epochs-1):
                    if not tuning:
                        print("[Stopping Early]")
                break
            
    best_accuracy = best_correct / best_total #Calculate Accuracy

    if not tuning:
        print(f" Best Epoch: {best_epoch+1}\n Best Val Loss: {best_loss:.3f}\n Best Accuracy: {best_accuracy*100:.2f}")
    
    #TEST PHASE
    #Testing the best performing model on the training data
    model.eval() #Swap to eval mode
    model.load_state_dict(torch.load('best_cnn_state.pth')) # Load the best performing state of the model
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += loss_function(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    train_acc = test_correct/test_total
    if not tuning:
        print(f"\nBest CNN Model Test (Training Dataset): size={test_total}, batch_size={batch_size}, lr={learning_rate}")
        print(f" Total Loss: {test_loss:.3f}\n Accuracy: {100*test_correct/test_total:.2f}")

    #Testing the best performing model on the validation data
    model.eval() #Swap to eval mode
    model.load_state_dict(torch.load('best_cnn_state.pth')) # Load the best performing state of the model
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in validation_data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += loss_function(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    val_acc = test_correct/test_total
    if not tuning:
        print(f"\nBest CNN Model Test (Validation Dataset): size={test_total}, batch_size={batch_size}, lr={learning_rate}")
        print(f" Total Loss: {test_loss:.3f}\n Accuracy: {100*test_correct/test_total:.2f}")

    #Testing the best performing model on the test data
    model.eval() #Swap to eval mode
    model.load_state_dict(torch.load('best_cnn_state.pth')) # Load the best performing state of the model
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += loss_function(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_acc = test_correct/test_total
    if not tuning:
        print(f"\nBest CNN Model Test (Test Dataset): size={test_total}, batch_size={batch_size}, lr={learning_rate}")
        print(f" Total Loss: {test_loss:.3f}\n Accuracy: {100*test_correct/test_total:.2f}\n")

    avg_acc = (train_acc + val_acc + test_acc) / 3
    print(f"Average Accuracy of CNN Model with: max_epochs={epochs}, patience={patience}, batch_size={batch_size}, lr={learning_rate}")
    print(f"> {avg_acc*100:.2f}%")
    if tuning:
        return avg_acc #Return average accuracy for tuning purposes
    


#LSTM MODEL:
def run_lstm_model(epochs, patience, learning_rate, batch_size, tuning = False):
    train_data, validation_data, test_data = dm.get_MNIST_loaders(train_batch_size=batch_size)
    model = LSTM_Classifier().to(device)

    if tuning:
        print("[LSTM Tuning]")
    if not tuning:
        print(f"LSTM Model: max_epochs={epochs}, early_stop_threshold={patience}, batch_size={batch_size}, lr={learning_rate}")

    loss_function = nn.CrossEntropyLoss() #Use cross entropy loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #Use Adam optimizer

    test_patience = 0

    #TRAINING PHASE
    for epoch in range(epochs):

        model.train() #Switch to training mode
        #Train
        for images, labels in train_data:
            images = images.squeeze(1).to(device) #Reshape data to 3 dimensions
            labels = labels.to(device)

            #Forward
            outputs = model(images) #Classify images
            loss = loss_function(outputs, labels) #Calculate loss

            #Backward/Optimizer
            optimizer.zero_grad() #Reset gradients to zero
            loss.backward() #Fill gradients
            optimizer.step() #Update weights

        model.eval() # Switch to eval mode (no model updates)
        
        #Validate
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for images, labels in validation_data:
                images = images.squeeze(1).to(device)
                labels = labels.to(device)
                outputs = model(images)
                val_loss += loss_function(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            if not tuning:
                print(f" Epoch: {epoch+1}\n  Total Loss: {val_loss:.3f}\n  Accuracy: {100*correct/total:.2f}")

        #Early Stopping
        if (epoch == 0):
            best_loss = val_loss
            best_epoch = epoch
        elif(val_loss <= best_loss):
            test_patience = 0
            best_loss = val_loss
            best_correct = correct
            best_total = total
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_lstm_state.pth') #Save the best state of the model
        else:
            test_patience += 1
            if (test_patience >= patience):
                if(epoch != epochs-1):
                    if not tuning:
                        print("[Stopping Early]")
                break
    best_accuracy = best_correct / best_total

    if not tuning:
        print(f" Best Epoch: {best_epoch+1}\n Best Val Loss: {best_loss:.3f}\n Best Accuracy: {best_accuracy*100:.2f}\n")

    #TEST PHASE
    #Testing the best performing model on the training data
    model.eval()
    model.load_state_dict(torch.load('best_lstm_state.pth')) # Load the best performing state of the model
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in train_data:
            images = images.squeeze(1).to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_loss += loss_function(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    train_acc = test_correct/test_total
    if not tuning:
        print(f"\nBest LSTM Model Test (Train Dataset): size={test_total}, batch_size={batch_size}, lr={learning_rate}")
        print(f" Total Loss: {test_loss:.3f}\n Accuracy: {100*test_correct/test_total:.2f}")

    #Testing the best performing model on the validation data
    model.eval()
    model.load_state_dict(torch.load('best_lstm_state.pth')) # Load the best performing state of the model
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in validation_data:
            images = images.squeeze(1).to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_loss += loss_function(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    val_acc = test_correct/test_total
    if not tuning:
        print(f"\nBest LSTM Model Test (Validation Dataset): size={test_total}, batch_size={batch_size}, lr={learning_rate}")
        print(f" Total Loss: {test_loss:.3f}\n Accuracy: {100*test_correct/test_total:.2f}")

    #Testing the best performing model on the test data
    model.eval()
    model.load_state_dict(torch.load('best_lstm_state.pth')) # Load the best performing state of the model
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_data:
            images = images.squeeze(1).to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_loss += loss_function(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_acc = test_correct/test_total
    if not tuning:
        print(f"\nBest LSTM Model Test (Test Dataset): size={test_total}, batch_size={batch_size}, lr={learning_rate}")
        print(f" Total Loss: {test_loss:.3f}\n Accuracy: {100*test_correct/test_total:.2f}\n")

    avg_acc = (train_acc + val_acc + test_acc) / 3
    if tuning:
        print("\r")
    print(f"Average Accuracy of LSTM Model with: max_epochs={epochs}, patience={patience}, batch_size={batch_size}, lr={learning_rate}")
    print(f"> {avg_acc*100:.2f}%\n")
    if tuning:
        return avg_acc #Return average accuracy for tuning purposes



#MAIN:
parser = argparse.ArgumentParser(description="cnn or lstm")
parser.add_argument('--network', type=str, default='none', help='Choose model: cnn or lstm')
parser.add_argument('--tuning', type=str, default='False', help='Tuning mode: True or False')
parser.add_argument('--batch_size', type=int, default=64, help='Set batch size: Default is 64')
parser.add_argument('--lr', type=float, default=0.001, help='Set learning rate: Default is 0.001')
parser.add_argument('--epochs', type=int, default=25, help='Set max epochs: Default is 25')
parser.add_argument('--patience', type=int, default=3, help='Set patience threshold: Default is 3')

args = parser.parse_args()

network_map = {
    'cnn': run_cnn_model,
    'lstm': run_lstm_model
}

#TUNING:
if args.tuning == 'True':
    lr_array = [0.001, 0.002, 0.005, 0.010]
    bs_array = [64, 128, 256]
    best_acc = 0
    best_lr = 0
    best_bs = 0

    print("Tuning Mode: On\n")
    if args.network == 'none':
        print("Please choose a network to tune")
    elif args.network == 'cnn':
        for lr in lr_array:
            for bs in bs_array:
                acc = run_cnn_model(epochs=args.epochs, patience=args.patience, learning_rate=lr, batch_size=bs, tuning=True)
                if acc > best_acc:
                    best_acc = acc
                    best_lr = lr
                    best_bs = bs

        print("\n===========================")
        print(f"CNN MODEL TUNING:\n Best Accuracy={best_acc*100:.2f}%\n Best Learning Rate={best_lr}\n Best Batch Size={best_bs}")
        print("===========================\n")

        best_acc = 0
        best_lr = 0
        best_bs = 0

    elif args.network == 'lstm':
        for lr in lr_array:
            for bs in bs_array:
                acc = run_lstm_model(epochs=args.epochs, patience=args.patience, learning_rate=lr, batch_size=bs, tuning=True)
                if acc > best_acc:
                    best_acc = acc
                    best_lr = lr
                    best_bs = bs

        print("\n===========================")
        print(f"LSTM MODEL TUNING:\n Best Accuracy={best_acc*100:.2f}%\n Best Learning Rate={best_lr}\n Best Batch Size={best_bs}")
        print("===========================\n")

#NOT TUNING:
else:
    if args.network in network_map:
        network_map[args.network](patience=args.patience,epochs=args.epochs,batch_size=args.batch_size, learning_rate=args.lr)
    elif args.network != 'none':
        print("Invalid Network")

