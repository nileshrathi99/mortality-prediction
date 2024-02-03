import numpy as np
import torch.optim as optim
import tqdm
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# def validate(dataloader):
#     with torch.no_grad():
#         v_loss = 0
#         total = 0
#         y_true = np.zeros(dataloader.batch_size * len(dataloader))
#         y_pred = np.zeros(dataloader.batch_size * len(dataloader))
#         for x, y in dataloader:
#             x= x.to(device)
#             y= y.to(device)
#             #y = y.float().unsqueeze(1)
#             y_hat = model(x)
#             pred = torch.argmax(y_hat, dim= 1)
#             y_pred[total*test_loader.batch_size: total*test_loader.batch_size + test_loader.batch_size] = pred.cpu().detach().numpy()
#             y_true[total*test_loader.batch_size: total*test_loader.batch_size + test_loader.batch_size] = y.cpu().detach().numpy()
#             v_loss += criterion(y_hat, y).item()
#             total += 1    


def train(model, train_loader, test_loader, criteria, lr = 1e-3, weight_decay = 1e-4, max_epochs = 50, device = 'cuda', name = 'nn_model.pth'):

    train_loss = np.zeros(max_epochs )
    val_loss = np.zeros(max_epochs ) 
    val_acc = np.zeros(max_epochs)
    val_auc = np.zeros(max_epochs)
    train_acc = np.zeros(max_epochs)

    criterion = criteria
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay= weight_decay)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10], gamma= 0.1)
    model = model.to(device)
    t = tqdm.trange(max_epochs, desc= 'Starting ...', leave= True)
    
    best_auc = 0.5
    val_auc_patience = 10
    current_lr  = lr
    patience = 0

    for epoch in t:
        model.train(True)
        t_loss = 0
        total = 0
        y_true = np.zeros(train_loader.batch_size * len(train_loader))
        y_pred = np.zeros(train_loader.batch_size * len(train_loader))
        for x, y in train_loader:
            x= x.to(device)
            y= y.to(device)
            #y = y.float().unsqueeze(1)
            outputs = model(x)
            # print(outputs)
            pred = torch.argmax(outputs, dim= 1)
            y_pred[total*train_loader.batch_size: total*train_loader.batch_size + train_loader.batch_size] = pred.cpu().detach().numpy()
            y_true[total*train_loader.batch_size: total*train_loader.batch_size + train_loader.batch_size] = y.cpu().detach().numpy()
            
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            total += 1
        
        # break
        # scheduler.step()
        # print(len(train_loss))
        train_acc[epoch] = 100* np.sum(y_true == y_pred)/len(y_true)
        train_loss[epoch] = t_loss
        
        model.eval()

        with torch.no_grad():
            v_loss = 0
            total = 0
            y_true = np.zeros(test_loader.batch_size * len(test_loader))
            y_pred = np.zeros(test_loader.batch_size * len(test_loader))
            for x, y in test_loader:
                x= x.to(device)
                y= y.to(device)
                #y = y.float().unsqueeze(1)
                y_hat = model(x)
                pred = torch.argmax(y_hat, dim= 1)
                y_pred[total*test_loader.batch_size: total*test_loader.batch_size + test_loader.batch_size] = pred.cpu().detach().numpy()
                y_true[total*test_loader.batch_size: total*test_loader.batch_size + test_loader.batch_size] = y.cpu().detach().numpy()
                v_loss += criterion(y_hat, y).item()
                total += 1

        val_loss[epoch] = v_loss
        val_acc[epoch] = 100* np.sum(y_true == y_pred)/len(y_true)
        val_auc[epoch] = roc_auc_score(y_true, y_pred)

        t.set_description(f'{epoch}/{max_epochs}: Train:Loss = {train_loss[epoch]}, Val:Loss = {val_loss[epoch]}, Train:Acc = {train_acc[epoch]}, Val:Acc = {val_acc[epoch]}, Val:AUC = {val_auc[epoch]}', refresh= True)

        # if val_auc[epoch] > best_auc:
        #     best_auc = val_auc[epoch]
        #     patience = 0
        # else:
        #     patience += 1
        #     if patience == val_auc_patience/2:
        #         current_lr *= 0.1
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = current_lr
        #     if patience == val_auc_patience:
        #         break
    #print(scheduler.get_lr())
    return train_loss, val_loss, train_acc, val_acc, val_auc


def view_training_graphs(training, validation, title = ''):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the training loss
    ax1.plot(training, color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Training')

    # Plot the validation loss
    ax2.plot(validation, color='blue')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Validation')

    # Display the figure
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig