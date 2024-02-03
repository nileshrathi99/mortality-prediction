import pandas as pd
from torch.utils.data import DataLoader
from utils.configs import *
from utils.data_utils import *
from utils.dataloader import *
from utils.train import *
from models.LSTM import *
from models.CNN import *

def save_training_plots(train_loss, val_loss, train_acc, val_acc, val_auc, folder_path='results'):
    # Create the main results folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create a subfolder with the model name
    model_folder = os.path.join(folder_path, which_model + f'{input_dim}D')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Create another subfolder with X_Y
    subfolder = os.path.join(model_folder, f'{X}_{Y}')
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    plt.plot(val_auc)
    plt.title('AUC')
    plt.savefig(os.path.join(subfolder, 'auc.png'))

    fig = view_training_graphs(train_loss, val_loss, title = "LOSS")
    fig.savefig(os.path.join(subfolder, 'loss.png'))

    fig = view_training_graphs(train_acc, val_acc, title = "ACCURACY")
    fig.savefig(os.path.join(subfolder, 'accuracy.png'))

    plt.close()


def fetch_new_data():
    data = Data(os.path.join(base_dir, 'db', 'MV_Mortality.csv'), x= X, y = Y, num_workers= workers, modality= modality)
    data.select_patients(threshold= 0.5)
    table = data.get_table()
    print(len(table[table['label'] == 1]), len(table))
    
    selected_table = data.get_selected_table()
    create_splits(selected_table, n_folds)
    selected_table.to_csv(f'../db/selected_x{X}_y{Y}.csv')


def get_train_test_split():
    selected_table = pd.read_csv(os.path.abspath(f'db/selected_x{X}_y{Y}.csv'))
    print(len(selected_table[selected_table['label']== 1]), len(selected_table[selected_table['label']== 0]))
    return get_fold_split(selected_table, 0)


def get_train_test_dataloader(train_df, test_df):
    train_dataset = ECG_Data(train_df, x= X, y = Y, modality= modality, dim = input_dim) # set dim = 2 for 2D input
    test_dataset = ECG_Data(test_df, x= X, y = Y, modality= modality, dim = input_dim)
    print(len(test_dataset), len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size= 64, sampler= train_dataset.sampler, num_workers= workers, pin_memory= True, drop_last= True)
    test_loader = DataLoader(test_dataset, batch_size= len(test_dataset), shuffle= False, num_workers= 1, pin_memory= True)
    return train_loader, test_loader


def train_model(model, train_loader, test_loader):
    print('Traning Started')
    train_loss, val_loss, train_acc, val_acc, val_auc = train(model, train_loader, test_loader, criteria= nn.CrossEntropyLoss())
    save_training_plots(train_loss, val_loss, train_acc, val_acc, val_auc)
    print('Traning Over, plots saved')


def main():
    print(X, Y)
    print(which_model, input_dim)
    if fetch_new_window:
        fetch_new_data()
    
    train_df, test_df = get_train_test_split()
    train_loader, test_loader = get_train_test_dataloader(train_df, test_df)

    if which_model == 'LSTM':
        print(next(iter(train_loader))[0].shape)
        fdim, tdim = next(iter(train_loader))[0].shape[-2:]
        if input_dim == 1:
            train_model(LSTM1D(tdim = tdim), train_loader, test_loader)
        else:
            train_model(LSTM2D(fdim = fdim, tdim = tdim), train_loader, test_loader)
    
    if which_model == 'CNN':
        if input_dim == 1:
            train_model(RonNet1D(), train_loader, test_loader)
        else:
            train_model(RonNet2D(), train_loader, test_loader)


if __name__ == "__main__":
    main()