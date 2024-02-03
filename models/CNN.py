import torch
import torch.nn as nn
import torchvision.models as models


class RonNet1D(nn.Module):
    def __init__(self):
        super(RonNet1D, self).__init__()

        self.model = nn.Sequential()
        # Convolutional layers'
        self.model.append(nn.BatchNorm1d(1))
        self.model.append(nn.Conv1d(1, 64, kernel_size= 5,))
        self.model.append(nn.BatchNorm1d(64))
        self.model.append(nn.ReLU())
        self.model.append(nn.MaxPool1d(kernel_size= 2, stride= 2))
        # self.dropout1 = nn.Dropout2d(p=0.25)

        self.model.append(nn.Conv1d(64, 128, kernel_size= 3, padding= 1))
        self.model.append(nn.BatchNorm1d(128))
        self.model.append(nn.ReLU())
        self.model.append(nn.MaxPool1d(kernel_size= 2, stride= 2))
        # # self.dropout2 = nn.Dropout2d(p=0.25)

        self.model.append(nn.Conv1d(128, 256, kernel_size= 3, padding= 1))
        self.model.append(nn.BatchNorm1d(256))
        self.model.append(nn.ReLU())
        self.model.append(nn.MaxPool1d(kernel_size= 2, stride= 2))

        # Fully connected layers
        self.model.append(nn.AdaptiveAvgPool1d(1))
        self.fc1 = nn.Linear(256, 2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(1024, 2)  # Assuming input size of 64x64

        for layer in self.model.children():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nonlinearity= 'relu')

    def forward(self, x):
        x = x.float()
        x = self.model(x)
        x = torch.flatten(x, 1)
        # print(x.shape)   
        # x = self.act(self.fc1(x))
        x = self.fc1(x)
        return x


class RonNet2D(nn.Module):
    def __init__(self):
        super(RonNet2D, self).__init__()

        self.model = models.resnet18(pretrained = False)
        self.model.conv1 = nn.Conv2d(1, 64, 5, 2, 3, bias= False)
        self.model.fc = nn.Linear(512, 2)

        # self.model = nn.Sequential()
        # # Convolutional layers
        # self.model.append(nn.Conv2d(1, 64, kernel_size= 5, padding= 2))
        # self.model.append(nn.BatchNorm2d(64))
        # self.model.append(nn.ReLU())
        # self.model.append(nn.MaxPool2d(kernel_size= 2, stride= 2))
        # # self.dropout1 = nn.Dropout2d(p=0.25)

        # self.model.append(nn.Conv2d(64, 128, kernel_size= 3, padding= 1))
        # self.model.append(nn.BatchNorm2d(128))
        # self.model.append(nn.ReLU())
        # self.model.append(nn.MaxPool2d(kernel_size= 2, stride= 2))
        # # self.dropout2 = nn.Dropout2d(p=0.25)

        # self.model.append(nn.Conv2d(128, 256, kernel_size= 3, padding= 1))
        # self.model.append(nn.BatchNorm2d(256))
        # self.model.append(nn.ReLU())
        # self.model.append(nn.MaxPool2d(kernel_size= 2, stride= 2))

        # # Fully connected layers
        # self.model.append(nn.AdaptiveAvgPool2d(1))
        # self.fc1 = nn.Linear(256, 2)
        # self.act = nn.ReLU()
        # self.fc2 = nn.Linear(128, 2)  # Assuming input size of 64x64

        # for layer in self.model.children():
        #     if isinstance(layer, (nn.Conv2d, nn.Linear)):
        #         nn.init.kaiming_normal_(layer.weight, nonlinearity= 'relu')

    def forward(self, x):
        x = x.float()
        x = self.model(x)
        # x = torch.flatten(x, 1)
        # # print(x.shape)   
        # x = self.act(self.fc1(x))
        # x = self .fc2(x)
        return x