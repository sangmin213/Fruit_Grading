import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from matplotlib import pyplot as plt, transforms
from glob import glob
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import torchvision
from torchvision import transforms
from torch.optim import lr_scheduler


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock,self).__init__()

        # BN에 bias가 포함되어 있으므로, conv2d의 bias=False
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=3, init_weights=True, best_model_save_path="./ResNet_best_model.pt"):
        super().__init__()

        self.best_model_save_path = best_model_save_path
        self.in_channels=64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def train_(self, train_loader, val_loader, learning_rate, epochs, device):
        self.train_accuracy = []
        self.train_loss = []
        self.val_accuracy = []
        self.val_loss = []
        self.pred_labels_train = []
        self.real_labels_train = []
        self.pred_labels_val = None
        self.real_labels_val = None

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_f=nn.CrossEntropyLoss()

        best_epoch = -1
        best_acc = -1 
        
        for epoch in range(1, epochs+1):
            total = 0
            correct = 0
            start_time = time.time()
            epoch_loss = 0.0
            epoch_acc = 0.0
            self.train()

            for batch_idx, (batch_data, batch_label) in enumerate(tqdm(train_loader)):

                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)

                self.optimizer.zero_grad()

                pred = self.forward(batch_data) # (batch_size)
                loss = self.loss_f(pred, batch_label)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                _, pred_indices = torch.max(pred, axis=1)
                total += batch_data.shape[0]

                # print(pred_indices)
                # print(batch_label)

                # batch_label = torch.max(batch_label, axis=1)[1]
                correct += pred_indices.eq(batch_label).sum().item()
                
                if epoch==epochs: #last epoch
                    self.pred_labels_train.append(pred_indices)
                    self.real_labels_train.append(batch_label)
            
            end_time = time.time()
            print(f"epoch {epoch} time: {end_time-start_time}sec(s).")
            

            epoch_loss /= len(train_loader)
            self.train_loss.append(epoch_loss)
            epoch_acc = correct / total
            self.train_accuracy.append(epoch_acc)
            print(f"epoch {epoch} train accuracy: {epoch_acc}")
            print(f"epoch {epoch} loss: {epoch_loss}")  


            predicted, labels, val_loss = self.predict(val_loader, device)
            if epoch==epochs: #last epoch
                self.pred_labels_val=predicted.cpu().numpy()
                self.real_labels_val=labels.cpu().numpy()
            val_acc = predicted.eq(labels).sum().item() / len(predicted)
            print(f"epoch {epoch} val accuracy: {val_acc}")
            print(f"epoch {epoch} val loss: {val_loss}")

            if val_acc > epoch_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(self.state_dict(), self.best_model_save_path)
            
            self.val_accuracy.append(val_acc)
            self.val_loss.append(val_loss)
        
        self.pred_labels_train = torch.cat(self.pred_labels_train, dim=0)
        self.real_labels_train = torch.cat(self.real_labels_train, dim=0)
        self.pred_labels_train = self.pred_labels_train.cpu().numpy()
        self.real_labels_train = self.real_labels_train.cpu().numpy()
            
            
        print("Finish!")
        
        return best_acc, best_epoch
            
    def predict(self, test_loader, device):
        self.eval()
        labels = []
        predicted = []
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (batch_data, batch_label) in enumerate(tqdm(test_loader)):
                
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)
                
                pred = self.forward(batch_data)

                _, pred_indices = torch.max(pred, axis=1)
                loss = self.loss_f(pred, batch_label)
                
                val_loss += loss.item()

                predicted.append(pred_indices)
                # batch_label = torch.max(batch_label, axis=1)[1]
                labels.append(batch_label)
        val_loss /= len(test_loader)
        predicted = torch.cat(predicted, dim=0)
        labels = torch.cat(labels, dim=0)

        return predicted, labels, val_loss
    
    def plot(self, which):
        
        X = [i for i in range(1, len(self.train_accuracy) + 1)]
        if which=='train_loss':
            y = self.train_loss
        elif which=='train_acc':
            y = self.train_accuracy
        elif which=='val_acc':
            y = self.val_accuracy
        elif which=='val_loss':
            y = self.val_loss
        elif which=='confusion_train':
            ConfusionMatrixDisplay.from_predictions(self.real_labels_train, self.pred_labels_train, display_labels=["fresh","rotten"])
            plt.title('train confusion matrix')
            plt.savefig(f"./result/{which}.png")
            return
        elif which=='confusion_normalize_train':
            ConfusionMatrixDisplay.from_predictions(self.real_labels_train, self.pred_labels_train, display_labels=["fresh","rotten"], normalize='true')
            plt.title('train confusion matrix')
            plt.savefig(f"./result/{which}.png")
            return        
        elif which=='confusion_val':
            ConfusionMatrixDisplay.from_predictions(self.real_labels_val, self.pred_labels_val, display_labels=["fresh","rotten"])
            plt.title('val confusion matrix')
            plt.savefig(f"./result/{which}.png")
            return
        elif which=='confusion_normalize_val':
            ConfusionMatrixDisplay.from_predictions(self.real_labels_val, self.pred_labels_val, display_labels=["fresh","rotten"], normalize='true')
            plt.title('val confusion matrix')
            plt.savefig(f"./result/{which}.png")
            return   

        plt.xlabel("epoch")
        plt.ylabel(which)
        plt.title(which)
        plt.plot(X, y)
        plt.savefig(f"./result/{which}.png")
        plt.clf() # 현재 그려진 plot 삭제

    def resnet18():
        return ResNet(BasicBlock, [2,2,2,2], num_classes=2)

    def resnet34():
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=2)

    def resnet50():
        return ResNet(BottleNeck, [3,4,6,3], num_classes=2)

    def resnet101():
        return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=2)

    def resnet152():
        return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=2)


''' Dataset '''
train_transform=torchvision.transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

test_transform=torchvision.transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

train_dataset=torchvision.datasets.ImageFolder("./PreprocessedData/train/",transform=train_transform)
test_dataset=torchvision.datasets.ImageFolder("./PreprocessedData/test/",transform=test_transform)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=4) # ,num_workers=4
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=True,num_workers=4)

# print(train_dataset[0][0]) # Check data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=ResNet.resnet50().to(device)

model.train_(train_loader=train_loader, val_loader=test_loader, learning_rate=0.001, epochs=30, device=device)
model.plot('train_loss')
model.plot('train_acc')
model.plot('val_loss')
model.plot('val_acc')
model.plot('confusion_train')
model.plot('confusion_normalize_train')
model.plot('confusion_val')
model.plot('confusion_normalize_val')

