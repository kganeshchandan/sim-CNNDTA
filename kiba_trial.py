import torch 
import json,pickle,math
import pandas as pd
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

LSM = pickle.load(open('./kiba_ligand_similarity_matrix.pkl', 'rb'))
PSM = pickle.load(open('./kiba_protein_similarity_matrix.pkl', 'rb'))
df = pd.read_csv(open('./kiba_all_pairs.csv','r'))

SMILES = json.load(open('./data/KIBA/SMILES.txt'))
TARGETS = json.load(open('./data/KIBA/target_seq.txt'))
SMILES=list(SMILES.values())
TARGETS=list(TARGETS.values())

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# Hyper parameters
num_epochs = 5
# num_classes = 10
batch_size = 12
learning_rate = 0.001

class custom_dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, smiles, targets, LSM,PSM,transform=None):
        self.df = pd.read_csv(open(csv_file))
#         self.root_dir = root_dir
        self.smiles =smiles
        self.targets = targets
        self.LSM = LSM
        self.PSM = PSM
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        smi = self.df.iloc[idx]['SMILES']
        seq = self.df.iloc[idx]['Target Sequence']
        s_i = self.smiles.index(smi)
        t_i = self.targets.index(seq)
        
        ki=self.LSM[s_i]
        kj=self.PSM[t_i]
        
        ki_x_kj = np.outer(ki,kj)
        ki_x_kj = torch.tensor([ki_x_kj])
        output = {'outer_product': ki_x_kj , 'Label':self.df.iloc[idx]['Label']}
        return output


dataset = custom_dataset(csv_file = './kiba_all_pairs.csv', smiles=SMILES, targets = TARGETS, LSM=LSM,PSM=PSM)
full_dataset = dataset

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader= torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader= torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1,32, 5).double()
        self.pool1 = nn.MaxPool2d(2,2).double()
        self.conv2 = nn.Conv2d(32,18,3).double()
        self.pool2 = nn.MaxPool2d(2,2).double()
        self.fc1 = nn.Linear(18*525*55, 128).double()
        self.fc2 = nn.Linear(128,1).double()
        self.dropout = nn.Dropout(0.1).double()
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1,18*525*55)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        
        return x
    
model = ConvNet().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def rmse(y,f):
    rmse = math.sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
from lifelines.utils import concordance_index
def ci(y,f):
    return concordance_index(y,f)


# Train the model
best_mse = 1000
best_ci = 0
model_file_name = 'best_sim-CNN-DTA_kiba.model'
result_file_name = 'best_result_sim-CNNDTA_kiba.csv'
total_step = len(train_loader)
for epoch in range(num_epochs):
    c=0
    for i in train_loader:
        c=c+1
        images = i['outer_product']
        labels = i['Label']
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.flatten(), labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
           
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
               .format(epoch+1, num_epochs, c, total_step, loss.item()), flush =True)
    
    # taking best model so far
#     G,P = predicting(model, device, test_loader)
#     ret = [rmse(G, P), mse(G, P), pearson(G, P), ci(G, P)]
#     if ret[1] < best_mse:
#         torch.save(model.state_dict(), model_file_name)
#         with open(result_file_name, 'w') as f:
#             f.write(','.join(map(str, ret)))
#         best_epoch = epoch+1
#         best_mse = ret[1]
#         best_ci = ret[-1]
#         best_r = ret[2]
        
#         print('rmse improved at epoch ', best_epoch,
#                       '; best_mse,best_ci,best_r:', best_mse, best_ci,best_r)
        
        
model.eval()
# eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
total_preds = np.array([])
total_labels = np.array([])
with torch.no_grad():
    correct = 0
    total = 0
    for i in test_loader:
        images = i['outer_product']
        labels = i['Label']
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images) 
        outputs = outputs.cpu().detach().numpy().flatten()
        labels =labels.cpu().detach().numpy().flatten()
        total_preds = np.concatenate([total_preds, outputs])
        total_labels = np.concatenate([total_labels, labels])
#         total_preds = torch.cat(total_preds, outputs.cpu(), 0 )
#         total_labels = torch.cat(total_labels, labels.cpu(), 0)
#         break

G,P = total_labels, total_preds

print(pearson(G,P), ci(G,P), rmse(G,P), mse(G,P), flush =True)

