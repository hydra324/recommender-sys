import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from database import get_user_feedback



class MatrixFactorization(torch.nn.Module):
    
    def __init__(self, n_users, n_items, embed_dim=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, embed_dim,dtype=torch.float32)
        self.item_factors = torch.nn.Embedding(n_items, embed_dim,dtype=torch.float32)
        self.user_biases = torch.nn.Embedding(n_users, 1,dtype=torch.float32)
        self.item_biases = torch.nn.Embedding(n_items,1,dtype=torch.float32)
        torch.nn.init.xavier_uniform_(self.user_factors.weight)
        torch.nn.init.xavier_uniform_(self.item_factors.weight)
        self.user_biases.weight.data.fill_(0.)
        self.item_biases.weight.data.fill_(0.)
        
    def forward(self, user, item):
        pred = self.user_biases(user) + self.item_biases(item)
        pred += (self.user_factors(user) * self.item_factors(item)).sum(1, keepdim=True)
        return pred.squeeze()

class RatingDataset(Dataset):
    def __init__(self, train, label):
        '''
        A function to construct dataset.

        Args:
            train (ndarray): 2D user_id-video_id array.
            label (ndarray): 1D ratings array i.e, like/dislike array.

        Returns:
            None.
        '''
        self.feature_= train
        self.label_= label
    def __len__(self):
        #return size of dataset
        return len(self.feature_)
    def __getitem__(self, idx):
        return torch.tensor(self.feature_[idx],dtype=torch.int32),torch.tensor(self.label_[idx],dtype=torch.float32)

def main():
    # define params
    embed_dim = 100
    n_users = 1000
    n_videos = 10000

    model = MatrixFactorization(n_users,n_videos,embed_dim)

    batch_size = 100
    # get data
    x = get_user_feedback()
    # feedback_matrix = np.random.choice([0,1], size=(n_users,n_videos))
    np.random.shuffle(x)
    split_idx = int(0.75*(x.shape[0]))
    x_train = x[:split_idx,:]
    x_test = x[split_idx:,:]
    train_data = DataLoader(RatingDataset(x_train[:,:2],x_train[:,2]),batch_size=batch_size,shuffle=True)
    test_data = DataLoader(RatingDataset(x_test[:,:2],x_test[:,2]),batch_size=batch_size,shuffle=True)

    # TRAIN
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    loss_func = torch.nn.MSELoss()
    model.to(device)
    n_epochs = 100
    learning_rate = 0.01
    for epoch in range(n_epochs):
        progress_bar = tqdm(enumerate(train_data),total=len(train_data))
        count = 0.
        cum_loss = 0.
        for i,(train_batch,label_batch) in progress_bar:
            count = 1+i
            # Alternating least squares
            # First fix video matrix and optimize user matrix
            optimizer = torch.optim.SGD([model.user_biases.weight,model.user_factors.weight],
                                        lr=learning_rate,
                                        weight_decay=1e-5)
            pred = model(train_batch[:,0].to(device),
                        train_batch[:,1].to(device))
            loss = loss_func(pred,label_batch.to(device))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Now fix user matrix and optimize video matrix
            optimizer = torch.optim.SGD([model.item_biases.weight,model.item_factors.weight],
                                        lr=learning_rate,
                                        weight_decay=1e-5)
            pred = model(train_batch[:,0].to(device),
                        train_batch[:,1].to(device))
            loss = loss_func(pred,label_batch.to(device))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            cum_loss += loss.item()
            progress_bar.set_description(f'training loss at {epoch} batch {i}: {loss.item()}')

    train_loss = cum_loss/count


    # TEST

    progress_bar = tqdm(enumerate(test_data),total=len(test_data))
    cum_loss = 0.
    count = 0
    for i,(test_batch,label_batch) in progress_bar:
        count = 1+i
        with torch.no_grad():
            pred = model(test_batch[:,0].to(device),
                        test_batch[:,1].to(device))
            loss = loss_func(pred,label_batch.to(device))
            cum_loss += loss.item()
            progress_bar.set_description(f'test loss at {epoch} batch {i}: {loss.item()}')
    test_loss = cum_loss/count
    print(f'avg train_loss: {train_loss}, avg test_loss: {test_loss}')


from numpy import dot
from numpy.linalg import norm
from scipy import spatial

def cosSim(a, b):
    return 1 - spatial.distance.cosine(a, b)

def getSortedSimilarity(userIdx, userMatrix, ItemMatrix):
    res=[]
    u= userMatrix[userIdx,:]
    for i in range(0, ItemMatrix.shape[1]):
        v= ItemMatrix[:,i]
        res.append([cosSim(u,v), i])
    res.sort(reverse=True)
    res= np.array(res)
    return res[:, 1]

if __name__=='__main__':
    main()






