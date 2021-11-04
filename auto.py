import torch
import torch.nn as nn
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append(r"E:\data")
sys.path.append(r"E:\data\images")
sys.path.append(r"R:\git projects\auto-robot")
import model as mod

def train(model, x, y, alpha, epochs, batch, device = 'cuda'):
    
    optimizer = torch.optim.Adam(model.parameters(), lr = alpha)
    
    loss = nn.MSELoss()
    
    model.train()
    
    loss_list = []
    
    for i in range(epochs):
        
        print('Epoch', i+1, ':')
        
        shuffled_idx = torch.randperm(x.shape[0])
        
        for j in range(0, len(shuffled_idx) - batch, batch):
            
            optimizer.zero_grad(set_to_none=True)
            
            x_batch = x[shuffled_idx[j:j+batch]].to(device, dtype=torch.float)
            
            y_batch = y[shuffled_idx[j:j+batch]].to(device, dtype=torch.float)
            
            yp = model.forward(x_batch).squeeze()
            
            L = loss(y_batch, yp)
            
            L.backward()
            
            optimizer.step()
            
            del x_batch, y_batch
            
            loss_list.append(float(L))
            
            if j%10 == 0:
                
                print('     Batch',(j/batch)+1,': Loss =', float(L))
    
    torch.save(model.state_dict(), r'R:\git projects\auto-robot\model1.pth')
    
    return loss_list
     
#%%
N = int(input('Please enter the number of training samples: '))
y = torch.from_numpy(pd.read_csv('E:\data\ground_truth.csv')[['Left Wheel F/R','Right Wheel F/R']].values)
x = torch.zeros((N,3,360,360))
for i in range(N):
    x[i] = torch.from_numpy(cv2.imread('E:\data\images\img (%d).png'%(i+1)).T)
print(y.shape, x.shape)
print(y[0], x[0])

if torch.cuda.is_available():

    model = mod.auto().to('cuda')
    
    loss_list = train(model, x, y, 0.0001, 100, 5)
    
    plt.plot(loss_list)
    plt.show()
    
    '''
    for i in range(1, 50):
        
        xc = x[i:i+1].to('cuda')
        yp = model.forward(xc).to('cpu').detach().squeeze().numpy()
        ye = y[i].squeeze().numpy() == 1
        yp = yp > 0.5
        print(yp==ye)
        del xc, yp
    '''
else: print('CUDA not available. END.')