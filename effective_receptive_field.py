import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from modules.convolution import Conv2d

def effective_receptive_field_torch(x):
    x = torch.tensor(x).to(torch.float32).requires_grad_(True)
    conv = nn.Conv2d(in_channels=3, out_channels=10,
                     kernel_size=3, stride=1,
                     padding=0, bias=False)
    z = conv(x)
    loss_vec = z[:, :, z.size(-2)//2, z.size(-1)//2]
    loss = torch.sum(loss_vec)
    loss.backward()
    
    grad_x =  x.grad[0, 0].detach().numpy()
    grad_weight = conv.weight.grad[0, 0].detach().numpy()
    print(grad_x[112])
    print(grad_weight) 
    

def effective_receptive_field_np(x):
    conv = Conv2d(in_channels=3, out_channels=10, padding=0, stride=1, kernel_size=3)
    z = conv.forward(x)
    
    grad_z = np.zeros_like(z)
    grad_z[:, :, z.shape[-2]//2, z.shape[-1]//2] = 1
    
    grad_x, grad_weight = conv.backpropagation(np.float32(grad_z))
    print(grad_x[0, 0, 112])
    print(grad_weight[0, 0]) 
    
if __name__ == "__main__":
    x = np.array([[
        [1, 2, 3, 4, 5],
        [7, 8, 9, 2, 1],
        [5, 6, 7, 8, 1],
        [3, 2, 1, 6, 2],
        [2, 4, 5, 8, 9],
    ]])
    x = np.expand_dims(np.float32(x), 1)
    
    image = Image.open("samples/cock.jpg").resize((224, 224))
    x = np.array(image)[np.newaxis, :, :]
    x = np.transpose(x, (0, 3, 1, 2))


    effective_receptive_field_np(x)
    print()
    effective_receptive_field_torch(x)