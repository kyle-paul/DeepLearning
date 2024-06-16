import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from modules.convolution import Conv2d

common_weight = None

def effective_receptive_field_torch(x: np.ndarray) -> None:
    x = torch.tensor(x).to(torch.float32).requires_grad_(True)
    conv = nn.Conv2d(in_channels=3, out_channels=4,
                     kernel_size=3, stride=1,
                     padding=1, bias=False)
    global common_weight
    conv.weight = torch.nn.Parameter(torch.tensor(common_weight))
    z = conv(x)
      
    loss_vec = z[:, :, z.size(-2)//2, z.size(-1)//2]
    loss = torch.sum(loss_vec)
    loss.backward()
    
    grad_x =  x.grad[0, 0].detach().numpy()
    grad_weight = conv.weight.grad[0, 0].detach().numpy()
    print(grad_x.shape)
    print(grad_x[112])
    print(grad_weight) 
    

def effective_receptive_field_np(x: np.array) -> None:
    conv = Conv2d(in_channels=3, out_channels=4, 
                  padding=1, stride=1, kernel_size=3)
    global common_weight
    common_weight = conv.weight
    z = conv.forward(x)
    
    grad_z = np.zeros_like(z)
    grad_z[:, :, z.shape[-2]//2, z.shape[-1]//2] = 1
    
    grad_x, grad_weight = conv.backpropagation(np.float32(grad_z))
    print(grad_x.shape)
    print(grad_x[0, 0, 112])
    print(grad_weight[0, 0]) 
    
if __name__ == "__main__":
    
    image = np.array(Image.open("samples/cock.jpg").resize((224, 224)))
    image2 = np.array(Image.open("samples/cats.jpg").resize((224, 224)))
    
    image = np.transpose(image, (2, 0, 1))
    image2 = np.transpose(image2, (2, 0, 1))
    batch = np.stack([image, image2], axis=0)

    effective_receptive_field_np(batch)
    effective_receptive_field_torch(batch)