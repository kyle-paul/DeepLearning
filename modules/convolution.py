import numpy as np

def initialize_weight(kernel_size):
    k1 = np.array([
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
    ])
    k2 = np.array([
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
    ])
    k3 = np.array([
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
    ])
    k4 = np.array([
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
    ])
    k5 = np.array([
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
    ])
    k6 = np.array([
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
    ])
    k7 = np.array([
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
    ])
    k8 = np.array([
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
    ])
    k9 = np.array([
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],

        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
    ])

    k10 = np.array([
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
        
        [[0.5, 0.6, 0.8],
        [0.3, 0.2, 0.1],
        [0.2, 0.8, 0.9]],
    ])
    
    weight = np.stack([k1, k2, k3, k4, k5, k6, k7, k8, k9, k10])
    weight = np.float32(weight)
    return weight


class Conv2d():
    def __init__(self, in_channels, out_channels, padding, stride, kernel_size):
        super().__init__()
    
        self.weight = initialize_weight(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size        
        
    def compute(self):
        # Add padding to the input array
        if self.padding > 0:
            self.x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                            mode='constant', constant_values=0)
            
        # Get the dimensions of the input and weight
        self.batch_size, _, in_height, in_width = self.x.shape
        _, _, self.kernel_height, self.kernel_width = self.weight.shape
        
        # Calculate the dimensions of the output array
        self.out_height = (in_height - self.kernel_height) // self.stride + 1
        self.out_width = (in_width - self.kernel_width) // self.stride + 1
        
        # Initialize the output array
        self.output = np.zeros((self.batch_size, self.out_channels, self.out_height, self.out_width))
        
    def backpropagation(self, grad_z):
        grad_x = np.zeros_like(self.x)
        grad_x = np.float32(grad_x)
        grad_weight = np.zeros_like(self.weight)
        grad_weight = np.float32(grad_weight)

        for b in range(self.batch_size):
            for o in range(self.out_channels):
                for c in range(self.in_channels): 
                
                    for i in range(self.out_height):
                        start_i = i * self.stride
                        end_i = start_i + self.kernel_height
                        for j in range(self.out_width):
                            start_j = j * self.stride
                            end_j = start_j + self.kernel_width
                            patch = self.x[b, c, start_i:end_i, start_j:end_j]
                            grad_weight[o, c] += patch * grad_z[b, o, i, j]
                            grad_x[b, c, start_i:end_i, start_j:end_j] += self.weight[o, c] * grad_z[b, o, i, j]
        
        return grad_x, grad_weight
        
    def forward(self, x):
        self.x = x
        self.compute()
        
        for b in range(self.batch_size):
            for o in range(self.out_channels):
                for c in range(self.in_channels):
                
                    for i in range(self.out_height):
                        start_i = i * self.stride
                        end_i = start_i + self.kernel_height
                        for j in range(self.out_width):
                            start_j = j * self.stride
                            end_j = start_j + self.kernel_width
                            patch = x[b, c, start_i:end_i, start_j:end_j]
                            self.output[b, o, i, j] += np.sum(patch * self.weight[o, c])
        
        return self.output