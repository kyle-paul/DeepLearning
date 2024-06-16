import numpy as np

def initialize_weight(in_channels, out_channels, kernel_size):

    fan_in = in_channels * kernel_size * kernel_size
    gain = np.sqrt(2.0 / (1 + 0**2)) 
    
    bound = gain * np.sqrt(3.0 / fan_in)
    weight = np.random.uniform(
        -bound, bound, 
        size=(out_channels, in_channels, 
        kernel_size, kernel_size)
    ).astype(np.float32)

    return weight


class Conv2d():
    def __init__(self, in_channels, out_channels, padding, stride, kernel_size):
        super().__init__()
    
        self.weight = initialize_weight(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size        
        
    def compute(self):
        if self.padding > 0:
            self.padded_x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            self.padded_x = self.x
        
        # Get the dimensions of the input
        self.batch_size, _, in_height, in_width = self.x.shape
        
        # Calculate the dimensions of the output array
        self.out_height = (in_height + 2*self.padding - self.kernel_size) // self.stride + 1
        self.out_width = (in_width + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize the output array
        self.output = np.zeros((self.batch_size, self.out_channels, self.out_height, self.out_width))
        
    def backpropagation(self, grad_z):
        grad_padded_x = np.float32(np.zeros_like(self.padded_x))
        grad_weight = np.float32(np.zeros_like(self.weight))

        for b in range(self.batch_size):
            for o in range(self.out_channels):
                for i in range(self.out_height):
                    start_i = i * self.stride
                    end_i = start_i + self.kernel_size
                    for j in range(self.out_width):
                        start_j = j * self.stride
                        end_j = start_j + self.kernel_size
                        patch = self.padded_x[b, :, start_i:end_i, start_j:end_j]
                        grad_weight[o, :] += patch * grad_z[b, o, i, j]
                        grad_padded_x[b, :, start_i:end_i, start_j:end_j] += self.weight[o, :] * grad_z[b, o, i, j]
                        
        if self.padding > 0:
            grad_padded_x = grad_padded_x[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return grad_padded_x, grad_weight
        
    def forward(self, x):
        self.x = x
        self.compute()
        
        for b in range(self.batch_size):
            for o in range(self.out_channels):
                for i in range(self.out_height):
                    start_i = i * self.stride
                    end_i = start_i + self.kernel_size
                    for j in range(self.out_width):
                        start_j = j * self.stride
                        end_j = start_j + self.kernel_size
                        patch = self.padded_x[b, :, start_i:end_i, start_j:end_j]
                        self.output[b, o, i, j] += np.sum(patch * self.weight[o, :])
    
        return self.output