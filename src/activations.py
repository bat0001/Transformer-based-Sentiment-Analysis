import torch
import torch.nn.functional as F

class Softmax:
    def __init__(self, dim=1):
        self.dim = dim

    def forward_prop(self, x):
        self.output = F.softmax(x, dim=self.dim)
        return self.output

    def backward_prop(self, grad_output):
        softmax_out = self.output
        d_out_d_in = softmax_out * (torch.eye(softmax_out.size(1)).to(softmax_out.device) - softmax_out).unsqueeze(1)
        grad_input = torch.einsum('ijk,ik->ij', d_out_d_in, grad_output)
        return grad_input

class Linear:
    def __init__(self, input_dim, output_dim):
        self.weights = torch.randn(input_dim, output_dim, requires_grad=True) * 0.01
        self.bias = torch.zeros(output_dim, requires_grad=True)

    def forward(self, x):
        self.x = x
        return torch.matmul(x, self.weights) + self.bias

    def backward(self, grad_output):
        grad_input = torch.matmul(grad_output, self.weights.t())
        grad_weights = torch.matmul(self.x.t(), grad_output)
        grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weights, grad_bias
