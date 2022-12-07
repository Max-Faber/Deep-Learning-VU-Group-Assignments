import torch

class Unfold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        U = torch.nn.functional.unfold(input, kernel_size=3)
        ctx.save_for_backward(input)
        return U

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * input

class Multiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, W):
        Y_prime = torch.matmul(U, W)
        ctx.save_for_backward(U)
        ctx.save_for_backward(W)
        return Y_prime

    @staticmethod
    def backward(ctx, grad_output):
        U, W, = ctx.saved_tensors
        return torch.matmul(grad_output, W), torch.matmul(grad_output, U)

class Reshape(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        Y = torch.reshape(input)
        ctx.save_for_backward(input)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * input

if __name__ == '__main__':
    X = torch.rand(10, 10)
    W = torch.rand(3, 3)
    U = Unfold(X)
    Y_prime = Multiply(U, W)
    Y = Reshape(Y_prime)
    Unfold()
