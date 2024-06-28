import torch
import torch.nn as nn
import pdb


class ValueGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t):
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    # this is the gradient that is used in equation (3) in paper, to guide the reverse process!
    def gradients(self, x, stop_grad,*args):
        x.requires_grad_()

        y = self(x, *args)
        
        # CHANGED GRAD WHILE VALUE FUNCTION IS A CONSTANT
        #grad=torch.ones(x.shape,requires_grad=True)
        grad = torch.autograd.grad([y.sum()], [x])[0]

        if stop_grad:
            grad=torch.zeros_like(grad)
            #print(grad)

        x.detach()
        return y, grad
