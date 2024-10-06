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

        y = self(x, *args) # this calls ValueDiffusion

        #create_graph=True makes it so that a second derivative (now w.r.t. value model parameters) can be taken
        grad = torch.autograd.grad([y.sum()], [x],create_graph=True)[0]

        if stop_grad:
            grad=torch.zeros_like(grad)

        return y, grad
