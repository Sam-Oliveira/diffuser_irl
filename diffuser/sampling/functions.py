import torch

from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)

# So this function is used as the sampling function for guided sampling(instead of default_sample_fn() in diffusion.py). It only does 1 step of the reverse diffusion! 
# I think n_guide_steps is just number of steps in direction of grad? Idk. Also don't understand structure of grad and what is happening when we index into it with t<t_stop_grad since t is some scalar, no?

# DONT UNDERSTAND WHY THEY FIRST ALTER X, AND THEN FEED THAT TO GET THE MEAN, INSTEADD OF GETTING ITS MEAN AND ALTERING IT WITH GRAD?

def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True, stop_grad=False
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    # just about how many steps we take in direction of guide gradient I think
    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, stop_grad,cond, t)
        if scale_grad_by_std:
            grad = model_var * grad
        #grad.register_hook(lambda grad: print(grad))
        grad[t < t_stopgrad] = 0
        #grad.register_hook(lambda grad: print(grad))
        #x.register_hook(lambda grad: print(grad))
        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y # y is the predicted value! 
