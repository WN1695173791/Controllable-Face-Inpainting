"""Miscellaneous utils."""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch._six import container_abcs, string_classes


def split_labels(labels, label_lengths):
    r"""Split concatenated labels into their parts.

    Args:
        labels (torch.Tensor): Labels obtained through concatenation.
        label_lengths (OrderedDict): Containing order of labels & their lengths.

    Returns:

    """
    assert isinstance(label_lengths, OrderedDict)
    start = 0
    outputs = {}
    for data_type, length in label_lengths.items():
        end = start + length
        if labels.dim() == 5:
            outputs[data_type] = labels[:, :, start:end]
        elif labels.dim() == 4:
            outputs[data_type] = labels[:, start:end]
        elif labels.dim() == 3:
            outputs[data_type] = labels[start:end]
        start = end
    return outputs


def requires_grad(model, require=True):
    r""" Set a model to require gradient or not.

    Args:
        model (nn.Module): Neural network model.
        require (bool): Whether the network requires gradient or not.

    Returns:

    """
    for p in model.parameters():
        p.requires_grad = require


def to_device(data, device):
    r"""Move all tensors inside data to device.

    Args:
        data (dict, list, or tensor): Input data.
        device (str): 'cpu' or 'cuda'.
    """
    assert device in ['cpu', 'cuda']
    if isinstance(data, torch.Tensor):
        data = data.to(torch.device(device))
        return data
    elif isinstance(data, container_abcs.Mapping):
        return {key: to_device(data[key], device) for key in data}
    elif isinstance(data, container_abcs.Sequence) and \
            not isinstance(data, string_classes):
        return [to_device(d, device) for d in data]
    else:
        return data


def to_cuda(data):
    r"""Move all tensors inside data to gpu.

    Args:
        data (dict, list, or tensor): Input data.
    """
    return to_device(data, 'cuda')


def to_cpu(data):
    r"""Move all tensors inside data to cpu.

    Args:
        data (dict, list, or tensor): Input data.
    """
    return to_device(data, 'cpu')


def to_half(data):
    r"""Move all floats to half.

    Args:
        data (dict, list or tensor): Input data.
    """
    if isinstance(data, torch.Tensor) and torch.is_floating_point(data):
        data = data.half()
        return data
    elif isinstance(data, container_abcs.Mapping):
        return {key: to_half(data[key]) for key in data}
    elif isinstance(data, container_abcs.Sequence) and \
            not isinstance(data, string_classes):
        return [to_half(d) for d in data]
    else:
        return data


def to_float(data):
    r"""Move all halfs to float.

    Args:
        data (dict, list or tensor): Input data.
    """
    if isinstance(data, torch.Tensor) and torch.is_floating_point(data):
        data = data.float()
        return data
    elif isinstance(data, container_abcs.Mapping):
        return {key: to_float(data[key]) for key in data}
    elif isinstance(data, container_abcs.Sequence) and \
            not isinstance(data, string_classes):
        return [to_float(d) for d in data]
    else:
        return data


def get_and_setattr(cfg, name, default):
    r"""Get attribute with default choice. If attribute does not exist, set it
    using the default value.

    Args:
        cfg (obj) : Config options.
        name (str) : Attribute name.
        default (obj) : Default attribute.

    Returns:
        (obj) : Desired attribute.
    """
    if not hasattr(cfg, name) or name not in cfg.__dict__:
        setattr(cfg, name, default)
    return getattr(cfg, name)


def get_nested_attr(cfg, attr_name, default):
    r"""Iteratively try to get the attribute from cfg. If not found, return
    default.

    Args:
        cfg (obj): Config file.
        attr_name (str): Attribute name (e.g. XXX.YYY.ZZZ).
        default (obj): Default return value for the attribute.

    Returns:
        (obj): Attribute value.
    """
    names = attr_name.split('.')
    atr = cfg
    for name in names:
        if not hasattr(atr, name):
            return default
        atr = getattr(atr, name)
    return atr


def gradient_norm(model):
    r"""Return the gradient norm of model.

    Args:
        model (PyTorch module): Your network.

    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)


def random_shift(x, offset=0.05, mode='bilinear', padding_mode='reflection'):
    r"""Randomly shift the input tensor.

    Args:
        x (4D tensor): The input batch of images.
        offset (int): The maximum offset ratio that is between [0, 1].
        The maximum shift is offset * image_size for each direction.
        mode (str): The resample mode for 'F.grid_sample'.
        padding_mode (str): The padding mode for 'F.grid_sample'.

    Returns:
        x (4D tensor) : The randomly shifted image.
    """
    assert x.dim() == 4, "Input must be a 4D tensor."
    batch_size = x.size(0)
    theta = torch.eye(2, 3, device=x.device).unsqueeze(0).repeat(
        batch_size, 1, 1)
    theta[:, :, 2] = 2 * offset * torch.rand(batch_size, 2) - offset
    grid = F.affine_grid(theta, x.size())
    x = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)
    return x


def truncated_gaussian(threshold, size, seed=None, device=None):
    r"""Apply the truncated gaussian trick to trade diversity for quality

    Args:
        threshold (float): Truncation threshold.
        size (list of integer): Tensor size.
        seed (int): Random seed.
        device:
    """
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-threshold, threshold,
                           size=size, random_state=state)
    return torch.tensor(values, device=device).float()


def apply_imagenet_normalization(input):
    r"""Normalize using ImageNet mean and std.

    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [-1, 1].

    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    # normalize the input back to [0, 1]
    normalized_input = (input + 1) / 2
    # normalize the input using the ImageNet mean and std
    mean = normalized_input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = normalized_input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (normalized_input - mean) / std
    return output


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def draw_landmarks(img, landmark, color='r', step=2):
    """
    Return:
        img              -- torch.tensor, (B, 3, H, W) img with landmark, RGB order, range (0, 255)
        

    Parameters:
        img              -- torch.tensor, (B, 3, H, W), RGB order, range (0, 255)
        landmark         -- torch.tensor, (B, 68, 2), y direction is opposite to v direction
        color            -- str, 'r' or 'b' (red or blue)
    """
    device = 'cpu'
    if torch.is_tensor(img):
        device = img.get_device() if img.is_cuda else 'cpu'
        img = img.detach().permute(0,2,3,1).cpu().numpy()
    if torch.is_tensor(landmark):
        landmark = landmark.detach().cpu().numpy()

    if color =='r':
        c = np.array([255., 0, 0])
    else:
        c = np.array([0, 0, 255.])

    _, H, W, _ = img.shape
    img, landmark = img.copy(), landmark.copy()
    landmark[..., 1] = H - 1 - landmark[..., 1]
    landmark = np.round(landmark).astype(np.int32)
    for i in range(landmark.shape[1]):
        x, y = landmark[:, i, 0], landmark[:, i, 1]
        for j in range(-step, step):
            for k in range(-step, step):
                u = np.clip(x + j, 0, W - 1)
                v = np.clip(y + k, 0, H - 1)
                for m in range(landmark.shape[0]):
                    img[m, v[m], u[m]] = c
    
    return torch.from_numpy(img).permute(0, 3, 1, 2).to(device)
