import abc
import torch

class Model(torch.nn.Module):
    """
    a basic template class for models
    """
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """
        initiate the loss function class with needed parameters
        """
        super().__init__()
        pass


    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        implement the loss calculation and return the value
        """
        raise NotImplementedError()

