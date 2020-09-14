from abc import ABC, abstractmethod
import torch

class Environment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self, state, action, agent_index):
        pass
