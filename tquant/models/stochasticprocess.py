from abc import ABC, abstractmethod

class StochasticProcess(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def size(self):
        pass
    
    @property
    def factors(self):
        return self.size()
    
    @abstractmethod
    def initial_values(self):
        pass

    @abstractmethod
    def drift(self, t0, x0, dt):
        pass

    @abstractmethod
    def diffusion(self, t0, x0, dt):
        pass

    def expectation(self, t0, x0, dt):
        """ aggiungere euler discretization"""
        return x0 + self.drift(t0, x0, dt) 

    def std_deviation(self, t0, x0, dt):
        return self.diffusion(t0, x0, dt)
    
    def evolve(self, t0, x0, dt, dw):
        return self.expectation(t0=t0, x0=x0, dt=dt) + self.std_deviation(t0=t0, x0=x0, dt=dt)*dw
