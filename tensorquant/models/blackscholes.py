from .stochasticprocess import StochasticProcess

class BlackScholesProcess(StochasticProcess):

    def __init__(self, x0, term_structure, dividend_yield, vol_term_structure):
        self._x0 = x0 
        self._term_structure = term_structure
        self._dividend_yield = dividend_yield
        self._vol_term_structure = vol_term_structure

    def drift(self, t0, x0, dt):
        return #TODO black scholes process implementation
    
    def diffusion(self, t0, x0, dt):
        return 
    
    def initial_values(self):
        return self._x0 
    
    def size(self):
        return 1