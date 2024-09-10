
class LinearInterp:
    ''' 
    interpolazione lineare per tensorflow
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def interpolate(self, term):
        for i in range(0, len(self.x) - 1):
            if term < self.x[i + 1]:
                dtr = 1 / (self.x[i + 1] - self.x[i])
                w1 = (self.x[i + 1] - term) * dtr
                w2 = (term - self.x[i]) * dtr
                r1 = w1 * self.y[i]
                r2 = w2 * self.y[i + 1]
                return r1 + r2
