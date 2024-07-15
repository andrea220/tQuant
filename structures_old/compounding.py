from enum import Enum

class Compounding(Enum):
    Simple = 0               # 1+rt
    Compounded = 1           # (1+r)^t
    Continuous = 2           # e^{rt}


class Frequency(Enum):
    NoFrequency = -1
    Once = 0
    Annual = 1
    Semiannual = 2
    EveryFourthMonth = 3
    Quarterly = 4
    Bimonthly = 6
    Monthly = 12
    EveryFourthWeek = 13
    Biweekly = 26
    Weekly = 52
    Daily = 365
    OtherFrequency = 999