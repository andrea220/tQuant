from dataclasses import dataclass
from tquant.index import InflationIndex 
from tquant.utilities import TARGET, TimeUnit, Frequency

#TODO Implementare daycounter indice?

@dataclass
class Inflation_Index:
    """
    Dataclass to store different instances of InflationIndex.

    Attributes:
    -----------
        euhicp: InflationIndex

    """

    euhicp = InflationIndex(name='HICP',
                        tenor=1,
                        time_unit=TimeUnit.Months,
                        fixing_calendar=TARGET(),
                        frequency = Frequency.Monthly,
                        fixing_days=None,
                        time_series=None,
                        revised = False)
   