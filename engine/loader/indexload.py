from dataclasses import dataclass
from tquant.index import IborIndex 
from tquant.markethandles import RateCurve
from tquant.utilities import TimeUnit, Settings, TARGET
from engine.globalconfig import GlobalConfig
import pandas as pd
from datetime import datetime

@dataclass
class MarketIndex:
    """
    Dataclass to store different instances of IborIndex.

    Attributes:
    -----------
        eur_1m: IborIndex
            euribor 1m 
        eur_3m: IborIndex
            euribor 3m 
        eur_6m: IborIndex
            euribor 6m 
        eur_12m: IborIndex
            euribor 12m 

    """
    eur_1w = IborIndex(name='EUR-EURIBOR-1W',
                        fixing_calendar=TARGET(),
                        tenor=1,
                        time_unit=TimeUnit.Weeks,
                        fixing_days=None,
                        time_series=None)
    
    eur_1m = IborIndex(name='EUR-EURIBOR-1M',
                        fixing_calendar=TARGET(),
                        tenor=1,
                        time_unit=TimeUnit.Months,
                        fixing_days=None,
                        time_series=None)
    
    eur_3m = IborIndex(name='EUR-EURIBOR-3M',
                        fixing_calendar=TARGET(),
                        tenor=3,
                        time_unit=TimeUnit.Months,
                        fixing_days=None,
                        time_series=None)
    
    eur_6m = IborIndex(name='EUR-EURIBOR-6M',
                        fixing_calendar=TARGET(),
                        tenor=6,
                        time_unit=TimeUnit.Months,
                        fixing_days=None,
                        time_series=None)
    
    eur_12m = IborIndex(name='EUR-EURIBOR-12M',
                        fixing_calendar=TARGET(),
                        tenor=12,
                        time_unit=TimeUnit.Months,
                        fixing_days=None,
                        time_series=None)

    def _load_data(self, config:GlobalConfig):
        """
        Load market curves from the source.

        Returns:
            dict: Market data loaded from the specified source.
        """ 
        if config.config['source'] == "local":
            return self._load_local_data()
        elif config.config['source']  == "database":
            return self._load_database_data()
        else:
            raise ValueError("Invalid source parameter. Source must be 'local' or 'database'.")
        
    def _load_local_data(self):
        """
        Load market data from a local source.

        Returns:
            dict: Market data loaded from a local source.
        """
        evaluation_date = Settings.evaluation_date
        y =  str(evaluation_date.year)
        m = evaluation_date.strftime("%m")
        d = str(evaluation_date.day)
        date_ = y + m + d
        file_name_fixings = '../data/' + date_ +'/market/MarketFixing.csv'
        fixings = pd.read_csv(file_name_fixings,on_bad_lines='skip')

        fixings['Date'] = fixings['Date'].apply(conversion_string_data)
    
        for name, index in vars(MarketIndex).items():
            if isinstance(index, IborIndex):  # Filtro solo gli attributi di tipo IborIndex
                fixings[fixings['Index'] == index.name].apply(lambda x: add_fixing_df(index, x), axis=1)
                
        

    def _load_database_data(self):
        """
        Load market data from a database.

        Returns:
            dict: Market data loaded from a database.
        """
        raise ValueError("Implement logic to load data from a database")
    
    def _load_curves(self, mdl):

        curves = {} #Sarà il dizionario con indici e curve

        for name, index in vars(MarketIndex).items():
            if isinstance(index, IborIndex):  # Filtro solo gli attributi di tipo IborIndex
                try:
                    rates = mdl._ir_curves(index.name)['quote'].values
                    times = mdl._ir_curves(index.name)['daycount'].values/365
                    curves[index.name] = RateCurve(times, rates)
                except:
                    curves[index.name] = 'error'
        return curves
    

def add_fixing_df(index,row):
    
    index.add_fixing(row['Date'], row['Fixing'])

def conversion_string_data(str_data):
    try:
        data = pd.to_datetime(str_data, format="%d-%b-%y")
    except:
        data = pd.to_datetime(str_data, format="%d-%b-%Y")
    data = data.to_pydatetime().date()

    return data
   