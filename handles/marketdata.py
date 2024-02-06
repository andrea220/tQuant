from utilities.time import TimeUnit, Settings, BusinessDayConvention
from utilities.targetcalendar import TARGET, Calendar
from data.config import FileNames
from datetime import date
from datetime import datetime

import pandas as pd
import re


def set_period_unit(period_string):
    ''' 
    Trasforma il formato stringa-TimeUnit
    '''
    if period_string == 'Y':
        period_unit = TimeUnit.Years
    elif period_string == 'M':
        period_unit = TimeUnit.Months
    elif period_string == 'W':
        period_unit = TimeUnit.Weeks
    elif period_string == 'BD':
        period_unit = TimeUnit.Days
    return period_unit

class MarketDataLoader:
    def __init__(self,
                 eur_calendar: Calendar = TARGET() ) -> None:
        # read raw data
        self.raw_dfs = pd.read_csv('..' + FileNames.ROOT + FileNames.MARKET_DFS, 
                                skiprows= 2,
                                on_bad_lines='skip')
        
        self.raw_quotes = pd.read_csv('..' + FileNames.ROOT + FileNames.MARKET_QUOTES, 
                                skiprows= 2,
                                        on_bad_lines='skip')
        ## conventions
        Settings.evaluation_date = date(int(FileNames.MARKET_DFS.split('_')[1]), 
                                    int(FileNames.MARKET_DFS.split('_')[2]),
                                    int(FileNames.MARKET_DFS.split('_')[3])
                                )
        self.evaluation_date = Settings.evaluation_date
        self.eur_calendar = eur_calendar

        self._market_quotes = None
        self._market_discount = None

        self._ir_curve_data = None
        self._ir_curve_data_disc = None

        self._ir_vol_data = None
        self._ir_vol_data_disc = None

    @property
    def market_quotes(self):
        if self._market_quotes is None:
            self._filter_quotes()
        return self._market_quotes
    
    def _filter_quotes(self):
        ''' 
        splitta properties 
        '''
        quotes_properties = pd.DataFrame([self.raw_quotes['Property'][i].split(".") for i in range(self.raw_quotes.shape[0])] )
        market_quotes = pd.concat([quotes_properties, self.raw_quotes['Label']], axis=1)
        market_quotes.columns = ['type',
                                'name',
                                'curve_instrument',
                                'quote_type',
                                'bo1', # ??? 
                                'bo2',
                                'bo3',
                                'market_quote']
        self._market_quotes = market_quotes
    
    @property
    def market_discount(self):
        if self._market_discount is None:
            self._filter_discount()
        return self._market_discount
    
    def _filter_discount(self):
        quotes_properties = pd.DataFrame([self.raw_dfs['Property'][i].split(".") for i in range(self.raw_dfs.shape[0])] )
        market_discount = pd.concat([quotes_properties, self.raw_dfs['Label']], axis=1)
        market_discount.columns = ['type',
                                'name',
                                'curve_instrument',
                                'quote_type',
                                'bo1', # ??? 
                                'bo2',
                                'bo3',
                                'market_quote'] 
        self._market_discount = market_discount

    @property
    def ir_curve_data(self):
        if self._ir_curve_data is None:
            self._ir_quotes()
        return self._ir_curve_data
    
    @property
    def ir_vol_data(self):
        if self._ir_vol_data is None:
            self._ir_quotes()
        return self._ir_vol_data
    
    def _ir_quotes(self):
        ir_data = self.market_quotes[self.market_quotes['type'] == 'IR']
        ir_curve_data = ir_data[ir_data['quote_type'] == 'MID']

        self._ir_curve_data = ir_curve_data[['type', 'name', 'curve_instrument', 'quote_type', 'market_quote']]
        self._ir_vol_data = ir_data[ir_data['quote_type'] == 'SWPT']

    @property
    def ir_curve_data_disc(self):
        if self._ir_curve_data_disc is None:
            self._ir_discount()
        return self._ir_curve_data_disc
    
    @property
    def ir_vol_data_disc(self):
        if self._ir_vol_data_disc is None:
            self._ir_discount()
        return self._ir_vol_data_disc
    
    def _ir_discount(self):
        ir_data = self.market_discount[self.market_discount['type'] == 'IR']
        ir_curve_data = ir_data[ir_data['quote_type'] != 'SWPT']

        self._ir_curve_data_disc = ir_curve_data[['type', 'name', 'curve_instrument', 'quote_type', 'market_quote']]
        self._ir_vol_data_disc = ir_data[ir_data['quote_type'] == 'SWPT']

    @property
    def ir_eur_curve_1m(self):
        return self._ir_curves('EUR-EURIBOR-1M')
    @property
    def ir_eur_discount_1m(self):
        return self._ir_discounts('EUR-EURIBOR-1M')
    
    @property
    def ir_eur_curve_3m(self):
        return self._ir_curves('EUR-EURIBOR-3M')
    @property
    def ir_eur_discount_3m(self):
        return self._ir_discounts('EUR-EURIBOR-3M')
    
    @property
    def ir_eur_curve_6m(self):
        return self._ir_curves('EUR-EURIBOR-6M')
    @property
    def ir_eur_discount_6m(self):
        return self._ir_discounts('EUR-EURIBOR-6M')

    @property
    def ir_eur_curve_12m(self):
        return self._ir_curves('EUR-EURIBOR-12M')
    @property
    def ir_eur_discount_12m(self):
        return self._ir_discounts('EUR-EURIBOR-12M')
    
    @property
    def ir_eur_curve_estr(self):
        return self._ir_curves('EUR-ESTR-ON')
    @property
    def ir_eur_discount_estr(self):
        return self._ir_discounts('EUR-ESTR-ON')
    
    def _ir_curves(self, curve_name):
        ''' 
        Fa operazioni sulle stringhe per filtrare il df
        '''
        curve = self.ir_curve_data[self.ir_curve_data['name'] == curve_name]
        temp = re.compile("([0-9]+)([a-zA-Z]+)")
        df_curve = pd.DataFrame()
        for i in range( curve.shape[0]):
            split_temp = curve['curve_instrument'].iloc[i].split('-')
            if split_temp[0] == 'CASH' or split_temp[0] == 'SWAP':
                start_date = Settings.evaluation_date

                res = temp.match(split_temp[1]).groups()
                n_period = int(res[0])
                period_unit = set_period_unit(res[1]) 

                maturity = self.eur_calendar.advance(Settings.evaluation_date,
                                                    n_period, 
                                                    period_unit,
                                                    BusinessDayConvention.ModifiedFollowing 
                                                    )
                dt = (maturity - Settings.evaluation_date).days
                df_temp = pd.DataFrame( [split_temp[0], start_date, maturity, split_temp[1], dt, curve['market_quote'].iloc[i]] ).T
                df_curve = pd.concat([df_curve, df_temp], axis = 0)

            elif split_temp[0] == 'FRA':
                res_start = temp.match(split_temp[1]).groups()
                res_end = temp.match(split_temp[2]).groups()

                period_unit_start = set_period_unit(res_start[1]) 
                period_unit_end = set_period_unit(res_end[1]) 

                start_date = self.eur_calendar.advance(Settings.evaluation_date,
                                                    int(res_start[0]), 
                                                    period_unit_start,
                                                    BusinessDayConvention.ModifiedFollowing 
                                                    )
                maturity = self.eur_calendar.advance(start_date,
                                                    int(res_end[0]), 
                                                    period_unit_end,
                                                    BusinessDayConvention.ModifiedFollowing 
                                                    )
                dt = (maturity - Settings.evaluation_date).days
                df_temp = pd.DataFrame( [split_temp[0], start_date, maturity, curve['curve_instrument'].iloc[i], dt, curve['market_quote'].iloc[i]] ).T
                df_curve = pd.concat([df_curve, df_temp], axis = 0)


        df_curve.columns = ['type',
                            'start',
                            'maturity',
                            'tenor',
                            'daycount',
                            'quote']
        return df_curve

    def _ir_discounts(self, curve_name):
        ### 
        curve = self.ir_curve_data_disc[self.ir_curve_data_disc['name'] == curve_name].copy()
        curve = curve[curve['curve_instrument'] != 'PROJECTION']
        curve.reset_index(drop=True, inplace=True)
        curve_dates_str = [curve['curve_instrument'].iloc[i].replace('DF-','') for i in range(curve.shape[0])]
        curve_dates = pd.DataFrame([datetime.strptime(str_temp, '%d-%b-%Y').date() for str_temp in curve_dates_str],
                                columns = ['maturity_date'])
        dt = pd.DataFrame([i.days for i in (curve_dates.loc[:,'maturity_date'] - Settings.evaluation_date) ],
                        columns = ['daycount'])
        curve = pd.concat([curve, curve_dates, dt], axis = 1)
        return curve