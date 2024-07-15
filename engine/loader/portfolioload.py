from tquant.utilities import TimeUnit, Settings, TARGET, BusinessDayConvention, DayCounter,DayCounterConvention, CompoundingType
from tquant.instruments import InterestRateSwap2, Cap, Floor
from tquant.flows import FixedRateLeg, FloatingRateLeg
from engine.globalconfig import GlobalConfig
from .indexload import MarketIndex
import pandas as pd
from datetime import date, datetime
import re
import json

#TODO Aggionare inflation swap
#TODO Daycounter floating leg
#TODO Controllare compouinding
#TODO Chiedere se inserie i cross-currency, fx-forward (Inserire il sistema di currency allora), FX SWAP

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


class PortfolioLoader:
    def __init__(self,
                 config: GlobalConfig) -> None:
        self._config = config
        self.eur_calendar = TARGET()
        self.portfolio = {}#[]
        self.errors = {}
        for filename in self._load_portfolio():
            with open(filename , 'r') as file:
                # self.trade_data = json.load(file)
                self.make_portfolio(trade_data = json.load(file))
                # portfolio, errors = self.make_portfolio(trade_data = json.load(file))
                # self.portfolio.extend(portfolio)
                # self.errors.extend(errors)

        

    def _load_portfolio(self):
        """
        Load market curves from the source.

        Returns:
            dict: Market data loaded from the specified source.
        """
        if self._config.config['source'] == "local":
            return self._load_local_data()
        elif self._config.config['source']  == "database":
            return self._load_database_data()
        else:
            raise ValueError("Invalid source parameter. Source must be 'local' or 'database'.")
        
    def _load_local_data(self):
        """
        Load market data from a local source.

        Returns:
            dict: Market data loaded from a local source.
        """
        evaluation_date = self._config.config['evaluationDate']
        y =  evaluation_date[:4]
        m = evaluation_date[4:6]
        d = evaluation_date[6:8]
        Settings.evaluation_date = date(int(y), int(m), int(d))
        file_name = '../data/' + evaluation_date
        local_data = [file_name + '/trades/TradesAggregated_' + str(i) + '.json' for i in range(1,4)]
        return local_data
        

    def _load_database_data(self):
        """
        Load market data from a database.

        Returns:
            dict: Market data loaded from a database.
        """
        raise ValueError("Implement logic to load data from a database")
        
    def make_swap(self,
                  trade):
        ''' 
        Read json and returns tq.Swap
        '''
        currency = trade['data']['currency']
        
        leg1_schedule_len = len(trade['data']['Leg1.schedule']['rows'])
        leg2_schedule_len = len(trade['data']['Leg2.schedule']['rows'])

        leg1_is_fixed = trade['data']['Leg1.isFixedLeg']
        leg2_is_fixed = trade['data']['Leg2.isFixedLeg']

        if not leg1_is_fixed:
            leg1_index = trade['data']['Leg1.index']
        if not leg2_is_fixed:
            leg2_index = trade['data']['Leg2.index']

        date_format = '%Y-%m-%d'
        leg1_payment_dates = [trade['data']['Leg1.schedule']['rows'][i]['paymentDate'] for i in range(leg1_schedule_len)]
        leg1_payment_dates.insert(0, trade['data']['Leg1.schedule']['rows'][0]['accrualStartDate'])
        leg1_payment_dates = [datetime.strptime(dt, date_format).date() for dt in leg1_payment_dates]
        leg2_payment_dates = [trade['data']['Leg2.schedule']['rows'][i]['paymentDate'] for i in range(leg2_schedule_len)]
        leg2_payment_dates.insert(0, trade['data']['Leg2.schedule']['rows'][0]['accrualStartDate'])
        leg2_payment_dates = [datetime.strptime(dt, date_format).date() for dt in leg2_payment_dates]

        leg1_notionals = [trade['data']['Leg1.notionalTable']['rows'][i]['value'] for i in range(leg1_schedule_len)]
        leg2_notionals = [trade['data']['Leg2.notionalTable']['rows'][i]['value'] for i in range(leg2_schedule_len)]

        leg1_margins = [trade['data']['Leg1.marginTable']['rows'][i]['value'] for i in range(leg1_schedule_len)]
        leg2_margins = [trade['data']['Leg2.marginTable']['rows'][i]['value'] for i in range(leg2_schedule_len)]

        leg1_rate_factor = [trade['data']['Leg1.rateFactorTable']['rows'][i]['value'] for i in range(leg1_schedule_len)]
        leg2_rate_factor = [trade['data']['Leg2.rateFactorTable']['rows'][i]['value'] for i in range(leg2_schedule_len)]

        leg1_compounding_type = trade['data']['Leg1.compoundingType']

        if leg1_compounding_type == 'Linear':
            leg1_compouning = CompoundingType.Simple
        if leg1_compounding_type == 'Exponential':
            leg1_compounding_type = CompoundingType.Compounded

        leg2_compounding_type = trade['data']['Leg2.compoundingType']

        if leg2_compounding_type == 'Linear':
            leg2_compouning = CompoundingType.Simple
        if leg2_compounding_type == 'Exponential':
            leg2_compounding_type = CompoundingType.Compounded

        #DayCounter per entrambe le gambe
        leg1_day_counter_convention = trade['data']['Leg1.schedule.basis']

        if leg1_day_counter_convention == 'ACT/360':
            leg1_day_counter = DayCounter(DayCounterConvention.Actual360)
        elif leg1_day_counter_convention == 'ACT/365':
            leg1_day_counter = DayCounter(DayCounterConvention.Actual365)
        elif leg1_day_counter_convention == '30/360':
            leg1_day_counter = DayCounter(DayCounterConvention.Thirty360)
        elif leg1_day_counter_convention == '30E/360':
            leg1_day_counter = DayCounter(DayCounterConvention.Thirty360E)
        elif leg1_day_counter_convention == 'ACT/ACT':
            leg1_day_counter = DayCounter(DayCounterConvention.ActualActual)
        else:
            #print(leg1_day_counter_convention, "Da implementare come convenzione di Daycounter")
            raise ValueError('No convention available')

        try:
            leg2_day_counter_convention = trade['data']['Leg2.schedule.basis']

            if leg2_day_counter_convention == 'ACT/360':
                leg2_day_counter = DayCounter(DayCounterConvention.Actual360)
            elif leg2_day_counter_convention == 'ACT/365':
                leg2_day_counter = DayCounter(DayCounterConvention.Actual365)
            elif leg2_day_counter_convention == '30/360':
                leg2_day_counter = DayCounter(DayCounterConvention.Thirty360)
            elif leg2_day_counter_convention == '30E/360':
                leg2_day_counter = DayCounter(DayCounterConvention.Thirty360E)
            elif leg2_day_counter_convention == 'ACT/ACT':
                leg2_day_counter = DayCounter(DayCounterConvention.ActualActual)
            else:
                #print(leg2_day_counter_convention, "Da implementare come convenzione di Daycounter")
                raise ValueError('No convention available')
            
        except:

            leg2_day_counter = DayCounter(DayCounterConvention.Actual360)

        # Creo la gamba 2
        if not leg2_is_fixed:

            if leg2_index == 'EUR-EURIBOR-1W':
                index2 = MarketIndex.eur_1w
            elif leg2_index == 'EUR-EURIBOR-1M':
                index2 = MarketIndex.eur_1m
            elif leg2_index == 'EUR-EURIBOR-3M':
                index2 = MarketIndex.eur_3m
            elif leg2_index == 'EUR-EURIBOR-6M':
                index2 = MarketIndex.eur_6m
            elif leg2_index == 'EUR-EURIBOR-12M':
                index2 = MarketIndex.eur_12m
            else:
                #print(leg2_index, "Indice Da implementare")
                raise ValueError('No index available')
            
            leg2_gearing = [1]*len(leg2_notionals)
            leg2 = FloatingRateLeg(leg2_payment_dates, leg2_notionals, leg2_gearing, leg2_margins, index2, DayCounter(DayCounterConvention.Actual360))
        
        else:
            
            leg2 = FixedRateLeg(leg2_payment_dates, leg2_notionals,leg2_margins,leg2_day_counter,leg2_compouning)
            
        #Creo la gamba 1
        if not leg1_is_fixed:

            if leg1_index == 'EUR-EURIBOR-1W':
                index1 = MarketIndex.eur_1w
            if leg1_index == 'EUR-EURIBOR-1M':
                index1 = MarketIndex.eur_1m
            elif leg1_index == 'EUR-EURIBOR-3M':
                index1 = MarketIndex.eur_3m
            elif leg1_index == 'EUR-EURIBOR-6M':
                index1 = MarketIndex.eur_6m
            elif leg1_index == 'EUR-EURIBOR-12M':
                index1= MarketIndex.eur_12m
            else:
                #print(leg1_index, "Indice da implementare")
                raise ValueError('No index available')
            
            leg1_gearing = [1]*len(leg1_notionals)
            leg1 = FloatingRateLeg(leg1_payment_dates, leg1_notionals, leg1_gearing, leg1_margins, index1, DayCounter(DayCounterConvention.Actual360))

        else:
            leg1 = FixedRateLeg(leg1_payment_dates, leg1_notionals,leg1_margins,leg1_day_counter,leg1_compouning)
        
            
        if leg1_is_fixed:
            swap_obj = InterestRateSwap2(leg1, leg2, currency)
        else:
            swap_obj = InterestRateSwap2(leg2, leg1, currency)
                    
        return swap_obj
    

    def make_capfloor(self,
                      trade):
        ''' 
        Read json and returns tq.CapFloor
        '''
        
        index_name = trade['data']['OptionOnIndex.index']

        if index_name == 'EUR-EURIBOR-1W':
            index = MarketIndex.eur_1w
        elif index_name == 'EUR-EURIBOR-1M':
            index = MarketIndex.eur_1m
        elif index_name == 'EUR-EURIBOR-3M':
            index = MarketIndex.eur_3m
        elif index_name == 'EUR-EURIBOR-6M':
            index = MarketIndex.eur_6m
        elif index_name == 'EUR-EURIBOR-12M':
            index = MarketIndex.eur_12m
        else:
            #print(index_name, "Indice da implementare")
            raise ValueError('No index available')
        
        date_format = '%Y-%m-%d'
        start_date = trade['data']['OptionOnIndex.schedule.startDate']
        start_date = datetime.strptime(start_date, date_format).date()
        end_date = trade['data']['OptionOnIndex.schedule.endDate']
        end_date = datetime.strptime(end_date, date_format).date()

        day_counter_convention = trade['data']['OptionOnIndex.schedule.basis']

        if day_counter_convention == 'ACT/360':
            day_counter = DayCounter(DayCounterConvention.Actual360)
        elif day_counter_convention == 'ACT/365':
            day_counter = DayCounter(DayCounterConvention.Actual365)
        elif day_counter_convention == '30/365':
            day_counter = DayCounter(DayCounterConvention.Thirty360)
        elif day_counter_convention == '30E/365':
            day_counter = DayCounter(DayCounterConvention.Thirty360E)
        elif day_counter_convention == 'ACT/ACT':
            day_counter = DayCounter(DayCounterConvention.ActualActual)
        else:
            raise ValueError('No convention available')

        lencapfloor = len(trade['data']['OptionOnIndex.amortNotional']['rows'])
        
        dates = [trade['data']['OptionOnIndex.schedule']['rows'][i]['paymentDate'] for i in range(lencapfloor)]
        dates = [datetime.strptime(dt, date_format).date() for dt in dates]

        notionals = [trade['data']['OptionOnIndex.amortNotional']['rows'][i]['notional'] for i in range(lencapfloor)]
        
        try:
            strikes = [trade['data']['OptionOnIndex.stepStrikes']['rows'][i]['strike'] for i in range(lencapfloor)]
        except:
            strikes = [0 for i in range(lencapfloor)]
            #print("Strikes Missing")

        gearing = [0]*len(dates) #TODO parlare del gearing e spreads
        spreads = [0]*len(dates)
        
        leg = FloatingRateLeg(dates, notionals, gearing, spreads, index,day_counter)
        
        if trade['data']['OptionOnIndex.optionType'] =='FLOOR':
            capfloor_obj = Floor(leg,strikes[0])
        else:
            capfloor_obj = Cap(leg,strikes[0])

            
        return capfloor_obj
    
    def make_cross_currency_amm_swap(self,
                                     trade):
        cross_currency_amm_swap_obj = 0
        return cross_currency_amm_swap_obj
    
    def make_fx_forward(self,
                        trade):
        fx_forward_obj = 0
        return fx_forward_obj
    
    def make_fx_swap(self,
                     trade):
        fx_swap_obj = 0
        return fx_swap_obj
    
    def make_european_vanilla_option(self,
                     trade):
        european_vanilla_option_obj = 0
        return european_vanilla_option_obj
    
    def make_european_swaption(self,
                     trade):
        european_swaption_obj = 0
        return european_swaption_obj

    def make_inflation_swap(self,
                     trade):
        inflation_swap_obj = 0
        return inflation_swap_obj

    def make_portfolio(self, trade_data):
        i = 0
        for trade_json in trade_data:

            try:
                if trade_json['data']['productType'] == 'InterestRateSwapScript':
                    trade_json['trade_object'] = self.make_swap(trade_json)
                    trade_json['trade_object'].product_type = 'InterestRateSwap'
                elif trade_json['data']['productType'] == 'CapFloor':
                    trade_json['trade_object'] = self.make_capfloor(trade_json)
                elif trade_json['data']['productType'] == 'CrossCurrencyAmortizingSwap':
                    trade_json['trade_object'] = self.make_cross_currency_amm_swap(trade_json)
                elif trade_json['data']['productType'] == 'FXForward':
                    trade_json['trade_object'] = self.make_fx_forward(trade_json)
                elif trade_json['data']['productType'] == 'FXSwap':
                    trade_json['trade_object'] = self.make_fx_swap(trade_json)
                elif trade_json['data']['productType'] == 'EuropeanVanillaOption':
                    trade_json['trade_object'] = self.make_european_vanilla_option(trade_json)
                elif trade_json['data']['productType'] == 'EuropeanSwaption':
                    trade_json['trade_object'] = self.make_european_swaption(trade_json)
                if trade_json['data']['productType'] == 'InflationSwap':
                    trade_json['trade_object'] = self.make_inflation_swap(trade_json)
                self.portfolio[trade_json['ref']] = trade_json['trade_object']

                # portfolio.append({"ref": trade_json['ref'] , "obj": trade_json['trade_object']})
            except:
                self.errors[trade_json['ref']] = 'error'
                # errors.append(trade_json['ref'])
                # print("Errore",i)
            i += 1
        return
        # return portfolio, errors