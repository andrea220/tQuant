from utilities.time import *
from interface.index import Index, Index_dev
from structures.ratecurves import DiscountCurveSimple
from datetime import datetime, timedelta


# class CurveRateIndex(Index):
#     def __init__(self,
#                  name: str,
#                  fixing_calendar: Calendar,
#                  term_structure: DiscountCurveSimple,
#                  tenor,
#                  fixing_days = None,
#                  time_series: dict = None) -> None:
#         super().__init__(name, fixing_calendar, time_series)      
#         self._fixing_days = fixing_days
#         self._term_structure = term_structure
#         self._tenor = tenor # da implementare oggetto tenor

#         #self._update_evaluation_date() #si setta la data di valutazione uguale alla prima data della curva
    
#     @property
#     def fixing_days(self):
#         if self._fixing_days == None:
#             return 0
#         else:
#             return self._fixing_days
    
#     def _update_evaluation_date(self):
#         Settings.evaluation_date = self.term_structure.pillars[0] # da aggiungere date alle curve e settare prima data della curva come evaluation_date 

#     def _fixing_maturity(self, fixing_date):
#         return fixing_date + timedelta(182) # da implementare calendar.advance

#     def forecast_fixing(self,
#                         fixing_date: date):
#         ''' 
#         Calcola il tasso forward
#         '''
#         d1 = fixing_date
#         d2 = self._fixing_maturity(d1)
#         t = (d2 - d1).days / 365 # il daycounter yearfraction va sviluppato 
#         df1 = self._term_structure.discount((d1- Settings.evaluation_date).days / 365).numpy() # il daycounter yearfraction va sviluppato 
#         df2 = self._term_structure.discount((d2 - Settings.evaluation_date).days / 365).numpy() # il daycounter yearfraction va sviluppato 
#         return (df1 / df2 -1) / t


class IborIndex(Index_dev):
    def __init__(self,
                 name: str,
                 fixing_calendar: Calendar,
                 tenor,
                 fixing_days = None,
                 time_series: dict = None) -> None:
        super().__init__(name,
                         fixing_calendar,
                         time_series)      
        self._fixing_days = fixing_days
        self._tenor = tenor # serve per la maturity del forward per il fixing
        
    
    @property
    def fixing_days(self):
        if self._fixing_days == None:
            return 0
        else:
            return self._fixing_days
    
    def _update_evaluation_date(self):
        Settings.evaluation_date = self.term_structure.pillars[0] # da aggiungere date alle curve e settare prima data della curva come evaluation_date 

    def _fixing_maturity(self, fixing_date):
        return fixing_date + timedelta(182) # da implementare calendar.advance

    def forecast_fixing(self,
                        fixing_date: date,
                        term_structure: DiscountCurveSimple):
        ''' 
        Calcola il tasso forward.
        TBD: cambiare interfaccia perché index.fixing() prenda la curva in input
        TBD: creare una funzione vettoriale che restituisca i forward per tutti i fixing
        '''
        d1 = fixing_date
        d2 = self._fixing_maturity(d1)
        t = (d2 - d1).days / 365 # il daycounter yearfraction va sviluppato 
        df1 = term_structure.discount((d1- Settings.evaluation_date).days / 365).numpy() # il daycounter yearfraction va sviluppato 
        df2 = term_structure.discount((d2 - Settings.evaluation_date).days / 365).numpy() # il daycounter yearfraction va sviluppato 
        return (df1 / df2 -1) / t