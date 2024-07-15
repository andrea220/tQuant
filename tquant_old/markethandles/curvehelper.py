# Helper per il bootstrapping dei discount usando i Depositi, OIS, FRA e Swap
#TODO Utilizzare gli indici libor per tutti i parametri che servono, così da passare solo l'indice e non anche la durata temporale o le convenzioni
from typing import Optional
from ..interface.curve import AbstractCurve
from ..interface.helper import AbstractHelper
from ..utilities.daycounter import DayCounter
from datetime import date, timedelta
from ..utilities.utils import Settings, TimeUnit, BusinessDayConvention
from ..interface.tqcalendar import Calendar
from ..utilities.interpolation import InterpolationType

class DepositRateHelper(AbstractHelper):

    def __init__(self, 
                 rate: float, 
                 period: int,
                 time_unit: TimeUnit, 
                 calendar: Calendar,
                 convention: BusinessDayConvention, 
                 settlement_days: Optional[int] = None,
                 day_counter: Optional[DayCounter] = None):
      super().__init__(rate, calendar, settlement_days, day_counter)
      self.period = period
      self.time_unit = time_unit
      self.convention = convention 

    def pillar(self): #TODO controllare giorno di settlement
       d1 = Settings.evaluation_date + timedelta(days =self.settlement_days)
       d1 = self.calendar.advance(d1, self.period, self.time_unit, self.convention)
       return(d1)
    
    def bootstrap(self, curve: AbstractCurve): #TODO dividere funzione
        # Viene utilizzato il simple per calcolare il dfs
        d2 = Settings.evaluation_date + timedelta(days =self.settlement_days)
        d2 = self.calendar.advance(d2, self.period, self.time_unit, self.convention)
        dfs = 1.0 / (self.rate * self.day_counter.year_fraction(curve.today,d2) + 1.0)
        curve.insert_pillar(self.pillar(), dfs, 0)


class OISRateHelper(AbstractHelper):
    # Se interpreto bene da Quantlib, il pagamento è annuale solitamente - Per ora implemento il monocurva
    def __init__(self, 
                 rate: float, 
                 period: int,
                 time_unit: TimeUnit,
                 calendar: Calendar,
                 convention: BusinessDayConvention,
                 schedule: int,
                 time_unit_s: TimeUnit, #TODO togliere da qua, è il time unit annuo per l'annuity
                 settlement_days: Optional[int] = None,
                 day_counter: Optional[DayCounter] = None,
                 fixing_days: Optional[int] = None):
      super().__init__(rate, calendar, settlement_days, day_counter)
      self.period = period 
      self.time_unit = time_unit
      self.time_unit_s = time_unit_s
      self.convention = convention 
      self.fixing_days = fixing_days if fixing_days is not None else 0 # Questi sono i fixing Days che dipendono dalla curva (Bisognerà passargli una curva come argomento)
      self.schedule = schedule

    def pillar(self):
       # Pillar per la curva di sconto
       d1 = Settings.evaluation_date + timedelta(days =self.settlement_days)
       d1 = self.calendar.advance(d1, self.period, self.time_unit, self.convention)
       return(d1)

    def bootstrap(self,curve: AbstractCurve):
        
        # Bootstrapping dei dfs
        di = self.calendar.advance(Settings.evaluation_date, self.settlement_days, TimeUnit.Days, self.convention)
        df = self.calendar.advance(Settings.evaluation_date + timedelta(days=self.settlement_days), self.period, self.time_unit, self.convention)
        Ac, d1 = curve.annuity(di, df, self.schedule, self.time_unit_s, self.convention, self.fixing_days)
        # Calcolo il dfs implicito nello strumento
        dfs = (1.0 - Ac*self.rate)/(1.0 + self.rate* self.day_counter.year_fraction(d1,df))
        # Inserisco il dfs nella curve
        curve.insert_pillar(self.pillar(), dfs, self.settlement_days)

class DatedOISRateHelper(AbstractHelper):

    def __init__(self,
                 rate:float,
                 start_date: date, 
                 end_date: date,
                 calendar: Calendar,
                 convention: BusinessDayConvention,
                 schedule: int, 
                 time_unit_s: TimeUnit, 
                 settlement_days: Optional[int] = None, 
                 day_counter: Optional[DayCounter] = None,
                 fixing_days: Optional[int] = None):
      super().__init__(rate,  calendar, settlement_days, day_counter)
      self.start_date = start_date
      self.end_date = end_date
      self.fixing_days = fixing_days if fixing_days is not None else 2 # Questi sono i fixing Days che dipendono dalla curva (Bisognerà passargli una curva come argomento)
      self.schedule = schedule
      self.time_unit_s = time_unit_s
      self.convention = convention
    
    def pillar(self):
       # Pillar per la curva di sconto
       return(self.end_date)
    
    def bootstrap(self, curve: AbstractCurve):
      # Bootstrapping dei dfs
      Ac, d1 = curve.annuity(self.start_date, self.end_date, self.schedule, self.time_unit_s, self.convention, self.fixing_days) #TODO controllare gestione Annuity in ql
      # Calcolo il dfs implicito nello strumento
      dfs = (1.0 - Ac*self.rate)/(1.0 + self.rate*self.day_counter.year_fraction(d1, self.end_date))
      # Inserisco il dfs nella curve
      curve.insert_pillar(self.pillar(),dfs,(self.start_date-Settings.evaluation_date).days) #TODO rivedere logica P(T_a)

class FRAHelper(AbstractHelper):
    # Per ora è definita come dati solo lo start date e un unico periodo. Si può implementare qualcosa con star-end e period.
    def __init__(self,
                 rate: float, 
                 start_date: date,
                 period: int,
                 time_unit: TimeUnit,
                 calendar: Calendar,
                 convention: BusinessDayConvention,
                 settlement_days: Optional[int] = None,
                 day_counter: Optional[DayCounter] = None,
                 fixing_days: Optional[int] = None):
      super().__init__(rate, calendar, settlement_days, day_counter)
      self.start_date = start_date #In giorni da oggi
      self.period = period #In giorni
      self.fixing_days = fixing_days if fixing_days is not None else 2 # Questi sono i fixing Days che dipendono dalla curva (Bisognerà passargli una curva come argomento)
      self.time_unit = time_unit
      self.convention = convention

    def pillar(self):
       d1 = self.start_date + timedelta(days= self.settlement_days) + timedelta(days= self.fixing_days)
       d1 = self.calendar.adjust(d1,self.convention)
       d1 = self.calendar.advance(d1, self.period, self.time_unit, self.convention)
       return(d1)
    
    def bootstrap(self, curve: AbstractCurve):
      d1 = self.start_date + timedelta(days= self.settlement_days) + timedelta(days= self.fixing_days)
      d1 = self.calendar.adjust(d1,self.convention)
      if d1 < curve.today:
        Rpre = 1.0
      else:
        Rpre = curve.discount(self.calendar.advance(d1,-self.fixing_days,TimeUnit.Days,self.convention), interpolation= InterpolationType.Linear)
      dfs = Rpre/(1.0+self.rate*self.day_counter.year_fraction(d1,self.pillar()))
      
      #Inserisco il dfs nella curva
      curve.insert_pillar(self.pillar(), dfs, self.settlement_days)


class SwapHelper(AbstractHelper):

    def __init__(self, 
                 rate: float, 
                 period: int, 
                 time_unit: TimeUnit,
                 discount_curve: AbstractCurve, 
                 fixed_schedule: int, 
                 fixed_time_unit: TimeUnit,
                 convention: BusinessDayConvention, 
                 floating_schedule:int, 
                 floating_time_unit:TimeUnit,
                 calendar: Calendar,
                 floating_convention: BusinessDayConvention,
                 settlement_days: Optional[int] = None, 
                 fixed_day_counter: Optional[DayCounter] = None, 
                 floating_day_counter: Optional[DayCounter] = None, 
                 fixing_days: Optional[int] = None):
      super().__init__(rate, calendar, settlement_days, fixed_day_counter)
      self.period = period 
      self.time_unit = time_unit
      self.discount_curve = discount_curve
      self.fixed_schedule = fixed_schedule
      self.fixed_time_unit = fixed_time_unit
      self.convention = convention
      self.floating_schedule = floating_schedule
      self.floating_time_unit = floating_time_unit
      self.floating_day_counter = floating_day_counter
      self.floating_convention = floating_convention
      self.fixing_days = fixing_days if fixing_days is not None else 2 # Questi sono i fixing Days che dipendono dalla curva (Bisognerà passargli una curva come argomento)

    def pillar(self):
       d1 = Settings.evaluation_date + timedelta(days= self.settlement_days) + timedelta(days= self.fixing_days)
       d1 = self.calendar.advance(d1, self.period, self.time_unit, self.floating_convention)
       return(d1)
    
    def price_leg(self, 
                 curve:AbstractCurve, 
                 start_date: date, 
                 end_date: date):
       
      R = 0.0
      d1 = start_date
      while d1 <= end_date:
        d2 = self.calendar.advance(d1, self.fixing_days,TimeUnit.Days,self.convention)
        di = self.calendar.advance(d1, -self.floating_schedule,self.floating_time_unit,self.convention)
        df = d1
        R = self.discount_curve.discount(d2)*curve.forward_rate(di, df, interpolation= InterpolationType.Linear)*self.floating_day_counter.year_fraction(di,df) + R
        d1 = self.calendar.advance(d1, self.floating_schedule, self.floating_time_unit, self.convention)
      return R
          
    
    def bootstrap(self, 
                  curve: AbstractCurve):
        # Seguiamo Eq. del paper
        dpre = self.calendar.advance(self.pillar(),-self.fixed_schedule,self.fixed_time_unit,self.convention)
        dAc = self.calendar.advance(dpre,self.fixing_days,TimeUnit.Days,self.convention)
        Acpre, _ = self.discount_curve.annuity(self.discount_curve.today, dAc, self.fixed_schedule, self.fixed_time_unit, self.convention, self.fixing_days, self.day_counter)
        Rpre = self.price_leg(curve, self.calendar.advance(curve.today, self.floating_schedule,self.floating_time_unit,self.convention), dpre)/Acpre
        dAc = self.calendar.advance(self.pillar(),self.fixing_days,TimeUnit.Days,self.convention)
        Ac, _ = self.discount_curve.annuity(self.discount_curve.today, dAc,  self.fixed_schedule, self.fixed_time_unit, self.convention, self.fixing_days, self.day_counter)
        R = self.rate
        dpre = self.calendar.advance(self.pillar(), self.fixing_days, TimeUnit.Days,self.convention)
        D = self.discount_curve.discount(dpre)
        dpre = self.calendar.advance(self.pillar(),-self.floating_schedule,self.floating_time_unit,self.convention)
        X = curve.discount(dpre, interpolation= InterpolationType.Linear)
        di = self.calendar.advance(self.pillar(), self.floating_schedule,self.floating_time_unit,BusinessDayConvention.Unadjusted)
        di = self.calendar.advance(di,-self.fixed_schedule,self.fixed_time_unit,self.convention)
        df = self.calendar.advance(self.pillar(),-self.floating_schedule,self.floating_time_unit,self.convention)
        Sum = self.price_leg(curve, di, df)

        dfs = D*X/(R*Ac - Rpre*Acpre + D - Sum)
        
        #Inserisco il dfs nella curva
        curve.insert_pillar(self.pillar(), dfs, self.settlement_days)

        #Rpre = self.price_leg(curve, self.calendar.advance(curve.today, self.floating_schedule,self.floating_time_unit,self.convention), self.pillar())/Ac
        #differenza = (R - Rpre)/R
        #while( abs(differenza) > 1e-5):
          #dfs = curve.dfs.pop(len(curve.dfs)-1)
          #curve.dfs.append(dfs - differenza*dfs/100.0)
          #Rpre = self.price_leg(curve, self.calendar.advance(curve.today, self.floating_schedule,self.floating_time_unit,self.convention), self.pillar())/Ac
          #differenza1 = (R - Rpre)/R
          #if abs(differenza1)>abs(differenza):
            #dfs = curve.dfs.pop(len(curve.dfs)-1)
            #curve.dfs.append(dfs*100.0/(100.0-differenza) + differenza*dfs/(100.0-differenza))
            #Rpre = self.price_leg(curve, self.calendar.advance(curve.today, self.floating_schedule,self.floating_time_unit,self.convention), self.pillar())/Ac
            #differenza1 = (R - Rpre)/R
          #differenza = differenza1
        #print(R,Rpre)

        # Su quantlib usa le funzioni fixedlegBPS e floatinglegBPS - Mi sa non la abbiamo al momento

        # Si può iterare la procedura
        i = 0
        while i < 0:
           i = i + 1
           dpre = self.calendar.advance(self.pillar(),-self.fixed_schedule,self.fixed_time_unit,self.convention)
           Rpre = self.price_leg(curve, self.calendar.advance(curve.today, self.floating_schedule,self.floating_time_unit,self.convention), dpre)/Acpre
           dpre = self.calendar.advance(self.pillar(),-self.floating_schedule,self.floating_time_unit,self.convention)
           X = curve.discount(dpre, interpolation= InterpolationType.Linear)
           Sum = self.price_leg(curve, di, df)
           dfsnew = D*X/(R*Ac - Rpre*Acpre + D - Sum)
           curve.dfs.pop(len(curve.dfs)-1)
           curve.dfs.append(dfsnew)

        