import tensorflow as tf
# import numpy as np
# from scipy.interpolate import interp1d
from ..interface.curve import AbstractCurve
from typing import Optional, Union
from ..interface.helper import AbstractHelper
from ..utilities.daycounter import DayCounter
from ..utilities.interpolation import InterpolationType, interpolate
from datetime import date, timedelta
from ..utilities.utils import Settings, TimeUnit, BusinessDayConvention, CompoundingType
from ..interface.tqcalendar import Calendar


class YieldCurve(AbstractCurve):

    def __init__(self,
                 calendar: Calendar,
                 settlement_days: Optional[int] = None, 
                 pillars: Optional[Union[date, list[date]]] = None, 
                 dfs: Optional[Union[float, list[float]]] = None,
                 helpers: Optional[Union[list[AbstractHelper],AbstractHelper]] = None, 
                 day_counter: Optional[DayCounter] = None,
                 T1: Optional[Union[int, float, list[int], list[float]]] = None,
                 interpolation: Optional[InterpolationType] = None):
        # Inizializza le liste solo se non sono fornite come argomenti
        super().__init__(calendar, day_counter, pillars, dfs, T1)
        self.interpolation = interpolation if interpolation is not None else InterpolationType.Linear
        self.settlement_days = settlement_days if settlement_days is not None else 0
        self.today = self.calendar.advance(Settings.evaluation_date,self.settlement_days,TimeUnit.Days,BusinessDayConvention.Following)

        if dfs is not None and pillars is not None:
          self.dfs =  tf.Variable(dfs, dtype=tf.float64)

        if helpers is not None:

            self.pillars = [self.today]
            self.dfs = [1.0]
            self.T1 = [0]
            pillarshelper = []

            for helper in helpers:
              pillarshelper.append((helper.pillar() - self.today).days)

            # Ordino in base al pillar
            helper_pillar = list(zip(pillarshelper, helpers))
            coppie_ordinate = sorted(helper_pillar, key=lambda x: x[0])
            pillars_ordinata, helpers_ordinata = zip(*coppie_ordinate)
            # Converti le tuple in liste
            pillars_ordinata = list(pillars_ordinata)
            helpers_ordinata = list(helpers_ordinata)

            for helper in helpers_ordinata:
               helper.bootstrap(self)

            self.sort()

            self.dfs =  tf.Variable(self.dfs, dtype=tf.float64)

    def sort(self):
        #Devo ordinare in base al pillar
        difference_days = [(pillar - self.today).days for pillar in self.pillars]
        pillars_dfs = list(zip(difference_days, self.pillars, self.dfs, self.T1))
        coppie_ordinate = sorted(pillars_dfs, key=lambda x: x[0])
        difference_days, pillars_ordinata, dfs_ordinata, T1_ordinata = zip(*coppie_ordinate)
        # Converti le tuple in liste
        pillars_ordinata = list(pillars_ordinata)
        dfs_ordinata = list(dfs_ordinata)
        T1_ordinata = list(T1_ordinata)
        self.pillars = pillars_ordinata
        self.dfs = dfs_ordinata
        self.T1 = T1_ordinata

        #Spostiamo tutte le T1 a T0
        for T1 in self.T1:
            if T1 != 0:
                posizione = self.T1.index(T1)
                pillar = self.pillars[posizione]
                prepillar = 0
                i = 0
                while prepillar <= T1:
                    i = i + 1
                    prepillar = (self.pillars[i] - self.today).days
                if (self.pillars[i-1] - self.today).days != 0:
                    rate = (1.0/self.dfs[i-1]-1.0)/self.day_counter.year_fraction(self.today, self.pillars[i-1])
                    df = 1.0/(rate* self.day_counter.year_fraction(self.today, self.today + timedelta(days = T1)) + 1.0)*self.dfs[posizione]
                else:
                    df = self.dfs[posizione]

                self.pillars.pop(posizione)
                self.dfs.pop(posizione)
                self.T1.pop(posizione)
                self.pillars.insert(posizione, pillar)
                self.dfs.insert(posizione, df)
                self.T1.insert(posizione, 0)

    # Il fattore di sconto ad una data precisa
    def discount(self, 
                 d1: date, #TODO inserire yearfraction
                 interpolation: Optional[InterpolationType] = None ): #TODO interpolazione nel costruttore della classe
      if interpolation is None:
        interpolation = self.interpolation
      if d1 > self.pillars[-1]:
        interpolation = InterpolationType.Linear #L'estrapolazione è fatta linearmente su Quantlib. Questo spiega le discrepanze in estrapolazione che avevamo
      if d1 < self.today:
         raise ValueError("Date not correct")
      else:
          difference_days = [(pillar - self.today).days for pillar in self.pillars]
          difference = (d1 - self.today).days
          return interpolate(difference_days,self.dfs,difference,interpolation,logaritm=True)

    # Calcolo dello zero rate dal fattore di sconto
    def zero_rate(self,
                 d1: date,
                 interpolation: Optional[InterpolationType] = None,
                 compounding: Optional[CompoundingType] = None):
      if compounding is None:
         compounding = CompoundingType.Compounded
      if interpolation is None:
        interpolation = self.interpolation

      if compounding == CompoundingType.Compounded:
          result =(1.0/self.discount(d1, interpolation=interpolation))**(1.0/self.day_counter.year_fraction(self.today, d1)) - 1.0
      
      if compounding == CompoundingType.Continuous:
          result = -(1.0/self.day_counter.year_fraction(self.today, d1))*tf.math.log(self.discount(d1, interpolation=interpolation))

      if compounding == CompoundingType.Simple:
          result = (1.0/self.discount(d1, interpolation=interpolation)-1.0)*(1.0/self.day_counter.year_fraction(self.today, d1))

      result = tf.cast(result,dtype = tf.float64)
      return result
         
    def forward_rate(self, #TODO controllare inst. forward e interpolazione
                     start_date: date,
                     end_date: date,
                     interpolation: Optional[InterpolationType] = None,
                     compounding: Optional[CompoundingType] = None):
          if compounding is None:
            compounding = CompoundingType.Simple
          if interpolation is None:
            interpolation = self.interpolation

          if compounding == CompoundingType.Compounded:
            F = (self.discount(start_date,interpolation=interpolation)/self.discount(end_date,interpolation=interpolation))**(1.0/self.day_counter.year_fraction(start_date,end_date)) - 1.0 
          
          if compounding == CompoundingType.Continuous:
            F = -tf.math.log(self.discount(end_date,interpolation=interpolation)/self.discount(start_date,interpolation=interpolation))/self.day_counter.year_fraction(start_date,end_date)

          if compounding == CompoundingType.Simple:
            F = (self.discount(start_date,interpolation=interpolation)/self.discount(end_date,interpolation=interpolation) - 1.0)/self.day_counter.year_fraction(start_date,end_date)
          
          F = tf.cast(F, dtype = tf.float64)
          return F
    
    def annuity(self,
                start_date: date,
                end_date: date,
                schedule: int,
                time_unit: TimeUnit,
                convention: BusinessDayConvention,
                fixing_days: Optional[int] = None,
                day_counter: Optional[DayCounter] = None,
                interpolation: Optional[InterpolationType] = None):
      # Imposto l'interpolazione
      if interpolation is None:
        interpolation = self.interpolation
      
      #Imposto il daycounter
      if day_counter is None:
        day_counter = self.day_counter
      # Calcolo Annuity e la data finale del calcolo
      Ac = 0.0
      d1 = start_date
      d1O = start_date
      while d1 < self.calendar.advance(end_date, -schedule, time_unit, convention):
        d1 = self.calendar.advance(d1, schedule, time_unit, convention)
        Ac = Ac + self.discount(self.calendar.advance(d1, - fixing_days, TimeUnit.Days, convention), interpolation=interpolation)*day_counter.year_fraction(d1O,d1)
        d1O = d1
      
      return(Ac, d1)
    
    def insert_pillar(self,
                     pillar: date, 
                     rate: float,
                     T1: Optional[Union[int,float]] = None):
       if isinstance(rate,tf.Variable) or isinstance(rate,tf.Tensor):
          rate = rate.numpy()
       return super().insert_pillar(pillar, rate, T1)