import datetime

from tquant import RateCurve
from tquant.bootstrapping.curve_assignment_loader import CurveAssignmentLoader
from tquant.bootstrapping.product_builders_loader import ProductBuildersLoader
from tquant.bootstrapping.zero_curve_stripper import ZeroCurveStripper


class ZeroCurveStripperManager:
    def __init__(self, stripper: ZeroCurveStripper):
        builders_loader = ProductBuildersLoader("product_builders_data.json")
        self.builders = builders_loader.load_product_builders()
        curve_assignment_loader = CurveAssignmentLoader('curve_assignment_data.json')
        self.curve_assignment = curve_assignment_loader.load_curve_assignment()
        self.stripper = stripper

    def strip(self,
              bootstrapping_curve_name: str,
              trade_date: datetime,
              generators: list[str],
              maturities: list[str],
              quotes: list[float],
              market_data: dict[str, RateCurve],
              solver_type: str = "LOCAL",
              is_spread_curve: bool = False,
              base_curve_name: str = ""):
        return self.stripper.strip(bootstrapping_curve_name,
                                   trade_date,
                                   generators,
                                   maturities,
                                   quotes,
                                   self.curve_assignment,
                                   self.builders,
                                   market_data,
                                   solver_type,
                                   is_spread_curve,
                                   base_curve_name)
