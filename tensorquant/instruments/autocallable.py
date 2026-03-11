import datetime
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .product import Product
from ..markethandles.utils import Currency


class AutocallableOption(Product):
    """
    Equity autocallable option instrument.

    Questa classe contiene solo le **anagrafiche di prodotto** (schedule e parametri di payoff);
    la logica di pricing è demandata a un pricer esterno (es. il notebook Monte Carlo).

    Convenzioni:
    - Tutti i livelli sono espressi in **percentuale dello strike/fixing**:
      - `autocall_barrier = 100` → 100% dello strike
      - `coupon_barriers[i] = 70` → 70% dello strike
      - `coupon_rates[i] = 5` → 5% di cedola
    - `date_grid` è la griglia di simulazione già costruita esternamente
      (merge tra coupon, autocall e griglia di discretizzazione aggiuntiva).
    """

    def __init__(
        self,
        ccy: Currency,
        notional: float,
        start_date: datetime.date,
        end_date: datetime.date,
        strike: float,
        # --- COUPON LEG ---
        coupon_fixing_dates: Sequence[datetime.date],
        coupon_payment_dates: Sequence[datetime.date],
        coupon_rates: Sequence[float],
        coupon_barriers: Sequence[float],
        memory: bool,
        # --- AUTOCALL LEG ---
        autocall_fixing_dates: Sequence[datetime.date],
        autocall_payment_dates: Sequence[datetime.date],
        autocall_barrier: Sequence[float],
        # --- FINAL REDEMPTION (OPTION ONLY) ---
        payoff_barrier: float,
        payoff_participation: float = 1.0,
        payoff_type: str = "put",
        underlying: Optional[str] = None,
    ) -> None:
        """
        Initialize an AutocallableOption.

        Tutte le schedule (coupon/autocall fixing e payment) sono passate dall'esterno,
        così puoi costruirle con `ScheduleGenerator` nel modo che preferisci
        (incluso merge con altre date di discretizzazione per il modello).
        """
        super().__init__(ccy=ccy, start_date=start_date, end_date=end_date)

        self.notional = notional
        self.strike = strike
        self.underlying = underlying

        # schedule principali
        self.coupon_fixing_dates: List[datetime.date] = list(coupon_fixing_dates)
        self.coupon_payment_dates: List[datetime.date] = list(coupon_payment_dates)
        self.autocall_fixing_dates: List[datetime.date] = list(autocall_fixing_dates)
        self.autocall_payment_dates: List[datetime.date] = list(autocall_payment_dates)

        # sanity check sulle lunghezze
        n_coupons = len(self.coupon_fixing_dates)
        if len(self.coupon_payment_dates) != n_coupons:
            raise ValueError(
                "coupon_fixing_dates and coupon_payment_dates must have the same length "
                f"({n_coupons} vs {len(self.coupon_payment_dates)})"
            )

        if len(self.autocall_fixing_dates) != len(self.autocall_payment_dates):
            raise ValueError(
                "autocall_fixing_dates and autocall_payment_dates must have the same length "
                f"({len(self.autocall_fixing_dates)} vs {len(self.autocall_payment_dates)})"
            )

        # parametri di payoff scalar
        self.payoff_barrier = float(payoff_barrier)
        self.payoff_participation = float(payoff_participation)
        self.memory = memory

        # vettori coupon/barriere (con broadcast)
        def _broadcast(x: Sequence[float], name: str, n: int) -> List[float]:
            if len(x) == 1 and n > 1:
                return [float(x[0])] * n
            if len(x) != n:
                raise ValueError(
                    f"{name} length ({len(x)}) must be 1 or equal to target length ({n})"
                )
            return [float(v) for v in x]

        n_coupons = len(self.coupon_fixing_dates)
        n_autocall = len(self.autocall_fixing_dates)

        self.coupon_rates: List[float] = _broadcast(coupon_rates, "coupon_rates", n_coupons)
        self.coupon_barriers: List[float] = _broadcast(coupon_barriers, "coupon_barriers", n_coupons)
        self.autocall_barrier: List[float] = _broadcast(
            autocall_barrier, "autocall_barrier", n_autocall
        )

        # payoff finale (solo derivato, senza capitale)
        self.redemption_payoff: Optional[Callable[..., float]] = None
        if payoff_type.lower() == "put":
            # stessa firma usata nel notebook: final_redemption(S_T, strike, partecipation)
            self.redemption_payoff = self.final_redemption

    @property
    def maturity_date(self) -> datetime.date:
        """Alias per end_date, più esplicito in contesto equity."""
        return self.end_date

    def schedule_dataframe(self) -> "pd.DataFrame":
        """
        Ritorna un DataFrame con i dettagli delle schedule coupon e autocall,
        mostrandole **affiancate**: se una data è sia coupon che autocall,
        viene riportata sulla stessa riga.

        Colonne principali:
            - fixing_date, payment_date: chiave comune
            - coupon_rate_pct, coupon_barrier_pct
            - autocall_barrier_pct
            - memory (costante per il prodotto)
            - payoff_barrier_pct
        """
        rows: List[dict] = []

        # chiavi (fixing, payment) per ciascuna gamba
        coupon_keys = list(zip(self.coupon_fixing_dates, self.coupon_payment_dates))
        autocall_keys = list(zip(self.autocall_fixing_dates, self.autocall_payment_dates))

        all_keys = sorted(set(coupon_keys) | set(autocall_keys))

        for key in all_keys:
            fix, pay = key

            try:
                i_c = coupon_keys.index(key)
            except ValueError:
                i_c = None

            try:
                j_a = autocall_keys.index(key)
            except ValueError:
                j_a = None

            rows.append(
                {
                    "fixing_date": fix,
                    "payment_date": pay,
                    "coupon_rate_pct": self.coupon_rates[i_c] if i_c is not None else None,
                    "coupon_barrier_pct": self.coupon_barriers[i_c] if i_c is not None else None,
                    "autocall_barrier_pct": self.autocall_barrier[j_a] if j_a is not None else None,
                    "memory": self.memory,
                }
            )

        df = pd.DataFrame(rows)

        # payoff_barrier si applica solo alla riga finale (maturity)
        if not df.empty:
            df["payoff_barrier_pct"] = None
            df.loc[df.index[-1], "payoff_barrier_pct"] = self.payoff_barrier

        return df

    @staticmethod
    def final_redemption(S_T, strike: float, partecipation: float):
        """
        Payoff di una put “nuda” in percentuale del notional:

            final_redemption(S_T, K, partecipation)
                = max(K - S_T, 0) / K * partecipation
        """
        return np.maximum(strike - S_T, 0) / strike * partecipation
