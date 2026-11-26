import numpy as np
import tensorflow as tf

from .pricer import Pricer
from ..instruments.autocallable import AutocallableOption, AutocallableSwap
from ..markethandles.marketenvironment import MarketEnvironment
from ..models.brownian import FINE_HORIZON, FINE_TIME_STEP, simulation_grid
from ..models.localvolatility import LocalVolatilityModel
from ..models.equitymodel import bind, check_model
from ..models.multiasset import MultiAssetProcess
from ..numericalhandles.dtypes import DTYPE
from .quanto import apply_quanto, composite_fx_paths, quanto_drift
from ..timehandles.utils import Settings


class AutocallableMCPricer(Pricer):
    """Monte Carlo pricer for :class:`AutocallableOption` (phoenix: conditional coupons with optional
    memory, autocall, capital-at-risk put at maturity), from the **investor's side**:

    ``NPV = notional * E[ sum_i coupon_i P(0, pay_i) - put P(0, T) ]``,
    ``put = participation * max(K_put - S_T, 0) / K`` if ``S_T <= payoff barrier`` and not called (``K`` the
    strike (initial level), the reference of every barrier; ``K_put = put_strike % * K``, default ``K``).
    Forward start (``product.strike_date`` after the evaluation date): ``K = S(strike_date)`` on each path, the
    strike date is simulated with the fixings.

    Live trades: the observations (and the strike date) up to the evaluation date use the fixings already known,
    ``product.fixings`` first, then the market's (``market_env.get_eq_fixing``, the closes); one on the evaluation date
    that is in neither is the spot. They set the state of the trade on the same payoff code (exact indicators):
    strike, coupons paid or missed (memory), autocall already triggered; payments before the evaluation date are
    settled and left out. A past observation without a fixing raises.

    The model is a class, built from the market at every pricing call
    (``model.from_equity(market_env.get_equity(product.underlying), maturity, strike)``, greeks through the
    calibration: one ``pricer.tape.gradient(product.price, market_env.risk_factors())``), or an instance
    (:class:`EquityModel`) bound with ``model.process(equity)`` (parameters held: model greeks on
    ``model.parameters``, par greeks with :func:`par_greeks`); see :class:`VanillaMCPricer`.
    Paths are simulated (stateless seed) on the union of the future fixing dates and a grid of daily steps
    (``fine_time_step``) up to ``fine_horizon`` years, of ``max_time_step`` after (:func:`simulation_grid`), with the
    vol-surface day count.

    Coupon, autocall and put barriers are digitals: their pathwise derivative is zero, so AAD would
    miss most of delta and vega. With ``barrier_smoothing = eps > 0`` every indicator
    ``1{S >= B}`` becomes ``sigmoid((S - B) / (eps K))`` (eps = 1% of the strike by default):
    a small, controlled price bias for smooth, differentiable greeks. ``eps = 0`` gives the
    exact (hard) payoff.

    Args:
        model: Model class with ``from_equity`` (default :class:`LocalVolatilityModel`) or a model instance.
        n_paths: Number of paths.
        seed: Seed of the stateless Gaussian draws.
        max_time_step: Largest simulation step (years).
        barrier_smoothing: Width of the smoothed digitals as a fraction of the strike (0 = hard).
        fine_time_step: Step over the first months (years).
        fine_horizon: End of the fine steps (years; 0 = uniform grid of ``max_time_step``).

    After pricing: ``product.coupon_pv``, ``product.put_pv`` (both per unit notional, put as a
    positive value paid by the investor), ``product.standard_error`` (currency units);
    ``pricer.paths`` ``[n_paths, n_fixings]`` at the future fixings ``pricer.date_grid`` (times in
    ``pricer.time_grid``) for diagnostics.
    """

    def __init__(
        self,
        model=LocalVolatilityModel,
        n_paths: int = 20_000,
        seed: int = 42,
        max_time_step: float = 1.0 / 52.0,
        barrier_smoothing: float = 0.01,
        fine_time_step: float = FINE_TIME_STEP,
        fine_horizon: float = FINE_HORIZON,
    ) -> None:
        super().__init__()
        for m in (model.values() if isinstance(model, dict) else [model]):     # one model, or one per ticker
            check_model(m)
        if barrier_smoothing < 0:
            raise ValueError("barrier_smoothing must be >= 0")
        self._model = model
        self._n_paths = n_paths
        self._seed = seed
        self._max_time_step = max_time_step
        self._fine_time_step = fine_time_step
        self._fine_horizon = fine_horizon
        self._eps = barrier_smoothing
        self.paths = None
        self.time_grid = None
        self.date_grid = None

    # ------------------------------------------------------------------
    # Pricer interface
    # ------------------------------------------------------------------

    def calculate_price(self, product: AutocallableOption, market_env: MarketEnvironment) -> tf.Tensor:
        """Price an :class:`AutocallableOption` by Monte Carlo (investor's side).

        Args:
            product: The autocallable (``underlying`` must be set).
            market_env: Market data (underlying via ``get_equity``).

        Returns:
            tf.Tensor: float64 NPV in currency units, differentiable w.r.t. the market risk factors.

        Raises:
            ValueError: If *product* is not an :class:`AutocallableOption`, has no underlying,
                or the underlying has no vol surface.
        """
        if not isinstance(product, AutocallableOption):
            raise ValueError("AutocallableMCPricer only supports AutocallableOption")
        if isinstance(product, AutocallableSwap) and not isinstance(self, AutocallableSwapPricer):
            raise ValueError("an AutocallableSwap has a funding leg: price it with AutocallableSwapPricer")
        self._market = market_env
        if product.underlying is None:
            raise ValueError("AutocallableOption.underlying must be set")
        if product.is_worst_of:
            return self._worst_of(product, market_env)
        eq = market_env.get_equity(product.underlying, ccy=product.ccy)
        if eq.vol_surface is None:
            raise ValueError(f"No volatility surface for underlying '{product.underlying}'")

        eval_date = Settings.evaluation_date
        daycounter = eq.vol_surface.daycounter
        strike_date = getattr(product, "strike_date", None)
        forward_start = strike_date is not None and strike_date > eval_date
        known = self._known_levels(product, market_env, [eq], eval_date)
        if known and product.is_composite:
            raise ValueError("composite products with past fixings are not supported (the FX fixings are needed)")
        if product.strike is not None:
            strike = float(product.strike)
        elif not forward_start:
            strike = known[strike_date][0]                       # struck in the past: the fixing of the strike date
        fixings = self._future_dates(product, eval_date, forward_start)
        curve = eq.discount_curve
        perf = {d: v[0] / strike for d, v in known.items() if d in self._observations(product)}
        self.paths, self.time_grid, self.date_grid = None, np.zeros(0), []
        n = 1
        if fixings:
            fixing_times = np.array([daycounter.year_fraction(eval_date, d) for d in fixings])
            uniform = simulation_grid(fixing_times[-1], self._max_time_step, self._fine_time_step,
                                      self._fine_horizon)
            time_grid = np.unique(np.concatenate([uniform, fixing_times]))
            column = {d: i for i, d in enumerate(fixings)}     # paths are returned at the fixings only

            level = float(strike) if not forward_start else float(eq.spot)
            if product.is_composite and product.strike is not None:     # the level of S X: back into the underlying's
                level /= float(market_env.get_fx_spot(eq.currency, product.payment_ccy))
            model = bind(self._model_for(eq.ticker), eq, product.end_date, level)
            curve = apply_quanto(market_env, product, eq, model, float(fixing_times[-1]))   # quanto: drift and curve
            k = model.noise_size()
            dw = tf.random.stateless_normal([self._n_paths, len(time_grid)] + ([k] if k > 1 else []),
                                            seed=[self._seed, 0], dtype=DTYPE)
            paths = model.evolve(tf.constant(time_grid, DTYPE), dw, observation_times=fixing_times)
            if product.is_composite:                        # the underlying in the payment currency: S X
                paths = paths * composite_fx_paths(market_env, product, eq, time_grid, dw, fixing_times, self._seed)
            self.paths, self.time_grid, self.date_grid = paths, fixing_times, fixings
            k_path = paths[:, column[strike_date]] if forward_start else strike
            perf.update({d: paths[:, column[d]] / k_path for d in fixings if d in self._observations(product)})
            n = self._n_paths
        elif product.is_quanto:
            curve = market_env.get_ir_curve(product.payment_ccy)
        self._equities = [eq]
        return self._value(product, perf, set(known), curve, eval_date, n)

    def _model_for(self, ticker: str):
        """The model of *ticker*: the pricer's one, or its entry when the pricer has one per ticker."""
        if isinstance(self._model, dict):
            if ticker not in self._model:
                raise ValueError(f"no model for {ticker}")
            return self._model[ticker]
        return self._model

    def _worst_of(self, product: AutocallableOption, market_env: MarketEnvironment) -> tf.Tensor:
        """Worst-of: the underlyings simulated jointly (:class:`MultiAssetProcess`, the market correlations), each
        underlying in another currency than the product's with its quanto drift; the levels on the worst performance
        ``min_i S_i / K_i``; payments discounted on the curve of the product's currency."""
        tickers = product.underlyings
        equities = [market_env.get_equity(t) for t in tickers]
        if any(eq.vol_surface is None for eq in equities):
            raise ValueError(f"no volatility surface for an underlying of {tickers}")
        payment_ccy = product.payment_ccy
        eval_date = Settings.evaluation_date
        daycounter = equities[0].vol_surface.daycounter
        strike_date = getattr(product, "strike_date", None)
        forward_start = strike_date is not None and strike_date > eval_date
        known = self._known_levels(product, market_env, equities, eval_date)
        if product.strike is not None:
            strikes = [float(k) for k in product.strikes]
        elif not forward_start:
            strikes = known[strike_date]                         # struck in the past: the fixings of the strike date
        observations = self._observations(product)
        perf = {d: tf.reduce_min(tf.stack([v / k for v, k in zip(values, strikes)]))
                for d, values in known.items() if d in observations}
        fixings = self._future_dates(product, eval_date, forward_start)
        product.quanto = {}
        self.paths, self.time_grid, self.date_grid, self.performance = None, np.zeros(0), [], None
        self._by_underlying, self._column = None, {}
        n = 1
        if fixings:
            fixing_times = np.array([daycounter.year_fraction(eval_date, d) for d in fixings])
            uniform = simulation_grid(fixing_times[-1], self._max_time_step, self._fine_time_step,
                                      self._fine_horizon)
            time_grid = np.unique(np.concatenate([uniform, fixing_times]))
            column = {d: i for i, d in enumerate(fixings)}

            levels = [float(eq.spot) for eq in equities] if forward_start else [float(k) for k in strikes]
            processes = []
            for eq, level in zip(equities, levels):
                process = bind(self._model_for(eq.ticker), eq, product.end_date, level)
                if eq.currency != payment_ccy:                # quanto: paid in the product's currency
                    product.quanto[eq.ticker] = quanto_drift(market_env, eq, payment_ccy, process,
                                                             float(fixing_times[-1]))
                processes.append(process)
            names = [f"EQ:{eq.currency.name}:{eq.ticker}" for eq in equities]
            model = MultiAssetProcess(processes, market_env.get_correlation(names))
            dw = tf.random.stateless_normal([self._n_paths, len(time_grid), len(tickers)], seed=[self._seed, 0],
                                            dtype=DTYPE)
            paths = model.evolve(tf.constant(time_grid, DTYPE), dw, observation_times=fixing_times)    # [P, F, n]
            self.paths, self.time_grid, self.date_grid = paths, fixing_times, fixings
            k_paths = (paths[:, column[strike_date], :] if forward_start
                       else tf.stack([tf.cast(k, DTYPE) for k in strikes])[None, :])
            by_underlying = paths / k_paths[:, None, :]
            performance = tf.reduce_min(by_underlying, axis=2)                                         # worst of
            self.performance = performance
            self._by_underlying, self._column = by_underlying, column      # projections: who is the worst
            perf.update({d: performance[:, column[d]] for d in fixings if d in observations})
            n = self._n_paths
        curve = market_env.get_ir_curve(payment_ccy)
        self._equities = equities
        return self._value(product, perf, set(known), curve, eval_date, n)

    # ------------------------------------------------------------------
    # Live trades: the fixings already observed
    # ------------------------------------------------------------------

    @staticmethod
    def _observations(product) -> set:
        return set(product.coupon_fixing_dates) | set(product.autocall_fixing_dates)

    def _future_dates(self, product, eval_date, forward_start: bool) -> list:
        """Dates to simulate: the observations after the evaluation date (and a future strike date)."""
        strike_date = getattr(product, "strike_date", None)
        return sorted(d for d in self._observations(product) | ({strike_date} if forward_start else set())
                      if d > eval_date)

    def _known_levels(self, product, market_env, equities, eval_date) -> dict:
        """Levels already fixed: ``{date: [value per underlying]}`` on the observation and strike dates up to the
        evaluation date. From ``product.fixings`` first, then the market fixings (``get_eq_fixing``); a fixing on the
        evaluation date not in either is the spot (today's close is not known yet). A past date without a fixing
        raises."""
        dates = self._observations(product)
        strike_date = getattr(product, "strike_date", None)
        if strike_date is not None:
            dates.add(strike_date)
        given = getattr(product, "fixings", None) or {}
        known = {}
        for d in sorted(x for x in dates if x <= eval_date):
            if d in given:
                values = list(given[d]) if isinstance(given[d], (list, tuple)) else [given[d]]
                if len(values) != len(equities):
                    raise ValueError(f"fixing of {d}: {len(values)} values for {len(equities)} underlyings")
                known[d] = [tf.constant(float(v), DTYPE) for v in values]
                continue
            row = []
            for eq in equities:
                stored = market_env.get_eq_fixings(eq.ticker, eq.currency)
                value = None if stored is None else stored.get(d)
                if value is not None:
                    row.append(tf.constant(float(value), DTYPE))
                elif d == eval_date:
                    row.append(tf.cast(eq.spot, DTYPE))          # today's close: not known yet, the spot
                else:
                    row.append(tf.constant(market_env.get_eq_fixing(eq.ticker, d, eq.currency), DTYPE))  # raises
            known[d] = row
        return known

    def projections(self) -> dict:
        """Expected values of the last pricing, from its paths (exact indicators, not discounted), per unit notional.

        Returns:
            dict: ``cash_flows`` (per payment date after the evaluation date: ``coupon``, ``redemption`` (the notional
            back, at an autocall or at maturity), ``put`` (minus the put paid by the investor at maturity), ``funding``
            (autocallable swap: minus the funding paid while alive), ``total``; the client's flows, all of the other
            sign for the issuer), ``autocall`` (per autocall observation:
            ``probability`` of being called there, ``cumulative``), ``maturity`` (``reached`` and ``knock_in``
            probabilities, the expected ``put``, ``loss_given_knock_in``, ``expected_life``: the mean time to
            the redemption, autocall or maturity, in years), ``worst`` (worst-of: per underlying the share of the paths
            where it is the worst at maturity, among those reaching it and among those knocked in), ``forwards`` (per underlying and future observation: the mean
            of the simulated level, its 10th and 90th percentiles ``p10`` / ``p90``, the market forward; with a quanto drift
            the mean differs from the forward).

        Raises:
            ValueError: If nothing was priced yet.
        """
        if getattr(self, "_last", None) is None:
            raise ValueError("price an autocallable first")
        last = self._last
        product, n, eval_date, known = last["product"], last["n"], last["eval_date"], last["known"]
        perf = {d: np.broadcast_to(np.asarray(v.numpy() if hasattr(v, "numpy") else v, float), (n,))
                for d, v in last["perf"].items()}
        coupon_index = {d: i for i, d in enumerate(product.coupon_fixing_dates)}
        autocall_index = {d: i for i, d in enumerate(product.autocall_fixing_dates)}
        last_coupon = product.coupon_fixing_dates[-1]
        alive, unpaid = np.ones(n, bool), np.zeros(n)
        flows: dict = {}
        autocall, maturity, worst = [], {}, []
        years = lambda pay: max((pay - eval_date).days, 0) / 365.0
        life = np.zeros(n)                                              # redemption time of each path (years)

        survival = []                                                   # (autocall payment date, alive after)

        def add(pay, kind, values):
            if pay >= eval_date:
                row = flows.setdefault(pay, {"date": pay, "coupon": 0.0, "redemption": 0.0, "put": 0.0,
                                             "funding": 0.0})
                row[kind] += float(np.mean(values))

        for d in sorted(perf):
            x = perf[d]
            if d in coupon_index:
                i = coupon_index[d]
                above = x >= product.coupon_barriers[i] / 100.0
                amount = product.coupon_rates[i] / 100.0 * ((1.0 + unpaid) if product.memory else 1.0)
                add(product.coupon_payment_dates[i], "coupon", np.where(alive & above, amount, 0.0))
                if d == last_coupon:                    # maturity: the paths not called (also not on this date)
                    called_now = (alive & (x >= product.autocall_barrier[autocall_index[d]] / 100.0)
                                  if d in autocall_index else np.zeros(n, bool))
                    reached = alive & ~called_now
                    put = np.zeros(n)
                    knock_in = reached & (x < product.payoff_barrier / 100.0) if product.redemption_payoff is not None                         else np.zeros(n, bool)
                    if knock_in.any():
                        put = np.where(knock_in, product.payoff_participation
                                       * np.maximum(getattr(product, "put_strike", 100.0) / 100.0 - x, 0.0), 0.0)
                    add(product.coupon_payment_dates[i], "redemption", reached.astype(float))    # the notional
                    add(product.coupon_payment_dates[i], "put", -put)                            # paid by the investor
                    life = np.where(reached, years(product.coupon_payment_dates[i]), life)
                    by_und = getattr(self, "_by_underlying", None)
                    if product.is_worst_of and by_und is not None and d in self._column:
                        worst_idx = np.argmin(by_und.numpy()[:, self._column[d], :], axis=1)
                        for k, ticker in enumerate(product.underlyings):
                            is_worst = worst_idx == k
                            worst.append({"ticker": ticker,
                                          "maturity": float((is_worst & reached).sum() / max(reached.sum(), 1)),
                                          "knock_in": float((is_worst & knock_in).sum() / max(knock_in.sum(), 1))})
                    maturity = {"reached": float(reached.mean()), "put": float(put.mean()),
                                "knock_in": float(knock_in.mean()),
                                "loss_given_knock_in": float(put[knock_in].mean()) if knock_in.any() else 0.0}
                unpaid = np.where(alive & above, 0.0, np.where(alive & ~above, unpaid + 1.0, unpaid))
            if d in autocall_index:
                j = autocall_index[d]
                called = alive & (x >= product.autocall_barrier[j] / 100.0)
                add(product.autocall_payment_dates[j], "redemption", called.astype(float))
                life = np.where(called, years(product.autocall_payment_dates[j]), life)
                autocall.append({"date": d, "probability": float(called.mean()), "known": d in known})
                alive = alive & ~called
                survival.append((product.autocall_payment_dates[j], alive.copy()))
        for start, pay, amount in last.get("funding", []):               # paid by the client while alive
            alive_at = next((a for p, a in reversed(survival) if p <= start), np.ones(n, bool))
            add(pay, "funding", -amount * alive_at)
        sign = getattr(product, "side_sign", 1.0)                       # the issuer: every flow the other way
        for row in flows.values():
            for kind in ("coupon", "redemption", "put", "funding"):
                row[kind] *= sign
        total = 0.0
        for row in autocall:
            total += row["probability"]
            row["cumulative"] = total
        cash_flows = [dict(r, total=r["coupon"] + r["redemption"] + r["put"] + r["funding"])
                      for _, r in sorted(flows.items())]
        maturity["expected_life"] = float(life.mean())
        forwards = []
        paths = None if self.paths is None else self.paths.numpy()
        for i, eq in enumerate(getattr(self, "_equities", [])):
            rows = []
            for c, d in enumerate(self.date_grid):
                level = paths[:, c] if paths.ndim == 2 else paths[:, c, i]
                rows.append({"date": d, "mean": float(level.mean()),            # composite: S X, no forward of S
                             "p10": float(np.percentile(level, 10)), "p90": float(np.percentile(level, 90)),
                             "forward": None if product.is_composite else float(eq.forward(d))})
            forwards.append({"ticker": eq.ticker, "rows": rows})
        return {"cash_flows": cash_flows, "autocall": autocall, "maturity": maturity, "forwards": forwards,
                "worst": worst, "n_paths": n}

    def trade_state(self, product: AutocallableOption, market_env: MarketEnvironment) -> dict:
        """State of a live trade from the fixings already observed (the same fixings as the price).

        Returns:
            dict: ``strikes`` (initial levels, ``None`` before a future strike date), ``rows`` (one per past
            observation: ``date``, ``levels`` per underlying, ``performance`` (the worst), ``coupon`` "paid" /
            "missed" / ``None``, ``coupon_paid`` (% of the notional, with the memory), ``autocall`` (bool)),
            ``called_on`` (date or ``None``), ``unpaid_coupons`` (coupons in memory after the last past date).
        """
        eval_date = Settings.evaluation_date
        equities = [market_env.get_equity(t) for t in product.underlyings]
        known = self._known_levels(product, market_env, equities, eval_date)
        strike_date = getattr(product, "strike_date", None)
        if product.strike is not None:
            strikes = [float(k) for k in product.strikes]
        elif strike_date is not None and strike_date <= eval_date:
            strikes = [float(v) for v in known[strike_date]]
        else:
            strikes = None
        coupon_index = {d: i for i, d in enumerate(product.coupon_fixing_dates)}
        autocall_index = {d: i for i, d in enumerate(product.autocall_fixing_dates)}
        rows, unpaid, called_on = [], 0, None
        for d in sorted(x for x in known if x in self._observations(product)):
            if called_on is not None:
                break
            levels = [float(v) for v in known[d]]
            perf = min(v / k for v, k in zip(levels, strikes))
            row = {"date": d, "levels": levels, "performance": perf, "coupon": None, "coupon_paid": 0.0,
                   "autocall": False}
            if d in coupon_index:
                i = coupon_index[d]
                if perf >= product.coupon_barriers[i] / 100.0:
                    row["coupon"] = "paid"
                    row["coupon_paid"] = product.coupon_rates[i] * ((1 + unpaid) if product.memory else 1)
                    unpaid = 0
                else:
                    row["coupon"] = "missed"
                    unpaid += 1
            if d in autocall_index and perf >= product.autocall_barrier[autocall_index[d]] / 100.0:
                row["autocall"] = True
                called_on = d
            rows.append(row)
        return {"strikes": strikes, "rows": rows, "called_on": called_on,
                "unpaid_coupons": unpaid if product.memory else 0}

    def _value(self, product, perf: dict, known: set, curve, eval_date, n: int) -> tf.Tensor:
        """Price from the performance on every observation date (past ones as numbers, future ones on the paths)."""
        self._last = {"product": product, "perf": perf, "known": known, "eval_date": eval_date, "n": n}
        coupon_pv, put_pv = self._payoff(product, perf, known, curve, eval_date, n)
        total = coupon_pv - put_pv                                     # the client's value, per unit notional
        product.coupon_pv = tf.reduce_mean(coupon_pv)
        product.put_pv = tf.reduce_mean(put_pv)
        funding_pv = self._funding(product, eval_date, n)
        if funding_pv is not None:                                     # autocallable swap: the client pays it
            total = total - funding_pv
            product.funding_pv = tf.reduce_mean(funding_pv)
        product.standard_error = (product.notional * tf.math.reduce_std(tf.stop_gradient(total)) / n ** 0.5
                                  if n > 1 else tf.constant(0.0, DTYPE))
        return getattr(product, "side_sign", 1.0) * product.notional * tf.reduce_mean(total)

    def _funding(self, product, eval_date, n: int):
        """Path-wise discounted funding leg per unit notional (``[n]``), or ``None`` (an autocallable without one:
        :class:`AutocallableSwapPricer` prices it)."""
        return None

    # ------------------------------------------------------------------
    # Payoff
    # ------------------------------------------------------------------

    def _above(self, x: tf.Tensor, level: float, hard: bool = False) -> tf.Tensor:
        """``1{x >= level}`` on the performance, smoothed with a sigmoid of width ``eps`` when ``eps > 0`` (a past
        fixing, *hard*: the exact indicator)."""
        if hard or self._eps == 0.0:
            return tf.cast(x >= level, DTYPE)
        return tf.sigmoid((x - level) / self._eps)

    def _payoff(self, product, perf: dict, known: set, curve, eval_date, n: int):
        """Path-wise discounted coupons and put, per unit notional (``[n]`` each), on the performance *perf*
        (``{date: S / K}``, or the worst of the underlyings; the levels in % of the strike). Observations in *known*
        are past fixings (exact indicators, no smoothing); payments before the evaluation date are already settled
        (they only move the state: memory, autocall)."""
        coupon_index = {d: i for i, d in enumerate(product.coupon_fixing_dates)}
        autocall_index = {d: i for i, d in enumerate(product.autocall_fixing_dates)}
        last_coupon = product.coupon_fixing_dates[-1]

        alive = tf.ones([n], DTYPE)
        unpaid = tf.zeros([n], DTYPE)
        coupon_pv = tf.zeros([n], DTYPE)
        put_pv = tf.zeros([n], DTYPE)
        self._survival = []             # (autocall payment date, alive after it): the end of a funding leg

        for d in sorted(perf):
            s = perf[d] * tf.ones([n], DTYPE)
            hard = d in known
            if d in coupon_index:
                i = coupon_index[d]
                above = self._above(s, product.coupon_barriers[i] / 100.0, hard)
                rate = product.coupon_rates[i] / 100.0
                amount = rate * (1.0 + unpaid) if product.memory else rate * tf.ones_like(s)
                pay = product.coupon_payment_dates[i]
                df_pay = tf.cast(curve.discount(pay), DTYPE) if pay >= eval_date else tf.constant(0.0, DTYPE)
                coupon_pv = coupon_pv + alive * above * amount * df_pay
                if d == last_coupon and product.redemption_payoff is not None:
                    below = 1.0 - self._above(s, product.payoff_barrier / 100.0, hard)
                    k_put = getattr(product, "put_strike", 100.0) / 100.0
                    put = product.payoff_participation * tf.maximum(k_put - s, 0.0)
                    put_pv = put_pv + alive * below * put * df_pay
                # memory: reset when paid, +1 when missed (alive paths)
                unpaid = alive * (1.0 - above) * (unpaid + 1.0) + (1.0 - alive) * unpaid
            if d in autocall_index:
                j = autocall_index[d]
                alive = alive * (1.0 - self._above(s, product.autocall_barrier[j] / 100.0, hard))
                self._survival.append((product.autocall_payment_dates[j], alive))
        return coupon_pv, put_pv


class AutocallableSwapPricer(AutocallableMCPricer):
    """:class:`AutocallableSwap` by Monte Carlo: the autocallable as :class:`AutocallableMCPricer` and the funding
    leg weighted by the probability that the product is alive at the start of each period, on the same paths (the
    same smoothed autocall indicators, so delta, vega and skew include the termination of the funding).

    The rate is not simulated: period ``i`` pays ``(F_i + margin) tau_i`` at its end, ``F_i = (P(s_i) / P(e_i) - 1) /
    tau_i`` on the funding curve (``get_ir_curve(payment_ccy, funding_curve)``), discounted on the payment currency's
    default curve; rho per pillar reaches both curves by AAD. A period already started takes the forward from the
    evaluation date to its end (no rate fixings in the market yet); periods paid before the evaluation date are
    settled. Without an autocall the leg equals the floating leg of a swap on the same periods.
    """

    def _funding(self, product, eval_date, n: int):
        if not isinstance(product, AutocallableSwap):
            return None
        ccy = product.payment_ccy
        projection = self._market.get_ir_curve(ccy, ticker=product.funding_curve)
        discount = self._market.get_ir_curve(ccy)
        margin = product.funding_margin / 1e4
        survival = getattr(self, "_survival", [])
        pv = tf.zeros([n], DTYPE)
        periods = []
        for start, end, pay, tau in zip(product.funding_start_dates, product.funding_end_dates,
                                        product.funding_payment_dates, product.funding_accruals):
            if pay < eval_date:
                continue
            first = max(start, eval_date)
            tau_fwd = product.funding_day_count.year_fraction(first, end)
            forward = ((tf.cast(projection.discount(first), DTYPE) / tf.cast(projection.discount(end), DTYPE) - 1.0)
                       / tau_fwd if tau_fwd > 0 else tf.constant(0.0, DTYPE))
            amount = (forward + margin) * tau
            alive = next((a for p, a in reversed(survival) if p <= start), tf.ones([n], DTYPE))
            pv = pv + alive * amount * tf.cast(discount.discount(pay), DTYPE)
            periods.append((start, pay, float(amount)))
        self._last["funding"] = periods                                  # the projections
        return pv
