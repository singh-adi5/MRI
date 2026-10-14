"""
mri.data.generate
-----------------

Synthetic marketplace generator for the MRI fraud-intelligence pipeline.

What changed vs. the original generator
=======================================
The previous version drew every field i.i.d. uniform, injected **no fraud at
all**, and reused 5 hardcoded device IDs across every account. The downstream
rule engine therefore flagged every account that had ever logged in, the
"label" collapsed to ``login_events > 0`` and every metric in the repo was
meaningless.

This version produces:

* 2000 customers / 200 sellers with heavy-tailed device and IP-region
  cardinality (most accounts own 1-2 devices; a *small* number of devices are
  genuinely shared, by benign households as well as by fraud rings, so device
  sharing is a selective signal rather than a universal one).
* Five labelled fraud patterns at a realistic base rate (~5% of entities):
  ``bust_out``, ``collusive_ring``, ``refund_abuse``, ``ato``, ``rating_farm``.
* Deliberately overlapping benign behaviour (heavy refunders, new fast-growing
  sellers, travellers, device upgraders, extreme raters, repeat raters,
  households) so that no single feature separates fraud from benign.

The CSV schemas (customers, sellers, orders, refunds, ratings,
login_activity) are **unchanged** - only the contents, scale and realism.

#############################################################################
#                                                                           #
#  ground_truth.csv IS EVALUATION ONLY.                                     #
#                                                                           #
#  data/raw/ground_truth.csv carries the injected fraud labels. It MUST     #
#  NEVER be read by any detector, rule, feature-engineering step, graph     #
#  builder, scoring model, fusion stage or API handler. Reading it anywhere  #
#  in src/mri/{detection,features,models,fusion,api} is label leakage and   #
#  invalidates every metric the project reports. It exists solely so that    #
#  evaluation code (and this module's own self-check) can measure detection #
#  quality out-of-band.                                                     #
#                                                                           #
#############################################################################

Determinism
-----------
A single seeded ``numpy.random.Generator`` drives every draw, nothing iterates
over an unordered set, and every frame is sorted on a unique key before being
written. Two runs produce byte-identical CSVs.

Usage::

    PYTHONPATH=src python -m mri.data.generate
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from mri.config import RAW_DATA_DIR

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEED = 20260315

TODAY = datetime(2026, 3, 15)
WINDOW_DAYS = 180          # transactional observation window
SELLER_HISTORY_DAYS = 730  # sellers may have joined up to 2 years ago
CUSTOMER_HISTORY_DAYS = 540

N_CUSTOMERS = 2000
N_SELLERS = 200
PRODUCTS_PER_SELLER = 15

# Fraud cohort sizes (~5.4% of the 2200 entities).
N_BUST_OUT_SELLERS = 7
N_RINGS = 4
RING_CUSTOMERS = 6         # customers per collusive ring (+1 seller each)
N_REFUND_ABUSERS = 28
N_ATO_CUSTOMERS = 20
N_ATO_SELLERS = 2
N_RATING_FARMS = 3
FARM_ACCOUNTS = 10         # customer accounts per rating farm (+1 target seller)

# Benign look-alike cohorts - these keep every individual feature non-separating.
N_BENIGN_HOUSEHOLDS = 18   # 3-4 related accounts sharing a device + region
N_BENIGN_SHARED_DEVICES = 90
N_BENIGN_NEW_RAMP_SELLERS = 12
N_BENIGN_POOR_QUALITY_SELLERS = 14
N_BENIGN_REPEAT_RATERS = 70

COARSE_REGIONS = ["Europe", "North America", "South America", "Asia", "Africa", "Oceania"]
DEVICE_TYPES = ["mobile", "desktop", "tablet"]
CATEGORIES = ["electronics", "fashion", "home", "beauty", "sports", "toys", "books"]
PAYMENT_STATUSES = ["completed", "pending", "cancelled"]
REFUND_REASONS = ["defective", "late delivery", "changed mind", "fraud"]
REFUND_STATUSES = ["approved", "pending", "rejected"]

WINDOW_START = TODAY - timedelta(days=WINDOW_DAYS)

FRAUD_PATTERNS = ["bust_out", "collusive_ring", "refund_abuse", "ato", "rating_farm"]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _day(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _jitter(rng: np.random.Generator, dt: datetime) -> datetime:
    """Give a date a plausible time-of-day (diurnal, not uniform)."""
    hour = int(np.clip(rng.normal(14.5, 4.5), 0, 23.999))
    return dt.replace(hour=0, minute=0, second=0) + timedelta(
        hours=hour, minutes=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 60))
    )


def _between(rng: np.random.Generator, start: datetime, end: datetime) -> datetime:
    """Uniform datetime in [start, end]; collapses to start if the span is empty."""
    span = (end - start).total_seconds()
    if span <= 0:
        return start
    return start + timedelta(seconds=float(rng.uniform(0, span)))


def _zipf_weights(n: int, exponent: float, offset: float) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=float)
    w = 1.0 / np.power(ranks + offset, exponent)
    return w / w.sum()


def _build_region_vocabulary() -> list[str]:
    """~130 fine-grained IP regions, so region sharing carries information.

    The original 6 continents over 350 entities made region-sharing a no-op.
    """
    blocks = [
        ("EU", ["DE", "FR", "GB", "ES", "IT", "PL", "NL", "SE", "IE", "PT"]),
        ("NA", ["US", "CA", "MX"]),
        ("SA", ["BR", "AR", "CL", "CO"]),
        ("AS", ["IN", "CN", "JP", "SG", "VN", "PH", "ID", "TR"]),
        ("AF", ["NG", "ZA", "EG", "KE"]),
        ("OC", ["AU", "NZ"]),
    ]
    regions: list[str] = []
    for continent, countries in blocks:
        for country in countries:
            # Larger markets get more distinct metro-level regions.
            n_metro = 9 if country in ("US", "DE", "GB", "IN", "BR", "CN") else 5
            for metro in range(1, n_metro + 1):
                regions.append(f"{continent}-{country}-{metro:02d}")
    return regions


# --------------------------------------------------------------------------
# Entity model
# --------------------------------------------------------------------------


@dataclass
class Account:
    kind: str                      # "customer" | "seller"
    eid: int
    uid: str                       # "C123" / "S45"
    signup: datetime
    coarse_region: str
    home_region: str               # fine-grained login region
    devices: list[str] = field(default_factory=list)
    fraud_pattern: str = ""
    # behavioural knobs
    refund_propensity: float = 0.05
    rating_bias: str = "normal"    # normal | extreme_high | extreme_low
    rating_rate: float = 0.26      # P(leaves a rating | order)
    n_orders: int = 0
    # pattern bookkeeping
    takeover_at: datetime | None = None
    secondary_region: str | None = None
    secondary_from: datetime | None = None
    active_until: datetime | None = None
    login_scale: float = 1.0
    # bust-out timing (sellers only)
    bust_day: int = 0
    ramp_peak: float = 0.0
    # a device that only appears partway through the window
    late_device: str | None = None
    late_device_from: datetime | None = None
    # ATO only: the device the attacker brings
    takeover_device: str | None = None


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------


class MarketplaceGenerator:
    def __init__(self, seed: int = SEED) -> None:
        self.rng = np.random.default_rng(seed)
        self.regions = _build_region_vocabulary()
        # Long-tailed region popularity: the biggest metros hold a few percent of
        # accounts each, while the tail holds single-digit counts - which is where
        # ring co-location actually shows up as a signal.
        self.region_weights = _zipf_weights(len(self.regions), exponent=1.0, offset=9.0)

        self.customers: dict[int, Account] = {}
        self.sellers: dict[int, Account] = {}
        self.orders: list[dict] = []
        self.ratings: list[dict] = []
        self.logins: list[dict] = []

        self._device_counter = 0
        self.device_owners: dict[str, list[str]] = {}

        # populated by _assign_fraud_roles
        self.bust_out_sellers: list[int] = []
        self.rings: list[dict] = []
        self.refund_abusers: list[int] = []
        self.ato_customers: list[int] = []
        self.ato_sellers: list[int] = []
        self.farms: list[dict] = []
        self.benign_ramp_sellers: list[int] = []
        self.households: list[list[str]] = []
        self.repeat_raters: list[int] = []

    # -- infrastructure ----------------------------------------------------

    def _new_device(self) -> str:
        self._device_counter += 1
        return f"dev_{self._device_counter:06d}"

    def _attach_device(self, acct: Account, device: str) -> None:
        if device not in acct.devices:
            acct.devices.append(device)
        self.device_owners.setdefault(device, [])
        if acct.uid not in self.device_owners[device]:
            self.device_owners[device].append(acct.uid)

    def _pick_region(self) -> str:
        return self.regions[int(self.rng.choice(len(self.regions), p=self.region_weights))]

    def _pick_rare_region(self) -> str:
        """A region from the thin tail - used for co-located rings."""
        tail = self.regions[int(len(self.regions) * 0.6):]
        return tail[int(self.rng.integers(0, len(tail)))]

    def _choice(self, items: list, weights: np.ndarray | None = None):
        idx = int(self.rng.choice(len(items), p=weights))
        return items[idx]

    # -- 1. entity skeletons ----------------------------------------------

    def build_entities(self) -> None:
        rng = self.rng
        # Account age is front-loaded (the marketplace is growing), not uniform.
        # A uniform spread would make "account is only a few weeks old" a near
        # perfect tell for rating-farm accounts; in reality young accounts are
        # the single largest cohort.
        for cid in range(1, N_CUSTOMERS + 1):
            age = 1.0 + float(rng.beta(0.85, 1.5)) * (CUSTOMER_HISTORY_DAYS - 1)
            signup = TODAY - timedelta(days=age)
            self.customers[cid] = Account(
                kind="customer",
                eid=cid,
                uid=f"C{cid}",
                signup=signup,
                coarse_region=self._choice(COARSE_REGIONS),
                home_region=self._pick_region(),
                # Beta(1.6, 18) -> mean ~8%, but a real tail of benign customers
                # sitting at 30-45% refund rates. This is what stops refund_rate
                # from being a one-feature separator for refund abusers.
                refund_propensity=float(rng.beta(1.6, 18.0)),
            )
        for sid in range(1, N_SELLERS + 1):
            join = TODAY - timedelta(
                days=1.0 + float(rng.beta(0.9, 1.3)) * (SELLER_HISTORY_DAYS - 1)
            )
            self.sellers[sid] = Account(
                kind="seller",
                eid=sid,
                uid=f"S{sid}",
                signup=join,
                coarse_region=self._choice(COARSE_REGIONS),
                home_region=self._pick_region(),
                refund_propensity=float(rng.beta(2.0, 20.0)),
            )

        # Rating style: a sizeable benign minority only ever gives 5s (or 1s),
        # and ~9% are prolific reviewers who rate nearly everything they buy.
        # Both cohorts overlap the rating farms on rating volume and extremity.
        for cid in sorted(self.customers):
            acct = self.customers[cid]
            r = rng.random()
            if r < 0.13:
                acct.rating_bias = "extreme_high"
            elif r < 0.17:
                acct.rating_bias = "extreme_low"
            acct.rating_rate = 0.80 if rng.random() < 0.09 else float(rng.uniform(0.12, 0.34))

    # -- 2. fraud role assignment -----------------------------------------

    def assign_fraud_roles(self) -> None:
        rng = self.rng
        cust_pool = list(rng.permutation(np.arange(1, N_CUSTOMERS + 1)))
        sell_pool = list(rng.permutation(np.arange(1, N_SELLERS + 1)))

        def take_c(n: int) -> list[int]:
            out = [int(cust_pool.pop()) for _ in range(n)]
            return out

        def take_s(n: int) -> list[int]:
            out = [int(sell_pool.pop()) for _ in range(n)]
            return out

        # bust-out sellers. Timing is fixed here, before any orders are drawn,
        # so that organic demand also respects the disappearance date.
        self.bust_out_sellers = take_s(N_BUST_OUT_SELLERS)
        for sid in self.bust_out_sellers:
            s = self.sellers[sid]
            s.fraud_pattern = "bust_out"
            s.signup = TODAY - timedelta(days=float(rng.integers(55, 96)))
            s.home_region = self._pick_region()
            life = max((TODAY - s.signup).days, 20)
            s.bust_day = int(life * rng.uniform(0.60, 0.75))
            s.ramp_peak = float(rng.uniform(3.5, 8.0))
            # goes dark a few days after the refund wave
            s.active_until = min(
                s.signup + timedelta(days=s.bust_day + float(rng.uniform(4, 11))), TODAY
            )

        # collusive rings
        for _ in range(N_RINGS):
            members_c = take_c(RING_CUSTOMERS)
            member_s = take_s(1)[0]
            ring_start = TODAY - timedelta(days=float(rng.integers(70, 141)))
            region = self._pick_rare_region()
            devices = [self._new_device() for _ in range(int(rng.integers(1, 3)))]
            ring = {
                "customers": members_c,
                "seller": member_s,
                "start": ring_start,
                "region": region,
                "devices": devices,
            }
            self.rings.append(ring)
            for cid in members_c:
                c = self.customers[cid]
                c.fraud_pattern = "collusive_ring"
                c.signup = ring_start + timedelta(days=float(rng.uniform(0, 10)))
                c.home_region = region
                c.refund_propensity = float(rng.beta(1.2, 40.0))
            s = self.sellers[member_s]
            s.fraud_pattern = "collusive_ring"
            s.signup = ring_start + timedelta(days=float(rng.uniform(-5, 5)))
            s.home_region = region
            s.refund_propensity = float(rng.beta(1.2, 40.0))

        # refund abusers
        self.refund_abusers = take_c(N_REFUND_ABUSERS)
        for cid in self.refund_abusers:
            self.customers[cid].fraud_pattern = "refund_abuse"

        # account takeover
        self.ato_customers = take_c(N_ATO_CUSTOMERS)
        for cid in self.ato_customers:
            c = self.customers[cid]
            c.fraud_pattern = "ato"
            # ATO targets established accounts.
            c.signup = TODAY - timedelta(days=float(rng.integers(210, CUSTOMER_HISTORY_DAYS)))
            c.takeover_at = TODAY - timedelta(days=float(rng.uniform(8, 35)))
        self.ato_sellers = take_s(N_ATO_SELLERS)
        for sid in self.ato_sellers:
            s = self.sellers[sid]
            s.fraud_pattern = "ato"
            s.signup = TODAY - timedelta(days=float(rng.integers(300, SELLER_HISTORY_DAYS)))
            s.takeover_at = TODAY - timedelta(days=float(rng.uniform(8, 35)))

        # rating farms
        for _ in range(N_RATING_FARMS):
            accounts = take_c(FARM_ACCOUNTS)
            target = take_s(1)[0]
            farm_start = TODAY - timedelta(days=float(rng.integers(25, 75)))
            self.farms.append({"accounts": accounts, "target": target, "start": farm_start})
            for cid in accounts:
                c = self.customers[cid]
                c.fraud_pattern = "rating_farm"
                c.signup = farm_start + timedelta(days=float(rng.uniform(0, 14)))
                c.login_scale = 0.25
            t = self.sellers[target]
            t.fraud_pattern = "rating_farm"

        # ---- benign look-alikes (labelled benign on purpose) ----
        self.benign_ramp_sellers = take_s(N_BENIGN_NEW_RAMP_SELLERS)
        for sid in self.benign_ramp_sellers:
            s = self.sellers[sid]
            s.signup = TODAY - timedelta(days=float(rng.integers(45, 110)))
            # Genuinely poor-quality new sellers: refund rates that reach into
            # bust-out territory without any bust-out behaviour.
            s.refund_propensity = float(rng.uniform(0.22, 0.48))

        # Established but low-quality sellers (dropshippers, fragile goods).
        # Without these the ">30% seller refund rate" rule would be a perfect
        # fraud oracle instead of a noisy heuristic.
        for sid in take_s(N_BENIGN_POOR_QUALITY_SELLERS):
            self.sellers[sid].refund_propensity = float(rng.uniform(0.28, 0.55))

        self.repeat_raters = take_c(N_BENIGN_REPEAT_RATERS)

        # Benign households: related accounts sharing a device and a region.
        for _ in range(N_BENIGN_HOUSEHOLDS):
            size = int(rng.integers(3, 5))
            members = take_c(size)
            region = self._pick_rare_region()
            device = self._new_device()
            base_signup = TODAY - timedelta(days=float(rng.integers(60, 400)))
            uids = []
            for cid in members:
                c = self.customers[cid]
                c.home_region = region
                c.signup = base_signup + timedelta(days=float(rng.uniform(0, 12)))
                self._attach_device(c, device)
                uids.append(c.uid)
            self.households.append(uids)

    # -- 3. devices --------------------------------------------------------

    def build_devices(self) -> None:
        """Heavy-tailed personal device ownership + a thin layer of real sharing."""
        rng = self.rng
        counts = [1, 2, 3, 4, 5]
        probs = np.array([0.52, 0.28, 0.12, 0.05, 0.03])

        all_accounts = [self.customers[c] for c in sorted(self.customers)] + [
            self.sellers[s] for s in sorted(self.sellers)
        ]
        for acct in all_accounts:
            n = counts[int(rng.choice(len(counts), p=probs))]
            if acct.fraud_pattern == "rating_farm" and acct.kind == "customer":
                n = 1  # farm accounts are cheap and thin
            for _ in range(n):
                self._attach_device(acct, self._new_device())

        # Benign shared devices (household PCs, shared tablets, office machines).
        # Crucially some of these exceed the rule engine's ">3 accounts" threshold,
        # so device sharing produces genuine false positives rather than being a
        # perfect fraud oracle.
        share_sizes = [2, 3, 4, 5, 6]
        share_probs = np.array([0.45, 0.30, 0.15, 0.07, 0.03])
        benign_uids = [a.uid for a in all_accounts if a.fraud_pattern == ""]
        for _ in range(N_BENIGN_SHARED_DEVICES):
            k = share_sizes[int(rng.choice(len(share_sizes), p=share_probs))]
            picks = rng.choice(len(benign_uids), size=k, replace=False)
            device = self._new_device()
            for p in sorted(int(x) for x in picks):
                uid = benign_uids[p]
                acct = self._account_by_uid(uid)
                self._attach_device(acct, device)

        # Collusive rings: every member on the same 2-3 devices.
        for ring in self.rings:
            member_accounts = [self.customers[c] for c in ring["customers"]]
            member_accounts.append(self.sellers[ring["seller"]])
            for acct in member_accounts:
                for device in ring["devices"]:
                    self._attach_device(acct, device)

        # Rating farms: accounts pair up on devices (burner phones recycled),
        # but never all on one, so farm != device-sharing.
        for farm in self.farms:
            accounts = [self.customers[c] for c in farm["accounts"]]
            for i in range(0, len(accounts) - 1, 3):
                device = self._new_device()
                for acct in accounts[i:i + 3]:
                    self._attach_device(acct, device)

        # ATO: takeover introduces a brand-new device for the hijacked account.
        # Held off acct.devices so it cannot leak into pre-takeover logins.
        for acct in self._ato_accounts():
            acct.takeover_device = self._new_device()

        # Benign device churn: ~12% of benign accounts replace a handset mid-window.
        # This is what makes "a new device suddenly appears" a non-exclusive ATO
        # signal - plenty of clean accounts do exactly the same thing.
        for acct in all_accounts:
            if acct.fraud_pattern == "" and rng.random() < 0.12:
                acct.late_device = self._new_device()
                acct.late_device_from = _between(
                    self.rng, WINDOW_START, TODAY - timedelta(days=5)
                )

        # Benign travel / relocation: a contiguous block of logins from elsewhere.
        for acct in all_accounts:
            if acct.fraud_pattern == "" and rng.random() < 0.10:
                acct.secondary_region = self._pick_region()
                if acct.secondary_from is None:
                    acct.secondary_from = _between(self.rng, WINDOW_START, TODAY - timedelta(days=5))

    def _account_by_uid(self, uid: str) -> Account:
        if uid.startswith("C"):
            return self.customers[int(uid[1:])]
        return self.sellers[int(uid[1:])]

    def _ato_accounts(self) -> list[Account]:
        return [self.customers[c] for c in self.ato_customers] + [
            self.sellers[s] for s in self.ato_sellers
        ]

    # -- 4. orders ---------------------------------------------------------

    def _seller_popularity(self) -> tuple[list[int], np.ndarray]:
        rng = self.rng
        sids = sorted(self.sellers)
        order = rng.permutation(len(sids))
        weights = _zipf_weights(len(sids), exponent=1.0, offset=4.0)
        w = np.empty(len(sids))
        w[order] = weights
        return sids, w / w.sum()

    def _product_for(self, seller_id: int) -> int:
        base = (seller_id - 1) * PRODUCTS_PER_SELLER
        return int(base + self.rng.integers(1, PRODUCTS_PER_SELLER + 1))

    def _payment_status(self) -> str:
        return self._choice(PAYMENT_STATUSES, np.array([0.88, 0.08, 0.04]))

    def _add_order(
        self,
        customer_id: int,
        seller_id: int,
        when: datetime,
        amount: float,
        refund: bool,
        status: str | None = None,
    ) -> None:
        self.orders.append(
            {
                "customer_id": int(customer_id),
                "seller_id": int(seller_id),
                "product_id": self._product_for(seller_id),
                "order_date": when,
                "amount": round(float(np.clip(amount, 3.0, 5000.0)), 2),
                "payment_status": status or self._payment_status(),
                "_refund": bool(refund),
            }
        )

    def _refund_decision(self, customer: Account, seller: Account) -> bool:
        # Noisy-OR rather than a weighted average: a genuinely bad seller drives
        # its own refund rate instead of being diluted towards the mean by its
        # customers. That is what puts honest low-quality sellers above the 30%
        # rule threshold alongside the bust-out sellers.
        p = 1.0 - (1.0 - 0.6 * customer.refund_propensity) * (1.0 - seller.refund_propensity)
        return bool(self.rng.random() < np.clip(p, 0.002, 0.95))

    def build_orders(self) -> None:
        rng = self.rng
        sids, pop = self._seller_popularity()

        signups = np.array(
            [np.datetime64(self.customers[c].signup) for c in sorted(self.customers)]
        )
        cust_ids = np.array(sorted(self.customers))
        sort_idx = np.argsort(signups, kind="stable")
        signups_sorted = signups[sort_idx]
        cust_sorted = cust_ids[sort_idx]

        def random_customer_active_by(when: datetime) -> int:
            k = int(np.searchsorted(signups_sorted, np.datetime64(when), side="right"))
            if k <= 0:
                k = 1
            return int(cust_sorted[int(rng.integers(0, k))])

        # ---- baseline organic demand ------------------------------------
        for cid in sorted(self.customers):
            c = self.customers[cid]
            if c.fraud_pattern in ("rating_farm",):
                c.n_orders = int(rng.integers(0, 3))
            elif c.fraud_pattern == "collusive_ring":
                c.n_orders = int(rng.integers(2, 5))      # non-ring purchases only
            elif c.fraud_pattern == "refund_abuse":
                c.n_orders = int(rng.integers(3, 9))      # baseline half of profile
            else:
                lam = float(np.clip(np.exp(rng.normal(1.25, 0.85)), 0.2, 45.0))
                c.n_orders = int(rng.poisson(lam))

            lo = max(WINDOW_START, c.signup)
            if c.fraud_pattern == "ato" and c.takeover_at is not None:
                hi = c.takeover_at
            else:
                hi = TODAY
            if hi <= lo:
                continue

            for _ in range(c.n_orders):
                sid = sids[int(rng.choice(len(sids), p=pop))]
                s = self.sellers[sid]
                lo_s = max(lo, s.signup)
                if s.active_until is not None:
                    hi_s = min(hi, s.active_until)
                else:
                    hi_s = hi
                if hi_s <= lo_s:
                    continue
                when = _jitter(rng, _between(rng, lo_s, hi_s))
                amount = float(rng.lognormal(mean=4.0, sigma=0.95))
                self._add_order(cid, sid, when, amount, self._refund_decision(c, s))

        # ---- bust-out sellers -------------------------------------------
        for sid in self.bust_out_sellers:
            s = self.sellers[sid]
            life = max((TODAY - s.signup).days, 20)
            bust_day = s.bust_day
            peak = s.ramp_peak
            for d in range(0, min(bust_day + 12, life)):
                day_dt = s.signup + timedelta(days=d)
                if day_dt >= s.active_until:
                    break
                # exponential ramp, flattening after the bust point
                ramp = min(d / max(bust_day, 1), 1.0)
                rate = 0.35 + peak * (ramp ** 2.1)
                n = int(rng.poisson(rate))
                for _ in range(n):
                    when = _jitter(rng, day_dt)
                    cust = random_customer_active_by(when)
                    # higher ticket than the marketplace median, but inside the tail
                    amount = float(rng.lognormal(mean=4.75, sigma=0.80))
                    late = d >= bust_day * 0.80
                    refund = bool(rng.random() < (0.70 if late else 0.08))
                    self._add_order(cust, sid, when, amount, refund)

        # Benign new sellers that ramp just as fast (and refund nearly as much).
        for sid in self.benign_ramp_sellers:
            s = self.sellers[sid]
            life = max((TODAY - s.signup).days, 20)
            peak = float(rng.uniform(2.5, 6.5))
            for d in range(0, life):
                day_dt = s.signup + timedelta(days=d)
                rate = 0.3 + peak * ((d / life) ** 1.8)
                n = int(rng.poisson(rate))
                for _ in range(n):
                    when = _jitter(rng, day_dt)
                    cust = random_customer_active_by(when)
                    amount = float(rng.lognormal(mean=4.6, sigma=0.85))
                    refund = bool(rng.random() < s.refund_propensity)
                    self._add_order(cust, sid, when, amount, refund)

        # ---- collusive rings --------------------------------------------
        for ring in self.rings:
            seller_id = ring["seller"]
            s = self.sellers[seller_id]
            for cid in ring["customers"]:
                c = self.customers[cid]
                lo = max(c.signup, s.signup, WINDOW_START)
                n = int(rng.integers(5, 11))
                for _ in range(n):
                    when = _jitter(rng, _between(rng, lo, TODAY))
                    amount = float(rng.lognormal(mean=4.55, sigma=0.65))
                    self._add_order(cid, seller_id, when, amount, bool(rng.random() < 0.03))

        # ---- refund abusers ---------------------------------------------
        for cid in self.refund_abusers:
            c = self.customers[cid]
            lo = max(c.signup, WINDOW_START)
            burst_len = float(rng.uniform(14, 24))
            latest_start = (TODAY - timedelta(days=burst_len))
            if latest_start <= lo:
                burst_start = lo
            else:
                burst_start = _between(rng, lo, latest_start)
            burst_end = min(burst_start + timedelta(days=burst_len), TODAY)
            n_burst = int(rng.integers(5, 12))
            burst_refund_p = float(rng.uniform(0.62, 0.88))
            for _ in range(n_burst):
                sid = sids[int(rng.choice(len(sids), p=pop))]
                s = self.sellers[sid]
                lo_s = max(burst_start, s.signup)
                hi_s = burst_end if s.active_until is None else min(burst_end, s.active_until)
                if hi_s <= lo_s:
                    continue
                when = _jitter(rng, _between(rng, lo_s, hi_s))
                amount = float(rng.lognormal(mean=4.2, sigma=0.9))
                self._add_order(cid, sid, when, amount, bool(rng.random() < burst_refund_p))

        # ---- ATO: post-takeover high-value burst ------------------------
        for acct in self._ato_accounts():
            if acct.kind != "customer" or acct.takeover_at is None:
                continue
            n = int(rng.integers(2, 7))
            for _ in range(n):
                sid = sids[int(rng.choice(len(sids), p=pop))]
                s = self.sellers[sid]
                lo_s = max(acct.takeover_at, s.signup)
                hi_s = TODAY if s.active_until is None else min(TODAY, s.active_until)
                if hi_s <= lo_s:
                    continue
                when = _jitter(rng, _between(rng, lo_s, hi_s))
                # high value, but drawn from the same family as the benign tail
                amount = float(rng.lognormal(mean=5.45, sigma=0.65))
                self._add_order(acct.eid, sid, when, amount, bool(rng.random() < 0.10))

        # ---- rating farms: a handful of token purchases -------------------
        for farm in self.farms:
            target = farm["target"]
            s = self.sellers[target]
            for cid in farm["accounts"]:
                if rng.random() < 0.55:
                    c = self.customers[cid]
                    lo = max(c.signup, s.signup, WINDOW_START)
                    when = _jitter(rng, _between(rng, lo, TODAY))
                    amount = float(rng.lognormal(mean=2.6, sigma=0.6))
                    self._add_order(cid, target, when, amount, False)

    # -- 5. ratings --------------------------------------------------------

    def _rating_value(self, acct: Account) -> int:
        rng = self.rng
        if acct.rating_bias == "extreme_high":
            return 5
        if acct.rating_bias == "extreme_low":
            return int(rng.choice([1, 2], p=[0.8, 0.2]))
        return int(rng.choice([1, 2, 3, 4, 5], p=[0.06, 0.05, 0.11, 0.27, 0.51]))

    def build_ratings(self, orders_df: pd.DataFrame) -> None:
        rng = self.rng

        # Organic post-purchase ratings.
        for row in orders_df.itertuples(index=False):
            c = self.customers[int(row.customer_id)]
            if rng.random() >= c.rating_rate:
                continue
            when = row.order_date + timedelta(days=float(rng.uniform(1, 21)))
            if when > TODAY:
                continue
            self.ratings.append(
                {
                    "customer_id": int(row.customer_id),
                    "seller_id": int(row.seller_id),
                    "rating_value": self._rating_value(c),
                    "rating_date": _jitter(rng, when),
                }
            )

        # Ring members inflate their own seller - spread out so that the naive
        # ">3 ratings in 30 days" rule only catches part of the ring.
        for ring in self.rings:
            for cid in ring["customers"]:
                c = self.customers[cid]
                n = int(rng.integers(1, 4))
                lo = max(c.signup, WINDOW_START)
                for _ in range(n):
                    when = _jitter(rng, _between(rng, lo, TODAY))
                    self.ratings.append(
                        {
                            "customer_id": cid,
                            "seller_id": ring["seller"],
                            "rating_value": int(rng.choice([4, 5], p=[0.2, 0.8])),
                            "rating_date": when,
                        }
                    )

        # Rating farms: bursts of 5s (or 1s for a competitor-attack farm).
        for i, farm in enumerate(self.farms):
            attack = (i == len(self.farms) - 1)   # one farm is a 1-star attack farm
            for cid in farm["accounts"]:
                n = int(rng.choice([1, 2, 3, 4, 5], p=[0.18, 0.22, 0.20, 0.25, 0.15]))
                lo = max(farm["start"], self.customers[cid].signup)
                span = max((TODAY - lo).days, 1)
                for _ in range(n):
                    when = _jitter(rng, lo + timedelta(days=float(rng.uniform(0, span))))
                    self.ratings.append(
                        {
                            "customer_id": cid,
                            "seller_id": farm["target"],
                            "rating_value": 1 if attack else 5,
                            "rating_date": when,
                        }
                    )

        # Benign repeat raters: loyal customers who rate a favourite seller
        # repeatedly inside 30 days -> false positives for the rating rule.
        sids = sorted(self.sellers)
        for cid in self.repeat_raters:
            c = self.customers[cid]
            favourite = sids[int(rng.integers(0, len(sids)))]
            s = self.sellers[favourite]
            lo = max(c.signup, s.signup, WINDOW_START)
            if (TODAY - lo).days < 35:
                continue
            anchor = _between(rng, lo, TODAY - timedelta(days=30))
            for _ in range(int(rng.integers(4, 7))):
                when = _jitter(rng, anchor + timedelta(days=float(rng.uniform(0, 28))))
                self.ratings.append(
                    {
                        "customer_id": cid,
                        "seller_id": favourite,
                        "rating_value": self._rating_value(c),
                        "rating_date": when,
                    }
                )

    # -- 6. logins ---------------------------------------------------------

    def _failed_attempts(self) -> int:
        return int(
            self.rng.choice(
                [0, 1, 2, 3, 4, 5],
                p=[0.855, 0.080, 0.040, 0.016, 0.006, 0.003],
            )
        )

    def build_logins(self) -> None:
        rng = self.rng
        accounts = [self.customers[c] for c in sorted(self.customers)] + [
            self.sellers[s] for s in sorted(self.sellers)
        ]
        seller_order_counts: dict[int, int] = {}
        for o in self.orders:
            sid = o["seller_id"]
            seller_order_counts[sid] = seller_order_counts.get(sid, 0) + 1

        for acct in accounts:
            lo = max(WINDOW_START, acct.signup)
            hi = TODAY if acct.active_until is None else min(TODAY, acct.active_until)
            if hi <= lo:
                hi = lo + timedelta(hours=6)

            if acct.kind == "customer":
                lam = (0.9 * acct.n_orders + 2.0) * acct.login_scale
            else:
                lam = 0.05 * seller_order_counts.get(acct.eid, 0) + 4.0
            n_logins = max(1, int(rng.poisson(max(lam, 0.5))))
            # Every device an account owns must appear at least once, otherwise a
            # shared device is invisible in login_activity and device sharing
            # silently disappears as a signal.
            n_logins = max(n_logins, len(acct.devices) + (1 if acct.late_device else 0))

            primary = acct.devices[0]
            times = sorted(_jitter(rng, _between(rng, lo, hi)) for _ in range(n_logins))

            # Device assignment is chronological: the churn device cannot appear
            # before the day it was bought.
            plan: list[str | None] = [None] * n_logins
            pending = list(acct.devices)          # must each be seen at least once
            late_slots = [
                i for i, t in enumerate(times)
                if acct.late_device is not None
                and acct.late_device_from is not None
                and t >= acct.late_device_from
            ]
            for i in range(n_logins):
                if pending and (n_logins - i) <= len(pending):
                    plan[i] = pending.pop(0)      # reserve the tail for coverage
                elif i in late_slots and rng.random() < 0.65:
                    plan[i] = acct.late_device
                elif len(acct.devices) == 1 or rng.random() < 0.72:
                    plan[i] = primary
                else:
                    plan[i] = acct.devices[int(rng.integers(0, len(acct.devices)))]
            if acct.late_device is not None and late_slots and acct.late_device not in plan:
                plan[late_slots[-1]] = acct.late_device

            for when, device in zip(times, plan):
                region = acct.home_region
                if acct.secondary_region is not None and acct.secondary_from is not None:
                    if when >= acct.secondary_from:
                        region = acct.secondary_region
                elif rng.random() < 0.05:
                    region = self._pick_region()
                self.logins.append(
                    {
                        "user_id": acct.uid,
                        "login_date": when,
                        "ip_region": region,
                        "device_id": device,
                        "failed_attempts": self._failed_attempts(),
                    }
                )

            # Benign credential fumbling: ~12% of accounts have one bad day with
            # 4-6 failed attempts, so "max_failed_attempts > 3" is not ATO-only -
            # it stays a noisy heuristic rather than a label proxy.
            if acct.fraud_pattern == "" and rng.random() < 0.12:
                when = _jitter(rng, _between(rng, lo, hi))
                self.logins.append(
                    {
                        "user_id": acct.uid,
                        "login_date": when,
                        "ip_region": acct.home_region,
                        "device_id": primary,
                        "failed_attempts": int(rng.integers(4, 7)),
                    }
                )

        # ---- ATO: failed-login burst, then new device + new region --------
        for acct in self._ato_accounts():
            if acct.takeover_at is None:
                continue
            new_device = acct.takeover_device or self._new_device()
            new_region = self._pick_region()
            while new_region == acct.home_region:
                new_region = self._pick_region()

            for _ in range(int(rng.integers(3, 8))):
                when = acct.takeover_at - timedelta(minutes=float(rng.uniform(5, 600)))
                self.logins.append(
                    {
                        "user_id": acct.uid,
                        "login_date": when,
                        "ip_region": new_region,
                        "device_id": new_device,
                        "failed_attempts": int(rng.integers(2, 5)),
                    }
                )
            for _ in range(int(rng.integers(3, 10))):
                when = _between(rng, acct.takeover_at, TODAY)
                self.logins.append(
                    {
                        "user_id": acct.uid,
                        "login_date": when,
                        "ip_region": new_region,
                        "device_id": new_device,
                        "failed_attempts": int(rng.choice([0, 1], p=[0.85, 0.15])),
                    }
                )

    # -- 7. assembly -------------------------------------------------------

    def _orders_frame(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame(self.orders)
        df = df.sort_values(
            ["order_date", "customer_id", "seller_id", "amount"], kind="mergesort"
        ).reset_index(drop=True)
        df.insert(0, "order_id", np.arange(1, len(df) + 1, dtype=np.int64))

        rng = self.rng
        refund_rows = []
        refund_src = df[df["_refund"]].copy()
        for row in refund_src.itertuples(index=False):
            delay = float(rng.gamma(shape=2.2, scale=4.0)) + 1.0
            when = row.order_date + timedelta(days=delay)
            if when > TODAY:
                continue  # still in-flight at the observation cut-off
            refund_rows.append(
                {
                    "order_id": int(row.order_id),
                    "refund_date": when,
                    "refund_reason": str(
                        rng.choice(REFUND_REASONS, p=[0.34, 0.26, 0.33, 0.07])
                    ),
                    "refund_status": str(
                        rng.choice(REFUND_STATUSES, p=[0.72, 0.19, 0.09])
                    ),
                }
            )
        refunds = pd.DataFrame(refund_rows, columns=["order_id", "refund_date", "refund_reason", "refund_status"])
        refunds = refunds.sort_values(["refund_date", "order_id"], kind="mergesort").reset_index(drop=True)
        refunds.insert(0, "refund_id", np.arange(1, len(refunds) + 1, dtype=np.int64))
        refunds["refund_date"] = refunds["refund_date"].apply(_ts)

        orders_out = df.drop(columns=["_refund"]).copy()
        orders_out["order_date"] = orders_out["order_date"].apply(_ts)
        orders_out = orders_out[
            ["order_id", "customer_id", "seller_id", "product_id", "order_date", "amount", "payment_status"]
        ]
        return orders_out, refunds, df

    def _customers_frame(self) -> pd.DataFrame:
        rows = []
        for cid in sorted(self.customers):
            c = self.customers[cid]
            rows.append(
                {
                    "customer_id": cid,
                    "signup_date": _day(c.signup),
                    "region": c.coarse_region,
                    "device_type": self._choice(DEVICE_TYPES, np.array([0.58, 0.32, 0.10])),
                    "account_age_days": (TODAY - c.signup).days,
                }
            )
        return pd.DataFrame(rows)

    def _sellers_frame(self) -> pd.DataFrame:
        rng = self.rng
        rows = []
        for sid in sorted(self.sellers):
            s = self.sellers[sid]
            # Prior risk history is weakly (not deterministically) related to fraud:
            # plenty of clean sellers carry a 'minor' flag and most fraud sellers
            # carry none, so this column cannot act as a label proxy.
            if s.fraud_pattern:
                hist = str(rng.choice(["none", "minor", "major"], p=[0.62, 0.26, 0.12]))
            else:
                hist = str(rng.choice(["none", "minor", "major"], p=[0.74, 0.21, 0.05]))
            rows.append(
                {
                    "seller_id": sid,
                    "join_date": _day(s.signup),
                    "seller_region": s.coarse_region,
                    "category": self._choice(CATEGORIES),
                    "risk_flag_history": hist,
                }
            )
        return pd.DataFrame(rows)

    def _ratings_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.ratings)
        df = df.sort_values(
            ["rating_date", "customer_id", "seller_id"], kind="mergesort"
        ).reset_index(drop=True)
        df.insert(0, "rating_id", np.arange(1, len(df) + 1, dtype=np.int64))
        df["rating_date"] = df["rating_date"].apply(_ts)
        return df[["rating_id", "customer_id", "seller_id", "rating_value", "rating_date"]]

    def _logins_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.logins)
        df = df.sort_values(
            ["login_date", "user_id", "device_id"], kind="mergesort"
        ).reset_index(drop=True)
        df["login_date"] = df["login_date"].apply(_ts)
        return df[["user_id", "login_date", "ip_region", "device_id", "failed_attempts"]]

    def _ground_truth_frame(self) -> pd.DataFrame:
        rows = []
        for cid in sorted(self.customers):
            p = self.customers[cid].fraud_pattern
            rows.append(
                {
                    "entity_type": "customer",
                    "entity_id": str(cid),
                    "is_fraud": 1 if p else 0,
                    "fraud_pattern": p,
                }
            )
        for sid in sorted(self.sellers):
            p = self.sellers[sid].fraud_pattern
            rows.append(
                {
                    "entity_type": "seller",
                    "entity_id": str(sid),
                    "is_fraud": 1 if p else 0,
                    "fraud_pattern": p,
                }
            )
        return pd.DataFrame(rows, columns=["entity_type", "entity_id", "is_fraud", "fraud_pattern"])

    # -- orchestration -----------------------------------------------------

    def run(self) -> dict[str, pd.DataFrame]:
        self.build_entities()
        self.assign_fraud_roles()
        self.build_devices()
        self.build_orders()
        orders_df, refunds_df, orders_raw = self._orders_frame()
        self.build_ratings(orders_raw)
        self.build_logins()
        return {
            "customers": self._customers_frame(),
            "sellers": self._sellers_frame(),
            "orders": orders_df,
            "refunds": refunds_df,
            "ratings": self._ratings_frame(),
            "login_activity": self._logins_frame(),
            "ground_truth": self._ground_truth_frame(),
        }


# --------------------------------------------------------------------------
# Self-check: the injected fraud must not be trivially separable
# --------------------------------------------------------------------------


def build_eval_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rebuild the pipeline's entity feature space in-memory.

    Mirrors src/mri/features/engineering.py so the separability check is run on
    exactly the features a model would see. Kept local (rather than importing)
    because engineering.py reads the SQLite build, which does not exist yet at
    generation time.
    """
    customers = tables["customers"].copy()
    sellers = tables["sellers"].copy()
    orders = tables["orders"].copy()
    refunds = tables["refunds"].copy()
    ratings = tables["ratings"].copy()
    login = tables["login_activity"].copy()

    orders["order_date"] = pd.to_datetime(orders["order_date"])
    ratings["rating_date"] = pd.to_datetime(ratings["rating_date"])

    def safe_div(a, b):
        return (a / b.replace(0, np.nan)).fillna(0)

    seller_orders = orders.groupby("seller_id").agg(
        total_orders=("order_id", "count"),
        avg_order_value=("amount", "mean"),
        total_revenue=("amount", "sum"),
        unique_customers=("customer_id", "nunique"),
    ).reset_index()
    rs = refunds.merge(orders[["order_id", "seller_id"]], on="order_id", how="left")
    seller_refunds = rs.groupby("seller_id").agg(total_refunds=("refund_id", "count")).reset_index()
    seller_ratings = ratings.groupby("seller_id").agg(
        avg_rating=("rating_value", "mean"),
        rating_count=("rating_id", "count"),
        rating_std=("rating_value", "std"),
    ).reset_index()
    orders["order_day"] = orders["order_date"].dt.date
    daily = orders.groupby(["seller_id", "order_day"]).size().reset_index(name="daily_orders")
    spikes = daily.groupby("seller_id").agg(
        max_daily_orders=("daily_orders", "max"),
        avg_daily_orders=("daily_orders", "mean"),
    ).reset_index()
    spikes["order_spike_ratio"] = safe_div(spikes["max_daily_orders"], spikes["avg_daily_orders"])

    sf = (sellers.merge(seller_orders, on="seller_id", how="left")
                 .merge(seller_refunds, on="seller_id", how="left")
                 .merge(seller_ratings, on="seller_id", how="left")
                 .merge(spikes, on="seller_id", how="left"))
    sf["total_refunds"] = sf["total_refunds"].fillna(0)
    sf["refund_rate"] = safe_div(sf["total_refunds"], sf["total_orders"].fillna(0))
    sf["entity_id"] = sf["seller_id"].astype(str)
    sf["entity_type"] = "seller"

    customer_orders = orders.groupby("customer_id").agg(
        total_orders=("order_id", "count"),
        avg_order_value=("amount", "mean"),
        total_spend=("amount", "sum"),
        unique_sellers=("seller_id", "nunique"),
    ).reset_index()
    rc = refunds.merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
    customer_refunds = rc.groupby("customer_id").agg(total_refunds=("refund_id", "count")).reset_index()
    customer_ratings = ratings.groupby("customer_id").agg(
        total_ratings=("rating_id", "count"),
        avg_rating_given=("rating_value", "mean"),
    ).reset_index()

    login["entity_type"] = login["user_id"].str[0].map({"C": "customer", "S": "seller"})
    login["numeric_id"] = login["user_id"].str[1:]
    cl = (login[login["entity_type"] == "customer"].groupby("numeric_id").agg(
        login_events=("user_id", "count"),
        avg_failed_attempts=("failed_attempts", "mean"),
        max_failed_attempts=("failed_attempts", "max"),
        unique_devices=("device_id", "nunique"),
        unique_regions=("ip_region", "nunique"),
    ).reset_index().rename(columns={"numeric_id": "customer_id"}))
    cl["customer_id"] = cl["customer_id"].astype(int)

    cf = (customers.merge(customer_orders, on="customer_id", how="left")
                   .merge(customer_refunds, on="customer_id", how="left")
                   .merge(customer_ratings, on="customer_id", how="left")
                   .merge(cl, on="customer_id", how="left"))
    cf["total_refunds"] = cf["total_refunds"].fillna(0)
    cf["refund_rate"] = safe_div(cf["total_refunds"], cf["total_orders"].fillna(0))
    cf["entity_id"] = cf["customer_id"].astype(str)
    cf["entity_type"] = "customer"

    seller_cols = ["entity_id", "entity_type", "total_orders", "avg_order_value", "total_revenue",
                   "unique_customers", "total_refunds", "refund_rate", "avg_rating", "rating_count",
                   "rating_std", "max_daily_orders", "avg_daily_orders", "order_spike_ratio"]
    customer_cols = ["entity_id", "entity_type", "total_orders", "avg_order_value", "total_spend",
                     "unique_sellers", "total_refunds", "refund_rate", "total_ratings",
                     "avg_rating_given", "login_events", "avg_failed_attempts", "max_failed_attempts",
                     "unique_devices", "unique_regions", "account_age_days"]
    for col in seller_cols:
        if col not in sf.columns:
            sf[col] = 0
    for col in customer_cols:
        if col not in cf.columns:
            cf[col] = 0

    allf = pd.concat([sf[seller_cols], cf[customer_cols]], ignore_index=True, sort=False).fillna(0)
    return allf


def separability_check(tables: dict[str, pd.DataFrame]) -> dict:
    """Train a depth-2 decision tree on the raw features against the labels.

    NOTE: this is the ONLY place ground truth may legitimately meet the feature
    space, and it exists purely to prove the dataset is not trivially separable.
    An AUC above ~0.95 here means the fraud is too easy and must be made subtler.
    """
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:  # pragma: no cover - sklearn is a declared dependency
        return {"error": "scikit-learn unavailable; separability check skipped"}

    feats = build_eval_features(tables)
    gt = tables["ground_truth"].copy()
    gt["entity_id"] = gt["entity_id"].astype(str)
    merged = feats.merge(gt[["entity_type", "entity_id", "is_fraud"]],
                         on=["entity_type", "entity_id"], how="left")
    merged["is_fraud"] = merged["is_fraud"].fillna(0).astype(int)

    feature_cols = [c for c in merged.columns
                    if c not in ("entity_id", "entity_type", "is_fraud")]
    X = merged[feature_cols].to_numpy(dtype=float)
    y = merged["is_fraud"].to_numpy(dtype=int)

    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    in_sample = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    proba = cross_val_predict(
        DecisionTreeClassifier(max_depth=2, random_state=0), X, y, cv=cv, method="predict_proba"
    )[:, 1]
    cv_auc = float(roc_auc_score(y, proba))

    best = feature_cols[int(np.argmax(clf.feature_importances_))]
    return {
        "in_sample_auc": in_sample,
        "cv_auc": cv_auc,
        "top_split_feature": best,
        "n_rows": int(len(merged)),
        "n_fraud": int(y.sum()),
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _report(tables: dict[str, pd.DataFrame]) -> None:
    gt = tables["ground_truth"]
    login = tables["login_activity"]

    print("\n--- entity counts ---")
    print(f"customers : {len(tables['customers'])}")
    print(f"sellers   : {len(tables['sellers'])}")
    print(f"orders    : {len(tables['orders'])}")
    print(f"refunds   : {len(tables['refunds'])}")
    print(f"ratings   : {len(tables['ratings'])}")
    print(f"logins    : {len(login)}")

    print("\n--- ground truth ---")
    n_fraud = int(gt["is_fraud"].sum())
    print(f"entities  : {len(gt)}  fraudulent: {n_fraud} ({n_fraud / len(gt):.2%})")
    counts = gt[gt["is_fraud"] == 1].groupby(["fraud_pattern", "entity_type"]).size()
    for (pattern, etype), n in counts.items():
        print(f"  {pattern:<16} {etype:<9} {n}")

    print("\n--- device / region cardinality ---")
    per_device = login.groupby("device_id")["user_id"].nunique()
    per_account = login.groupby("user_id")["device_id"].nunique()
    per_region = login.groupby("ip_region")["user_id"].nunique()
    print(f"distinct devices          : {per_device.size}")
    print(f"accounts per device       : mean {per_device.mean():.2f} max {per_device.max()} "
          f"| shared by >3: {(per_device > 3).sum()} devices")
    print(f"devices per account       : mean {per_account.mean():.2f} max {per_account.max()}")
    print(f"distinct ip_regions       : {per_region.size}")
    print(f"accounts per region       : mean {per_region.mean():.2f} "
          f"max {per_region.max()} median {per_region.median():.0f}")


def main() -> None:
    base_dir = str(RAW_DATA_DIR)
    os.makedirs(base_dir, exist_ok=True)

    tables = MarketplaceGenerator(SEED).run()

    for name, df in tables.items():
        df.to_csv(os.path.join(base_dir, f"{name}.csv"), index=False)

    print(f"Data generated and saved to: {base_dir}")
    _report(tables)

    result = separability_check(tables)
    print("\n--- separability self-check (depth-2 decision tree vs is_fraud) ---")
    if "error" in result:
        print(result["error"])
        return
    print(f"in-sample AUC : {result['in_sample_auc']:.4f}")
    print(f"5-fold CV AUC : {result['cv_auc']:.4f}")
    print(f"dominant split: {result['top_split_feature']}")
    if result["in_sample_auc"] > 0.95:
        print("WARNING: AUC > 0.95 - the injected fraud is too easy, make the patterns subtler.")
    else:
        print("OK: fraud is not trivially separable by a shallow single/two-feature rule.")


if __name__ == "__main__":
    main()
