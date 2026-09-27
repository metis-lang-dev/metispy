"""Continuous RSS reference: Lemma 2's closed-form d_min evaluated on raw
CARLA geometry (meters, m/s — no cells, no buckets).

The ad-rss sensor computes client-side and needs the RSS-variant
PythonAPI wheel (`make PythonAPI.rss`), which this lab's wheels lack.
The sensor's longitudinal core is this same formula, so the cross-check
keeps its substance: does our *discretized* verdict stay conservative
against the *exact* continuous one on real distances? The same physical
parameters are used on both sides (cells/ticks converted to SI)."""

from .profile import CAR, EnvProfile

# the CAR profile's discrete parameters in SI units (import-compat)
ACC = CAR.acc_si
BMIN = CAR.bmin_si
BMAX = CAR.bmax_si
RHO_S = CAR.rho_si


def d_min_continuous(v_r: float, v_f: float,
                     profile: EnvProfile = CAR) -> float:
    """RSS Lemma 2 (same direction), front-most to rear-most points:
    d_min = [v_r*rho + a*rho^2/2 + (v_rho)^2/(2*b_min)
             - v_f^2/(2*b_max)]_+  — parameters from the environment
    profile's SI mapping of the same discrete constants. v_rho is
    capped at the platform's top speed (the discrete series caps at
    v_max buckets; a speed-limited platform cannot accelerate past its
    cap during rho — ad-rss's maxSpeedOnAcceleration, and our
    actuator-mirror doctrine applied to the cross-check). For the car
    profile the cap (15 m/s) sits far above the 6 m/s policy cap, so
    behavior is unchanged."""
    acc, bmin, bmax, rho = (profile.acc_si, profile.bmin_si,
                            profile.bmax_si, profile.rho_si)
    v_top = profile.cells_to_speed(profile.v_max)
    v_rho = min(v_r + rho * acc, max(v_r, v_top))
    d = (v_r * rho + 0.5 * acc * rho ** 2
         + v_rho ** 2 / (2.0 * bmin)
         - v_f ** 2 / (2.0 * bmax))
    return max(0.0, d)


class ContinuousRssCheck:
    """Per-tick agreement between the bucketed verdict and the continuous
    formula on real gaps. counts[(ours, cont)] like the sensor variant."""

    def __init__(self, profile: EnvProfile = CAR):
        self.profile = profile
        self.counts = {(a, b): 0 for a in (0, 1) for b in (0, 1)}
        self.margin_min = float("inf")   # worst continuous margin seen
        self.samples = 0
        self.no_front = 0
        self.disagreements = []

    def record(self, ours_dangerous: bool, gap_m, v_r, v_f):
        if gap_m is None:
            self.no_front += 1
            return
        d = d_min_continuous(v_r, v_f, self.profile)
        cont = gap_m < d
        self.counts[(int(ours_dangerous), int(cont))] += 1
        if int(ours_dangerous) != int(cont) and \
                len(self.disagreements) < 12:
            self.disagreements.append(
                {"ours": ours_dangerous, "cont": cont,
                 "gap_m": round(gap_m, 1), "v_r": round(v_r, 1),
                 "v_f": round(v_f, 1), "d_min": round(d, 1)})
        self.margin_min = min(self.margin_min, gap_m - d)
        self.samples += 1

    def report(self) -> str:
        c = self.counts
        lines = [
            f"continuous-RSS cross-check: {self.samples} ticks with a "
            f"front vehicle, {self.no_front} without",
            f"  both safe      : {c[(0, 0)]}",
            f"  both dangerous : {c[(1, 1)]}",
            f"  ours-only      : {c[(1, 0)]}   (discretization "
            "conservatism)",
            f"  cont-only      : {c[(0, 1)]}   (MUST be 0: discrete "
            "verdict missed a continuous danger)",
        ]
        if self.samples:
            lines.append(f"  worst continuous margin: "
                         f"{self.margin_min:+.1f} m")
        for d in self.disagreements:
            lines.append(f"    disagree: {d}")
        return "\n".join(lines)
