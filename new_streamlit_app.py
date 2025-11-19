# === BMW AI Implementation Cost & Benefit Model — Redesigned Dashboard ===
# Author: Hank 
# Goal: Professional presentation-grade dashboard for AI deployment analysis

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy import stats
from scipy.stats import shapiro, jarque_bera, norm
from scipy.optimize import minimize, fsolve
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import io
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from fpdf import FPDF

# ============================================================================
# DATA BREACH RISK VALUATION CLASSES
# ============================================================================
# These classes are used in Page 6 for data breach risk analysis

@dataclass
class BreachProbabilitySources:
    """Breach probability parameter estimates with uncertainty distributions."""
    MANUF_BASE_RATE: float = 0.048
    MANUF_BASE_RATE_STD: float = 0.008
    SIZE_SCALING_EXPONENT: float = 0.32
    SIZE_SCALING_EXPONENT_STD: float = 0.04
    RATING_PROBABILITIES: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'A': (0.028, 0.005), 'B+': (0.071, 0.012), 'B': (0.115, 0.018), 'C': (0.198, 0.025)
    })
    SIZE_RATING_CORRELATION: float = 0.45

@dataclass
class ImpactEstimationSources:
    """Conditional loss estimation parameters."""
    BENCHMARK_BASE_COST: float = 360
    BENCHMARK_COST_STD: float = 120
    INSURANCE_POLICY_LIMIT: float = 1000
    INSURANCE_LOSS_RATIO: float = 0.42
    INSURANCE_LOADING_FACTOR: float = 1.40
    INSURANCE_LOADING_FACTOR_STD: float = 0.15

@dataclass
class CombinedWeights:
    """Weighting scheme for impact signal fusion."""
    benchmark: float = 0.60
    insurance: float = 0.40

class BreachProbabilityModel:
    """Estimates annual breach probability via multi-signal fusion."""
    def __init__(self, sources: Optional[BreachProbabilitySources] = None):
        self.sources = sources or BreachProbabilitySources()
    
    def estimate_size_adjusted_probability(self, revenue: float, n_sim: int = 10000) -> Tuple[float, float]:
        base_rate = self.sources.MANUF_BASE_RATE
        median_revenue = 50.0
        exponents = np.random.normal(self.sources.SIZE_SCALING_EXPONENT, self.sources.SIZE_SCALING_EXPONENT_STD, n_sim)
        size_factors = (revenue / median_revenue) ** exponents
        adjusted_rates = base_rate * size_factors
        return np.mean(adjusted_rates), np.std(adjusted_rates)
    
    def estimate_rating_probability(self, rating: str) -> Tuple[float, float]:
        if rating not in self.sources.RATING_PROBABILITIES:
            raise ValueError(f"Rating {rating} not in {list(self.sources.RATING_PROBABILITIES.keys())}")
        return self.sources.RATING_PROBABILITIES[rating]
    
    def combined_probability_estimation(self, revenue: float, rating: str, n_sim: int = 100000) -> Dict:
        p_size, std_size = self.estimate_size_adjusted_probability(revenue)
        p_rating, std_rating = self.estimate_rating_probability(rating)
        precision_size = 1 / (std_size ** 2)
        precision_rating = 1 / (std_rating ** 2)
        total_precision = precision_size + precision_rating
        p_combined = (precision_size * p_size + precision_rating * p_rating) / total_precision
        correlation = self.sources.SIZE_RATING_CORRELATION
        var_combined = (1 / total_precision) + (2 * correlation * std_size * std_rating * (precision_size * precision_rating) / (total_precision ** 2))
        std_combined = np.sqrt(var_combined)
        cov_matrix = [[std_size**2, correlation * std_size * std_rating], [correlation * std_size * std_rating, std_rating**2]]
        samples = np.random.multivariate_normal([p_size, p_rating], cov_matrix, n_sim)
        weights = np.array([precision_size, precision_rating]) / total_precision
        combined_samples = np.sum(samples * weights, axis=1)
        return {
            'expected': p_combined, 'std_dev': std_combined, 'distribution': combined_samples,
            'components': {'size_adjusted': (p_size, std_size), 'rating_adjusted': (p_rating, std_rating)}
        }

class ImpactEstimationModel:
    """Estimates conditional loss magnitude given breach occurs."""
    def __init__(self, sources: Optional[ImpactEstimationSources] = None):
        self.sources = sources or ImpactEstimationSources()
    
    def estimate_benchmark_value(self, revenue: float, n_sim: int = 50000) -> Dict:
        base_cost = self.sources.BENCHMARK_BASE_COST
        cost_std = self.sources.BENCHMARK_COST_STD
        revenue_factor = np.log(revenue / 50.0) * 0.15 + 1.0
        log_mean = np.log(base_cost * revenue_factor)
        log_std = cost_std / base_cost
        simulated_costs = np.random.lognormal(log_mean, log_std, n_sim)
        return {'expected': np.mean(simulated_costs), 'std_dev': np.std(simulated_costs), 'distribution': simulated_costs}
    
    def estimate_insurance_implied_value(self, n_sim: int = 50000) -> Dict:
        policy_limit = self.sources.INSURANCE_POLICY_LIMIT
        loss_ratio = self.sources.INSURANCE_LOSS_RATIO
        loading_factor = self.sources.INSURANCE_LOADING_FACTOR
        expected_loss = (policy_limit * loss_ratio) / loading_factor
        loading_samples = np.random.normal(loading_factor, self.sources.INSURANCE_LOADING_FACTOR_STD, n_sim)
        simulated_values = (policy_limit * loss_ratio) / loading_samples
        return {'expected': np.mean(simulated_values), 'std_dev': np.std(simulated_values), 'distribution': simulated_values}
    
    def combined_impact_estimation(self, revenue: float, weights: Optional[CombinedWeights] = None) -> Dict:
        if weights is None:
            weights = CombinedWeights()
        benchmark_result = self.estimate_benchmark_value(revenue)
        insurance_result = self.estimate_insurance_implied_value()
        weight_dict = {'benchmark': weights.benchmark, 'insurance': weights.insurance}
        total_weight = sum(weight_dict.values())
        normalized_weights = {k: v/total_weight for k, v in weight_dict.items()}
        min_len = min(len(benchmark_result['distribution']), len(insurance_result['distribution']))
        combined_samples = (normalized_weights['benchmark'] * benchmark_result['distribution'][:min_len] + 
                          normalized_weights['insurance'] * insurance_result['distribution'][:min_len])
        return {
            'expected': np.mean(combined_samples), 'std_dev': np.std(combined_samples), 'distribution': combined_samples,
            'components': {'benchmark': benchmark_result, 'insurance_implied': insurance_result},
            'weights': normalized_weights
        }

class MonteCarloValuation:
    """Joint simulation of breach probability and impact with correlation."""
    def __init__(self, breach_model: BreachProbabilityModel, impact_model: ImpactEstimationModel, p_l_correlation: float = 0.0):
        self.breach_model = breach_model
        self.impact_model = impact_model
        self.p_l_correlation = p_l_correlation
    
    def _induce_correlation_via_copula(self, p_samples: np.ndarray, l_samples: np.ndarray, correlation: float) -> Tuple[np.ndarray, np.ndarray]:
        if abs(correlation) < 1e-6:
            return p_samples, l_samples
        n = min(len(p_samples), len(l_samples))
        p_samples = p_samples[:n]
        l_samples = l_samples[:n]
        p_ranks = stats.rankdata(p_samples) / (n + 1)
        l_ranks = stats.rankdata(l_samples) / (n + 1)
        p_normal = norm.ppf(p_ranks)
        l_normal = norm.ppf(l_ranks)
        l_normal_corr = correlation * p_normal + np.sqrt(1 - correlation**2) * l_normal
        l_uniform_corr = norm.cdf(l_normal_corr)
        l_sorted = np.sort(l_samples)
        l_correlated = l_sorted[(l_uniform_corr * (n-1)).astype(int)]
        return p_samples, l_correlated
    
    def run_simulation(self, revenue: float, rating: str, n_simulations: int = 100000) -> Dict:
        p_results = self.breach_model.combined_probability_estimation(revenue, rating, n_simulations)
        l_results = self.impact_model.combined_impact_estimation(revenue)
        p_samples = p_results['distribution'].copy()
        l_samples = l_results['distribution'].copy()
        p_final, l_final = self._induce_correlation_via_copula(p_samples, l_samples, self.p_l_correlation)
        min_samples = min(len(p_final), len(l_final), n_simulations)
        p_final = p_final[:min_samples]
        l_final = l_final[:min_samples]
        expected_loss = p_final * l_final
        return {
            'expected_loss_distribution': expected_loss, 'expected_value': np.mean(expected_loss), 'std_dev': np.std(expected_loss),
            'confidence_interval_95': (np.percentile(expected_loss, 2.5), np.percentile(expected_loss, 97.5)),
            'percentile_5': np.percentile(expected_loss, 5), 'percentile_25': np.percentile(expected_loss, 25),
            'percentile_50': np.percentile(expected_loss, 50), 'percentile_75': np.percentile(expected_loss, 75),
            'percentile_95': np.percentile(expected_loss, 95), 'probability_distribution': p_final, 'impact_distribution': l_final,
            'components': {'p_breach': (np.mean(p_final), np.std(p_final)), 'l_impact': (np.mean(l_final), np.std(l_final))},
            'actual_correlation': np.corrcoef(p_final, l_final)[0,1]
        }

def validate_against_benchmarks(expected_loss: float, revenue: float, p_breach: float) -> Dict:
    """Validates model output against industry benchmarks."""
    revenue_millions = revenue * 1000
    loss_ratio = expected_loss / revenue_millions
    return {
        'breach_frequency_check': {'estimate': p_breach, 'benchmark_range': (0.04, 0.06), 'pass': 0.03 <= p_breach <= 0.08, 'source': 'Verizon DBIR 2023'},
        'loss_ratio_check': {'estimate': loss_ratio, 'benchmark_range': (0.005, 0.02), 'pass': 0.003 <= loss_ratio <= 0.03, 'source': 'Aon 2024'},
        'magnitude_sanity': {'estimate': expected_loss, 'benchmark_range': (100, 800), 'pass': 50 <= expected_loss <= 1000, 'source': 'IBM-Ponemon 2023'}
    }

def sensitivity_analysis(breach_model: BreachProbabilityModel, impact_model: ImpactEstimationModel, 
                         base_revenue: float, base_rating: str, p_l_correlation: float = 0.0) -> Dict:
    """Tornado analysis of key parameter sensitivities."""
    scenarios = {
        'Base Case': {'revenue': base_revenue, 'rating': base_rating},
        'Worse Security (B)': {'revenue': base_revenue, 'rating': 'B'},
        'Better Security (A)': {'revenue': base_revenue, 'rating': 'A'},
        'Revenue +20%': {'revenue': base_revenue * 1.2, 'rating': base_rating},
        'Revenue -20%': {'revenue': base_revenue * 0.8, 'rating': base_rating},
    }
    sensitivity_results = {}
    mc = MonteCarloValuation(breach_model, impact_model, p_l_correlation)
    for scenario, params in scenarios.items():
        result = mc.run_simulation(revenue=params['revenue'], rating=params['rating'], n_simulations=100000)
        sensitivity_results[scenario] = {
            'expected': result['expected_value'], 'std_dev': result['std_dev'], 'ci_95': result['confidence_interval_95'],
            'p5': result['percentile_5'], 'p95': result['percentile_95'],
            'p_breach': result['components']['p_breach'][0], 'l_impact': result['components']['l_impact'][0]
        }
    return sensitivity_results

# ============================================================================
# PARAMETER SYSTEM: SINGLE SOURCE OF TRUTH
# ============================================================================

@dataclass
class Params:
    """Single source of truth for all dashboard parameters. All values in USD."""
    # =========================
    # PAIRED COST PARAMETERS  (Before & After)
    # =========================
    wage_usd_per_hour_before: float = 30.0        # w (before)
    wage_usd_per_hour_after: float = 30.0         # w (after)
    overhead_multiplier_before: float = 1.3      # φ (before)
    overhead_multiplier_after: float = 1.3       # φ (after)

    scrap_rate_before: float = 0.002               # s_before
    scrap_rate_after: float = 0.0015                # s_after

    security_spend_usd_per_year_before: float = 100000.0   # S_before
    security_spend_usd_per_year_after: float = 150000.0    # S_after

    # =========================
    # BEFORE-ONLY COSTS (capital before AI)
    # =========================
    capex_usd_before: float = 50000.0            # legacy/prev capital $ (before)
    useful_life_years_before: float = 7.0        # Y_before (for annualization)
    
    # Before AI operations (typically smaller than after, but included for completeness)
    labeling_time_hours_per_label_before: float = 0.001   # τ (before) - much smaller than after (0.005)
    labels_per_year_before: float = 1000                 # n_ell (before) - much smaller than after (12000)
    dataset_tb_before: float = 1.0                      # dataset size before AI
    storage_usd_per_tb_year_before: float = 276.0       # c_TB (before)
    etl_usd_per_tb_year_before: float = 480.0           # α (before)
    mlops_usd_per_model_year_before: float = 0.0         # β (before) - keep at 0
    models_deployed_before: int = 0                      # n_m (before) - keep at 0

    # =========================
    # AFTER-ONLY COSTS (AI program)
    # =========================
    labeling_time_hours_per_label_after: float = 0.005   # τ
    labels_per_year_after: float = 12000                 # n_ell
    dataset_tb_after: float = 5.0                        # dataset size for storage/ETL

    storage_usd_per_tb_year_after: float = 276.0        # c_TB
    etl_usd_per_tb_year_after: float = 480.0            # α

    mlops_usd_per_model_year_after: float = 8400.0       # β
    models_deployed_after: int = 3                  # n_m

    capex_usd_after: float = 360000.0                      # CapEx
    useful_life_years_after: float = 7.0              # Y

    # =========================
    # BENEFIT PARAMETERS
    # =========================
    oee_improvement_fraction: float = 0.03             # ΔOEE
    downtime_hours_avoided_per_year: float = 75.0      # H_dt
    overhead_usd_per_downtime_hour: float = 5000.0       # c_oh
    restart_scrap_fraction: float = 0.02               # r_restart
    contribution_margin_usd_per_unit: float = 250.0     # CM

    # =========================
    # SHARED CONTEXT
    # =========================
    units_per_year: float = 100000.0                       # u
    operating_hours_per_year: float = 4000.0             # h_op
    material_cost_usd_per_unit: float = 300.0           # c_mat

    breach_loss_envelope_usd: float = 280000.0             # L_breach (annualized)
    security_effectiveness_per_dollar: float = np.log(2) / 100000.0    # η  (e.g., ln(2)/100000)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for compatibility with legacy code."""
        return {k: v for k, v in self.__dict__.items()}

def log_debug(msg: str, payload: Optional[Dict] = None):
    """Lightweight debug logging utility."""
    if st.session_state.get('debug_mode', False):
        timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
        st.write(f"[{timestamp}] {msg}")
        if payload:
            st.json(payload)

# ============================================================================
# PURE COMPUTE FUNCTIONS
# ============================================================================
import math

def expected_risk_loss(S: float, L_breach: float, eta: float) -> float:
    """Calculate expected annual risk loss: L_breach * exp(-eta * S)"""
    return L_breach * math.exp(-eta * S)

def annualized_capex(capex: float, years: float) -> float:
    """Annualize capital expenditure."""
    return capex / years if years > 0 else 0.0

def scrap_cost(s: float, u: float, c_mat: float) -> float:
    """Calculate scrap cost: s * u * c_mat"""
    return s * u * c_mat

def labeling_cost_before(p: Params) -> float:
    """Calculate labeling cost before AI: τ * n_ell * w_before * φ_before"""
    return (p.labeling_time_hours_per_label_before * p.labels_per_year_before * 
            p.wage_usd_per_hour_before * p.overhead_multiplier_before)

def labeling_cost_after(p: Params) -> float:
    """Calculate labeling cost after AI: τ * n_ell * w_after * φ_after"""
    return (p.labeling_time_hours_per_label_after * p.labels_per_year_after * 
            p.wage_usd_per_hour_after * p.overhead_multiplier_after)

def storage_cost_before(p: Params) -> float:
    """Calculate storage cost before AI: c_TB * dataset_tb_before"""
    return p.storage_usd_per_tb_year_before * p.dataset_tb_before

def storage_cost_after(p: Params) -> float:
    """Calculate storage cost after AI: c_TB * dataset_tb_after"""
    return p.storage_usd_per_tb_year_after * p.dataset_tb_after

def ops_cost_before(p: Params) -> float:
    """Calculate operations cost before AI: C^ops = α_yr * D + β^ops_yr * N_m"""
    etl_component = p.etl_usd_per_tb_year_before * p.dataset_tb_before
    mlops_component = p.mlops_usd_per_model_year_before * p.models_deployed_before
    return etl_component + mlops_component

def ops_cost_after(p: Params) -> float:
    """Calculate operations cost after AI: C^ops = α_yr * D + β^ops_yr * N_m"""
    etl_component = p.etl_usd_per_tb_year_after * p.dataset_tb_after
    mlops_component = p.mlops_usd_per_model_year_after * p.models_deployed_after
    return etl_component + mlops_component

# Keep separate functions for breakdown display
def etl_cost_before(p: Params) -> float:
    """Calculate ETL cost before AI: α * dataset_tb_before"""
    return p.etl_usd_per_tb_year_before * p.dataset_tb_before

def etl_cost_after(p: Params) -> float:
    """Calculate ETL cost after AI: α * dataset_tb_after"""
    return p.etl_usd_per_tb_year_after * p.dataset_tb_after

def mlops_cost_before(p: Params) -> float:
    """Calculate MLOps cost before AI: β * n_m"""
    return p.mlops_usd_per_model_year_before * p.models_deployed_before

def mlops_cost_after(p: Params) -> float:
    """Calculate MLOps cost after AI: β * n_m"""
    return p.mlops_usd_per_model_year_after * p.models_deployed_after

def total_cost_before(p: Params) -> float:
    """Calculate total cost before AI: C^label + C^store + C^ops + C^risk(S) + C^mat(s) + C^cap"""
    cap = annualized_capex(p.capex_usd_before, p.useful_life_years_before)
    risk = expected_risk_loss(p.security_spend_usd_per_year_before, 
                             p.breach_loss_envelope_usd, 
                             p.security_effectiveness_per_dollar)
    scrap = scrap_cost(p.scrap_rate_before, p.units_per_year, p.material_cost_usd_per_unit)
    label = labeling_cost_before(p)
    store = storage_cost_before(p)
    ops = ops_cost_before(p)
    return cap + risk + scrap + label + store + ops

def total_cost_after(p: Params) -> float:
    """Calculate total cost after AI: C^label + C^store + C^ops + C^risk(S) + C^mat(s) + C^cap"""
    cap = annualized_capex(p.capex_usd_after, p.useful_life_years_after)
    risk = expected_risk_loss(p.security_spend_usd_per_year_after, 
                             p.breach_loss_envelope_usd, 
                             p.security_effectiveness_per_dollar)
    scrap = scrap_cost(p.scrap_rate_after, p.units_per_year, p.material_cost_usd_per_unit)
    label = labeling_cost_after(p)
    store = storage_cost_after(p)
    ops = ops_cost_after(p)
    return cap + risk + scrap + label + store + ops

def oee_benefit(p: Params) -> float:
    """Calculate OEE benefit: ΔOEE * u * CM"""
    return p.oee_improvement_fraction * p.units_per_year * p.contribution_margin_usd_per_unit

def downtime_benefit(p: Params) -> float:
    """Calculate downtime benefit: H_dt * c_oh"""
    return p.downtime_hours_avoided_per_year * p.overhead_usd_per_downtime_hour

def scrap_reduction_benefit(p: Params) -> float:
    """Calculate scrap reduction benefit: (s_before - s_after) * u * c_mat"""
    scrap_reduction = p.scrap_rate_before - p.scrap_rate_after
    return scrap_reduction * p.units_per_year * p.material_cost_usd_per_unit

def risk_benefit(p: Params) -> float:
    """Calculate risk reduction benefit: L(S_before) - L(S_after)"""
    risk_before = expected_risk_loss(p.security_spend_usd_per_year_before,
                                    p.breach_loss_envelope_usd,
                                    p.security_effectiveness_per_dollar)
    risk_after = expected_risk_loss(p.security_spend_usd_per_year_after,
                                   p.breach_loss_envelope_usd,
                                   p.security_effectiveness_per_dollar)
    return risk_before - risk_after

def total_benefits(p: Params) -> float:
    """Calculate total benefits: OEE + downtime + risk + scrap reduction"""
    return oee_benefit(p) + downtime_benefit(p) + risk_benefit(p) + scrap_reduction_benefit(p)

def compute_cost_before(p: Params) -> Dict:
    """Compute cost breakdown for Before AI period. Returns dict with components."""
    cap = annualized_capex(p.capex_usd_before, p.useful_life_years_before)
    risk = expected_risk_loss(p.security_spend_usd_per_year_before, 
                             p.breach_loss_envelope_usd, 
                             p.security_effectiveness_per_dollar)
    scrap = scrap_cost(p.scrap_rate_before, p.units_per_year, p.material_cost_usd_per_unit)
    label = labeling_cost_before(p)
    store = storage_cost_before(p)
    ops = ops_cost_before(p)
    # Keep separate components for breakdown display
    etl = etl_cost_before(p)
    mlops = mlops_cost_before(p)
    total = cap + risk + scrap + label + store + ops
    
    return {
        "capital": cap,
        "risk": risk,
        "scrap": scrap,
        "labeling": label,
        "storage": store,
        "ops": ops,
        "etl": etl,  # For breakdown display
        "mlops": mlops,  # For breakdown display
        "total": total,
        "point_estimate_usd": total
    }

def compute_cost_after(p: Params) -> Dict:
    """Compute cost breakdown for After AI period. Returns dict with components."""
    cap = annualized_capex(p.capex_usd_after, p.useful_life_years_after)
    risk = expected_risk_loss(p.security_spend_usd_per_year_after, 
                             p.breach_loss_envelope_usd, 
                             p.security_effectiveness_per_dollar)
    scrap = scrap_cost(p.scrap_rate_after, p.units_per_year, p.material_cost_usd_per_unit)
    label = labeling_cost_after(p)
    store = storage_cost_after(p)
    ops = ops_cost_after(p)
    # Keep separate components for breakdown display
    etl = etl_cost_after(p)
    mlops = mlops_cost_after(p)
    total = cap + risk + scrap + label + store + ops
    
    return {
        "capital": cap,
        "risk": risk,
        "scrap": scrap,
        "labeling": label,
        "storage": store,
        "ops": ops,
        "etl": etl,  # For breakdown display
        "mlops": mlops,  # For breakdown display
        "total": total,
        "point_estimate_usd": total
    }

def compute_benefits(p: Params) -> Dict:
    """Compute benefit breakdown. Returns dict with components."""
    oee = oee_benefit(p)
    downtime = downtime_benefit(p)
    risk = risk_benefit(p)
    scrap_reduction = scrap_reduction_benefit(p)
    total = oee + downtime + risk + scrap_reduction  # Calculate directly instead of calling total_benefits
    
    return {
        "oee_improvement": oee,
        "downtime_avoidance": downtime,
        "risk_reduction": risk,
        "scrap_reduction": scrap_reduction,
        "total": total,
        "point_estimate_usd": total,
        # Additional fields for display
        "downtime_cost_per_hour": p.overhead_usd_per_downtime_hour,
        "downtime_hours_avoided": p.downtime_hours_avoided_per_year
    }

def monte_carlo_cost(p: Params, period: str, n_simulations: int = 10000) -> Dict:
    """Monte Carlo simulation for cost. period in {"before", "after"}. Pure function."""
    if period == "before":
        base_cost = total_cost_before(p)
    elif period == "after":
        base_cost = total_cost_after(p)
    else:
        raise ValueError(f"period must be 'before' or 'after', got {period}")
    
    # Simple uncertainty model: add 10% coefficient of variation
    cv = 0.10
    mean = base_cost
    std = mean * cv
    
    # Generate samples (lognormal to ensure positive)
    if mean > 0:
        log_mean = np.log(mean / np.sqrt(1 + (std/mean)**2))
        log_std = np.sqrt(np.log(1 + (std/mean)**2))
        samples = np.random.lognormal(log_mean, log_std, n_simulations)
    else:
        samples = np.zeros(n_simulations)
    
    return {
        "point_estimate_usd": mean,
        "distribution": samples,
        "p50": np.percentile(samples, 50),
        "p5": np.percentile(samples, 5),
        "p95": np.percentile(samples, 95),
        "mean": np.mean(samples),
        "std": np.std(samples)
    }

def compute_net_value(p: Params) -> Dict:
    """Compute net value: Benefits - (Cost_after - Cost_before). Pure function."""
    cost_before_val = total_cost_before(p)
    cost_after_val = total_cost_after(p)
    benefits_val = total_benefits(p)
    cost_delta = cost_after_val - cost_before_val
    net_value = benefits_val - cost_delta
    
    return {
        "benefits_usd": benefits_val,
        "cost_before_usd": cost_before_val,
        "cost_after_usd": cost_after_val,
        "cost_delta_usd": cost_delta,
        "net_value_usd": net_value,
        "point_estimate_usd": net_value
    }

def get_params_from_session() -> Params:
    """Get Params from session_state, initializing with defaults if needed."""
    if 'params' not in st.session_state:
        st.session_state.params = Params()
    return st.session_state.params

def update_params_in_session(p: Params):
    """Update Params in session_state and mark for recomputation."""
    st.session_state.params = p
    st.session_state['last_update_time'] = pd.Timestamp.now().strftime('%H:%M:%S')
    # Clear cached results to force recomputation
    if 'cost_before_result' in st.session_state:
        del st.session_state['cost_before_result']
    if 'cost_after_result' in st.session_state:
        del st.session_state['cost_after_result']
    if 'benefits_result' in st.session_state:
        del st.session_state['benefits_result']
    if 'net_value_result' in st.session_state:
        del st.session_state['net_value_result']
    if 'mc_cost_before' in st.session_state:
        del st.session_state['mc_cost_before']
    if 'mc_cost_after' in st.session_state:
        del st.session_state['mc_cost_after']

def get_or_compute_cost_before(p: Params) -> Dict:
    """Get cached result or compute cost before."""
    cache_key = 'cost_before_result'
    if cache_key not in st.session_state:
        st.session_state[cache_key] = compute_cost_before(p)
    return st.session_state[cache_key]

def get_or_compute_cost_after(p: Params) -> Dict:
    """Get cached result or compute cost after."""
    cache_key = 'cost_after_result'
    if cache_key not in st.session_state:
        st.session_state[cache_key] = compute_cost_after(p)
    return st.session_state[cache_key]

def get_or_compute_benefits(p: Params) -> Dict:
    """Get cached result or compute benefits."""
    cache_key = 'benefits_result'
    if cache_key not in st.session_state:
        st.session_state[cache_key] = compute_benefits(p)
    return st.session_state[cache_key]

def get_or_compute_net_value(p: Params) -> Dict:
    """Get cached result or compute net value."""
    cache_key = 'net_value_result'
    if cache_key not in st.session_state:
        st.session_state[cache_key] = compute_net_value(p)
    return st.session_state[cache_key]

st.set_page_config(
    page_title="BMW AI Cost & Benefit Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# ============================================================================
# BMW-STYLE GLOBAL CSS - ENHANCED PROFESSIONAL DESIGN
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Reset & Base Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Main Background - Enhanced gradient */
    .main {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 30%, #fafbfc 100%);
        padding: 0;
    }
    
    /* Streamlit Container Improvements */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3.5rem;
        max-width: 1400px;
    }
    
    /* Section Headers - Modern Design */
    h2, h3, h4 {
        color: #0a1929;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h2 {
        font-size: 2rem;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        font-size: 1.5rem;
        color: #1e293b;
    }
    
    h4 {
        font-size: 1.25rem;
        color: #334155;
    }
    
    .help-callout {
        background: linear-gradient(135deg, rgba(0,102,204,0.08) 0%, rgba(255,255,255,0.9) 100%);
        border-left: 4px solid #0066cc;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 1rem 0 1.5rem 0;
        color: #0a1929;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    
    .help-callout strong {
        color: #0066cc;
    }
    
    .help-icon {
        font-size: 1.2rem;
        margin-right: 0.35rem;
    }
    
    /* Header Container - Premium Design */
    .header-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #ffffff 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 2.5rem;
        border: 1px solid rgba(0,102,204,0.08);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #F66733 0%, #0066cc 100%);
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    .logo-text {
        font-size: 1.9rem;
        font-weight: 800;
        color: #1a1d29;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
    
    .clemson-color {
        color: #F66733;
        font-weight: 700;
    }
    
    .bmw-color {
        color: #0066cc;
        font-weight: 700;
    }
    
    .mvp-badge {
        display: inline-block;
        background: linear-gradient(135deg, #F66733 0%, #e55a20 50%, #0066cc 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 24px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-left: 1rem;
        box-shadow: 0 4px 12px rgba(246,103,51,0.25), 0 2px 4px rgba(0,102,204,0.2);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0a1929;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    
    .subheader {
        font-size: 1.05rem;
        color: #5a6c7d;
        margin-bottom: 2rem;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* KPI Cards - Enhanced Design */
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%);
        padding: 1.75rem;
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,102,204,0.08);
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #0066cc 0%, #F66733 100%);
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.1), 0 4px 8px rgba(0,0,0,0.06);
        border-color: rgba(0,102,204,0.15);
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0a1929;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    
    /* Metric Cards - Streamlit Native */
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.03);
        border: 1px solid rgba(0,102,204,0.08);
        border-left: 4px solid #0066cc;
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.08), 0 2px 6px rgba(0,0,0,0.04);
    }
    
    .stMetric label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0a1929;
        letter-spacing: -0.02em;
    }
    
    /* Navigation Buttons - Modern Design */
    .stButton>button {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        color: #0a1929;
        border-radius: 10px;
        font-weight: 600;
        padding: 1rem 1.25rem;
        border: 1.5px solid rgba(0,102,204,0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        min-height: 56px;
        white-space: normal;
        word-wrap: break-word;
        line-height: 1.3;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,102,204,0.15), 0 2px 6px rgba(0,0,0,0.06);
        border-color: rgba(0,102,204,0.3);
    }
    
    /* Expander Styling - Enhanced */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border: 1px solid rgba(0,102,204,0.1);
        font-weight: 600;
        color: #0a1929;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border-color: rgba(0,102,204,0.2);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .streamlit-expanderContent {
        background: #ffffff;
        border-radius: 0 0 10px 10px;
        padding: 1.5rem;
        border: 1px solid rgba(0,102,204,0.1);
        border-top: none;
        margin-top: -1px;
    }
    
    /* Number Input Styling - Enhanced */
    .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1.5px solid rgba(0,102,204,0.15);
        padding: 0.75rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stNumberInput>div>div>input:focus {
        border-color: #0066cc;
        box-shadow: 0 0 0 3px rgba(0,102,204,0.1);
        outline: none;
    }
    
    /* File Uploader Styling */
    .stFileUploader>div {
        border-radius: 10px;
        border: 2px dashed rgba(0,102,204,0.2);
        padding: 1.5rem;
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        transition: all 0.3s ease;
    }
    
    .stFileUploader>div:hover {
        border-color: rgba(0,102,204,0.4);
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    }
    
    /* Download Button Styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,102,204,0.25);
    }
    
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #0052a3 0%, #0066cc 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,102,204,0.35);
    }
    
    /* Chart Containers - Enhanced */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        background: white;
        padding: 1rem;
    }
    
    /* Divider Lines - Enhanced */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, rgba(0,102,204,0.2) 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Success/Error Messages - Enhanced */
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #0066cc 0%, #0080ff 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 12px rgba(0,102,204,0.25), 0 2px 4px rgba(0,102,204,0.15);
        font-weight: 700;
    }
    
    .stButton>button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0080ff 0%, #0066cc 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,102,204,0.3), 0 4px 8px rgba(0,102,204,0.2);
    }
    
    .stButton>button[kind="secondary"] {
        background: white;
        color: #0066cc;
        border: 1.5px solid #0066cc;
        font-weight: 600;
    }
    
    .stButton>button[kind="secondary"]:hover {
        background: #f0f7ff;
        border-color: #0052a3;
        color: #0052a3;
    }
    
    /* Typography Improvements */
    h1, h2, h3, h4, h5, h6 {
        color: #0a1929;
        font-weight: 700;
        letter-spacing: -0.01em;
        line-height: 1.3;
    }
    
    h1 {
        font-size: 2.25rem;
        font-weight: 800;
    }
    
    h2 {
        font-size: 1.75rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Sidebar Enhancements */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid rgba(0,0,0,0.06);
    }
    
    section[data-testid="stSidebar"] .element-container {
        padding: 0.5rem 0;
    }
    
    /* Slider Improvements */
    .stSlider {
        margin-bottom: 1.25rem;
    }
    
    .stSlider label {
        font-weight: 600;
        color: #0a1929;
        font-size: 0.9rem;
    }
    
    /* Number Input Improvements */
    .stNumberInput label {
        font-weight: 600;
        color: #0a1929;
        font-size: 0.9rem;
    }
    
    /* Selectbox Improvements */
    .stSelectbox label {
        font-weight: 600;
        color: #0a1929;
        font-size: 0.9rem;
    }
    
    /* Radio Button Improvements */
    .stRadio label {
        font-weight: 600;
        color: #0a1929;
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    /* Info/Error/Success Boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    /* Expander Improvements */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #0a1929;
    }
    
    /* Chart Container Background */
    [data-testid="stPlotlyChart"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    /* Markdown Text Improvements */
    p {
        line-height: 1.7;
        color: #334155;
    }
    
    /* Code Blocks */
    code {
        background: #f1f5f9;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.9em;
        color: #0066cc;
    }
    
    /* Horizontal Rules */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(0,102,204,0.2) 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Table Improvements */
    table {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Focus States for Accessibility */
    button:focus-visible,
    input:focus-visible,
    select:focus-visible {
        outline: 2px solid #0066cc;
        outline-offset: 2px;
    }
    
    /* Smooth Scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: #0066cc transparent transparent transparent;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PARAMETER RANGES (from new cost model)
# ============================================================================
ranges = {
    # Labor / Labeling - Before AI
    "w_before": (22.0, 30.0, 40.0),                      # $/hour
    "tau_before": (0.003, 0.005, 0.012),                 # hours per label
    "phi_before": (1.2, 1.3, 1.5),                       # overhead multiplier
    
    # Labor / Labeling - After AI
    "w_after": (22.0, 30.0, 40.0),                      # $/hour
    "tau_after": (0.003, 0.005, 0.012),                 # hours per label
    "phi_after": (1.2, 1.3, 1.5),                       # overhead multiplier
    
    # Data Ops (annualized) - Before AI
    "cTB_yr_before": (216.0, 276.0, 360.0),              # $/TB-year storage
    "alpha_yr_before": (300.0, 480.0, 720.0),            # $/TB-year processing/ETL
    "beta_ops_yr_before": (4800.0, 8400.0, 14400.0),      # $/model-year MLOps
    
    # Data Ops (annualized) - After AI
    "cTB_yr_after": (216.0, 276.0, 360.0),              # $/TB-year storage
    "alpha_yr_after": (300.0, 480.0, 720.0),            # $/TB-year processing/ETL
    "beta_ops_yr_after": (4800.0, 8400.0, 14400.0),      # $/model-year MLOps
    
    # Legacy single values (for backward compatibility)
    "w": (22.0, 30.0, 40.0),                      # $/hour
    "tau": (0.003, 0.005, 0.012),                 # hours per label
    "phi": (1.2, 1.3, 1.5),                       # overhead multiplier
    "cTB_yr": (216.0, 276.0, 360.0),              # $/TB-year storage
    "alpha_yr": (300.0, 480.0, 720.0),            # $/TB-year processing/ETL
    "beta_ops_yr": (4800.0, 8400.0, 14400.0),      # $/model-year MLOps
    
    # Operations - Before AI
    "n_labels_before": (0, 0, 1000),  # Typically 0 before AI
    "size_tb_before": (0.0, 0.0, 1.0),  # Typically 0 before AI
    "n_models_before": (0, 0, 1),  # Typically 0 before AI
    
    # Operations - After AI
    "n_labels_after": (5000, 12000, 20000),
    "size_tb_after": (2.0, 5.0, 10.0),
    "n_models_after": (1, 3, 6),
    
    # Risk (annualized)
    "eta": (np.log(2)/140000.0, np.log(2)/100000.0, np.log(2)/60000.0),
    "L_breach_yr": (180000.0, 280000.0, 480000.0),
    
    # Material
    "scrap_rate_before": (0.0, 0.002, 0.005),     # baseline scrap rate
    "scrap_rate_after": (0.0005, 0.0015, 0.003),   # scrap rate after AI (improved, should be 20-40% of benefits)
    "material_cost_per_unit": (250.0, 300.0, 350.0),
    "units_per_year": (80000, 100000, 120000),
    
    # Production / Downtime
    "operating_hours_year": (3000.0, 4000.0, 5000.0),
    "cm_per_unit": (150.0, 250.0, 350.0),         # BMW-scale contribution margin
    "downtime_hours_avoided": (50.0, 75.0, 150.0),  # Reduced for low-downtime environments (5-20% of benefits)
    "oee_improvement_rate": (0.02, 0.03, 0.045),   # 2%-4.5% performance efficiency (should be 40-70% of benefits)
    
    # Capital - Before AI
    "capex_before": (0.0, 0.0, 50000.0),  # Typically 0 or minimal existing infrastructure
    
    # Capital - After AI
    "capex_after": (300000.0, 360000.0, 420000.0),
    "useful_life_years": (6.0, 7.0, 8.0),
}

# Parameter name to key mapping (Excel parameter names -> dual_input keys)
PARAM_KEY_MAPPING = {
    # Before AI parameters
    'n_labels_before': 'n_labels_before',
    'size_tb_before': 'size_tb_before',
    'n_models_before': 'n_models_before',
    'capex_before': 'capex_before',
    # After AI parameters
    'n_labels_after': 'n_labels_after',
    'n_labels': 'n_labels_after',  # Legacy support
    'size_tb_after': 'size_tb_after',
    'size_tb': 'size_tb_after',  # Legacy support
    'n_models_after': 'n_models_after',
    'n_models': 'n_models_after',  # Legacy support
    'capex_after': 'capex_after',
    'capex': 'capex_after',  # Legacy support
    # Cost coefficients - Before AI
    'w_before': 'w_before',
    'tau_before': 'tau_before',
    'phi_before': 'phi_before',
    'cTB_yr_before': 'cTB_yr_before',
    'alpha_yr_before': 'alpha_yr_before',
    'beta_ops_yr_before': 'beta_ops_yr_before',
    # Cost coefficients - After AI
    'w_after': 'w_after',
    'tau_after': 'tau_after',
    'phi_after': 'phi_after',
    'cTB_yr_after': 'cTB_yr_after',
    'alpha_yr_after': 'alpha_yr_after',
    'beta_ops_yr_after': 'beta_ops_yr_after',
    # Legacy single values (for backward compatibility)
    'w': 'w',
    'tau': 'tau',
    'phi': 'phi',
    'cTB_yr': 'cTB_yr',
    'alpha_yr': 'alpha_yr',
    'beta_ops_yr': 'beta_ops_yr',
    'security_before': 'sec_before',
    'security_after': 'sec_after',
    'eta': 'eta',
    'L_breach_yr': 'L_breach',
    'units_per_year': 'units',
    'material_cost_per_unit': 'mat_cost',
    'scrap_rate_before': 'scrap_before',
    'scrap_rate_after': 'scrap_after',
    'operating_hours_year': 'op_hours',
    'cm_per_unit': 'cm',
    'downtime_hours_avoided': 'downtime',
    'oee_improvement_rate': 'oee_rate',
    'capex_before': 'capex_before',
    'capex_after': 'capex_after',
    'capex': 'capex_after',  # Legacy support
    'useful_life_years': 'life'
}

# Parameter explanations
PARAM_EXPLANATIONS = {
    "w": "Average labor wage per hour for data labeling specialists. Higher wages reflect skilled workforce requirements.",
    "w_before": "Average labor wage per hour for data labeling specialists before AI implementation.",
    "w_after": "Average labor wage per hour for data labeling specialists after AI implementation.",
    "tau": "Time required to label one data point in hours. Lower values indicate more efficient labeling processes.",
    "tau_before": "Time required to label one data point in hours before AI implementation.",
    "tau_after": "Time required to label one data point in hours after AI implementation.",
    "phi": "Overhead multiplier accounting for management, infrastructure, and indirect costs beyond direct labor.",
    "phi_before": "Overhead multiplier before AI implementation.",
    "phi_after": "Overhead multiplier after AI implementation.",
    "cTB_yr": "Annual storage cost per terabyte. Includes cloud storage, backup, and data retention infrastructure.",
    "cTB_yr_before": "Annual storage cost per terabyte before AI implementation.",
    "cTB_yr_after": "Annual storage cost per terabyte after AI implementation.",
    "alpha_yr": "Annual ETL and data processing cost per TB. Covers data pipeline operations and transformation.",
    "alpha_yr_before": "Annual ETL and data processing cost per TB before AI implementation.",
    "alpha_yr_after": "Annual ETL and data processing cost per TB after AI implementation.",
    "beta_ops_yr": "Annual MLOps cost per model. Includes deployment, monitoring, retraining, and model management.",
    "beta_ops_yr_before": "Annual MLOps cost per model before AI implementation (typically 0 or minimal).",
    "beta_ops_yr_after": "Annual MLOps cost per model after AI implementation.",
    "eta": "Security effectiveness parameter. Higher values mean security spending reduces breach risk more effectively.",
    "L_breach_yr": "Expected annual financial loss if a data breach occurs. Includes fines, recovery, and reputation costs.",
    "scrap_rate_before": "Baseline scrap rate (fraction of units) before AI implementation. Represents current manufacturing quality.",
    "scrap_rate_after": "Scrap rate after AI implementation. Typically lower due to improved quality control and defect detection.",
    "material_cost_per_unit": "Cost of materials per manufactured unit. Used to calculate scrap cost impact.",
    "units_per_year": "Annual production volume. Affects both material costs and benefit scaling.",
    "operating_hours_year": "Annual operating hours for production. Used to calculate production throughput and downtime costs.",
    "cm_per_unit": "Contribution margin per unit ($). Revenue minus variable costs per unit. Used in downtime cost calculation.",
    "downtime_hours_avoided": "Annual hours of unplanned downtime avoided due to AI predictive maintenance. Reduces production losses. Typical range: 50-150 hrs/year for low downtime environments.",
    "oee_improvement_rate": "Overall Equipment Effectiveness improvement rate (as fraction, e.g., 0.01 = 1%). Typical range: 0.2%-2%. Measures throughput and quality gains excluding downtime.",
    "n_labels_before": "Number of data labels per year before AI implementation. Typically 0 or minimal for baseline operations.",
    "n_labels_after": "Number of data labels per year after AI implementation. Required for AI model training and validation.",
    "size_tb_before": "Dataset size in terabytes before AI implementation. Typically 0 or minimal for baseline operations.",
    "size_tb_after": "Dataset size in terabytes after AI implementation. Includes all training and validation data.",
    "n_models_before": "Number of AI models maintained before AI implementation. Typically 0 for baseline operations.",
    "n_models_after": "Number of AI models maintained after AI implementation. Each model requires MLOps infrastructure.",
    "capex_before": "Capital expenditure before AI implementation. Typically 0 or minimal existing infrastructure.",
    "capex_after": "Total capital expenditure for AI infrastructure after implementation. Includes hardware, software licenses, and implementation.",
    "useful_life_years": "Asset depreciation period. Determines annual capital cost allocation.",
}

def triangular(a, m, b, size=None):
    return np.random.triangular(a, m, b, size=size)

def get_ranges_for_sampling():
    """Get ranges dictionary for sampling - uses current parameter values as middle values when available."""
    # Always try to use current params from session state as baseline
    modified_ranges = {}
    
    # Get current params if available
    current_params = None
    if 'params' in st.session_state:
        params = st.session_state['params']
        # Create a mapping from parameter names to current values
        current_params = {
            'w_before': params.wage_usd_per_hour_before,
            'w_after': params.wage_usd_per_hour_after,
            'tau_before': params.labeling_time_hours_per_label_before,
            'tau_after': params.labeling_time_hours_per_label_after,
            'phi_before': params.overhead_multiplier_before,
            'phi_after': params.overhead_multiplier_after,
            'cTB_yr_before': params.storage_usd_per_tb_year_before,
            'cTB_yr_after': params.storage_usd_per_tb_year_after,
            'alpha_yr_before': params.etl_usd_per_tb_year_before,
            'alpha_yr_after': params.etl_usd_per_tb_year_after,
            'beta_ops_yr_before': params.mlops_usd_per_model_year_before,
            'beta_ops_yr_after': params.mlops_usd_per_model_year_after,
            'n_labels_before': params.labels_per_year_before,
            'n_labels_after': params.labels_per_year_after,
            'size_tb_before': params.dataset_tb_before,
            'size_tb_after': params.dataset_tb_after,
            'n_models_before': params.models_deployed_before,
            'n_models_after': params.models_deployed_after,
            'scrap_rate_before': params.scrap_rate_before,
            'scrap_rate_after': params.scrap_rate_after,
            'security_before': params.security_spend_usd_per_year_before,
            'security_after': params.security_spend_usd_per_year_after,
            'capex_before': params.capex_usd_before,
            'capex_after': params.capex_usd_after,
            'useful_life_years_before': params.useful_life_years_before,
            'useful_life_years_after': params.useful_life_years_after,
            'units_per_year': params.units_per_year,
            'material_cost_per_unit': params.material_cost_usd_per_unit,
            'oee_improvement_rate': params.oee_improvement_fraction,
            'downtime_hours_avoided': params.downtime_hours_avoided_per_year,
            'cm_per_unit': params.contribution_margin_usd_per_unit,
            'L_breach_yr': params.breach_loss_envelope_usd,
            'eta': params.security_effectiveness_per_dollar,
            'operating_hours_year': params.operating_hours_per_year,
            'downtime_cost_per_hour': params.overhead_usd_per_downtime_hour,
            # Legacy keys
            'w': params.wage_usd_per_hour_after,
            'tau': params.labeling_time_hours_per_label_after,
            'phi': params.overhead_multiplier_after,
            'cTB_yr': params.storage_usd_per_tb_year_after,
            'alpha_yr': params.etl_usd_per_tb_year_after,
            'beta_ops_yr': params.mlops_usd_per_model_year_after,
            'n_labels': params.labels_per_year_after,
            'size_tb': params.dataset_tb_after,
            'n_models': params.models_deployed_after,
            'capex': params.capex_usd_after,
            'useful_life_years': params.useful_life_years_after,
        }
    
    # Check for Excel mode
    if st.session_state.get('input_mode') == 'Excel Import':
        # Create modified ranges with Excel values as middle values
        for param_name, (low, mid, high) in ranges.items():
            # Map parameter name to Excel key
            key = PARAM_KEY_MAPPING.get(param_name, param_name)
            excel_key = f"excel_{key}"
            
            # Use Excel value as middle if available, otherwise use current param or original middle
            if excel_key in st.session_state:
                excel_val = float(st.session_state[excel_key])
                spread = (high - low) / 2
                new_low = max(0, excel_val - spread)
                new_high = excel_val + spread
                modified_ranges[param_name] = (new_low, excel_val, new_high)
            elif current_params and param_name in current_params:
                # Use current param value as middle
                current_val = current_params[param_name]
                spread = (high - low) / 2
                new_low = max(0, current_val - spread)
                new_high = current_val + spread
                modified_ranges[param_name] = (new_low, current_val, new_high)
            else:
                modified_ranges[param_name] = (low, mid, high)
        return modified_ranges
    elif current_params:
        # Use current params as middle values
        for param_name, (low, mid, high) in ranges.items():
            if param_name in current_params:
                current_val = current_params[param_name]
                # Calculate spread proportionally
                spread = (high - low) / 2
                new_low = max(0, current_val - spread)
                new_high = current_val + spread
                modified_ranges[param_name] = (new_low, current_val, new_high)
            else:
                modified_ranges[param_name] = (low, mid, high)
        return modified_ranges
    else:
        return ranges

def sample_params(r=None):
    if r is None:
        r = get_ranges_for_sampling()
    return {k: float(triangular(*v)) for k, v in r.items()}

def likely_params(r=None):
    """Get likely parameter values. Uses current Params from session state if available, otherwise uses ranges."""
    # First try to get current params from session state
    if 'params' in st.session_state:
        params = st.session_state['params']
        # Convert Params object to dictionary format compatible with legacy code
        p_dict = params.to_dict()
        # Add legacy keys for backward compatibility
        p_dict['w'] = params.wage_usd_per_hour_after
        p_dict['w_before'] = params.wage_usd_per_hour_before
        p_dict['w_after'] = params.wage_usd_per_hour_after
        p_dict['tau'] = params.labeling_time_hours_per_label_after
        p_dict['tau_before'] = params.labeling_time_hours_per_label_before
        p_dict['tau_after'] = params.labeling_time_hours_per_label_after
        p_dict['phi'] = params.overhead_multiplier_after
        p_dict['phi_before'] = params.overhead_multiplier_before
        p_dict['phi_after'] = params.overhead_multiplier_after
        p_dict['cTB_yr'] = params.storage_usd_per_tb_year_after
        p_dict['cTB_yr_before'] = params.storage_usd_per_tb_year_before
        p_dict['cTB_yr_after'] = params.storage_usd_per_tb_year_after
        p_dict['alpha_yr'] = params.etl_usd_per_tb_year_after
        p_dict['alpha_yr_before'] = params.etl_usd_per_tb_year_before
        p_dict['alpha_yr_after'] = params.etl_usd_per_tb_year_after
        p_dict['beta_ops_yr'] = params.mlops_usd_per_model_year_after
        p_dict['beta_ops_yr_before'] = params.mlops_usd_per_model_year_before
        p_dict['beta_ops_yr_after'] = params.mlops_usd_per_model_year_after
        p_dict['n_labels'] = params.labels_per_year_after
        p_dict['n_labels_before'] = params.labels_per_year_before
        p_dict['n_labels_after'] = params.labels_per_year_after
        p_dict['size_tb'] = params.dataset_tb_after
        p_dict['size_tb_before'] = params.dataset_tb_before
        p_dict['size_tb_after'] = params.dataset_tb_after
        p_dict['n_models'] = params.models_deployed_after
        p_dict['n_models_before'] = params.models_deployed_before
        p_dict['n_models_after'] = params.models_deployed_after
        p_dict['scrap_rate_before'] = params.scrap_rate_before
        p_dict['scrap_rate_after'] = params.scrap_rate_after
        p_dict['security_before'] = params.security_spend_usd_per_year_before
        p_dict['security_after'] = params.security_spend_usd_per_year_after
        p_dict['capex'] = params.capex_usd_after
        p_dict['capex_before'] = params.capex_usd_before
        p_dict['capex_after'] = params.capex_usd_after
        p_dict['useful_life_years'] = params.useful_life_years_after
        p_dict['useful_life_years_before'] = params.useful_life_years_before
        p_dict['useful_life_years_after'] = params.useful_life_years_after
        p_dict['units_per_year'] = params.units_per_year
        p_dict['material_cost_per_unit'] = params.material_cost_usd_per_unit
        p_dict['oee_improvement_rate'] = params.oee_improvement_fraction
        p_dict['downtime_hours_avoided'] = params.downtime_hours_avoided_per_year
        p_dict['cm_per_unit'] = params.contribution_margin_usd_per_unit
        p_dict['L_breach_yr'] = params.breach_loss_envelope_usd
        p_dict['eta'] = params.security_effectiveness_per_dollar
        p_dict['operating_hours_year'] = params.operating_hours_per_year
        p_dict['downtime_cost_per_hour'] = params.overhead_usd_per_downtime_hour
        return p_dict
    
    # Fallback to ranges if no params in session state
    if r is None:
        r = get_ranges_for_sampling()
    return {k: float(v[1]) for k, v in r.items()}

# ============================================================================
# COST MODEL (updated with new structure)
# ============================================================================
def total_cost_breakdown(n_labels, size_tb, security_spend_yr, n_models, 
                         scrap_rate, p=None, use_before=False):
    """Returns detailed cost breakdown. Handles Executive mode direct calculations.
    
    Args:
        use_before: If True, use _before cost parameters; if False, use _after parameters.
                    Falls back to single values if before/after not available.
    """
    if p is None:
        p = likely_params()
    
    # Select cost coefficients based on use_before flag, with fallback to single values
    w = p.get(f'w_{"before" if use_before else "after"}', p.get('w', 30.0))
    tau = p.get(f'tau_{"before" if use_before else "after"}', p.get('tau', 0.005))
    phi = p.get(f'phi_{"before" if use_before else "after"}', p.get('phi', 1.3))
    cTB_yr = p.get(f'cTB_yr_{"before" if use_before else "after"}', p.get('cTB_yr', 276.0))
    alpha_yr = p.get(f'alpha_yr_{"before" if use_before else "after"}', p.get('alpha_yr', 480.0))
    beta_ops_yr = p.get(f'beta_ops_yr_{"before" if use_before else "after"}', p.get('beta_ops_yr', 8400.0))
    capex = p.get(f'capex_{"before" if use_before else "after"}', p.get('capex', 0.0))
    
    # Labeling cost
    # Executive mode: use direct annual_labeling_cost if available
    if '_exec_annual_labeling_cost' in p:
        C_label = p['_exec_annual_labeling_cost']
    else:
        C_label = w * tau * n_labels * phi
    
    C_store = cTB_yr * size_tb
    C_ops = alpha_yr * size_tb + beta_ops_yr * n_models
    
    # Risk cost
    # Executive/Analyst mode: use direct calculation if available
    if '_exec_expected_breach_loss' in p and '_exec_breach_likelihood_reduction' in p:
        # For Executive mode, compute risk directly
        # p_before = some baseline, p_after = p_before * (1 - reduction)
        # For simplicity, use a baseline probability
        baseline_prob = 0.5  # Default baseline breach probability
        p_after = baseline_prob * (1 - p['_exec_breach_likelihood_reduction'])
        L_risk = p['_exec_expected_breach_loss'] * p_after
    elif p.get('eta', 0) == 0:
        # Analyst mode with eta=0, use direct calculation
        baseline_prob = 0.5
        if '_exec_breach_likelihood_reduction' in p:
            p_after = baseline_prob * (1 - p['_exec_breach_likelihood_reduction'])
            L_risk = p.get('L_breach_yr', 280000) * p_after
        else:
            L_risk = p.get('L_breach_yr', 280000) * 0.3  # Default
    else:
        # Engineering mode: use exponential formula
        p_breach = np.exp(-p['eta'] * security_spend_yr)
        L_risk = p_breach * p['L_breach_yr']
    
    C_material = scrap_rate * p['units_per_year'] * p['material_cost_per_unit']
    
    # Capital cost
    # Executive mode: use direct annual_capital_cost if available
    if '_exec_annual_capital_cost' in p:
        C_capital = p['_exec_annual_capital_cost']
    else:
        C_capital = capex / p.get('useful_life_years', 7.0)
    
    total = C_label + C_store + C_ops + L_risk + C_material + C_capital
    
    return {
        "labeling": C_label,
        "storage": C_store,
        "operations": C_ops,
        "risk": L_risk,
        "material": C_material,
        "capital": C_capital,
        "total": total
    }

def total_cost(n_labels, size_tb, security_spend_yr, n_models, scrap_rate, p=None):
    breakdown = total_cost_breakdown(n_labels, size_tb, security_spend_yr, n_models, scrap_rate, p)
    return breakdown["total"]

# ============================================================================
# DOWNTIME COST MODEL
# ============================================================================
def downtime_cost_per_hour(p, crew_size=15, restart_scrap_pct=0.02):
    """Calculate cost per hour of downtime."""
    units_per_hour = p['units_per_year'] / p['operating_hours_year']
    margin_loss = units_per_hour * p['cm_per_unit']
    # Handle before/after parameters: use 'after' values if available, otherwise fall back to single values
    w = p.get('w_after', p.get('w', 30.0))
    phi = p.get('phi_after', p.get('phi', 1.3))
    idle_labor = crew_size * w * phi
    restart_scrap = units_per_hour * p['material_cost_per_unit'] * restart_scrap_pct
    overhead = 500.0
    return margin_loss + idle_labor + restart_scrap + overhead

# ============================================================================
# CES PRODUCTION OPTIMIZATION FUNCTIONS
# ============================================================================
def ces_production(L, K, A, alpha, rho):
    """CES production: q = A[αL^(-ρ) + (1-α)K^(-ρ)]^(-1/ρ)"""
    L_safe = max(L, 1e-6)
    K_safe = max(K, 1e-6)
    return A * (alpha * L_safe**(-rho) + (1 - alpha) * K_safe**(-rho))**(-1/rho)

def marginal_products(L, K, A, alpha, rho):
    """
    Analytical marginal products for CES.
    
    MPL = ∂q/∂L = α·A^ρ·q^(1+ρ)·L^(-1-ρ)
    MPK = ∂q/∂K = (1-α)·A^ρ·q^(1+ρ)·K^(-1-ρ)
    """
    L_safe = max(L, 1e-6)
    K_safe = max(K, 1e-6)
    q = ces_production(L_safe, K_safe, A, alpha, rho)
    
    MPL = alpha * (A**rho) * (q**(1 + rho)) * (L_safe**(-1 - rho))
    MPK = (1 - alpha) * (A**rho) * (q**(1 + rho)) * (K_safe**(-1 - rho))
    
    return MPL, MPK

def profit_function(L, K, p, w, phi, r, c_m, A, alpha, rho, C_fixed):
    """Compute profit: π = p·q − w·φ·L − r·K − c_m·q − C_fixed"""
    q = ces_production(L, K, A, alpha, rho)
    revenue = p * q
    labor_cost = w * phi * L
    capital_cost = r * K
    material_cost = c_m * q
    return revenue - labor_cost - capital_cost - material_cost - C_fixed

def lagrangian_focs(vars, A, alpha, rho, w, phi, r, p, c_m, q_min):
    """
    Lagrangian first-order conditions.
    
    Variables: [L, K, λ]
    
    FOCs:
        1. ∂L/∂L = (p - c_m)·MPL - wφ + λ·MPL = 0
        2. ∂L/∂K = (p - c_m)·MPK - r + λ·MPK = 0
        3. ∂L/∂λ = q(L,K) - q_min = 0
    """
    L, K, lam = vars
    
    # Enforce positivity
    L = max(L, 1e-6)
    K = max(K, 1e-6)
    
    q = ces_production(L, K, A, alpha, rho)
    MPL, MPK = marginal_products(L, K, A, alpha, rho)
    
    # Net marginal revenue (p - material cost)
    p_net = p - c_m
    
    # FOC for labor
    foc_L = p_net * MPL - w * phi + lam * MPL
    
    # FOC for capital
    foc_K = p_net * MPK - r + lam * MPK
    
    # Production constraint (equality if binding)
    foc_lambda = q - q_min
    
    return [foc_L, foc_K, foc_lambda]

def compute_ces_optimum(A, alpha, rho, w, phi, r, p, c_m, q_min, C_fixed, tol=1e-6):
    """
    Solve constrained optimization using Lagrangian method.
    Returns: (L_opt, K_opt, q_opt, profit_opt, lambda_opt, success, residuals)
    """
    def negative_profit(x):
        """For scipy.minimize"""
        return -profit_function(x[0], x[1], p, w, phi, r, c_m, A, alpha, rho, C_fixed)
    
    # Step 1: Unconstrained optimum
    try:
        result_unconstrained = minimize(
            negative_profit,
            x0=[10000, 2000000],
            bounds=[(100, 100000), (10000, 20000000)],
            method='L-BFGS-B',
            options={'ftol': 1e-12}
        )
        
        L_uncon, K_uncon = result_unconstrained.x
        q_uncon = ces_production(L_uncon, K_uncon, A, alpha, rho)
        
        # Step 2: Lagrangian solution (constraint may bind)
        if q_uncon >= q_min:
            initial_guess = [L_uncon, K_uncon, 0.0]  # Start with λ=0
        else:
            initial_guess = [15000, 3000000, 10.0]   # Start with λ>0
        
        # Wrap lagrangian_focs for fsolve
        def lagrangian_wrapper(vars):
            return lagrangian_focs(vars, A, alpha, rho, w, phi, r, p, c_m, q_min)
        
        solution = fsolve(
            lagrangian_wrapper,
            x0=initial_guess,
            full_output=True,
            xtol=1e-10
        )
        
        L_opt, K_opt, lambda_opt = solution[0]
        info = solution[1]
        
        # Validate solution
        q_opt = ces_production(L_opt, K_opt, A, alpha, rho)
        profit_opt = profit_function(L_opt, K_opt, p, w, phi, r, c_m, A, alpha, rho, C_fixed)
        residuals_list = lagrangian_wrapper([L_opt, K_opt, lambda_opt])
        max_residual = max(abs(r) for r in residuals_list)
        
        # Check if constraint binds
        constraint_binds = abs(q_opt - q_min) < 1.0  # Within 1 unit
        
        # Compute FOC residuals for validation
        MPL_opt, MPK_opt = marginal_products(L_opt, K_opt, A, alpha, rho)
        p_net = p - c_m
        
        labor_condition_lhs = p_net * MPL_opt - w * phi
        labor_condition_rhs = -lambda_opt * MPL_opt
        labor_error = abs(labor_condition_lhs - labor_condition_rhs)
        
        capital_condition_lhs = p_net * MPK_opt - r
        capital_condition_rhs = -lambda_opt * MPK_opt
        capital_error = abs(capital_condition_lhs - capital_condition_rhs)
        
        residuals = {
            'foc_L': labor_error,
            'foc_K': capital_error,
            'constraint': abs(q_opt - q_min),
            'max_residual': max_residual,
            'constraint_binds': constraint_binds
        }
        
        return (L_opt, K_opt, q_opt, profit_opt, lambda_opt, True, residuals)
        
    except Exception as e:
        # Fallback: Direct constrained optimization
        try:
            L_init = max(1000, (q_min / A) ** (1.0 / (1 - rho)) * 1.5) if rho != 0 else max(1000, np.sqrt(q_min / A) * 1.5)
            K_init = max(10000, (q_min / A) ** (1.0 / (1 - rho)) * 1.5) if rho != 0 else max(10000, np.sqrt(q_min / A) * 1.5)
            
            def constraint(x):
                L, K = x[0], x[1]
                q = ces_production(L, K, A, alpha, rho)
                return q - q_min
            
            result_constrained = minimize(
                negative_profit,
                x0=[L_init, K_init],
                bounds=[(100, 100000), (10000, 20000000)],
                constraints={'type': 'eq', 'fun': constraint},
                method='SLSQP',
                options={'ftol': 1e-12}
            )
            
            L_opt, K_opt = result_constrained.x
            q_opt = ces_production(L_opt, K_opt, A, alpha, rho)
            profit_opt = profit_function(L_opt, K_opt, p, w, phi, r, c_m, A, alpha, rho, C_fixed)
            lambda_opt = None  # Not available
            
            residuals = {'foc_L': 0, 'foc_K': 0, 'constraint': abs(q_opt - q_min), 'error': str(e)}
            return (L_opt, K_opt, q_opt, profit_opt, lambda_opt, True, residuals)
        except Exception as e2:
            return (None, None, None, None, None, False, {'error': str(e2)})

def compute_shadow_price(L, K, A, alpha, rho, p, c_m, w, phi, r, q_min, C_fixed, delta=0.01):
    """Compute shadow price by finite difference approximation."""
    q_base = ces_production(L, K, A, alpha, rho)
    if q_base < q_min:
        return 0.0
    
    # Increase q_min slightly and recompute
    q_min_new = q_min * (1 + delta)
    result_new = compute_ces_optimum(A, alpha, rho, w, phi, r, p, c_m, q_min_new, C_fixed)
    if result_new[5]:  # success
        profit_base = profit_function(L, K, p, w, phi, r, c_m, A, alpha, rho, C_fixed)
        profit_new = result_new[3]
        lambda_val = (profit_base - profit_new) / (q_min_new - q_min)
        return lambda_val
    return 0.0

# ============================================================================
# BENEFIT MODEL
# ============================================================================
def total_benefits_breakdown_legacy(p=None, security_before=100000, security_after=150000):
    """Legacy function: Returns benefit breakdown. Handles Executive mode direct calculations. Expects dict."""
    if p is None:
        p = likely_params()
    
    # Handle both dict and Params object
    if isinstance(p, Params):
        # Convert Params to dict-like access
        p_dict = p.to_dict()
        p_dict['scrap_rate_before'] = p.scrap_rate_before
        p_dict['scrap_rate_after'] = p.scrap_rate_after
        p_dict['units_per_year'] = p.units_per_year
        p_dict['material_cost_per_unit'] = p.material_cost_usd_per_unit
        p_dict['downtime_hours_avoided'] = p.downtime_hours_avoided_per_year
        p_dict['oee_improvement_rate'] = p.oee_improvement_fraction
        p_dict['cm_per_unit'] = p.contribution_margin_usd_per_unit
        p_dict['L_breach_yr'] = p.breach_loss_envelope_usd
        p_dict['eta'] = p.security_effectiveness_per_dollar
        p = p_dict
    
    # Scrap reduction benefit
    scrap_reduction = p['scrap_rate_before'] - p['scrap_rate_after']
    B_scrap = scrap_reduction * p['units_per_year'] * p['material_cost_per_unit']
    
    # Downtime avoidance
    # Executive mode: use direct downtime_cost_per_hour if available
    if '_exec_downtime_cost_per_hour' in p:
        dt_cost_per_hr = p['_exec_downtime_cost_per_hour']
    else:
        dt_cost_per_hr = downtime_cost_per_hour(p)
    B_downtime = p['downtime_hours_avoided'] * dt_cost_per_hr
    
    # OEE improvement (excludes unplanned downtime to avoid double counting)
    B_OEE = p['oee_improvement_rate'] * p['units_per_year'] * p['cm_per_unit']
    
    # Risk reduction (cyber)
    # Executive/Analyst mode: use direct calculation if available
    if '_exec_breach_likelihood_reduction' in p and '_exec_expected_breach_loss' in p:
        # Direct calculation: B_risk = expected_loss * reduction_pct
        B_risk = p['_exec_expected_breach_loss'] * p['_exec_breach_likelihood_reduction']
    elif '_exec_breach_likelihood_reduction' in p:
        # Analyst mode with direct reduction
        expected_loss = p.get('L_breach_yr', 280000)
        B_risk = expected_loss * p['_exec_breach_likelihood_reduction']
    else:
        # Engineering mode: use exponential formula
        before_risk = p['L_breach_yr'] * np.exp(-p['eta'] * security_before)
        after_risk = p['L_breach_yr'] * np.exp(-p['eta'] * security_after)
        B_risk = before_risk - after_risk
    
    total = B_scrap + B_downtime + B_OEE + B_risk
    
    return {
        "scrap_reduction": B_scrap,
        "downtime_avoidance": B_downtime,
        "downtime_cost_per_hour": dt_cost_per_hr,
        "downtime_hours_avoided": p['downtime_hours_avoided'],
        "OEE_improvement": B_OEE,
        "risk_reduction": B_risk,
        "total": total
    }

def total_benefits_legacy(p=None, security_before=100000, security_after=150000):
    """Legacy function: expects dict. Use total_benefits(Params) for new code."""
    breakdown = total_benefits_breakdown_legacy(p, security_before, security_after)
    return breakdown["total"]

# ============================================================================
# HELPER FUNCTION: Dual Input (Slider + Number Input)
# ============================================================================
def dual_input(label, min_val, max_val, default_val, step=None, key=None, format_str=None, help_text=None):
    """Create both slider and number input side by side with help tooltip. Supports Excel Import mode."""
    # Convert all values to float for consistency
    min_val = float(min_val)
    max_val = float(max_val)
    default_val = float(default_val)
    
    # Convert step to float if provided, otherwise None
    if step is not None:
        step = float(step)
    
    # Check if we're in Excel Import mode and have Excel value
    # Also check if Excel values are loaded (even if mode was switched)
    use_excel = False
    excel_val = None
    excel_key = f"excel_{key}"
    
    # Check for Excel value if in Excel Import mode OR if Excel values are loaded
    if (st.session_state.get('input_mode') == 'Excel Import' or 
        st.session_state.get('excel_loaded', False)):
        if excel_key in st.session_state:
            try:
                excel_val = float(st.session_state[excel_key])
                use_excel = True
            except (ValueError, TypeError):
                pass
    
    # Initialize session state for this parameter
    if f"val_{key}" not in st.session_state:
        st.session_state[f"val_{key}"] = excel_val if use_excel else default_val
    
    # Get current value - prefer Excel if available
    if use_excel and excel_val is not None:
        current_val = excel_val
        # Update session state with Excel value
        st.session_state[f"val_{key}"] = current_val
    else:
        current_val = float(st.session_state[f"val_{key}"])
    
    # Ensure it's within bounds
    current_val = max(min_val, min(max_val, current_val))
    
    col1, col2 = st.columns([2, 1])
    
    # Create slider with help text (disabled in Excel mode)
    with col1:
        slider_val = st.slider(
            label, 
            min_value=min_val, 
            max_value=max_val, 
            value=current_val, 
            step=step, 
            key=f"slider_{key}",
            help=help_text if help_text else None,
            disabled=use_excel  # Disable slider when using Excel values
        )
    
    # Create number input - use slider value to keep them in sync (disabled in Excel mode)
    with col2:
        if format_str:
            num_val = st.number_input(
                "", 
                min_value=min_val, 
                max_value=max_val, 
                value=float(slider_val), 
                step=step, 
                key=f"num_{key}", 
                format=format_str, 
                label_visibility="collapsed",
                disabled=use_excel  # Disable number input when using Excel values
            )
        else:
            num_val = st.number_input(
                "", 
                min_value=min_val, 
                max_value=max_val, 
                value=float(slider_val), 
                step=step, 
                key=f"num_{key}", 
                label_visibility="collapsed",
                disabled=use_excel  # Disable number input when using Excel values
            )
    
    # Determine which value to use
    if use_excel and excel_val is not None:
        final_val = excel_val
    else:
        final_val = float(num_val)
    
    # Clamp to valid range
    final_val = max(min_val, min(max_val, final_val))
    
    # Update session state
    st.session_state[f"val_{key}"] = final_val
    
    return final_val

# ============================================================================
# EXCEL IMPORT FUNCTIONALITY
# ============================================================================
def create_template_excel():
    """Create a template Excel file with all parameters matching the new Params structure."""
    # Get all parameter names and their default values
    template_data = {
        'Parameter': [],
        'Value': [],
        'Description': []
    }
    
    # Create a default Params instance to get default values
    default_params = Params()
    
    # Add all parameters with their defaults - organized by category
    # PAIRED COST PARAMETERS (Before & After)
    defaults = {
        # Paired parameters
        'wage_usd_per_hour_before': default_params.wage_usd_per_hour_before,
        'wage_usd_per_hour_after': default_params.wage_usd_per_hour_after,
        'overhead_multiplier_before': default_params.overhead_multiplier_before,
        'overhead_multiplier_after': default_params.overhead_multiplier_after,
        'scrap_rate_before': default_params.scrap_rate_before,
        'scrap_rate_after': default_params.scrap_rate_after,
        'security_spend_usd_per_year_before': default_params.security_spend_usd_per_year_before,
        'security_spend_usd_per_year_after': default_params.security_spend_usd_per_year_after,
        
        # BEFORE-ONLY COSTS
        'capex_usd_before': default_params.capex_usd_before,
        'useful_life_years_before': default_params.useful_life_years_before,
        'labeling_time_hours_per_label_before': default_params.labeling_time_hours_per_label_before,
        'labels_per_year_before': default_params.labels_per_year_before,
        'dataset_tb_before': default_params.dataset_tb_before,
        'storage_usd_per_tb_year_before': default_params.storage_usd_per_tb_year_before,
        'etl_usd_per_tb_year_before': default_params.etl_usd_per_tb_year_before,
        'mlops_usd_per_model_year_before': default_params.mlops_usd_per_model_year_before,
        'models_deployed_before': default_params.models_deployed_before,
        
        # AFTER-ONLY COSTS (AI operations)
        'labeling_time_hours_per_label_after': default_params.labeling_time_hours_per_label_after,
        'labels_per_year_after': default_params.labels_per_year_after,
        'dataset_tb_after': default_params.dataset_tb_after,
        'storage_usd_per_tb_year_after': default_params.storage_usd_per_tb_year_after,
        'etl_usd_per_tb_year_after': default_params.etl_usd_per_tb_year_after,
        'mlops_usd_per_model_year_after': default_params.mlops_usd_per_model_year_after,
        'models_deployed_after': default_params.models_deployed_after,
        'capex_usd_after': default_params.capex_usd_after,
        'useful_life_years_after': default_params.useful_life_years_after,
        
        # BENEFIT PARAMETERS
        'oee_improvement_fraction': default_params.oee_improvement_fraction,
        'downtime_hours_avoided_per_year': default_params.downtime_hours_avoided_per_year,
        'overhead_usd_per_downtime_hour': default_params.overhead_usd_per_downtime_hour,
        'restart_scrap_fraction': default_params.restart_scrap_fraction,
        'contribution_margin_usd_per_unit': default_params.contribution_margin_usd_per_unit,
        
        # SHARED CONTEXT
        'units_per_year': default_params.units_per_year,
        'operating_hours_per_year': default_params.operating_hours_per_year,
        'material_cost_usd_per_unit': default_params.material_cost_usd_per_unit,
        'breach_loss_envelope_usd': default_params.breach_loss_envelope_usd,
        'security_effectiveness_per_dollar': default_params.security_effectiveness_per_dollar,
    }
    
    # Descriptions for each parameter
    descriptions = {
        'wage_usd_per_hour_before': 'Hourly labor wage rate before AI ($/hr)',
        'wage_usd_per_hour_after': 'Hourly labor wage rate after AI ($/hr)',
        'overhead_multiplier_before': 'Labor overhead multiplier before AI (benefits, facilities, etc.)',
        'overhead_multiplier_after': 'Labor overhead multiplier after AI',
        'scrap_rate_before': 'Fraction of units scrapped before AI',
        'scrap_rate_after': 'Fraction of units scrapped after AI (improved)',
        'security_spend_usd_per_year_before': 'Annual cybersecurity spending before AI ($/year)',
        'security_spend_usd_per_year_after': 'Annual cybersecurity spending after AI ($/year)',
        'capex_usd_before': 'Capital expenditure before AI ($)',
        'useful_life_years_before': 'Useful life of capital assets before AI (years)',
        'labeling_time_hours_per_label_before': 'Time to label one data point before AI (hr, typically 0)',
        'labels_per_year_before': 'Number of data labels created per year before AI (typically 0)',
        'dataset_tb_before': 'Total dataset size in terabytes before AI (typically 0)',
        'storage_usd_per_tb_year_before': 'Annual storage cost per terabyte before AI ($/TB-year)',
        'etl_usd_per_tb_year_before': 'Annual ETL/processing cost per terabyte before AI ($/TB-year)',
        'mlops_usd_per_model_year_before': 'Annual MLOps cost per model before AI ($/model-year, typically 0)',
        'models_deployed_before': 'Number of ML models in production before AI (typically 0)',
        'labeling_time_hours_per_label_after': 'Time to label one data point after AI (hr)',
        'labels_per_year_after': 'Number of data labels created per year after AI',
        'dataset_tb_after': 'Total dataset size in terabytes after AI',
        'storage_usd_per_tb_year_after': 'Annual storage cost per terabyte after AI ($/TB-year)',
        'etl_usd_per_tb_year_after': 'Annual ETL/processing cost per terabyte after AI ($/TB-year)',
        'mlops_usd_per_model_year_after': 'Annual MLOps cost per model after AI ($/model-year)',
        'models_deployed_after': 'Number of ML models in production after AI',
        'capex_usd_after': 'Capital expenditure for AI implementation ($)',
        'useful_life_years_after': 'Useful life of AI capital assets (years)',
        'oee_improvement_fraction': 'Fractional improvement in effective output/quality (e.g., 0.03 = 3%)',
        'downtime_hours_avoided_per_year': 'Unplanned downtime hours avoided per year due to AI (hr/year)',
        'overhead_usd_per_downtime_hour': 'Cost per hour of unplanned downtime ($/hr)',
        'restart_scrap_fraction': 'Fraction of units scrapped during restart after downtime',
        'contribution_margin_usd_per_unit': 'Contribution margin (price - variable cost) per unit ($/unit)',
        'units_per_year': 'Total units produced per year',
        'operating_hours_per_year': 'Total operating hours per year',
        'material_cost_usd_per_unit': 'Material cost per unit produced ($/unit)',
        'breach_loss_envelope_usd': 'Annualized envelope of breach loss if breach occurs (USD)',
        'security_effectiveness_per_dollar': 'Security effectiveness per $; higher η means spend reduces risk faster. Default ≈ ln(2)/100000',
    }
    
    for param, default_val in defaults.items():
        template_data['Parameter'].append(param)
        template_data['Value'].append(default_val)
        template_data['Description'].append(descriptions.get(param, ''))
    
    df = pd.DataFrame(template_data)
    return df

def parse_excel_file(uploaded_file):
    """Parse Excel file and return dictionary of parameters."""
    try:
        # Try reading as Excel (try openpyxl first, then xlrd for .xls)
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        except:
            df = pd.read_excel(uploaded_file, engine='xlrd')
        
        # Check if it's two-column format (Parameter, Value)
        if 'Parameter' in df.columns and 'Value' in df.columns:
            param_dict = dict(zip(df['Parameter'], df['Value']))
        # Check if it's single-row format (parameters as headers)
        elif len(df) == 1:
            param_dict = df.iloc[0].to_dict()
        else:
            # Try first two columns
            if len(df.columns) >= 2:
                param_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
            else:
                raise ValueError("Excel format not recognized. Expected 'Parameter' and 'Value' columns or single row with parameter names.")
        
        return param_dict, None
    except Exception as e:
        return None, str(e)

def validate_excel_params(param_dict):
    """Validate that all required parameters are present. Accepts new Params field names, legacy keys, or grouped keys."""
    # New parameter names (from Params dataclass) - minimum required
    new_required = [
        'wage_usd_per_hour_before', 'wage_usd_per_hour_after',
        'overhead_multiplier_before', 'overhead_multiplier_after',
        'scrap_rate_before', 'scrap_rate_after',
        'security_spend_usd_per_year_before', 'security_spend_usd_per_year_after',
        'capex_usd_before', 'useful_life_years_before',
        'labeling_time_hours_per_label_after', 'labels_per_year_after',
        'dataset_tb_after', 'storage_usd_per_tb_year_after',
        'etl_usd_per_tb_year_after', 'mlops_usd_per_model_year_after',
        'models_deployed_after', 'capex_usd_after', 'useful_life_years_after',
        'oee_improvement_fraction', 'downtime_hours_avoided_per_year',
        'overhead_usd_per_downtime_hour', 'contribution_margin_usd_per_unit',
        'units_per_year', 'operating_hours_per_year',
        'material_cost_usd_per_unit', 'breach_loss_envelope_usd',
        'security_effectiveness_per_dollar'
    ]
    
    # Legacy parameter names (for backward compatibility)
    legacy_required = [
        'n_labels', 'size_tb', 'n_models', 'w', 'tau', 'phi', 'cTB_yr', 'alpha_yr', 'beta_ops_yr',
        'security_before', 'security_after', 'eta', 'L_breach_yr',
        'units_per_year', 'material_cost_per_unit', 'scrap_rate_before', 'scrap_rate_after',
        'operating_hours_year', 'cm_per_unit', 'downtime_hours_avoided', 'oee_improvement_rate',
        'capex', 'useful_life_years'
    ]
    
    # Simple mode grouped keys
    simple_required = [
        'total_ai_costs', 'size_tb', 'units_per_year', 'material_cost_per_unit', 'cm_per_unit',
        'delta_scrap_rate', 'downtime_hours_avoided', 'downtime_cost_per_hour',
        'oee_improvement_rate', 'expected_breach_loss_if_occurs', 'breach_likelihood_reduction_pct'
    ]
    
    # Check which format we have
    has_new = any(k in param_dict for k in ['wage_usd_per_hour_before', 'wage_usd_per_hour_after', 
                                            'labeling_time_hours_per_label_after', 'oee_improvement_fraction'])
    has_grouped = any(k in param_dict for k in ['total_ai_costs', 'annual_labeling_cost', 'data_ops_cost_per_TB_year'])
    has_legacy = any(k in param_dict for k in ['w', 'tau', 'cTB_yr'])
    
    if has_new:
        # Validate new parameter names - be lenient, only check critical ones
        critical_new = ['units_per_year', 'material_cost_usd_per_unit', 'scrap_rate_before', 'scrap_rate_after']
        missing = [p for p in critical_new if p not in param_dict]
    elif has_grouped:
        # Validate grouped keys
        missing = [p for p in simple_required if p not in param_dict]
    elif has_legacy:
        # Validate legacy keys
        missing = [p for p in legacy_required if p not in param_dict]
    else:
        # Neither set present, check for minimum required
        missing = ['units_per_year']  # At minimum need this
    
    invalid = []
    
    # Check for invalid values (non-numeric or negative where not allowed)
    for param, value in param_dict.items():
        try:
            val = float(value)
            if val < 0 and param not in ['scrap_rate_before', 'scrap_rate_after', 'delta_scrap_rate']:
                invalid.append(f"{param}: negative value not allowed")
        except (ValueError, TypeError):
            invalid.append(f"{param}: non-numeric value")
    
    return missing, invalid

# ============================================================================
# HELPER: Build Engine Parameters from UI Values
# ============================================================================
def build_engine_params_from_ui(detail_level, ui_values, excel_overrides=None):
    """
    Build parameter dict for engine from UI values based on detail level.
    Returns dict with legacy keys (w, tau, phi, cTB_yr, etc.) that engine expects.
    """
    if excel_overrides is None:
        excel_overrides = {}
    
    p = {}
    
    # Get value helper (Excel first, then UI)
    def get_val(key, default):
        excel_key = f"excel_{key}"
        
        # Priority 1: Check excel_overrides dict (most direct, passed in)
        if excel_overrides and excel_key in excel_overrides:
            try:
                return float(excel_overrides[excel_key])
            except (ValueError, TypeError):
                pass
        
        # Priority 2: Check session state if in Excel Import mode OR if Excel values exist
        # This ensures Excel values are used even if mode was temporarily switched
        if st.session_state.get('input_mode') == 'Excel Import' or st.session_state.get('excel_loaded', False):
            if excel_key in st.session_state:
                try:
                    return float(st.session_state[excel_key])
                except (ValueError, TypeError):
                    pass
        
        # Priority 3: Fall back to UI values
        return ui_values.get(key, default)
    
    if detail_level == "Simple":
        # Simple mode: derive from grouped inputs
        # First try to get grouped parameter, or calculate from legacy parameters if in Excel
        total_ai_costs = get_val('total_ai_costs', None)
        
        # If total_ai_costs not found, try to calculate from legacy parameters
        if total_ai_costs is None or (st.session_state.get('input_mode') == 'Excel Import' and f"excel_total_ai_costs" not in st.session_state):
            # Try to get legacy parameters and calculate total_ai_costs
            w_val = get_val('w', None)
            tau_val = get_val('tau', None)
            phi_val = get_val('phi', None)
            n_labels_val = get_val('n_labels', None)
            cTB_yr_val = get_val('cTB_yr', None)
            alpha_yr_val = get_val('alpha_yr', None)
            beta_ops_yr_val = get_val('beta_ops_yr', None)
            n_models_val = get_val('n_models', None)
            capex_val = get_val('capex', None)
            useful_life_val = get_val('life', None)
            size_tb_for_calc = get_val('size_tb', 5.0)
            
            # Calculate from legacy parameters if available
            # Default values (fallback if legacy params not available)
            default_annual_labeling_cost = 2340.0  # Typical labeling cost
            default_data_ops_total = 3780.0  # Typical storage + ETL cost
            default_annual_mlops_cost = 25200.0  # Typical MLOps cost
            default_annual_capital_cost = 51429.0  # Typical annualized CAPEX
            default_total_ai_costs = 86769.0  # Sum of defaults
            
            if all(v is not None for v in [w_val, tau_val, phi_val, n_labels_val]):
                annual_labeling_cost_calc = w_val * tau_val * n_labels_val * phi_val
            else:
                annual_labeling_cost_calc = default_annual_labeling_cost
            
            if all(v is not None for v in [cTB_yr_val, alpha_yr_val]):
                data_ops_total_calc = (cTB_yr_val + alpha_yr_val) * size_tb_for_calc
            else:
                data_ops_total_calc = default_data_ops_total
            
            if all(v is not None for v in [beta_ops_yr_val, n_models_val]):
                annual_mlops_cost_calc = beta_ops_yr_val * n_models_val
            else:
                annual_mlops_cost_calc = default_annual_mlops_cost
            
            if all(v is not None for v in [capex_val, useful_life_val]):
                annual_capital_cost_calc = capex_val / useful_life_val
            else:
                annual_capital_cost_calc = default_annual_capital_cost
            
            total_ai_costs = annual_labeling_cost_calc + data_ops_total_calc + annual_mlops_cost_calc + annual_capital_cost_calc
        
        # If still None, use default
        if total_ai_costs is None:
            default_total_ai_costs = 86769.0  # Sum of component defaults
            total_ai_costs = ui_values.get('total_ai_costs', default_total_ai_costs)
        
        size_tb = get_val('size_tb', ui_values.get('size_tb', 5.0))
        
        # Split total AI costs into components (using default proportions)
        # Default proportions: labeling ~25%, data ops ~20%, MLOps ~35%, capital ~20%
        annual_labeling_cost = total_ai_costs * 0.25
        data_ops_total = total_ai_costs * 0.20  # This is for the dataset size
        annual_mlops_cost = total_ai_costs * 0.35
        annual_capital_cost = total_ai_costs * 0.20
        
        # But if we have legacy parameters from Excel, use those instead
        if st.session_state.get('input_mode') == 'Excel Import':
            w_val = get_val('w', None)
            tau_val = get_val('tau', None)
            phi_val = get_val('phi', None)
            n_labels_val = get_val('n_labels', None)
            cTB_yr_val = get_val('cTB_yr', None)
            alpha_yr_val = get_val('alpha_yr', None)
            beta_ops_yr_val = get_val('beta_ops_yr', None)
            n_models_val = get_val('n_models', None)
            capex_val = get_val('capex', None)
            useful_life_val = get_val('life', None)
            
            if all(v is not None for v in [w_val, tau_val, phi_val, n_labels_val]):
                annual_labeling_cost = w_val * tau_val * n_labels_val * phi_val
            
            if all(v is not None for v in [cTB_yr_val, alpha_yr_val]):
                data_ops_total = (cTB_yr_val + alpha_yr_val) * size_tb
            
            if all(v is not None for v in [beta_ops_yr_val, n_models_val]):
                annual_mlops_cost = beta_ops_yr_val * n_models_val
            
            if all(v is not None for v in [capex_val, useful_life_val]):
                annual_capital_cost = capex_val / useful_life_val
        delta_scrap_rate = get_val('delta_scrap_rate', ui_values.get('delta_scrap_rate', 0.0005))
        downtime_cost_per_hour = get_val('downtime_cost_per_hour', ui_values.get('downtime_cost_per_hour', 5000))
        breach_likelihood_reduction_pct = get_val('breach_likelihood_reduction_pct', ui_values.get('breach_likelihood_reduction_pct', 0.40))
        expected_breach_loss_if_occurs = get_val('expected_breach_loss_if_occurs', ui_values.get('expected_breach_loss_if_occurs', 280000))
        
        # Derive legacy parameters
        # If we have legacy parameters from Excel, use those directly; otherwise derive from grouped values
        p['w'] = get_val('w', 30.0)
        p['tau'] = get_val('tau', 0.005)
        p['phi'] = get_val('phi', 1.3)
        p['n_labels'] = get_val('n_labels', None)
        if p['n_labels'] is None:
            # n_labels derived from annual_labeling_cost
            if p['w'] * p['tau'] * p['phi'] > 0:
                p['n_labels'] = annual_labeling_cost / (p['w'] * p['tau'] * p['phi'])
            else:
                p['n_labels'] = 12000
        
        # Data ops: if we have legacy params, use those; otherwise split 40/60
        p['cTB_yr'] = get_val('cTB_yr', None)
        p['alpha_yr'] = get_val('alpha_yr', None)
        if p['cTB_yr'] is None or p['alpha_yr'] is None:
            data_ops_cost_per_TB_year = data_ops_total / size_tb if size_tb > 0 else 756
            p['cTB_yr'] = data_ops_cost_per_TB_year * 0.4
            p['alpha_yr'] = data_ops_cost_per_TB_year * 0.6
        
        # MLOps - Before/After
        p['beta_ops_yr'] = get_val('beta_ops_yr', annual_mlops_cost)
        p['n_models_before'] = get_val('n_models_before', 0)  # Before AI: typically 0
        p['n_models_after'] = get_val('n_models_after', get_val('n_models', 1))  # After AI: from grouped or legacy
        
        # Operations - Before/After (Simple mode: before = 0, after = derived)
        p['n_labels_before'] = get_val('n_labels_before', 0)  # Before AI: typically 0
        p['n_labels_after'] = get_val('n_labels_after', p.get('n_labels', 12000))  # After AI: from grouped or legacy
        
        p['size_tb_before'] = get_val('size_tb_before', 0.0)  # Before AI: typically 0
        p['size_tb_after'] = get_val('size_tb_after', size_tb)  # After AI: from grouped input
        
        # Capital - Before/After
        p['useful_life_years'] = get_val('life', 7.0)
        p['capex_before'] = get_val('capex_before', 0.0)  # Before AI: typically 0
        p['capex_after'] = get_val('capex_after', None)
        if p['capex_after'] is None:
            p['capex_after'] = get_val('capex', annual_capital_cost * p['useful_life_years'])  # After AI: from grouped or legacy
        
        # Scrap
        # For Simple mode: scrap_rate_after is the improved (lower) rate
        # delta_scrap_rate is the reduction (positive = improvement)
        # But if we have legacy scrap rates from Excel, use those
        p['scrap_rate_after'] = get_val('scrap_after', None)
        p['scrap_rate_before'] = get_val('scrap_before', None)
        if p['scrap_rate_after'] is None:
            scrap_rate_after = 0.0015  # Improved rate after AI (should be 20-40% of benefits)
            p['scrap_rate_after'] = scrap_rate_after
        if p['scrap_rate_before'] is None:
            p['scrap_rate_before'] = max(delta_scrap_rate, 0) + p['scrap_rate_after']
        
        # Production (keep visible ones) - get from Excel if available, otherwise UI
        p['operating_hours_year'] = get_val('op_hours', ui_values.get('operating_hours_year', 4000))
        p['cm_per_unit'] = get_val('cm', ui_values.get('cm_per_unit', 250))
        p['downtime_hours_avoided'] = get_val('downtime', ui_values.get('downtime_hours_avoided', 75))
        p['oee_improvement_rate'] = get_val('oee_rate', ui_values.get('oee_improvement_rate', 0.03))
        
        # Material/Production
        p['material_cost_per_unit'] = get_val('mat_cost', ui_values.get('material_cost_per_unit', 300))
        p['units_per_year'] = get_val('units', ui_values.get('units_per_year', 100000))
        
        # Security (derive from risk reduction)
        # For Simple mode, we'll compute risk directly, but need security_before/after for engine
        # Get from Excel if available, otherwise use defaults
        p['security_before'] = get_val('sec_before', ui_values.get('security_before', 100000))
        p['security_after'] = get_val('sec_after', ui_values.get('security_after', 150000))
        p['eta'] = get_val('eta', 0.0)  # Get from Excel if available, otherwise 0 (not used in Simple mode)
        p['L_breach_yr'] = get_val('L_breach', expected_breach_loss_if_occurs)
        
        # Store Simple mode-specific values for cost/benefit calculation
        p['_exec_annual_labeling_cost'] = annual_labeling_cost
        p['_exec_annual_capital_cost'] = annual_capital_cost
        p['_exec_downtime_cost_per_hour'] = downtime_cost_per_hour
        p['_exec_breach_likelihood_reduction'] = breach_likelihood_reduction_pct
        p['_exec_expected_breach_loss'] = expected_breach_loss_if_occurs
        
    else:  # Full Detail mode
        # Full Detail mode: all parameters visible with before/after distinction
        
        # Cost coefficients - Before AI
        p['w_before'] = get_val('w_before', ui_values.get('w_before', None))
        p['tau_before'] = get_val('tau_before', ui_values.get('tau_before', None))
        p['phi_before'] = get_val('phi_before', ui_values.get('phi_before', None))
        p['cTB_yr_before'] = get_val('cTB_yr_before', ui_values.get('cTB_yr_before', None))
        p['alpha_yr_before'] = get_val('alpha_yr_before', ui_values.get('alpha_yr_before', None))
        p['beta_ops_yr_before'] = get_val('beta_ops_yr_before', ui_values.get('beta_ops_yr_before', None))
        
        # Cost coefficients - After AI
        p['w_after'] = get_val('w_after', ui_values.get('w_after', None))
        p['tau_after'] = get_val('tau_after', ui_values.get('tau_after', None))
        p['phi_after'] = get_val('phi_after', ui_values.get('phi_after', None))
        p['cTB_yr_after'] = get_val('cTB_yr_after', ui_values.get('cTB_yr_after', None))
        p['alpha_yr_after'] = get_val('alpha_yr_after', ui_values.get('alpha_yr_after', None))
        p['beta_ops_yr_after'] = get_val('beta_ops_yr_after', ui_values.get('beta_ops_yr_after', None))
        
        # Legacy support: if before/after not set, use legacy single values for both
        if p['w_before'] is None:
            p['w_before'] = get_val('w', ui_values.get('w', 30))
        if p['tau_before'] is None:
            p['tau_before'] = get_val('tau', ui_values.get('tau', 0.005))
        if p['phi_before'] is None:
            p['phi_before'] = get_val('phi', ui_values.get('phi', 1.3))
        if p['cTB_yr_before'] is None:
            p['cTB_yr_before'] = get_val('cTB_yr', ui_values.get('cTB_yr', 276))
        if p['alpha_yr_before'] is None:
            p['alpha_yr_before'] = get_val('alpha_yr', ui_values.get('alpha_yr', 480))
        if p['beta_ops_yr_before'] is None:
            p['beta_ops_yr_before'] = get_val('beta_ops_yr', ui_values.get('beta_ops_yr', 8400))
        
        if p['w_after'] is None:
            p['w_after'] = get_val('w', ui_values.get('w', 30))
        if p['tau_after'] is None:
            p['tau_after'] = get_val('tau', ui_values.get('tau', 0.005))
        if p['phi_after'] is None:
            p['phi_after'] = get_val('phi', ui_values.get('phi', 1.3))
        if p['cTB_yr_after'] is None:
            p['cTB_yr_after'] = get_val('cTB_yr', ui_values.get('cTB_yr', 276))
        if p['alpha_yr_after'] is None:
            p['alpha_yr_after'] = get_val('alpha_yr', ui_values.get('alpha_yr', 480))
        if p['beta_ops_yr_after'] is None:
            p['beta_ops_yr_after'] = get_val('beta_ops_yr', ui_values.get('beta_ops_yr', 8400))
        
        # Set legacy single values for backward compatibility (use 'after' values as defaults)
        p['w'] = p.get('w_after', p.get('w_before', 30.0))
        p['tau'] = p.get('tau_after', p.get('tau_before', 0.005))
        p['phi'] = p.get('phi_after', p.get('phi_before', 1.3))
        p['cTB_yr'] = p.get('cTB_yr_after', p.get('cTB_yr_before', 276.0))
        p['alpha_yr'] = p.get('alpha_yr_after', p.get('alpha_yr_before', 480.0))
        p['beta_ops_yr'] = p.get('beta_ops_yr_after', p.get('beta_ops_yr_before', 8400.0))
        p['capex'] = p.get('capex_after', p.get('capex_before', 0.0))
        
        # Operations - Before/After
        p['n_labels_before'] = get_val('n_labels_before', ui_values.get('n_labels_before', 0))
        p['n_labels_after'] = get_val('n_labels_after', ui_values.get('n_labels_after', 12000))
        p['size_tb_before'] = get_val('size_tb_before', ui_values.get('size_tb_before', 0.0))
        p['size_tb_after'] = get_val('size_tb_after', ui_values.get('size_tb_after', 5.0))
        p['n_models_before'] = get_val('n_models_before', ui_values.get('n_models_before', 0))
        p['n_models_after'] = get_val('n_models_after', ui_values.get('n_models_after', 3))
        
        # Legacy support: if old parameters exist, use them for "after"
        if p['n_labels_after'] == 0 and get_val('n_labels', None) is not None:
            p['n_labels_after'] = get_val('n_labels', 12000)
        if p['size_tb_after'] == 0.0 and get_val('size_tb', None) is not None:
            p['size_tb_after'] = get_val('size_tb', 5.0)
        if p['n_models_after'] == 0 and get_val('n_models', None) is not None:
            p['n_models_after'] = get_val('n_models', 3)
        
        # Capital - Before/After
        p['capex_before'] = get_val('capex_before', ui_values.get('capex_before', 0.0))
        p['capex_after'] = get_val('capex_after', ui_values.get('capex_after', 360000))
        # Legacy support
        if p['capex_after'] == 0.0 and get_val('capex', None) is not None:
            p['capex_after'] = get_val('capex', 360000)
        
        p['useful_life_years'] = get_val('life', ui_values.get('useful_life_years', 7))
        p['scrap_rate_before'] = get_val('scrap_before', ui_values.get('scrap_rate_before', 0.002))
        p['scrap_rate_after'] = get_val('scrap_after', ui_values.get('scrap_rate_after', 0.0015))
        p['operating_hours_year'] = get_val('op_hours', ui_values.get('operating_hours_year', 4000))
        p['cm_per_unit'] = get_val('cm', ui_values.get('cm_per_unit', 250))
        p['downtime_hours_avoided'] = get_val('downtime', ui_values.get('downtime_hours_avoided', 75))
        p['oee_improvement_rate'] = get_val('oee_rate', ui_values.get('oee_improvement_rate', 0.03))
        p['material_cost_per_unit'] = get_val('mat_cost', ui_values.get('material_cost_per_unit', 300))
        p['units_per_year'] = get_val('units', ui_values.get('units_per_year', 100000))
        p['security_before'] = get_val('sec_before', ui_values.get('security_before', 100000))
        p['security_after'] = get_val('sec_after', ui_values.get('security_after', 150000))
        p['eta'] = get_val('eta', ui_values.get('eta', np.log(2)/100000.0))
        p['L_breach_yr'] = get_val('L_breach', ui_values.get('L_breach_yr', 280000))
    
    # Common parameters (always needed)
    p['size_tb'] = get_val('size_tb', ui_values.get('size_tb', 5.0))
    
    return p

# ============================================================================
# NEW PARAMETER UI SYSTEM
# ============================================================================

def render_parameter_ui() -> Params:
    """Render the parameter UI with three collapsible sections: Costs Before, Costs After, Benefits."""
    p = get_params_from_session()
    
    st.markdown("### 🎛️ Model Parameters")
    
    # Toggle Mode: Manual Input vs Excel Import
    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = 'Manual'
    
    mode = st.radio(
        "**Input Mode:**",
        ['Manual', 'Excel Import'],
        index=0 if st.session_state.input_mode == 'Manual' else 1,
        key='input_mode_radio'
    )
    
    # If switching from Excel Import to Manual, clear Excel values and flag
    if st.session_state.input_mode == 'Excel Import' and mode == 'Manual':
        # Clear Excel values
        keys_to_clear = [k for k in st.session_state.keys() if k.startswith('excel_')]
        for key in keys_to_clear:
            del st.session_state[key]
        st.session_state['excel_loaded'] = False
        # Clear cached results
        if 'cost_before_result' in st.session_state:
            del st.session_state['cost_before_result']
        if 'cost_after_result' in st.session_state:
            del st.session_state['cost_after_result']
        if 'benefits_result' in st.session_state:
            del st.session_state['benefits_result']
        if 'net_value_result' in st.session_state:
            del st.session_state['net_value_result']
    
    st.session_state.input_mode = mode
    
    # Excel Import Section
    if mode == 'Excel Import':
        st.markdown("---")
        st.markdown("#### 📥 Excel Import")
        
        uploaded_file = st.file_uploader(
            "Upload Parameter File",
            type=['xlsx', 'xls'],
            help="Upload an Excel file with 'Parameter' and 'Value' columns, or a single row with parameter names as headers.",
            key='excel_uploader'
        )
        
        # Process file if uploaded (only process new files, not on every rerun)
        # Use file uploader's file_id to track new uploads
        if uploaded_file is not None:
            # Get unique file identifier
            try:
                current_file_id = uploaded_file.file_id if hasattr(uploaded_file, 'file_id') else id(uploaded_file)
            except:
                current_file_id = id(uploaded_file)
            
            last_processed_id = st.session_state.get('last_excel_file_id')
            
            # Only process if this is a new file
            if current_file_id != last_processed_id:
                with st.spinner("Loading Excel parameters..."):
                    param_dict, error = parse_excel_file(uploaded_file)
                    
                    if error:
                        st.error(f"❌ Error reading Excel file: {error}")
                    else:
                        missing, invalid = validate_excel_params(param_dict)
                        
                        if missing:
                            st.warning(f"⚠️ Missing parameters: {', '.join(missing)}")
                        if invalid:
                            st.error(f"❌ Invalid values: {', '.join(invalid)}")
                        
                        if not missing and not invalid:
                            # Store Excel values in session state
                            # Handle new parameter names (from Params dataclass) and legacy names for backward compatibility
                            
                            # Mapping from new Params field names to UI keys (for dual_input widgets)
                            new_to_ui_key = {
                                'wage_usd_per_hour_before': 'w_before',
                                'wage_usd_per_hour_after': 'w_after',
                                'overhead_multiplier_before': 'phi_before',
                                'overhead_multiplier_after': 'phi_after',
                                'scrap_rate_before': 'scrap_rate_before',
                                'scrap_rate_after': 'scrap_rate_after',
                                'security_spend_usd_per_year_before': 'security_before',
                                'security_spend_usd_per_year_after': 'security_after',
                                'capex_usd_before': 'capex_before',
                                'useful_life_years_before': 'useful_life_years_before',
                                # Before AI operations parameters
                                'labeling_time_hours_per_label_before': 'tau_before',
                                'labels_per_year_before': 'n_labels_before',
                                'dataset_tb_before': 'size_tb_before',
                                'storage_usd_per_tb_year_before': 'cTB_yr_before',
                                'etl_usd_per_tb_year_before': 'alpha_yr_before',
                                'mlops_usd_per_model_year_before': 'beta_ops_yr_before',
                                'models_deployed_before': 'n_models_before',
                                # After AI operations parameters
                                'labeling_time_hours_per_label_after': 'tau_after',
                                'labels_per_year_after': 'n_labels_after',
                                'dataset_tb_after': 'size_tb_after',
                                'storage_usd_per_tb_year_after': 'cTB_yr_after',
                                'etl_usd_per_tb_year_after': 'alpha_yr_after',
                                'mlops_usd_per_model_year_after': 'beta_ops_yr_after',
                                'models_deployed_after': 'n_models_after',
                                'capex_usd_after': 'capex_after',
                                'useful_life_years_after': 'useful_life_years_after',
                                # Benefit parameters
                                'oee_improvement_fraction': 'oee_rate',  # Maps to UI key 'oee_rate'
                                'downtime_hours_avoided_per_year': 'downtime_hours_avoided',
                                'overhead_usd_per_downtime_hour': 'downtime_cost_per_hour',
                                'restart_scrap_fraction': 'restart_scrap_fraction',
                                'contribution_margin_usd_per_unit': 'cm_per_unit',
                                # Shared context parameters
                                'units_per_year': 'units_per_year',
                                'operating_hours_per_year': 'operating_hours_year',
                                'material_cost_usd_per_unit': 'material_cost_per_unit',
                                'breach_loss_envelope_usd': 'L_breach_yr',
                                'security_effectiveness_per_dollar': 'eta',
                            }
                            
                            # Legacy parameter name mapping (for backward compatibility)
                            legacy_to_new = {
                                'w': 'wage_usd_per_hour_after',
                                'w_before': 'wage_usd_per_hour_before',
                                'w_after': 'wage_usd_per_hour_after',
                                'tau': 'labeling_time_hours_per_label_after',
                                'tau_before': 'labeling_time_hours_per_label_before',
                                'tau_after': 'labeling_time_hours_per_label_after',
                                'phi': 'overhead_multiplier_after',
                                'phi_before': 'overhead_multiplier_before',
                                'phi_after': 'overhead_multiplier_after',
                                'cTB_yr': 'storage_usd_per_tb_year_after',
                                'cTB_yr_before': 'storage_usd_per_tb_year_before',
                                'cTB_yr_after': 'storage_usd_per_tb_year_after',
                                'alpha_yr': 'etl_usd_per_tb_year_after',
                                'alpha_yr_before': 'etl_usd_per_tb_year_before',
                                'alpha_yr_after': 'etl_usd_per_tb_year_after',
                                'beta_ops_yr': 'mlops_usd_per_model_year_after',
                                'beta_ops_yr_before': 'mlops_usd_per_model_year_before',
                                'beta_ops_yr_after': 'mlops_usd_per_model_year_after',
                                'n_labels': 'labels_per_year_after',
                                'n_labels_before': 'labels_per_year_before',
                                'n_labels_after': 'labels_per_year_after',
                                'size_tb': 'dataset_tb_after',
                                'size_tb_before': 'dataset_tb_before',
                                'size_tb_after': 'dataset_tb_after',
                                'n_models': 'models_deployed_after',
                                'n_models_before': 'models_deployed_before',
                                'n_models_after': 'models_deployed_after',
                                'security_before': 'security_spend_usd_per_year_before',
                                'security_after': 'security_spend_usd_per_year_after',
                                'capex': 'capex_usd_after',
                                'capex_after': 'capex_usd_after',
                                'capex_before': 'capex_usd_before',
                                'useful_life_years': 'useful_life_years_after',
                                'useful_life_years_after': 'useful_life_years_after',
                                'useful_life_years_before': 'useful_life_years_before',
                                'oee_improvement_rate': 'oee_improvement_fraction',
                                'downtime_hours_avoided': 'downtime_hours_avoided_per_year',
                                'downtime_cost_per_hour': 'overhead_usd_per_downtime_hour',
                                'cm_per_unit': 'contribution_margin_usd_per_unit',
                                'operating_hours_year': 'operating_hours_per_year',
                                'L_breach_yr': 'breach_loss_envelope_usd',
                                'eta': 'security_effectiveness_per_dollar',
                            }
                            
                            for param_name, value in param_dict.items():
                                try:
                                    # Determine the UI key to use
                                    # First check if it's a new parameter name (from Params)
                                    if param_name in new_to_ui_key:
                                        ui_key = new_to_ui_key[param_name]
                                    # Then check if it's a legacy name that needs mapping
                                    elif param_name in legacy_to_new:
                                        # Map to new name, then to UI key
                                        new_name = legacy_to_new[param_name]
                                        ui_key = new_to_ui_key.get(new_name, param_name)
                                    # Otherwise use PARAM_KEY_MAPPING for backward compatibility
                                    else:
                                        ui_key = PARAM_KEY_MAPPING.get(param_name, param_name)
                                    
                                    # Store in session state with both the UI key and the original param name
                                    st.session_state[f"excel_{ui_key}"] = float(value)
                                    st.session_state[f"excel_{param_name}"] = float(value)  # Also store with original name
                                except (ValueError, TypeError):
                                    # Store as-is if not numeric (for string values)
                                    ui_key = new_to_ui_key.get(param_name, PARAM_KEY_MAPPING.get(param_name, param_name))
                                    st.session_state[f"excel_{ui_key}"] = value
                                    st.session_state[f"excel_{param_name}"] = value
                            
                            # Clear cached Monte Carlo results to force recalculation
                            if 'mc_results_cost' in st.session_state:
                                del st.session_state['mc_results_cost']
                            if 'mc_results_net' in st.session_state:
                                del st.session_state['mc_results_net']
                            
                            # Clear cached computation results
                            if 'cost_before_result' in st.session_state:
                                del st.session_state['cost_before_result']
                            if 'cost_after_result' in st.session_state:
                                del st.session_state['cost_after_result']
                            if 'benefits_result' in st.session_state:
                                del st.session_state['benefits_result']
                            if 'net_value_result' in st.session_state:
                                del st.session_state['net_value_result']
                            
                            # Clear all val_* session state values to force dual_input to use Excel values
                            keys_to_clear = [k for k in st.session_state.keys() if k.startswith('val_')]
                            for key in keys_to_clear:
                                del st.session_state[key]
                            
                            # Set flag to indicate Excel values are loaded
                            st.session_state['excel_loaded'] = True
                            st.session_state['last_excel_file_id'] = current_file_id
                            
                            st.success("✅ Parameters loaded successfully! Updating all calculations...")
                            
                            # Debug: Show what was loaded
                            if st.session_state.get('debug_excel', False):
                                with st.expander("🔍 Debug: Excel Values Loaded"):
                                    excel_debug = {k: v for k, v in st.session_state.items() if k.startswith('excel_')}
                                    st.json(excel_debug)
                            
                            # Show preview
                            with st.expander("📋 Preview Imported Values"):
                                preview_df = pd.DataFrame({
                                    'Parameter': list(param_dict.keys()),
                                    'Value': list(param_dict.values())
                                })
                                st.dataframe(preview_df, use_container_width=True, hide_index=True)
                            
                            # Immediately update Params object with Excel values
                            # This ensures calculations use Excel values even before UI renders
                            params = get_params_from_session()
                            # Create a temporary UI values dict from Excel values
                            temp_ui_vals = {}
                            for k, v in st.session_state.items():
                                if k.startswith('excel_'):
                                    ui_key = k.replace('excel_', '')
                                    temp_ui_vals[ui_key] = v
                            
                            # Map Excel values to Params object
                            def map_excel_to_params(excel_vals: Dict, params: Params) -> Params:
                                """Map Excel values directly to Params object. Handles both new full names and legacy names."""
                                # Helper to safely get and convert value
                                def get_val(key, default=None):
                                    if key in excel_vals:
                                        try:
                                            val = excel_vals[key]
                                            if val is None or val == '':
                                                return default
                                            return float(val) if not isinstance(val, (int, float)) else val
                                        except (ValueError, TypeError):
                                            return default
                                    return default
                                
                                # PAIRED COST PARAMETERS - Check both new full names and legacy names
                                # Wage
                                val = get_val('wage_usd_per_hour_before') or get_val('w_before')
                                if val is not None:
                                    params.wage_usd_per_hour_before = val
                                val = get_val('wage_usd_per_hour_after') or get_val('w_after')
                                if val is not None:
                                    params.wage_usd_per_hour_after = val
                                
                                # Overhead
                                val = get_val('overhead_multiplier_before') or get_val('phi_before')
                                if val is not None:
                                    params.overhead_multiplier_before = val
                                val = get_val('overhead_multiplier_after') or get_val('phi_after')
                                if val is not None:
                                    params.overhead_multiplier_after = val
                                
                                # Scrap rate
                                val = get_val('scrap_rate_before')
                                if val is not None:
                                    params.scrap_rate_before = val
                                val = get_val('scrap_rate_after')
                                if val is not None:
                                    params.scrap_rate_after = val
                                
                                # Security spend
                                val = get_val('security_spend_usd_per_year_before') or get_val('security_before')
                                if val is not None:
                                    params.security_spend_usd_per_year_before = val
                                val = get_val('security_spend_usd_per_year_after') or get_val('security_after')
                                if val is not None:
                                    params.security_spend_usd_per_year_after = val
                                
                                # BEFORE-ONLY COSTS
                                val = get_val('capex_usd_before') or get_val('capex_before')
                                if val is not None:
                                    params.capex_usd_before = val
                                val = get_val('useful_life_years_before')
                                if val is not None:
                                    params.useful_life_years_before = val
                                
                                val = get_val('labeling_time_hours_per_label_before') or get_val('tau_before')
                                if val is not None:
                                    params.labeling_time_hours_per_label_before = val
                                val = get_val('labels_per_year_before') or get_val('n_labels_before')
                                if val is not None:
                                    params.labels_per_year_before = int(val)
                                val = get_val('dataset_tb_before') or get_val('size_tb_before')
                                if val is not None:
                                    params.dataset_tb_before = val
                                val = get_val('storage_usd_per_tb_year_before') or get_val('cTB_yr_before')
                                if val is not None:
                                    params.storage_usd_per_tb_year_before = val
                                val = get_val('etl_usd_per_tb_year_before') or get_val('alpha_yr_before')
                                if val is not None:
                                    params.etl_usd_per_tb_year_before = val
                                val = get_val('mlops_usd_per_model_year_before') or get_val('beta_ops_yr_before')
                                if val is not None:
                                    params.mlops_usd_per_model_year_before = val
                                val = get_val('models_deployed_before') or get_val('n_models_before')
                                if val is not None:
                                    params.models_deployed_before = int(val)
                                
                                # AFTER-ONLY COSTS
                                val = get_val('labeling_time_hours_per_label_after') or get_val('tau_after')
                                if val is not None:
                                    params.labeling_time_hours_per_label_after = val
                                val = get_val('labels_per_year_after') or get_val('n_labels_after') or get_val('n_labels')
                                if val is not None:
                                    params.labels_per_year_after = int(val)
                                val = get_val('dataset_tb_after') or get_val('size_tb_after') or get_val('size_tb')
                                if val is not None:
                                    params.dataset_tb_after = val
                                val = get_val('storage_usd_per_tb_year_after') or get_val('cTB_yr_after')
                                if val is not None:
                                    params.storage_usd_per_tb_year_after = val
                                val = get_val('etl_usd_per_tb_year_after') or get_val('alpha_yr_after')
                                if val is not None:
                                    params.etl_usd_per_tb_year_after = val
                                val = get_val('mlops_usd_per_model_year_after') or get_val('beta_ops_yr_after')
                                if val is not None:
                                    params.mlops_usd_per_model_year_after = val
                                val = get_val('models_deployed_after') or get_val('n_models_after') or get_val('n_models')
                                if val is not None:
                                    params.models_deployed_after = int(val)
                                val = get_val('capex_usd_after') or get_val('capex_after') or get_val('capex')
                                if val is not None:
                                    params.capex_usd_after = val
                                val = get_val('useful_life_years_after')
                                if val is not None:
                                    params.useful_life_years_after = val
                                
                                # BENEFIT PARAMETERS
                                val = get_val('oee_improvement_fraction') or get_val('oee_rate') or get_val('oee_improvement_rate')
                                if val is not None:
                                    params.oee_improvement_fraction = val
                                val = get_val('downtime_hours_avoided_per_year') or get_val('downtime_hours_avoided')
                                if val is not None:
                                    params.downtime_hours_avoided_per_year = val
                                val = get_val('overhead_usd_per_downtime_hour') or get_val('downtime_cost_per_hour')
                                if val is not None:
                                    params.overhead_usd_per_downtime_hour = val
                                val = get_val('restart_scrap_fraction')
                                if val is not None:
                                    params.restart_scrap_fraction = val
                                val = get_val('contribution_margin_usd_per_unit') or get_val('cm_per_unit')
                                if val is not None:
                                    params.contribution_margin_usd_per_unit = val
                                
                                # SHARED CONTEXT
                                val = get_val('units_per_year')
                                if val is not None:
                                    params.units_per_year = val
                                val = get_val('operating_hours_per_year') or get_val('operating_hours_year')
                                if val is not None:
                                    params.operating_hours_per_year = val
                                val = get_val('material_cost_usd_per_unit') or get_val('material_cost_per_unit')
                                if val is not None:
                                    params.material_cost_usd_per_unit = val
                                val = get_val('breach_loss_envelope_usd') or get_val('L_breach_yr')
                                if val is not None:
                                    params.breach_loss_envelope_usd = val
                                val = get_val('security_effectiveness_per_dollar') or get_val('eta')
                                if val is not None:
                                    params.security_effectiveness_per_dollar = val
                                
                                return params
                            
                            # Map Excel values to Params
                            params = map_excel_to_params(temp_ui_vals, params)
                            update_params_in_session(params)
                            
                            # Force rerun immediately - this will refresh all calculations and pages
                            st.rerun()
            
            # Show preview if Excel values are already loaded (even if file uploader is None after rerun)
            if st.session_state.get('excel_loaded', False):
                with st.expander("📋 Preview Imported Values", expanded=False):
                    excel_params = {}
                    for k, v in st.session_state.items():
                        if k.startswith('excel_'):
                            param_name = k.replace('excel_', '')
                            excel_params[param_name] = v
                    if excel_params:
                        preview_df = pd.DataFrame({
                            'Parameter': list(excel_params.keys()),
                            'Value': list(excel_params.values())
                        })
                        st.dataframe(preview_df, use_container_width=True, hide_index=True)
        
        # Export Template Button
        st.markdown("---")
        if st.button("📥 Download Template Excel File"):
            template_df = create_template_excel()
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                template_df.to_excel(writer, index=False, sheet_name='Parameters')
            output.seek(0)
            st.download_button(
                label="⬇️ Download Template",
                data=output,
                file_name="parameter_template.xlsx",
                mime="application/vnd.openpyxl-officedocument.spreadsheetml.sheet"
            )
    
    st.markdown("---")
    
    # Collect all UI values in a dictionary
    ui_values = {}
    
    # ========================================================================
    # SECTION 1: COST BEFORE PARAMETERS (Collapsible)
    # ========================================================================
    with st.expander("💰 **Costs Before**", expanded=True):
        st.caption("Parameters for calculating costs before AI implementation")
        
        ui_values['w_before'] = dual_input(
            "Wage (w, $/hr)", 10.0, 200.0, 30.0, 1.0, "w_before", "%.2f",
            help_text="Hourly labor wage rate before AI"
        )
        ui_values['phi_before'] = dual_input(
            "Overhead multiplier (φ)", 1.0, 3.0, 1.3, 0.01, "phi_before", "%.2f",
            help_text="Labor overhead multiplier (benefits, facilities, etc.)"
        )
        ui_values['scrap_rate_before'] = dual_input(
            "Scrap rate (s_before)", 0.0, 0.1, 0.002, 0.0001, "scrap_before", "%.4f",
            help_text="Fraction of units scrapped before AI"
        )
        ui_values['security_before'] = dual_input(
            "Security spend (S_before, $/year)", 0.0, 1000000.0, 100000.0, 10000.0, "sec_before", "%.2f",
            help_text="Annual cybersecurity spending before AI"
        )
        ui_values['capex_before'] = dual_input(
            "CAPEX before ($)", 0.0, 5000000.0, 50000.0, 10000.0, "capex_before", "%.2f",
            help_text="Capital expenditure before AI"
        )
        ui_values['useful_life_years_before'] = dual_input(
            "Useful life before (Y_before, years)", 1.0, 20.0, 7.0, 0.5, "life_before", "%.2f",
            help_text="Useful life of capital assets before AI"
        )
        ui_values['tau_before'] = dual_input(
            "Time per label (τ, hr)", 0.0, 0.1, 0.001, 0.0001, "tau_before", "%.6f",
            help_text="Time to label one data point before AI (much smaller than after)"
        )
        ui_values['n_labels_before'] = dual_input(
            "Labels per year (n_ℓ)", 0, 1000000, 1000, 100, "n_labels_before", None,
            help_text="Number of data labels created per year before AI (much smaller than after)"
        )
        ui_values['size_tb_before'] = dual_input(
            "Dataset size (TB)", 0.0, 1000.0, 1.0, 0.5, "size_tb_before", "%.2f",
            help_text="Total dataset size in terabytes before AI"
        )
        ui_values['cTB_yr_before'] = dual_input(
            "Storage cost (c_TB, $/TB-year)", 50.0, 1000.0, 276.0, 10.0, "cTB_yr_before", "%.2f",
            help_text="Annual storage cost per terabyte before AI"
        )
        ui_values['alpha_yr_before'] = dual_input(
            "ETL cost (α, $/TB-year)", 100.0, 2000.0, 480.0, 10.0, "alpha_yr_before", "%.2f",
            help_text="Annual ETL/processing cost per terabyte before AI"
        )
        ui_values['beta_ops_yr_before'] = dual_input(
            "MLOps cost (β, $/model-year)", 0.0, 50000.0, 0.0, 100.0, "beta_ops_yr_before", "%.2f",
            help_text="Annual MLOps cost per model before AI (typically 0)"
        )
        ui_values['n_models_before'] = dual_input(
            "Number of models (n_m)", 0, 100, 0, 1, "n_models_before", None,
            help_text="Number of ML models in production before AI (keep at 0)"
        )
        
        st.markdown("#### Shared Production Parameters")
        ui_values['units_per_year'] = dual_input(
            "Units per year", 1000, 10000000, 100000, 1000, "units", None,
            help_text="Total units produced per year (used in cost calculations)"
        )
        ui_values['material_cost_per_unit'] = dual_input(
            "Material cost per unit ($)", 50.0, 2000.0, 300.0, 10.0, "mat_cost", "%.2f",
            help_text="Material cost per unit produced (used in scrap cost calculation)"
        )
    
    # ========================================================================
    # SECTION 2: COST AFTER PARAMETERS (Collapsible)
    # ========================================================================
    with st.expander("💰 **Costs After**", expanded=True):
        st.caption("Parameters for calculating costs after AI implementation")
        
        ui_values['w_after'] = dual_input(
            "Wage (w, $/hr)", 10.0, 200.0, 30.0, 1.0, "w_after", "%.2f",
            help_text="Hourly labor wage rate after AI"
        )
        ui_values['phi_after'] = dual_input(
            "Overhead multiplier (φ)", 1.0, 3.0, 1.3, 0.01, "phi_after", "%.2f",
            help_text="Labor overhead multiplier after AI"
        )
        ui_values['scrap_rate_after'] = dual_input(
            "Scrap rate (s_after)", 0.0, 0.1, 0.0015, 0.0001, "scrap_after", "%.4f",
            help_text="Fraction of units scrapped after AI (improved)"
        )
        ui_values['security_after'] = dual_input(
            "Security spend (S_after, $/year)", 0.0, 1000000.0, 150000.0, 10000.0, "sec_after", "%.2f",
            help_text="Annual cybersecurity spending after AI"
        )
        ui_values['tau_after'] = dual_input(
            "Time per label (τ, hr)", 0.0001, 0.1, 0.005, 0.0001, "tau_after", "%.6f",
            help_text="Time to label one data point after AI"
        )
        ui_values['n_labels_after'] = dual_input(
            "Labels per year (n_ℓ)", 0, 1000000, 12000, 100, "n_labels_after", None,
            help_text="Number of data labels created per year after AI"
        )
        ui_values['size_tb_after'] = dual_input(
            "Dataset size (TB)", 0.0, 1000.0, 5.0, 0.5, "size_tb_after", "%.2f",
            help_text="Total dataset size in terabytes after AI"
        )
        ui_values['cTB_yr_after'] = dual_input(
            "Storage cost (c_TB, $/TB-year)", 50.0, 1000.0, 276.0, 10.0, "cTB_yr_after", "%.2f",
            help_text="Annual storage cost per terabyte"
        )
        ui_values['alpha_yr_after'] = dual_input(
            "ETL cost (α, $/TB-year)", 100.0, 2000.0, 480.0, 10.0, "alpha_yr_after", "%.2f",
            help_text="Annual ETL/processing cost per terabyte"
        )
        ui_values['beta_ops_yr_after'] = dual_input(
            "MLOps cost (β, $/model-year)", 1000.0, 50000.0, 8400.0, 100.0, "beta_ops_yr_after", "%.2f",
            help_text="Annual MLOps cost per model"
        )
        ui_values['n_models_after'] = dual_input(
            "Number of models (n_m)", 0, 100, 3, 1, "n_models_after", None,
            help_text="Number of ML models in production after AI"
        )
        ui_values['capex_after'] = dual_input(
            "CAPEX after (CapEx, $)", 0.0, 5000000.0, 360000.0, 10000.0, "capex_after", "%.2f",
            help_text="Capital expenditure for AI implementation"
        )
        ui_values['useful_life_years_after'] = dual_input(
            "Useful life after (Y, years)", 1.0, 20.0, 7.0, 0.5, "life_after", "%.2f",
            help_text="Useful life of AI capital assets"
        )
    
    # ========================================================================
    # SECTION 3: BENEFIT PARAMETERS (Collapsible)
    # ========================================================================
    with st.expander("✨ **Benefits**", expanded=True):
        st.caption("Parameters for calculating all benefit components")
        
        ui_values['cm_per_unit'] = dual_input(
            "Contribution margin per unit ($)", 50.0, 2000.0, 250.0, 5.0, "cm", "%.2f",
            help_text="Contribution margin (price - variable cost) per unit"
        )
        ui_values['operating_hours_year'] = dual_input(
            "Operating hours per year", 1000.0, 8760.0, 4000.0, 100.0, "op_hours", "%.2f",
            help_text="Total operating hours per year"
        )
        ui_values['oee_improvement_rate'] = dual_input(
            "OEE improvement (ΔOEE)", 0.0, 0.2, 0.03, 0.001, "oee_rate", "%.4f",
            help_text="Fractional improvement in effective output/quality (e.g., 0.03 = 3%)"
        )
        ui_values['downtime_hours_avoided'] = dual_input(
            "Downtime hours avoided (H_dt, hr/year)", 0.0, 2000.0, 75.0, 10.0, "downtime", "%.2f",
            help_text="Unplanned downtime hours avoided per year due to AI"
        )
        ui_values['downtime_cost_per_hour'] = dual_input(
            "Downtime cost per hour (c_oh, $/hr)", 0.0, 50000.0, 5000.0, 100.0, "downtime_cost_per_hour", "%.2f",
            help_text="Cost per hour of unplanned downtime"
        )
        ui_values['L_breach_yr'] = dual_input(
            "Breach loss envelope (L_breach, $)", 50000.0, 2000000.0, 280000.0, 10000.0, "L_breach", "%.2f",
            help_text="Annualized envelope of breach loss if breach occurs (USD)"
        )
        ui_values['eta'] = dual_input(
            "Security effectiveness (η)", 0.0, 0.0001, np.log(2)/100000.0, 0.000001, "eta", "%.8f",
            help_text="Security effectiveness per $; higher η means spend reduces risk faster. Default ≈ ln(2)/100000"
        )
    
    st.markdown("---")
    st.markdown("#### 🎲 Monte Carlo")
    ui_values['mc_runs'] = dual_input(
        "Simulation runs", 0, 1000000, 5000, 500, "mc_runs", None,
        help_text="Number of Monte Carlo simulation iterations. Higher values provide more accurate distributions but take longer."
    )
    ui_values['mc_seed'] = st.number_input(
        "Random seed", 0, 10000, 42,
        help="Random seed for reproducibility. Same seed produces identical results."
    )
    
    mc_runs = ui_values['mc_runs']
    mc_seed = ui_values['mc_seed']
    
    # Store in session state for access outside this function
    st.session_state['mc_runs'] = mc_runs
    st.session_state['mc_seed'] = mc_seed
    
    # Map UI values to Params object
    def map_ui_to_params(ui_vals: Dict, params: Params) -> Params:
        """Map UI values to Params object with new field names."""
        # Paired parameters
        if 'w_before' in ui_vals:
            params.wage_usd_per_hour_before = float(ui_vals['w_before'])
        if 'w_after' in ui_vals:
            params.wage_usd_per_hour_after = float(ui_vals['w_after'])
        if 'phi_before' in ui_vals:
            params.overhead_multiplier_before = float(ui_vals['phi_before'])
        if 'phi_after' in ui_vals:
            params.overhead_multiplier_after = float(ui_vals['phi_after'])
        if 'scrap_rate_before' in ui_vals:
            params.scrap_rate_before = float(ui_vals['scrap_rate_before'])
        if 'scrap_rate_after' in ui_vals:
            params.scrap_rate_after = float(ui_vals['scrap_rate_after'])
        if 'security_before' in ui_vals:
            params.security_spend_usd_per_year_before = float(ui_vals['security_before'])
        if 'security_after' in ui_vals:
            params.security_spend_usd_per_year_after = float(ui_vals['security_after'])
        
        # Before-only parameters
        if 'capex_before' in ui_vals:
            params.capex_usd_before = float(ui_vals['capex_before'])
        if 'useful_life_years_before' in ui_vals:
            params.useful_life_years_before = float(ui_vals['useful_life_years_before'])
        elif 'useful_life_years' in ui_vals:
            # Fallback: use same value for both if only one provided
            params.useful_life_years_before = float(ui_vals['useful_life_years'])
        if 'tau_before' in ui_vals:
            params.labeling_time_hours_per_label_before = float(ui_vals['tau_before'])
        if 'n_labels_before' in ui_vals:
            params.labels_per_year_before = int(ui_vals['n_labels_before'])
        if 'size_tb_before' in ui_vals:
            params.dataset_tb_before = float(ui_vals['size_tb_before'])
        if 'cTB_yr_before' in ui_vals:
            params.storage_usd_per_tb_year_before = float(ui_vals['cTB_yr_before'])
        if 'alpha_yr_before' in ui_vals:
            params.etl_usd_per_tb_year_before = float(ui_vals['alpha_yr_before'])
        if 'beta_ops_yr_before' in ui_vals:
            params.mlops_usd_per_model_year_before = float(ui_vals['beta_ops_yr_before'])
        if 'n_models_before' in ui_vals:
            params.models_deployed_before = int(ui_vals['n_models_before'])
        
        # After-only parameters
        if 'tau_after' in ui_vals or 'tau' in ui_vals:
            params.labeling_time_hours_per_label_after = float(ui_vals.get('tau_after', ui_vals.get('tau', 0.005)))
        if 'n_labels_after' in ui_vals:
            params.labels_per_year_after = int(ui_vals['n_labels_after'])
        if 'size_tb_after' in ui_vals:
            params.dataset_tb_after = float(ui_vals['size_tb_after'])
        if 'cTB_yr_after' in ui_vals:
            params.storage_usd_per_tb_year_after = float(ui_vals['cTB_yr_after'])
        if 'alpha_yr_after' in ui_vals:
            params.etl_usd_per_tb_year_after = float(ui_vals['alpha_yr_after'])
        if 'beta_ops_yr_after' in ui_vals:
            params.mlops_usd_per_model_year_after = float(ui_vals['beta_ops_yr_after'])
        if 'n_models_after' in ui_vals:
            params.models_deployed_after = int(ui_vals['n_models_after'])
        if 'capex_after' in ui_vals:
            params.capex_usd_after = float(ui_vals['capex_after'])
        if 'useful_life_years_after' in ui_vals:
            params.useful_life_years_after = float(ui_vals['useful_life_years_after'])
        elif 'useful_life_years' in ui_vals:
            # Fallback: use same value for both if only one provided
            params.useful_life_years_after = float(ui_vals['useful_life_years'])
        
        # Benefit parameters
        if 'units_per_year' in ui_vals:
            params.units_per_year = float(ui_vals['units_per_year'])
        if 'material_cost_per_unit' in ui_vals:
            params.material_cost_usd_per_unit = float(ui_vals['material_cost_per_unit'])
        if 'oee_improvement_rate' in ui_vals:
            params.oee_improvement_fraction = float(ui_vals['oee_improvement_rate'])
        if 'downtime_hours_avoided' in ui_vals:
            params.downtime_hours_avoided_per_year = float(ui_vals['downtime_hours_avoided'])
        if 'operating_hours_year' in ui_vals:
            params.operating_hours_per_year = float(ui_vals['operating_hours_year'])
        if 'cm_per_unit' in ui_vals:
            params.contribution_margin_usd_per_unit = float(ui_vals['cm_per_unit'])
        if 'downtime_cost_per_hour' in ui_vals:
            params.overhead_usd_per_downtime_hour = float(ui_vals['downtime_cost_per_hour'])
        if 'restart_scrap_fraction' in ui_vals:
            params.restart_scrap_fraction = float(ui_vals['restart_scrap_fraction'])
        
        # Security/risk parameters
        if 'eta' in ui_vals:
            params.security_effectiveness_per_dollar = float(ui_vals['eta'])
        if 'L_breach_yr' in ui_vals:
            params.breach_loss_envelope_usd = float(ui_vals['L_breach_yr'])
        
        return params
    
    p = map_ui_to_params(ui_values, p)
    
    # Update session state
    update_params_in_session(p)
    
    # Show update indicator
    if 'last_update_time' in st.session_state:
        st.caption(f"✓ Updated {st.session_state['last_update_time']}")
    
    return p

# ============================================================================
# SIDEBAR: PARAMETER UI REMOVED (not affecting calculations)
# ============================================================================
# Parameters are now only used for the AI Operational Benefits chart on Net Value page

# ============================================================================
# PAGE NAVIGATION
# ============================================================================
# Header with Clemson X BMW Logo
st.markdown("""
<div class="header-container">
    <div class="logo-container">
        <span class="logo-text">
            <span class="clemson-color">CLEMSON</span> 
            <span style="color: #666;">×</span> 
            <span class="bmw-color">BMW</span>
        </span>
        <span class="mvp-badge">MVP • Final Version</span>
    </div>
    <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.95rem;">
        AI Optimization in Manufacturing • Cost & Benefit Analysis Dashboard
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown('<p class="main-header">🚗 BMW AI Implementation Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Module-level Automotive Manufacturing • Cost & Benefit Model</p>', unsafe_allow_html=True)

# Navigation buttons with enhanced professional styling
st.markdown("""
<div style="background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%); padding: 1.25rem; border-radius: 14px; box-shadow: 0 4px 16px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04); margin: 2rem 0; border: 1px solid rgba(0,102,204,0.08);">
<style>
    /* Ensure all navigation buttons have uniform size */
    button[kind="secondary"], button[kind="primary"] {
        min-height: 64px !important;
        height: auto !important;
        font-size: 15px !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        padding: 12px 16px !important;
        line-height: 1.4 !important;
    }
</style>
""", unsafe_allow_html=True)

col_nav1, col_nav2, col_nav3 = st.columns(3)

# Initialize page state if not exists
if 'page' not in st.session_state:
    st.session_state.page = 4

with col_nav1:
    page4 = st.button("💹 Net Value", use_container_width=True, type="primary" if st.session_state.get('page') == 4 else "secondary", key="nav_page4")
    if page4:
        st.session_state.page = 4
        st.rerun()

with col_nav2:
    page3 = st.button("🔒 Breach Risk", use_container_width=True, type="primary" if st.session_state.get('page') == 3 else "secondary", key="nav_page3")
    if page3:
        st.session_state.page = 3
        st.rerun()

with col_nav3:
    page2 = st.button("📖 Parameters", use_container_width=True, type="primary" if st.session_state.get('page') == 2 else "secondary", key="nav_page2")
    if page2:
        st.session_state.page = 2
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# Get current page (already initialized above in navigation section)
current_page = st.session_state.get('page', 4)

st.markdown("---")

# ============================================================================
# COMPUTE MODEL RESULTS
# ============================================================================
# Always rebuild p_current to ensure we have the latest values (especially Excel values)
# This ensures Excel values are always used when available
# IMPORTANT: Always check session state for Excel values, even if file uploader is None
# Excel values persist in session state and should be used whenever in Excel Import mode
excel_overrides_fresh = {}
excel_count = 0

# Always collect Excel values if they exist in session state (regardless of current mode)
# Get params from new system
params = get_params_from_session()

# For backward compatibility, create p_current from params
# This allows existing pages to work during transition
p_current = params.to_dict()
# Add legacy keys for compatibility
p_current['w'] = params.wage_usd_per_hour_after
p_current['w_before'] = params.wage_usd_per_hour_before
p_current['w_after'] = params.wage_usd_per_hour_after
p_current['tau'] = params.labeling_time_hours_per_label_after
p_current['tau_after'] = params.labeling_time_hours_per_label_after
p_current['phi'] = params.overhead_multiplier_after
p_current['phi_before'] = params.overhead_multiplier_before
p_current['phi_after'] = params.overhead_multiplier_after
p_current['cTB_yr'] = params.storage_usd_per_tb_year_after
p_current['cTB_yr_after'] = params.storage_usd_per_tb_year_after
p_current['alpha_yr'] = params.etl_usd_per_tb_year_after
p_current['alpha_yr_after'] = params.etl_usd_per_tb_year_after
p_current['beta_ops_yr'] = params.mlops_usd_per_model_year_after
p_current['beta_ops_yr_after'] = params.mlops_usd_per_model_year_after
p_current['n_labels'] = params.labels_per_year_after
p_current['n_labels_after'] = params.labels_per_year_after
p_current['size_tb'] = params.dataset_tb_after
p_current['size_tb_after'] = params.dataset_tb_after
p_current['n_models'] = params.models_deployed_after
p_current['n_models_after'] = params.models_deployed_after
p_current['models_count_after'] = params.models_deployed_after
p_current['security_before'] = params.security_spend_usd_per_year_before
p_current['security_after'] = params.security_spend_usd_per_year_after
p_current['scrap_rate_before'] = params.scrap_rate_before
p_current['scrap_rate_after'] = params.scrap_rate_after
p_current['units_per_year'] = params.units_per_year
p_current['material_cost_per_unit'] = params.material_cost_usd_per_unit
p_current['cm_per_unit'] = params.contribution_margin_usd_per_unit
p_current['operating_hours_year'] = params.operating_hours_per_year
p_current['oee_improvement_rate'] = params.oee_improvement_fraction
p_current['downtime_hours_avoided'] = params.downtime_hours_avoided_per_year
p_current['capex'] = params.capex_usd_after
p_current['capex_after'] = params.capex_usd_after
p_current['capex_before'] = params.capex_usd_before
p_current['useful_life_years'] = params.useful_life_years_after
p_current['useful_life_years_after'] = params.useful_life_years_after
p_current['useful_life_years_before'] = params.useful_life_years_before
p_current['eta'] = params.security_effectiveness_per_dollar
p_current['L_breach_yr'] = params.breach_loss_envelope_usd
p_current['annualized_capex_usd_after'] = annualized_capex(params.capex_usd_after, params.useful_life_years_after)

# Debug: Show if Excel values are being used (only in debug mode)
if st.session_state.get('debug_excel', False) and st.session_state.get('input_mode') == 'Excel Import':
    st.info(f"🔍 Debug: Using {excel_count} Excel parameters from session state")

# Re-extract all values from the rebuilt p_current
# Use before/after specific values if available, otherwise fall back to single values
actual_n_labels_before = p_current.get('n_labels_before', 0)
actual_n_labels_after = p_current.get('n_labels_after', p_current.get('n_labels', 12000))
actual_size_tb_before = p_current.get('size_tb_before', 0.0)
actual_size_tb_after = p_current.get('size_tb_after', p_current.get('size_tb', 5.0))
actual_n_models_before = p_current.get('n_models_before', 0)
actual_n_models_after = p_current.get('n_models_after', p_current.get('n_models', 3))
actual_security_before = p_current.get('security_before', 100000.0)
actual_security_after = p_current.get('security_after', 150000.0)
actual_scrap_rate_before = p_current.get('scrap_rate_before', 0.002)
actual_scrap_rate_after = p_current.get('scrap_rate_after', 0.0015)

# Legacy support: maintain old variable names for backward compatibility
actual_n_labels = actual_n_labels_after
actual_size_tb = actual_size_tb_after
actual_n_models = actual_n_models_after

# Also update the individual variables for backward compatibility
n_labels = p_current.get('n_labels', 12000)
size_tb = p_current.get('size_tb', 5.0)
n_models = p_current.get('n_models', 3)
w = p_current.get('w', 30.0)
tau = p_current.get('tau', 0.005)
phi = p_current.get('phi', 1.3)
cTB_yr = p_current.get('cTB_yr', 276.0)
alpha_yr = p_current.get('alpha_yr', 480.0)
beta_ops_yr = p_current.get('beta_ops_yr', 8400.0)
security_before = p_current.get('security_before', 100000.0)
security_after = p_current.get('security_after', 150000.0)
eta_val = p_current.get('eta', np.log(2)/100000.0)
L_breach_yr = p_current.get('L_breach_yr', 280000.0)
units_per_year = p_current.get('units_per_year', 100000)
material_cost_per_unit = p_current.get('material_cost_per_unit', 300.0)
scrap_rate_before = p_current.get('scrap_rate_before', 0.002)
scrap_rate_after = p_current.get('scrap_rate_after', 0.0015)
operating_hours_year = p_current.get('operating_hours_year', 4000.0)
cm_per_unit = p_current.get('cm_per_unit', 250.0)
downtime_hours_avoided = p_current.get('downtime_hours_avoided', 75.0)
oee_improvement_rate = p_current.get('oee_improvement_rate', 0.03)
capex = p_current.get('capex', 360000.0)
useful_life_years = p_current.get('useful_life_years', 7.0)

base_inputs = dict(
    n_labels=actual_n_labels,
    size_tb=actual_size_tb,
    security_spend_yr=actual_security_after,
    n_models=actual_n_models,
    scrap_rate=actual_scrap_rate_after
)

# Use new compute functions
cost_before_breakdown = get_or_compute_cost_before(params)
cost_after_breakdown = get_or_compute_cost_after(params)
benefits_breakdown = get_or_compute_benefits(params)
net_value_result = get_or_compute_net_value(params)

cost_before = cost_before_breakdown["total"]
cost_after = cost_after_breakdown["total"]
benefits = benefits_breakdown["total"]
net_value = net_value_result["net_value_usd"]
incremental_cost = net_value_result["cost_delta_usd"]

# ============================================================================
# HELPER: Symbol Mapping Expander
# ============================================================================
def show_symbol_mapping():
    """Display symbol to code key mapping expander."""
    with st.expander("📋 Symbol ↔ Code Key Mapping"):
        st.markdown("""
        | Symbol | Meaning | Code key |
        |---|---|---|
        | \\(w\\) | Wage ($/hr) | `w` |
        | \\(\\tau\\) | Time per label (hr) | `tau` |
        | \\(\\phi\\) | Overhead multiplier | `phi` |
        | \\(c^{TB}_{yr}\\) | Storage cost ($/TB·yr) | `cTB_yr` |
        | \\(\\alpha_{yr}\\) | ETL/processing cost ($/TB·yr) | `alpha_yr` |
        | \\(\\beta^{ops}_{yr}\\) | MLOps cost ($/model·yr) | `beta_ops_yr` |
        | \\(\\eta\\) | Security effectiveness param | `eta` |
        | \\(L^{breach}_{yr}\\) | Expected breach loss baseline ($/yr) | `L_breach_yr` |
        | \\(U\\) | Units per year | `units_per_year` |
        | \\(m\\) | Material cost per unit ($/unit) | `material_cost_per_unit` |
        | \\(s_{\\text{before}}\\) | Scrap rate before AI | `scrap_rate_before` |
        | \\(s_{\\text{after}}\\) | Scrap rate after AI | `scrap_rate_after` |
        | \\(H\\) | Operating hours per year (hr/yr) | `operating_hours_year` |
        | \\(CM\\) | Contribution margin per unit ($/unit) | `cm_per_unit` |
        | \\(h_{dt}\\) | Downtime hours avoided (hr/yr) | `downtime_hours_avoided` |
        | \\(r_{oee}\\) | OEE improvement rate (fraction) | `oee_improvement_rate` |
        | \\(K\\) | CAPEX ($) | `capex` |
        | \\(L\\) | Useful life (years) | `useful_life_years` |
        | \\(S_{\\text{before}}\\) | Security spend before AI ($/yr) | `security_before` |
        | \\(S_{\\text{after}}\\) | Security spend after AI ($/yr) | `security_after` |
        | \\(D\\) | Dataset size (TB) | `size_tb` |
        | \\(N_m\\) | Number of models | `n_models` |
        | \\(N_\\ell\\) | Labels per year | `n_labels` |
        """)

# ============================================================================
# PAGE 2: PARAMETER DEFINITIONS & MODEL EXPLANATION
# ============================================================================
if current_page == 2:
    st.markdown("## 📖 Parameter Definitions & Model Explanation")
    
    st.markdown("""
    ### Model Overview
    This dashboard analyzes the net financial impact of implementing AI in module-level automotive manufacturing. 
    The model considers both costs (labor, storage, ETL, MLOps, capital, cyber risk) and benefits 
    (scrap reduction, downtime avoidance, throughput/OEE gains, security risk reduction).
    """)
    
    st.markdown("### Parameter Definitions")
    
    param_data = []
    for k, (low, mid, high) in ranges.items():
        param_data.append({
            "Parameter": k,
            "Description": PARAM_EXPLANATIONS.get(k, "No description available"),
            "Current Value": f"{p_current.get(k, mid):,.4f}" if k in p_current else f"{mid:,.4f}",
            "Typical Range": f"{low:.4f} - {high:.4f}"
        })
    
    param_df = pd.DataFrame(param_data)
    st.dataframe(param_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ### Cost Model Components
    
    1. **Labeling Cost**: Labor wage × time per label × number of labels × overhead multiplier
    2. **Storage Cost**: Annual storage rate × dataset size (TB)
    3. **Operations Cost**: Processing cost × dataset size + MLOps cost × number of models
    4. **Risk Cost**: Expected breach loss × probability of breach (exponential decay with security spending)
    5. **Material Cost**: Scrap rate × units per year × material cost per unit
    6. **Capital Cost**: CAPEX ÷ useful life (depreciation)
    
    ### Benefit Model Components
    
    1. **Scrap Reduction**: (Scrap rate before - Scrap rate after) × units × material cost
    2. **Downtime Avoidance**: Hours avoided × cost per hour
    3. **OEE Improvement**: OEE gain × units × contribution margin
    4. **Risk Reduction**: Reduction in expected breach loss due to improved security spending
    """)
    
    # ============================================================================
    # COST-BENEFIT EQUATIONS (FULL SIMPLE AND DETAILED VERSIONS)
    # ============================================================================
    st.markdown("---")
    st.markdown("## 📐 Cost-Benefit Equations")
    
    # Simplified Version
    with st.container():
        st.markdown("""
        <div style="background: #f0f7ff; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #0066cc; margin-bottom: 1rem;">
        """, unsafe_allow_html=True)
        
        st.markdown("### 📐 Simplified: Full Cost-Benefit Equation")
        
        st.markdown("""
        **Simple Equation:**
        $$
        \\text{Net Value} = \\text{Total Benefits} - \\text{Incremental Cost}
        $$
        
        where:
        - **Total Benefits** = OEE Improvement + Downtime Avoidance + Risk Reduction + Scrap Reduction
        - **Incremental Cost** = Cost After AI - Cost Before AI
        
        **What this means:** This is your bottom line—how much money you save (or lose) each year from implementing AI.
        
        - **If positive:** AI pays for itself and generates savings
        - **If negative:** AI costs more than it saves
        
        **Example:** If benefits = $1,000,000 and incremental cost = $300,000, then net value = $700,000 per year.
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Detailed Version
    with st.container():
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #0066cc;">
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Detailed: Full Cost-Benefit Model Equations")
        
        st.markdown("""
        **Net Value Calculation:**
        $$
        \\text{Net} = B_{\\text{total}} - (C_{\\text{after}} - C_{\\text{before}})
        $$
        
        **Total Benefits:**
        $$
        B_{\\text{total}} = B^{scrap} + B^{dt} + B^{OEE} + B^{risk}
        $$
        
        where:
        - \\(B^{scrap} = (s_{\\text{before}} - s_{\\text{after}}) \\cdot U \\cdot m\\) = Scrap reduction benefit
        - \\(B^{dt} = h_{dt} \\cdot C_{hr}^{dt}\\) = Downtime avoidance benefit
        - \\(B^{OEE} = r_{OEE} \\cdot U \\cdot CM\\) = OEE improvement benefit
        - \\(B^{risk} = C^{risk}(S_{\\text{before}}) - C^{risk}(S_{\\text{after}})\\) = Risk reduction benefit
        
        **Cost Before AI:**
        $$
        C_{\\text{before}} = C_{\\text{total}}(s_{\\text{before}}, S_{\\text{before}})
        $$
        
        **Cost After AI:**
        $$
        C_{\\text{after}} = C_{\\text{total}}(s_{\\text{after}}, S_{\\text{after}})
        $$
        
        where \\(C_{\\text{total}}(s, S) = C^{label} + C^{store} + C^{ops} + C^{risk}(S) + C^{mat}(s) + C^{cap}\\) is the total cost function.
        
        **Component Definitions:**
        - \\(C^{label}\\) = Annual labeling cost (labor × labels per year × time per label × overhead)
        - \\(C^{store}\\) = Annual storage cost (dataset size × storage cost per TB-year)
        - \\(C^{ops}\\) = Annual operations cost (data processing + ETL pipelines)
        - \\(C^{risk}(S)\\) = Annual cyber risk cost (decreases with security spending \\(S\\))
        - \\(C^{mat}(s)\\) = Annual material cost from scrap (scrap rate \\(s\\) × units × material cost)
        - \\(C^{cap}\\) = Annual capital cost (CAPEX ÷ useful life years)
        
        **Rationale:** Net value represents the annual financial impact of AI implementation. Positive values indicate that benefits exceed incremental costs, justifying the investment. This metric is used for ROI calculations and decision-making.
        """)
        
        show_symbol_mapping()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Final Equation")
    st.latex(r"\text{Net Value} = \text{Benefits} - (\text{Cost After} - \text{Cost Before})")
    st.caption(f"**Benefits** = ${benefits:,.0f} | **Cost After** = ${cost_after:,.0f} | **Cost Before** = ${cost_before:,.0f} | **Net Value** = ${net_value:,.0f}")
    st.markdown("**Full equation:** Net Value = Benefits − (Cost_after − Cost_before)")

# ============================================================================
# PAGE 3: DATA BREACH RISK VALUATION
# ============================================================================
elif current_page == 3:
    st.markdown("## 🔒 Data Breach Risk Valuation")
    st.markdown("Monte Carlo-based estimation of expected annual data breach losses")
    
    # Model Parameters
    st.markdown("### Model Parameters")
    
    col_param1, col_param2, col_param3 = st.columns(3)
    
    with col_param1:
        breach_revenue = st.number_input(
            "Annual Revenue ($ billions)",
            min_value=1.0,
            max_value=1000.0,
            value=142.6,
            step=1.0,
            help="Firm's annual revenue in $ billions. Used for size-adjusted breach probability.",
            key="breach_revenue"
        )
    
    with col_param2:
        breach_rating = st.selectbox(
            "Security Rating",
            options=['A', 'B+', 'B', 'C'],
            index=1,  # Default to B+
            help="Security rating from SecurityScorecard. Lower ratings indicate higher breach risk.",
            key="breach_rating"
        )
    
    with col_param3:
        breach_p_l_correlation = st.slider(
            "P-L Correlation",
            min_value=0.0,
            max_value=0.5,
            value=0.25,
            step=0.05,
            help="Correlation between breach probability and impact magnitude. Higher values indicate that weaker security correlates with larger impacts.",
            key="breach_p_l_corr"
        )
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        breach_n_simulations = st.number_input(
            "Monte Carlo Simulations",
            min_value=10000,
            max_value=1000000,
            value=250000,
            step=25000,
            help="Number of Monte Carlo simulation iterations. Higher values provide more accurate distributions but take longer.",
            key="breach_n_sim"
        )
    
    with col_sim2:
        breach_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=10000,
            value=42,
            help="Random seed for reproducibility.",
            key="breach_seed"
        )
    
    # Run simulation button
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_breach_sim = st.button("Run Risk Valuation", type="primary", use_container_width=True, key="run_breach_btn")
    
    # Initialize models
    if run_breach_sim or 'breach_results' in st.session_state:
        if run_breach_sim:
            with st.spinner("Running data breach risk valuation..."):
                np.random.seed(int(breach_seed))
                
                # Initialize models
                breach_model = BreachProbabilityModel()
                impact_model = ImpactEstimationModel()
                mc_valuation = MonteCarloValuation(
                    breach_model=breach_model,
                    impact_model=impact_model,
                    p_l_correlation=breach_p_l_correlation
                )
                
                # Run simulation
                results = mc_valuation.run_simulation(
                    revenue=breach_revenue,
                    rating=breach_rating,
                    n_simulations=int(breach_n_simulations)
                )
                
                # Store results
                st.session_state.breach_results = results
        else:
            results = st.session_state.breach_results
        
        # Display Results
        st.markdown("---")
        st.markdown("### Valuation Results")
        
        # KPI Cards
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        
        with col_kpi1:
            st.metric("Expected Annual Loss", f"${results['expected_value']:.1f}M")
        with col_kpi2:
            st.metric("Standard Deviation", f"${results['std_dev']:.1f}M")
        with col_kpi3:
            st.metric("95th Percentile", f"${results['percentile_95']:.1f}M")
        with col_kpi4:
            p_breach_mean = results['components']['p_breach'][0]
            st.metric("Breach Probability", f"{p_breach_mean:.2%}")
        
        # Distribution Visualization
        st.markdown("---")
        st.markdown("### Loss Distribution")
        
        loss_dist = results['expected_loss_distribution']
        
        # Histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=loss_dist,
            nbinsx=100,
            name="Expected Loss Distribution",
            marker_color='#0066cc',
            opacity=0.7
        ))
        
        # Add percentile lines
        fig_hist.add_vline(
            x=results['percentile_50'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: ${results['percentile_50']:.1f}M"
        )
        fig_hist.add_vline(
            x=results['percentile_95'],
            line_dash="dash",
            line_color="orange",
            annotation_text=f"95th %ile: ${results['percentile_95']:.1f}M"
        )
        
        fig_hist.update_layout(
            title="Expected Annual Data Breach Loss Distribution",
            xaxis_title="Expected Loss ($ millions)",
            yaxis_title="Probability Density",
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Cumulative Distribution
        sorted_losses = np.sort(loss_dist)
        cumulative = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
        
        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(
            x=sorted_losses,
            y=cumulative,
            mode='lines',
            name='Cumulative Distribution',
            line=dict(color='#0066cc', width=2)
        ))
        fig_cdf.add_hline(
            y=0.95,
            line_dash="dash",
            line_color="orange",
            annotation_text="95% VaR"
        )
        fig_cdf.update_layout(
            title="Cumulative Distribution Function",
            xaxis_title="Expected Loss ($ millions)",
            yaxis_title="Cumulative Probability",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12)
        )
        st.plotly_chart(fig_cdf, use_container_width=True)
        
        # Statistics Table
        st.markdown("---")
        st.markdown("### Risk Statistics")
        
        stats_data = {
            'Metric': [
                'Expected Value',
                'Standard Deviation',
                '5th Percentile (VaR 95%)',
                '25th Percentile',
                '50th Percentile (Median)',
                '75th Percentile',
                '95th Percentile (Tail Risk)',
                '95% Confidence Interval (Lower)',
                '95% Confidence Interval (Upper)',
                'Breach Probability (Mean)',
                'Breach Probability (Std Dev)',
                'Conditional Impact (Mean)',
                'Conditional Impact (Std Dev)',
                'P-L Correlation'
            ],
            'Value': [
                f"${results['expected_value']:.2f}M",
                f"${results['std_dev']:.2f}M",
                f"${results['percentile_5']:.2f}M",
                f"${results['percentile_25']:.2f}M",
                f"${results['percentile_50']:.2f}M",
                f"${results['percentile_75']:.2f}M",
                f"${results['percentile_95']:.2f}M",
                f"${results['confidence_interval_95'][0]:.2f}M",
                f"${results['confidence_interval_95'][1]:.2f}M",
                f"{results['components']['p_breach'][0]:.4f} ({results['components']['p_breach'][0]:.2%})",
                f"{results['components']['p_breach'][1]:.4f}",
                f"${results['components']['l_impact'][0]:.2f}M",
                f"${results['components']['l_impact'][1]:.2f}M",
                f"{results['actual_correlation']:.3f}"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Component Breakdown
        st.markdown("---")
        st.markdown("### Component Breakdown")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.markdown("#### Breach Probability Components")
            breach_model = BreachProbabilityModel()
            p_results = breach_model.combined_probability_estimation(breach_revenue, breach_rating)
            
            comp_data = {
                'Component': ['Size-Adjusted', 'Rating-Based', 'Combined (Precision-Weighted)'],
                'Probability': [
                    f"{p_results['components']['size_adjusted'][0]:.4f}",
                    f"{p_results['components']['rating_adjusted'][0]:.4f}",
                    f"{p_results['expected']:.4f}"
                ],
                'Std Dev': [
                    f"{p_results['components']['size_adjusted'][1]:.4f}",
                    f"{p_results['components']['rating_adjusted'][1]:.4f}",
                    f"{p_results['std_dev']:.4f}"
                ]
            }
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        with col_comp2:
            st.markdown("#### Impact Estimation Components")
            impact_model = ImpactEstimationModel()
            l_results = impact_model.combined_impact_estimation(breach_revenue)
            
            impact_data = {
                'Source': ['Benchmark (IBM-Ponemon)', 'Insurance-Implied', 'Combined (Weighted)'],
                'Expected ($M)': [
                    f"{l_results['components']['benchmark']['expected']:.1f}",
                    f"{l_results['components']['insurance_implied']['expected']:.1f}",
                    f"{l_results['expected']:.1f}"
                ],
                'Std Dev ($M)': [
                    f"{l_results['components']['benchmark']['std_dev']:.1f}",
                    f"{l_results['components']['insurance_implied']['std_dev']:.1f}",
                    f"{l_results['std_dev']:.1f}"
                ]
            }
            impact_df = pd.DataFrame(impact_data)
            st.dataframe(impact_df, use_container_width=True, hide_index=True)
        
        # Validation against Benchmarks
        st.markdown("---")
        st.markdown("### Benchmark Validation")
        
        validation = validate_against_benchmarks(
            expected_loss=results['expected_value'],
            revenue=breach_revenue,
            p_breach=results['components']['p_breach'][0]
        )
        
        val_data = {
            'Check': list(validation.keys()),
            'Estimate': [f"{validation[k]['estimate']:.4f}" for k in validation.keys()],
            'Benchmark Range': [str(validation[k]['benchmark_range']) for k in validation.keys()],
            'Status': ['✅ PASS' if validation[k]['pass'] else '❌ FAIL' for k in validation.keys()],
            'Source': [validation[k]['source'] for k in validation.keys()]
        }
        val_df = pd.DataFrame(val_data)
        st.dataframe(val_df, use_container_width=True, hide_index=True)
        
        # Sensitivity Analysis
        st.markdown("---")
        st.markdown("### Sensitivity Analysis")
        
        sensitivity = sensitivity_analysis(
            breach_model=breach_model,
            impact_model=impact_model,
            base_revenue=breach_revenue,
            base_rating=breach_rating,
            p_l_correlation=breach_p_l_correlation
        )
        
        base_expected = sensitivity['Base Case']['expected']
        
        # Tornado Chart
        scenarios = []
        deviations = []
        for scenario, vals in sensitivity.items():
            if scenario != 'Base Case':
                deviation = vals['expected'] - base_expected
                scenarios.append(scenario)
                deviations.append(deviation)
        
        # Sort by absolute deviation
        sorted_data = sorted(zip(scenarios, deviations), key=lambda x: abs(x[1]), reverse=True)
        scenarios_sorted = [s[0] for s in sorted_data]
        deviations_sorted = [s[1] for s in sorted_data]
        colors = ['red' if d < 0 else 'green' for d in deviations_sorted]
        
        fig_tornado = go.Figure()
        fig_tornado.add_trace(go.Bar(
            y=scenarios_sorted,
            x=deviations_sorted,
            orientation='h',
            marker_color=colors,
            opacity=0.7,
            text=[f"${d:+.1f}M" for d in deviations_sorted],
            textposition='outside'
        ))
        fig_tornado.add_vline(x=0, line_width=2, line_color="black")
        fig_tornado.update_layout(
            title="Sensitivity Analysis: Impact on Expected Loss",
            xaxis_title="Change in Expected Loss ($ millions)",
            yaxis_title="Scenario",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12)
        )
        st.plotly_chart(fig_tornado, use_container_width=True)
        
        # Sensitivity Table
        sens_data = {
            'Scenario': list(sensitivity.keys()),
            'Expected Loss ($M)': [f"{sensitivity[k]['expected']:.1f}" for k in sensitivity.keys()],
            'Std Dev ($M)': [f"{sensitivity[k]['std_dev']:.1f}" for k in sensitivity.keys()],
            '95% CI Lower ($M)': [f"{sensitivity[k]['ci_95'][0]:.1f}" for k in sensitivity.keys()],
            '95% CI Upper ($M)': [f"{sensitivity[k]['ci_95'][1]:.1f}" for k in sensitivity.keys()],
            'Breach Prob.': [f"{sensitivity[k]['p_breach']:.4f}" for k in sensitivity.keys()],
            'Impact ($M)': [f"{sensitivity[k]['l_impact']:.1f}" for k in sensitivity.keys()]
        }
        sens_df = pd.DataFrame(sens_data)
        st.dataframe(sens_df, use_container_width=True, hide_index=True)
        
        # Risk Management Implications
        st.markdown("---")
        st.markdown("### Risk Management Implications")
        
        expected_loss = results['expected_value']
        tail_risk = results['percentile_95']
        
        st.info(f"""
        **Capital Allocation:** Consider ${expected_loss:.0f}M annual reserve for data risk
        
        **Insurance Coverage:** Target ${tail_risk:.0f}M+ for catastrophic protection
        
        **Security Investment:** Justified up to ${expected_loss:.0f}M annually for risk reduction
        
        **Risk Appetite:** {breach_rating} rating implies {p_breach_mean:.1%} annual breach probability
        """)
    
    else:
        st.info("👆 Click 'Run Risk Valuation' to perform the analysis.")

# ============================================================================
# PAGE 4: NET VALUE (COMBINED WITH OPERATIONS OPTIMIZATION)
# ============================================================================
elif current_page == 4:
    st.markdown("## 💹 Net Value")
    st.markdown("**Complete cost-benefit analysis with operations optimization.**")
    
    # Executive summary metrics
    roi_pct = None
    if incremental_cost not in (None, 0):
        roi_pct = (net_value / incremental_cost) * 100 if incremental_cost != 0 else None
    
    payback_years = None
    if net_value > 0:
        payback_years = incremental_cost / net_value if incremental_cost is not None else None
    
    payback_display = "N/A"
    if payback_years is not None:
        if payback_years < 1:
            payback_display = "< 1 yr"
        elif payback_years == np.inf:
            payback_display = "Not reached"
        else:
            payback_display = f"{payback_years:.1f} yrs"
    
    st.markdown("### 📊 Executive Summary")
    summary_cols = st.columns(5)
    with summary_cols[0]:
        st.metric("💰 Net Annual Value", f"${net_value:,.0f}")
    with summary_cols[1]:
        st.metric("📈 Total Benefits", f"${benefits:,.0f}")
    with summary_cols[2]:
        st.metric("📊 Incremental Cost", f"${incremental_cost:,.0f}")
    with summary_cols[3]:
        st.metric("🎯 ROI", f"{roi_pct:.1f}%" if roi_pct is not None else "—")
    with summary_cols[4]:
        st.metric("⏱ Payback Period", payback_display)
    
    st.markdown("""
    <div class="help-callout">
        <span class="help-icon">ℹ️</span>
        <strong>How to read this:</strong> Net Annual Value shows year-one profit impact, ROI normalizes benefits vs incremental cost, and Payback estimates how quickly savings offset investments.
    </div>
    """, unsafe_allow_html=True)
    
    # Adjustable Parameters Section
    st.markdown("### 🎛️ Adjustable Parameters")
    with st.expander("📊 Adjust AI Operational Benefits Parameters", expanded=True):
        st.markdown("""
        <div class="help-callout">
            <span class="help-icon">💡</span>
            <strong>Tip:</strong> Use these sliders to test upside/downside scenarios. Annual AI Costs are spread evenly across categories for the net view, so increasing costs boosts the blue bars in the chart below.
        </div>
        """, unsafe_allow_html=True)
        # Default values
        default_values = {
            'OEE Improvement': 1035000,
            'Downtime Avoidance': 2000000,
            'Scrap Reduction': 250125,
            'Energy Savings': 319200,
            'Workers Comp Reduction': 19866,
            'Annual AI Costs': 570425
        }
        
        # Store values in session state for persistence
        if 'benefits_oee' not in st.session_state:
            st.session_state.benefits_oee = default_values['OEE Improvement']
        if 'benefits_downtime' not in st.session_state:
            st.session_state.benefits_downtime = default_values['Downtime Avoidance']
        if 'benefits_scrap' not in st.session_state:
            st.session_state.benefits_scrap = default_values['Scrap Reduction']
        if 'benefits_energy' not in st.session_state:
            st.session_state.benefits_energy = default_values['Energy Savings']
        if 'benefits_workers_comp' not in st.session_state:
            st.session_state.benefits_workers_comp = default_values['Workers Comp Reduction']
        if 'benefits_ai_costs' not in st.session_state:
            st.session_state.benefits_ai_costs = default_values['Annual AI Costs']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Gross Benefits (USD/year)")
            oee_benefit = st.number_input(
                "OEE Improvement", 
                min_value=0, 
                max_value=10000000, 
                value=int(st.session_state.benefits_oee), 
                step=10000,
                key="oee_benefit"
            )
            downtime_benefit = st.number_input(
                "Downtime Avoidance", 
                min_value=0, 
                max_value=10000000, 
                value=int(st.session_state.benefits_downtime), 
                step=10000,
                key="downtime_benefit"
            )
            scrap_benefit = st.number_input(
                "Scrap Reduction", 
                min_value=0, 
                max_value=10000000, 
                value=int(st.session_state.benefits_scrap), 
                step=10000,
                key="scrap_benefit"
            )
        
        with col2:
            st.markdown("#### Gross Benefits (continued)")
            energy_benefit = st.number_input(
                "Energy Savings", 
                min_value=0, 
                max_value=10000000, 
                value=int(st.session_state.benefits_energy), 
                step=10000,
                key="energy_benefit"
            )
            workers_comp_benefit = st.number_input(
                "Workers Comp Reduction", 
                min_value=0, 
                max_value=10000000, 
                value=int(st.session_state.benefits_workers_comp), 
                step=1000,
                key="workers_comp_benefit"
            )
            st.markdown("---")
            ai_costs_annual = st.number_input(
                "Annual AI Costs", 
                min_value=0, 
                max_value=5000000, 
                value=int(st.session_state.benefits_ai_costs), 
                step=10000,
                key="ai_costs_annual",
                help="Total annual cost of AI implementation"
            )
        
        # Update session state when values change
        st.session_state.benefits_oee = oee_benefit
        st.session_state.benefits_downtime = downtime_benefit
        st.session_state.benefits_scrap = scrap_benefit
        st.session_state.benefits_energy = energy_benefit
        st.session_state.benefits_workers_comp = workers_comp_benefit
        st.session_state.benefits_ai_costs = ai_costs_annual
    
    st.markdown("---")
    
    # AI Operational Benefits: Gross vs Net chart
    st.markdown("#### AI Operational Benefits: Gross vs Net")
    categories_benefits = ['OEE Improvement', 'Downtime Avoidance', 'Scrap Reduction', 
                          'Energy Savings', 'Workers Comp Reduction']
    benefits_values = [oee_benefit, downtime_benefit, scrap_benefit, energy_benefit, workers_comp_benefit]
    net_benefits_values = [b - (ai_costs_annual/len(categories_benefits)) for b in benefits_values]
    
    fig_benefits = go.Figure()
    fig_benefits.add_trace(go.Bar(
        x=categories_benefits,
        y=[b/1000 for b in benefits_values],
        name='Gross Benefits',
        marker_color='#2ecc71',
        opacity=0.7,
        text=[f'${b/1000:.0f}K' for b in benefits_values],
        textposition='outside'
    ))
    fig_benefits.add_trace(go.Bar(
        x=categories_benefits,
        y=[b/1000 for b in net_benefits_values],
        name='Net Benefits',
        marker_color='#3498db',
        text=[f'${b/1000:.0f}K' for b in net_benefits_values],
        textposition='outside'
    ))
    fig_benefits.update_layout(
        title='AI Operational Benefits: Gross vs Net (Excluding Risk Reduction)',
        xaxis_title='Benefit Category',
        yaxis_title='Value (Thousand $)',
        barmode='group',
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter", size=12),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_benefits, use_container_width=True)
    
    # Export functionality
    st.markdown("### 📤 Export Results")
    summary_rows = [
        ("Net Annual Value", net_value),
        ("Total Benefits", benefits),
        ("Incremental Cost", incremental_cost),
        ("Cost Before AI", cost_before),
        ("Cost After AI", cost_after),
        ("ROI (%)", roi_pct if roi_pct is not None else np.nan),
        ("Payback (years)", payback_years if payback_years is not None else np.nan)
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    
    benefit_labels = {
        "oee_improvement": "OEE Improvement",
        "downtime_avoidance": "Downtime Avoidance",
        "risk_reduction": "Risk Reduction",
        "scrap_reduction": "Scrap Reduction"
    }
    benefits_df = pd.DataFrame(
        [(benefit_labels.get(k, k.replace("_", " ").title()), v) 
         for k, v in benefits_breakdown.items() if k in benefit_labels],
        columns=["Benefit Component", "Value ($)"]
    )
    
    def cost_df(cost_dict: Dict[str, float], title_prefix: str) -> pd.DataFrame:
        exclude = {"total", "point_estimate_usd"}
        records = []
        for k, v in cost_dict.items():
            if k in exclude or not isinstance(v, (int, float)):
                continue
            records.append((f"{title_prefix} - {k.replace('_', ' ').title()}", v))
        return pd.DataFrame(records, columns=["Cost Component", "Value ($)"])
    
    cost_before_df = cost_df(cost_before_breakdown, "Before AI")
    cost_after_df = cost_df(cost_after_breakdown, "After AI")
    
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index=False, sheet_name='Summary')
        benefits_df.to_excel(writer, index=False, sheet_name='Benefits')
        cost_before_df.to_excel(writer, index=False, sheet_name='Costs_Before_AI')
        cost_after_df.to_excel(writer, index=False, sheet_name='Costs_After_AI')
    excel_data = excel_buffer.getvalue()
    
    st.download_button(
        "⬇️ Download Excel Summary",
        data=excel_data,
        file_name="BMW_AI_Net_Value_Summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_net_value_excel"
    )
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "BMW AI Net Value Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Net Annual Value: ${net_value:,.0f}", ln=True)
    pdf.cell(0, 10, f"Total Benefits: ${benefits:,.0f}", ln=True)
    pdf.cell(0, 10, f"Incremental Cost: ${incremental_cost:,.0f}", ln=True)
    if roi_pct is not None:
        pdf.cell(0, 10, f"ROI: {roi_pct:.1f}%", ln=True)
    pdf.cell(0, 10, f"Payback Period: {payback_display}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Benefit Breakdown", ln=True)
    pdf.set_font("Arial", size=11)
    for _, row in benefits_df.iterrows():
        pdf.cell(0, 8, f"- {row['Benefit Component']}: ${row['Value ($)']:,.0f}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Key Notes", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, "Values are annualized. ROI compares Net Value vs incremental cost, and Payback estimates how quickly savings offset the incremental investment.")
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    
    st.download_button(
        "⬇️ Download PDF Summary",
        data=pdf_bytes,
        file_name="BMW_AI_Net_Value_Summary.pdf",
        mime="application/pdf",
        key="download_net_value_pdf"
    )
    
    st.markdown("---")
    
    st.markdown("""
    **THEORY:**
    - max π = pq - wφL - rK - material·q - C_fixed
    - s.t. q = A[αL^(-ρ) + (1-α)K^(-ρ)]^(-1/ρ) ≥ q_min
    
    **Lagrangian:** L = π(L,K) + λ[q(L,K) - q_min]
    
    **FOCs:**
    - ∂L/∂L: (p - c_m)·MPL - wφ + λ·MPL = 0
    - ∂L/∂K: (p - c_m)·MPK - r + λ·MPK = 0
    - ∂L/∂λ: q(L,K) - q_min = 0  (if binding)
    
    **Economic interpretation:**
    - λ = shadow price of production constraint ($/unit)
    - λ > 0 → constraint binds (produce exactly q_min)
    - λ = 0 || λ < 0 → unconstrained optimum dominates
    """)
    
    # Inputs panel with notebook defaults
    with st.expander("📥 Inputs (USD)", expanded=True):
        # Excel Import Section for CES Parameters
        st.markdown("#### 📊 Import Parameters from Excel")
        col_excel1, col_excel2 = st.columns([2, 1])
        
        with col_excel1:
            ces_uploaded_file = st.file_uploader(
                "Upload CES Production Parameters",
                type=['xlsx', 'xls'],
                key="ces_excel_upload",
                help="Upload an Excel file with Parameter and Value columns"
            )
        
        with col_excel2:
            # Create and download template for CES parameters
            ces_template_data = {
                'Parameter': [
                    'A (Total factor productivity)',
                    'α (Labor share parameter)',
                    'ρ (Substitution parameter)',
                    'w: Wage ($/hr)',
                    'φ: Overhead multiplier',
                    'r: Capital rate (annual)',
                    'Material cost ($/unit)',
                    'Selling price ($/unit)',
                    'q_min: Minimum output (units/year)',
                    'τ: Labeling time (hr/point)',
                    'Labels per year',
                    'Dataset size (TB)',
                    'Number of models',
                    'Storage cost ($/TB-yr)',
                    'ETL cost ($/TB-yr)',
                    'MLOps cost ($/model-yr)',
                    'Capital expenditure ($)',
                    'Useful life (years)',
                    'Expected annual loss pre-AI ($)',
                    'AI-driven risk reduction'
                ],
                'Value': [
                    1.2,
                    0.60,
                    0.43,
                    31.0,
                    1.35,
                    0.15,
                    725.0,
                    1460.0,
                    100000,
                    0.0065,
                    12000.0,
                    5.0,
                    3.0,
                    325.0,
                    650.0,
                    19000.0,
                    1000000.0,
                    8.75,
                    390000.0,
                    0.425
                ]
            }
            ces_template_df = pd.DataFrame(ces_template_data)
            
            # Convert to Excel bytes
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                ces_template_df.to_excel(writer, index=False, sheet_name='CES_Parameters')
            ces_excel_bytes = output.getvalue()
            
            st.download_button(
                "📥 Download Template",
                ces_excel_bytes,
                file_name="CES_Production_Parameters_Template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_ces_template"
            )
        
        # Default values for CES parameters
        ces_default_values = {
            'A (Total factor productivity)': 1.2,
            'α (Labor share parameter)': 0.60,
            'ρ (Substitution parameter)': 0.43,
            'w: Wage ($/hr)': 31.0,
            'φ: Overhead multiplier': 1.35,
            'r: Capital rate (annual)': 0.15,
            'Material cost ($/unit)': 725.0,
            'Selling price ($/unit)': 1460.0,
            'q_min: Minimum output (units/year)': 100000,
            'τ: Labeling time (hr/point)': 0.0065,
            'Labels per year': 12000.0,
            'Dataset size (TB)': 5.0,
            'Number of models': 3.0,
            'Storage cost ($/TB-yr)': 325.0,
            'ETL cost ($/TB-yr)': 650.0,
            'MLOps cost ($/model-yr)': 19000.0,
            'Capital expenditure ($)': 1000000.0,
            'Useful life (years)': 8.75,
            'Expected annual loss pre-AI ($)': 390000.0,
            'AI-driven risk reduction': 0.425
        }
        
        # Parse Excel file if uploaded
        if ces_uploaded_file is not None:
            current_file_id = ces_uploaded_file.file_id if hasattr(ces_uploaded_file, 'file_id') else id(ces_uploaded_file)
            last_processed_id = st.session_state.get('last_ces_excel_file_id')
            
            if current_file_id != last_processed_id:
                try:
                    df = pd.read_excel(ces_uploaded_file, engine='openpyxl')
                    
                    if 'Parameter' in df.columns and 'Value' in df.columns:
                        param_dict = dict(zip(df['Parameter'], df['Value']))
                    elif len(df.columns) >= 2:
                        param_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
                    else:
                        st.error("❌ Excel format not recognized. Expected 'Parameter' and 'Value' columns.")
                        param_dict = {}
                    
                    # Update default values with Excel values
                    for param_name, value in param_dict.items():
                        param_name_clean = str(param_name).strip()
                        for key in ces_default_values.keys():
                            if param_name_clean.lower() == key.lower():
                                try:
                                    ces_default_values[key] = float(value)
                                except (ValueError, TypeError):
                                    st.warning(f"⚠️ Invalid value for {key}: {value}")
                    
                    st.session_state.last_ces_excel_file_id = current_file_id
                    st.success("✅ CES parameters loaded from Excel!")
                    
                except Exception as e:
                    st.error(f"❌ Error reading Excel file: {str(e)}")
        
        # Store in session state (simplified keys)
        ces_param_map = {
            'A (Total factor productivity)': 'ces_A_val',
            'α (Labor share parameter)': 'ces_alpha_val',
            'ρ (Substitution parameter)': 'ces_rho_val',
            'w: Wage ($/hr)': 'ces_w_val',
            'φ: Overhead multiplier': 'ces_phi_val',
            'r: Capital rate (annual)': 'ces_r_val',
            'Material cost ($/unit)': 'ces_cm_val',
            'Selling price ($/unit)': 'ces_p_val',
            'q_min: Minimum output (units/year)': 'ces_qmin_val',
            'τ: Labeling time (hr/point)': 'ces_tau_val',
            'Labels per year': 'ces_nlabels_val',
            'Dataset size (TB)': 'ces_tb_val',
            'Number of models': 'ces_nmodels_val',
            'Storage cost ($/TB-yr)': 'ces_ctb_val',
            'ETL cost ($/TB-yr)': 'ces_alpha_yr_val',
            'MLOps cost ($/model-yr)': 'ces_beta_val',
            'Capital expenditure ($)': 'ces_capex_val',
            'Useful life (years)': 'ces_life_val',
            'Expected annual loss pre-AI ($)': 'ces_breach_val',
            'AI-driven risk reduction': 'ces_risk_reduction_val'
        }
        
        # Initialize session state
        for key, session_key in ces_param_map.items():
            if session_key not in st.session_state:
                st.session_state[session_key] = ces_default_values[key]
        
        # Update from Excel if loaded (after processing)
        if ces_uploaded_file is not None and 'last_ces_excel_file_id' in st.session_state:
            for key, session_key in ces_param_map.items():
                st.session_state[session_key] = ces_default_values[key]
        
        st.markdown("---")
        st.markdown("""
        <div class="help-callout">
            <span class="help-icon">🧠</span>
            <strong>Need guidance?</strong> Start with the Excel template, then tweak any field inline. Labor/Capital inputs (left) feed the CES production curve, while AI/Data costs (right) flow into fixed cost and risk models.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### CES Production Parameters")
            A = st.number_input("A (Total factor productivity)", 0.1, 100.0, 
                               float(st.session_state.ces_A_val), 0.1, key="ces_A",
                               help="TFP scaled to dollar inputs (Oberfield & Raval 2021)")
            st.session_state.ces_A_val = A
            alpha_ces = st.number_input("α (Labor share parameter)", 0.01, 0.99, 
                                       float(st.session_state.ces_alpha_val), 0.01, key="ces_alpha",
                                       help="Labor distribution weight")
            st.session_state.ces_alpha_val = alpha_ces
            rho = st.number_input("ρ (Substitution parameter)", -0.99, 0.99, 
                                 float(st.session_state.ces_rho_val), 0.01, key="ces_rho",
                                 help="σ = 0.70, capital-labor complements (Oberfield & Raval 2021)")
            st.session_state.ces_rho_val = rho
            
            st.markdown("#### Cost Parameters")
            w = st.number_input("w: Wage ($/hr)", 10.0, 200.0, 
                               float(st.session_state.ces_w_val), 1.0, key="ces_w")
            st.session_state.ces_w_val = w
            phi = st.number_input("φ: Overhead multiplier", 1.0, 3.0, 
                                 float(st.session_state.ces_phi_val), 0.1, key="ces_phi")
            st.session_state.ces_phi_val = phi
            r = st.number_input("r: Capital rate (annual)", 0.01, 0.50, 
                               float(st.session_state.ces_r_val), 0.01, key="ces_r",
                               help="Annual cost of capital as fraction (industry standard)")
            st.session_state.ces_r_val = r
            material_cost = st.number_input("Material cost ($/unit)", 50.0, 1000.0, 
                                            float(st.session_state.ces_cm_val), 10.0, key="ces_cm")
            st.session_state.ces_cm_val = material_cost
            selling_price = st.number_input("Selling price ($/unit)", 100.0, 2000.0, 
                                            float(st.session_state.ces_p_val), 10.0, key="ces_p")
            st.session_state.ces_p_val = selling_price
            
        with col2:
            st.markdown("#### Constraint & Fixed Costs")
            q_min = st.number_input("q_min: Minimum output (units/year)", 1000, 1000000, 
                                   int(st.session_state.ces_qmin_val), 1000, key="ces_qmin")
            st.session_state.ces_qmin_val = q_min
            
            st.markdown("#### AI/Data Costs (USD/year)")
            tau = st.number_input("τ: Labeling time (hr/point)", 0.001, 0.1, 
                                 float(st.session_state.ces_tau_val), 0.0001, key="ces_tau")
            st.session_state.ces_tau_val = tau
            n_labels = st.number_input("Labels per year", 1000.0, 100000.0, 
                                      float(st.session_state.ces_nlabels_val), 1000.0, key="ces_nlabels")
            st.session_state.ces_nlabels_val = n_labels
            TB = st.number_input("Dataset size (TB)", 1.0, 100.0, 
                                float(st.session_state.ces_tb_val), 0.5, key="ces_tb")
            st.session_state.ces_tb_val = TB
            n_models = st.number_input("Number of models", 1.0, 10.0, 
                                     float(st.session_state.ces_nmodels_val), 1.0, key="ces_nmodels")
            st.session_state.ces_nmodels_val = n_models
            cTB_yr = st.number_input("Storage cost ($/TB-yr)", 100.0, 1000.0, 
                                     float(st.session_state.ces_ctb_val), 10.0, key="ces_ctb")
            st.session_state.ces_ctb_val = cTB_yr
            alpha_yr = st.number_input("ETL cost ($/TB-yr)", 100.0, 2000.0, 
                                      float(st.session_state.ces_alpha_yr_val), 10.0, key="ces_alpha_yr")
            st.session_state.ces_alpha_yr_val = alpha_yr
            beta_ops_yr = st.number_input("MLOps cost ($/model-yr)", 5000.0, 50000.0, 
                                         float(st.session_state.ces_beta_val), 1000.0, key="ces_beta")
            st.session_state.ces_beta_val = beta_ops_yr
            
            st.markdown("#### Capital & Risk")
            capex = st.number_input("Capital expenditure ($)", 100000.0, 10000000.0, 
                                   float(st.session_state.ces_capex_val), 100000.0, key="ces_capex")
            st.session_state.ces_capex_val = capex
            useful_life_years = st.number_input("Useful life (years)", 1.0, 20.0, 
                                               float(st.session_state.ces_life_val), 0.25, key="ces_life")
            st.session_state.ces_life_val = useful_life_years
            L_breach_baseline = st.number_input("Expected annual loss pre-AI ($)", 10000.0, 1000000.0, 
                                               float(st.session_state.ces_breach_val), 10000.0, key="ces_breach")
            st.session_state.ces_breach_val = L_breach_baseline
            risk_reduction_rate = st.number_input("AI-driven risk reduction", 0.0, 1.0, 
                                                  float(st.session_state.ces_risk_reduction_val), 0.01, key="ces_risk_reduction")
            st.session_state.ces_risk_reduction_val = risk_reduction_rate
    
    # Calculate fixed costs
    C_label = tau * n_labels * w * phi
    C_store = cTB_yr * TB
    C_ops = alpha_yr * TB + beta_ops_yr * n_models
    C_cap_annual = capex / useful_life_years
    risk_after_AI = L_breach_baseline * (1 - risk_reduction_rate)
    C_fixed = C_label + C_store + C_ops + risk_after_AI + C_cap_annual
    
    # Compute optimization
    st.markdown("---")
    st.markdown("### 📊 Optimization Results")
    
    # Step 1: Unconstrained optimum
    st.markdown("#### [1] Unconstrained Optimum (λ = 0)")
    
    def negative_profit(x):
        return -profit_function(x[0], x[1], selling_price, w, phi, r, material_cost, A, alpha_ces, rho, C_fixed)
    
    result_unconstrained = minimize(
        negative_profit,
        x0=[10000, 2000000],
        bounds=[(100, 100000), (10000, 20000000)],
        method='L-BFGS-B',
        options={'ftol': 1e-12}
    )
    
    L_uncon, K_uncon = result_unconstrained.x
    q_uncon = ces_production(L_uncon, K_uncon, A, alpha_ces, rho)
    profit_uncon = profit_function(L_uncon, K_uncon, selling_price, w, phi, r, material_cost, A, alpha_ces, rho, C_fixed)
    
    col_uncon1, col_uncon2, col_uncon3, col_uncon4 = st.columns(4)
    with col_uncon1:
        st.metric("Labor", f"{L_uncon:,.0f} hours")
    with col_uncon2:
        st.metric("Capital", f"${K_uncon:,.0f}")
    with col_uncon3:
        st.metric("Output", f"{q_uncon:,.0f} units")
    with col_uncon4:
        st.metric("Profit", f"${profit_uncon:,.0f}")
    
    constraint_status = "✅ SATISFIED" if q_uncon >= q_min else "❌ VIOLATED"
    st.caption(f"Constraint: {constraint_status} (q ≥ {q_min:,.0f})")
    
    # Step 2: Lagrangian solution
    st.markdown("---")
    st.markdown("#### [2] Lagrangian Solution (constraint enforced)")
    
    result = compute_ces_optimum(A, alpha_ces, rho, w, phi, r, selling_price, material_cost, q_min, C_fixed)
    L_opt, K_opt, q_opt, profit_opt, lambda_opt, success, residuals = result
    
    if success and L_opt is not None:
        col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
        with col_opt1:
            st.metric("Labor", f"{L_opt:,.0f} hours")
        with col_opt2:
            st.metric("Capital", f"${K_opt:,.0f}")
        with col_opt3:
            st.metric("Output", f"{q_opt:,.0f} units")
        with col_opt4:
            st.metric("Profit", f"${profit_opt:,.0f}")
        
        if lambda_opt is not None:
            st.metric("λ (shadow price)", f"${lambda_opt:,.4f} per unit",
                     help="Marginal value of relaxing the minimum production constraint by one unit")
            constraint_binds = residuals.get('constraint_binds', False)
            st.caption(f"Max FOC residual: {residuals.get('max_residual', 0):.2e} | Constraint: {'BINDING' if constraint_binds else 'SLACK'}")
        
        # Validation Tests
        st.markdown("---")
        st.markdown("### ✅ Validation Tests")
        
        if lambda_opt is not None:
            MPL_opt, MPK_opt = marginal_products(L_opt, K_opt, A, alpha_ces, rho)
            p_net = selling_price - material_cost
            
            labor_condition_lhs = p_net * MPL_opt - w * phi
            labor_condition_rhs = -lambda_opt * MPL_opt
            labor_error = abs(labor_condition_lhs - labor_condition_rhs)
            
            capital_condition_lhs = p_net * MPK_opt - r
            capital_condition_rhs = -lambda_opt * MPK_opt
            capital_error = abs(capital_condition_lhs - capital_condition_rhs)
            
            test_col1, test_col2, test_col3, test_col4 = st.columns(4)
            
            with test_col1:
                test1_pass = labor_error < 1e-3
                st.metric("Labor FOC Error", f"{labor_error:.2e}", "✅ PASS" if test1_pass else "❌ FAIL")
            
            with test_col2:
                test2_pass = capital_error < 1e-3
                st.metric("Capital FOC Error", f"{capital_error:.2e}", "✅ PASS" if test2_pass else "❌ FAIL")
            
            with test_col3:
                constraint_error = abs(q_opt - q_min)
                test3_pass = constraint_error < 10
                st.metric("Constraint Error", f"{constraint_error:.2f} units", delta="✅ PASS" if test3_pass else "❌ FAIL")
            
            with test_col4:
                capital_labor_ratio = K_opt / L_opt if L_opt > 0 else 0
                test4_pass = 10 <= capital_labor_ratio <= 500
                st.metric("K/L Ratio", f"${capital_labor_ratio:.2f}/hr", delta="✅ PASS" if test4_pass else "❌ FAIL")
            
            # Returns to scale
            epsilon_L = (MPL_opt * L_opt) / q_opt
            epsilon_K = (MPK_opt * K_opt) / q_opt
            returns_to_scale = epsilon_L + epsilon_K
            st.caption(f"Returns to Scale: Labor elasticity = {epsilon_L:.3f}, Capital elasticity = {epsilon_K:.3f}, Sum = {returns_to_scale:.3f} {'✓' if 0.95 <= returns_to_scale <= 1.05 else '✗'}")
        
        # Visualizations - Security Investment Effectiveness Dashboard
        st.markdown("---")
        st.markdown("### 📈 Security Investment Effectiveness Dashboard")
        st.markdown("**AI-Enhanced Cybersecurity ROI Analysis**")
        
        # Parameters for security investment analysis
        risk_baseline = 26_000_000  # Annual breach loss without AI ($) - from notebook
        investment = 5_000_000  # AI implementation cost ($) - from notebook
        total_net_benefit = 8_500_000  # Annual operational benefit from AI ($) - from notebook
        
        # Generate risk reduction range: 0% to 99%
        risk_reduction_pct = np.linspace(0, 99, 100)
        risk_reduction_frac = risk_reduction_pct / 100
        
        # Calculate metrics
        risk_after = risk_baseline * (1 - risk_reduction_frac)
        annual_investment_cost = investment / 5  # Assume 5-year amortization
        risk_reduction_benefit = risk_baseline * (1 - risk_reduction_frac)
        net_profit = total_net_benefit - risk_reduction_benefit - annual_investment_cost
        roi = net_profit / investment
        payback_years = np.where(net_profit > 0, investment / net_profit, np.inf)
        
        # Color palette (colorblind-safe)
        COLOR_PROFIT = '#0173B2'      # Blue
        COLOR_ROI = '#DE8F05'         # Orange
        COLOR_PAYBACK = '#029E73'     # Green
        COLOR_THRESHOLD = '#CC78BC'   # Purple
        COLOR_DANGER = '#CA3542'      # Red
        
        # ============================================================================
        # FIGURE 1: MULTI-PANEL DASHBOARD (Interactive Plotly)
        # ============================================================================
        
        # Panel A: Net Annual Profit (full width on top)
        st.markdown("#### Panel A: Net Annual Profit vs Security Investment Effectiveness")
        fig_profit = go.Figure()
        
        # Profitable region fill
        profit_positive = net_profit > 0
        if np.any(profit_positive):
            fig_profit.add_trace(go.Scatter(
                x=risk_reduction_pct,
                y=np.maximum(net_profit / 1e6, 0),
                mode='none',
                fill='tozeroy',
                fillcolor='rgba(1, 115, 178, 0.2)',
                name='Profitable region',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        # Loss region fill
        profit_negative = net_profit <= 0
        if np.any(profit_negative):
            fig_profit.add_trace(go.Scatter(
                x=risk_reduction_pct,
                y=np.minimum(net_profit / 1e6, 0),
                mode='none',
                fill='tozeroy',
                fillcolor='rgba(202, 53, 66, 0.2)',
                name='Loss region',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        # Main line
        fig_profit.add_trace(go.Scatter(
            x=risk_reduction_pct,
            y=net_profit / 1e6,
            mode='lines+markers',
            name='Net Profit',
            line=dict(color=COLOR_PROFIT, width=2.5),
            marker=dict(size=5, symbol='circle', color=COLOR_PROFIT),
            showlegend=False,
            hovertemplate='Effectiveness: %{x:.1f}%<br>Net Profit: $%{y:.2f}M<extra></extra>'
        ))
        
        # Break-even point
        breakeven_idx = np.argmin(np.abs(net_profit))
        breakeven_pct = risk_reduction_pct[breakeven_idx]
        fig_profit.add_vline(
            x=breakeven_pct,
            line_dash="dash",
            line_color=COLOR_THRESHOLD,
            line_width=1.5,
            annotation_text=f'Break-even: {breakeven_pct:.1f}%',
            annotation_position="top"
        )
        fig_profit.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.8, opacity=0.5)
        
        # Annotations for key points
        key_points = [0, 25, 50, 75, 95, 99]
        for pct in key_points:
            idx = int(pct)
            if idx < len(net_profit):
                fig_profit.add_annotation(
                    x=pct,
                    y=net_profit[idx] / 1e6,
                    text=f'${net_profit[idx]/1e6:.1f}M',
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=COLOR_PROFIT,
                    bgcolor='white',
                    bordercolor=COLOR_PROFIT,
                    borderwidth=1,
                    font=dict(size=9, color='black', family='sans-serif')
                )
        
        fig_profit.update_layout(
            title=dict(text='Panel A: Net Annual Profit vs Security Investment Effectiveness', 
                      font=dict(size=12, family='sans-serif', color='black')),
            xaxis_title='Security Effectiveness (% Risk Reduction)',
            yaxis_title='Net Annual Profit (Million $)',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="sans-serif", size=11),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, 
                       bgcolor='rgba(255,255,255,0.9)', bordercolor='gray', borderwidth=1),
            xaxis=dict(range=[-2, 101], gridcolor='rgba(0,0,0,0.1)', gridwidth=1, showgrid=True),
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)', gridwidth=1, showgrid=True)
        )
        st.plotly_chart(fig_profit, use_container_width=True)
        
        # Panel B and C side by side (on bottom)
        col_b, col_c = st.columns(2)
        
        with col_b:
            st.markdown("#### Panel B: ROI Sensitivity")
            roi_clipped = np.clip(roi, -5, 10)
            
            fig_roi = go.Figure()
            
            # ROI > 1x region
            roi_above_one = roi_clipped > 1
            if np.any(roi_above_one):
                fig_roi.add_trace(go.Scatter(
                    x=risk_reduction_pct,
                    y=np.maximum(roi_clipped, np.ones_like(roi_clipped)),
                    mode='none',
                    fill='tonexty',
                    fillcolor='rgba(222, 143, 5, 0.2)',
                    name='ROI > 1x (profitable)',
                    showlegend=True,
                    hoverinfo='skip'
                ))
                fig_roi.add_trace(go.Scatter(
                    x=risk_reduction_pct,
                    y=np.ones_like(roi_clipped),
                    mode='none',
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # ROI < 1x region
            roi_below_one = roi_clipped <= 1
            if np.any(roi_below_one):
                fig_roi.add_trace(go.Scatter(
                    x=risk_reduction_pct,
                    y=np.ones_like(roi_clipped),
                    mode='none',
                    fill='tonexty',
                    fillcolor='rgba(202, 53, 66, 0.2)',
                    name='ROI < 1x (loss)',
                    showlegend=True,
                    hoverinfo='skip'
                ))
                fig_roi.add_trace(go.Scatter(
                    x=risk_reduction_pct,
                    y=np.minimum(roi_clipped, np.ones_like(roi_clipped)),
                    mode='none',
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Main line
            fig_roi.add_trace(go.Scatter(
                x=risk_reduction_pct,
                y=roi_clipped,
                mode='lines+markers',
                name='ROI',
                line=dict(color=COLOR_ROI, width=2.5),
                marker=dict(size=5, symbol='square', color=COLOR_ROI),
                showlegend=False,
                hovertemplate='Effectiveness: %{x:.1f}%<br>ROI: %{y:.2f}x<extra></extra>'
            ))
            
            # Threshold lines
            fig_roi.add_hline(y=1, line_dash="dash", line_color=COLOR_THRESHOLD, line_width=1.5,
                            annotation_text='ROI = 1x (break-even)', annotation_position="right")
            fig_roi.add_hline(y=3, line_dash="dot", line_color="gray", line_width=1, opacity=0.5)
            fig_roi.add_hline(y=5, line_dash="dot", line_color="gray", line_width=1, opacity=0.5)
            fig_roi.add_annotation(x=98, y=3.2, text='3x', showarrow=False, 
                                  font=dict(size=8, color='gray', family='sans-serif'))
            fig_roi.add_annotation(x=98, y=5.2, text='5x', showarrow=False, 
                                  font=dict(size=8, color='gray', family='sans-serif'))
            
            fig_roi.update_layout(
                title=dict(text='Panel B: ROI Sensitivity', 
                          font=dict(size=12, family='sans-serif', color='black')),
                xaxis_title='Security Effectiveness (%)',
                yaxis_title='Return on Investment (×)',
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="sans-serif", size=11),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                           bgcolor='rgba(255,255,255,0.9)', bordercolor='gray', borderwidth=1, font=dict(size=8)),
                xaxis=dict(range=[-2, 101], gridcolor='rgba(0,0,0,0.1)', gridwidth=1, showgrid=True),
                yaxis=dict(range=[-5.5, 10.5], gridcolor='rgba(0,0,0,0.1)', gridwidth=1, showgrid=True)
            )
            st.plotly_chart(fig_roi, use_container_width=True)
        
        with col_c:
            st.markdown("#### Panel C: Investment Payback Period")
            payback_clipped = np.clip(payback_years, 0, 20)
            
            fig_payback = go.Figure()
            
            # < 5 years region
            payback_fast = payback_clipped <= 5
            if np.any(payback_fast):
                fig_payback.add_trace(go.Scatter(
                    x=risk_reduction_pct,
                    y=np.maximum(payback_clipped, np.full_like(payback_clipped, 5.0)),
                    mode='none',
                    fill='tonexty',
                    fillcolor='rgba(2, 158, 115, 0.3)',
                    name='< 5 years (attractive)',
                    showlegend=True,
                    hoverinfo='skip'
                ))
                fig_payback.add_trace(go.Scatter(
                    x=risk_reduction_pct,
                    y=np.full_like(payback_clipped, 5.0),
                    mode='none',
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # > 5 years region
            payback_slow = payback_clipped > 5
            if np.any(payback_slow):
                fig_payback.add_trace(go.Scatter(
                    x=risk_reduction_pct,
                    y=np.full_like(payback_clipped, 5.0),
                    mode='none',
                    fill='tonexty',
                    fillcolor='rgba(202, 53, 66, 0.2)',
                    name='> 5 years (slow)',
                    showlegend=True,
                    hoverinfo='skip'
                ))
                fig_payback.add_trace(go.Scatter(
                    x=risk_reduction_pct,
                    y=np.minimum(payback_clipped, np.full_like(payback_clipped, 5.0)),
                    mode='none',
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Main line
            fig_payback.add_trace(go.Scatter(
                x=risk_reduction_pct,
                y=payback_clipped,
                mode='lines+markers',
                name='Payback Period',
                line=dict(color=COLOR_PAYBACK, width=2.5),
                marker=dict(size=5, symbol='triangle-up', color=COLOR_PAYBACK),
                showlegend=False,
                hovertemplate='Effectiveness: %{x:.1f}%<br>Payback: %{y:.2f} years<extra></extra>'
            ))
            
            # Threshold line
            fig_payback.add_hline(y=5, line_dash="dash", line_color=COLOR_THRESHOLD, line_width=1.5,
                                annotation_text='5-year threshold', annotation_position="right")
            
            fig_payback.update_layout(
                title=dict(text='Panel C: Investment Payback Period', 
                          font=dict(size=12, family='sans-serif', color='black')),
                xaxis_title='Security Effectiveness (%)',
                yaxis_title='Payback Period (Years)',
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="sans-serif", size=11),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                           bgcolor='rgba(255,255,255,0.9)', bordercolor='gray', borderwidth=1, font=dict(size=8)),
                xaxis=dict(range=[-2, 101], gridcolor='rgba(0,0,0,0.1)', gridwidth=1, showgrid=True),
                yaxis=dict(range=[-0.5, 20.5], gridcolor='rgba(0,0,0,0.1)', gridwidth=1, showgrid=True)
            )
            # Invert y-axis for intuitive interpretation (shorter is better)
            fig_payback.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_payback, use_container_width=True)
        
        # Summary Statistics Table
        st.markdown("---")
        st.markdown("### Summary Statistics")
        
        percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
        summary_data = {
            'Effectiveness (%)': [f"{pct:.0f}" for pct in percentiles],
            'Residual Risk ($)': [f"${risk_after[int(pct)]:,.0f}" for pct in percentiles],
            'Net Profit ($M)': [f"${net_profit[int(pct)]/1e6:.2f}M" for pct in percentiles],
            'ROI (×)': [f"{roi[int(pct)]:.2f}x" for pct in percentiles],
            'Payback (years)': [('∞' if payback_years[int(pct)] == np.inf else f'{payback_years[int(pct)]:.2f}') for pct in percentiles]
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Key Insights
        st.markdown("---")
        st.markdown("### Key Insights")
        
        breakeven_idx = np.argmin(np.abs(net_profit))
        roi_3x_idx = np.argmin(np.abs(roi - 3))
        payback_5y_idx = np.argmin(np.abs(payback_years - 5))
        max_profit_idx = np.argmax(net_profit)
        
        insights = f"""
        1. **Break-even point:** {risk_reduction_pct[breakeven_idx]:.1f}% risk reduction → Minimum viability threshold
        
        2. **Attractive ROI (3×):** {risk_reduction_pct[roi_3x_idx]:.1f}% risk reduction → Investment generates $3 profit per $1 invested
        
        3. **Fast payback (<5 years):** {risk_reduction_pct[payback_5y_idx]:.1f}% risk reduction → Investment recovers within typical budget cycles
        
        4. **Optimal effectiveness range:** {risk_reduction_pct[roi_3x_idx]:.0f}%-90% → Maximize value while maintaining realistic security goals
        
        5. **Maximum profit:** ${net_profit[max_profit_idx]/1e6:.2f}M at {risk_reduction_pct[max_profit_idx]:.0f}% effectiveness → Theoretical upper bound (99% effectiveness)
        """
        st.markdown(insights)
        
        # Summary
        st.markdown("---")
        st.markdown("### 📋 Final Summary")
        
        summary_data = {
            'Metric': ['Optimal Labor', 'Optimal Capital', 'Optimal Production', 'Maximum Profit'],
            'Value': [
                f"{L_opt:,.0f} hours/year",
                f"${K_opt:,.0f}",
                f"{q_opt:,.0f} units/year",
                f"${profit_opt:,.0f}/year"
            ]
        }
        if lambda_opt is not None:
            summary_data['Metric'].append('Shadow Price (λ)')
            summary_data['Value'].append(f"${lambda_opt:,.4f}/unit")
            summary_data['Metric'].append('Interpretation')
            summary_data['Value'].append(f"Relaxing minimum production by 1 unit would {'increase' if lambda_opt < 0 else 'decrease'} profit by ${abs(lambda_opt):.2f}")
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
    else:
        st.error("❌ Optimization failed. Please check input parameters and try again.")
        if 'error' in residuals:
            st.error(f"Error details: {residuals['error']}")

# ============================================================================
# PAGE 5: STATISTICAL VALIDATION (REMOVED)
# ============================================================================
# Page 5 has been removed - Validation functionality is no longer needed
if False and current_page == 5:
    st.markdown("## 📈 Statistical Validation")
    
    # Monte Carlo Simulation
    st.markdown("### Monte Carlo Simulation — Net Value Distribution")
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_mc = st.button("Run Monte Carlo Simulation", type="primary", use_container_width=True)
    
    if run_mc or 'mc_results_net' in st.session_state:
        if run_mc:
            with st.spinner("Running Monte Carlo simulation..."):
                mc_seed = st.session_state.get('mc_seed', 42)
                mc_runs = st.session_state.get('mc_runs', 5000)
                np.random.seed(int(mc_seed))
                
                def net_value_sample():
                    p = sample_params()
                    # Read values from sampled parameter dict
                    n_lab_before = p.get('n_labels_before', p.get('n_labels', 0))
                    size_before = p.get('size_tb_before', p.get('size_tb', 0.0))
                    n_mod_before = p.get('n_models_before', p.get('n_models', 0))
                    sec_before = p.get('security_before', p.get('security_spend_usd_per_year_before', 100000.0))
                    scrap_before = p.get('scrap_rate_before', 0.002)
                    
                    n_lab_after = p.get('n_labels_after', p.get('n_labels', 12000))
                    size_after = p.get('size_tb_after', p.get('size_tb', 5.0))
                    n_mod_after = p.get('n_models_after', p.get('n_models', 3))
                    sec_after = p.get('security_after', p.get('security_spend_usd_per_year_after', 150000.0))
                    scrap_after = p.get('scrap_rate_after', 0.0015)
                    
                    cb = total_cost_breakdown(n_lab_before, size_before, sec_before, 
                                            n_mod_before, scrap_before, p=p, use_before=True)["total"]
                    ca = total_cost_breakdown(n_lab_after, size_after, sec_after, 
                                            n_mod_after, scrap_after, p=p, use_before=False)["total"]
                    b = total_benefits_legacy(p=p, 
                                     security_before=sec_before, security_after=sec_after)
                    return b - (ca - cb)
                
                draws = np.array([net_value_sample() for _ in range(int(mc_runs))])
                
                # Store draws in session state for regression analysis
                st.session_state.mc_draws = draws
                st.session_state.mc_params = []
                
                # Collect parameter values for regression
                param_samples = []
                for _ in range(int(mc_runs)):
                    p = sample_params()
                    param_samples.append(p)
                st.session_state.mc_params = param_samples
                
                # Statistics
                p10, p25, p50, p75, p90 = np.percentile(draws, [10, 25, 50, 75, 90])
                mean_val = np.mean(draws)
                std_val = np.std(draws)
                prob_positive = np.mean(draws > 0) * 100
                
                # Store results in session state
                st.session_state.mc_results_net = {
                    'draws': draws,
                    'p10': p10, 'p25': p25, 'p50': p50, 'p75': p75, 'p90': p90,
                    'mean': mean_val, 'std': std_val, 'prob_positive': prob_positive
                }
        else:
            # Use stored results
            results = st.session_state.mc_results_net
            draws = results['draws']
            p10 = results['p10']
            p25 = results['p25']
            p50 = results['p50']
            p75 = results['p75']
            p90 = results['p90']
            mean_val = results['mean']
            std_val = results['std']
            prob_positive = results['prob_positive']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Mean", f"${mean_val:,.0f}")
        with col2:
            st.metric("Median (P50)", f"${p50:,.0f}")
        with col3:
            st.metric("P10", f"${p10:,.0f}")
        with col4:
            st.metric("P90", f"${p90:,.0f}")
        with col5:
            st.metric("P(NPV>0)", f"{prob_positive:.1f}%")
        
        # Histogram
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=draws,
            nbinsx=50,
            marker_color="#0066cc",
            opacity=0.7,
            name="Net Value Distribution"
        ))
        fig_mc.add_vline(
            x=p50,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Median: ${p50:,.0f}",
            annotation_position="top"
        )
        fig_mc.add_vline(
            x=0,
            line_dash="dot",
            line_color="red",
            annotation_text="Break-even",
            annotation_position="bottom"
        )
        fig_mc.update_layout(
            height=450,
            xaxis_title="Net Annual Value ($)",
            yaxis_title="Frequency",
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12)
        )
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # Summary text
        st.markdown("### Validation Summary")
        if prob_positive > 50:
            st.success(f"✅ **The model shows positive net value across all scenarios tested.** "
                      f"Probability of positive NPV: {prob_positive:.1f}%. "
                      f"Median net value: ${p50:,.0f}.")
        else:
            st.warning(f"⚠️ **Model shows mixed results.** Probability of positive NPV: {prob_positive:.1f}%. "
                      f"Median net value: ${p50:,.0f}.")
    
    # Regression Analysis
    if 'mc_draws' in st.session_state and 'mc_params' in st.session_state:
        st.markdown("---")
        st.markdown("### Regression Analysis")
        st.caption("OLS regression of net value on parameter values")
        
        if len(st.session_state.mc_params) > 0:
            # Prepare data for regression
            param_df = pd.DataFrame(st.session_state.mc_params)
            y = st.session_state.mc_draws
            
            # Select key parameters for regression
            # Use parameter names that exist in the dataframe (filter to only those that exist)
            potential_params = ['w', 'w_before', 'w_after', 'tau', 'tau_after', 'phi', 'phi_before', 'phi_after',
                              'cTB_yr', 'cTB_yr_after', 'alpha_yr', 'alpha_yr_after', 'beta_ops_yr', 'beta_ops_yr_after',
                              'eta', 'L_breach_yr', 'scrap_rate_before', 'scrap_rate_after',
                              'material_cost_per_unit', 'units_per_year', 'operating_hours_year', 'operating_hours_per_year',
                              'cm_per_unit', 'downtime_hours_avoided', 'downtime_hours_avoided_per_year',
                              'oee_improvement_rate', 'oee_improvement_fraction',
                              'capex', 'capex_after', 'capex_before', 'useful_life_years', 'useful_life_years_after',
                              'n_labels_after', 'labels_per_year_after', 'size_tb_after', 'dataset_tb_after',
                              'n_models_after', 'models_deployed_after', 'security_before', 'security_after',
                              'security_spend_usd_per_year_before', 'security_spend_usd_per_year_after']
            
            # Filter to only parameters that exist in the dataframe
            key_params = [p for p in potential_params if p in param_df.columns]
            
            # If no parameters found, use all numeric columns
            if len(key_params) == 0:
                key_params = [col for col in param_df.columns if param_df[col].dtype in ['float64', 'int64']]
            
            # Limit to reasonable number for regression
            if len(key_params) > 20:
                key_params = key_params[:20]
            
            X = param_df[key_params]
            X = sm.add_constant(X)  # Add intercept
            
            # Fit OLS model
            model = sm.OLS(y, X).fit()
            
            # Get residuals and fitted values for diagnostics
            resid = model.resid
            fitted = model.fittedvalues
            
            # Display regression results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Regression Coefficients")
                coef_df = pd.DataFrame({
                    'Parameter': ['Intercept'] + key_params,
                    'Coefficient': model.params,
                    'Std Error': model.bse,
                    'P-value': model.pvalues,
                    'Significant': ['✓' if p < 0.05 else '' for p in model.pvalues]
                })
                coef_df['Coefficient'] = coef_df['Coefficient'].apply(lambda x: f"{x:,.2f}")
                coef_df['Std Error'] = coef_df['Std Error'].apply(lambda x: f"{x:,.2f}")
                coef_df['P-value'] = coef_df['P-value'].apply(lambda x: f"{x:.4f}")
                st.dataframe(coef_df, use_container_width=True, hide_index=True)
                
                st.markdown(f"**R²**: {model.rsquared:.4f}  |  **Adj R²**: {model.rsquared_adj:.4f}")
                st.markdown(f"**F-statistic**: {model.fvalue:.2f}  |  **Prob (F-statistic)**: {model.f_pvalue:.4f}")
            
            with col2:
                st.markdown("#### Model Diagnostics")
                # Residuals vs Fitted
                fig_resid = go.Figure()
                fig_resid.add_trace(go.Scatter(
                    x=fitted,
                    y=resid,
                    mode='markers',
                    marker=dict(color="#0066cc", opacity=0.6),
                    name="Residuals"
                ))
                fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                fig_resid.update_layout(
                    height=300,
                    xaxis_title="Fitted Values",
                    yaxis_title="Residuals",
                    margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Inter", size=12)
                )
                st.plotly_chart(fig_resid, use_container_width=True)
            
            # QQ Plot using statsmodels approach
            st.markdown("#### Q-Q Plot of Residuals")
            # Compute QQ plot data using statsmodels approach (normal distribution)
            sorted_resid = np.sort(resid)
            n = len(sorted_resid)
            # Theoretical quantiles for normal distribution
            theoretical_quantiles = norm.ppf(np.linspace(0.01, 0.99, n))
            sample_quantiles = sorted_resid
            
            # Compute fitted line (45-degree line)
            # Fit line: y = a + b*x where b=1 and a is chosen to pass through median
            median_theoretical = np.median(theoretical_quantiles)
            median_sample = np.median(sample_quantiles)
            # For 45-degree line: a = median_sample - median_theoretical
            line_intercept = median_sample - median_theoretical
            line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
            line_y = line_intercept + line_x
            
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                marker=dict(color="#0066cc", opacity=0.6, size=4),
                name="Residuals"
            ))
            # Add 45-degree reference line
            fig_qq.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                line=dict(color="red", dash="dash", width=2),
                name="45° Reference Line"
            ))
            fig_qq.update_layout(
                height=450,
                xaxis_title="Theoretical Quantiles (Normal)",
                yaxis_title="Sample Quantiles",
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Inter", size=12),
                showlegend=True
            )
            st.plotly_chart(fig_qq, use_container_width=True)
            
            # Normality test
            shapiro_stat, shapiro_p = shapiro(resid[:5000])  # Limit for Shapiro test
            jb_stat, jb_p = jarque_bera(resid)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Normality Tests:**")
                st.markdown(f"- Shapiro-Wilk: Statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}")
                st.markdown(f"- Jarque-Bera: Statistic = {jb_stat:.4f}, p-value = {jb_p:.4f}")
            
            with col2:
                # Heteroscedasticity test
                bp_stat, bp_p, _, _ = het_breuschpagan(resid, X)
                st.markdown("**Heteroscedasticity Test:**")
                st.markdown(f"- Breusch-Pagan: Statistic = {bp_stat:.4f}, p-value = {bp_p:.4f}")
    
    # Tornado Sensitivity
    st.markdown("### Tornado Sensitivity Analysis")
    st.caption("Impact of varying each parameter from current values to min/max ranges")
    
    p_current = likely_params()  # This now returns current parameter values
    
    def net_with(param, val):
        p = p_current.copy()
        p[param] = val
        # Read values from parameter dict, not hardcoded variables
        n_lab_before = p.get('n_labels_before', p.get('n_labels', 0))
        size_before = p.get('size_tb_before', p.get('size_tb', 0.0))
        n_mod_before = p.get('n_models_before', p.get('n_models', 0))
        sec_before = p.get('security_before', p.get('security_spend_usd_per_year_before', 100000.0))
        scrap_before = p.get('scrap_rate_before', 0.002)
        
        n_lab_after = p.get('n_labels_after', p.get('n_labels', 12000))
        size_after = p.get('size_tb_after', p.get('size_tb', 5.0))
        n_mod_after = p.get('n_models_after', p.get('n_models', 3))
        sec_after = p.get('security_after', p.get('security_spend_usd_per_year_after', 150000.0))
        scrap_after = p.get('scrap_rate_after', 0.0015)
        
        cb = total_cost_breakdown(n_lab_before, size_before, sec_before, 
                                n_mod_before, scrap_before, p=p, use_before=True)["total"]
        ca = total_cost_breakdown(n_lab_after, size_after, sec_after, 
                                n_mod_after, scrap_after, p=p, use_before=False)["total"]
        b = total_benefits_legacy(p=p,
                          security_before=sec_before, security_after=sec_after)
        return b - (ca - cb)
    
    records = []
    tornado_ranges = get_ranges_for_sampling()
    for k, (low, mid, high) in tornado_ranges.items():
        v_min = net_with(k, low)
        v_current = net_with(k, mid)  # mid is now the current value
        v_max = net_with(k, high)
        records.append([k, v_current - v_min, v_max - v_current])
        
    tor = pd.DataFrame(records, columns=["param", "down", "up"])
    tor["span"] = tor[["down", "up"]].abs().max(axis=1)
    tor = tor.sort_values("span", ascending=True)
    
    # Tornado chart
    fig_tornado = go.Figure()
    
    fig_tornado.add_trace(go.Bar(
        y=tor["param"],
        x=-tor["down"],
        orientation="h",
        name="Min → Current",
        marker_color="#d62728",
        opacity=0.7
    ))
    
    fig_tornado.add_trace(go.Bar(
        y=tor["param"],
        x=tor["up"],
        orientation="h",
        name="Current → Max",
        marker_color="#00a86b",
        opacity=0.7
    ))
    
    fig_tornado.add_vline(x=0, line_width=2, line_color="black")
    fig_tornado.update_layout(
        height=600,
        xaxis_title="Δ Net Value ($)",
        yaxis_title="Parameter",
        barmode="overlay",
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter", size=12)
    )
    st.plotly_chart(fig_tornado, use_container_width=True)
    
    # Sensitivity table
    tor_display = tor.copy()
    tor_display["down"] = tor_display["down"].apply(lambda x: f"${x:,.0f}")
    tor_display["up"] = tor_display["up"].apply(lambda x: f"${x:,.0f}")
    tor_display["span"] = tor_display["span"].apply(lambda x: f"${x:,.0f}")
    tor_display.columns = ["Parameter", "Min → Current", "Current → Max", "Max Impact"]
    st.dataframe(tor_display[["Parameter", "Min → Current", "Current → Max", "Max Impact"]], 
                use_container_width=True, hide_index=True)

# PAGE 6: REMOVED (DUPLICATE OF PAGE 3)
# ============================================================================
# Page 6 was a duplicate of Page 3 (Data Breach Risk Valuation) and has been removed
if False and current_page == 6:
    st.markdown("## 🔒 Data Breach Risk Valuation")
    st.markdown("Monte Carlo-based estimation of expected annual data breach losses")
    
    # Import the data breach risk classes (they're defined at the end of the file)
    # We'll need to make sure they're accessible - for now, let's define them inline or import
    
    # Parameters sidebar section (if not already in sidebar, add here)
    st.markdown("### Model Parameters")
    
    col_param1, col_param2, col_param3 = st.columns(3)
    
    with col_param1:
        breach_revenue = st.number_input(
            "Annual Revenue ($ billions)",
            min_value=1.0,
            max_value=1000.0,
            value=142.6,
            step=1.0,
            help="Firm's annual revenue in $ billions. Used for size-adjusted breach probability."
        )
    
    with col_param2:
        breach_rating = st.selectbox(
            "Security Rating",
            options=['A', 'B+', 'B', 'C'],
            index=1,  # Default to B+
            help="Security rating from SecurityScorecard. Lower ratings indicate higher breach risk."
        )
    
    with col_param3:
        breach_p_l_correlation = st.slider(
            "P-L Correlation",
            min_value=0.0,
            max_value=0.5,
            value=0.25,
            step=0.05,
            help="Correlation between breach probability and impact magnitude. Higher values indicate that weaker security correlates with larger impacts."
        )
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        breach_n_simulations = st.number_input(
            "Monte Carlo Simulations",
            min_value=10000,
            max_value=1000000,
            value=250000,
            step=25000,
            help="Number of Monte Carlo simulation iterations. Higher values provide more accurate distributions but take longer."
        )
    
    with col_sim2:
        breach_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=10000,
            value=42,
            help="Random seed for reproducibility."
        )
    
    # Run simulation button
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_breach_sim = st.button("Run Risk Valuation", type="primary", use_container_width=True)
    
    # Initialize models (using the classes from the end of the file)
    # We need to make sure these classes are accessible
    # For now, let's check if they exist, otherwise we'll need to move them
    
    if run_breach_sim or 'breach_results' in st.session_state:
        if run_breach_sim:
            with st.spinner("Running data breach risk valuation..."):
                np.random.seed(int(breach_seed))
                
                # Initialize models
                breach_model = BreachProbabilityModel()
                impact_model = ImpactEstimationModel()
                mc_valuation = MonteCarloValuation(
                    breach_model=breach_model,
                    impact_model=impact_model,
                    p_l_correlation=breach_p_l_correlation
                )
                
                # Run simulation
                results = mc_valuation.run_simulation(
                    revenue=breach_revenue,
                    rating=breach_rating,
                    n_simulations=int(breach_n_simulations)
                )
                
                # Store results
                st.session_state.breach_results = results
        else:
            results = st.session_state.breach_results
        
        # Display Results
        st.markdown("---")
        st.markdown("### Valuation Results")
        
        # KPI Cards
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        
        with col_kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Expected Annual Loss</div>
                <div class="kpi-value">${results['expected_value']:.1f}M</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_kpi2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Standard Deviation</div>
                <div class="kpi-value">${results['std_dev']:.1f}M</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_kpi3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">95th Percentile</div>
                <div class="kpi-value">${results['percentile_95']:.1f}M</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_kpi4:
            p_breach_mean = results['components']['p_breach'][0]
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Breach Probability</div>
                <div class="kpi-value">{p_breach_mean:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Distribution Visualization
        st.markdown("---")
        st.markdown("### Loss Distribution")
        
        loss_dist = results['expected_loss_distribution']
        
        # Histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=loss_dist,
            nbinsx=100,
            name="Expected Loss Distribution",
            marker_color='#0066cc',
            opacity=0.7
        ))
        
        # Add percentile lines
        fig_hist.add_vline(
            x=results['percentile_50'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: ${results['percentile_50']:.1f}M"
        )
        fig_hist.add_vline(
            x=results['percentile_95'],
            line_dash="dash",
            line_color="orange",
            annotation_text=f"95th %ile: ${results['percentile_95']:.1f}M"
        )
        
        fig_hist.update_layout(
            title="Expected Annual Data Breach Loss Distribution",
            xaxis_title="Expected Loss ($ millions)",
            yaxis_title="Probability Density",
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Cumulative Distribution
        sorted_losses = np.sort(loss_dist)
        cumulative = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
        
        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(
            x=sorted_losses,
            y=cumulative,
            mode='lines',
            name='Cumulative Distribution',
            line=dict(color='#0066cc', width=2)
        ))
        fig_cdf.add_hline(
            y=0.95,
            line_dash="dash",
            line_color="orange",
            annotation_text="95% VaR"
        )
        fig_cdf.update_layout(
            title="Cumulative Distribution Function",
            xaxis_title="Expected Loss ($ millions)",
            yaxis_title="Cumulative Probability",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12)
        )
        st.plotly_chart(fig_cdf, use_container_width=True)
        
        # Statistics Table
        st.markdown("---")
        st.markdown("### Risk Statistics")
        
        stats_data = {
            'Metric': [
                'Expected Value',
                'Standard Deviation',
                '5th Percentile (VaR 95%)',
                '25th Percentile',
                '50th Percentile (Median)',
                '75th Percentile',
                '95th Percentile (Tail Risk)',
                '95% Confidence Interval (Lower)',
                '95% Confidence Interval (Upper)',
                'Breach Probability (Mean)',
                'Breach Probability (Std Dev)',
                'Conditional Impact (Mean)',
                'Conditional Impact (Std Dev)',
                'P-L Correlation'
            ],
            'Value': [
                f"${results['expected_value']:.2f}M",
                f"${results['std_dev']:.2f}M",
                f"${results['percentile_5']:.2f}M",
                f"${results['percentile_25']:.2f}M",
                f"${results['percentile_50']:.2f}M",
                f"${results['percentile_75']:.2f}M",
                f"${results['percentile_95']:.2f}M",
                f"${results['confidence_interval_95'][0]:.2f}M",
                f"${results['confidence_interval_95'][1]:.2f}M",
                f"{results['components']['p_breach'][0]:.4f} ({results['components']['p_breach'][0]:.2%})",
                f"{results['components']['p_breach'][1]:.4f}",
                f"${results['components']['l_impact'][0]:.2f}M",
                f"${results['components']['l_impact'][1]:.2f}M",
                f"{results['actual_correlation']:.3f}"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Component Breakdown
        st.markdown("---")
        st.markdown("### Component Breakdown")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.markdown("#### Breach Probability Components")
            p_results = breach_model.combined_probability_estimation(breach_revenue, breach_rating)
            
            comp_data = {
                'Component': ['Size-Adjusted', 'Rating-Based', 'Combined (Precision-Weighted)'],
                'Probability': [
                    f"{p_results['components']['size_adjusted'][0]:.4f}",
                    f"{p_results['components']['rating_adjusted'][0]:.4f}",
                    f"{p_results['expected']:.4f}"
                ],
                'Std Dev': [
                    f"{p_results['components']['size_adjusted'][1]:.4f}",
                    f"{p_results['components']['rating_adjusted'][1]:.4f}",
                    f"{p_results['std_dev']:.4f}"
                ]
            }
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        with col_comp2:
            st.markdown("#### Impact Estimation Components")
            l_results = impact_model.combined_impact_estimation(breach_revenue)
            
            impact_data = {
                'Source': ['Benchmark (IBM-Ponemon)', 'Insurance-Implied', 'Combined (Weighted)'],
                'Expected ($M)': [
                    f"{l_results['components']['benchmark']['expected']:.1f}",
                    f"{l_results['components']['insurance_implied']['expected']:.1f}",
                    f"{l_results['expected']:.1f}"
                ],
                'Std Dev ($M)': [
                    f"{l_results['components']['benchmark']['std_dev']:.1f}",
                    f"{l_results['components']['insurance_implied']['std_dev']:.1f}",
                    f"{l_results['std_dev']:.1f}"
                ]
            }
            impact_df = pd.DataFrame(impact_data)
            st.dataframe(impact_df, use_container_width=True, hide_index=True)
        
        # Validation against Benchmarks
        st.markdown("---")
        st.markdown("### Benchmark Validation")
        
        validation = validate_against_benchmarks(
            expected_loss=results['expected_value'],
            revenue=breach_revenue,
            p_breach=results['components']['p_breach'][0]
        )
        
        val_data = {
            'Check': list(validation.keys()),
            'Estimate': [f"{validation[k]['estimate']:.4f}" for k in validation.keys()],
            'Benchmark Range': [str(validation[k]['benchmark_range']) for k in validation.keys()],
            'Status': ['✅ PASS' if validation[k]['pass'] else '❌ FAIL' for k in validation.keys()],
            'Source': [validation[k]['source'] for k in validation.keys()]
        }
        val_df = pd.DataFrame(val_data)
        st.dataframe(val_df, use_container_width=True, hide_index=True)
        
        # Sensitivity Analysis
        st.markdown("---")
        st.markdown("### Sensitivity Analysis")
        
        sensitivity = sensitivity_analysis(
            breach_model=breach_model,
            impact_model=impact_model,
            base_revenue=breach_revenue,
            base_rating=breach_rating,
            p_l_correlation=breach_p_l_correlation
        )
        
        base_expected = sensitivity['Base Case']['expected']
        
        # Tornado Chart
        scenarios = []
        deviations = []
        for scenario, vals in sensitivity.items():
            if scenario != 'Base Case':
                deviation = vals['expected'] - base_expected
                scenarios.append(scenario)
                deviations.append(deviation)
        
        # Sort by absolute deviation
        sorted_data = sorted(zip(scenarios, deviations), key=lambda x: abs(x[1]), reverse=True)
        scenarios_sorted = [s[0] for s in sorted_data]
        deviations_sorted = [s[1] for s in sorted_data]
        colors = ['red' if d < 0 else 'green' for d in deviations_sorted]
        
        fig_tornado = go.Figure()
        fig_tornado.add_trace(go.Bar(
            y=scenarios_sorted,
            x=deviations_sorted,
            orientation='h',
            marker_color=colors,
            opacity=0.7,
            text=[f"${d:+.1f}M" for d in deviations_sorted],
            textposition='outside'
        ))
        fig_tornado.add_vline(x=0, line_width=2, line_color="black")
        fig_tornado.update_layout(
            title="Sensitivity Analysis: Impact on Expected Loss",
            xaxis_title="Change in Expected Loss ($ millions)",
            yaxis_title="Scenario",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12)
        )
        st.plotly_chart(fig_tornado, use_container_width=True)
        
        # Sensitivity Table
        sens_data = {
            'Scenario': list(sensitivity.keys()),
            'Expected Loss ($M)': [f"{sensitivity[k]['expected']:.1f}" for k in sensitivity.keys()],
            'Std Dev ($M)': [f"{sensitivity[k]['std_dev']:.1f}" for k in sensitivity.keys()],
            '95% CI Lower ($M)': [f"{sensitivity[k]['ci_95'][0]:.1f}" for k in sensitivity.keys()],
            '95% CI Upper ($M)': [f"{sensitivity[k]['ci_95'][1]:.1f}" for k in sensitivity.keys()],
            'Breach Prob.': [f"{sensitivity[k]['p_breach']:.4f}" for k in sensitivity.keys()],
            'Impact ($M)': [f"{sensitivity[k]['l_impact']:.1f}" for k in sensitivity.keys()]
        }
        sens_df = pd.DataFrame(sens_data)
        st.dataframe(sens_df, use_container_width=True, hide_index=True)
        
        # Risk Management Implications
        st.markdown("---")
        st.markdown("### Risk Management Implications")
        
        expected_loss = results['expected_value']
        tail_risk = results['percentile_95']
        
        st.info(f"""
        **Capital Allocation:** Consider ${expected_loss:.0f}M annual reserve for data risk
        
        **Insurance Coverage:** Target ${tail_risk:.0f}M+ for catastrophic protection
        
        **Security Investment:** Justified up to ${expected_loss:.0f}M annually for risk reduction
        
        **Risk Appetite:** {breach_rating} rating implies {p_breach_mean:.1%} annual breach probability
        """)
    
    else:
        st.info("👆 Click 'Run Risk Valuation' to perform the analysis.")

# ============================================================================
# PAGE 4: OPERATIONS OPTIMIZATION (CES + LAGRANGIAN) - REMOVED (DUPLICATE)
# ============================================================================
# This was a duplicate of the Page 4 content above and has been removed
if False:  # This page is now page 4, handled above
    st.markdown("## ⚙️ Operations Optimization (CES + Lagrangian)")
    st.markdown("**Maximize profit with CES production under minimum output constraint.**")
    st.caption("Diagnostic appendix only. Results do not feed into Net Value calculations.")
    
    # Inputs panel
    with st.expander("📥 Inputs (USD)", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### CES Production Parameters")
            A = st.number_input("A (Total factor productivity)", 0.1, 100.0, 1.0, 0.1, key="ces_A")
            alpha = st.number_input("α (Labor share parameter)", 0.01, 0.99, 0.5, 0.01, key="ces_alpha")
            rho = st.number_input("ρ (Substitution parameter)", -0.99, 0.99, 0.5, 0.01, key="ces_rho",
                                 help="ρ = 0 gives Cobb-Douglas, ρ > 0 gives elasticity < 1")
            
            st.markdown("#### Cost Parameters")
            w = st.number_input("w: Wage ($/hr)", 10.0, 200.0, 30.0, 1.0, key="ces_w")
            phi = st.number_input("φ: Overhead multiplier", 1.0, 3.0, 1.3, 0.1, key="ces_phi")
            r = st.number_input("r: Capital rate ($/$/year)", 0.01, 0.50, 0.10, 0.01, key="ces_r",
                               help="Annual cost of capital as fraction")
            c_m = st.number_input("c_m: Material cost ($/unit)", 50.0, 1000.0, 300.0, 10.0, key="ces_cm")
            p = st.number_input("p: Selling price ($/unit)", 100.0, 2000.0, 550.0, 10.0, key="ces_p")
            
        with col2:
            st.markdown("#### Constraint & Fixed Costs")
            q_min = st.number_input("q_min: Minimum output (units/year)", 1000, 1000000, 100000, 1000, key="ces_qmin")
            
            st.markdown("#### Fixed Costs (USD/year)")
            # Defaults calculated from typical dashboard values (after AI scenario):
            # labeling: w=30 * tau=0.005 * n_labels=12000 * phi=1.3 = 2340
            # storage: cTB_yr=276 * size_tb=5 = 1380
            # ETL: alpha_yr=480 * size_tb=5 = 2400
            # MLOps: beta_ops_yr=8400 * n_models=3 = 25200
            # annualized_capex: capex=360000 / useful_life=7 = 51429
            # residual_risk: L_breach_yr=280000 * exp(-eta*security_after) with eta=log(2)/100000, security_after=150000 ≈ 99000
            labeling_cost = st.number_input("Labeling cost ($/year)", 0.0, 1000000.0, 2340.0, 100.0, key="ces_labeling")
            storage_cost = st.number_input("Storage cost ($/year)", 0.0, 1000000.0, 1380.0, 100.0, key="ces_storage")
            etl_cost = st.number_input("ETL cost ($/year)", 0.0, 1000000.0, 2400.0, 100.0, key="ces_etl")
            mlops_cost = st.number_input("MLOps cost ($/year)", 0.0, 1000000.0, 25200.0, 1000.0, key="ces_mlops")
            annualized_capex = st.number_input("Annualized CAPEX ($/year)", 0.0, 1000000.0, 51429.0, 1000.0, key="ces_capex")
            residual_risk_postAI = st.number_input("Residual risk post-AI ($/year)", 0.0, 1000000.0, 99000.0, 1000.0, key="ces_risk")
    
    C_fixed = labeling_cost + storage_cost + etl_cost + mlops_cost + annualized_capex + residual_risk_postAI
    
    # Compute optimization
    st.markdown("---")
    st.markdown("### 📊 Optimization Results")
    
    result = compute_ces_optimum(A, alpha, rho, w, phi, r, p, c_m, q_min, C_fixed)
    L_opt, K_opt, q_opt, profit_opt, lambda_opt, success, residuals = result
    
    if success and L_opt is not None:
        # Results cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Optimal Labor", f"{L_opt:,.0f} hours/year")
        with col2:
            st.metric("Optimal Capital", f"${K_opt:,.0f}")
        with col3:
            st.metric("Output", f"{q_opt:,.0f} units/year")
        with col4:
            st.metric("Profit", f"${profit_opt:,.0f}/year")
        
        st.markdown("---")
        col_lambda = st.columns(1)[0]
        with col_lambda:
            st.metric("Shadow Price λ", f"${lambda_opt:,.2f} USD/unit",
                     help="Marginal value of relaxing the minimum output constraint by one unit")
        
        # Sanity checks
        st.markdown("---")
        st.markdown("### ✅ Sanity Checks")
        
        tolerance = 1.0  # units
        constraint_ok = abs(q_opt - q_min) < tolerance
        foc_max = max(residuals.get('foc_L', 1e10), residuals.get('foc_K', 1e10))
        foc_ok = foc_max < 1.0  # reasonable threshold
        kl_ratio = K_opt / L_opt if L_opt > 0 else 0
        kl_ok = 10 <= kl_ratio <= 500
        
        check_col1, check_col2, check_col3 = st.columns(3)
        
        with check_col1:
            if constraint_ok:
                st.success(f"✅ Constraint satisfied: |q - q_min| = {abs(q_opt - q_min):.2f} < {tolerance}")
            else:
                st.error(f"❌ Constraint violation: |q - q_min| = {abs(q_opt - q_min):.2f} >= {tolerance}")
        
        with check_col2:
            if foc_ok:
                st.success(f"✅ First-order conditions: max residual = {foc_max:.2e}")
            else:
                st.warning(f"⚠️ First-order conditions: max residual = {foc_max:.2e}")
        
        with check_col3:
            if kl_ok:
                st.success(f"✅ K/L ratio: ${kl_ratio:.2f} per labor hour (plausible)")
            else:
                st.warning(f"⚠️ K/L ratio: ${kl_ratio:.2f} per labor hour (outside 10-500 range)")
        
        # Sensitivity analysis
        st.markdown("---")
        st.markdown("### 📈 Sensitivity Analysis")
        st.caption("Adjust parameters ±20% and see impact on Profit and Shadow Price")
        
        sens_col1, sens_col2, sens_col3 = st.columns(3)
        
        with sens_col1:
            q_min_sens = st.slider("q_min variation (%)", -20, 20, 0, 1, key="sens_qmin")
            q_min_new = q_min * (1 + q_min_sens / 100)
        with sens_col2:
            w_sens = st.slider("w (wage) variation (%)", -20, 20, 0, 1, key="sens_w")
            w_new = w * (1 + w_sens / 100)
        with sens_col3:
            c_m_sens = st.slider("c_m (material cost) variation (%)", -20, 20, 0, 1, key="sens_cm")
            c_m_new = c_m * (1 + c_m_sens / 100)
        
        # Recompute with sensitivity parameters
        result_sens = compute_ces_optimum(A, alpha, rho, w_new, phi, r, p, c_m_new, q_min_new, C_fixed)
        L_sens, K_sens, q_sens, profit_sens, lambda_sens, success_sens, _ = result_sens
        
        if success_sens and profit_sens is not None:
            profit_delta = profit_sens - profit_opt
            lambda_delta = lambda_sens - lambda_opt
            
            sens_result_col1, sens_result_col2 = st.columns(2)
            with sens_result_col1:
                st.metric("Profit Change", f"${profit_delta:,.0f}", 
                         f"{profit_delta/profit_opt*100:.1f}%" if profit_opt != 0 else "N/A")
            with sens_result_col2:
                st.metric("Shadow Price Change", f"${lambda_delta:,.2f}",
                         f"{lambda_delta/lambda_opt*100:.1f}%" if lambda_opt != 0 else "N/A")
        else:
            st.warning("⚠️ Sensitivity analysis failed. Check parameter ranges.")
        
        # Visualizations
        st.markdown("---")
        st.markdown("### 📈 Visualizations")
        
        # Chart 1: Production Isoquant and Optimal Point
        st.markdown("#### Production Function: Isoquant & Optimal Point")
        L_range = np.linspace(L_opt * 0.5, L_opt * 2.0, 50)
        K_range = np.linspace(K_opt * 0.5, K_opt * 2.0, 50)
        L_grid, K_grid = np.meshgrid(L_range, K_range)
        q_grid = np.zeros_like(L_grid)
        for i in range(L_grid.shape[0]):
            for j in range(L_grid.shape[1]):
                q_grid[i, j] = ces_production(L_grid[i, j], K_grid[i, j], A, alpha, rho)
        
        # Find isoquant for q_min
        isoquant_L = []
        isoquant_K = []
        for L_val in L_range:
            # Solve for K such that q(L, K) = q_min
            try:
                def iso_eqn(K_val):
                    return ces_production(L_val, K_val, A, alpha, rho) - q_min
                K_iso = fsolve(iso_eqn, K_opt)[0]
                if K_iso > 0 and K_iso < K_range.max() * 1.1:  # Allow slightly beyond range
                    isoquant_L.append(L_val)
                    isoquant_K.append(K_iso)
            except:
                pass
        
        fig_isoquant = go.Figure()
        
        # Add isoquant line
        if len(isoquant_L) > 0:
            fig_isoquant.add_trace(go.Scatter(
                x=isoquant_L, y=isoquant_K,
                mode='lines',
                name=f'Isoquant (q = {q_min:,.0f})',
                line=dict(color='#0066cc', width=2, dash='dash'),
                fill='tonexty' if len(isoquant_L) > 1 else None
            ))
        
        # Add optimal point
        fig_isoquant.add_trace(go.Scatter(
            x=[L_opt], y=[K_opt],
            mode='markers+text',
            name='Optimal Point',
            marker=dict(size=15, color='#d62728', symbol='star'),
            text=[f'Optimal<br>L={L_opt:,.0f}<br>K=${K_opt:,.0f}'],
            textposition='top center',
            showlegend=True
        ))
        
        fig_isoquant.update_layout(
            title='Production Isoquant & Optimal Labor-Capital Mix',
            xaxis_title='Labor (hours/year)',
            yaxis_title='Capital ($)',
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_isoquant, use_container_width=True)
        
        # Chart 2: Profit vs q_min (constraint sensitivity)
        st.markdown("#### Profit Sensitivity to Minimum Output Constraint")
        q_min_range = np.linspace(q_min * 0.5, q_min * 1.5, 30)
        profit_range = []
        lambda_range = []
        for qm in q_min_range:
            try:
                res = compute_ces_optimum(A, alpha, rho, w, phi, r, p, c_m, qm, C_fixed)
                if res[5]:  # success
                    profit_range.append(res[3])
                    lambda_range.append(res[4])
                else:
                    profit_range.append(None)
                    lambda_range.append(None)
            except:
                profit_range.append(None)
                lambda_range.append(None)
        
        fig_constraint = go.Figure()
        fig_constraint.add_trace(go.Scatter(
            x=q_min_range, y=profit_range,
            mode='lines+markers',
            name='Profit',
            line=dict(color='#00a86b', width=3),
            marker=dict(size=6)
        ))
        fig_constraint.add_trace(go.Scatter(
            x=[q_min], y=[profit_opt],
            mode='markers',
            name='Current Optimum',
            marker=dict(size=15, color='#d62728', symbol='star'),
            showlegend=True
        ))
        fig_constraint.add_vline(x=q_min, line_dash="dash", line_color="gray", 
                                annotation_text=f"Current q_min = {q_min:,.0f}")
        
        fig_constraint.update_layout(
            title='Profit vs Minimum Output Constraint',
            xaxis_title='Minimum Output (units/year)',
            yaxis_title='Profit ($/year)',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_constraint, use_container_width=True)
        
        # Chart 3: Cost Breakdown
        st.markdown("#### Cost Breakdown at Optimal Point")
        revenue = p * q_opt
        labor_cost = w * phi * L_opt
        capital_cost = r * K_opt
        material_cost = c_m * q_opt
        
        cost_categories = ['Labor', 'Capital', 'Material', 'Fixed Costs']
        cost_values = [labor_cost, capital_cost, material_cost, C_fixed]
        cost_colors = ['#0066cc', '#00a86b', '#F66733', '#9b59b6']
        
        fig_costs = go.Figure()
        fig_costs.add_trace(go.Bar(
            x=cost_categories,
            y=cost_values,
            marker_color=cost_colors,
            text=[f'${v:,.0f}' for v in cost_values],
            textposition='outside',
            name='Costs'
        ))
        fig_costs.update_layout(
            title=f'Cost Breakdown at Optimal Point (Total Revenue: ${revenue:,.0f})',
            xaxis_title='Cost Category',
            yaxis_title='Cost ($/year)',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12),
            showlegend=False
        )
        st.plotly_chart(fig_costs, use_container_width=True)
        
        # Investment Payback Profile
        st.markdown("#### Investment Payback Profile")
        investment = 1000000  # $1M capex
        total_net_benefit = sum(net_benefits_values)
        roi = total_net_benefit / investment
        payback_years = investment / total_net_benefit if total_net_benefit > 0 else 0
        
        years = np.arange(0, 4)
        cumulative_cashflow = np.array([-investment] + [-investment + total_net_benefit * year for year in range(1, 4)])
        
        fig_payback = go.Figure()
        fig_payback.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        # Add fill for profit zone (above zero)
        profit_zone_x = []
        profit_zone_y = []
        loss_zone_x = []
        loss_zone_y = []
        for i, cf in enumerate(cumulative_cashflow):
            if cf >= 0:
                profit_zone_x.append(years[i])
                profit_zone_y.append(cf/1000)
            else:
                loss_zone_x.append(years[i])
                loss_zone_y.append(cf/1000)
        
        # Add profit zone fill
        if len(profit_zone_x) > 0:
            profit_zone_x_full = [years[0]] + profit_zone_x + [profit_zone_x[-1] if profit_zone_x else years[-1]]
            profit_zone_y_full = [0] + profit_zone_y + [0]
            fig_payback.add_trace(go.Scatter(
                x=profit_zone_x_full,
                y=profit_zone_y_full,
                mode='none',
                name='Profit Zone',
                fill='tozeroy',
                fillcolor='rgba(46, 204, 113, 0.3)',
                showlegend=True
            ))
        
        # Add loss zone fill
        if len(loss_zone_x) > 0:
            loss_zone_x_full = [years[0]] + loss_zone_x + [loss_zone_x[-1] if loss_zone_x else years[-1]]
            loss_zone_y_full = [0] + loss_zone_y + [0]
            fig_payback.add_trace(go.Scatter(
                x=loss_zone_x_full,
                y=loss_zone_y_full,
                mode='none',
                name='Investment Period',
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.3)',
                showlegend=True
            ))
        
        # Add main line
        fig_payback.add_trace(go.Scatter(
            x=years,
            y=cumulative_cashflow/1000,
            mode='lines+markers',
            name='Cumulative Cash Flow',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10)
        ))
        fig_payback.update_layout(
            title=f'Investment Payback Profile (ROI: {roi:.1f}x | Payback: {payback_years:.1f} years)',
            xaxis_title='Years',
            yaxis_title='Cumulative Cash Flow (Thousand $)',
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_payback, use_container_width=True)
        
        # Details accordion
        st.markdown("---")
        with st.expander("📖 Details: Equations & Interpretation", expanded=False):
            st.markdown("""
            #### First-Order Conditions (FOCs)
            
            The Lagrangian is:
            $$
            \\mathcal{L} = p \\cdot q - w \\cdot \\phi \\cdot L - r \\cdot K - c_m \\cdot q - C_{fixed} + \\lambda (q - q_{min})
            $$
            
            First-order conditions:
            $$
            \\frac{\\partial \\mathcal{L}}{\\partial L} = (p - c_m) \\frac{\\partial q}{\\partial L} - w \\phi + \\lambda \\frac{\\partial q}{\\partial L} = 0
            $$
            
            $$
            \\frac{\\partial \\mathcal{L}}{\\partial K} = (p - c_m) \\frac{\\partial q}{\\partial K} - r + \\lambda \\frac{\\partial q}{\\partial K} = 0
            $$
            
            $$
            \\frac{\\partial \\mathcal{L}}{\\partial \\lambda} = q - q_{min} \\geq 0, \\quad \\lambda \\geq 0, \\quad \\lambda (q - q_{min}) = 0
            $$
            
            #### Shadow Price Interpretation
            
            The shadow price $\\lambda$ represents the marginal value of relaxing the minimum output constraint by one unit.
            
            - **If $\\lambda > 0$**: The constraint is binding. Increasing $q_{min}$ by one unit would reduce profit by approximately $\\lambda$ dollars.
            - **If $\\lambda < 0$**: This indicates the fixed costs ($C_{fixed}$) create a burden at the constrained scale. The firm would prefer to produce less than $q_{min}$ if unconstrained, but the constraint forces higher output, reducing profitability.
            - **If $\\lambda \\approx 0$**: The constraint is not binding; the unconstrained optimum already exceeds $q_{min}$.
            
            When $\\lambda < 0$, it suggests that the fixed-cost structure (labeling, storage, ETL, MLOps, CAPEX, residual risk) makes it unprofitable to operate at the minimum output level, but the constraint forces production anyway.
            """)
    else:
        st.error("❌ Optimization failed. Please check input parameters and try again.")
        if 'error' in residuals:
            st.error(f"Error details: {residuals['error']}")

# Footer - Enhanced Design (show on all pages 2-4)
if current_page in [2, 3, 4]:
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); padding: 1.5rem; border-radius: 14px; margin-top: 3rem; border: 1px solid rgba(0,102,204,0.08); box-shadow: 0 4px 16px rgba(0,0,0,0.04);">
        <p style="margin: 0; color: #94a3b8; font-size: 0.85rem; text-align: center; font-weight: 500;">
            <span class="clemson-color" style="font-weight: 700;">Clemson University</span> × <span class="bmw-color" style="font-weight: 700;">BMW</span> • AI Optimization in Manufacturing
        </p>
    </div>
    """, unsafe_allow_html=True)
