
import numpy as np
import pandas as pd
from collections import deque
from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation.forecasting import check_y_X, check_fh

from pgmpy.models import (
    DiscreteBayesianNetwork,
    LinearGaussianBayesianNetwork,
)
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator, HillClimbSearch
from pgmpy.inference.ExactInference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from scipy.stats import norm

__all__ = ["PgmpyDBNForecaster"]


class PgmpyDBNForecaster(BaseForecaster):
    """
    Flexible DBN forecaster (sktime + pgmpy), supporting:
      • Discrete DBN forecasting (MLE, Bayesian)
      • Causal / counterfactual forecasting via do() for discrete models
      • Automated structure learning for both discrete and continuous models
      • Continuous DBN forecasting via manual forward inference
    """

    _tags = {
        "scitype:y": "univariate",
        "requires-y": True,
        "capability:pred_int": True,
        "handles-missing-data": False,
        "requires-fh-in-fit": True,
        "python_dependencies": ["pgmpy", "scipy"],
    }

    def __init__(
        self,
        structure: list | None = None,
        *,
        max_lag: int = 1,
        continuous: bool = False,
        state_names: dict | None = None,
        auto_structure: bool = False,
        estimator=MaximumLikelihoodEstimator,
        estimator_kwargs: dict | None = None,
    ):
        super().__init__()
        self.structure = structure
        self.max_lag = max_lag
        self.continuous = continuous
        self.state_names = state_names
        self.auto_structure = auto_structure
        self.estimator = estimator
        self.estimator_kwargs = estimator_kwargs or {}
        # This attribute is used for Gaussian intervals
        self._pred_variances = None

    # THIS METHOD HAS BEEN MOVED OUT OF __init__ TO THE CORRECT CLASS LEVEL
    def _gaussian_forward_inference(self, node: str, evidence: dict) -> tuple[float, float]:
        """
        Manually compute conditional mean & variance of `node` given `evidence`
        using stored LinearGaussianCPD objects. This replaces GaussianInference.
        """
        cpd = self.dbn_.get_cpds(node)  # This is a LinearGaussianCPD object
        
        # Correctly unpack the beta array for this pgmpy version.
        # The first element is the intercept, the rest are parent coefficients.
        intercept = cpd.beta[0]
        parent_betas = cpd.beta[1:]
        
        # Calculate the mean using the linear regression formula: μ = β₀ + Σ(βᵢ * parentᵢ)
        mean = intercept + sum(
            c * evidence[par] for c, par in zip(parent_betas, cpd.evidence)
        )
        
        # The variance is the residual variance of the regression
        variance = cpd.std ** 2

        
        return mean, variance

    def _fit(self, y: pd.Series, X: pd.DataFrame = None, fh=None):
        """Build DBN, fit CPDs or regression, and prepare for forecasting."""
        y, X = check_y_X(y, X)
        self._y_name = y.name

        dtype = float if self.continuous else int
        df = pd.DataFrame(y.astype(dtype))
        if X is not None:
            df = pd.concat([df, X.astype(dtype)], axis=1)
        lagged = self._create_time_lags(df, self.max_lag)

        if not self.continuous:
            lagged = lagged.astype("category")

        if self.auto_structure:
            score_type = "bic-g" if self.continuous else "aic-d"
            hc = HillClimbSearch(lagged)
            kwargs = {} if self.auto_structure is True else self.auto_structure
            best_model = hc.estimate(scoring_method=score_type, **kwargs)
            edges = list(best_model.edges())
        elif self.structure is not None:
            edges = [(f"{u[0]}_{u[1]}", f"{v[0]}_{v[1]}") for u, v in self.structure]
        else:
            raise ValueError("Either `structure` must be provided or `auto_structure` must be True.")

        if self.continuous:
            self.dbn_ = LinearGaussianBayesianNetwork(edges)
            self.dbn_.fit(lagged)
            self.inference_ = None
        else:
            self.dbn_ = DiscreteBayesianNetwork(edges)
            pgmpy_state_names = {}
            if self.state_names:
                for var_name, states in self.state_names.items():
                    for lag in range(self.max_lag + 1):
                        pgmpy_state_names[f"{var_name}_{lag}"] = states
            
            # FIX: Instantiate the estimator with ONLY the arguments it expects.
            estimator_instance = self.estimator(
                model=self.dbn_, data=lagged,
                state_names=pgmpy_state_names
            )
            
            # FIX: Pass the specific estimator kwargs to the .get_parameters() method.
            cpds = estimator_instance.get_parameters(**self.estimator_kwargs)
            self.dbn_.add_cpds(*cpds)
            if not self.dbn_.check_model():
                raise RuntimeError("The fitted pgmpy model is invalid after manual fitting.")
            self.inference_ = VariableElimination(self.dbn_)

        self.is_fitted_ = True
        return self

    def _create_time_lags(self, df: pd.DataFrame, L: int) -> pd.DataFrame:
        """
        Return DataFrame with columns 'var_0'...'var_L'.
        IMPORTANT: `var_0` is current time t, `var_1` is t-1, etc.
        """
        dtype = float if self.continuous else int
        cols = {}
        for lag in range(L + 1):
            sh = df.shift(lag)
            for c in df.columns:
                cols[f"{c}_{lag}"] = sh[c]
        lagged = pd.DataFrame(cols).dropna().astype(dtype)
        return lagged

    def _predict(self, fh, X: pd.DataFrame = None):
            """Iteratively forecast point estimates."""
            self.check_is_fitted()
            fh = check_fh(fh)
            abs_idx = fh.to_absolute(self.cutoff).to_pandas()
            
            last_y = self._y.iloc[-1]
            last_X = self._X.iloc[-1] if self._X is not None else pd.Series(dtype=object)
            
            preds, variances = [], []
            nodes = set(self.dbn_.nodes())

            for t_abs in fh.to_absolute(self.cutoff):
                evidence = {f"{self._y_name}_1": last_y}
                for c, v in last_X.items():
                    evidence[f"{c}_1"] = v

                if X is not None and t_abs in X.index:
                    for c in X.columns:
                        evidence[f"{c}_0"] = X.loc[t_abs, c]
                
                evidence = {k: v for k, v in evidence.items() if k in nodes}
                node_to_predict = f"{self._y_name}_0"

                if self.continuous:
                    mean, var = self._gaussian_forward_inference(node_to_predict, evidence)
                    val = mean
                    variances.append(var)
                else:
                    res = self.inference_.query([node_to_predict], evidence=evidence, show_progress=False)
                    max_prob_idx = np.argmax(res.values)
                    val = int(res.state_names[node_to_predict][max_prob_idx])
                
                preds.append(val)
                
                last_y = val
                if X is not None and t_abs in X.index:
                    last_X = X.loc[t_abs]

            self._pred_variances = variances
            return pd.Series(preds, index=abs_idx, name=self._y_name)

    def predict_causal(self, fh, X=None, intervention=None):
        """Counterfactual forecast under a constant intervention."""
        if self.continuous:
            raise NotImplementedError("Causal `do()` operator is not supported for LinearGaussianBayesianNetwork in this version of pgmpy.")
            
        if intervention is None:
            return self.predict(fh=fh, X=X)
            
        fh = check_fh(fh)
        abs_idx = fh.to_absolute(self.cutoff).to_pandas()
        history_y = deque(self._y.iloc[-self.max_lag:].astype(int),
                          maxlen=self.max_lag)

        preds = []
        for t_abs in fh.to_absolute(self.cutoff):
            # 1) copy & do()-intervene
            intervened = self.dbn_.copy()
            nodes = set(intervened.nodes())
            for key, val in intervention.items():
                var, lag = (key if isinstance(key, tuple) else (key, 0))
                node = f"{var}_{lag}"
                if node in nodes:
                    intervened.do({node: val})

            infer = VariableElimination(intervened)
            nodes = set(intervened.nodes())

            # 2) build evidence ONLY for nodes that exist
            evidence = {}
            # lagged y's
            for lag_i, past_val in enumerate(reversed(history_y), start=1):
                node = f"{self._y_name}_{lag_i}"
                if node in nodes:
                    # if you intervened on this lag, use that; else use actual history
                    evidence[node] = intervention.get((self._y_name, lag_i), past_val)

            # any X you might have (same pattern) 
            if X is not None and t_abs in X.index:
                for c in X.columns:
                    node = f"{c}_0"
                    if node in nodes and (c, 0) not in intervention:
                        evidence[node] = X.loc[t_abs, c]

            # 3) query safely
            dist = infer.query(
                [f"{self._y_name}_0"],
                evidence=evidence,
                show_progress=False,
            )

            # 4) extract the mode & roll forward
            mode = dist.state_names[self._y_name + "_0"][dist.values.argmax()]
            preds.append(int(mode))
            history_y.append(int(mode))

        return pd.Series(preds, index=abs_idx, name=self._y_name)


    def _predict_interval(self, fh, X=None, coverage: float = 0.90):
        """Return probabilistic prediction intervals."""
        # The logic in _predict is general, so this call works for all lags.
        y_pred = self._predict(fh=fh, X=X)

        if isinstance(coverage, float):
            coverages = [coverage]
        else:
            coverages = coverage

        var_name = self._y_name

        if self.continuous:
            pred_int = pd.DataFrame()
            std_devs = np.sqrt(self._pred_variances)
            for cov in coverages:
                z_score = norm.ppf(0.5 + cov / 2.0)
                lower = y_pred - z_score * std_devs
                upper = y_pred + z_score * std_devs
                pred_int[(var_name, cov, "lower")] = lower
                pred_int[(var_name, cov, "upper")] = upper
            pred_int.columns = pd.MultiIndex.from_tuples(pred_int.columns, names=["variable", "coverage", "bound"])
            return pred_int

        # Discrete logic needs the same robust history mechanism.
        pred_int_discrete = {}
        for cov in coverages:
            history_y = deque(self._y.iloc[-self.max_lag:].astype(int), maxlen=self.max_lag)
            history_X = self._X.iloc[-self.max_lag:] if self._X is not None else None
            lower_bounds, upper_bounds = [], []
            nodes = set(self.dbn_.nodes())

            for t_abs in fh.to_absolute(self.cutoff):
                evidence = {}
                for i, val_hist in enumerate(history_y, 1):
                    evidence[f"{var_name}_{i}"] = val_hist
                if history_X is not None:
                    for lag_idx, hist_row in enumerate(history_X.itertuples(), 1):
                        for col_name, val_hist in hist_row._asdict().items():
                            if col_name != "Index": evidence[f"{col_name}_{lag_idx}"] = val_hist
                if X is not None and t_abs in X.index:
                    for c in X.columns: evidence[f"{c}_0"] = X.loc[t_abs, c]
                
                evidence = {k: v for k, v in evidence.items() if k in nodes}
                node_to_predict = f"{var_name}_0"
                dist = self.inference_.query([node_to_predict], evidence=evidence, show_progress=False)

                states = dist.state_names[node_to_predict]
                probs = dist.values
                mode_idx = np.argmax(probs)
                prob_sum = probs[mode_idx]
                interval_indices = {mode_idx}
                l_ptr, r_ptr = mode_idx - 1, mode_idx + 1
                while prob_sum < cov and (l_ptr >= 0 or r_ptr < len(probs)):
                    l_prob = probs[l_ptr] if l_ptr >= 0 else -1
                    r_prob = probs[r_ptr] if r_ptr < len(probs) else -1
                    if l_prob >= r_prob:
                        prob_sum += l_prob; interval_indices.add(l_ptr); l_ptr -= 1
                    else:
                        prob_sum += r_prob; interval_indices.add(r_ptr); r_ptr += 1
                
                interval_states = [states[i] for i in sorted(interval_indices)]
                lower_bounds.append(min(interval_states))
                upper_bounds.append(max(interval_states))

                history_y.append(y_pred.loc[t_abs])
                if X is not None and t_abs in X.index:   # Only update the X history if future X was provided for this step.
                    if history_X is not None:
                        new_X_row = X.loc[[t_abs]]
                        history_X = pd.concat([history_X.iloc[1:], new_X_row])
            
            pred_int_discrete[(var_name, cov, "lower")] = pd.Series(lower_bounds, index=y_pred.index)
            pred_int_discrete[(var_name, cov, "upper")] = pd.Series(upper_bounds, index=y_pred.index)
            
        pred_int = pd.DataFrame(pred_int_discrete)
        pred_int.columns = pd.MultiIndex.from_tuples(pred_int.columns, names=["variable", "coverage", "bound"])
        return pred_int