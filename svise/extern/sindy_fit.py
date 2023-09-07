import pysindy
import numpy as np
from typing import List
from sklearn.exceptions import ConvergenceWarning
import warnings


# tvregdiff is quite slow and unstable
# method = pysindy.SINDyDerivative(
#     kind="trend_filtered",
#     order=0,
#     alpha=1e-2,
#     normalize=True,
#     max_iter=100000,
#     tol=1e-2,
# )


class SparsePolynomialSINDySTLSQ:
    """
    Fits a sparse polynomial model using STLSQ and derivative approximation.

    Thresholding is determined using 5-fold cross validation.
    """

    def __init__(
        self,
        d: int,
        degree: int,
        train_t: np.ndarray,
        train_x: np.ndarray,
        input_labels: List = None,
    ) -> None:
        if input_labels is None:
            input_labels = ["x{}".format(i) for i in range(d)]
        feature_library = pysindy.feature_library.PolynomialLibrary(degree=degree)
        # works well for the relatively smooth problems in this work
        diff_method = pysindy.SINDyDerivative(
            kind="savitzky_golay", left=0.5, right=0.5, order=3
        )
        # selecting threshold using 5-fold cross validation
        threshold_list = [1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
        eval_error = {threshold: [] for threshold in threshold_list}
        n_folds = 5
        idx = np.arange(len(train_t))
        np.random.shuffle(idx)
        n = len(train_t) // n_folds
        split_idx = [idx[n * i : n * (i + 1)] for i in range(n_folds)]
        for threshold in threshold_list:
            for split in split_idx:
                train_idx = np.array(list(set(idx) - set(split)))
                t = train_t[train_idx]
                x = train_x[train_idx]
                optimizer = pysindy.STLSQ(threshold=threshold)
                model = pysindy.SINDy(
                    feature_names=input_labels,
                    feature_library=feature_library,
                    differentiation_method=diff_method,
                    optimizer=optimizer,
                )
                model.fit(x, t=t)
                t_eval = train_t[split]
                x_eval = train_x[split]
                sort_idx = np.argsort(t_eval)
                # the higher the better
                eval_loss = -model.score(x_eval[sort_idx], t=t_eval[sort_idx])
                eval_error[threshold].append(eval_loss)
            eval_error[threshold] = np.array(eval_error[threshold]).mean()
        best_threshold = threshold_list[0]
        eval_best = eval_error[best_threshold]
        for threshold in threshold_list:
            if eval_error[threshold] < eval_best:
                best_threshold = threshold
                eval_best = eval_error[threshold]
        self.threshold = best_threshold
        optimizer = pysindy.STLSQ(threshold=self.threshold)
        self.model = pysindy.SINDy(
            feature_names=input_labels,
            feature_library=feature_library,
            differentiation_method=diff_method,
            optimizer=optimizer,
        )
        self.model.fit(train_x, t=train_t)

    @property
    def W(self):
        return self.model.coefficients()

    def __str__(self) -> str:
        return "; ".join(self.model.equations())


class SparsePolynomialSINDySR3:
    """
    Fits a sparse polynomial model using SR3 and derivative approximation.

    Thresholding is determined using 5-fold cross validation.
    """

    def __init__(
        self,
        d: int,
        degree: int,
        train_t: np.ndarray,
        train_x: np.ndarray,
        input_labels: List = None,
    ) -> None:
        if input_labels is None:
            input_labels = ["x{}".format(i) for i in range(d)]
        feature_library = pysindy.feature_library.PolynomialLibrary(degree=degree)
        # works well for the relatively smooth problems in this work
        diff_method = pysindy.SINDyDerivative(
            kind="savitzky_golay", left=0.5, right=0.5, order=3
        )
        # selecting threshold using 5-fold cross validation
        threshold_list = [1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
        nu = [1e-3, 1e-2, 1e-1, 1e-0, 1e1]
        hyperparams = [(nu_i, thresh_i) for nu_i in nu for thresh_i in threshold_list]
        eval_error = {hpm: [] for hpm in hyperparams}
        n_folds = 5
        idx = np.arange(len(train_t))
        np.random.shuffle(idx)
        n = len(train_t) // n_folds
        split_idx = [idx[n * i : n * (i + 1)] for i in range(n_folds)]
        for hpm in hyperparams:
            nu_i, thresh_i = hpm
            for split in split_idx:
                train_idx = np.array(list(set(idx) - set(split)))
                t = train_t[train_idx]
                x = train_x[train_idx]
                optimizer = pysindy.SR3(
                    threshold=thresh_i, nu=nu_i, thresholder="L0", max_iter=1000
                )
                model = pysindy.SINDy(
                    feature_names=input_labels,
                    feature_library=feature_library,
                    differentiation_method=diff_method,
                    optimizer=optimizer,
                )
                model.fit(x, t=t)
                t_eval = train_t[split]
                x_eval = train_x[split]
                sort_idx = np.argsort(t_eval)
                # the higher the better
                eval_loss = -model.score(x_eval[sort_idx], t=t_eval[sort_idx])
                eval_error[hpm].append(eval_loss)
            eval_error[hpm] = np.array(eval_error[hpm]).mean()
        best_hpm = hyperparams[0]
        eval_best = eval_error[best_hpm]
        for hpm in hyperparams:
            if eval_error[hpm] < eval_best:
                best_hpm = hpm
                eval_best = eval_error[hpm]
        self.nu, self.threshold = best_hpm
        optimizer = pysindy.SR3(
            threshold=self.threshold, nu=self.nu, thresholder="L0", max_iter=1000
        )
        model = pysindy.SINDy(
            feature_names=input_labels,
            feature_library=feature_library,
            differentiation_method=diff_method,
            optimizer=optimizer,
        )
        model.fit(train_x, t=train_t)
        self._W = model.coefficients()

    @property
    def W(self):
        return self._W

class EnsembleSINDy:
    """
    Fits a sparse polynomial model using SR3 and derivative approximation.

    Thresholding is determined using 5-fold cross validation.
    """

    def __init__(
        self,
        d: int,
        degree: int,
        train_t: np.ndarray,
        train_x: np.ndarray,
        input_labels: List = None,
    ) -> None:
        if input_labels is None:
            input_labels = ["x{}".format(i) for i in range(d)]
        feature_library = pysindy.feature_library.PolynomialLibrary(degree=degree)
        # works well for the relatively smooth problems in this work
        diff_method = pysindy.SINDyDerivative(
            kind="savitzky_golay", left=0.5, right=0.5, order=3
        )
        # selecting threshold using 5-fold cross validation
        threshold_list = [1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
        eval_error = {threshold: [] for threshold in threshold_list}
        n_folds = 5
        idx = np.arange(len(train_t))
        np.random.shuffle(idx)
        n = len(train_t) // n_folds
        split_idx = [idx[n * i : n * (i + 1)] for i in range(n_folds)]
        for threshold in threshold_list:
            for split in split_idx:
                train_idx = np.array(list(set(idx) - set(split)))
                t = train_t[train_idx]
                x = train_x[train_idx]
                ensemble_optimizer = pysindy.EnsembleOptimizer(pysindy.STLSQ(threshold=threshold),bagging=True,n_subset=int(0.6*x.shape[0]))
                model = pysindy.SINDy(
                    feature_names=input_labels,
                    feature_library=feature_library,
                    differentiation_method=diff_method,
                    optimizer=ensemble_optimizer,
                )
                model.fit(x, t=t)
                t_eval = train_t[split]
                x_eval = train_x[split]
                sort_idx = np.argsort(t_eval)
                # the higher the better
                eval_loss = -model.score(x_eval[sort_idx], t=t_eval[sort_idx])
                eval_error[threshold].append(eval_loss)
            eval_error[threshold] = np.array(eval_error[threshold]).mean()
        best_threshold = threshold_list[0]
        eval_best = eval_error[best_threshold]
        for threshold in threshold_list:
            if eval_error[threshold] < eval_best:
                best_threshold = threshold
                eval_best = eval_error[threshold]
        self.threshold = best_threshold
        ensemble_optimizer = pysindy.EnsembleOptimizer(pysindy.STLSQ(threshold=self.threshold),bagging=True,n_subset=int(0.6*train_x.shape[0]))
        model = pysindy.SINDy(
            feature_names=input_labels,
            feature_library=feature_library,
            differentiation_method=diff_method,
            optimizer=ensemble_optimizer,
        )
        model.fit(train_x, t=train_t)
        self._W = np.median(np.stack(ensemble_optimizer.coef_list),axis=0)

    @property
    def W(self):
        return self._W
