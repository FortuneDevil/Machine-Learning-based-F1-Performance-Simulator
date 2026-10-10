"""
reference_model.py

Shared definition of the dataset-grounded reference aerodynamic model.

The class lives in its own importable module so that the Phase-7 pickle
records it as ``reference_model.ReferenceAerodynamicModel``. Phases 7, 8 and 9
all import this single definition instead of keeping private copies.

Interpolation method
--------------------
- Standardize all six continuous input dimensions.
- Find the K nearest observations.
- Fit a distance-weighted local polynomial (degree 1 or 2) through those
  neighbours, centred on the query point.
- The fitted intercept is the reference response at the query.

A local polynomial fit reproduces linear (degree 1) or quadratic (degree 2)
response surfaces exactly. A plain inverse-distance weighted average does
not: it is biased wherever the response has a slope, which made the earlier
reference less accurate than the ML surrogate it was meant to benchmark.

Queries outside the observed input envelope are rejected by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# Use a small epsilon to detect exact matches with dataset points.
DISTANCE_EPSILON = 1e-12

# Tolerance applied to the observed min/max when checking the envelope.
BOUNDARY_TOLERANCE = 1e-9


def polynomial_design_matrix(
    dx: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Design matrix [1, dx, (dx_i * dx_j)] for a centred local fit."""
    columns = [np.ones(len(dx))]
    columns.extend(dx.T)

    if degree == 2:
        n_features = dx.shape[1]
        for i in range(n_features):
            for j in range(i, n_features):
                columns.append(dx[:, i] * dx[:, j])

    return np.column_stack(columns)


class ReferenceAerodynamicModel:
    """
    Dataset-grounded local-polynomial interpolator for aerodynamic responses.

    Parameters
    ----------
    X:
        Full dataset input matrix.
    y:
        Full dataset target matrix.
    input_columns:
        Names of continuous input variables.
    target_columns:
        Names of continuous targets.
    k_neighbors:
        Number of neighbours used for each local fit.
    degree:
        Degree of the local polynomial (1 = linear, 2 = quadratic).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        input_columns: list[str],
        target_columns: list[str],
        k_neighbors: int,
        degree: int,
    ) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 2:
            raise ValueError("y must be a 2D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")

        if X.shape[1] != len(input_columns):
            raise ValueError(
                "Number of X columns does not match input_columns."
            )

        if y.shape[1] != len(target_columns):
            raise ValueError(
                "Number of y columns does not match target_columns."
            )

        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values.")

        if not np.isfinite(y).all():
            raise ValueError("y contains non-finite values.")

        if degree not in (1, 2):
            raise ValueError("degree must be 1 or 2.")

        n_coefficients = polynomial_design_matrix(
            np.zeros((1, X.shape[1])),
            degree,
        ).shape[1]

        if k_neighbors <= n_coefficients:
            raise ValueError(
                "k_neighbors must exceed the number of local polynomial "
                f"coefficients ({n_coefficients})."
            )

        if k_neighbors >= len(X):
            raise ValueError(
                "k_neighbors must be smaller than the number of observations."
            )

        self.X_raw = X
        self.y = y
        self.input_columns = list(input_columns)
        self.target_columns = list(target_columns)
        self.k_neighbors = int(k_neighbors)
        self.degree = int(degree)

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X_raw)

        self.neighbor_search = NearestNeighbors(
            n_neighbors=self.k_neighbors,
            algorithm="auto",
            metric="euclidean",
            n_jobs=1,
        )
        self.neighbor_search.fit(self.X_scaled)

        self.feature_min = self.X_raw.min(axis=0)
        self.feature_max = self.X_raw.max(axis=0)

    # -----------------------------------------------------------------
    # Input handling
    # -----------------------------------------------------------------

    def _prepare_query(
        self,
        X_query: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Convert a query to an array with correct feature ordering."""
        if isinstance(X_query, pd.DataFrame):
            missing = [
                col
                for col in self.input_columns
                if col not in X_query.columns
            ]

            if missing:
                raise ValueError(
                    "Reference query is missing columns: "
                    + ", ".join(missing)
                )

            query = X_query[self.input_columns].to_numpy(dtype=float)
        else:
            query = np.asarray(X_query, dtype=float)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.ndim != 2:
            raise ValueError("Reference query must be a 1D or 2D array.")

        if query.shape[1] != len(self.input_columns):
            raise ValueError(
                "Reference query has the wrong number of features. "
                f"Expected {len(self.input_columns)}."
            )

        if not np.isfinite(query).all():
            raise ValueError("Reference query contains non-finite values.")

        return query

    def check_bounds(
        self,
        X_query: pd.DataFrame | np.ndarray,
    ) -> pd.DataFrame:
        """
        Check whether each query lies inside the observed input envelope.

        Returns a row-level report. This does not raise an exception.
        """
        query = self._prepare_query(X_query)
        rows = []

        for index, row in enumerate(query):
            below = row < (self.feature_min - BOUNDARY_TOLERANCE)
            above = row > (self.feature_max + BOUNDARY_TOLERANCE)

            outside_columns = [
                self.input_columns[i]
                for i in range(len(self.input_columns))
                if below[i] or above[i]
            ]

            rows.append(
                {
                    "query_index": index,
                    "inside_observed_envelope": len(outside_columns) == 0,
                    "outside_columns": ", ".join(outside_columns),
                }
            )

        return pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Local fit
    # -----------------------------------------------------------------

    def _local_fit(
        self,
        query_scaled: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_distances: np.ndarray,
    ) -> np.ndarray:
        """
        Weighted local polynomial fit for one query.

        Neighbours are weighted with a tricube kernel whose bandwidth is the
        distance to the farthest neighbour, so influence falls smoothly to
        zero at the edge of the neighbourhood.
        """
        bandwidth = float(neighbor_distances.max())

        if bandwidth <= DISTANCE_EPSILON:
            return self.y[neighbor_indices].mean(axis=0)

        # Slightly widen the bandwidth so the farthest neighbour keeps a
        # small positive weight.
        scaled_distance = neighbor_distances / (bandwidth * 1.05)
        weights = (1.0 - scaled_distance**3) ** 3
        sqrt_weights = np.sqrt(weights)

        design = polynomial_design_matrix(
            self.X_scaled[neighbor_indices] - query_scaled,
            self.degree,
        )

        coefficients, *_ = np.linalg.lstsq(
            design * sqrt_weights[:, None],
            self.y[neighbor_indices] * sqrt_weights[:, None],
            rcond=None,
        )

        # The fit is centred on the query, so the intercept is the response.
        return coefficients[0]

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------

    def predict(
        self,
        X_query: pd.DataFrame | np.ndarray,
        reject_outside: bool = True,
        return_distance: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Predict reference outputs using a local polynomial fit.

        Parameters
        ----------
        X_query:
            One or more continuous operating points.
        reject_outside:
            If True, reject points outside the observed six-dimensional
            feature envelope. Extrapolation is intentionally avoided.
        return_distance:
            If True, also return the distance to the nearest neighbour.
        """
        query = self._prepare_query(X_query)

        if reject_outside:
            bounds = self.check_bounds(query)
            invalid = ~bounds["inside_observed_envelope"].to_numpy()

            if invalid.any():
                bad_indices = np.where(invalid)[0]
                raise ValueError(
                    "Reference-model query contains points outside the "
                    "observed operating envelope. "
                    f"Indices: {bad_indices.tolist()}"
                )

        query_scaled = self.scaler.transform(query)

        distances, indices = self.neighbor_search.kneighbors(
            query_scaled,
            n_neighbors=self.k_neighbors,
            return_distance=True,
        )

        predictions = np.empty(
            (len(query), self.y.shape[1]),
            dtype=float,
        )

        for i in range(len(query)):
            # If the query exactly matches a dataset point, return that
            # observation exactly.
            if distances[i, 0] <= DISTANCE_EPSILON:
                predictions[i] = self.y[indices[i, 0]]
                continue

            predictions[i] = self._local_fit(
                query_scaled[i],
                indices[i],
                distances[i],
            )

        if return_distance:
            return predictions, distances[:, 0]

        return predictions

    def predict_dataframe(
        self,
        X_query: pd.DataFrame,
        reject_outside: bool = True,
    ) -> pd.DataFrame:
        """Predict and return named target columns."""
        predictions = self.predict(
            X_query,
            reject_outside=reject_outside,
        )

        return pd.DataFrame(
            predictions,
            columns=self.target_columns,
            index=X_query.index,
        )

    def predict_leave_one_out(
        self,
        row_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct stored observations from their neighbours only.

        Each requested dataset row is removed from its own neighbourhood
        before the local fit, so the result measures genuine interpolation
        quality instead of trivially returning the stored response.

        Returns
        -------
        predictions:
            Array of shape (n_rows, n_targets).
        nearest_nonself_distance:
            Distance from each row to its nearest other observation.
        """
        row_indices = np.asarray(row_indices, dtype=int)
        query_scaled = self.X_scaled[row_indices]

        # One extra neighbour because the first match is the row itself.
        distances, indices = self.neighbor_search.kneighbors(
            query_scaled,
            n_neighbors=self.k_neighbors + 1,
            return_distance=True,
        )

        predictions = np.empty(
            (len(row_indices), self.y.shape[1]),
            dtype=float,
        )
        nearest_nonself_distance = np.empty(len(row_indices), dtype=float)

        for i, row_index in enumerate(row_indices):
            nonself = indices[i] != row_index

            d = distances[i][nonself][: self.k_neighbors]
            idx = indices[i][nonself][: self.k_neighbors]

            predictions[i] = self._local_fit(
                query_scaled[i],
                idx,
                d,
            )
            nearest_nonself_distance[i] = d[0]

        return predictions, nearest_nonself_distance
