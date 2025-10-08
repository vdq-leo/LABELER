import numpy as np
import pandas as pd

class Labeler:
    """
    Labels upward and downward momentum (or trend) movements where:
    - price movement amplitude exceeds a threshold
    - movement establishes a new high or low within a maximum inactivity period
    - movement is not invalidated by a reversal beyond the threshold
    """

    def __init__(self, amplitude_threshold: float, max_inactive_period: int):
        """
        :param amplitude_threshold: minimum movement required to define a trend (bps or %)
        :param max_inactive_period: maximum number of samples allowed without extending the trend
        """
        self.amplitude_threshold = float(amplitude_threshold)
        self.max_inactive_period = int(max_inactive_period)
        self.df = None

    # =====================================================
    def label(self, prices: pd.Series) -> pd.DataFrame:
        """
        Perform labeling on a price series.

        :param prices: pandas Series of prices or cumulative returns
        :return: DataFrame with 'price' and 'label'
        """
        prices = pd.Series(prices).astype(float).dropna().reset_index(drop=True)

        cumr = np.log(prices / prices.iloc[0]) * 1e4  # convert to cumulative basis points
        n = len(cumr)
        labels = np.zeros(n, dtype=float)

        self._pass1(cumr, labels)
        self._filter(cumr, labels)

        self.df = pd.DataFrame({
            'price': cumr,
            'label': labels
        })
        return self.df

    def _pass1(self, cumr: np.ndarray, labels: np.ndarray):
        """
        First-pass trend labeling using amplitude (minamp) and inactivity (Tinactive).

        Logic:
        - Detects new uptrends or downtrends when price moves beyond minamp from the last extreme.
        - Ends a trend if price goes flat too long (Tinactive samples) or reverses enough.
        """

        n = len(cumr)
        if n == 0:
            return

        # --- Initialize tracking variables ---
        start_idx = 0              # Start of current segment
        cursor = 0                 # Current position in the time series
        min_idx = max_idx = 0      # Indices of local min and max
        min_val = max_val = cumr[0]

        while cursor < n:
            v = cumr[cursor]  # current value

            # ==========================================================
            # 1️⃣ Direction Reversal Detection
            # ==========================================================

            # Case 1: Previously falling (min after max) → now rising enough
            if (max_val - min_val) >= self.amplitude_threshold  and min_idx > max_idx and (v - min_val) >= self.amplitude_threshold :
                # label the last downward move
                self._apply_label(labels, start_idx, max_idx - 1, 0.0)
                self._apply_label(labels, max_idx, min_idx, -1.0)
                # start new upward leg
                start_idx = min_idx
                max_idx = cursor
                max_val = v

            # Case 2: Previously rising (max after min) → now falling enough
            elif (max_val - min_val) >= self.amplitude_threshold  and max_idx > min_idx and (max_val - v) >= self.amplitude_threshold :
                # label the last upward move
                self._apply_label(labels, start_idx, min_idx - 1, 0.0)
                self._apply_label(labels, min_idx, max_idx, +1.0)
                # start new downward leg
                start_idx = max_idx
                min_idx = cursor
                min_val = v

            # ==========================================================
            # 2️⃣ Inactivity Handling
            # ==========================================================

            # (a) Uptrend but flat for too long
            elif max_idx > min_idx and (cursor - max_idx) >= self.max_inactive_period  and v <= max_val:
                if (max_val - min_val) >= self.amplitude_threshold :
                    # Up move confirmed, then flat
                    self._apply_label(labels, start_idx, min_idx - 1, 0.0)
                    self._apply_label(labels, min_idx, max_idx, +1.0)
                    self._apply_label(labels, max_idx + 1, cursor, 0.0)
                else:
                    # Weak move — just flat
                    self._apply_label(labels, start_idx, cursor, 0.0)

                start_idx = cursor
                min_idx = max_idx = cursor
                min_val = max_val = v

            # (b) Downtrend but flat for too long
            elif min_idx > max_idx and (cursor - min_idx) >= self.max_inactive_period  and v >= min_val:
                if (max_val - min_val) >= self.amplitude_threshold :
                    # Down move confirmed, then flat
                    self._apply_label(labels, start_idx, max_idx - 1, 0.0)
                    self._apply_label(labels, max_idx, min_idx, -1.0)
                    self._apply_label(labels, min_idx + 1, cursor, 0.0)
                else:
                    self._apply_label(labels, start_idx, cursor, 0.0)

                start_idx = cursor
                min_idx = max_idx = cursor
                min_val = max_val = v

            # ==========================================================
            # 3️⃣ Update Local Extremes
            # ==========================================================
            if v >= max_val:
                max_idx = cursor
                max_val = v
            if v <= min_val:
                min_idx = cursor
                min_val = v

            cursor += 1

        # ==========================================================
        # 4️⃣ Finalize labeling for end of series
        # ==========================================================
        if (max_val - min_val) >= self.amplitude_threshold  and min_idx > max_idx:
            self._apply_label(labels, start_idx, max_idx - 1, 0.0)
            self._apply_label(labels, max_idx, min_idx, -1.0)
            self._apply_label(labels, min_idx + 1, cursor - 1, 0.0)
        elif (max_val - min_val) >= self.amplitude_threshold  and max_idx > min_idx:
            self._apply_label(labels, start_idx, min_idx - 1, 0.0)
            self._apply_label(labels, min_idx, max_idx, +1.0)
            self._apply_label(labels, max_idx + 1, cursor - 1, 0.0)
        else:
            self._apply_label(labels, start_idx, cursor - 1, 0.0)

    def _filter(self, cumr: np.ndarray, labels: np.ndarray):
        """
        Refine the raw trend labels using OLS-based filtering.

        This step removes weak or inconsistent moves.
        For each contiguous region of +1 or -1 labels:
        - Fit OLS slopes forward and backward
        - Measure how strong the directional trend is
        - Keep only strong sections; set weak ones to 0 (neutral)

        Parameters
        ----------
        cumr : np.ndarray
            Cumulative returns (e.g., log(price) * 1e4)
        labels : np.ndarray
            Array of labels (+1 for uptrend, -1 for downtrend, 0 for neutral)
        """

        n = len(cumr)
        pos = 0  # current position in array

        while pos < n:
            direction = labels[pos]

            # Skip neutral regions
            if direction == 0.0:
                pos += 1
                continue

            # -----------------------------
            # Identify contiguous labeled region
            # -----------------------------
            start = pos
            end = pos
            while end < n and labels[end] == direction:
                end += 1
            end -= 1  # move back to last matching index

            # -----------------------------
            # Measure OLS slope strength
            # -----------------------------
            forward_strength, forward_max_idx = self._ols_distance(
                cumr, start, end, direction, forward=True
            )
            backward_strength, backward_max_idx = self._ols_distance(
                cumr, start, end, direction, forward=False
            )

            # -----------------------------
            # Remove weak or inconsistent regions
            # -----------------------------
            if forward_strength < self.amplitude_threshold  and backward_strength < self.amplitude_threshold :
                # Entire region too weak — reset to neutral
                self._apply_label(labels, start, end, 0.0)
            else:
                # Forward (start → forward_max_idx)
                if forward_strength >= self.amplitude_threshold :
                    self._apply_label(labels, start, forward_max_idx, direction)
                    self._apply_label(labels, forward_max_idx + 1, backward_max_idx - 1, 0.0)
                else:
                    self._apply_label(labels, start, backward_max_idx, 0.0)

                # Backward (backward_max_idx → end)
                if backward_strength >= self.amplitude_threshold :
                    self._apply_label(labels, backward_max_idx, end, direction)
                else:
                    neutral_start = max(backward_max_idx, forward_max_idx + 1)
                    self._apply_label(labels, neutral_start, end, 0.0)

            # Move to the next region
            pos = end + 1

    # =====================================================
    def _ols_distance(self, cumr: np.ndarray, start: int, end: int, direction: float, forward=True):
        """
        Estimate how far prices deviate from a local OLS trend within a region.

        Parameters
        ----------
        cumr : np.ndarray
            Cumulative return or price series (already log-transformed or normalized).
        start : int
            Starting index of the segment to evaluate.
        end : int
            Ending index of the segment to evaluate.
        direction : float
            +1 for upward trends, -1 for downward trends.
        forward : bool, optional
            If True, evaluate forward from start→end.
            If False, evaluate backward from end→start.

        Returns
        -------
        (max_distance, max_index)
            max_distance : float
                Maximum directional deviation in OLS slope.
            max_index : int
                Index where that maximum occurs.
        """
        max_distance = 0.0
        max_index = start if forward else end

        # Define iteration direction
        if forward:
            indices = range(start, end + 1)
        else:
            indices = range(end, start - 1, -1)

        # Track cumulative sums for OLS regression
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_xx = 0.0

        for i, idx in enumerate(indices):
            x = float(i)
            y = cumr[idx]

            # Incrementally update sums
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_xx += x * x

            # Need at least 2 points to fit OLS
            if i < 1:
                continue

            n = i + 1
            denom = (sum_xx - (sum_x ** 2) / n)
            if denom == 0:
                continue

            # Compute slope (beta) of OLS regression line
            beta = (sum_xy - (sum_x * sum_y) / n) / denom

            # Compute deviation along the trend direction
            distance = direction * beta * x if forward else -direction * beta * x

            # Track maximum deviation
            if distance > max_distance:
                max_distance = distance
                max_index = idx

        return max_distance, max_index

    # =====================================================
    @staticmethod
    def _apply_label(labels: np.ndarray, start_idx: int, end_idx: int, direction: float):
        """
        Assign a trend label to a contiguous segment of the labels array.

        Parameters:
        -----------
        labels : np.ndarray
            Array storing trend labels.
        start_idx : int
            Starting index of the segment.
        end_idx : int
            Ending index of the segment.
        direction : float
            Trend direction:
            +1 → uptrend
            -1 → downtrend
            0  → neutral / no trend

        Updates the labels array in-place.
        """
        if end_idx < start_idx:
            return  # nothing to label
        labels[start_idx:end_idx + 1] = direction
