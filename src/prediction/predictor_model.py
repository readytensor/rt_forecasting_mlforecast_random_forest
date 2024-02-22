import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Iterable, Union
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError
from mlforecast import MLForecast
from sklearn.ensemble import RandomForestRegressor
from mlforecast.target_transforms import LocalMinMaxScaler
from logger import get_logger

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"
logger = get_logger(task_name="model_training")


class Forecaster:
    """A wrapper class for the Random Forest Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "Random Forest Forecaster"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        history_forecast_ratio: int = None,
        lags_forecast_ratio: Union[int, float] = None,
        lags: Optional[Iterable] = None,
        n_estimators: int = 100,
        criterion: str = "squared_error",
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: int = 1,
        use_exogenous: bool = True,
        random_state: int = 0,
        **kwargs,
    ):
        """Construct a new Random Forest Forecaster

        Args:

            data_schema (ForecastingSchema):
                Schema of training data.

            history_forecast_ratio (int):
                Sets the history length depending on the forecast horizon.
                For example, if the forecast horizon is 20 and the history_forecast_ratio is 10,
                history length will be 20*10 = 200 samples.

            lags_forecast_ratio (int):
                Sets the lags parameters depending on the forecast horizon.
                lags = forecast horizon * lags_forecast_ratio
                This parameters overides lags parameter and uses the most recent values as lags.

            lags (Optional[Iterable]): Lags of the target to use as features.

            n_estimators (int): The number of trees in the forest.

            criterion (str): {“squared_error”, “absolute_error”, “friedman_mse”, “poisson”},
                The function to measure the quality of a split. Supported criteria are “squared_error” for the mean squared error,
                which is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each terminal node,
                “friedman_mse”, which uses mean squared error with Friedman’s improvement score for potential splits,
                “absolute_error” for the mean absolute error, which minimizes the L1 loss using the median of each terminal node,
                and “poisson” which uses reduction in Poisson deviance to find splits.
                Training using “absolute_error” is significantly slower than when using “squared_error”.

            min_samples_split (Union[int, float]): The minimum number of samples required to split an internal node:
                If int, then consider min_samples_split as the minimum number.
                If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.


            min_samples_leaf (Union[int, float]): The minimum number of samples required to be at a leaf node.
                A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
                If int, then consider min_samples_leaf as the minimum number.
                If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

            use_exogenous (bool): If true, uses covariates in training.

            random_state (int): Sets the underlying random seed at model initialization time.

            **kwargs (dict): Additional parameters accepted by the forecaster.
        """
        self.data_schema = data_schema
        self.lags = lags
        self.use_exogenous = use_exogenous
        self.random_state = random_state
        self._is_trained = False
        self.kwargs = kwargs
        self.history_length = None

        if history_forecast_ratio:
            self.history_length = (
                self.data_schema.forecast_length * history_forecast_ratio
            )

        if lags_forecast_ratio:
            lags = int(lags_forecast_ratio * self.data_schema.forecast_length)
            self.lags = [i for i in range(1, lags + 1)]

        self.models = [
            RandomForestRegressor(
                n_estimators=n_estimators,
                criterion=criterion,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                **kwargs,
            )
        ]

    def map_frequency(self, frequency: str) -> str:
        """
        Maps the frequency in the data schema to the frequency expected by mlforecast.

        Args:
            frequency (str): The frequency from the schema.

        Returns (str): The mapped frequency.
        """
        if self.data_schema.time_col_dtype == "INT":
            return 1

        frequency = frequency.lower()
        frequency = frequency.split("frequency.")[1]
        if frequency == "yearly":
            return "Y"
        if frequency == "quarterly":
            return "Q"
        if frequency == "monthly":
            return "M"
        if frequency == "weekly":
            return "W"
        if frequency == "daily":
            return "D"
        if frequency == "hourly":
            return "H"
        if frequency == "minutely":
            return "min"
        if frequency == "secondly":
            return "S"
        else:
            return 1

    def _validate_lags_and_history_length(self, series_length: int):
        """
        Validate the value of lags and that history length is at least double the forecast horizon.
        If the provided lags value is invalid (too large), lags are set to the largest possible value.

        Args:
            series_length (int): The length of the history.

        Returns: None
        """
        forecast_length = self.data_schema.forecast_length
        if series_length < 2 * forecast_length:
            raise ValueError(
                f"Training series is too short. History should be at least double the forecast horizon. history_length = ({series_length}), forecast horizon = ({forecast_length})"
            )

        if self.lags[-1] >= series_length:
            self.lags = [i for i in range(1, series_length)]
            logger.warning(
                f"The provided lags value >= available history length. Lags are set to to (history length - 1) = {series_length-1}"
            )

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the training data by dropping past covariates, converting the index to datetime if available
        and drops or keeps other covariates depending on use_exogenous.

            Args:
                data (pd.DataFrame): The training data.
        """
        data.drop(columns=self.data_schema.past_covariates, inplace=True)

        if self.data_schema.time_col_dtype in ["DATE", "DATETIME"]:
            data[self.data_schema.time_col] = pd.to_datetime(
                data[self.data_schema.time_col]
            )

        groups_by_ids = data.groupby(self.data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())

        all_series = [groups_by_ids.get_group(id_).reset_index() for id_ in all_ids]

        if self.history_length:
            for index, series in enumerate(all_series):
                all_series[index] = series.iloc[-self.history_length :]
            data = pd.concat(all_series).drop(columns="index")

        if not self.use_exogenous:
            if self.data_schema.future_covariates:
                data.drop(columns=self.data_schema.future_covariates, inplace=True)

            if self.data_schema.static_covariates:
                data.drop(columns=self.data_schema.static_covariates, inplace=True)

        return data

    def fit(
        self,
        history: pd.DataFrame,
    ) -> None:
        """Fit the Forecaster to the training data.

        Args:
            history (pandas.DataFrame): The features of the training data.

        """
        np.random.seed(self.random_state)

        if self.use_exogenous and len(self.data_schema.static_covariates) > 0:
            static_features = self.data_schema.static_covariates
        else:
            static_features = []

        history = self.prepare_data(history)

        series_length = (
            history.groupby(self.data_schema.id_col)[self.data_schema.target]
            .count()
            .iloc[0]
        )

        self._validate_lags_and_history_length(series_length=series_length)

        if self.lags[-1] > len(history):
            self.lags = [i for i in range(1, len(history))]
            logger.warning(
                f"The provided lags value is greater than the available history length. Lags are set to to history length = {len(history)}"
            )

        self.model = MLForecast(
            models=self.models,
            freq=self.map_frequency(self.data_schema.frequency),
            lags=self.lags,
            target_transforms=[LocalMinMaxScaler()],
        )

        self.model.fit(
            df=history,
            time_col=self.data_schema.time_col,
            id_col=self.data_schema.id_col,
            target_col=self.data_schema.target,
            static_features=static_features,
        )
        self._is_trained = True

    def predict(
        self, test_data: pd.DataFrame, prediction_col_name: str
    ) -> pd.DataFrame:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            pd.DataFrame: The prediction dataframe.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        if self.use_exogenous and self.data_schema.future_covariates:
            future_df = self.model.make_future_dataframe(
                self.data_schema.forecast_length
            )
            future_df[self.data_schema.future_covariates] = test_data[
                self.data_schema.future_covariates
            ]
        else:
            future_df = None

        forecast = self.model.predict(self.data_schema.forecast_length, X_df=future_df)
        forecast[prediction_col_name] = forecast.drop(
            columns=[self.data_schema.time_col, self.data_schema.id_col]
        ).mean(axis=1)
        forecast[self.data_schema.time_col] = test_data[self.data_schema.time_col]
        return forecast

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """

    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(
        history=history,
    )
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
