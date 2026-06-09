from prediction.arima import ARIMA_ORDER as ARIMA_ORDER
from prediction.arima import predict_prices as predict_arima
from prediction.prophet_model import predict_prices_prophet


def predict_prices(gold, model="arima", steps=12):
    if model == "prophet":
        return predict_prices_prophet(gold, steps)
    return predict_arima(gold, steps)
