from time import sleep

from thrift.transport.TSocket import TSocket

from generated.chan_estimator_api import ChanEstimatorService
from generated.chan_estimator_api.ttypes import Status, Code
from thrift.transport.TTransport import TBufferedTransport, TTransportException
from thrift.protocol import TBinaryProtocol

import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import r2_score

THRIFT_CONNECT_RETRY_TIMEOUT = 5


class ExponentionalAverager:
    def __init__(self, start_value, window_size):
        assert window_size >= 0
        self.val = start_value
        self.alpha = 2. / (window_size + 1)
        self.window_size = window_size

    def add_value(self, val):
        self.val = self.alpha * val + (1. - self.alpha) * self.val

    def get_value(self):
        return self.val


def open_thrift_transport(thrift_transport):
    retries = 100
    while retries:
        try:
            thrift_transport.open()
            break
        except TTransportException:
            retries -= 1
            sleep(THRIFT_CONNECT_RETRY_TIMEOUT)
            continue
        except IOError:
            raise
    else:
        raise IOError("Could not connect to thrift server after 100 retries")


def main():
    data = pd.read_csv("/Users/rmnigm/Downloads/frames_errors_12-03.csv", header=None)
    error_time_series = list(data[2].dropna())
    ema = ExponentionalAverager(0.01, 5)
    
    transp = TBufferedTransport(TSocket('0.0.0.0', 8080))
    open_thrift_transport(transp)
    client = ChanEstimatorService.Client(TBinaryProtocol.TBinaryProtocol(transp))
    
    predictions = [0.01]
    true_values = []
    for qber in tqdm(error_time_series):
        ema.add_value(qber)
        qber_ema = ema.get_value()
        est = client.retrieveEst(qber, qber_ema, 0.02, 0.02, 0.05, 0.003, 0.0004, 0.00005, False)
        if est.res == Code.OK:
            predictions.append(est.est)
            true_values.append(qber)

    true_values = true_values[25000:]
    predictions = predictions[25000:-1]
    np.save("predictions.npy", predictions)
    print(r2_score(true_values, predictions))


if __name__ == '__main__':
    main()
