#!/usr/bin/python3

import argparse
import dataclasses
from enum import Enum
from queue import Queue, Empty as QueueEmpty
from threading import Thread

from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport
from thrift.transport.TSocket import TServerSocket
from thrift.server import TServer

from estimator import Estimator
from generated.chan_estimator_api import ChanEstimatorService
from generated.chan_estimator_api.ttypes import Status, Code


class ServerModel(Enum):
    SINGLE_THREAD = 1
    MULTI_THREAD = 2


@dataclasses.dataclass
class EstRequest:
    eMu: float
    eMuX: float
    eMuEma: float
    eNu1: float
    eNu2: float
    qMu: float
    qNu1: float
    qNu2: float
    maintenance: bool


class CBM:
    def __init__(self, path):
        self.is_active = False
        self.is_ready = False
        self.thread = None
        self.refit_count = 0
        self.req_queue = Queue()
        self.resp_queue = Queue()
        self.refit_threshold = 25000
        self.estimator = Estimator(7, self.refit_threshold)
        self.estimator.load_model(path)

    def retrieve_est(self, eMu, eMuX, eMuEma, eNu1, eNu2, qMu, qNu1, qNu2, maintenance):
        req = EstRequest(eMu, eMuX, eMuEma, eNu1, eNu2, qMu, qNu1, qNu2, maintenance)
        self.req_queue.put(req)
        resp = None
        max_tries = 5
        for i in range(max_tries):
            try:
                resp = self.resp_queue.get(block=True, timeout=1)
                break
            except QueueEmpty:
                resp = Status(Code.INTERNAL_ERROR, 0.)
        return resp

    def start(self):
        self.is_active = True
        self.thread = Thread(target=self.loop)
        print("Starting model prediction thread")
        self.thread.start()

    def stop(self):
        if not self.is_active:
            raise RuntimeError("E: model thread is stopped already")
        self.is_active = False
        self.thread.join()

    def ready(self):
        return self.is_ready

    def refit(self):
        self.is_ready = False
        self.refit_count += 1
        self.estimator.refit()
        self.is_ready = True

    def loop(self):
        self.is_ready = True
        iteration = 0
        while self.is_active:
            refit_required = iteration > self.refit_threshold
            req = None
            try:
                req = self.req_queue.get(True, 1)
            except QueueEmpty:
                continue
            self.estimator.update(req.eMu, req.eMuEma)
            est = self.estimator.predict()
            self.resp_queue.put(Status(res=Code.OK, est=est))
            iteration += 1
            if refit_required:
                self.refit()
                iteration = 0


class ChanEstimatorHandler:
    def __init__(self, est_model):
        self.model = est_model

    def retrieveEst(self, *args, **kwargs):
        if self.model.ready():
            return self.model.retrieve_est(*args, **kwargs)
        else:
            return Status(Code.TRY_LATER, 0.)


def main():
    parser = argparse.ArgumentParser(description='ML-based channel estimator')
    parser.add_argument("--model",
                        default='simple', required=False,
                        help='chan_estimator thread model')
    args = parser.parse_args()
    
    if args.model == "simple":
        server_model = ServerModel.SINGLE_THREAD
    elif args.model == "multi":
        server_model = ServerModel.MULTI_THREAD
    else:
        raise RuntimeError("E: invalid server model obtained")

    est_model = CBM("./model.cbm")
    est_model.start()

    handler = ChanEstimatorHandler(est_model)
    processor = ChanEstimatorService.Processor(handler)
    transport = TServerSocket('0.0.0.0', 8080)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = None
    if server_model == ServerModel.SINGLE_THREAD:
        server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    elif server_model == ServerModel.MULTI_THREAD:
        server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)

    thrift_server_thread = Thread(target=server.serve, name='Thrift server')
    print("Starting Thrift server thread")
    thrift_server_thread.start()


if __name__ == '__main__':
    main()
