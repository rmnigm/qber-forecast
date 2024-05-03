# qber-forecast
Small repository for forecasting model microservices with thrift, designed for QBER estimation experiments.

## Currently ready
* Periodically refitting CatBoost on lagged values
* CatBoostRegressor with window statistics
* Composite model (exponental smoothing and LGBM on residuals)
* Torch models (in progress)

## Setup
1. Install Docker
2. Clone the repository to local machine
```bash
$ git clone <repo>
$ cd <repo>
```
3. Build the docker image and run it!
```bash
$ cd <service>
$ docker compose up --build
```
