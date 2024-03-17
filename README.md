# qber-forecast
Small repository for forecasting model microservices with thrift, designed for QBER estimation experiments.

## Currently ready
* CatBoostRegressor estimator
* Composite model (exponental smoothing and LGBMRegressor)
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
