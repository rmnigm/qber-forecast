# qber-forecast
Small repository for qber forecasting microservice with catboost model.

## Currently ready
* CatBoostRegressor estimator
* Composite model (linear regression and LGBMRegressor)

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
