#!/usr/bin/env bash

export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__EXECUTOR="SequentialExecutor"
export AIRFLOW__CORE__AUTH_MANAGER="airflow.providers.fab.auth_manager.fab_auth_manager.FabAuthManager"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="sqlite:///$AIRFLOW_HOME/airflow.db"
