# AI Customer Churn Prediction & MLOps Platform

A production-style Machine Learning system that predicts customer churn, serves predictions through a FastAPI REST API, tracks experiments with MLflow, runs inside Docker, and monitors prediction behavior and data drift.

## 🚀 Project Overview

This project demonstrates an end-to-end MLOps workflow:

Dataset → Preprocessing → Model Training → MLflow → Best Model → FastAPI → Docker → Dashboard → Monitoring → Drift Detection

## 🏗️ Architecture

```text
Customer Churn Dataset
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
ML Model Training
        ↓
MLflow Experiment Tracking
        ↓
Best Model Selection
        ↓
FastAPI REST API
        ↓
Docker Container
        ↓
Web Dashboard
        ↓
Prediction Monitoring
        ↓
Data Drift Detection
