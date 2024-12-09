# Programming Assignment 2: Wine Quality Prediction in Spark over AWS

## Overview
This project involves building a machine learning (ML) application for predicting wine quality. The task utilizes **Apache Spark's MLlib** for training and validation, and **Docker** for containerizing the application for deployment.

### Key Features:
1. Parallel model training using **4 EC2 instances** in AWS.
2. Model evaluation and prediction on a **single EC2 instance**.
3. Containerization of the prediction application using **Docker**.

---

## How to Set Up the Cloud Environment
1. **Launch EC2 Instances**:
   - Set up 4 Ubuntu Linux EC2 instances for training.
   - Set up 1 additional Ubuntu Linux EC2 instance for prediction.

2. **Install Apache Spark**:
   - Download and install Apache Spark on all instances.
   - Set up a Spark cluster for parallel computation.

3. **Transfer Code and Data**:
   - Clone the project repository from GitHub onto all instances.
   - Upload `TrainingDataset.csv` and `ValidationDataset.csv` to the training instances.

4. **Prepare Docker** (on prediction instance):
   - Install Docker and verify installation.

---

## How to Run the Application
### Parallel Model Training
1. SSH into one of the training instances and start the Spark master node.
2. SSH into the other instances and configure them as Spark workers.
3. Run the training script:
   ```bash
   spark-submit --master spark://<master-node-ip>:7077 src/TrainModel.py
