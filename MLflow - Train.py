# Databricks notebook source
# MAGIC %md ## Training and Loading Models with Databricks Hosted Tracking Server
# MAGIC 
# MAGIC This demo walks you through the day in the life as you work with designing your models.  Training the models, and loading the models analysis.
# MAGIC 
# MAGIC We’ll:
# MAGIC * Load/Prep data
# MAGIC * Train a model using linear regression on a diabetes dataset using ElasticNet - recorded in MLFlow
# MAGIC * Load the model from MLFlow into other tools (sklearn and pyspark UDF)
# MAGIC 
# MAGIC This notebook uses the `diabetes` dataset in scikit-learn and predicts the progression metric (a quantitative measure of disease progression after one year after) based on BMI, blood pressure, etc. It uses the scikit-learn ElasticNet linear regression model, where we vary the `alpha` and `l1_ratio` parameters for tuning. For more information on ElasticNet, refer to:
# MAGIC   * [Elastic net regularization](https://en.wikipedia.org/wiki/Elastic_net_regularization)
# MAGIC   * [Regularization and Variable Selection via the Elastic Net](https://web.stanford.edu/~hastie/TALKS/enet_talk.pdf)
# MAGIC 
# MAGIC To get started, you will first need to 
# MAGIC 
# MAGIC 1. Be part of the Databricks Hosted MLflow early adopter program and have MLflow tracking server enabled on your shard.
# MAGIC 2. Install the most recent version of MLflow and Python ML and math libraries on your Databricks cluster (see details in the next cell).

# COMMAND ----------

# MAGIC %md ### Install MLflow on Your Databricks Cluster
# MAGIC 
# MAGIC ####NOTE: ML Runtime 5.0 beta or newer is required!
# MAGIC  
# MAGIC  
# MAGIC 1. Ensure you are using or [create a cluster](https://docs.databricks.com/user-guide/clusters/create.html) specifying 
# MAGIC   * **Databricks Runtime Version:** Databricks Runtime "5.0 beta" 
# MAGIC   * **Python Version:** Python 3
# MAGIC 2. Install `mlflow` as a [PiPy library](https://docs.databricks.com/user-guide/libraries.html#upload-a-python-pypi-package-or-python-egg).
# MAGIC   1. Choose **PyPi** and enter `mlflow==0.8.0`.
# MAGIC 3. For our ElasticNet Descent Path visualizations, install the latest `scikit-learn` and `matplotlib` as a PyPI libraries.
# MAGIC   1. Choose **PyPi** and enter `scikit-learn==0.19.1`
# MAGIC   1. Choose **PyPi** and enter `matplotlib==2.2.2`

# COMMAND ----------

# MAGIC %md #### Write Your ML Code Based on the`train_diabetes.py` Code
# MAGIC This tutorial is based on the MLflow's [train_diabetes.py](https://github.com/databricks/mlflow/blob/master/example/tutorial/train_diabetes.py), which uses the `sklearn.diabetes` built-in dataset to predict disease progression based on various factors.

# COMMAND ----------

dbutils.widgets.text("experiment_id", "", "Experiment ID")

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

# Import various libraries including matplotlib, sklearn, mlflow
import os
import warnings
import sys

import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

# Import mlflow
import mlflow
import mlflow.sklearn


# COMMAND ----------

# MAGIC %md #### Load external data i.e. Diabetes dataset from sklearn

# COMMAND ----------

# Load Diabetes datasets
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Create pandas DataFrame for sklearn ElasticNet linear_model
Y = np.array([y]).transpose()
d = np.concatenate((X, Y), axis=1)
cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
data = pd.DataFrame(d, columns=cols)

# COMMAND ----------

# MAGIC %md #### Organize MLflow Runs into Experiments
# MAGIC 
# MAGIC As you start using MLFlow for more tasks, it will make things easier to navigate your runs by grouping them into experiments.  I've already created this in advance.

# COMMAND ----------

# MAGIC %md #### Plot the ElasticNet Descent Path
# MAGIC As an example of recording arbitrary output files in MLflow, we'll plot the [ElasticNet Descent Path](http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html) for the ElasticNet model by *alpha* for the specified *l1_ratio*.
# MAGIC 
# MAGIC The `plot_enet_descent_path` function below:
# MAGIC * Returns an image that can be displayed in our Databricks notebook via `display`
# MAGIC * As well as saves the figure `ElasticNet-paths.png` to the Databricks cluster's driver node
# MAGIC * This file is then uploaded to MLflow using the `log_artifact` within `train_diabetes`

# COMMAND ----------

def plot_enet_descent_path(X, y, l1_ratio):
    # Compute paths
    eps = 5e-3  # the smaller it is the longer is the path

    # Reference the global image variable
    global image
    
    print("Computing regularization path using the elastic net.")
    alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=l1_ratio, fit_intercept=False)

    # Display results
    fig = plt.figure(1)
    ax = plt.gca()

    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_alphas_enet = -np.log10(alphas_enet)
    for coef_e, c in zip(coefs_enet, colors):
        l1 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)

    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    title = 'ElasticNet Path by alpha for l1_ratio = ' + str(l1_ratio)
    plt.title(title)
    plt.axis('tight')

    # Display images
    image = fig
    
    # Save figure
    fig.savefig("ElasticNet-paths.png")

    # Close plot
    plt.close(fig)

    # Return images
    return image    

# COMMAND ----------

# MAGIC %md #### Train the Diabetes Model
# MAGIC The next function trains Elastic-Net linear regression based on the input parameters of `alpha (in_alpha)` and `l1_ratio (in_l1_ratio)`.
# MAGIC 
# MAGIC In addition, this function uses MLflow Tracking to record its
# MAGIC * parameters
# MAGIC * metrics
# MAGIC * model
# MAGIC * arbitrary files, namely the above noted Lasso Descent Path plot.
# MAGIC 
# MAGIC **Tip:** We use `with mlflow.start_run:` in the Python code to create a new MLflow run. This is the recommended way to use MLflow in notebook cells. Whether your code completes or exits with an error, the `with` context will make sure that we close the MLflow run, so you don't have to call `mlflow.end_run` later in the code.

# COMMAND ----------

# train_diabetes
#   Uses the sklearn Diabetes dataset to predict diabetes progression using ElasticNet
#       The predicted "progression" column is a quantitative measure of disease progression one year after baseline
#       http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
#
#   Returns: The MLflow RunInfo associated with this training run, see
#            https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunInfo
#            We will use this later in the notebook to demonstrate ways to access the output of this
#            run and do useful things with it!
def train_diabetes(data, in_alpha, in_l1_ratio):
  # Evaluate metrics
  def eval_metrics(actual, pred):
      rmse = np.sqrt(mean_squared_error(actual, pred))
      mae = mean_absolute_error(actual, pred)
      r2 = r2_score(actual, pred)
      return rmse, mae, r2

  warnings.filterwarnings("ignore")
  np.random.seed(40)

  # Split the data into training and test sets. (0.75, 0.25) split.
  train, test = train_test_split(data)

  # The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
  train_x = train.drop(["progression"], axis=1)
  test_x = test.drop(["progression"], axis=1)
  train_y = train[["progression"]]
  test_y = test[["progression"]]

  if float(in_alpha) is None:
    alpha = 0.05
  else:
    alpha = float(in_alpha)
    
  if float(in_l1_ratio) is None:
    l1_ratio = 0.05
  else:
    l1_ratio = float(in_l1_ratio)
  
  # Start an MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
  with mlflow.start_run() as run:
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # Print out ElasticNet model metrics
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # Set tracking_URI first and then reset it back to not specifying port
    # Note, we had specified this in an earlier cell
    #mlflow.set_tracking_uri(mlflow_tracking_URI)

    # Log mlflow attributes for mlflow UI
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(lr, "model")
    
    # Call plot_enet_descent_path
    image = plot_enet_descent_path(X, y, l1_ratio)
    
    # Log artifacts (output files)
    mlflow.log_artifact("ElasticNet-paths.png")
    
    print("Inside MLflow Run with id %s" % run.info.run_uuid)
    
    # return our RunUUID so we can use it when we try out some other APIs later in this notebook.
    return run.info

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://docs.databricks.com/_static/images/mlflow/elasticnet-paths-by-alpha-per-l1-ratio.png)

# COMMAND ----------

# MAGIC %md #### Experiment with Different Parameters
# MAGIC 
# MAGIC Now that we have a `train_diabetes` function that records MLflow runs, we can simply call it with different parameters to explore them. Later, we'll be able to visualize all these runs on our MLflow tracking server.

# COMMAND ----------

# Start with alpha and l1_ratio values of 0.01, 0.01
run_info_1 = train_diabetes(data, 0.01, 0.01)

# COMMAND ----------

display(image)

# COMMAND ----------

# Start with alpha and l1_ratio values of 0.01, 0.75
run_info_2 = train_diabetes(data, 0.01, 0.75)

# COMMAND ----------

display(image)

# COMMAND ----------

# Start with alpha and l1_ratio values of 0.01, 1
run_info_3 = train_diabetes(data, 0.01, 1)

# COMMAND ----------

display(image)

# COMMAND ----------

# MAGIC %md ## Review the MLflow UI
# MAGIC Visit your tracking server in a web browser by going to `https://your_shard_id.cloud.databricks.com/mlflow`

# COMMAND ----------

# MAGIC %md
# MAGIC The MLflow UI should look something similar to the animated GIF below. Inside the UI, you can:
# MAGIC * View your experiments and runs
# MAGIC * Review the parameters and metrics on each run
# MAGIC * Click each run for a detailed view to see the the model, images, and other artifacts produced.
# MAGIC 
# MAGIC <img src="https://docs.databricks.com/_static/images/mlflow/mlflow-ui.gif"/>

# COMMAND ----------

# MAGIC %md ## Load MLflow model back as a Scikit-learn model
# MAGIC Here we demonstrate using the MLflow API to load model from the MLflow server that was produced by a given run. To do so we have to specify the run_id.
# MAGIC 
# MAGIC Once we load it back in, it is a just a scikitlearn model object like any other and we can explore it or use it.

# COMMAND ----------

import mlflow.sklearn
model = mlflow.sklearn.load_model("runs:/" + run_info_1.run_uuid + "/model")
model.coef_

# COMMAND ----------

#Get a prediction for a row of the dataset
model.predict(data[0:1].drop(["progression"], axis=1))

# COMMAND ----------

# MAGIC %md ## Use an MLflow Model for Batch inference
# MAGIC We can also get a pyspark UDF to do some batch inference using one of the models you logged above. For more on this see https://mlflow.org/docs/latest/models.html#apache-spark

# COMMAND ----------

# First let's create a Spark DataFrame out of our original pandas
# DataFrame minus the column we want to predict. We'll use this
# to simulate what this would be like if we had a big data set
# that was regularly getting updated that we were routinely wanting
# to score, e.g. click logs.
dataframe = spark.createDataFrame(data.drop(["progression"], axis=1))

# COMMAND ----------

# Next we use the MLflow API to create a PySpark UDF given our run.
# See the API docs for this function call here:
# https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf
# the spark_udf function takes our SparkSession, the path to the model within artifact
# repository, and the ID of the run that produced this model.
pyfunc_udf = mlflow.pyfunc.spark_udf(spark, "runs:/" + run_info_1.run_uuid + "/model")

# COMMAND ----------

#withColumns adds a column to the data by applying the python UDF to the DataFrame
predicted_df = dataframe.withColumn("prediction", pyfunc_udf(
  'age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'))
display(predicted_df)

# COMMAND ----------

# MAGIC %md ## Summary
# MAGIC 
# MAGIC We walked through:
# MAGIC * Hosted Tracking Server
# MAGIC * Trained a linear regression model using ElasictNet - recorded to MLFlow
# MAGIC * Load the a model recorded in MLFlow to sklearn - ran a prediction
# MAGIC * Load the a different model record in MFlow flow into a Spark UDF - ran batch inference

# COMMAND ----------

dbutils.notebook.exit(0)

# COMMAND ----------

# MAGIC %md ## Model Serving Example

# COMMAND ----------

import os
import requests
import pandas as pd

def score_model(dataset: pd.DataFrame):
  url = 'https://eastus2.azuredatabricks.net/model/rgdemo/1/invocations'
  headers = {'Authorization': f'Bearer dapie4bb8df1411b77ea3a0eb632aed2ab59'}
  data_json = dataset.to_dict(orient='split')
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

score_model(data[0:1].drop(["progression"], axis=1))

# COMMAND ----------


