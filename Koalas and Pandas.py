# Databricks notebook source
# MAGIC %md
# MAGIC ##<img src="https://databricks.com/wp-content/themes/databricks/assets/images/header_logo_2x.png" alt="logo" width="150"/> 
# MAGIC # Koalas
# MAGIC * https://github.com/databricks/koalas
# MAGIC * https://databricks.com/blog/2019/04/24/koalas-easy-transition-from-pandas-to-apache-spark.html
# MAGIC * https://databricks.com/session_eu19/koalas-pandas-on-apache-spark
# MAGIC * https://koalas.readthedocs.io/en/latest/index.html
# MAGIC 
# MAGIC ### Koalas vs Pandas: The Ultimate Showdown Demo 
# MAGIC 
# MAGIC #### In this Demo: 
# MAGIC * Basics of Koalas 
# MAGIC * Why we should use Koalas for Big Data with an example
# MAGIC 
# MAGIC #### Dataset: 
# MAGIC * https://www.gbif.org/dataset/8a863029-f435-446a-821e-275f4f641165
# MAGIC * dbfs:/ml-workshop-datasets/nature

# COMMAND ----------

# MAGIC %fs ls /ml-workshop-datasets/nature/

# COMMAND ----------

# MAGIC %fs ls /ml-workshop-datasets/nature/nature-data-nl.csv

# COMMAND ----------

# DBTITLE 1,Import Koalas
import pandas as pd
import numpy as np
import databricks.koalas as ks
from pyspark.sql import SparkSession

# COMMAND ----------

# DBTITLE 1,Creating a Koalas DataFrame by passing a dict of objects that can be converted to series-like.
kdf = ks.DataFrame(
    {'a': [1, 2, 3, 4, 5, 6],
     'b': [100, 200, 300, 400, 500, 600],
     'c': ["one", "two", "three", "four", "five", "six"]},
    index=[10, 20, 30, 40, 50, 60])

display(kdf)

# COMMAND ----------

# DBTITLE 1,Creating a pandas DataFrame
dates = pd.date_range('20130101', periods=6)
pdf = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

pdf

# COMMAND ----------

# DBTITLE 1,Now, this pandas DataFrame can be converted to a Koalas DataFrame
kdf = ks.from_pandas(pdf)

# COMMAND ----------

type(kdf)

# COMMAND ----------

# DBTITLE 1,It looks and behaves the same as a pandas DataFrame though
kdf

# COMMAND ----------

kdf.head()

# COMMAND ----------

# DBTITLE 1,Describe shows a quick statistic summary of your data
kdf.describe()

# COMMAND ----------

# DBTITLE 1,Getting Data in/out: CSV (other formats as well)
kdf.to_csv('foo.csv')
ks.read_csv('foo.csv').head(10)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Let's do Some Natural Science 
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSEZHbOBUglDoCu_k2x12lH6ZKA1fpVJx1wxFFt1fTkj2b-sd-r" width="420"/>
# MAGIC </div>
# MAGIC 
# MAGIC Let's dig in a little deeper with a real dataset. 
# MAGIC 
# MAGIC We will be looking at a dataset containing the occurrence of flora and fauna species (biodiversity) in the Netherlands on a 5 x 5 km scale. 
# MAGIC 
# MAGIC Data is available from Observation.org, Nature data from the Netherlands.
# MAGIC 
# MAGIC 
# MAGIC ### Agenda
# MAGIC * Load Data
# MAGIC * Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md ## Load the Data

# COMMAND ----------

# DBTITLE 1,Load full dataset using Pandas
#import pandas as pd
#import glob

#pandas_full_df = pd.concat([pd.read_csv(f) for f in glob.glob('/dbfs/koalas-demo-sais19eu/nature-data-nl.csv/*.csv')], ignore_index=True)
#pandas_full_df.head()

# NOTE: This cell takes 10 mins and gets an OOM error

# COMMAND ----------

# DBTITLE 1,Load sample of the dataset using Pandas
import pandas as pd

pandas_df = pd.read_csv("/dbfs//ml-workshop-datasets/nature/part.csv")
pandas_df.head()

# COMMAND ----------

# DBTITLE 1,The Pandas DataFrame contains 269,047 rows and 50 columns
pandas_df.shape

# COMMAND ----------

# MAGIC %fs ls dbfs:/ml-workshop-datasets/nature/nature-data-nl.csv

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# DBTITLE 1,Load full dataset using Koalas
from databricks import koalas as ks

ks.set_option('compute.default_index_type', 'distributed')

koalas_df = ks.read_csv("dbfs:/ml-workshop-datasets/nature/nature-data-nl.csv") 
koalas_df.head()

# COMMAND ----------

# DBTITLE 1,The Koalas DataFrame contains 26,859,363 rows and 50 columns
koalas_df.shape

# COMMAND ----------

# Grab the column names
koalas_df.columns

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Exploratory Data Analysis

# COMMAND ----------

# DBTITLE 1,Looking at the value counts for the kingdom column
koalas_df["kingdom"].value_counts(normalize=True)

# COMMAND ----------

# DBTITLE 1,Data visualization using pandas (sample of dataset)
import matplotlib.pyplot as plt

plt.clf()
display(pandas_df["kingdom"].value_counts(normalize=True).plot.bar(rot=25, title="Bar plot of kingdom column using Pandas DataFrame").figure)

# COMMAND ----------

# DBTITLE 1,Data visualization using Koalas (full dataset)
plt.clf()
display(koalas_df["kingdom"].value_counts(normalize=True).plot.bar(rot=25, title="Bar plot of kingdom column using Koalas DataFrame").figure)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Note that Bacteria was not present as a category of the kingdom column for the Pandas DataFrame.

# COMMAND ----------

# DBTITLE 1,Did they find any Koalas or Pandas? 
koalasAndPandas=koalas_df.loc[(koalas_df['species'] == "Phascolarctos cinereus") & (koalas_df['species'] == "Ailuropoda melanoleuca")]
display(koalasAndPandas)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Since the data was taken from the Netherlands, I guess it makes sense we didn't find these adorable bears... 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://zootles.files.wordpress.com/2017/01/panda-koala.png" width="620"/>
# MAGIC </div>
