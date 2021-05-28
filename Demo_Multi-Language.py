# Databricks notebook source
# MAGIC %md 
# MAGIC ## Let's do SQL

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS ronguerrero.all_stocks_5yr;
# MAGIC CREATE TABLE ronguerrero.all_stocks_5yr
# MAGIC USING com.databricks.spark.csv
# MAGIC OPTIONS (path "/ronguerrero/all_stocks_5yr.csv", header "true", inferSchema "true");

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC    * 
# MAGIC FROM 
# MAGIC    ronguerrero.all_stocks 
# MAGIC WHERE name = 'AAPL'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC    Name, min(low) as minLow, max(high) as maxHigh 
# MAGIC FROM 
# MAGIC    ronguerrero.all_stocks_5yr 
# MAGIC GROUP BY 
# MAGIC    Name;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's work with Python

# COMMAND ----------

# DBTITLE 1,Pandas 
import pandas as pd
import numpy as np

pdStocks = pd.read_csv('/dbfs/ronguerrero/all_stocks_5yr.csv')
pdStocks.loc[pdStocks['Name'] == "AAPL"].head(10)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Koalas - Pandas at Scale!

# COMMAND ----------

import databricks.koalas as ks

ksStocks = ks.read_csv('/ronguerrero/all_stocks_5yr.csv')
ksStocks.loc[ksStocks['Name'] == "AAPL"].head(10)

# COMMAND ----------

ksStocks.shape

# COMMAND ----------

display(ksStocks)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's do some R

# COMMAND ----------

# MAGIC %r
# MAGIC library(sparklyr)
# MAGIC library(dplyr)
# MAGIC 
# MAGIC sc <- spark_connect(method = "databricks")
# MAGIC 
# MAGIC ## Read a table from a table or global temporary view
# MAGIC fromTable <- spark_read_table(sc, "aapl_delta") 
# MAGIC 
# MAGIC head(fromTable)

# COMMAND ----------


