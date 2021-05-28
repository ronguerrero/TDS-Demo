# Databricks notebook source
# MAGIC %md
# MAGIC # Work with cloud storage within a Notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ##Let's make the data available for use.  

# COMMAND ----------

# MAGIC %md 
# MAGIC Databricks provides DBFS as part of a workspace deployment. It's primarily used for scratch space.

# COMMAND ----------

from pyspark.sql.functions import upper, col

spark.conf.set(
  "fs.azure.account.key.ronguerreroblob.blob.core.windows.net",
  "UCcEJW/ccAh4i/jRkac+q6nyW48H+9AGUt06hiFc3/PSPH5KiWR6JnLDItPZbswzzduVR4wmv6lM1lqLyW5vuw==")


# COMMAND ----------

# MAGIC %fs
# MAGIC ls  wasbs://bridgestone@ronguerreroblob.blob.core.windows.net/lending/

# COMMAND ----------

# Read loanstats_2012_2017.parquet
loan_stats = spark.read.parquet("wasbs://bridgestone@ronguerreroblob.blob.core.windows.net/lending/")

# Select only the columns needed, and do a light transformation
loan_stats = loan_stats.select(col("addr_state"), upper(col("loan_status")))
display(loan_stats)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Other ways to get data using spark connectors:  
# MAGIC * JDBC - https://docs.databricks.com/data/data-sources/sql-databases.html  
# MAGIC * EventHub - https://docs.databricks.com/spark/latest/structured-streaming/streaming-event-hubs.html
# MAGIC * Synapse - https://docs.databricks.com/data/data-sources/azure/synapse-analytics.html
# MAGIC * Many more

# COMMAND ----------

# MAGIC %md
# MAGIC ## Other ways to get data without Spark

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -N -P /dbfs/ml/streaming-stock-analysis/parquet/stocksDailyPricesSample/ https://pages.databricks.com/rs/094-YMS-629/images/stocksDailyPricesSample.snappy.parquet

# COMMAND ----------

# MAGIC %sh
# MAGIC dir /dbfs/ml/streaming-stock-analysis/parquet/stocksDailyPricesSample/ 

# COMMAND ----------

parquetPricePath = "/ml/streaming-stock-analysis/parquet/stocksDailyPricesSample/"
display(spark.read.parquet(parquetPricePath))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Extracting Data
# MAGIC * Use the Download button
# MAGIC * Use ADLS APIs/CLI to extract data directly from the ADLS
# MAGIC * Push the data to a remote location

# COMMAND ----------

# MAGIC %md 

# COMMAND ----------


