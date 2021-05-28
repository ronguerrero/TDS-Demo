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

from pyspark.sql.functions import col, upper

# Read loanstats_2012_2017.parquet
loan_stats = spark.read.parquet("dbfs:/ronguerrero/devdays/loanstats_2012_2017_pq.parquet")

# Select only the columns needed, and do a light transformation
loan_stats = loan_stats.select(col("addr_state"), upper(col("loan_status")))
display(loan_stats)

# COMMAND ----------

# MAGIC %sh 
# MAGIC cat /dbfs/ronguerrero/devdays/loanstats_2012_2017_pq.parquet

# COMMAND ----------

# MAGIC %md 
# MAGIC It was in parquet format, let's write it into CSV

# COMMAND ----------

loan_stats.write.option("overwrite", "true").csv("s3a:/home/ronguerrero/aviva_csv")

# COMMAND ----------

# MAGIC %sh
# MAGIC ls  /dbfs//home/ronguerrero/aviva_csv

# COMMAND ----------

# MAGIC %sh
# MAGIC cat  /dbfs//home/ronguerrero/aviva_csv/part-00000-tid-3787202180570828088-8d8e09a6-19f9-41d6-9c7d-44f3eea673c4-221-1-c000.csv

# COMMAND ----------

# MAGIC %md
# MAGIC Let's write this out as JSON

# COMMAND ----------

loan_stats.write.option("overwrite", "true").json("dbfs:/home/ronguerrero/aviva_json")

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/home/ronguerrero/aviva_json

# COMMAND ----------

# MAGIC %sh
# MAGIC cat /dbfs/home/ronguerrero/aviva_json/part-00000-tid-8087776953574435089-f14f91f1-968b-46e9-b6b0-68dbeb3885c2-223-1-c000.json

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Other ways to write data using spark connectors:  
# MAGIC * JDBC - https://docs.databricks.com/data/data-sources/sql-databases.html  
# MAGIC * Kinesis - https://docs.databricks.com/spark/latest/structured-streaming/kinesis.html
# MAGIC * Redshift - https://docs.databricks.com/data/data-sources/aws/amazon-redshift.html
# MAGIC * Many more

# COMMAND ----------

# MAGIC %md
# MAGIC ## Other ways to export data without Spark

# COMMAND ----------

# MAGIC %md 
# MAGIC %sh 
# MAGIC scp
