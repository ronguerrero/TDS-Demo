-- Databricks notebook source
show databases

-- COMMAND ----------

use ronguerrero

-- COMMAND ----------

drop table if exists tds1;
create table tds1 (a integer, b string) using delta;

-- COMMAND ----------

insert into tds1 values(1, 'ron');
insert into tds1 values(2, 'paul');


-- COMMAND ----------

update tds1 set b = 'peter' where a = 2;

-- COMMAND ----------

delete from tds1 where a = 1;

-- COMMAND ----------

select * from tds1;


-- COMMAND ----------


