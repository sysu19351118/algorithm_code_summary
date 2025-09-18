# -*- encoding: utf-8 -*-
'''
    * @File               :   2_get_sku_wo_col_name0.py
    * @Author             :   jianghao163 
    * @Date               :   2025-03-10 17:16:56
    * @Last Modified by   :   jianghao163
    * @Contact            :   jianghao163@jd.com
    * @Last Modified time :   2025-03-10 17:16:56
'''


from xml.etree.ElementPath import get_parent_map
from pyspark.sql.types import ArrayType, StringType, IntegerType, DoubleType, StructType, StructField
from pyspark.sql.functions import udf, explode, split, col, collect_list, concat_ws, row_number, avg, size, expr, first, length, when, lit, count, struct, array, posexplode, sum, coalesce
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession

from datetime import datetime, timedelta, date
import sys
import time
import zipfile
from ctypes import cdll, c_char_p, c_longlong, c_int


def get_spark(name):
    spk = SparkSession \
        .builder \
        .appName(name) \
        .enableHiveSupport() \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config('spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT', '1') \
        .config('spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT', '1') \
        .config("spark.sql.hive.convertMetastoreOrc", "false") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.default.parallelism", 1000) \
        .config("spark.sql.shuffle.partitions", 1000) \
        .getOrCreate()
    spk.sparkContext.setLogLevel("ERROR")
    return spk

def date_before(days: int):
    """
    :param days: 天数
    :return: 日期
    """
    return F.date_sub(F.current_date(), days)

sc = get_spark("personalized_gene_recall_stage_1@jianghao163")


csv_file_path = ['本地的csv表格']

schema = StructType( [
    StructField("col_name1", StringType(), True),
    StructField("cid", StringType(), True),
    StructField("brand_id", StringType(), True),
] )


sku_info_wo_col_name0_info = sc.read.csv(csv_file_path, header=False, sep='\t', schema=schema).select(
    "col_name1",
    "cid", 
    "brand_id", 
    "sku_name", 
    "shop_name", 
    "brand_name"
)


def truncate_list(col_name0_list):
    return "\x01".join( col_name0_list[:15] )

truncate_list_udf = F.udf(truncate_list, StringType())

relevance_log_raw = sc.table("ad_search.retr_relevance_predictor_log").filter(
    (col("dt") == start_date_str) \
    & ( col("score") > 0.6 )
).select(
    "col_name0",
    "col_name1"
).groupBy("col_name1").agg(
    collect_list("col_name0").alias("col_name0_list_raw")
).withColumn(
    "relevance_col_name0", truncate_list_udf( col("col_name0_list_raw") )
)


search_clk_req_raw = sc.table("ad_search.search_platform_data_sum").filter( # sc.table 从spark context中选取自己感兴趣的表格，并使用filter进行条件过滤
    (col("dt") == start_date_str) \
    & ( (col("pos_id") == 633) | (col("pos_id") == 3494) | (col("pos_id") == 10138) ) \
    & ( col("page") <= 10 ) \
    & ~( ( col("col_name0").rlike("^[0-9]+$") ) & ( length(col("col_name0")) > 10 ) )  # 过滤掉仅包含数字的col_name0 -- 仅包含数字长 col_name0 大多为 col_name1 定向搜索
).select(  # 选取其中感兴趣的两列
    "col_name0",
    "col_name1"
).groupBy("col_name1").agg(# 按照列col_name1进行分组（列中有很多重复的，比如某个商品会重复出现，然后使用agg方法将col_name0聚合成一个list，并命名为col_name0_list_raw
    collect_list("col_name0").alias("col_name0_list_raw")
).withColumn(
    "clk_col_name0", truncate_list_udf( col("col_name0_list_raw") )
)


sku_info_add_col_name0_info = sku_info_wo_col_name0_info.join(clk_req_raw, "col_name1", "left").join( # data frame 操作， 将clk_req_raw的col_name1列 join到表的左边
    search_clk_req_raw, "col_name1", "left"  # 也是类似的join操作
    "col_name1",
    coalesce(search_clk_req_raw.clk_col_name0).alias("clk_col_name0") # 选出search_clk_req_raw的clk_col_name0列也加入到这里面来，列名称为clk_col_name0
).join( # 也是join操作
    relevance_log_raw, "col_name1", "left"
)
# "relevance_col_name0"


sku_info_add_col_name0 = sku_info_add_col_name0_info.filter(
    col("clk_col_name0").isNotNull() | col("relevance_col_name0").isNotNull() # 根据条件筛选行元素
).select(   # 选取其中想要的列
    "col_name1",
    "clk_col_name0",
    "relevance_col_name0"
)



sku_info_add_col_name0.repartition(200).write.option("sep", "\t").csv(
    "save_path", 
    mode='overwrite'
)

