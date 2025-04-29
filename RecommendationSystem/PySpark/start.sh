date=$1

spark-submit \
  --num-executors 200 \
  --name "tzx_task_01" \
  --queue bdp_jmart_ad.jd_ad_search \
  --master yarn \
  --deploy-mode cluster \
  --conf spark.shuffle.service.enabled=true \
  --conf spark.shuffle.service.port=7337 \
  --conf spark.dynamicAllocation.enabled=true \
  --conf spark.dynamicAllocation.minExecutors=2 \
  --conf spark.dynamicAllocation.maxExecutors=300 \
  --conf spark.yarn.maxAppAttempts=1 \
  --conf spark.yarn.am.cores=4 \
  --conf spark.yarn.am.memory=4g \
  --conf spark.executor.memory=32G \
  --conf spark.executor.memoryOverhead=16g \
  --conf spark.executor.cores=3 \
  --conf spark.driver.memory=4G \
  --conf spark.driver.maxResultSize=8g \
  --conf spark.sql.shuffle.partitions=1200 \
  --conf spark.sql.autoBroadcastJoinThreshold=-1 \
  --conf "spark.pyspark.python=./spk/bin/python"  \
  --conf "spark.yarn.dist.archives=hdfs://ns1017/user/jd_ad/ads_search/targeting/lizhixuan5/ad_search_pretrained_models/uniform_retrieval_finetune/version_3-0/data/tools/spark.tar.gz#spk" \
  ./code.py --dat "$date"