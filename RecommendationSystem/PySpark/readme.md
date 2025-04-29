# PySpark 学习
本代码仅用于个人复习使用，代码中的敏感信息已经全部删除
spark有很多核心的功能，本文只对搜推场景下取数功能进行解析

## 基本工作逻辑
电商场景下数据量往往非常大，因此需要进行分布式存储和读取。现有的存储方案：
- [x] MapReduce
- [x] Hadoop
- [x] Spark

## 核心模块
* Spark Core:   提供 Spark 最核心的功能，是 Spark SQL、Spark Streaming 等其他模块实现的基础
* Spark SQL:	提供基于 SQL 或 HQL（Apache Hive 提供的 SQL 方言）进行数据查询的组件
* Spark Streaming:	Spark 平台上针对实时数据进行流式计算的组件
* MLlib	Spark: 平台的机器学习算法库
* GraphX:	Spark 面向图计算的组件与算法库

## Spark的运行架构
![alt text](images/image.png)

根据运行架构图，我们来简述一下 Spark 应用的运行过程：
* Driver 执行用户程序（Application）的 main() 方法并创建 SparkContext，与 Cluster Manager 建立通信
* Cluster Manager 为用户程序分配计算资源，返回可供使用的 Executor 列表
* 获取 Executor 资源后，Spark 会将用户程序代码及依赖包（Application jar）传递给 Executor（即移动计算）
* 最后，SparkContext 发送 tasks（经拆解后的任务序列）到 Executor，由其执行计算任务并输出结果

## Spark 常见术语
Application	基于 Spark 构建的用户程序
Application jar	包含用户 Spark 应用程序的 jar 包（不包含 Hadoop 和 Spark 依赖包，运行时由集群导入）
Driver Program	运行用户程序 main() 函数并创建 SparkContext 的进程
SparkContext	用户程序与 Spark 集群交互的主要入口，用于创建 RDD、累加器和广播变量等
Cluster Manager	集群资源管理器，其实现可以是 Standalone、Mesos、YARN 或 Kubernetes
Master Node	独立部署集群中的主节点，负责资源调度，类比 Yarn 中的 ResourceManager
Worker Node	独立部署集群中的从节点，负责执行计算任务，类比 Yarn 中的 NodeManager
Executor	Worker 节点上负责执行实际计算任务的组件
Task	分区级别的计算任务，是 Spark 中最基本的任务执行单元

## Spark 运行方式
搜广推场景下，Spark主要用于分布式数据的获取。公司会收集每天用户的多维度行为数据保存到一个很大的Hive表到分布式存储系统中，如果我们只用本地的小型服务器不可能在短时间内处理这么多的数据，因此需要使用Spark分布式框架来进行多进程的数据处理，从海量的数据维度中提取出我们想要的数据。简单类比，可以把spark堪称一个多线程表格数据读取器，用于对一个非常非常大的Excel表格进行数据抓取。

主要通过Spark的RDD分区以及相关RDD算子进行取数（数据读取）操作

### RDD算子逻辑
filter(func)	筛选出满足条件的元素，并返回一个新的数据集
map(func)	将每个元素传递到函数 func 中，返回一个新的数据集，每个输入元素会映射到 1 个输出结果
flatMap(func)	与 map 相似，但每个输入元素都可以映射到 0 或多个输出结果
mapPartitions(func)	与 map 相似，但是传递给函数 func 的是每个分区数据集对应的迭代器
distinct(func)	对原数据集进行去重，并返回新的数据集
groupByKey([numPartitions])	应用于 (K, V) 形式的数据集，返回一个新的 (K, Iterable<V>) 形式的数据集，可通过 numPartitions 指定新数据集的分区数
reduceByKey(func, [numPartitions])	应用于 (K, V) 形式的数据集，返回一个新的 (K, V) 形式的数据集，新数据集中的 V 是原有数据集中每个 K 对应的 V 传递到 func 中进行聚合后的结果
aggregateByKey(zeroValue)(seqOp, combOp, [numPartitions])	应用于 (K, V) 形式的数据集，返回一个新的 (K, U) 形式的数据集，新数据集中的 U 是原有数据集中每个 K 对应的 V 传递到 seqOp 与 combOp 的联合函数且与 zeroValue 聚合后的结果
sortByKey([ascending], [numPartitions])	应用于 (K, V) 形式的数据集，返回一个根据 K 排序的数据集，K 按升序或降序排序由 ascending 指定
union(func)	将两个数据集中的元素合并到一个新的数据集
join(func)	表示内连接，对于给定的两个形式分别为 (K, V) 和 (K, W) 的数据集，只有在两个数据集中都存在的 K 才会被输出，最终得到一个 (K, (V, W)) 类型的数据集
repartition(numPartitions)	对数据集进行重分区，新的分区数由 numPartitions 指定

### 实际代码逻辑讲解
由于需要公司内部环境才能运行，并且涉及很多公司内容，已做强马赛克处理
使用start.sh中的代码运行，主要通过spark-submit函数进行任务递交，递交到集群之后会根据sh中的配置安排driver（领导）和executor（工作者）进行数据读取
sh最后一行要给上pyspark代码的入口文件
pyspark文件主要通过以下的代码声明上下文，
```python
spk = SparkSession
```
通过以下的代码获得一个巨大的HIVE表
```python
sc.table("ad_search.retr_relevance_predictor_log")
```
然后再使用前面提到的RDD算子对这个表格进行进一步的处理，然后运行下面的代码将处理好的数据保存为csv到指定路径

```python 
sku_info_add_col_name0.repartition(200).write.option("sep", "\t").csv(
    "save_path", 
    mode='overwrite'
)
```