from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import os

spark = SparkSession.builder \
    .appName("RecsysFeatureEngineering") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("Spark session started!")

# ── Define schema explicitly ───────────────────────────────
schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("rating", IntegerType(), True),
    StructField("summary", StringType(), True),
    StructField("text", StringType(), True),
    StructField("review_length", IntegerType(), True),
    StructField("is_positive", IntegerType(), True),
    StructField("product_avg_rating", FloatType(), True),
    StructField("user_review_count", IntegerType(), True),
    StructField("user_avg_rating", FloatType(), True)
])

# ── Load data ──────────────────────────────────────────────
print("\nLoading data...")
df = spark.read.csv(
    "data/processed/clean_reviews.csv",
    header=True,
    schema=schema,
    quote='"',
    escape='"',
    multiLine=True
)
print(f"Total rows: {df.count():,}")

# ── Feature 1: User activity stats ────────────────────────
print("\nComputing user features...")
user_features = df.groupBy("user_id").agg(
    F.count("*").alias("total_reviews"),
    F.avg("rating").alias("avg_rating_given"),
    F.stddev("rating").alias("rating_std"),
    F.min("rating").alias("min_rating"),
    F.max("rating").alias("max_rating"),
    F.sum(F.when(F.col("rating") >= 4, 1).otherwise(0)).alias("positive_reviews"),
    F.avg("review_length").alias("avg_review_length")
)
print(f"User features: {user_features.count():,} users")
user_features.show(5)

# ── Feature 2: Product popularity stats ───────────────────
print("\nComputing product features...")
product_features = df.groupBy("product_id").agg(
    F.count("*").alias("total_reviews"),
    F.avg("rating").alias("avg_rating"),
    F.stddev("rating").alias("rating_std"),
    F.countDistinct("user_id").alias("unique_reviewers"),
    F.sum(F.when(F.col("rating") >= 4, 1).otherwise(0)).alias("positive_reviews"),
    F.avg("review_length").alias("avg_review_length")
).withColumn(
    "popularity_score",
    (F.col("avg_rating") * 0.7) +
    (F.col("total_reviews") / 609 * 5 * 0.3)
)
print(f"Product features: {product_features.count():,} products")
product_features.show(5)

# ── Feature 3: Rating distribution ────────────────────────
print("\nComputing rating distribution...")
rating_dist = df.groupBy("product_id", "rating").count() \
    .groupBy("product_id").pivot("rating", [1, 2, 3, 4, 5]) \
    .agg(F.first("count")) \
    .fillna(0)
rating_dist = rating_dist.toDF(
    "product_id", "rating_1", "rating_2",
    "rating_3", "rating_4", "rating_5"
)
print("Rating distribution computed!")

# ── Save features ──────────────────────────────────────────
print("\nSaving features...")
os.makedirs("data/processed/spark_features", exist_ok=True)

user_features.toPandas().to_csv(
    "data/processed/spark_features/user_features.csv", index=False
)
product_features.toPandas().to_csv(
    "data/processed/spark_features/product_features.csv", index=False
)
rating_dist.toPandas().to_csv(
    "data/processed/spark_features/rating_distribution.csv", index=False
)

print("\n" + "=" * 50)
print("PYSPARK FEATURE ENGINEERING COMPLETE!")
print(f"Users processed:    {user_features.count():,}")
print(f"Products processed: {product_features.count():,}")
print("=" * 50)

spark.stop()