package ratingprediction

  import org.apache.spark.sql.SparkSession
  import org.apache.log4j.{Level, LogManager, Logger}
  import com.github.fommil.netlib
  import math.pow
  import org.apache.spark.ml.Pipeline
  import org.apache.spark.sql.functions.regexp_replace
  import org.apache.spark.ml.evaluation.RegressionEvaluator
  import org.apache.spark.ml.feature.VectorIndexer
  import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
  import org.apache.spark.ml.linalg.Vector
  import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
  import org.apache.spark.sql.Row
  import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassificationModel, RandomForestClassifier, LinearSVC}
  import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
  import org.apache.spark.ml.feature.{HashingTF, IDF, IndexToString, StopWordsRemover, StringIndexer, Tokenizer, VectorIndexer, VectorAssembler}

  /** Computes an approximation to pi */
  object ratingPrediction {
    def main(args: Array[String]) {
      Logger.getLogger("org").setLevel(Level.ERROR)
      val log = LogManager.getRootLogger
      log.info("Start")

      val spark = SparkSession
        .builder()
        .appName("Rating Prediction")
        .config("spark.master", "local")
        .getOrCreate()

      val data = spark
        .read
        .option("charset", "UTF8")
        .json("file:///Users/thomas.kottas/Documents/Apache_Spark/hadoop/yelp_dataset_round_13_2019/review.json")
        .limit(10000)
      log
        .info("Data loaded success.")

      val pos_neg_data = data
        .withColumn("category",data("stars") > 3)
        .withColumn("text", regexp_replace(data("text"), "[,.!?:;]]", ""))

      pos_neg_data.show(100)

      val tokenizer = new Tokenizer()
        .setInputCol("text")
        .setOutputCol("words")
      val wordsData = tokenizer
        .transform(pos_neg_data)
      log
        .info("Words tokenized success.")

//      wordsData.show(10)

      val hashingTF = new HashingTF()
        .setInputCol("words")
        .setOutputCol("rawFeatures")
        .setNumFeatures(65535)
      val featurizedData = hashingTF
        .transform(wordsData)
      log
        .info("Hashing TF success.")

//      featurizedData.show(10)

      val idf = new IDF()
        .setInputCol("rawFeatures")
        .setOutputCol("idfFeatures")
      val idfModel = idf
        .fit(featurizedData)
      val rescaledData = idfModel
        .transform(featurizedData)
      log
        .info("IDF-TF success.")

//      rescaledData.show(10)

      val assembler = new VectorAssembler()
        .setInputCols(Array("idfFeatures", "category"))
        .setOutputCol("features")
      val transformed = assembler
        .transform(rescaledData)
      log
        .info("Assembly multiple features success.")

//       Index labels, adding metadata to the label column.
//       Fit on whole dataset to include all labels in index.
      val labelIndexer = new StringIndexer()
        .setInputCol("stars")
        .setOutputCol("indexedLabel")
        .fit(transformed)
      log
        .info("Label indexing success.")

//       Automatically identify categorical features, and index them.
//       Set maxCategories so features with > 4 distinct values are treated as continuous.
      val featureIndexer = new VectorIndexer()
        .setInputCol("features")
        .setOutputCol("indexedFeatures")
        .setMaxCategories(4)
        .fit(transformed)
      log
        .info("Feature indexing success.")

//       Split the data into training and test sets (30% held out for testing).
      val Array(trainingData, testData) = transformed
        .randomSplit(Array(0.7, 0.3))
      log
        .info("Split data to training and test success.")

//      val rf = new RandomForestClassifier()
//        .setLabelCol("indexedLabel")
//        .setFeaturesCol("indexedFeatures")
//        .setNumTrees(10)
//      val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]


      val lr = new LogisticRegression()
        .setMaxIter(10)
        .setRegParam(0.3)
        .setElasticNetParam(0.8)
        .setLabelCol("indexedLabel")
        .setFeaturesCol("indexedFeatures")

      log
        .info("Train Model Initialization success.")

//       Convert indexed labels back to original labels.
      val labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("predictedLabel")
        .setLabels(labelIndexer.labels)
      log
        .info("Label Indexed Converter success.")

//       Chain indexers and forest in a Pipeline.
      val pipeline = new Pipeline()
        .setStages(Array(labelIndexer, featureIndexer, lr, labelConverter))
      log
        .info("Pipeline Set Stages success.")

//       Train model. This also runs the indexers.
      val model = pipeline.
        fit(trainingData)
      log
        .info("Model Train success.")

//       Make predictions.
      val predictions = model.
        transform(testData)
      log
        .info("Make Prediction success.")

//       Select example rows to display.
//      predictions.select("predictedLabel", "stars", "features").show(5)

      // Select (prediction, true label) and compute test error.
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("indexedLabel")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")
      val accuracy = evaluator.evaluate(predictions)
      println("Success = " + (accuracy))

      spark.stop
    }
}


