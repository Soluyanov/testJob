package com.test.core.streaming

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.log4j.Logger

object Fake {

val Log = Logger.getLogger(Fake.this.getClass().getSimpleName())
  def main(args: Array[String]) {
    
if (args.length < 3) {
      System.err.println(
        "Usage: test message " +
          "path/to/spam path/to/ham")
      System.exit(1)
    }

  val Array(testMessage, pathToSpamSet, pathToHamSet) = args
  val conf = new SparkConf().setAppName("Fake").setMaster("local[*]")
  val sc = new SparkContext(conf)
  val spam = sc.textFile(pathToSpamSet)
  val normal = sc.textFile(pathToHamSet)
  val tf = new HashingTF(numFeatures = 10000)
  val spamFeatures = spam.map(email => tf.transform(email.split(" ")))
  val normalFeatures = normal.map(email => tf.transform(email.split(" ")))
  val positiveExamples = spamFeatures.map(features => LabeledPoint(1, features))
  val negativeExamples = normalFeatures.map(features => LabeledPoint(0, features))
  val trainingData = positiveExamples.union(negativeExamples)
  trainingData.cache()
  val model = new LogisticRegressionWithSGD().run(trainingData)
  
  val test = tf.transform(testMessage.split(" "))
  println("Prediction for test example: " + model.predict(test))
  
}
}
