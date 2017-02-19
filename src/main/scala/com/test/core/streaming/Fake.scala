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
/**    
if (args.length < 3) {
      System.err.println(
        "Usage: KafkaWordCountProducer <metadataBrokerList> <topic> " +
          "<messagesPerSec> <wordsPerMessage>")
      System.exit(1)
    }
*/
  val conf = new SparkConf().setAppName("Fake").setMaster("local[*]")
  val sc = new SparkContext(conf)
  val spam = sc.textFile("/home/alexander/Downloads/spam1.txt", 4)
  val normal = sc.textFile("/home/alexander/Downloads/ham1.txt", 4)
  val tf = new HashingTF(numFeatures = 10000)
  val spamFeatures = spam.map(email => tf.transform(email.split(" ")))
  val normalFeatures = normal.map(email => tf.transform(email.split(" ")))
  val positiveExamples = spamFeatures.map(features => LabeledPoint(1, features))
  val negativeExamples = normalFeatures.map(features => LabeledPoint(0, features))
  val trainingData = positiveExamples.union(negativeExamples)
  trainingData.cache()
  val model = new LogisticRegressionWithSGD().run(trainingData)
  //Test on a positive example (spam) and a negative one (normal).
  val posTest = tf.transform(
    "REMINDER FROM O2  To get 2 50 pounds free call credit and details of great offers pls reply 2 this text with your valid name  house no and postcode".split(" "))
  val negTest = tf.transform(
    "Ok which your another number".split(" "))
  println("Prediction for positive test example: " + model.predict(posTest))
  println("Prediction for negative test example: " + model.predict(negTest))
}
}
