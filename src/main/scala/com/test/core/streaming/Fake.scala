package com.test.core.streaming

import org.apache.spark._
import org.apache.log4j.Logger

/**
  * Приложение позволяет классифицировать sms-сообщение как спам или не спам
  */
object Fake {

  val Log = Logger.getLogger(Fake.this.getClass().getSimpleName())
  def main(args: Array[String]) {
    if (args.length < 3) {
      System.err.println(
        "Usage: test message" " +
          "path/to/spam path/to/ham")
      System.exit(1)
    }

    val Array(testMessage, pathToSpamSet, pathToHamSet) = args
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("Fake")
    val sc = new SparkContext(sparkConf)

    /** Преобразуем подготовленные наборы сообщений в RDD */
    val spam = sc.textFile(pathToSpamSet)
    val normal = sc.textFile(pathToHamSet)

    /** Для каждого набора определяем общее количество уникальных слов и частоту каждого слова */
    val spamWords = spam.flatMap(line => line.split("\t"))
    val spamWordsCount =
      spamWords.map(x => (x, 1)).reduceByKey((x, y) => x + y)
    val sumSpamWords = spamWordsCount.count()

    val normalWords = normal.flatMap(line => line.split("\t"))
    val normalWordsCount =
      normalWords.map(x => (x, 1)).reduceByKey((x, y) => x + y)
    val sumNormWords = normalWordsCount.count()

    /** Определяем количество уникальных слов в обоих набоах */
    val sumSmsWords = normalWordsCount
      .union(spamWordsCount)
      .reduceByKey((x, y) => x + y)
      .count()

    /** Подготавливаем тестовое сообщение */
    val testWords = testMessage.split(" ")

    /** Предполагаем, что априорная вероятность сообщения попасть в какую-либо группу - 50% */
    var probabilitySpam = java.lang.Math.log(0.5)
    var probabilityNotSpam = java.lang.Math.log(0.5)

    /** Применяем наивный байесовский классификатор */
    testWords.foreach(word => {
      var m =
        if (spamWordsCount.lookup(word).length == 0) 0
        else spamWordsCount.lookup(word)(0).toDouble

      probabilitySpam += java.lang.Math
        .log((m + 1) / (sumSpamWords + sumSmsWords))

      var n =
        if (normalWordsCount.lookup(word).length == 0) 0
        else normalWordsCount.lookup(word)(0).toDouble

      probabilityNotSpam += java.lang.Math
        .log((n + 1) / (sumNormWords + sumSmsWords))
    })

    /** В зависимости от результатов классификации выводим соощение */
    if (probabilitySpam > probabilityNotSpam)
      println("Сообщение похоже на спам")
    else println("Сообщение НЕ спам")
  }

}

