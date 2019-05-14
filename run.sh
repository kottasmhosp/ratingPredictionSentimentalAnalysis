#!/usr/bin/env bash

spark-submit \
--class "ratingprediction.ratingPrediction" \
--master local[4]  \
target/scala-2.11/ratingPrediction-assembly-0.4.jar 15
