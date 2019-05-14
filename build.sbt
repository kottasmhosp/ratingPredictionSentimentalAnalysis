name := "ratingPrediction"

version := "0.4"

scalaVersion := "2.11.12"

mainClass in (Compile, packageBin) := Some("ratingprediction.ratingPrediction")

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.4.3",
  "org.apache.spark" %% "spark-sql" % "2.4.3",
  "org.apache.spark" %% "spark-mllib" % "2.4.3"
)

mainClass in assembly := Some("ratingprediction.ratingPrediction")

// META-INF discarding for the FAT JAR
assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}
