/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mllogisticsregression;

import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 *
 * @author Student
 */
public class LogiscsWeather {
    public static void main(String[] args) {
        // TODO code application logic here
        String training_dataset = "weather_training.arff";
        String testing_dataset = "weather_testing.arff";
        String predicting_dataset = "weather_predicts.arff";

        process(training_dataset, testing_dataset, predicting_dataset);
    }

    public static void process(String training, String testing, String predict) {

        try {
            Instances traingDataSet = getDataSet(training);
            Instances testingDataSet = getDataSet(testing);
            // Instances predictDataSet = getDataSet(predict);
            Classifier classifier = new Logistic();
            classifier.buildClassifier(traingDataSet);
            Evaluation eval = new Evaluation(traingDataSet);
            eval.evaluateModel(classifier, testingDataSet);
            System.out.println("Logistics Regression Evaluate with Dataset");
            System.out.println(eval.toSummaryString());
            System.out.println(classifier);

            System.out.println("Prediction");
            for (int i = 0; i < 3; i++) {
                Instance predictDataset = getDataSet(predict).instance(i);
                double value = classifier.classifyInstance(predictDataset);
                System.out.println(value);
            }
        } catch (Exception ex) {
            Logger.getLogger(LogisticsRegression.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static Instances getDataSet(String filename) {

        try {
            int classIndex = 4; // index of find value
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File(filename));
            Instances dataSet = loader.getDataSet();
            dataSet.setClassIndex(classIndex);
            return dataSet;
        } catch (IOException ex) {
            Logger.getLogger(LogisticsRegression.class.getName()).log(Level.SEVERE, null, ex);
        }

        return null;
    }
}
