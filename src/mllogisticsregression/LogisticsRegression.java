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
public class LogisticsRegression {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        String training_dataset = "diabete_train2.arff";
        String testing_dataset = "diabete_testing.arff";
        String predicting_dataset = "diabete_predict.arff";

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
            for (int i = 0; i < 4; i++) {
                Instance predictDataset = getDataSet(predict).instance(i);
                double value = classifier.classifyInstance(predictDataset);
                System.out.println("Number of Pragnancy "+predictDataset.value(0));
                System.out.println("Number of blood pressure "+predictDataset.value(2));
                System.out.println(value);
                if (value == 0) {
                    System.out.println("You got diabete :) Please see the doctor");
                }
            }
        } catch (Exception ex) {
            Logger.getLogger(LogisticsRegression.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static Instances getDataSet(String filename) {

        try {
            int classIndex = 8; // index of find value
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
