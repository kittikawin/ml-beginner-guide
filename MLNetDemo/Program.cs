using Microsoft.ML;
using MLNetDemo.Models;

var context = new MLContext(seed: 1);

// Load data
var data = context.Data.LoadFromTextFile<CustomerData>("data.csv", hasHeader: true, separatorChar: ',');

// Split for training and testing
var trainTestSplit = context.Data.TrainTestSplit(data, testFraction: 0.2);
var trainingData = trainTestSplit.TrainSet;

// Define the pipeline
var dataProcessPipeline =
    context.Transforms.Concatenate("Features",
        nameof(CustomerData.Age),
        nameof(CustomerData.Income),
        nameof(CustomerData.PreviousPurchases));

var trainer = context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
var trainingPipeline = dataProcessPipeline.Append(trainer);

// Train the model
var trainedModel = trainingPipeline.Fit(trainingData);

// Make a prediction
var predictionEngine = context.Model.CreatePredictionEngine<CustomerData, PurchasePrediction>(trainedModel);

var sample = new CustomerData { Age = 28, Income = 55000, PreviousPurchases = 1 };
var prediction = predictionEngine.Predict(sample);

Console.WriteLine($"Predicted Purchase: {prediction.Prediction}");
