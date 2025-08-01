using Microsoft.ML;
using MLNetDemo.Models;

var context = new MLContext();

// Load data
var data = context.Data.LoadFromTextFile<CustomerData>("data.csv", hasHeader: true, separatorChar: ',');

// Split for training and testing
var split = context.Data.TrainTestSplit(data, testFraction: 0.2);

// Define the pipeline
var pipeline =
    context.Transforms.Concatenate("Features",
            nameof(CustomerData.Age),
            nameof(CustomerData.Income),
            nameof(CustomerData.PreviousPurchases))
        .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label",
            featureColumnName: "Features"));

// Train the model
var model = pipeline.Fit(split.TrainSet);

var predictions = model.Transform(split.TestSet);
var metrics = context.BinaryClassification.Evaluate(predictions);

Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");

// Make a prediction
var predictionEngine = context.Model.CreatePredictionEngine<CustomerData, PurchasePrediction>(model);

var sample = new CustomerData { Age = 28, Income = 55000, PreviousPurchases = 1 };
var prediction = predictionEngine.Predict(sample);

Console.WriteLine($"Predicted Purchase: {prediction.Prediction}");