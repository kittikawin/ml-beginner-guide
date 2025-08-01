using Microsoft.ML.Data;

namespace MLNetDemo.Models;

public class PurchasePrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
}