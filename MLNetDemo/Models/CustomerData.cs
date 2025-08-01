using Microsoft.ML.Data;

namespace MLNetDemo.Models;

public class CustomerData
{
    [LoadColumn(0)] public float Age { get; set; }

    [LoadColumn(1)] public float Income { get; set; }

    [LoadColumn(2)] public float PreviousPurchases { get; set; }

    [LoadColumn(3)] public bool Label { get; set; }
}