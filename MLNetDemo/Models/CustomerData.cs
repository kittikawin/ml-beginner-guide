using Microsoft.ML.Data;

namespace MLNetDemo.Models;

public class CustomerData
{
    [LoadColumn(0), ColumnName("Age")] public float Age { get; set; }

    [LoadColumn(1), ColumnName("Income")] public float Income { get; set; }

    [LoadColumn(2), ColumnName("PreviousPurchases")] public float PreviousPurchases { get; set; }

    [LoadColumn(3), ColumnName("Label")] public bool Label { get; set; }
}