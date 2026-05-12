# Results Summary Template

This file provides a standard template for saving and organizing results across all experiments.

## Results File Structure

Place all results in the `results/` folder with these standard filenames:

### Metrics Files

- `tfidf_metrics.csv` - TF-IDF + Logistic Regression metrics
- `embedding_lr_metrics.csv` - Embedding + Logistic Regression metrics
- `embedding_svm_metrics.csv` - Embedding + SVM metrics
- `embedding_rf_metrics.csv` - Embedding + Random Forest metrics (optional)

### Confusion Matrix Figures

- `confusion_matrix_tfidf.png`
- `confusion_matrix_embedding_lr.png`
- `confusion_matrix_embedding_svm.png`
- `confusion_matrix_embedding_rf.png` (optional)

### Metrics CSV Format

Each metrics file should contain exactly these columns (in this order):

```
Model,Classifier,Accuracy,Precision,Recall,F1-Score,True_Negatives,False_Positives,False_Negatives,True_Positives
TF-IDF,LogisticRegression,0.XX,0.XX,0.XX,0.XX,XXXX,XXXX,XXXX,XXXX
Embedding,LogisticRegression,0.XX,0.XX,0.XX,0.XX,XXXX,XXXX,XXXX,XXXX
...
```

### Error Analysis

- `error_analysis.md` - Short notes about misclassified examples

### Aggregated Results Table

- `final_results_table.csv` - Merged results for the final report

### Model Artifacts

- `models/tfidf_model.pkl` - Trained TF-IDF vectorizer and classifier
- `models/embedding_lr.pkl` - Trained embedding + LR classifier
- `models/embedding_svm.pkl` - Trained embedding + SVM classifier
- `models/embeddings_train.npy` - Cached train embeddings (optional, if reusable)
- `models/embeddings_test.npy` - Cached test embeddings (optional, if reusable)

## Why This Matters

- Standard filenames let all teammates know where to find outputs
- Standard CSV format lets Person 3 merge results without reformatting
- Standard figure names link directly to report sections
- Standard paths ensure reproducibility in final submission
