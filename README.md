# Spam Detection Pipeline

This repository contains a spam detection pipeline implemented in Python. The pipeline utilizes various machine learning models to classify SMS messages as spam or ham (not spam).

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd spam-classifier
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline

To launch the spam detection pipeline, execute the following command:

```bash
python pipeline.py
```

### Configuration

You can configure the model type and embedding type by modifying the parameters in the `SpamDetectionPipeline` class within `pipeline.py`. The default model type is `xgboost` and the embedding type is `tfidf`.

### Evaluation

After running the pipeline, the results will include a summary of Recall for each model type used in the classification, as False Negative are more expensive than false positives (we want to ideally avoid getting spams in our mailbox).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Scikit-learn for machine learning algorithms
- XGBoost for gradient boosting
- TensorFlow for deep learning models
