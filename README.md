# SMS Spam Classification

## Project Overview

This project aims to classify SMS messages as either 'spam' or 'ham' using machine learning techniques. The classification model is developed using various text preprocessing methods and machine learning algorithms, including Random Forest and Gradient Boosting. The dataset used is the SMS Spam Collection dataset, which is available in tab-separated format.

## Dataset

The dataset used in this project is the SMS Spam Collection, which contains SMS messages labeled as 'spam' or 'ham'. The file is named `SMSSpamCollection.tsv`. Each row in the dataset represents an SMS message, with columns:

- **Label**: Indicates whether the message is 'spam' or 'ham'.
- **Message**: The text content of the SMS message.

## Features

- **Text Preprocessing**: Includes removing punctuation, tokenization, removing stopwords, stemming, and lemmatization.
- **Feature Engineering**: Creation of features such as message length and percentage of punctuation.
- **Vectorization**: Transformation of text data into numerical features using Count Vectorization and TF-IDF.
- **Modeling**: Application of Random Forest and Gradient Boosting classifiers.

## Installation

To run this project, you need Python and several Python packages. Follow these steps to set up the environment:

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/your-username/sms-spam-classification.git
    cd sms-spam-classification
    ```

2. **Install Dependencies**:

    ```bash
    pip install pandas numpy scikit-learn nltk matplotlib
    ```

3. **Download the Dataset**:

    Ensure the `SMSSpamCollection.tsv` file is placed in the project directory. You can download it from [here](https://archive.ics.uci.edu/ml/datasets/sms_spam_collection).

## Usage

1. **Data Preprocessing and Exploration**:

    The initial steps include reading the dataset, exploring its structure, and performing data cleaning. Execute the `preprocessing.py` script to run these steps:

    ```bash
    python preprocessing.py
    ```

2. **Model Training and Evaluation**:

    After preprocessing, you can train and evaluate models using `modeling.py`:

    ```bash
    python modeling.py
    ```

## Code Explanation

### 1. Reading and Preprocessing the Data

- **Reading the File**:
    - The dataset is read and split into labels and body text.
    - Discrepancies are investigated and corrected.

- **Creating a Data Frame**:
    - The raw data is organized into a DataFrame.

- **Exploratory Data Analysis (EDA)**:
    - Overview of data distribution and class counts.

### 2. Text Preprocessing

- **Removing Punctuations**:
    - Punctuation is removed to clean the text data.

- **Tokenization**:
    - Text is split into tokens for further processing.

- **Removing Stopwords**:
    - Common stopwords are removed from the tokenized text.

- **Stemming and Lemmatization**:
    - Words are reduced to their root forms using stemming and lemmatization.

- **Feature Creation**:
    - Additional features like message length and punctuation percentage are calculated.

### 3. Vectorization

- **Count Vectorizer**:
    - Converts text data into numerical features using token counts.

- **N-Grams**:
    - Generates n-grams (bigrams) for improved feature representation.

- **TF-IDF Vectorizer**:
    - Computes Term Frequency-Inverse Document Frequency (TF-IDF) for feature extraction.

### 4. Model Training and Evaluation

- **Random Forest Classifier**:
    - A Random Forest model is trained and evaluated using cross-validation.

- **Gradient Boosting Classifier**:
    - A Gradient Boosting model is trained and evaluated.

- **Hyperparameter Tuning**:
    - Grid Search and Randomized Search are used for hyperparameter optimization.

## Results

- **Model Performance**:
    - The Random Forest and Gradient Boosting models are evaluated based on precision, recall, and accuracy.
    - Performance metrics are displayed for both models.

- **Feature Importance**:
    - Key features contributing to the model's predictions are identified.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements or fixes.

## Contact

For questions or suggestions, please contact me at [pradeepakaarthi@gmail.com](mailto:your.email@example.com).

## Acknowledgements

- The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms_spam_collection).
- This project utilizes NLTK and scikit-learn for text processing and machine learning.

## Future Work

- Explore additional machine learning models and techniques.
- Implement deep learning approaches for text classification.
- Investigate other text features and preprocessing methods.

