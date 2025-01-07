# Flight_Delay_Prediction
## Overview
The **Flight Delay Classification Project** aims to predict whether a flight will be delayed using various machine learning models. This project explores the dataset in detail, performs preprocessing and exploratory data analysis, and evaluates multiple classification algorithms to determine the best-performing model.

## Dataset
### Source
The dataset used in this project is sourced from Kaggle: [Flight Delay Prediction Dataset](https://www.kaggle.com/datasets/divyansh22/flight-delay-prediction?select=Jan_2019_ontime.csv). It contains flight information for January 2019, including flight schedules, delays, and other flight-related details.

### Key Features
- **OP_CARRIER:** Airline carrier code.
- **ORIGIN:** Departure airport.
- **DEST:** Arrival airport.
- **DEP_DELAY:** Departure delay in minutes.
- **ARR_DELAY:** Arrival delay in minutes.
- **DISTANCE:** Distance between the departure and arrival airports.
- **DEP_TIME_BLK:** Departure time block (e.g., "0600-0659").

### Target Variable
- **ARR_DELAY_GROUP:** A binary classification for flight delays:
    - **0:** No delay.
    - **1:** Delayed.

## Objectives
1. Perform data cleaning and preprocessing to prepare the dataset for analysis.
2. Conduct exploratory data analysis (EDA) to identify trends and correlations.
3. Build and compare multiple machine learning models to predict flight delays.
4. Evaluate model performance using standard metrics like accuracy, precision, and recall.

## Workflow
### 1. Data Preprocessing
- Handled missing values in critical features like DEP_DELAY and ARR_DELAY.
- Encoded categorical variables such as OP_CARRIER, ORIGIN, and DEST using one-hot encoding.
- Scaled numerical features like DEP_DELAY and DISTANCE.

### 2. Exploratory Data Analysis
- Visualized the distribution of departure and arrival delays.
- Analyzed the relationship between departure time blocks and delays.
- Identified key airports and carriers contributing to delays.

### 3. Machine Learning Models
The following models were trained and evaluated:
- **Decision Tree Classifier**
- **Naive Bayes Classifier**
- **Multilayer Perceptron (MLP) Classifier**
- **Logistic Regression**
- **Random Forest Classifier**

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Specificity

## Results
### Model Performance Summary

| Model                         | Accuracy | Precision | Recall | F1 Score |
|-------------------------------|----------|-----------|--------|----------|
| Decision Tree                 |  87.70%  |  79.51%   | 80.15% |  79.83%  |
| Naive Bayes                   |  91.75%  |  86.95%   | 85.07% |  85.96%  |
| Multilayer Perceptron (MLP)   |  91.97%  |  87.80%   | 84.67% |  86.11%  |
| Logistic Regression           |  91.75%  |  86.95%   | 85.07% |  85.96%  |
| Random Forest                 |  91.88%  |  87.92%   | 84.11% |  85.83%  |

## Key Findings
- The **MLP Classifier** performed best with an accuracy of 91.97% and an F1 score of 86.11%.
- All models except the Decision Tree achieved accuracy above 91%.
- The **Random Forest Classifier** had the highest precision (87.93%).
- The Naive Bayes and Logistic Regression models showed identical performance.

<!-- ## Future Enhancements
- Integrate additional data sources, such as weather conditions and air traffic volume.
- Extend the analysis to multiple months and years.
- Deploy a real-time prediction API or web app for user interaction. -->

## Usage
### Prerequisites
Ensure the following dependencies are installed:
- Python 3.8 or higher
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

### Running the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/Prateek1607/Flight_Delay_Classifier.git
   cd Flight_Delay_Classifier
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook Flight_Delay_Classifier.ipynb
   ```

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to enhance the project.


## License
This project is licensed under the MIT License. See `LICENSE` for more information.
