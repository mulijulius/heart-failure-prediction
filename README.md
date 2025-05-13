# heart-failure-prediction
Machine learning models for predicting heart disease using clinical features
# Heart Failure Prediction Analysis

## Project Overview
This repository contains a machine learning analysis for predicting heart disease based on various clinical parameters. The project includes exploratory data analysis (EDA), data visualization, and implementation of classification models to predict the presence of heart disease.

## Dataset
The analysis uses the "Heart Failure Prediction" dataset which contains the following features:
- **Age**: Age of the patient
- **Sex**: Gender of the patient
- **ChestPainType**: Type of chest pain experienced
- **RestingBP**: Resting blood pressure (in mm Hg)
- **Cholesterol**: Serum cholesterol level
- **FastingBS**: Fasting blood sugar level
- **RestingECG**: Resting electrocardiogram results
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina
- **Oldpeak**: ST depression induced by exercise relative to rest
- **ST_Slope**: Slope of the peak exercise ST segment
- **HeartDisease**: Target variable (1 = heart disease, 0 = no heart disease)

## Requirements
- Python 3.7+
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

You can install all required dependencies using:
```
pip install -r requirements.txt
```

## Repository Structure
```
heart-failure-prediction/
│
├── heart_failure_prediction.py    # Main analysis script
├── heart_failure_prediction.csv   # Dataset
├── plots/                         # Generated visualization plots
│   ├── countplot_*.png
│   ├── histogram_*.png
│   ├── boxplot_*.png
│   ├── violinplot_*.png
│   ├── pairplot.png
│   ├── correlation_heatmap.png
│   ├── piechart_target.png
│   └── feature_importance.png
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

## Analysis Features

### Exploratory Data Analysis (EDA)
- Basic dataset information and descriptive statistics
- Distribution analysis of features
- Relationship between features and target variable
- Correlation analysis

### Visualizations
- Count plots for categorical variables
- Histograms for numerical features
- Box plots comparing feature distributions by heart disease status
- Violin plots for distribution comparison
- Pair plots for feature relationships
- Correlation heatmap
- Pie chart of class distribution

### Machine Learning Models
1. **Logistic Regression**
   - Standard scaling of features
   - Performance metrics: accuracy, precision, recall, F1-score
   - Confusion matrix

2. **Decision Tree Classifier**
   - Feature importance analysis
   - Performance metrics comparison

## How to Run
1. Clone this repository:
```
git clone https://github.com/[your-username]/heart-failure-prediction.git
cd heart-failure-prediction
```

2. Create and activate a virtual environment:

**For Windows:**
```
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**
```
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Run the analysis script:
```
python heart_failure_prediction.py
```

5. Deactivate the virtual environment when finished:
```
deactivate
```

4. Review the generated plots in the `plots` directory

## Results
The script provides classification reports and confusion matrices for both models implemented (Logistic Regression and Decision Tree). The analysis reveals important patterns and risk factors associated with heart disease.

Key insights include:
- Feature importance ranking for heart disease prediction
- Performance comparison between different classification algorithms
- Visual patterns of clinical parameters affecting heart disease risk

## Future Work
- Implement additional machine learning models (Random Forest, SVM, etc.)
- Perform hyperparameter tuning to improve model performance
- Implement cross-validation for more robust evaluation
- Create a simple web interface for predictions

## License
[Choose appropriate license]

## Author
MSc Dissertation Project (LD7083)

## Acknowledgements
- Dataset source: [Include source of the dataset if known]
- References to relevant medical/ML literaturef
