# Insurance Premium Predictor

A machine learning solution for predicting insurance premiums using LightGBM. This project demonstrates end-to-end ML pipeline development with focus on code quality and maintainability.

## 🎯 Project Overview

This project implements a robust machine learning pipeline for predicting insurance premiums. It features:
- Custom feature engineering for insurance data
- LightGBM model with optimized parameters
- 5-fold cross-validation
- RMSLE (Root Mean Squared Logarithmic Error) optimization
- Type-safe implementation with comprehensive error handling

## 🛠️ Technical Stack

- **Python**: 3.11+
- **Core Libraries**:
  - `lightgbm`: Gradient boosting framework
  - `pandas`: Data manipulation
  - `numpy`: Numerical operations
  - `scikit-learn`: ML utilities
  
## 📁 Project Structure

```
insurance-premium-predictor/
├── src/
│   ├── features/           # Feature engineering
│   │   └── feature_engineering.py
│   ├── models/            # Model implementations
│   │   └── models.py
│   ├── utils/             # Utility functions
│   │   └── metrics.py
│   └── train.py          # Training pipeline
├── data/                  # Data directory
├── submissions/          # Model predictions
└── requirements.txt      # Project dependencies
```

## 🔧 Features

### Feature Engineering
- Automated categorical variable handling
- Domain-specific feature creation:
  - Income per dependent
  - Claims per year
  - Policy duration analysis

### Model Implementation
- LightGBM with early stopping
- Optimized hyperparameters
- Cross-validation for robust evaluation

### Quality Assurance
- Type hints throughout
- Comprehensive error handling
- Detailed logging
- Modular code structure

## 📊 Model Performance

The model is evaluated using 5-fold cross-validation with RMSLE as the metric:
- Mean CV RMSLE: [Your Score]
- Standard Deviation: [Your Score]

d
## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Your Name
- GitHub: [@psukh28](https://github.com/psukh28)
- LinkedIn: [surya-praanv-sukumaran](https://linkedin.com/in/surya-praanv-sukumaran)

## 🌟 Acknowledgments

- Data source: [Playground Series S4-E12](https://www.kaggle.com/competitions/playground-series-s4e12)
- Inspiration: Insurance premium prediction challenge
- Libraries: LightGBM, scikit-learn, pandas
