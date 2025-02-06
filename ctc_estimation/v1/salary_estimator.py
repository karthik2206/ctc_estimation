import pandas as pd
from river import linear_model, preprocessing, compose, metrics, optim, feature_extraction
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

class SalaryEstimator:
    def __init__(self):
        # Create preprocessing steps for each feature
        self.preprocessors = {
            'yoe': preprocessing.StandardScaler(),
            'college_type': preprocessing.OneHotEncoder(),
            'company_category': preprocessing.OneHotEncoder()
        }
        
        # Create polynomial feature extractor for interactions
        self.poly = feature_extraction.PolynomialExtender(degree=2, interaction_only=True)
        
        # Create the regressor with better regularization
        self.regressor = linear_model.LinearRegression(
            optimizer=optim.SGD(lr=0.001),
            l2=1.0,
            intercept_lr=0.001
        )
        
        # Initialize metrics
        self.mae = metrics.MAE()
        self.rmse = metrics.RMSE()
        self.r2 = metrics.R2()
        
        # Store feature importance
        self.feature_importance = {}
        
        # Store salary statistics
        self.salary_stats = {
            'min': float('inf'),
            'max': float('-inf'),
            'p95': None,
            'mean': None,
            'std': None
        }
        
        # Store error analysis
        self.error_analysis = defaultdict(lambda: {
            'count': 0,
            'mae': 0,
            'percentage_error': 0,
            'under_predictions': 0,
            'over_predictions': 0
        })
        
    def _compute_salary_stats(self, df):
        """Compute salary statistics from training data"""
        salaries = df['current_ctc'].dropna()
        self.salary_stats['min'] = float(salaries.min())
        self.salary_stats['max'] = float(salaries.max())
        self.salary_stats['p95'] = float(salaries.quantile(0.95))
        self.salary_stats['mean'] = float(salaries.mean())
        self.salary_stats['std'] = float(salaries.std())
        
        print("\nSalary Statistics from Training Data:")
        print(f"Min: {self.salary_stats['min']:.2f} LPA")
        print(f"Max: {self.salary_stats['max']:.2f} LPA")
        print(f"95th Percentile: {self.salary_stats['p95']:.2f} LPA")
        print(f"Mean: {self.salary_stats['mean']:.2f} LPA")
        print(f"Std Dev: {self.salary_stats['std']:.2f} LPA")
        
    def _bound_prediction(self, pred, yoe):
        """Bound the prediction within reasonable limits based on YOE"""
        # Base minimum salary (3 LPA)
        min_salary = max(3.0, yoe * 2)
        
        # Maximum salary based on YOE and 95th percentile
        max_salary = min(self.salary_stats['p95'], 
                        self.salary_stats['mean'] + (2 * self.salary_stats['std']))
        
        # Additional cap based on YOE
        yoe_based_max = min(yoe * 10, max_salary)
        
        # Bound the prediction
        bounded_pred = max(min_salary, min(pred, yoe_based_max))
        return bounded_pred
        
    def evaluate_and_save_predictions(self, data_path, output_path):
        """Evaluate model on all data and save predictions to CSV"""
        print("\nEvaluating model on all data...")
        
        # Read the data
        df = pd.read_csv(data_path)
        
        # Create lists to store results
        actual_values = []
        predicted_values = []
        differences = []
        
        # Make predictions for each row
        for idx, row in df.iterrows():
            # Skip rows with missing values
            if pd.isna(row['current_ctc']) or pd.isna(row['yoe']) or \
               pd.isna(row['college_type']) or pd.isna(row['company_category']):
                continue
            
            # Get prediction
            pred = self.predict(
                yoe=row['yoe'],
                college_type=row['college_type'],
                company_category=row['company_category']
            )
            
            actual_values.append(row['current_ctc'])
            predicted_values.append(pred)
            differences.append(pred - row['current_ctc'])
            
            # Print progress
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1} predictions...")
        
        # Add predictions to dataframe
        df['predicted_ctc'] = pd.Series(predicted_values, index=df.index[:len(predicted_values)])
        df['prediction_difference'] = pd.Series(differences, index=df.index[:len(differences)])
        
        # Calculate percentage difference
        df['percentage_difference'] = (df['prediction_difference'] / df['current_ctc']) * 100
        
        # Add some summary statistics
        print("\nPrediction Summary:")
        print(f"Mean Absolute Difference: {np.mean(np.abs(differences)):.2f} LPA")
        print(f"Median Absolute Difference: {np.median(np.abs(differences)):.2f} LPA")
        print(f"Mean Percentage Difference: {np.mean(np.abs(df['percentage_difference'].dropna())):.2f}%")
        print(f"Median Percentage Difference: {np.median(np.abs(df['percentage_difference'].dropna())):.2f}%")
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"\nSaved predictions to: {output_path}")
        
        return df
        
    def _analyze_prediction_error(self, actual, predicted, features):
        """Analyze prediction error for different feature combinations"""
        error = abs(predicted - actual)
        percentage_error = (error / actual) * 100
        
        # Analyze by YOE ranges
        yoe = features['yoe']
        yoe_range = f"{5 * (yoe // 5)}-{5 * (yoe // 5) + 4}" if yoe < 15 else "15+"
        
        # Update error statistics for YOE range
        yoe_stats = self.error_analysis[f"YOE_{yoe_range}"]
        yoe_stats['count'] += 1
        yoe_stats['mae'] = (yoe_stats['mae'] * (yoe_stats['count'] - 1) + error) / yoe_stats['count']
        yoe_stats['percentage_error'] = (yoe_stats['percentage_error'] * (yoe_stats['count'] - 1) + percentage_error) / yoe_stats['count']
        yoe_stats['under_predictions'] += 1 if predicted < actual else 0
        yoe_stats['over_predictions'] += 1 if predicted > actual else 0
        
        # Analyze by college type
        college_stats = self.error_analysis[f"College_{features['college_type']}"]
        college_stats['count'] += 1
        college_stats['mae'] = (college_stats['mae'] * (college_stats['count'] - 1) + error) / college_stats['count']
        college_stats['percentage_error'] = (college_stats['percentage_error'] * (college_stats['count'] - 1) + percentage_error) / college_stats['count']
        college_stats['under_predictions'] += 1 if predicted < actual else 0
        college_stats['over_predictions'] += 1 if predicted > actual else 0
        
        # Analyze by company category
        company_stats = self.error_analysis[f"Company_{features['company_category']}"]
        company_stats['count'] += 1
        company_stats['mae'] = (company_stats['mae'] * (company_stats['count'] - 1) + error) / company_stats['count']
        company_stats['percentage_error'] = (company_stats['percentage_error'] * (company_stats['count'] - 1) + percentage_error) / company_stats['count']
        company_stats['under_predictions'] += 1 if predicted < actual else 0
        company_stats['over_predictions'] += 1 if predicted > actual else 0
        
    def _preprocess_features(self, X):
        """Preprocess features using the appropriate transformers"""
        # First apply standard preprocessing
        processed = {}
        for feat, value in X.items():
            if feat in self.preprocessors:
                self.preprocessors[feat].learn_one({feat: value})
                transformed = self.preprocessors[feat].transform_one({feat: value})
                processed.update(transformed)
        
        # Then generate interaction terms
        processed = self.poly.transform_one(processed)
        return processed
        
    def train(self, data_path):
        """Train the model on the cleaned data"""
        print("\nTraining salary estimation model...")
        
        # Read the cleaned CSV file
        df = pd.read_csv(data_path)
        
        # Compute salary statistics
        self._compute_salary_stats(df)
        
        # Reset error analysis
        self.error_analysis.clear()
        
        # Train the model one sample at a time (online learning)
        for idx, row in df.iterrows():
            X = {
                'yoe': row['yoe'],
                'college_type': row['college_type'],
                'company_category': row['company_category']
            }
            y = row['current_ctc']
            
            # Skip rows with missing values or extreme outliers
            if pd.isna(y) or any(pd.isna(v) for v in X.values()):
                continue
                
            # Skip extreme outliers
            if y > self.salary_stats['p95']:
                continue
                
            # Preprocess features with interactions
            X_processed = self._preprocess_features(X)
            
            # Update the model
            self.regressor.learn_one(X_processed, y)
            
            # Get bounded prediction
            y_pred = self._bound_prediction(
                self.regressor.predict_one(X_processed),
                X['yoe']
            )
            
            # Analyze prediction error
            self._analyze_prediction_error(y, y_pred, X)
            
            # Update metrics
            self.mae.update(y, y_pred)
            self.rmse.update(y, y_pred)
            self.r2.update(y, y_pred)
            
            # Print progress every 1000 samples
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1} samples...")
                print(f"Current MAE: {self.mae.get():.2f} LPA")
                print(f"Current RMSE: {self.rmse.get():.2f} LPA")
                print(f"Current R²: {self.r2.get():.3f}")
        
        print("\nTraining completed!")
        print(f"Final MAE: {self.mae.get():.2f} LPA")
        print(f"Final RMSE: {self.rmse.get():.2f} LPA")
        print(f"Final R²: {self.r2.get():.3f}")
        
        # Print error analysis
        self._print_error_analysis()
        
        # Extract feature importance
        self.feature_importance = {
            feat: abs(weight)
            for feat, weight in self.regressor.weights.items()
        }
    
    def _print_error_analysis(self):
        """Print detailed error analysis"""
        print("\nError Analysis:")
        
        # Print YOE range analysis
        print("\nAnalysis by Years of Experience:")
        yoe_ranges = sorted([k for k in self.error_analysis.keys() if k.startswith("YOE_")])
        for yoe_range in yoe_ranges:
            stats = self.error_analysis[yoe_range]
            print(f"\n{yoe_range.replace('YOE_', '')} years:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean Absolute Error: {stats['mae']:.2f} LPA")
            print(f"  Mean Percentage Error: {stats['percentage_error']:.2f}%")
            print(f"  Under-predictions: {stats['under_predictions']} ({stats['under_predictions']/stats['count']*100:.1f}%)")
            print(f"  Over-predictions: {stats['over_predictions']} ({stats['over_predictions']/stats['count']*100:.1f}%)")
        
        # Print College Type analysis
        print("\nAnalysis by College Type:")
        college_types = sorted([k for k in self.error_analysis.keys() if k.startswith("College_")])
        for college_type in college_types:
            stats = self.error_analysis[college_type]
            print(f"\n{college_type.replace('College_', '')}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean Absolute Error: {stats['mae']:.2f} LPA")
            print(f"  Mean Percentage Error: {stats['percentage_error']:.2f}%")
            print(f"  Under-predictions: {stats['under_predictions']} ({stats['under_predictions']/stats['count']*100:.1f}%)")
            print(f"  Over-predictions: {stats['over_predictions']} ({stats['over_predictions']/stats['count']*100:.1f}%)")
        
        # Print Company Category analysis
        print("\nAnalysis by Company Category:")
        company_cats = sorted([k for k in self.error_analysis.keys() if k.startswith("Company_")])
        for company_cat in company_cats:
            stats = self.error_analysis[company_cat]
            print(f"\n{company_cat.replace('Company_', '')}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean Absolute Error: {stats['mae']:.2f} LPA")
            print(f"  Mean Percentage Error: {stats['percentage_error']:.2f}%")
            print(f"  Under-predictions: {stats['under_predictions']} ({stats['under_predictions']/stats['count']*100:.1f}%)")
            print(f"  Over-predictions: {stats['over_predictions']} ({stats['over_predictions']/stats['count']*100:.1f}%)")
    
    def predict(self, yoe, college_type, company_category):
        """Predict salary for given features"""
        X = {
            'yoe': yoe,
            'college_type': college_type,
            'company_category': company_category
        }
        X_processed = self._preprocess_features(X)
        pred = self.regressor.predict_one(X_processed)
        return self._bound_prediction(pred, yoe)
    
    def save_model(self, model_path):
        """Save the trained model to a file"""
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert model state to JSON-serializable format
        model_state = {
            'preprocessors': {
                name: {
                    'class': preprocessor.__class__.__name__,
                    'stats': preprocessor._stats if hasattr(preprocessor, '_stats') else {}
                }
                for name, preprocessor in self.preprocessors.items()
            },
            'regressor': {
                'weights': {str(k): float(v) for k, v in self.regressor.weights.items()},
                'intercept': float(self.regressor.intercept)
            },
            'metrics': {
                'mae': float(self.mae.get()),
                'rmse': float(self.rmse.get()),
                'r2': float(self.r2.get())
            },
            'feature_importance': {
                str(k): float(v) for k, v in self.feature_importance.items()
            }
        }
        
        with open(model_path, 'w') as f:
            json.dump(model_state, f, indent=2)
        print(f"\nModel saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model from a file"""
        with open(model_path, 'r') as f:
            model_state = json.load(f)
        
        # Restore preprocessors state
        for name, state in model_state['preprocessors'].items():
            if state['class'] == 'StandardScaler':
                self.preprocessors[name]._stats = state['stats']
            elif state['class'] == 'OneHotEncoder':
                self.preprocessors[name]._stats = state['stats']
        
        # Restore regressor state
        self.regressor.weights.clear()
        for k, v in model_state['regressor']['weights'].items():
            self.regressor.weights[k] = v
        self.regressor.intercept = model_state['regressor']['intercept']
        
        self.feature_importance = {
            k: v for k, v in model_state['feature_importance'].items()
        }
        
        print(f"\nModel loaded from {model_path}")
        print("Model metrics from saved state:")
        print(f"MAE: {model_state['metrics']['mae']:.2f} LPA")
        print(f"RMSE: {model_state['metrics']['rmse']:.2f} LPA")
        print(f"R²: {model_state['metrics']['r2']:.3f}")

def main():
    # Initialize the model
    estimator = SalaryEstimator()
    
    # Define paths
    data_path = "/Users/karthiksridharan/Desktop/talent_profiles_export_output_v4.csv"  # Update this path
    model_path = "models/salary_estimator.json"
    predictions_path = "/Users/karthiksridharan/Desktop/salary_predictions.csv"
    
    # Train the model
    estimator.train(data_path)
    
    # Save the trained model
    estimator.save_model(model_path)
    
    # Generate and save predictions
    estimator.evaluate_and_save_predictions(data_path, predictions_path)
    
    # Example prediction
    example_features = {
        'yoe': 5,
        'college_type': 'Tier 1',
        'company_category': 'Product-funded'
    }
    
    predicted_salary = estimator.predict(**example_features)
    print(f"\nExample prediction:")
    print(f"Features: {example_features}")
    print(f"Predicted salary: {predicted_salary:.2f} LPA")
    
    # Print feature importance
    print("\nFeature importance:")
    sorted_features = sorted(
        estimator.feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    for feature, importance in sorted_features[:10]:
        print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    main() 