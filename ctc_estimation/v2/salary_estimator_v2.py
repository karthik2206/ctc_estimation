import pandas as pd
from river import linear_model, preprocessing, compose, metrics, optim, feature_extraction
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
import math

class SalaryEstimator:
    def __init__(self):
        # Create preprocessing steps for each feature
        self.preprocessors = {
            'yoe': preprocessing.StandardScaler(),
            'college_type': preprocessing.OneHotEncoder(),
            'company_type': preprocessing.OneHotEncoder(),
            'funding_status': preprocessing.OneHotEncoder(),
            'city_category': preprocessing.OneHotEncoder(),
            'specialization': preprocessing.OneHotEncoder()
        }
        
        # Create polynomial feature extractor for interactions
        self.poly = feature_extraction.PolynomialExtender(degree=2, interaction_only=True)
        
        # Create the regressor with original working configuration
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
            'std': None,
            'percentiles': {}
        }
        
        # Store error analysis
        self.error_analysis = defaultdict(lambda: {
            'count': 0,
            'mae': 0,
            'percentage_error': 0,
            'under_predictions': 0,
            'over_predictions': 0,
            'problematic_records': []  # Store indices of problematic records
        })
        
        # Store problematic records
        self.problematic_records = []
        
    def _create_advanced_features(self, X):
        """Create advanced features from basic features"""
        features = X.copy()
        
        # Keep features simple like the original version
        features['is_product'] = 1 if features['company_type'] == 'Product' else 0
        features['is_funded'] = 1 if features['funding_status'] == 'Funded' else 0
        
        return features
        
    def _is_problematic_record(self, actual, predicted, features):
        """Identify if a record is problematic based on prediction error"""
        error = abs(predicted - actual)
        percentage_error = (error / actual) * 100
        
        # Calculate expected salary range based on features
        yoe = features['yoe']
        expected_min = max(3.0, yoe * 2)  # Base minimum salary
        expected_max = self.salary_stats['p95']  # Use 95th percentile as max
        
        # Conditions for problematic records
        conditions = [
            percentage_error > 40,  # High percentage error
            actual < expected_min,  # Salary too low for experience
            actual > expected_max,  # Salary too high for experience
            abs(actual - self.salary_stats['mean']) > 2.5 * self.salary_stats['std']  # Statistical outlier
        ]
        
        return any(conditions)
        
    def _analyze_problematic_records(self, df):
        """Analyze patterns in problematic records"""
        prob_df = pd.DataFrame(self.problematic_records)
        if len(prob_df) == 0:
            print("\nNo problematic records found.")
            return
            
        print("\nAnalysis of Problematic Records:")
        print(f"Total problematic records: {len(prob_df)}")
        
        # Analyze by feature
        for feature in ['college_type', 'company_type', 'funding_status', 'city_category', 'specialization']:
            print(f"\nProblematic records by {feature}:")
            value_counts = prob_df[feature].value_counts()
            percentages = value_counts / len(prob_df) * 100
            for val, count in value_counts.items():
                print(f"  {val}: {count} ({percentages[val]:.1f}%)")
        
        # Analyze YOE distribution
        print("\nYOE distribution in problematic records:")
        print(prob_df['yoe'].describe())
        
        # Analyze salary distribution
        print("\nSalary distribution in problematic records:")
        print(prob_df['actual_ctc'].describe())
        
        # Find common patterns
        print("\nCommon patterns in problematic records:")
        pattern_counts = prob_df.groupby(
            ['college_type', 'company_type', 'funding_status']
        ).size().sort_values(ascending=False).head(5)
        for pattern, count in pattern_counts.items():
            print(f"  {' + '.join(pattern)}: {count} records")
        
    def _compute_salary_stats(self, df):
        """Compute salary statistics from training data"""
        salaries = df['current_ctc'].dropna()
        self.salary_stats['min'] = float(salaries.min())
        self.salary_stats['max'] = float(salaries.max())
        self.salary_stats['p95'] = float(salaries.quantile(0.95))
        self.salary_stats['mean'] = float(salaries.mean())
        self.salary_stats['std'] = float(salaries.std())
        
        # Compute percentiles for more granular analysis
        for p in range(5, 100, 5):
            self.salary_stats['percentiles'][p] = float(salaries.quantile(p/100))
        
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
        
        # Analyze by company type and funding
        company_stats = self.error_analysis[f"Company_{features['company_type']}_{features['funding_status']}"]
        company_stats['count'] += 1
        company_stats['mae'] = (company_stats['mae'] * (company_stats['count'] - 1) + error) / company_stats['count']
        company_stats['percentage_error'] = (company_stats['percentage_error'] * (company_stats['count'] - 1) + percentage_error) / company_stats['count']
        company_stats['under_predictions'] += 1 if predicted < actual else 0
        company_stats['over_predictions'] += 1 if predicted > actual else 0
        
    def _preprocess_features(self, X):
        """Preprocess features using the appropriate transformers"""
        # First create advanced features
        X_advanced = self._create_advanced_features(X)
        
        # Then apply standard preprocessing
        processed = {}
        for feat, value in X_advanced.items():
            if feat in self.preprocessors:
                self.preprocessors[feat].learn_one({feat: value})
                transformed = self.preprocessors[feat].transform_one({feat: value})
                processed.update(transformed)
            else:
                # For numerical features we created
                processed[feat] = value
        
        # Then generate interaction terms
        processed = self.poly.transform_one(processed)
        return processed
        
    def train(self, data_path):
        """Train the model on the cleaned data"""
        print("\nTraining salary estimation model...")
        
        # Read the cleaned CSV file
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records from cleaned dataset")
        
        # Compute salary statistics
        self._compute_salary_stats(df)
        
        # Reset error analysis and problematic records
        self.error_analysis.clear()
        self.problematic_records.clear()
        
        # First pass: identify problematic records
        print("\nIdentifying problematic records...")
        problematic_indices = set()
        for idx, row in df.iterrows():
            X = {
                'yoe': row['yoe'],
                'college_type': row['college_type'],
                'company_type': row['company_type'],
                'funding_status': row['funding_status'],
                'city_category': row['city_category'],
                'specialization': row['specialization']
            }
            y = row['current_ctc']
            
            # Skip rows with missing values or extreme outliers
            if pd.isna(y) or any(pd.isna(list(X.values()))):
                continue
                
            # Skip extreme outliers like in original version
            if y > self.salary_stats['p95']:
                continue
            
            # Preprocess features with interactions
            X_processed = self._preprocess_features(X)
            
            # Get bounded prediction
            y_pred = self._bound_prediction(
                self.regressor.predict_one(X_processed),
                X['yoe']
            )
            
            # Check if record is problematic
            if self._is_problematic_record(y, y_pred, X):
                problematic_indices.add(idx)
                prob_record = {**X, 'actual_ctc': y, 'predicted_ctc': y_pred}
                self.problematic_records.append(prob_record)
        
        print(f"Found {len(problematic_indices)} problematic records")
        
        # Reset the model and preprocessors for actual training
        self.__init__()
        self._compute_salary_stats(df)
        
        # Second pass: train on non-problematic records only
        print("\nTraining on clean records...")
        processed_count = 0
        for idx, row in df.iterrows():
            if idx in problematic_indices:
                continue
                
            X = {
                'yoe': row['yoe'],
                'college_type': row['college_type'],
                'company_type': row['company_type'],
                'funding_status': row['funding_status'],
                'city_category': row['city_category'],
                'specialization': row['specialization']
            }
            y = row['current_ctc']
            
            # Skip rows with missing values
            if pd.isna(y) or any(pd.isna(list(X.values()))):
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
            
            processed_count += 1
            # Print progress every 1000 samples
            if processed_count % 1000 == 0:
                print(f"Processed {processed_count} samples...")
                print(f"Current MAE: {self.mae.get():.2f} LPA")
                print(f"Current RMSE: {self.rmse.get():.2f} LPA")
                print(f"Current R²: {self.r2.get():.3f}")
        
        print("\nTraining completed!")
        print(f"Final MAE: {self.mae.get():.2f} LPA")
        print(f"Final RMSE: {self.rmse.get():.2f} LPA")
        print(f"Final R²: {self.r2.get():.3f}")
        
        # Analyze problematic records
        self._analyze_problematic_records(df)
        
        # Print error analysis
        self._print_error_analysis()
        
    def _print_error_analysis(self):
        """Print detailed error analysis"""
        print("\nError Analysis by Feature Groups:")
        
        # Sort error analysis by MAE
        sorted_analysis = sorted(
            self.error_analysis.items(),
            key=lambda x: x[1]['mae'],
            reverse=True
        )
        
        # Print analysis for each feature group
        for feature_group, stats in sorted_analysis:
            if stats['count'] < 50:  # Skip groups with too few samples
                continue
                
            print(f"\n{feature_group}:")
            print(f"  Samples: {stats['count']}")
            print(f"  MAE: {stats['mae']:.2f} LPA")
            print(f"  Percentage Error: {stats['percentage_error']:.2f}%")
            print(f"  Under-predictions: {stats['under_predictions']} ({stats['under_predictions']/stats['count']*100:.1f}%)")
            print(f"  Over-predictions: {stats['over_predictions']} ({stats['over_predictions']/stats['count']*100:.1f}%)")
        
    def predict(self, yoe, college_type, company_type, funding_status, city_category, specialization):
        """Make a salary prediction for given features"""
        X = {
            'yoe': yoe,
            'college_type': college_type,
            'company_type': company_type,
            'funding_status': funding_status,
            'city_category': city_category,
            'specialization': specialization
        }
        
        # Preprocess features
        X_processed = self._preprocess_features(X)
        
        # Get prediction and bound it
        pred = self.regressor.predict_one(X_processed)
        bounded_pred = self._bound_prediction(pred, yoe)
        
        return bounded_pred
        
    def save_model(self, model_path):
        """Save the trained model to a file"""
        model_data = {
            'preprocessors': {
                name: {
                    'class': preprocessor.__class__.__name__,
                    'stats': preprocessor._stats if hasattr(preprocessor, '_stats') else {}
                }
                for name, preprocessor in self.preprocessors.items()
            },
            'regressor': {
                'weights': self.regressor.weights,
                'intercept': self.regressor.intercept
            },
            'metrics': {
                'mae': self.mae.get(),
                'rmse': self.rmse.get(),
                'r2': self.r2.get()
            },
            'salary_stats': self.salary_stats
        }
        
        with open(model_path, 'w') as f:
            json.dump(model_data, f)
            
        print(f"\nModel saved to {model_path}")
        
    def load_model(self, model_path):
        """Load a trained model from a file"""
        with open(model_path, 'r') as f:
            model_data = json.load(f)
            
        # Load preprocessors
        for name, state in model_data['preprocessors'].items():
            if state['class'] == 'StandardScaler':
                self.preprocessors[name] = preprocessing.StandardScaler()
                self.preprocessors[name]._stats = state['stats']
            elif state['class'] == 'OneHotEncoder':
                self.preprocessors[name] = preprocessing.OneHotEncoder()
                self.preprocessors[name]._stats = state['stats']
            
        # Load regressor weights and intercept
        self.regressor.weights.clear()
        self.regressor.weights.update(model_data['regressor']['weights'])
        self.regressor.intercept = model_data['regressor']['intercept']
        
        # Load salary statistics
        self.salary_stats = model_data['salary_stats']
        
        print(f"\nModel loaded from {model_path}")
        print("Model metrics from saved state:")
        print(f"MAE: {model_data['metrics']['mae']:.2f} LPA")
        print(f"RMSE: {model_data['metrics']['rmse']:.2f} LPA")
        print(f"R²: {model_data['metrics']['r2']:.3f}")

def main():
    # Initialize the model
    model = SalaryEstimator()
    
    # Define paths
    input_file = "/Users/karthiksridharan/Desktop/talent_profiles_export_output_v4.csv"  # Use the original input file
    script_dir = Path(__file__).parent
    model_path = script_dir / "models" / "salary_estimator_v2.json"
    
    # Train the model
    model.train(input_file)
    
    # Save the trained model
    model_path.parent.mkdir(exist_ok=True)  # Ensure models directory exists
    model.save_model(model_path)
    
    # Example prediction
    example = {
        'yoe': 5,
        'college_type': 'Tier 1',
        'company_type': 'Product',
        'funding_status': 'Funded',
        'city_category': 'Metro',
        'specialization': 'Backend Developer'
    }
    
    predicted_salary = model.predict(**example)
    print(f"\nExample prediction:")
    print(f"Features: {example}")
    print(f"Predicted salary: {predicted_salary:.2f} LPA")

if __name__ == "__main__":
    main() 