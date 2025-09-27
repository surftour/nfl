"""
Linear regression models for NFL game prediction using team statistics.
"""
import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Optional, Tuple, Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

# Import data collection utilities
from .data import get_game_features_and_targets


class GamePredictionModel:
    """
    Linear regression model for predicting NFL game outcomes using team statistics.
    """
    
    def __init__(self, model_type: str = 'linear'):
        """
        Initialize the model.
        
        Args:
            model_type (str): Type of regression model ('linear', 'ridge', 'lasso')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.feature_importance_ = None
        
        # Initialize model based on type
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=1.0)
        else:
            raise ValueError("model_type must be 'linear', 'ridge', or 'lasso'")
    
    
    
    
    def normalize_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Normalize features using StandardScaler.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            
        Returns:
            np.ndarray: Scaled feature matrix
        """
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Train the model and return evaluation metrics.
        
        Args:
            X (np.ndarray): Scaled feature matrix
            y (np.ndarray): Target values
            test_size (float): Proportion of data for testing
            random_state (int): Random seed
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on model coefficients.
        
        Returns:
            pandas.DataFrame: Feature importance rankings
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model does not have coefficients")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        self.feature_importance_ = importance_df
        return importance_df
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Feature matrix (will be scaled)
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and scaler to a pickle file.
        
        Args:
            filepath (str): Path where to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type,
            'feature_importance_': self.feature_importance_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'GamePredictionModel':
        """
        Load a trained model from a pickle file.
        
        Args:
            filepath (str): Path to the saved model file
            
        Returns:
            GamePredictionModel: Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_cols = model_data['feature_cols']
        instance.feature_importance_ = model_data.get('feature_importance_')
        
        print(f"Model loaded from {filepath}")
        return instance


def build_game_prediction_model(
    years: List[int],
    target_col: str,
    weeks: Optional[List[int]] = None,
    model_type: str = 'linear',
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: Optional[str] = None
) -> Tuple[GamePredictionModel, Dict[str, float], pd.DataFrame]:
    """
    Complete pipeline to build and evaluate a game prediction model.
    
    Args:
        years (List[int]): Years to collect data from
        target_col (str): Target variable to predict
        weeks (List[int], optional): Weeks to include. Defaults to 1-17
        model_type (str): Type of regression model
        test_size (float): Test set proportion
        random_state (int): Random seed
        save_path (str, optional): Path to save the trained model
        
    Returns:
        Tuple containing:
        - GamePredictionModel: Trained model
        - Dict[str, float]: Evaluation metrics
        - pandas.DataFrame: Feature importance
    """
    print(f"Building {model_type} regression model to predict {target_col}")
    
    # Initialize model
    model = GamePredictionModel(model_type=model_type)
    
    # Get cleaned features and targets
    print("Getting features and targets...")
    X, y = get_game_features_and_targets(years, target_col, weeks)
    
    # Store feature columns in model (all columns in X are features)
    model.feature_cols = list(X.columns)
    
    print("Normalizing features...")
    X_scaled = model.normalize_data(X)
    
    print("Training model...")
    metrics = model.train(X_scaled, y[target_col].values, test_size=test_size, random_state=random_state)
    
    print("Getting feature importance...")
    importance = model.get_feature_importance()
    
    # Print results
    print("\n=== Model Performance ===")
    print(f"Test R²: {metrics['test_r2']:.3f}")
    print(f"Test MAE: {metrics['test_mae']:.2f}")
    print(f"Test MSE: {metrics['test_mse']:.2f}")
    print(f"CV R² (mean ± std): {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")
    
    print(f"\n=== Top 10 Most Important Features ===")
    print(importance.head(10)[['feature', 'coefficient']].to_string(index=False))
    
    # Save model if path is provided
    if save_path:
        model.save_model(save_path)
    
    return model, metrics, importance


# Example usage and convenience functions
def predict_visiting_team_score(years: List[int], **kwargs) -> Tuple[GamePredictionModel, Dict[str, float], pd.DataFrame]:
    """Build model to predict visiting team score."""
    return build_game_prediction_model(years, 'result_visiting_score', **kwargs)


def predict_home_team_score(years: List[int], **kwargs) -> Tuple[GamePredictionModel, Dict[str, float], pd.DataFrame]:
    """Build model to predict home team score."""
    return build_game_prediction_model(years, 'result_home_score', **kwargs)


def predict_total_points(years: List[int], **kwargs) -> Tuple[GamePredictionModel, Dict[str, float], pd.DataFrame]:
    """Build model to predict total points in game."""
    return build_game_prediction_model(years, 'result_total_points', **kwargs)


def predict_point_differential(years: List[int], **kwargs) -> Tuple[GamePredictionModel, Dict[str, float], pd.DataFrame]:
    """Build model to predict point differential (home - visiting)."""
    return build_game_prediction_model(years, 'result_point_differential', **kwargs)


if __name__ == "__main__":
    # Example usage
    print("Example: Building model to predict visiting team score")
    
    try:
        model, metrics, importance = predict_visiting_team_score(
            years=[2023, 2024],
            model_type='linear',
            weeks=list(range(1, 10))  # First 9 weeks only
        )
        
        print("\nModel training completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have game data available in the expected directory structure.")