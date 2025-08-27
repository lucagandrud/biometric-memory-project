# src/analysis/ml_pipeline.py
"""
ML Analysis Pipeline for Biometric-Cognitive Performance Project
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class BiometricAnalyzer:
    """
    Main analysis class for biometric-cognitive performance research
    """
    
    def __init__(self, data_path="../data/brainfit-paper/data"):
        self.data_path = Path(data_path)
        self.behavioral_data = None
        self.fitness_data = None
        self.combined_data = None
        self.scaler = StandardScaler()
        
    def load_manning_data(self):
        """Load the Manning dataset"""
        print("Loading Manning Dataset...")
        
        # Load behavioral data
        with open(self.data_path / "behavioral_summary.pkl", 'rb') as f:
            self.behavioral_data = pickle.load(f)
            
        # Load fitness data  
        with open(self.data_path / "fitness_summary.pkl", 'rb') as f:
            self.fitness_data = pickle.load(f)
            
        print("Data loaded successfully!")
        self._explore_data_structure()
        
    def _explore_data_structure(self):
        """Understand the data structure"""
        print("\nBEHAVIORAL DATA STRUCTURE:")
        print(f"Type: {type(self.behavioral_data)}")
        if isinstance(self.behavioral_data, dict):
            for key, value in self.behavioral_data.items():
                print(f"  {key}: {type(value)} - {np.array(value).shape if hasattr(value, '__len__') else 'scalar'}")
                
        print("\nFITNESS DATA STRUCTURE:")
        print(f"Type: {type(self.fitness_data)}")
        if isinstance(self.fitness_data, dict):
            for key, value in self.fitness_data.items():
                print(f"  {key}: {type(value)} - {np.array(value).shape if hasattr(value, '__len__') else 'scalar'}")
    
    def engineer_consistency_features(self, participant_data):
        """
        Engineer activity consistency features for hypothesis testing
        
        Args:
            participant_data: Individual participant's activity data
            
        Returns:
            dict: Consistency metrics
        """
        features = {}
        
        # Weekly activity variance (lower = more consistent)
        if 'daily_steps' in participant_data:
            steps = np.array(participant_data['daily_steps'])
            weekly_steps = steps.reshape(-1, 7).sum(axis=1)  # Weekly totals
            features['weekly_activity_variance'] = np.var(weekly_steps)
            features['activity_consistency_score'] = 1 / (1 + np.var(weekly_steps))
            
        # Training frequency stability
        if 'active_days' in participant_data:
            active_days = np.array(participant_data['active_days'])
            # Calculate weekly active day counts
            weekly_active = active_days.reshape(-1, 7).sum(axis=1)
            features['training_frequency_stability'] = 1 / (1 + np.var(weekly_active))
            
        # HRV pattern consistency (if available)
        if 'hrv_daily' in participant_data:
            hrv_values = np.array(participant_data['hrv_daily'])
            features['hrv_consistency'] = 1 / (1 + np.var(hrv_values))
            
        return features
    
    def identify_top_performers(self, performance_threshold=75):
        """
        Identify top 25% performers in each cognitive domain
        
        Args:
            performance_threshold: Percentile threshold (default 75 for top 25%)
            
        Returns:
            dict: Top performer indices for each task
        """
        top_performers = {}
        
        # This will need to be adapted based on actual data structure
        # Placeholder for now - will update after data exploration
        
        if isinstance(self.behavioral_data, dict):
            for task, scores in self.behavioral_data.items():
                if isinstance(scores, (list, np.ndarray)):
                    threshold = np.percentile(scores, performance_threshold)
                    top_performers[task] = np.where(np.array(scores) >= threshold)[0]
                    
        return top_performers
    
    def run_pca_analysis(self, features, n_components=5):
        """
        Run PCA analysis on biometric features
        
        Args:
            features: Feature matrix (participants x features)
            n_components: Number of principal components
            
        Returns:
            dict: PCA results
        """
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Run PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(features_scaled)
        
        # Results dictionary
        results = {
            'components': components,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'feature_loadings': pca.components_,
            'pca_model': pca
        }
        
        return results
    
    def create_consistency_hypothesis_test(self):
        """
        Main function to test consistency hypothesis
        """
        print("Testing Consistency Hypothesis...")
        
        # Step 1: Load data
        if self.behavioral_data is None:
            self.load_manning_data()
            
        # Step 2: Engineer consistency features
        # (This will need participant-level data processing)
        
        # Step 3: Identify top performers
        top_performers = self.identify_top_performers()
        
        # Step 4: Compare consistency patterns
        # (Implementation depends on data structure)
        
        print("Hypothesis test framework ready - needs data structure adaptation")
        
        return top_performers

def create_sample_analysis():
    """
    Sample analysis to get started
    """
    analyzer = BiometricAnalyzer()
    analyzer.load_manning_data()
    
    # Test the framework
    top_performers = analyzer.create_consistency_hypothesis_test()
    
    return analyzer, top_performers

if __name__ == "__main__":
    analyzer, results = create_sample_analysis()
    