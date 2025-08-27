import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# =============================================================================
# STEP 1: Load Your Extracted Data (FIXED PARTICIPANT ID ISSUE)
# =============================================================================

def load_extracted_data():
    """Load all the data you've already extracted with proper ID matching"""
    
    # Load participant timeseries data
    with open('participant_timeseries_data.pkl', 'rb') as f:
        participant_timeseries = pickle.load(f)
    
    # Load behavioral data with FIXED participant ID formatting
    behavioral_data = pd.read_csv('behavioral_data_properly_mapped.csv', index_col=0)
    # Convert behavioral IDs to 4-digit zero-padded format to match biometric data
    behavioral_data.index = behavioral_data.index.astype(str).str.zfill(4)
    
    # Load participant ID mapping for reference
    mapping_data = pd.read_csv('participant_id_mapping.csv')
    
    print(f"Loaded timeseries data for {len(participant_timeseries)} participants")
    print(f"Loaded behavioral data for {len(behavioral_data)} participants")
    print(f"Memory tasks available: {list(behavioral_data.columns)}")
    
    # Debug participant ID matching
    bio_ids = set(participant_timeseries.keys())
    behav_ids = set(behavioral_data.index)
    common_ids = bio_ids.intersection(behav_ids)
    
    print(f"\nParticipant ID Check:")
    print(f"Biometric IDs (first 5): {sorted(list(bio_ids))[:5]}")
    print(f"Behavioral IDs (first 5): {sorted(list(behav_ids))[:5]}")
    print(f"Common participants: {len(common_ids)}")
    
    return participant_timeseries, behavioral_data, mapping_data

# =============================================================================
# STEP 2: Neural Mechanism Feature Engineering Functions (Same as before)
# =============================================================================

def calculate_bdnf_features(participant_data, demographics):
    """
    BDNF features including sleep consistency based on BDNF-sleep plasticity connection.
    Now includes 6 features total: 4 exercise-based + 2 sleep-based
    """
    
    timeseries = participant_data['timeseries']
    total_days = len(timeseries)
    
    if total_days < 7:  # Need minimum data
        return {col: np.nan for col in ['BDNF_training_consistency_cv', 'BDNF_weeks_meeting_3x_threshold', 
                                       'BDNF_sex_adjusted_consistency', 'BDNF_optimal_session_frequency',
                                       'BDNF_sleep_duration_consistency_cv', 'BDNF_sleep_optimal_duration_adherence']}
    
    # Exercise-based BDNF features
    fb_mins_series = timeseries.get('fb_mins', pd.Series(dtype=float))
    cardio_mins_series = timeseries.get('cardio_mins', pd.Series(dtype=float))
    
    if fb_mins_series.empty or cardio_mins_series.empty:
        exercise_features = {
            'BDNF_training_consistency_cv': np.nan,
            'BDNF_weeks_meeting_3x_threshold': np.nan,
            'BDNF_sex_adjusted_consistency': np.nan,
            'BDNF_optimal_session_frequency': np.nan
        }
    else:
        # Convert to daily values
        fb_daily = fb_mins_series.fillna(0).values
        cardio_daily = cardio_mins_series.fillna(0).values
        
        # Weekly analysis
        weekly_active_days = []
        optimal_bdnf_sessions = []
        
        for week_start in range(0, len(fb_daily), 7):
            week_end = min(week_start + 7, len(fb_daily))
            week_fb = fb_daily[week_start:week_end]
            week_cardio = cardio_daily[week_start:week_end]
            
            week_active_days = 0
            for i in range(len(week_fb)):
                optimal_mins = week_fb[i] + week_cardio[i]
                if optimal_mins >= 30:  # BDNF-optimizing session
                    week_active_days += 1
                    optimal_bdnf_sessions.append(optimal_mins)
            
            weekly_active_days.append(week_active_days)
        
        if not weekly_active_days or all(days == 0 for days in weekly_active_days):
            exercise_features = {
                'BDNF_training_consistency_cv': np.nan,
                'BDNF_weeks_meeting_3x_threshold': np.nan,
                'BDNF_sex_adjusted_consistency': np.nan,
                'BDNF_optimal_session_frequency': np.nan
            }
        else:
            # Sex-moderated BDNF response
            sex = demographics.get('gender', 'unknown')
            sex_multiplier = 1.2 if sex == 'male' else 0.8
            
            # Calculate exercise features
            mean_weekly = np.mean(weekly_active_days)
            std_weekly = np.std(weekly_active_days)
            training_consistency_cv = std_weekly / mean_weekly if mean_weekly > 0 else np.nan
            weeks_meeting_3x_threshold = np.mean([days >= 3 for days in weekly_active_days])
            optimal_session_frequency = len(optimal_bdnf_sessions) / (total_days / 7)
            
            exercise_features = {
                'BDNF_training_consistency_cv': training_consistency_cv,
                'BDNF_weeks_meeting_3x_threshold': weeks_meeting_3x_threshold,
                'BDNF_sex_adjusted_consistency': training_consistency_cv * sex_multiplier,
                'BDNF_optimal_session_frequency': optimal_session_frequency
            }
    
    # Sleep-based BDNF features (BDNF mediates plasticity-related changes during sleep)
    sleep_series = timeseries.get('sleep_duration', pd.Series(dtype=float))
    
    if not sleep_series.empty and sleep_series.notna().sum() >= 7:
        sleep_durations = sleep_series.dropna().values
        sleep_duration_cv = np.std(sleep_durations) / np.mean(sleep_durations)
        optimal_nights = sum([1 for d in sleep_durations if 7 <= d <= 9])
        optimal_adherence = optimal_nights / len(sleep_durations)
        
        sleep_features = {
            'BDNF_sleep_duration_consistency_cv': sleep_duration_cv,
            'BDNF_sleep_optimal_duration_adherence': optimal_adherence
        }
    else:
        sleep_features = {
            'BDNF_sleep_duration_consistency_cv': np.nan,
            'BDNF_sleep_optimal_duration_adherence': np.nan
        }
    
    # Combine all BDNF features
    all_features = {}
    all_features.update(exercise_features)
    all_features.update(sleep_features)
    
    return all_features

def calculate_cbf_features(participant_data):
    """CBF features based on intensity-dependent responses and structural adaptations. 4 features total."""
    
    timeseries = participant_data['timeseries']
    total_days = len(timeseries)
    total_weeks = total_days / 7
    
    if total_days < 7:
        return {col: np.nan for col in ['CBF_mean_daily_optimization_score', 'CBF_intensity_consistency_cv',
                                       'CBF_structural_adaptation_eligible', 'CBF_sustained_aerobic_score']}
    
    # Get daily HR zone data
    fb_series = timeseries.get('fb_mins', pd.Series(dtype=float)).fillna(0)
    cardio_series = timeseries.get('cardio_mins', pd.Series(dtype=float)).fillna(0) 
    peak_series = timeseries.get('peak_mins', pd.Series(dtype=float)).fillna(0)
    
    daily_cbf_scores = []
    for i in range(len(fb_series)):
        fb_mins = fb_series.iloc[i] if i < len(fb_series) else 0
        cardio_mins = cardio_series.iloc[i] if i < len(cardio_series) else 0
        peak_mins = peak_series.iloc[i] if i < len(peak_series) else 0
        
        # CBF optimization based on physiological response curve
        cbf_score = (fb_mins * 1.0 +           # Linear CBF increase zone
                    cardio_mins * 0.3 +        # Diminishing CBF returns  
                    peak_mins * (-0.1))        # CBF decline
        daily_cbf_scores.append(cbf_score)
    
    # Structural adaptation (requires 12+ weeks)
    if total_weeks >= 12:
        structural_adaptation_eligible = 1
        # Calculate weekly aerobic totals
        weekly_aerobic = []
        for week_start in range(0, len(daily_cbf_scores), 7):
            week_scores = daily_cbf_scores[week_start:week_start+7] 
            weekly_total = sum([max(0, score) for score in week_scores])
            weekly_aerobic.append(weekly_total)
        
        sustained_aerobic_score = np.mean([1 for week in weekly_aerobic if week >= 150]) if weekly_aerobic else 0
    else:
        structural_adaptation_eligible = 0
        sustained_aerobic_score = 0
    
    # Calculate consistency
    mean_cbf = np.mean(daily_cbf_scores)
    std_cbf = np.std(daily_cbf_scores)
    consistency_cv = std_cbf / mean_cbf if mean_cbf > 0 else np.nan
    
    return {
        'CBF_mean_daily_optimization_score': mean_cbf,
        'CBF_intensity_consistency_cv': consistency_cv,
        'CBF_structural_adaptation_eligible': structural_adaptation_eligible,
        'CBF_sustained_aerobic_score': sustained_aerobic_score
    }

def calculate_cholinergic_features(participant_data, demographics):
    """Cholinergic features based on age-dependent responses and exercise initiation. 3 features total."""
    
    timeseries = participant_data['timeseries']
    total_days = len(timeseries)
    
    if total_days < 7:
        return {col: np.nan for col in ['CHOL_daily_initiation_frequency', 'CHOL_age_weighted_preservation_score', 'CHOL_participant_age']}
    
    # Age-dependent cholinergic vulnerability
    current_year = 2024
    birth_year = demographics.get('birthyear', current_year - 30)
    participant_age = current_year - birth_year
    
    if participant_age < 25:
        age_multiplier = 0.2
    elif participant_age < 45:
        age_multiplier = 0.6
    else:
        age_multiplier = 1.5
    
    # Get activity data
    light_series = timeseries.get('light_act_mins', pd.Series(dtype=float)).fillna(0)
    fair_series = timeseries.get('fair_act_mins', pd.Series(dtype=float)).fillna(0)
    very_series = timeseries.get('very_act_mins', pd.Series(dtype=float)).fillna(0)
    
    daily_initiations = []
    moderate_intensity_days = []
    
    for i in range(min(len(light_series), len(fair_series), len(very_series))):
        total_active_mins = light_series.iloc[i] + fair_series.iloc[i] + very_series.iloc[i]
        daily_initiations.append(1 if total_active_mins >= 30 else 0)
        
        moderate_mins = light_series.iloc[i] + fair_series.iloc[i]
        moderate_intensity_days.append(1 if moderate_mins >= 30 else 0)
    
    cholinergic_regularity = np.mean(daily_initiations) if daily_initiations else 0
    moderate_frequency = np.mean(moderate_intensity_days) if moderate_intensity_days else 0
    
    return {
        'CHOL_daily_initiation_frequency': cholinergic_regularity,
        'CHOL_age_weighted_preservation_score': moderate_frequency * age_multiplier,
        'CHOL_participant_age': participant_age
    }

def calculate_cognitive_enrichment_features(participant_data):
    """Cognitive enrichment based on HRV and activity complexity. 2 features total."""
    
    timeseries = participant_data['timeseries']
    total_days = len(timeseries)
    
    if total_days < 7:
        return {'COG_enrichment_hrv_proxy': np.nan, 'COG_weekly_activity_variety': np.nan}
    
    # Get relevant data series
    hrv_series = timeseries.get('HRV_RMSSD', pd.Series(dtype=float))
    fair_series = timeseries.get('fair_act_mins', pd.Series(dtype=float)).fillna(0)
    very_series = timeseries.get('very_act_mins', pd.Series(dtype=float)).fillna(0)
    floors_series = timeseries.get('floors', pd.Series(dtype=float)).fillna(0)
    cardio_series = timeseries.get('cardio_mins', pd.Series(dtype=float)).fillna(0)
    
    hrv_cognitive_load_scores = []
    activity_variety_scores = []
    
    # Weekly analysis
    for week_start in range(0, total_days, 7):
        week_end = min(week_start + 7, total_days)
        week_cognitive_scores = []
        week_activity_types = set()
        
        for i in range(week_start, week_end):
            # HRV-based cognitive load
            if i < len(hrv_series) and not pd.isna(hrv_series.iloc[i]):
                hrv_val = hrv_series.iloc[i]
                exercise_mins = (fair_series.iloc[i] if i < len(fair_series) else 0) + \
                              (very_series.iloc[i] if i < len(very_series) else 0)
                
                if exercise_mins >= 20 and hrv_val > 0:
                    cognitive_load_proxy = 1 / (hrv_val + 1)
                    week_cognitive_scores.append(cognitive_load_proxy)
            
            # Activity variety
            if i < len(floors_series) and floors_series.iloc[i] > 0:
                week_activity_types.add('vertical')
            if i < len(very_series) and very_series.iloc[i] > 10:
                week_activity_types.add('vigorous')  
            if i < len(cardio_series) and cardio_series.iloc[i] > 20:
                week_activity_types.add('sustained')
        
        if week_cognitive_scores:
            hrv_cognitive_load_scores.append(np.mean(week_cognitive_scores))
        activity_variety_scores.append(len(week_activity_types))
    
    return {
        'COG_enrichment_hrv_proxy': np.mean(hrv_cognitive_load_scores) if hrv_cognitive_load_scores else 0,
        'COG_weekly_activity_variety': np.mean(activity_variety_scores) if activity_variety_scores else 0
    }

# =============================================================================
# STEP 3: Integrate All Features for All Participants 
# =============================================================================

def engineer_all_features(participant_timeseries):
    """Engineer neural mechanism features for all participants. Now produces exactly 15 features."""
    
    all_features = []
    participant_ids = []
    
    print("Engineering neural mechanism features...")
    print("Expected feature count: 15 (6 BDNF + 4 CBF + 3 Cholinergic + 2 Cognitive Enrichment)")
    
    for i, (pid, data) in enumerate(participant_timeseries.items()):
        if i % 20 == 0:
            print(f"Processing participant {i+1}/{len(participant_timeseries)}")
        
        # Extract demographics
        demographics = data['demographics']
        
        # Calculate all feature sets
        features = {}
        
        try:
            # BDNF features (6 total: 4 exercise + 2 sleep)
            features.update(calculate_bdnf_features(data, demographics))
            
            # CBF features (4 total)
            features.update(calculate_cbf_features(data))
            
            # Cholinergic features (3 total)
            features.update(calculate_cholinergic_features(data, demographics))  
            
            # Cognitive enrichment features (2 total)
            features.update(calculate_cognitive_enrichment_features(data))
            
        except Exception as e:
            print(f"Error processing participant {pid}: {e}")
            # Fill with NaN values for all 15 features
            feature_names = [
                'BDNF_training_consistency_cv', 'BDNF_weeks_meeting_3x_threshold', 
                'BDNF_sex_adjusted_consistency', 'BDNF_optimal_session_frequency',
                'BDNF_sleep_duration_consistency_cv', 'BDNF_sleep_optimal_duration_adherence',
                'CBF_mean_daily_optimization_score', 'CBF_intensity_consistency_cv',
                'CBF_structural_adaptation_eligible', 'CBF_sustained_aerobic_score',
                'CHOL_daily_initiation_frequency', 'CHOL_age_weighted_preservation_score', 'CHOL_participant_age',
                'COG_enrichment_hrv_proxy', 'COG_weekly_activity_variety'
            ]
            
            features = {name: np.nan for name in feature_names}
        
        all_features.append(features)
        participant_ids.append(pid)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features, index=participant_ids)
    
    print(f"\nEngineered features for {len(features_df)} participants")
    print(f"Actual feature count: {len(features_df.columns)}")
    print(f"Feature columns: {list(features_df.columns)}")
    
    return features_df

# =============================================================================
# STEP 4: Merge with Behavioral Data and Run PCA
# =============================================================================

def merge_and_analyze(features_df, behavioral_data):
    """Merge biometric features with behavioral data and run PCA analysis."""
    
    print("\n=== MERGING BIOMETRIC FEATURES WITH BEHAVIORAL DATA ===")
    
    # Find common participants
    common_participants = features_df.index.intersection(behavioral_data.index)
    print(f"Common participants: {len(common_participants)}")
    
    if len(common_participants) < 10:
        print("ERROR: Too few common participants for meaningful analysis")
        print(f"Biometric participants (first 5): {list(features_df.index[:5])}")
        print(f"Behavioral participants (first 5): {list(behavioral_data.index[:5])}")
        return None, None
    
    # Create merged dataset
    features_common = features_df.loc[common_participants]
    behavioral_common = behavioral_data.loc[common_participants]
    
    # Select all 15 numeric features for PCA
    numeric_features = features_common.select_dtypes(include=[np.number]).columns
    print(f"Numeric features for PCA: {len(numeric_features)} features")
    print(f"Features: {list(numeric_features)}")
    
    # Prepare data for PCA
    X_biometric = features_common[numeric_features]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_biometric),
        columns=X_biometric.columns,
        index=X_biometric.index
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=X_imputed.columns,
        index=X_imputed.index
    )
    
    print(f"Final dataset shape: {X_scaled.shape}")
    
    return X_scaled, behavioral_common

def run_pca_analysis(X_scaled, behavioral_data):
    """Run PCA and create visualizations."""
    
    print("\n=== RUNNING PCA ANALYSIS ===")
    
    # Run PCA
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)
    
    # Create PCA DataFrame
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(len(pca_result[0]))],
        index=X_scaled.index
    )
    
    # Print variance explained
    print("Variance explained by each component:")
    for i, var in enumerate(pca.explained_variance_ratio_[:6]):
        print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    
    print(f"Cumulative variance (first 3 PCs): {pca.explained_variance_ratio_[:3].sum():.3f}")
    
    # Feature loadings
    loadings = pd.DataFrame(
        pca.components_[:6].T,  # First 6 PCs
        columns=[f'PC{i+1}' for i in range(6)],
        index=X_scaled.columns
    )
    
    print("\nTop feature loadings for PC1:")
    pc1_loadings = loadings['PC1'].abs().sort_values(ascending=False)
    for feature, loading in pc1_loadings.head(5).items():
        print(f"  {feature}: {loading:.3f}")
    
    # Create visualizations
    create_pca_visualizations(pca, loadings, pca_df, behavioral_data, X_scaled.columns)
    
    return pca_df, pca, loadings

def create_pca_visualizations(pca, loadings, pca_df, behavioral_data, feature_names):
    """Create PCA visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scree plot
    axes[0,0].plot(range(1, min(11, len(pca.explained_variance_ratio_)+1)), 
                   pca.explained_variance_ratio_[:10], 'bo-')
    axes[0,0].set_xlabel('Principal Component')
    axes[0,0].set_ylabel('Variance Explained')
    axes[0,0].set_title('Scree Plot - Neural Mechanism Features')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. PC1 vs PC2 scatter
    axes[0,1].scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
    axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0,1].set_title('PCA: PC1 vs PC2')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Feature loadings heatmap for first 3 PCs
    top_features = loadings.abs().sum(axis=1).sort_values(ascending=False).head(10).index
    sns.heatmap(loadings.loc[top_features, ['PC1', 'PC2', 'PC3']], 
                annot=True, cmap='RdBu_r', center=0, ax=axes[1,0])
    axes[1,0].set_title('Feature Loadings (Top 10 Features)')
    
    # 4. PC1 vs memory performance example
    memory_cols = [col for col in behavioral_data.columns if 'Free recall' in col]
    if memory_cols:
        memory_col = memory_cols[0]  # Use first available memory task
        memory_scores = behavioral_data[memory_col]
        common_idx = pca_df.index.intersection(memory_scores.index)
        
        if len(common_idx) > 5:
            axes[1,1].scatter(pca_df.loc[common_idx, 'PC1'], 
                            memory_scores.loc[common_idx], alpha=0.7)
            axes[1,1].set_xlabel('PC1 (Neural Mechanism Pattern)')
            axes[1,1].set_ylabel(f'{memory_col} Score')
            axes[1,1].set_title('PC1 vs Memory Performance')
            axes[1,1].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = pca_df.loc[common_idx, 'PC1'].corr(memory_scores.loc[common_idx])
            axes[1,1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1,1].transAxes,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[1,1].text(0.5, 0.5, 'Insufficient data\nfor memory correlation', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
    else:
        axes[1,1].text(0.5, 0.5, 'No memory tasks found', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    plt.savefig('neural_mechanism_pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'neural_mechanism_pca_analysis.png'")

# =============================================================================
# STEP 5: Main Execution Function  
# =============================================================================

def main():
    """Complete pipeline execution."""
    
    print("NEURAL MECHANISM-INFORMED BIOMETRIC ANALYSIS")
    print("=" * 50)
    
    # Step 1: Load extracted data with fixed participant ID matching
    participant_timeseries, behavioral_data, mapping_data = load_extracted_data()
    
    # Step 2: Engineer all neural mechanism features (15 total)
    features_df = engineer_all_features(participant_timeseries)
    
    # Verify feature count
    if len(features_df.columns) != 15:
        print(f"WARNING: Expected 15 features, got {len(features_df.columns)}")
        print(f"Features: {list(features_df.columns)}")
    
    # Step 3: Merge with behavioral data
    X_scaled, behavioral_common = merge_and_analyze(features_df, behavioral_data)
    
    if X_scaled is None:
        print("Analysis failed due to insufficient data overlap")
        return
    
    # Step 4: Run PCA analysis
    pca_df, pca_model, loadings = run_pca_analysis(X_scaled, behavioral_common)
    
    # Step 5: Test correlations with memory tasks
    print("\n=== TESTING MEMORY CORRELATIONS ===")
    
    memory_correlations = {}
    for memory_task in behavioral_common.columns:
        if behavioral_common[memory_task].notna().sum() > 10:
            correlation = pca_df['PC1'].corr(behavioral_common[memory_task])
            memory_correlations[memory_task] = correlation
            print(f"PC1 vs {memory_task}: r = {correlation:.3f}")
    
    # Feature interpretation by mechanism
    print("\n=== FEATURE LOADINGS BY NEURAL MECHANISM ===")
    bdnf_features = [col for col in loadings.index if col.startswith('BDNF')]
    cbf_features = [col for col in loadings.index if col.startswith('CBF')]
    chol_features = [col for col in loadings.index if col.startswith('CHOL')]
    cog_features = [col for col in loadings.index if col.startswith('COG')]
    
    print(f"BDNF mechanism features ({len(bdnf_features)}): {bdnf_features}")
    print(f"CBF mechanism features ({len(cbf_features)}): {cbf_features}")
    print(f"Cholinergic mechanism features ({len(chol_features)}): {chol_features}")
    print(f"Cognitive enrichment features ({len(cog_features)}): {cog_features}")
    
    # Save results
    features_df.to_csv('neural_mechanism_features.csv')
    pca_df.to_csv('pca_components.csv')
    loadings.to_csv('pca_feature_loadings.csv')
    
    print(f"\nResults saved:")
    print("- neural_mechanism_features.csv (15 mechanism-informed features)")
    print("- pca_components.csv (principal components)") 
    print("- pca_feature_loadings.csv (feature contributions to PCs)")
    print("- neural_mechanism_pca_analysis.png (visualizations)")
    
    return features_df, pca_df, loadings, behavioral_common

# Run the complete analysis
if __name__ == "__main__":
    results = main()