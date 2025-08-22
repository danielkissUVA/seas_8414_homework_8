# train_model.py
from typing import Dict, Any

import pandas as pd
import numpy as np
# Import both classification and clustering modules distinctly
from pycaret.classification import setup as setup_clf, compare_models, finalize_model, save_model as save_clf_model, plot_model as plot_clf_model
from pycaret.clustering import setup as setup_clu, create_model as create_clu_model, save_model as save_clu_model, plot_model as plot_clu_model
import os

def generate_synthetic_data(num_samples: int = 500) -> pd.DataFrame:
    """
    Generates a synthetic dataset of URL features for phishing detection and threat attribution.

    The function creates a balanced dataset consisting of two main categories:
    - Benign URLs: Represents legitimate, safe websites.
    - Phishing URLs: These are further sub-divided into three distinct threat actor profiles
      to simulate different attack methodologies.

    The features are designed to mimic a real-world phishing detection dataset, with values
    represented as 1 (positive indicator of phishing), 0 (neutral), or -1 (negative indicator).

    Args:
        num_samples (int): The total number of phishing URLs to generate. The number of
                           benign URLs will be equal to this number. The phishing URLs
                           will be split equally among the three threat profiles.

    Returns:
        pd.DataFrame: A shuffled DataFrame containing the synthetic data with a 'label'
                      column for classification and a 'threat_profile' column for clustering.
    """
    print("Generating synthetic dataset...")

    # Define the list of features to be used in the synthetic data.
    features = [
        'having_IP_Address', 'URL_Length', 'Shortining_Service',
        'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
        'having_Sub_Domain', 'SSLfinal_State', 'URL_of_Anchor', 'Links_in_tags',
        'SFH', 'Abnormal_URL'
    ]

    # Calculate the number of samples for each threat profile.
    num_per_profile = num_samples // 3
    # Set the number of benign samples equal to the total number of phishing samples.
    num_benign = num_samples

    # --- Threat Profile Definitions ---
    # Each profile is a dictionary defining the probability distribution for each feature.
    # The keys are feature names and the values are probability weights for [-1, 0, 1].

    # Profile 1: State-Sponsored (Subtle and Sophisticated)
    # This profile simulates advanced, targeted attacks.
    state_sponsored_data: Dict[str, Any] = {
        'having_IP_Address': np.random.choice([1, -1], num_per_profile, p=[0.05, 0.95]),  # Rarely use IPs
        'URL_Length': np.random.choice([1, 0, -1], num_per_profile, p=[0.2, 0.7, 0.1]),  # Normal length URLs
        'Shortining_Service': np.random.choice([1, -1], num_per_profile, p=[0.1, 0.9]),  # Rarely use shorteners
        'having_At_Symbol': np.random.choice([1, -1], num_per_profile, p=[0.1, 0.9]),
        'double_slash_redirecting': np.random.choice([1, -1], num_per_profile, p=[0.1, 0.9]),
        'Prefix_Suffix': np.random.choice([1, -1], num_per_profile, p=[0.9, 0.1]),  # High use of deceptive subdomains
        'having_Sub_Domain': np.random.choice([1, 0, -1], num_per_profile, p=[0.8, 0.1, 0.1]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], num_per_profile, p=[0.05, 0.05, 0.9]),  # Almost always use valid SSL
        'URL_of_Anchor': np.random.choice([-1, 0, 1], num_per_profile, p=[0.6, 0.2, 0.2]),  # Deceptive anchors
        'Links_in_tags': np.random.choice([-1, 0, 1], num_per_profile, p=[0.5, 0.3, 0.2]),
        'SFH': np.random.choice([-1, 0, 1], num_per_profile, p=[0.1, 0.8, 0.1]),  # Often points to legitimate domains
        'Abnormal_URL': np.random.choice([1, -1], num_per_profile, p=[0.2, 0.8]),
        'has_political_keyword': np.random.choice([1, -1], num_per_profile, p=[0.1, 0.9])  # Rarely political
    }

    # Profile 2: Organized Cybercrime (High-Volume and Noisy)
    # This profile simulates widespread, financially motivated attacks.
    organized_crime_data: Dict[str, Any] = {
        'having_IP_Address': np.random.choice([1, -1], num_per_profile, p=[0.8, 0.2]),  # Frequently use IPs
        'URL_Length': np.random.choice([1, 0, -1], num_per_profile, p=[0.7, 0.2, 0.1]),  # Often long URLs
        'Shortining_Service': np.random.choice([1, -1], num_per_profile, p=[0.9, 0.1]),  # Heavily rely on shorteners
        'having_At_Symbol': np.random.choice([1, -1], num_per_profile, p=[0.6, 0.4]),
        'double_slash_redirecting': np.random.choice([1, -1], num_per_profile, p=[0.7, 0.3]),
        'Prefix_Suffix': np.random.choice([1, -1], num_per_profile, p=[0.8, 0.2]),
        'having_Sub_Domain': np.random.choice([1, 0, -1], num_per_profile, p=[0.7, 0.2, 0.1]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], num_per_profile, p=[0.8, 0.15, 0.05]),  # Rarely use valid SSL
        'URL_of_Anchor': np.random.choice([-1, 0, 1], num_per_profile, p=[0.8, 0.1, 0.1]),
        'Links_in_tags': np.random.choice([-1, 0, 1], num_per_profile, p=[0.7, 0.2, 0.1]),
        'SFH': np.random.choice([-1, 0, 1], num_per_profile, p=[0.9, 0.05, 0.05]),  # Submitted to many blacklists
        'Abnormal_URL': np.random.choice([1, -1], num_per_profile, p=[0.9, 0.1]),  # Often abnormal URLs
        'has_political_keyword': np.random.choice([1, -1], num_per_profile, p=[0.05, 0.95]) # Rarely political
    }

    # Profile 3: Hacktivist (Opportunistic and Thematic)
    # This profile simulates attacks driven by ideological or political motives.
    hacktivist_data: Dict[str, Any] = {
        'having_IP_Address': np.random.choice([1, -1], num_per_profile, p=[0.3, 0.7]),
        'URL_Length': np.random.choice([1, 0, -1], num_per_profile, p=[0.4, 0.4, 0.2]),
        'Shortining_Service': np.random.choice([1, -1], num_per_profile, p=[0.5, 0.5]),
        'having_At_Symbol': np.random.choice([1, -1], num_per_profile, p=[0.3, 0.7]),
        'double_slash_redirecting': np.random.choice([1, -1], num_per_profile, p=[0.4, 0.6]),
        'Prefix_Suffix': np.random.choice([1, -1], num_per_profile, p=[0.6, 0.4]),
        'having_Sub_Domain': np.random.choice([1, 0, -1], num_per_profile, p=[0.5, 0.3, 0.2]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], num_per_profile, p=[0.5, 0.4, 0.1]),  # Mix of SSL usage
        'URL_of_Anchor': np.random.choice([-1, 0, 1], num_per_profile, p=[0.5, 0.3, 0.2]),
        'Links_in_tags': np.random.choice([-1, 0, 1], num_per_profile, p=[0.5, 0.3, 0.2]),
        'SFH': np.random.choice([-1, 0, 1], num_per_profile, p=[0.6, 0.3, 0.1]),
        'Abnormal_URL': np.random.choice([1, -1], num_per_profile, p=[0.4, 0.6]),
        'has_political_keyword': np.random.choice([1, -1], num_per_profile, p=[0.9, 0.1]) # High chance of political keywords
    }

    # Benign Data (as a baseline)
    # This simulates legitimate, safe URLs.
    benign_data: Dict[str, Any] = {
        'having_IP_Address': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'URL_Length': np.random.choice([1, 0, -1], num_benign, p=[0.1, 0.6, 0.3]),
        'Shortining_Service': np.random.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'having_At_Symbol': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'double_slash_redirecting': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'Prefix_Suffix': np.random.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'having_Sub_Domain': np.random.choice([1, 0, -1], num_benign, p=[0.1, 0.4, 0.5]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], num_benign, p=[0.05, 0.15, 0.8]),
        'URL_of_Anchor': np.random.choice([-1, 0, 1], num_benign, p=[0.1, 0.2, 0.7]),
        'Links_in_tags': np.random.choice([-1, 0, 1], num_benign, p=[0.1, 0.2, 0.7]),
        'SFH': np.random.choice([-1, 0, 1], num_benign, p=[0.1, 0.1, 0.8]),
        'Abnormal_URL': np.random.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'has_political_keyword': np.random.choice([1, -1], num_benign, p=[0.01, 0.99])  # Very low chance
    }

    # Create DataFrames from the generated data dictionaries.
    df_state = pd.DataFrame(state_sponsored_data)
    df_state['threat_profile'] = 'State-Sponsored'

    df_crime = pd.DataFrame(organized_crime_data)
    df_crime['threat_profile'] = 'Organized Cybercrime'

    df_hacktivist = pd.DataFrame(hacktivist_data)
    df_hacktivist['threat_profile'] = 'Hacktivist'

    df_benign = pd.DataFrame(benign_data)
    df_benign['threat_profile'] = 'Benign'

    # Combine the three phishing threat profiles into a single DataFrame.
    df_phishing = pd.concat([df_state, df_crime, df_hacktivist], ignore_index=True)
    # Assign a classification label of 1 to all phishing URLs.
    df_phishing['label'] = 1

    # Assign a classification label of 0 to all benign URLs.
    df_benign['label'] = 0

    # Combine the phishing and benign DataFrames to create the final dataset.
    final_df = pd.concat([df_phishing, df_benign], ignore_index=True)
    # Shuffle the combined DataFrame to randomize the order of the samples.
    return final_df.sample(frac=1).reset_index(drop=True)

def train():
    """
    Trains and saves a machine learning pipeline for a multi-stage cybersecurity threat analysis.

    This function first trains a classification model to detect phishing URLs and then
    applies a clustering model to profile and attribute detected phishing threats.

    The workflow includes the following steps:
    1. **Data Generation:** Synthetic data is generated to simulate phishing and legitimate URLs.
    2. **Phishing Detection (Classification Workflow):**
       - Uses PyCaret to train a classification model (Random Forest, Extra Trees, or LightGBM).
       - Selects the best-performing model to detect phishing URLs based on a 'label' target column.
       - The final model is saved to `models/phishing_url_detector.pkl`.
       - A feature importance plot is generated and saved as `models/feature_importance.png`.
    3. **Threat Profiling (Clustering Workflow):**
       - The dataset is filtered to include only the detected phishing URLs.
       - A K-Modes clustering model is trained on these phishing URLs to group them into distinct
         threat profiles.
       - The clustering model is saved to `models/threat_actor_profiler.pkl`.
       - A cluster plot is generated and saved as `models/cluster_plot.png`.

    The function checks for existing model artifacts (`.pkl` files) and skips the training process
    if they are already present, preventing redundant work.
    """
    # Define file paths for the classification and clustering models, and the feature importance plot.
    classification_model_path = 'models/phishing_url_detector'
    clustering_model_path = 'models/threat_actor_profiler'
    plot_path = 'models/feature_importance.png'

    # Check if both model files already exist. If they do, print a message and exit the function.
    if os.path.exists(classification_model_path + '.pkl') and os.path.exists(clustering_model_path + '.pkl'):
        print("All models and plots already exist. Skipping training.")
        return

    # Generate a synthetic dataset for training. This data mimics real-world phishing and legitimate URLs.
    data = generate_synthetic_data()

    # Create the 'data' directory if it doesn't already exist.
    os.makedirs('data', exist_ok=True)

    # Save the newly generated synthetic data to a CSV file for future use or analysis.
    data.to_csv('data/phishing_synthetic_profiles.csv', index=False)

    # --- 1. CLASSIFICATION WORKFLOW ---
    print("\n--- Starting Classification Workflow ---")

    print("Initializing PyCaret Classification Setup...")
    # Initialize the PyCaret classification environment. It prepares the data for model training
    # by setting the target variable ('label'), specifying a session ID for reproducibility,
    # and ignoring the 'threat_profile' feature since it's used only for the clustering phase.
    s_class = setup_clf(data, target='label', session_id=42, verbose=False, ignore_features=['threat_profile'])

    print("Comparing classification models...")
    # Train and compare a selection of popular classification models (Random Forest, Extra Trees, and LightGBM)
    # to find the best-performing one based on default evaluation metrics.
    best_model = compare_models(n_select=1, include=['rf', 'et', 'lightgbm'])

    print("Finalizing classification model...")
    # Finalize the best-performing model by retraining it on the entire dataset. This is a common
    # practice to ensure the model learns from all available data, not just the training subset.
    final_model = finalize_model(best_model)

    print("Saving feature importance plot...")
    # Create the 'models' directory if it doesn't exist.
    os.makedirs('models', exist_ok=True)

    # Generate and save a plot that visualizes the importance of each feature in the final model.
    # This helps in understanding which URL attributes are most predictive of phishing.
    plot_clf_model(final_model, plot='feature', save=True)

    # PyCaret saves the plot with a generic name ('Feature Importance.png').
    # This line renames it to a more specific path.
    if os.path.exists('Feature Importance.png'):
        os.rename('Feature Importance.png', plot_path)

    print("Saving classification model...")
    # Save the finalized classification model as a pickle file for later use in prediction.
    save_clf_model(final_model, classification_model_path)
    print("Classification model and plot saved successfully.")

    # --- 2. CLUSTERING WORKFLOW FOR ATTRIBUTION ---
    print("\n--- Starting Clustering Workflow for Threat Attribution ---")

    # Isolate only the data points that were identified as phishing URLs (where 'label' is 1).
    # This ensures that clustering is performed only on the malicious URLs.
    phishing_data = data[data['label'] == 1].copy()

    # For clustering, we only need the features. We drop the 'label' and 'threat_profile'
    # columns as they are not needed for this unsupervised learning task.
    phishing_features = phishing_data.drop(['label', 'threat_profile'], axis=1)

    print("Initializing PyCaret Clustering Setup...")
    # Initialize the PyCaret clustering environment.
    s_clust = setup_clu(phishing_features, session_id=42, verbose=False)

    print("Creating K-Modes clustering model...")
    # Create a K-Modes clustering model with a specified number of clusters (in this case, 3).
    # K-Modes is particularly suitable for categorical data, which is common in URL features.
    kmodes = create_clu_model('kmodes', num_clusters=3)

    print("Saving clustering model...")
    # Save the trained clustering model for later use in attributing new phishing URLs to existing
    # threat profiles.
    save_clu_model(kmodes, clustering_model_path)
    print("Clustering model saved successfully.")

    cluster_plot_path = 'models/cluster_plot.png'
    print(f"Saving cluster plot to {cluster_plot_path}...")

    # Generate and save a plot that visualizes the clusters. This helps in understanding the
    # separation and characteristics of the different threat profiles.
    plot_clu_model(kmodes, plot='cluster', save=True)

    # Rename the generically saved cluster plot to a more descriptive file path.
    if os.path.exists('Cluster Plot.png'):
        os.rename('Cluster Plot.png', cluster_plot_path)

if __name__ == "__main__":
    train()