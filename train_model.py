# train_model.py
import pandas as pd
import numpy as np
# Import both classification and clustering modules distinctly
from pycaret.classification import setup as setup_clf, compare_models, finalize_model, save_model as save_clf_model, plot_model as plot_clf_model
from pycaret.clustering import setup as setup_clu, create_model as create_clu_model, save_model as save_clu_model, plot_model as plot_clu_model
import os
import matplotlib.pyplot as plt


def generate_synthetic_data(num_samples=500):
    """Generates a synthetic dataset of phishing and benign URL features."""
    print("Generating synthetic dataset...")

    features = [
        'having_IP_Address', 'URL_Length', 'Shortining_Service',
        'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
        'having_Sub_Domain', 'SSLfinal_State', 'URL_of_Anchor', 'Links_in_tags',
        'SFH', 'Abnormal_URL'
    ]

    num_per_profile = num_samples // 3
    num_benign = num_samples  # Create an equal number of benign samples

    # Profile 1: State-Sponsored (Subtle and Sophisticated)
    state_sponsored_data = {
        'having_IP_Address': np.random.choice([1, -1], num_per_profile, p=[0.05, 0.95]),  # Rarely use IPs
        'URL_Length': np.random.choice([1, 0, -1], num_per_profile, p=[0.2, 0.7, 0.1]),  # Normal length URLs
        'Shortining_Service': np.random.choice([1, -1], num_per_profile, p=[0.1, 0.9]),  # Rarely use shorteners
        'having_At_Symbol': np.random.choice([1, -1], num_per_profile, p=[0.1, 0.9]),
        'double_slash_redirecting': np.random.choice([1, -1], num_per_profile, p=[0.1, 0.9]),
        'Prefix_Suffix': np.random.choice([1, -1], num_per_profile, p=[0.9, 0.1]),  # High use of deceptive subdomains
        'having_Sub_Domain': np.random.choice([1, 0, -1], num_per_profile, p=[0.8, 0.1, 0.1]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], num_per_profile, p=[0.05, 0.05, 0.9]),
        # Almost always use valid SSL
        'URL_of_Anchor': np.random.choice([-1, 0, 1], num_per_profile, p=[0.6, 0.2, 0.2]),  # Deceptive anchors
        'Links_in_tags': np.random.choice([-1, 0, 1], num_per_profile, p=[0.5, 0.3, 0.2]),
        'SFH': np.random.choice([-1, 0, 1], num_per_profile, p=[0.1, 0.8, 0.1]),  # Often points to legitimate domains
        'Abnormal_URL': np.random.choice([1, -1], num_per_profile, p=[0.2, 0.8]),
        'has_political_keyword': np.random.choice([1, -1], num_per_profile, p=[0.1, 0.9])  # Rarely political
    }

    # Profile 2: Organized Cybercrime (High-Volume and Noisy)
    organized_crime_data = {
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
        'has_political_keyword': np.random.choice([1, -1], num_per_profile, p=[0.05, 0.95])
    }

    # Profile 3: Hacktivist (Opportunistic and Thematic)
    hacktivist_data = {
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
        'has_political_keyword': np.random.choice([1, -1], num_per_profile, p=[0.9, 0.1])
        # High chance of political keywords
    }

    # Benign Data (as a baseline)
    benign_data = {
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

    # Create DataFrames and assign labels
    df_state = pd.DataFrame(state_sponsored_data)
    df_state['threat_profile'] = 'State-Sponsored'

    df_crime = pd.DataFrame(organized_crime_data)
    df_crime['threat_profile'] = 'Organized Cybercrime'

    df_hacktivist = pd.DataFrame(hacktivist_data)
    df_hacktivist['threat_profile'] = 'Hacktivist'

    df_benign = pd.DataFrame(benign_data)
    df_benign['threat_profile'] = 'Benign'

    # Combine phishing profiles
    df_phishing = pd.concat([df_state, df_crime, df_hacktivist], ignore_index=True)
    df_phishing['label'] = 1

    df_benign['label'] = 0

    # Combine all data and shuffle
    final_df = pd.concat([df_phishing, df_benign], ignore_index=True)
    return final_df.sample(frac=1).reset_index(drop=True)


def train():
    classification_model_path = 'models/phishing_url_detector'
    clustering_model_path = 'models/threat_actor_profiler'
    plot_path = 'models/feature_importance.png'

    # Check if all artifacts already exist
    if os.path.exists(classification_model_path + '.pkl') and os.path.exists(clustering_model_path + '.pkl'):
        print("All models and plots already exist. Skipping training.")
        return

    data = generate_synthetic_data()
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/phishing_synthetic_profiles.csv', index=False)

    # --- 1. CLASSIFICATION WORKFLOW ---
    print("\n--- Starting Classification Workflow ---")
    print("Initializing PyCaret Classification Setup...")
    s_class = setup_clf(data, target='label', session_id=42, verbose=False, ignore_features=['threat_profile'])

    print("Comparing classification models...")
    best_model = compare_models(n_select=1, include=['rf', 'et', 'lightgbm'])

    print("Finalizing classification model...")
    final_model = finalize_model(best_model)

    print("Saving feature importance plot...")
    os.makedirs('models', exist_ok=True)
    plot_clf_model(final_model, plot='feature', save=True)
    if os.path.exists('Feature Importance.png'):
        os.rename('Feature Importance.png', plot_path)

    print("Saving classification model...")
    save_clf_model(final_model, classification_model_path)
    print("Classification model and plot saved successfully.")

    # --- 2. CLUSTERING WORKFLOW FOR ATTRIBUTION ---
    print("\n--- Starting Clustering Workflow for Threat Attribution ---")

    # Isolate only the phishing data for profiling
    phishing_data = data[data['label'] == 1].copy()

    # For clustering, we only need the features
    phishing_features = phishing_data.drop(['label', 'threat_profile'], axis=1)

    print("Initializing PyCaret Clustering Setup...")
    s_clust = setup_clu(phishing_features, session_id=42, verbose=False)

    print("Creating K-Modes clustering model...")
    kmodes = create_clu_model('kmodes', num_clusters=3)

    print("Saving clustering model...")
    save_clu_model(kmodes, clustering_model_path)
    print("Clustering model saved successfully.")

    cluster_plot_path = 'models/cluster_plot.png'
    print(f"Saving cluster plot to {cluster_plot_path}...")
    plot_clu_model(kmodes, plot='cluster', save=True)
    if os.path.exists('Cluster Plot.png'):
        os.rename('Cluster Plot.png', cluster_plot_path)

if __name__ == "__main__":
    train()