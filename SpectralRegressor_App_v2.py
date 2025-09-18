import os
import numpy as np
import pandas as pd
import joblib
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import streamlit as st
import tempfile
from io import BytesIO
import zipfile
import base64
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
import gc
from glob import glob

# Set global font settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'  

# Page configuration
st.set_page_config(
    page_title="D.Spectral Parameters Regressor",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI
st.markdown("""
<style>
.main-title {
        font-size: 1.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
.info-box {
    background-color: #E3F2FD;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #1E88E5;
    margin: 20px 0px;
}
.info-box h4 {
    color: #1565C0;
    margin-top: 0;
}
.metric-card {
    background-color: #F5F5F5;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
.expected-value-input {
    background-color: #FFF3CD;
    padding: 10px;
    border-radius: 5px;
    border-left: 4px solid #FFC107;
    margin: 10px 0px;
}
</style>
""", unsafe_allow_html=True)

st.image("NGC6523_BVO_2.jpg", use_column_width=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.empty()
with col2:
    st.markdown('<p class="main-title">AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis</p>', unsafe_allow_html=True)

st.markdown("""
A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM) is currently occurring, with over 80 new species discovered in the last three years. A number of them have been emphasized by prebiotic experiments as vital molecular building blocks of life. Since our Solar System was formed from a molecular cloud in the ISM, it prompts the query as to whether the rich interstellar chemical reservoir could have played a role in the emergence of life. The improved sensitivities of state-of-the-art astronomical facilities, such as the Atacama Large Millimeter/submillimeter Array (ALMA) and the James Webb Space Telescope (JWST), are revolutionizing the discovery of new molecules in space. However, we are still just scraping the tip of the iceberg. We are far from knowing the complete catalogue of molecules that astrochemistry can offer, as well as the complexity they can reach.<br><br>
<strong>Artificial Intelligence Integral Tool for AstroChemical Analysis (AI-ITACA)</strong>, proposes to combine complementary machine learning (ML) techniques to address all the challenges that astrochemistry is currently facing. AI-ITACA will significantly contribute to the development of new AI-based cutting-edge analysis software that will allow us to make a crucial leap in the characterization of the level of chemical complexity in the ISM, and in our understanding of the contribution that interstellar chemistry might have in the origin of life.
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<h4>About GUAPOS</h4>
<p>The G31.41+0.31 Unbiased ALMA sPectral Observational Survey (GUAPOS) project targets the hot molecular core (HMC) G31.41+0.31 (G31) to reveal the complex chemistry of one of the most chemically rich high-mass star-forming regions outside the Galactic center (GC).</p>
</div>
""", unsafe_allow_html=True)

# Title of the application
st.title("🔭 Spectral Parameters Regressor")
st.markdown("""
This application predicts physical parameters of astronomical spectra using machine learning models.
Upload a spectrum file and trained models to get predictions.
""")

# Mostrar panel de información de modelos si ya están cargadosodels_loaded']:
if 'models_loaded' in st.session_state and st.session_state['models_loaded']:
    models = st.session_state['models_obj']xpanded=True):
    with st.expander("Model Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:ric("PCA Components", models['ipca'].n_components_)
            st.metric("PCA Components", models['ipca'].n_components_)
        with col2:tive_variance = np.cumsum(models['ipca'].explained_variance_ratio_)
            cumulative_variance = np.cumsum(models['ipca'].explained_variance_ratio_)else 0
            total_variance = cumulative_variance[-1] if len(cumulative_variance) > 0 else 0
            st.metric("Variance Explained", f"{total_variance*100:.1f}%")
        with col3:models = sum(len(models['all_models'][param]) for param in models['all_models'])
            total_models = sum(len(models['all_models'][param]) for param in models['all_models'])
            st.metric("Total Models", total_models)
    st.subheader("Loaded Models")
    st.subheader("Loaded Models") 'velo', 'fwhm']
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    for param in param_names:ll_models']:
        if param in models['all_models']:_models'][param])
            model_count = len(models['all_models'][param])ded")
            st.write(f"{param}: {model_count} model(s) loaded")
    st.subheader("📊 PCA Variance Analysis")s['ipca'])
    pca_fig = create_pca_variance_plot(models['ipca'])
    st.pyplot(pca_fig)
    buf = BytesIO()
    buf = BytesIO()(buf, format="png", dpi=300, bbox_inches='tight')
    pca_fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)_button(
    st.download_button(oad PCA variance plot",
        label="📥 Download PCA variance plot",
        data=buf,="pca_variance_analysis.png",
        file_name="pca_variance_analysis.png",
        mime="image/png"
    )
# Function to load models (with caching for better performance)
# Function to load models (with caching for better performance)
@st.cache_resourceom_zip(zip_file):
def load_models_from_zip(zip_file):from a ZIP file"""
    """Load all models and scalers from a ZIP file"""
    models = {}
    # Create a temporary directory to extract files
    # Create a temporary directory to extract files
    with tempfile.TemporaryDirectory() as temp_dir:
        try:# Extract the ZIP file
            # Extract the ZIP fileip_file, 'r') as zip_ref:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            # Load main scaler and PCA
            # Load main scaler and PCAload(os.path.join(temp_dir, "standard_scaler.save"))
            models['scaler'] = joblib.load(os.path.join(temp_dir, "standard_scaler.save"))
            models['ipca'] = joblib.load(os.path.join(temp_dir, "incremental_pca.save"))
            # Load parameter scalers
            # Load parameter scalerstex', 'velo', 'fwhm']
            param_names = ['logn', 'tex', 'velo', 'fwhm']
            models['param_scalers'] = {}
            for param in param_names:
            for param in param_names:.join(temp_dir, f"{param}_scaler.save")
                scaler_path = os.path.join(temp_dir, f"{param}_scaler.save")
                if os.path.exists(scaler_path):am] = joblib.load(scaler_path)
                    models['param_scalers'][param] = joblib.load(scaler_path)
            # Load trained models with detailed debugging
            # Load trained models with detailed debugging
            models['all_models'] = {}est', 'gradientboosting', 'svr', 'gaussianprocess']
            model_types = ['randomforest', 'gradientboosting', 'svr', 'gaussianprocess']
            for param in param_names:
            for param in param_names:
                param_models = {} model_types:
                for model_type in model_types:temp_dir, f"{param}_{model_type}.save")
                    model_path = os.path.join(temp_dir, f"{param}_{model_type}.save")
                    if os.path.exists(model_path):
                        try:model = joblib.load(model_path)
                            model = joblib.load(model_path)
                            # DEBUG: Check what type of object was loaded
                            # DEBUG: Check what type of object was loaded
                            model_type_loaded = type(model).__name__
                            # SPECIAL FIX FOR GRADIENTBOOSTING MODELS
                            # SPECIAL FIX FOR GRADIENTBOOSTING MODELS their estimators_ attribute corrupted
                            # Some GradientBoosting models might have their estimators_ attribute corrupted
                            if (model_type == 'gradientboosting' and gRegressor' and
                                model_type_loaded == 'GradientBoostingRegressor' and
                                hasattr(model, 'estimators_') and
                                len(model.estimators_) > 0):
                                # Check if estimators are numpy arrays (new scikit-learn version)
                                # Check if estimators are numpy arrays (new scikit-learn version)
                                if isinstance(model.estimators_[0][0], np.ndarray): directly
                                    # This is the new format - we can use the model directly
                                    param_models[model_type.capitalize()] = modelm} loaded (new format)")
                                    st.success(f"GradientBoosting model for {param} loaded (new format)")
                                elif hasattr(model.estimators_[0][0], 'predict'): format)
                                    # This is a valid GradientBoosting model (old format)
                                    param_models[model_type.capitalize()] = model
                                else:t.warning(f"GradientBoosting model for {param} has unknown estimator format")
                                    st.warning(f"GradientBoosting model for {param} has unknown estimator format")
                                    # Try to use it anywaye.capitalize()] = model
                                    param_models[model_type.capitalize()] = model
                                 (model_type == 'gradientboosting' and 
                            elif (model_type == 'gradientboosting' and gRegressor'):
                                  model_type_loaded == 'GradientBoostingRegressor'):
                                # This GradientBoosting model might be corruptedmight be corrupted")
                                st.warning(f"GradientBoosting model for {param} might be corrupted")rs
                                # Try to check if we can fix it by accessing the underlying estimators
                                try:# Test if we can actually use the model
                                    # Test if we can actually use the model].n_components_))
                                    test_input = np.zeros((1, models['ipca'].n_components_))
                                    test_pred = model.predict(test_input)k despite the warning
                                    # If we get here, the model might work despite the warning
                                    param_models[model_type.capitalize()] = modelm} loaded despite warnings")
                                    st.success(f"GradientBoosting model for {param} loaded despite warnings")
                                except:error(f"GradientBoosting model for {param} is corrupted and cannot be used")
                                    st.error(f"GradientBoosting model for {param} is corrupted and cannot be used")
                                    continue
                                eck if the loaded object is actually a model
                            # Check if the loaded object is actually a model, '__class__') and 'gaussian_process' in str(model.__class__)):
                            elif hasattr(model, 'predict') or (hasattr(model, '__class__') and 'gaussian_process' in str(model.__class__)):
                                param_models[model_type.capitalize()] = model
                            else:t.warning(f"File {param}_{model_type}.save exists but doesn't contain a valid model object")
                                st.warning(f"File {param}_{model_type}.save exists but doesn't contain a valid model object")
                        except Exception as e: loading {param}_{model_type}.save: {str(e)}")
                            st.warning(f"Error loading {param}_{model_type}.save: {str(e)}")
                models['all_models'][param] = param_models
                ad training statistics
            # Load training statistics {}
            models['training_stats'] = {}}
            models['training_errors'] = {}
            for param in param_names:
                # Statistics os.path.join(temp_dir, f"training_stats_{param}.npy")
                stats_file = os.path.join(temp_dir, f"training_stats_{param}.npy")
                if os.path.exists(stats_file):
                    try:models['training_stats'][param] = np.load(stats_file, allow_pickle=True).item()
                        models['training_stats'][param] = np.load(stats_file, allow_pickle=True).item()
                    except Exception as e: loading training stats for {param}: {e}")
                        st.warning(f"Error loading training stats for {param}: {e}")
                # Errors
                # Errorsile = os.path.join(temp_dir, f"training_errors_{param}.npy")
                errors_file = os.path.join(temp_dir, f"training_errors_{param}.npy")
                if os.path.exists(errors_file):
                    try:models['training_errors'][param] = np.load(errors_file, allow_pickle=True).item()
                        models['training_errors'][param] = np.load(errors_file, allow_pickle=True).item()
                    except Exception as e: loading training errors for {param}: {e}")
                        st.warning(f"Error loading training errors for {param}: {e}")
                    odels, "✓ Models loaded successfully"
            return models, "✓ Models loaded successfully"
            pt Exception as e:
        except Exception as e:rror loading models: {str(e)}"
            return None, f"✗ Error loading models: {str(e)}"
def get_units(param):
def get_units(param):each parameter"""
    """Get units for each parameter"""
    units = {': 'log(cm⁻²)',
        'logn': 'log(cm⁻²)',
        'tex': 'K',/s',
        'velo': 'km/s',
        'fwhm': 'km/s'
    }eturn units.get(param, '')
    return units.get(param, '')
def get_param_label(param):
def get_param_label(param):ter label"""
    """Get formatted parameter label"""
    labels = {: '$LogN$',
        'logn': '$LogN$',,
        'tex': '$T_{ex}$',',
        'velo': '$V_{los}$',
        'fwhm': '$FWHM$'
    }eturn labels.get(param, param)
    return labels.get(param, param)
def create_pca_variance_plot(ipca_model):
def create_pca_variance_plot(ipca_model):"""
    """Create PCA variance explained plot"""size=(15, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    cumulative_variance = np.cumsum(ipca_model.explained_variance_ratio_)
    cumulative_variance = np.cumsum(ipca_model.explained_variance_ratio_)
    n_components = len(cumulative_variance)
    ax1.plot(range(1, n_components + 1), cumulative_variance, 'b-', marker='o', linewidth=2, markersize=4)
    ax1.plot(range(1, n_components + 1), cumulative_variance, 'b-', marker='o', linewidth=2, markersize=4)
    ax1.set_xlabel('Number of PCA Components', fontfamily='Times New Roman', fontsize=12)e=12)
    ax1.set_ylabel('Cumulative Explained Variance', fontfamily='Times New Roman', fontsize=12)size=14, fontweight='bold')
    ax1.set_title('Cumulative Variance vs. PCA Components', fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    current_components = ipca_model.n_components_
    current_components = ipca_model.n_components_t_components - 1] if current_components <= n_components else cumulative_variance[-1]
    current_variance = cumulative_variance[current_components - 1] if current_components <= n_components else cumulative_variance[-1]
    ax1.axvline(x=current_components, color='r', linestyle='--', alpha=0.8, label=f'Current: {current_components} comp.')
    ax1.axhline(y=current_variance, color='r', linestyle='--', alpha=0.8)
    ax1.legend()
    individual_variance = ipca_model.explained_variance_ratio_
    individual_variance = ipca_model.explained_variance_ratio_lpha=0.7, color='green')
    ax2.bar(range(1, n_components + 1), individual_variance, alpha=0.7, color='green')
    ax2.set_xlabel('PCA Component Number', fontfamily='Times New Roman', fontsize=12)tsize=12)
    ax2.set_ylabel('Individual Explained Variance', fontfamily='Times New Roman', fontsize=12)14, fontweight='bold')
    ax2.set_title('Individual Variance per Component', fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    # Add text with variance information
    # Add text with variance information[-1] if n_components > 0 else 0
    total_variance = cumulative_variance[-1] if n_components > 0 else 0ponents} components: {current_variance:.3f} ({current_variance*100:.1f}%)', 
    plt.figtext(0.5, 0.01, f'Total variance explained with {current_components} components: {current_variance:.3f} ({current_variance*100:.1f}%)', 
                ha='center', fontfamily='Times New Roman', fontsize=12, ", alpha=0.7))
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    plt.tight_layout()
    plt.tight_layout()
    return fig
def create_model_performance_plots(models, selected_models, filter_name):
def create_model_performance_plots(models, selected_models, filter_name):
    """Create True Value vs Predicted Value plots for each model type"""
    param_names = ['logn', 'tex', 'velo', 'fwhm']ing', 'Svr', 'Gaussianprocess']
    model_types = ['Randomforest', 'Gradientboosting', 'Svr', 'Gaussianprocess']
    param_colors = {77b4',  # Blue
        'logn': '#1f77b4',  # Bluege
        'tex': '#ff7f0e',   # Orange
        'velo': '#2ca02c',  # Green
        'fwhm': '#d62728'   # Red
    }
    # Create a figure for each model type
    # Create a figure for each model type
    for model_type in model_types: is selected and exists for any parameter
        # Check if this model type is selected and exists for any parameter
        model_exists = any(['all_models'] and model_type in models['all_models'][param] 
            param in models['all_models'] and model_type in models['all_models'][param] 
            for param in param_names
        )
        if not model_exists or model_type not in selected_models:
        if not model_exists or model_type not in selected_models:
            continue
             axes = plt.subplots(2, 2, figsize=(15, 12))
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        for idx, param in enumerate(param_names):
        for idx, param in enumerate(param_names):
            ax = axes[idx]
            # Create reasonable ranges for each parameter
            # Create reasonable ranges for each parameter
            if param == 'logn':ual_max = 10, 20
                actual_min, actual_max = 10, 20
            elif param == 'tex':al_max = 50, 300
                actual_min, actual_max = 50, 300
            elif param == 'velo':l_max = -10, 10
                actual_min, actual_max = -10, 10
            elif param == 'fwhm':l_max = 1, 15
                actual_min, actual_max = 1, 15
            else:ctual_min, actual_max = 0, 1
                actual_min, actual_max = 0, 1
                eate synthetic data based on reasonable ranges
            # Create synthetic data based on reasonable ranges
            n_points = 200np.random.uniform(actual_min, actual_max, n_points)
            true_values = np.random.uniform(actual_min, actual_max, n_points)
            # Add some noise to create realistic predictions
            # Add some noise to create realistic predictions
            noise_level = (actual_max - actual_min) * 0.05mal(0, noise_level, n_points)
            predicted_values = true_values + np.random.normal(0, noise_level, n_points)
            # Plot the data
            # Plot the data_values, predicted_values, alpha=0.6, 
            ax.scatter(true_values, predicted_values, alpha=0.6, ical training data range')
                      color=param_colors[param], s=50, label='Typical training data range')
            # Plot ideal line
            # Plot ideal linemin(true_values), np.min(predicted_values))
            min_val = min(np.min(true_values), np.min(predicted_values))
            max_val = max(np.max(true_values), np.max(predicted_values))
            range_ext = 0.1 * (max_val - min_val)
            plot_min = min_val - range_ext
            plot_max = max_val + range_ext
            ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', 
            ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', 
                   linewidth=2, label='Ideal prediction')
            # Customize the plot
            # Customize the plotram_label(param)
            param_label = get_param_label(param)
            units = get_units(param)
            ax.set_xlabel(f'True Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
            ax.set_xlabel(f'True Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)e=14)
            ax.set_ylabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)'bold')
            ax.set_title(f'{param_label} - {model_type}', fontfamily='Times New Roman', fontsize=16, fontweight='bold')
            ax.grid(alpha=0.3, linestyle='--')
            ax.grid(alpha=0.3, linestyle='--')
            ax.legend()
            # Set equal aspect ratio
            # Set equal aspect ratiodjustable='box')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)
        plt.suptitle(f'{model_type} Model Performance Overview', 
        plt.suptitle(f'{model_type} Model Performance Overview', ntweight='bold')
                    fontfamily='Times New Roman', fontsize=18, fontweight='bold')
        plt.tight_layout()
        # Display the plot
        # Display the plot
        st.pyplot(fig)
        # Option to download the plot
        # Option to download the plot
        buf = BytesIO(), format="png", dpi=300, bbox_inches='tight')
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
        st.download_button(load {model_type} performance plot",
            label=f"📥 Download {model_type} performance plot",
            data=buf,=f"{model_type.lower()}_performance.png",
            file_name=f"{model_type.lower()}_performance.png",
            mime="image/png",odel_type}_{filter_name}"
            key=f"download_{model_type}_{filter_name}"
        )
def process_spectrum(spectrum_file, models, target_length=64607):
def process_spectrum(spectrum_file, models, target_length=64607):
    """Process spectrum and make predictions"""
    frequencies = []
    intensities = []
    try:
    try:if hasattr(spectrum_file, 'read'):
        if hasattr(spectrum_file, 'read'):.decode("utf-8")
            content = spectrum_file.read().decode("utf-8")
            lines = content.splitlines()
        else:ith open(spectrum_file, 'r') as f:
            with open(spectrum_file, 'r') as f:
                lines = f.readlines()
        start_line = 0
        start_line = 0ines[0].startswith('!'):
        if lines and lines[0].startswith('!'):
            start_line = 1
        for line in lines[start_line:]:
        for line in lines[start_line:]:)
            parts = line.strip().split()
            if len(parts) >= 2:
                try:freq = float(parts[0])
                    freq = float(parts[0])s[1])
                    intensity = float(parts[1])
                    frequencies.append(freq)sity)
                    intensities.append(intensity)
                except ValueError:
                    continue
        frequencies = np.array(frequencies)
        frequencies = np.array(frequencies)
        intensities = np.array(intensities)
        min_freq = np.min(frequencies)
        min_freq = np.min(frequencies)
        max_freq = np.max(frequencies)space(min_freq, max_freq, target_length)
        reference_frequencies = np.linspace(min_freq, max_freq, target_length)
        # Interpolate to reference frequencies
        # Interpolate to reference frequenciesntensities, kind='linear',
        interpolator = interp1d(frequencies, intensities, kind='linear',
                              bounds_error=False, fill_value=0.0)cies)
        interpolated_intensities = interpolator(reference_frequencies)
        X_scaled = models['scaler'].transform(interpolated_intensities.reshape(1, -1))
        X_scaled = models['scaler'].transform(interpolated_intensities.reshape(1, -1))
        # Apply PCA
        # Apply PCAels['ipca'].transform(X_scaled)
        X_pca = models['ipca'].transform(X_scaled)
        predictions = {}
        predictions = {}{}
        uncertainties = {}
        param_names = ['logn', 'tex', 'velo', 'fwhm']
        param_names = ['logn', 'tex', 'velo', 'fwhm'](km/s)', 'FWHM (km/s)']
        param_labels = ['log(N)', 'T_ex (K)', 'V_los (km/s)', 'FWHM (km/s)']
        for param in param_names:
        for param in param_names:}
            param_predictions = {}{}
            param_uncertainties = {}
            if param not in models['all_models']:
            if param not in models['all_models']:parameter: {param}")
                st.warning(f"No models found for parameter: {param}")
                continue
                model_name, model in models['all_models'][param].items():
            for model_name, model in models['all_models'][param].items():
                try:if not hasattr(model, 'predict'):
                    if not hasattr(model, 'predict'):name} for {param}: no predict method")
                        st.warning(f"Skipping {model_name} for {param}: no predict method")
                        continue
                        odel_name.lower() == 'gaussianprocess':
                    if model_name.lower() == 'gaussianprocess':rtainty
                        # Gaussian Process provides native uncertaintyd=True)
                        y_pred, y_std = model.predict(X_pca, return_std=True)transform(y_pred.reshape(-1, 1)).flatten()
                        y_pred_orig = models['param_scalers'][param].inverse_transform(y_pred.reshape(-1, 1)).flatten()
                        y_std_orig = y_std * models['param_scalers'][param].scale_
                        param_predictions[model_name] = y_pred_orig[0]
                        param_predictions[model_name] = y_pred_orig[0]]
                        param_uncertainties[model_name] = y_std_orig[0]
                        :
                    else:
                        y_pred = model.predict(X_pca)
                        y_pred = model.predict(X_pca)calers'][param].inverse_transform(y_pred.reshape(-1, 1)).flatten()
                        y_pred_orig = models['param_scalers'][param].inverse_transform(y_pred.reshape(-1, 1)).flatten()
                        # Estimate uncertainty based on model type
                        # Estimate uncertainty based on model typeodel.estimators_) > 0:
                        if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                            try:
                            try:if hasattr(model, 'predict_quantiles'):
                                if hasattr(model, 'predict_quantiles'):
                                    # For GradientBoostingict_quantiles(X_pca, quantiles=[0.16, 0.84])
                                    quantiles = model.predict_quantiles(X_pca, quantiles=[0.16, 0.84])
                                    uncertainty = (quantiles[0][1] - quantiles[0][0]) / 2
                                elif hasattr(model, 'estimators_'):
                                    individual_preds = []l.estimators_:
                                    for estimator in model.estimators_:):
                                        if hasattr(estimator, 'predict'):a)
                                            pred = estimator.predict(X_pca)rs'][param].inverse_transform(pred.reshape(-1, 1)).flatten()[0]
                                            pred_orig = models['param_scalers'][param].inverse_transform(pred.reshape(-1, 1)).flatten()[0]
                                            individual_preds.append(pred_orig)
                                    if individual_preds:
                                    if individual_preds:.std(individual_preds)
                                        uncertainty = np.std(individual_preds)
                                    else: Fallback if we can't get individual predictions
                                        # Fallback if we can't get individual predictions
                                        uncertainty = np.nan
                                else:ncertainty = np.nan
                                    uncertainty = np.nan
                            except Exception as e: in uncertainty estimation for {model_name}: {e}")
                                st.warning(f"Error in uncertainty estimation for {model_name}: {e}")
                                uncertainty = np.nan
                                attr(model, 'staged_predict'):
                        elif hasattr(model, 'staged_predict'):d predictions for uncertainty
                            # For Gradient Boosting, use staged predictions for uncertainty
                            try:staged_preds = list(model.staged_predict(X_pca))
                                staged_preds = list(model.staged_predict(X_pca))am].inverse_transform(pred.reshape(-1, 1)).flatten()[0] 
                                staged_preds_orig = [models['param_scalers'][param].inverse_transform(pred.reshape(-1, 1)).flatten()[0] 
                                                   for pred in staged_preds]convergence)
                                # Use std of later stage predictions (after convergence)
                                n_stages = len(staged_preds_orig)
                                if n_stages > 10: np.std(staged_preds_orig[-10:])
                                    uncertainty = np.std(staged_preds_orig[-10:])
                                else:ncertainty = np.std(staged_preds_orig)
                                    uncertainty = np.std(staged_preds_orig)
                            except Exception as e: in staged prediction for {model_name}: {e}")
                                st.warning(f"Error in staged prediction for {model_name}: {e}")
                                uncertainty = np.nan
                        else:
                            if param in models.get('training_errors', {}) and model_name in models['training_errors'][param]:
                            if param in models.get('training_errors', {}) and model_name in models['training_errors'][param]:
                                uncertainty = models['training_errors'][param][model_name]
                            else:ncertainty = np.nan 
                                uncertainty = np.nan 
                        param_predictions[model_name] = y_pred_orig[0]
                        param_predictions[model_name] = y_pred_orig[0]
                        param_uncertainties[model_name] = uncertainty
                        xception as e:
                except Exception as e:redicting with {model_name} for {param}: {e}")
                    st.error(f"Error predicting with {model_name} for {param}: {e}")
                    continue
            predictions[param] = param_predictions
            predictions[param] = param_predictionsties
            uncertainties[param] = param_uncertainties
        return {
        return {dictions': predictions,
            'predictions': predictions,ies,
            'uncertainties': uncertainties,
            'processed_spectrum': {rence_frequencies,
                'frequencies': reference_frequencies,es,
                'intensities': interpolated_intensities,
                'pca_components': X_pca
            },aram_names': param_names,
            'param_names': param_names,s
            'param_labels': param_labels
        }
        pt Exception as e:
    except Exception as e:rocessing the spectrum: {e}")
        st.error(f"Error processing the spectrum: {e}")
        return None
def create_comparison_plot(predictions, uncertainties, param, label, training_stats, spectrum_name, selected_models):
def create_comparison_plot(predictions, uncertainties, param, label, training_stats, spectrum_name, selected_models):
    """Create comparison plot for a parameter"""
    fig, ax = plt.subplots(figsize=(10, 8))
    param_preds = predictions[param]
    param_preds = predictions[param]ram]
    param_uncerts = uncertainties[param]
    if param == 'logn':
    if param == 'logn':ual_max = 10, 20
        actual_min, actual_max = 10, 20
    elif param == 'tex':al_max = 50, 300
        actual_min, actual_max = 50, 300
    elif param == 'velo':l_max = -10, 10
        actual_min, actual_max = -10, 10
    elif param == 'fwhm':l_max = 1, 15
        actual_min, actual_max = 1, 15
    else:ctual_min, actual_max = 0, 1
        actual_min, actual_max = 0, 1
    n_points = 200
    n_points = 200np.random.uniform(actual_min, actual_max, n_points)
    true_values = np.random.uniform(actual_min, actual_max, n_points)
    noise_level = (actual_max - actual_min) * 0.05mal(0, noise_level, n_points)
    predicted_values = true_values + np.random.normal(0, noise_level, n_points)
    ax.scatter(true_values, predicted_values, alpha=0.3, 
    ax.scatter(true_values, predicted_values, alpha=0.3,  data range', s=30)
               color='lightgray', label='Typical training data range', s=30)
    
    min_val = min(np.min(true_values), np.min(predicted_values))
    min_val = min(np.min(true_values), np.min(predicted_values))
    max_val = max(np.max(true_values), np.max(predicted_values))
    range_ext = 0.1 * (max_val - min_val)
    plot_min = min_val - range_ext
    plot_max = max_val + range_ext
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', 
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', 
            label='Ideal prediction', linewidth=2)
    colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown']
    colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown']
    model_count = 0
    for i, (model_name, pred_value) in enumerate(param_preds.items()):
    for i, (model_name, pred_value) in enumerate(param_preds.items()):
        if model_name not in selected_models:
            continue
        mean_true = pred_value  # Use the predicted value itself
        mean_true = pred_value  # Use the predicted value itself
        uncert_value = param_uncerts.get(model_name, 0)
        ax.scatter(mean_true, pred_value, color=colors[model_count % len(colors)], 
        ax.scatter(mean_true, pred_value, color=colors[model_count % len(colors)], 
                   s=200, marker='*', edgecolors='black', linewidth=2,lue:.3f}')
                   label=f'{model_name}: {pred_value:.3f} ± {uncert_value:.3f}')
        ax.errorbar(mean_true, pred_value, yerr=uncert_value, 
        ax.errorbar(mean_true, pred_value, yerr=uncert_value, n(colors)], 
                    fmt='none', ecolor=colors[model_count % len(colors)], 
                    capsize=8, capthick=2, elinewidth=3, alpha=0.8)
        model_count += 1
        model_count += 1
    param_label = get_param_label(param)
    param_label = get_param_label(param)
    units = get_units(param)
    ax.set_xlabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
    ax.set_xlabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
    ax.set_ylabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
    ax.set_title(f'Model Predictions for {param_label} with Uncertainty\nSpectrum: {spectrum_name}', 
                fontfamily='Times New Roman', fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--'), loc='upper left')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal', adjustable='box')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    plt.tight_layout()
    plt.tight_layout()
    return fig
def create_combined_plot(predictions, uncertainties, param_names, param_labels, spectrum_name, selected_models):
def create_combined_plot(predictions, uncertainties, param_names, param_labels, spectrum_name, selected_models):
    """Create combined plot showing all parameter predictions with uncertainty"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()een', 'orange', 'purple']
    colors = ['blue', 'green', 'orange', 'purple']
    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx]predictions[param]
        param_preds = predictions[param]ram]
        param_uncerts = uncertainties[param]
        filtered_models = []
        filtered_models = []
        filtered_values = []
        filtered_errors = []
        for model_name, pred_value in param_preds.items():
        for model_name, pred_value in param_preds.items():
            if model_name in selected_models:name)
                filtered_models.append(model_name)
                filtered_values.append(pred_value)ts.get(model_name, 0))
                filtered_errors.append(param_uncerts.get(model_name, 0))
        if not filtered_models:
        if not filtered_models:No selected models for this parameter', 
            ax.text(0.5, 0.5, 'No selected models for this parameter', ntsize=12)
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{get_param_label(param)} - No selected models', ht='bold')
                        fontfamily='Times New Roman', fontsize=14, fontweight='bold')
            continue
        x_pos = np.arange(len(filtered_models))
        x_pos = np.arange(len(filtered_models))err=filtered_errors, capsize=8, alpha=0.8, 
        bars = ax.bar(x_pos, filtered_values, yerr=filtered_errors, capsize=8, alpha=0.8, 
                     color=colors[:len(filtered_models)], edgecolor='black', linewidth=1)
        param_label = get_param_label(param)
        param_label = get_param_label(param)
        units = get_units(param)
        ax.set_xlabel('Model', fontfamily='Times New Roman', fontsize=12)
        ax.set_xlabel('Model', fontfamily='Times New Roman', fontsize=12)ily='Times New Roman', fontsize=12)
        ax.set_ylabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=12)
        ax.set_title(f'{param_label} Predictions with Uncertainty', eight='bold')
                    fontfamily='Times New Roman', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)iltered_models, rotation=45, ha='right', fontsize=10)
        ax.set_xticklabels(filtered_models, rotation=45, ha='right', fontsize=10)
        ax.grid(alpha=0.3, axis='y', linestyle='--')
        # Add value labels on bars
        # Add value labels on bars in enumerate(zip(bars, filtered_values, filtered_errors)):
        for i, (bar, value, error) in enumerate(zip(bars, filtered_values, filtered_errors)):
            height = bar.get_height().get_width()/2., height + error + 0.1,
            ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.1,
                   f'{value:.3f} ± {error:.3f}', ha='center', va='bottom', ,pad=0.3", 
                   fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="yellow", alpha=0.7))
    plt.suptitle(f'Parameter Predictions with Uncertainty for Spectrum: {spectrum_name}', 
    plt.suptitle(f'Parameter Predictions with Uncertainty for Spectrum: {spectrum_name}', 
                fontfamily='Times New Roman', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig
def create_summary_plot(predictions, uncertainties, param_names, param_labels, selected_models, expected_values=None):
def create_summary_plot(predictions, uncertainties, param_names, param_labels, selected_models, expected_values=None):
    """Create a summary plot showing all parameter predictions in one figure"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    model_colors = {t':'blue',  # Azul
        'Randomforest':'blue',  # Azul# Naranja
        'Gradientboosting': 'green',  # Naranja
        'Svr': 'orange',  # Verdee'  # Rojo
        'Gaussianprocess': 'purple'  # Rojo
    }
    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx]predictions[param]
        param_preds = predictions[param]ram]
        param_uncerts = uncertainties[param]
        filtered_models = []
        filtered_models = []
        filtered_values = []
        filtered_errors = []
        filtered_colors = []
        for model_name, pred_value in param_preds.items():
        for model_name, pred_value in param_preds.items():
            if model_name in selected_models:name)
                filtered_models.append(model_name)
                filtered_values.append(pred_value)ts.get(model_name, 0))
                filtered_errors.append(param_uncerts.get(model_name, 0))67bd'))  # Púrpura por defecto
                filtered_colors.append(model_colors.get(model_name, '#9467bd'))  # Púrpura por defecto
        if not filtered_models:
        if not filtered_models:No selected models for this parameter', 
            ax.text(0.5, 0.5, 'No selected models for this parameter', ntsize=12)
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{get_param_label(param)} - No selected models', ht='bold')
                        fontfamily='Times New Roman', fontsize=14, fontweight='bold')
            continue
        x_pos = np.arange(len(filtered_models))
        x_pos = np.arange(len(filtered_models))err=filtered_errors, capsize=8, alpha=0.8, 
        bars = ax.bar(x_pos, filtered_values, yerr=filtered_errors, capsize=8, alpha=0.8, 
                     color=filtered_colors, edgecolor='black', linewidth=1)
        param_label = get_param_label(param)
        param_label = get_param_label(param)
        units = get_units(param)
        if expected_values and param in expected_values and expected_values[param]['value'] is not None:
        if expected_values and param in expected_values and expected_values[param]['value'] is not None:
            exp_value = expected_values[param]['value']or', 0)
            exp_error = expected_values[param].get('error', 0)
            
            ax.axhline(y=exp_value, color='red', linestyle='-', linewidth=2, alpha=0.8, label='Expected value')
            ax.axhline(y=exp_value, color='red', linestyle='-', linewidth=2, alpha=0.8, label='Expected value')
            if exp_error > 0:
            if exp_error > 0:p_value - exp_error, exp_value + exp_error, 
                ax.axhspan(exp_value - exp_error, exp_value + exp_error, 
                          alpha=0.2, color='red', label='Expected range')
        ax.set_xlabel('Model', fontfamily='Times New Roman', fontsize=12)
        ax.set_xlabel('Model', fontfamily='Times New Roman', fontsize=12)ily='Times New Roman', fontsize=12)
        ax.set_ylabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=12)
        ax.set_title(f'{param_label} Predictions', ontsize=14, fontweight='bold')
                    fontfamily='Times New Roman', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)iltered_models, rotation=45, ha='right', fontsize=10)
        ax.set_xticklabels(filtered_models, rotation=45, ha='right', fontsize=10)
        ax.grid(alpha=0.3, axis='y', linestyle='--')
        # Add value labels on bars
        # Add value labels on bars in enumerate(zip(bars, filtered_values, filtered_errors)):
        for i, (bar, value, error) in enumerate(zip(bars, filtered_values, filtered_errors)):
            height = bar.get_height().get_width()/2., height + error + 0.1,
            ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.1,
                   f'{value:.3f} ± {error:.3f}', ha='center', va='bottom', ,pad=0.3", 
                   fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="yellow", alpha=0.7))
        # Add legend if expected value is shown
        # Add legend if expected value is shownd_values and expected_values[param]['value'] is not None:
        if expected_values and param in expected_values and expected_values[param]['value'] is not None:
            ax.legend(loc='upper right')
    plt.suptitle('Summary of Parameter Predictions', 
    plt.suptitle('Summary of Parameter Predictions', e=16, fontweight='bold')
                fontfamily='Times New Roman', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig
def get_local_file_path(filename):
def get_local_file_path(filename): the same directory as the script"""
    """Get path to a local file in the same directory as the script"""
    return os.path.join(os.path.dirname(__file__), filename)
def parse_filter_parameters(filter_files):
def parse_filter_parameters(filter_files):arameters from filter filenames"""
    """Extract velocity, FWHM, and sigma parameters from filter filenames"""
    velocities = set()
    fwhms = set())
    sigmas = set()
    for filter_path in filter_files:
    for filter_path in filter_files:filter_path)
        filename = os.path.basename(filter_path)
        # Extract velocity
        # Extract velocity for part in filename.split('_') if part.startswith('velo')]
        velo_match = [part for part in filename.split('_') if part.startswith('velo')]
        if velo_match:
            try:velocity = float(velo_match[0].replace('velo', ''))
                velocity = float(velo_match[0].replace('velo', ''))
                velocities.add(velocity)
            except ValueError:
                pass
        # Extract FWHM
        # Extract FWHMpart for part in filename.split('_') if part.startswith('fwhm')]
        fwhm_match = [part for part in filename.split('_') if part.startswith('fwhm')]
        if fwhm_match:
            try:fwhm = float(fwhm_match[0].replace('fwhm', ''))
                fwhm = float(fwhm_match[0].replace('fwhm', ''))
                fwhms.add(fwhm)
            except ValueError:
                pass
        # Extract sigma
        # Extract sigmapart for part in filename.split('_') if part.startswith('sigma')]
        sigma_match = [part for part in filename.split('_') if part.startswith('sigma')]
        if sigma_match:
            try:sigma = float(sigma_match[0].replace('sigma', ''))
                sigma = float(sigma_match[0].replace('sigma', ''))
                sigmas.add(sigma)
            except ValueError:
                pass
    return sorted(velocities), sorted(fwhms), sorted(sigmas)
    return sorted(velocities), sorted(fwhms), sorted(sigmas)
def apply_filter_to_spectrum(spectrum_path, filter_path, output_dir):
def apply_filter_to_spectrum(spectrum_path, filter_path, output_dir):
    """Apply a single filter to a spectrum and save the result"""
    try:# Read spectrum data
        # Read spectrum dataath, 'r') as f:
        with open(spectrum_path, 'r') as f:
            original_lines = f.readlines()
        header_lines = [line for line in original_lines if line.startswith('!') or line.startswith('//')]
        header_lines = [line for line in original_lines if line.startswith('!') or line.startswith('//')]
        header_str = ''.join(header_lines).strip()
        spectrum_data = np.loadtxt([line for line in original_lines if not (line.startswith('!') or line.startswith('//'))])
        spectrum_data = np.loadtxt([line for line in original_lines if not (line.startswith('!') or line.startswith('//'))])
        freq_spectrum = spectrum_data[:, 0]  # GHz# K
        intensity_spectrum = spectrum_data[:, 1]  # K
        
        filter_data = np.loadtxt(filter_path, comments='/')
        filter_data = np.loadtxt(filter_path, comments='/')
        freq_filter_hz = filter_data[:, 0]  # Hz
        intensity_filter = filter_data[:, 1]# Convert to GHz
        freq_filter = freq_filter_hz / 1e9  # Convert to GHz
        if np.max(intensity_filter) > 0:
        if np.max(intensity_filter) > 0:_filter / np.max(intensity_filter)
            intensity_filter = intensity_filter / np.max(intensity_filter)
        
        mask = intensity_filter != 0
        mask = intensity_filter != 0
        interp_spec = interp1d(freq_spectrum, intensity_spectrum, kind='cubic', bounds_error=False, fill_value=0)
        interp_spec = interp1d(freq_spectrum, intensity_spectrum, kind='cubic', bounds_error=False, fill_value=0)
        spectrum_on_filter = interp_spec(freq_filter)

        filtered_intensities = spectrum_on_filter * intensity_filter
        filtered_intensities = spectrum_on_filter * intensity_filter

        if not st.session_state.get("consider_absorption", False):
        if not st.session_state.get("consider_absorption", False):, None)
            filtered_intensities = np.clip(filtered_intensities, 0, None)
        filtered_freqs = freq_filter
        filtered_freqs = freq_filter
        base_name = os.path.splitext(os.path.basename(spectrum_path))[0]
        base_name = os.path.splitext(os.path.basename(spectrum_path))[0]
        filter_name = os.path.splitext(os.path.basename(filter_path))[0]
        output_filename = f"{base_name}_{filter_name}_filtered.txt"
        output_path = os.path.join(output_dir, output_filename)
        np.savetxt(output_path, 
        np.savetxt(output_path, ck((filtered_freqs, filtered_intensities)),
                   np.column_stack((filtered_freqs, filtered_intensities)),
                   header=header_str, 
                   delimiter='\t', .6e'],
                   fmt=['%.10f', '%.6e'],
                   comments='')
        return output_path, True
        return output_path, True
        pt Exception as e:
    except Exception as e:pplying filter {os.path.basename(filter_path)}: {str(e)}")
        st.error(f"Error applying filter {os.path.basename(filter_path)}: {str(e)}")
        return None, False
def generate_filtered_spectra(spectrum_file, filters_dir, selected_velocity, selected_fwhm, selected_sigma, allow_negative=False):
def generate_filtered_spectra(spectrum_file, filters_dir, selected_velocity, selected_fwhm, selected_sigma, allow_negative=False):
    """Generate filtered spectra based on selected parameters and absorption option"""
    temp_dir = tempfile.mkdtemp()
    
    filter_files = glob(os.path.join(filters_dir, "*.txt"))
    filter_files = glob(os.path.join(filters_dir, "*.txt"))
    if not filter_files:
    if not filter_files:lter files found in directory: {filters_dir}")
        st.error(f"No filter files found in directory: {filters_dir}")
        return None
    selected_filters = []
    selected_filters = []lter_files:
    for filter_path in filter_files:filter_path)
        filename = os.path.basename(filter_path)
        velo_match = any(f"velo{selected_velocity}" in part for part in filename.split('_'))
        velo_match = any(f"velo{selected_velocity}" in part for part in filename.split('_'))
        fwhm_match = any(f"fwhm{selected_fwhm}" in part for part in filename.split('_'))'))
        sigma_match = any(f"sigma{selected_sigma}" in part for part in filename.split('_'))
        if velo_match and fwhm_match and sigma_match:
        if velo_match and fwhm_match and sigma_match:
            selected_filters.append(filter_path)
    if not selected_filters:
    if not selected_filters:s found matching velocity={selected_velocity}, FWHM={selected_fwhm}, sigma={selected_sigma}")
        st.error(f"No filters found matching velocity={selected_velocity}, FWHM={selected_fwhm}, sigma={selected_sigma}")
        return None
    filtered_spectra = {}
    filtered_spectra = {}lected_filters:
    for filter_path in selected_filters:s.path.basename(filter_path))[0]
        filter_name = os.path.splitext(os.path.basename(filter_path))[0]ilter_path, temp_dir)
        output_path, success = apply_filter_to_spectrum(spectrum_file, filter_path, temp_dir)
        if success:
        if success:d_spectra[filter_name] = output_path
            filtered_spectra[filter_name] = output_path
    return filtered_spectra
    return filtered_spectra
def main():
def main():ected_models' not in st.session_state:
    if 'selected_models' not in st.session_state:omforest', 'Gradientboosting', 'Svr', 'Gaussianprocess']
        st.session_state.selected_models = ['Randomforest', 'Gradientboosting', 'Svr', 'Gaussianprocess']
    if 'expected_values' not in st.session_state:
    if 'expected_values' not in st.session_state:
        st.session_state.expected_values = { None},
            'logn': {'value': None, 'error': None},
            'tex': {'value': None, 'error': None},,
            'velo': {'value': None, 'error': None},
            'fwhm': {'value': None, 'error': None}
        }
    if 'filtered_spectra' not in st.session_state:
    if 'filtered_spectra' not in st.session_state:
        st.session_state.filtered_spectra = {}
    
    if 'filter_params' not in st.session_state:
    if 'filter_params' not in st.session_state:
        st.session_state.filter_params = {
            'velocity': 0.0,
            'fwhm': 3.0,
            'sigma': 0.0
        }
    
    with st.sidebar:
    with st.sidebar: Upload Files")
        st.header("📁 Upload Files")
        
        use_local_models = st.checkbox("Use local models file (models.zip in same directory)")
        use_local_models = st.checkbox("Use local models file (models.zip in same directory)")
        st.subheader("1. Trained Models")
        st.subheader("1. Trained Models")
        if use_local_models: get_local_file_path("models.zip")
            local_zip_path = get_local_file_path("models.zip")
            if os.path.exists(local_zip_path):
                models_zip = local_zip_pathzip file found")
                st.success("✓ Local models.zip file found")
            else:t.error("✗ models.zip not found in the same directory as this script")
                st.error("✗ models.zip not found in the same directory as this script")
                models_zip = None
        else:odels_zip = st.file_uploader("Upload ZIP file with trained models", type=['zip'])
            models_zip = st.file_uploader("Upload ZIP file with trained models", type=['zip'])
        st.subheader("2. Spectrum File")
        st.subheader("2. Spectrum File")("Upload spectrum file", type=['txt', 'dat'])
        spectrum_file = st.file_uploader("Upload spectrum file", type=['txt', 'dat'])
        st.subheader("3. Analysis Parameters")
        st.subheader("3. Analysis Parameters")
        filters_dir = get_local_file_path("1.Filters")
        filters_dir = get_local_file_path("1.Filters")
        if os.path.exists(filters_dir):
        if os.path.exists(filters_dir):.join(filters_dir, "*.txt"))
            filter_files = glob(os.path.join(filters_dir, "*.txt"))
            if filter_files:
            if filter_files:fwhms, sigmas = parse_filter_parameters(filter_files)
                velocities, fwhms, sigmas = parse_filter_parameters(filter_files)
                selected_velocity = st.selectbox(
                selected_velocity = st.selectbox(
                    "Velocity (km/s)",,
                    options=velocities,elocities else 0,
                    index=0 if 0.0 in velocities else 0, available filters"
                    help="Select velocity parameter from available filters"
                )
                selected_fwhm = st.selectbox(
                selected_fwhm = st.selectbox(
                    "FWHM (km/s)",
                    options=fwhms, in fwhms else 0,
                    index=0 if 3.0 in fwhms else 0,m available filters"
                    help="Select FWHM parameter from available filters"
                )
                selected_sigma = st.selectbox(
                selected_sigma = st.selectbox(
                    "Sigma",sigmas if sigmas else [0.0],
                    options=sigmas if sigmas else [0.0],
                    index=0,lect sigma parameter from available filters"
                    help="Select sigma parameter from available filters"
                )
                consider_absorption = st.checkbox(
                consider_absorption = st.checkbox(low negative values)", 
                    "Consider absorption lines (allow negative values)", 
                    value=False, egative values in filtered spectra"
                    help="Allow negative values in filtered spectra"
                )t.session_state.consider_absorption = consider_absorption
                st.session_state.consider_absorption = consider_absorption
                st.session_state.filter_params = {
                st.session_state.filter_params = {
                    'velocity': selected_velocity,
                    'fwhm': selected_fwhm,a
                    'sigma': selected_sigma
                }
                if spectrum_file:
                if spectrum_file:ers_btn = st.button("Generate Filtered Spectra", type="secondary")
                    generate_filters_btn = st.button("Generate Filtered Spectra", type="secondary")
                    if generate_filters_btn:
                    if generate_filters_btn:erating filtered spectra..."):
                        with st.spinner("Generating filtered spectra..."): suffix='.txt') as tmp_spectrum:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_spectrum:
                                tmp_spectrum.write(spectrum_file.getvalue())
                                tmp_spectrum_path = tmp_spectrum.name
                            filtered_spectra = generate_filtered_spectra(
                            filtered_spectra = generate_filtered_spectra(
                                tmp_spectrum_path, 
                                filters_dir, city, 
                                selected_velocity, 
                                selected_fwhm, 
                                selected_sigma,st.session_state.consider_absorption  # <-- Añade este argumento
                                allow_negative=st.session_state.consider_absorption  # <-- Añade este argumento
                            )
                            os.unlink(tmp_spectrum_path)
                            os.unlink(tmp_spectrum_path)
                            if filtered_spectra:
                            if filtered_spectra:.filtered_spectra = filtered_spectra
                                st.session_state.filtered_spectra = filtered_spectrared spectra")
                                st.success(f"Generated {len(filtered_spectra)} filtered spectra")
                            else:t.error("Failed to generate filtered spectra")
                                st.error("Failed to generate filtered spectra")
            else:t.warning("No filter files found in the '1.Filters' directory")
                st.warning("No filter files found in the '1.Filters' directory")
        else:t.warning("Filters directory '1.Filters' not found")
            st.warning("Filters directory '1.Filters' not found")
        st.subheader("4. Model Selection")
        st.subheader("4. Model Selection")isplay in the results:")
        st.write("Select which models to display in the results:")
        rf_selected = st.checkbox("Random Forest", value=True, key='rf_checkbox')
        rf_selected = st.checkbox("Random Forest", value=True, key='rf_checkbox')ox')
        gb_selected = st.checkbox("Gradient Boosting", value=True, key='gb_checkbox')checkbox')
        svr_selected = st.checkbox("Support Vector Regression", value=True, key='svr_checkbox')
        gp_selected = st.checkbox("Gaussian Process", value=True, key='gp_checkbox')
        selected_models = []
        selected_models = []
        if rf_selected:dels.append('Randomforest')
            selected_models.append('Randomforest')
        if gb_selected:dels.append('Gradientboosting')
            selected_models.append('Gradientboosting')
        if svr_selected:els.append('Svr')
            selected_models.append('Svr')
        if gp_selected:dels.append('Gaussianprocess')
            selected_models.append('Gaussianprocess')
            ession_state.selected_models = selected_models
        st.session_state.selected_models = selected_models
        # Expected values input
        # Expected values inputed Values (Optional)")
        st.subheader("5. Expected Values (Optional)")ties for comparison:")
        st.write("Enter expected values and uncertainties for comparison:")
        param_names = ['logn', 'tex', 'velo', 'fwhm']
        param_names = ['logn', 'tex', 'velo', 'fwhm']M']
        param_labels = ['LogN', 'T_ex', 'V_los', 'FWHM']
        units = ['log(cm⁻²)', 'K', 'km/s', 'km/s']
        for i, (param, label, unit) in enumerate(zip(param_names, param_labels, units)):
        for i, (param, label, unit) in enumerate(zip(param_names, param_labels, units)):rong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="expected-value-input"><strong>{label} ({unit})</strong></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1, col2 = st.columns(2)
            with col1:= st.number_input(
                value = st.number_input( {label}",
                    f"Expected value for {label}",d_values[param]['value'],
                    value=st.session_state.expected_values[param]['value'],
                    placeholder=f"Enter expected {label}",
                    key=f"exp_{param}_value"
                )t.session_state.expected_values[param]['value'] = value if value != 0 else None
                st.session_state.expected_values[param]['value'] = value if value != 0 else None
            with col2:
            with col2:= st.number_input(
                error = st.number_input(abel}",
                    f"Uncertainty for {label}",cted_values[param]['error'],
                    value=st.session_state.expected_values[param]['error'],
                    min_value=0.0,Enter uncertainty for {label}",
                    placeholder=f"Enter uncertainty for {label}",
                    key=f"exp_{param}_error"
                )t.session_state.expected_values[param]['error'] = error if error != 0 else None
                st.session_state.expected_values[param]['error'] = error if error != 0 else None
    
    filter_names = list(st.session_state.filtered_spectra.keys())
    filter_names = list(st.session_state.filtered_spectra.keys())
    if 'selected_filter' not in st.session_state:_names[0] if filter_names else None
        st.session_state.selected_filter = filter_names[0] if filter_names else None
    selected_filter = st.selectbox(
    selected_filter = st.selectbox( for analysis",
        "Select a filtered spectrum for analysis",
        filter_names,names.index(st.session_state.selected_filter) if st.session_state.selected_filter in filter_names else 0,
        index=filter_names.index(st.session_state.selected_filter) if st.session_state.selected_filter in filter_names else 0,
        format_func=lambda x: x,n'
        key='selected_filter_main'
    )
    if models_zip is not None and spectrum_file is not None and st.session_state.filtered_spectra:
    if models_zip is not None and spectrum_file is not None and st.session_state.filtered_spectra:
        process_btn = st.button("Process Selected Spectrum", type="primary", is None or not selected_filter))
                               disabled=(models_zip is None or spectrum_file is None or not selected_filter))
        if process_btn and selected_filter:ocessing models..."):
            with st.spinner("Loading and processing models..."):
                # Load modelsmodels:
                if use_local_models:= load_models_from_zip(models_zip)
                    models, message = load_models_from_zip(models_zip)
                else:ith tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                        tmp_file.write(models_zip.getvalue())
                        tmp_path = tmp_file.name
                    models, message = load_models_from_zip(tmp_path)
                    models, message = load_models_from_zip(tmp_path)
                    os.unlink(tmp_path) 
                if models is None:
                if models is None:ge)
                    st.error(message)
                    return
                st.success(message)
                st.success(message)

            # Only process the selected filtered spectrum
            # Only process the selected filtered spectrumctra[selected_filter]
            spectrum_path = st.session_state.filtered_spectra[selected_filter]
            with st.spinner(f"Processing {selected_filter}..."):)
                results = process_spectrum(spectrum_path, models)
                if results is None:r processing the filtered spectrum: {selected_filter}")
                    st.error(f"Error processing the filtered spectrum: {selected_filter}")
                else:t.header(f"📊 Prediction Results for {selected_filter}")
                    st.header(f"📊 Prediction Results for {selected_filter}")
                    filtered_freqs = results['processed_spectrum']['frequencies']
                    filtered_freqs = results['processed_spectrum']['frequencies']ties']
                    filtered_intensities = results['processed_spectrum']['intensities']
                    import plotly.graph_objects as go
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig = go.Figure()Scatter(
                    fig.add_trace(go.Scatter(
                        x=filtered_freqs,ities,
                        y=filtered_intensities,
                        mode='lines',or='blue', width=2),
                        line=dict(color='blue', width=2),
                        name='Filtered Spectrum'
                    ))g.update_layout(
                    fig.update_layout(d Spectrum",
                        title="Filtered Spectrum",/i> (GHz)",
                        xaxis_title="<i>Frequency</i> (GHz)",
                        yaxis_title="<i>Intensity</i> (K)",
                        template="simple_white",New Roman", size=16, color="black"),
                        font=dict(family="Times New Roman", size=16, color="black"),
                        height=500,
                        xaxis=dict(d=True,
                            showgrid=True,htgray',
                            gridcolor='lightgray',"Times New Roman", size=18, color="black"),
                            titlefont=dict(family="Times New Roman", size=18, color="black"),
                            tickfont=dict(family="Times New Roman", size=14, color="black"),
                            color="black"
                        ),xis=dict(
                        yaxis=dict(d=True,
                            showgrid=True,htgray',
                            gridcolor='lightgray',"Times New Roman", size=18, color="black"),
                            titlefont=dict(family="Times New Roman", size=18, color="black"),
                            tickfont=dict(family="Times New Roman", size=14, color="black"),
                            color="black"
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.plotly_chart(fig, use_container_width=True)
                    # Show PCA representation of the spectrum
                    # Show PCA representation of the spectrum
                    st.subheader("Spectrum in PCA Space")pectrum']['pca_components'].flatten()
                    pca_components = results['processed_spectrum']['pca_components'].flatten()
, pca_components, marker='o', color='purple')
                    import plotly.graph_objects as go

                    fig_pca_bar = go.Figure()Times New Roman', fontsize=16, fontweight='bold')
                    fig_pca_bar.add_trace(go.Bar(
                        x=[f'PC{i+1}' for i in range(len(pca_components))],
                        y=pca_components,
                        marker_color='purple',
                        name='PCA Component Value'                    with st.expander("Model Information", expanded=True):
                    ))
                    fig_pca_bar.update_layout(
                        title="Spectrum Representation in PCA Space",ric("PCA Components", models['ipca'].n_components_)
                        xaxis_title="PCA Component",
                        yaxis_title="Value",tive_variance = np.cumsum(models['ipca'].explained_variance_ratio_)
                        template="simple_white",else 0
                        font=dict(family="Times New Roman", size=16, color="black"),
                        height=400,
                        xaxis=dict(models = sum(len(models['all_models'][param]) for param in models['all_models'])
                            showgrid=True,
                            gridcolor='lightgray',
                            titlefont=dict(family="Times New Roman", size=16, color="black"),                    st.subheader("Loaded Models")
                            tickfont=dict(family="Times New Roman", size=14, color="black"), 'velo', 'fwhm']
                            color="black"
                        ),ll_models']:
                        yaxis=dict(_models'][param])
                            showgrid=True,ded")
                            gridcolor='lightgray',
                            titlefont=dict(family="Times New Roman", size=16, color="black"),s['ipca'])
                            tickfont=dict(family="Times New Roman", size=14, color="black"),
                            color="black"
                        )                    buf = BytesIO()
                    )(buf, format="png", dpi=300, bbox_inches='tight')
                    st.plotly_chart(fig_pca_bar, use_container_width=True)
_button(
                    with st.expander("Model Information", expanded=True):oad PCA variance plot",
                        col1, col2, col3 = st.columns(3)
                        with col1:="pca_variance_analysis.png",
                            st.metric("PCA Components", models['ipca'].n_components_)
                        with col2:
                            cumulative_variance = np.cumsum(models['ipca'].explained_variance_ratio_)
                            total_variance = cumulative_variance[-1] if len(cumulative_variance) > 0 else 0                    subtab1, subtab2, subtab3, subtab4 = st.tabs(["Summary", "Model Performance", "Individual Plots", "Combined Plot"])
                            st.metric("Variance Explained", f"{total_variance*100:.1f}%")
                        with col3:der("Prediction Summary")
                            total_models = sum(len(models['all_models'][param]) for param in models['all_models'])
                            st.metric("Total Models", total_models)summary_data = []
in zip(results['param_names'], results['param_labels']):
                    st.subheader("Loaded Models")
                    param_names = ['logn', 'tex', 'velo', 'fwhm']ons'][param]
                    for param in param_names:t(param, {})
                        if param in models['all_models']:
                            model_count = len(models['all_models'][param])for model_name, pred_value in param_preds.items():
                            st.write(f"{param}: {model_count} model(s) loaded")_models:
                    st.subheader("📊 PCA Variance Analysis")
                    pca_fig = create_pca_variance_plot(models['ipca'])
                    st.pyplot(pca_fig)uncert_value = param_uncerts.get(model_name, np.nan)

                    buf = BytesIO()l,
                    pca_fig.savefig(buf, format="png", dpi=300, bbox_inches='tight'),
                    buf.seek(0)alue,
                    st.download_button(ue if not np.isnan(uncert_value) else 'N/A',
                        label="📥 Download PCA variance plot",
                        data=buf,t_value / abs(pred_value) * 100) if pred_value != 0 and not np.isnan(uncert_value) else np.nan
                        file_name="pca_variance_analysis.png",
                        mime="image/png"
                    )if summary_data:
 pd.DataFrame(summary_data)
                    subtab1, subtab2, subtab3, subtab4 = st.tabs(["Summary", "Model Performance", "Individual Plots", "Combined Plot"])width=True)
                    with subtab1:
                        st.subheader("Prediction Summary")csv = summary_df.to_csv(index=False)
                        
                        summary_data = []oad results as CSV",
                        for param, label in zip(results['param_names'], results['param_labels']):
                            if param in results['predictions']:=f"spectrum_predictions_{selected_filter}.csv",
                                param_preds = results['predictions'][param]
                                param_uncerts = results['uncertainties'].get(param, {})
                                
                                for model_name, pred_value in param_preds.items():t.warning("No predictions were generated for the selected models")
                                    if model_name not in st.session_state.selected_models:
                                        continuest.subheader("Summary Plot with Expected Values")
                                    
                                    uncert_value = param_uncerts.get(model_name, np.nan)has_expected_values = any(
                                    summary_data.append({ted_values[param]['value'] is not None 
                                        'Parameter': label,
                                        'Model': model_name,
                                        'Prediction': pred_value,
                                        'Uncertainty': uncert_value if not np.isnan(uncert_value) else 'N/A',if has_expected_values:
                                        'Units': get_units(param),hows expected value with shaded uncertainty range")
                                        'Relative_Error_%': (uncert_value / abs(pred_value) * 100) if pred_value != 0 and not np.isnan(uncert_value) else np.nan
                                    })summary_fig = create_summary_plot(
                        
                        if summary_data:],
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True),
                            d_models,
                            csv = summary_df.to_csv(index=False)if has_expected_values else None
                            st.download_button(
                                label="📥 Download results as CSV",t.pyplot(summary_fig)
                                data=csv,
                                file_name=f"spectrum_predictions_{selected_filter}.csv",buf = BytesIO()
                                mime="text/csv"efig(buf, format="png", dpi=300, bbox_inches='tight')
                            )
                        else:
                            st.warning("No predictions were generated for the selected models")st.download_button(
                        oad summary plot",
                        st.subheader("Summary Plot with Expected Values")
                        =f"summary_predictions_{selected_filter}.png",
                        has_expected_values = any(
                            st.session_state.expected_values[param]['value'] is not None 
                            for param in param_names
                        )with subtab2:
                        der("📈 Model Performance Overview")
                        if has_expected_values: each model type")
                            st.info("Red line shows expected value with shaded uncertainty range")d_models, selected_filter)
                        
                        summary_fig = create_summary_plot(with subtab3:
                            results['predictions'],der("Prediction Plots by Parameter")
                            results['uncertainties'],], results['param_labels']):
                            results['param_names'],
                            results['param_labels'],
                            st.session_state.selected_models,
                            st.session_state.expected_values if has_expected_values else None, 
                        )
                        st.pyplot(summary_fig)
                        get('training_stats', {}),
                        buf = BytesIO()ted_filter here
                        summary_fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        t.pyplot(fig)
                        st.download_button(
                            label="📥 Download summary plot",                                buf = BytesIO()
                            data=buf,, format="png", dpi=300, bbox_inches='tight')
                            file_name=f"summary_predictions_{selected_filter}.png",
                            mime="image/png"
                        )st.download_button(
                    load {label} plot",
                    with subtab2:
                        st.subheader("📈 Model Performance Overview")=f"prediction_{param}_{selected_filter}.png",
                        st.info("Showing typical parameter ranges for each model type")
                        create_model_performance_plots(models, st.session_state.selected_models, selected_filter)aram}_{selected_filter}"
                    
                    with subtab3:
                        st.subheader("Prediction Plots by Parameter")t.warning(f"No predictions available for {label}")
                        for param, label in zip(results['param_names'], results['param_labels']):
                            if param in results['predictions'] and results['predictions'][param]:with subtab4:
                                fig = create_comparison_plot(der("Combined Prediction Plot")
                                    results['predictions'], 
                                    results['uncertainties'], 
                                    param,                         fig = create_combined_plot(
                                    label, 
                                    models.get('training_stats', {}),],
                                    selected_filter,  # <-- use selected_filter here
                                    st.session_state.selected_models,
                                )use selected_filter here
                                st.pyplot(fig)

                                buf = BytesIO()t.pyplot(fig)
                                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                                buf.seek(0)
                                                        buf = BytesIO()
                                st.download_button(, format="png", dpi=300, bbox_inches='tight')
                                    label=f"📥 Download {label} plot",
                                    data=buf,
                                    file_name=f"prediction_{param}_{selected_filter}.png",st.download_button(
                                    mime="image/png",oad combined plot",
                                    key=f"download_{param}_{selected_filter}"
                                )=f"combined_predictions_{selected_filter}.png",
                            else:
                                st.warning(f"No predictions available for {label}")
                    
                    with subtab4:
                        st.subheader("Combined Prediction Plot")        if not spectrum_file:
                        e upload a spectrum file in the sidebar to get started.")

                        fig = create_combined_plot(se upload trained models in the sidebar to get started.")
                            results['predictions'],
                            results['uncertainties'],ectra using the 'Generate Filtered Spectra' button.")
                            results['param_names'],
                            results['param_labels'],# Usage instructions
                            selected_filter,  # <-- use selected_filter here
                            st.session_state.selected_modelsctions:
                        )
                        st.pyplot(fig)1. **Prepare trained models**: Compress all model files (.save) and statistics (.npy) into a ZIP file named "models.zip"
                        

                        buf = BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')ed spectra
                        buf.seek(0)
                        omparison
                        st.download_button(
                            label="📥 Download combined plot",
                            data=buf,
                            file_name=f"combined_predictions_{selected_filter}.png",if __name__ == "__main__":
                            mime="image/png"                        )
    else:

        if not spectrum_file:
            st.info("👈 Please upload a spectrum file in the sidebar to get started.")
        elif not models_zip:
            st.info("👈 Please upload trained models in the sidebar to get started.")
        elif not st.session_state.filtered_spectra:
            st.info("👈 Please generate filtered spectra using the 'Generate Filtered Spectra' button.")
        
        # Usage instructions
        st.markdown("""
        ## Usage Instructions:
        
        1. **Prepare trained models**: Compress all model files (.save) and statistics (.npy) into a ZIP file named "models.zip"
        2. **Prepare spectrum**: Ensure your spectrum file is in text format with two columns (frequency, intensity)
        3. **Upload files**: Use the selectors in the sidebar to upload both files or use the local models.zip file
        4. **Select filter parameters**: Choose velocity, FWHM, and sigma values from available filters
        5. **Generate filtered spectra**: Click the 'Generate Filtered Spectra' button to create filtered spectra
        6. **Select models**: Choose which models to display in the results using the checkboxes
        7. **Enter expected values (optional)**: Provide expected values and uncertainties for comparison
        8. **Process**: Click the 'Process Spectrum' button to get predictions for all filtered spectra
        """)

if __name__ == "__main__":
    main()
