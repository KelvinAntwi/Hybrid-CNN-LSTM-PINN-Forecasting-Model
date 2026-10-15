# Physics-Informed CNN–LSTM for Solar PV Energy Forecasting

This project evaluates a conventional CNN–LSTM model and a physics-informed CNN–LSTM model for forecasting daily energy generation from land-based and floating solar photovoltaic (PV) systems. 

The physics-informed approach combines data-driven prediction with an approximate solar-energy estimate based on irradiation and temperature. The aim is to examine whether adding this physical constraint improves forecasting performance on a chronological test set.

## Project objectives

- Forecast daily energy generation for land-based and floating PV systems.
- Compare a conventional CNN–LSTM model with a physics-informed version (CNN–LSTM-PINN).
- Examine the effect of the physics-loss weight, λ, on validation performance.
- Evaluate both approaches using a chronological train/validation/test split.
- Visualize model predictions, training behaviour, and evaluation metrics.

## Data and input features

The notebook reads `merged_hybrid_generation_weather_filled.csv`. The dataset must be available in the notebook's working directory; it is not embedded in the notebook itself.

The workflow sorts observations by date, fills missing downtime hours with zero, creates cyclical month features, removes records with non-positive values for the selected energy target, and applies an interquartile-range (IQR) filter to the energy-to-irradiation ratio. It then creates lagged energy features and a trailing three-observation rolling mean.

Input features include:

- Global irradiation
- Sunshine duration
- Operating hours and downtime hours
- Month represented by sine and cosine features
- Energy lags 1, 2, and 3
- Trailing three-observation energy rolling mean
- Maximum air temperature
- Maximum wind speed
- Maximum relative humidity
- Daily shortwave-radiation sum

The target is selected separately for land-based generation (`LANDBASED_ENERGY_MWH`) and floating generation (`FLOATING_ENERGY_MWH`).

## Model architecture

The TensorFlow/Keras model contains:

1. A one-dimensional convolutional layer with 8 filters, kernel size 1, ReLU activation, and causal padding.
2. An LSTM layer with 16 units.
3. A dense hidden layer with 8 ReLU units.
4. A single linear output for the energy prediction.

Input features and target values are scaled using `MinMaxScaler`. The scalers are fitted on the training partition only and then applied to validation and test data.

**Implementation detail:** each row is reshaped into a sequence of length one. The model therefore does not receive a multi-timestep sequence directly; historical information is supplied mainly through the engineered lag and rolling features.

## Physics-informed learning

The physics-informed model adds a physics-based penalty to the usual prediction error.

**Total Loss = Data Loss + λ × Physics Loss**

The approximate daily energy estimate used by the notebook is:

**Eₚₕᵧₛᵢ꜀ₛ = max(0, η꜀ₐₗ × G × [1 + αₚ × (T − Tₛₜ꜀)])**

where:

- **η꜀ₐₗ** is a typical efficiency calibrated from the training partition.
- **G** is global irradiation.
- **T** is maximum air temperature.
- **Tₛₜ꜀** is the standard test-condition temperature of 25 °C.
- **αₚ** is the temperature coefficient set to −0.004.
- **λ** controls the contribution of the physics penalty to the total loss.

The physics estimate is an adapted approximation for daily energy, not a complete PV-system simulation. The baseline model uses λ = 0, while the main physics-informed comparison uses λ = 0.1. The notebook also evaluates λ = 0.2.

## Training and evaluation procedure

- Chronological split: 60% training, 20% validation, and 20% testing.
- Optimizer: Adam.
- Initial learning rate: 0.002.
- Maximum training duration: 80 epochs.
- Batch size: 16.
- Early stopping: validation loss, patience of 12 epochs, with best weights restored.
- Learning-rate reduction: factor of 0.5 after 5 epochs without validation-loss improvement; minimum learning rate of 0.00001.
- Random seeds: 1, 2, 3, 42, and 100 for the main comparison.
- Model selection: the run with the highest validation R² is selected; the test set is not used to choose the seed.

## Results recorded in the notebook

The notebook's printed final-comparison cell recorded the following test metrics. RMSE and MAE are calculated on the **MinMax-scaled target**, so these values are normalized-scale errors rather than errors in MWh.

| PV system | Model | Test R² | Test RMSE (scaled) | Test MAE (scaled) |
|---|---|---:|---:|---:|
| Land-based | CNN–LSTM baseline | 0.5335 | 0.1178 | 0.1045 |
| Land-based | CNN–LSTM-PINN, λ = 0.1 | 0.8866 | 0.0581 | 0.0474 |
| Floating | CNN–LSTM baseline | 0.9331 | 0.0626 | 0.0456 |
| Floating | CNN–LSTM-PINN, λ = 0.1 | 0.9344 | 0.0620 | 0.0434 |

In that recorded comparison, the physics-informed model improved land-based test R² from 0.5335 to 0.8866. For the floating system, the change was smaller, from 0.9331 to 0.9344. These figures describe the saved run, not a guarantee that every retraining will produce identical results. Neural-network initialization and the notebook's repeated training runs can lead to variation; rerun the full experiment before treating these values as a final reproducible benchmark.

## Figures and visual results

The `images/` directory contains figures exported from the notebook's plotting outputs. They provide visual checks of the loss-weight experiment, forecasts, residual agreement, and training behaviour.

### 1. Effect of physics-loss weight on validation R²

![Effect of physics-loss weight on validation R²]<img width="690" height="440" alt="lambda_grid_search" src="https://github.com/user-attachments/assets/72a0518f-bcb2-44ea-9aa6-4eeaf1b0480c" />


This figure compares validation R² for land-based and floating PV across λ values of 0.0, 0.1, and 0.2. It illustrates how the physics penalty can affect the two systems differently and why λ should be selected using validation data rather than assumed to improve performance at every setting.

### 2. Actual versus predicted daily energy

![Actual versus predicted daily energy]<img width="1089" height="790" alt="actual_vs_predicted" src="https://github.com/user-attachments/assets/443d1f2d-889b-4ee9-8e37-98af33c46fd3" />


The chronological test-set plot overlays measured energy with predictions from the baseline CNN–LSTM and the physics-informed model. It helps reveal how closely the forecasts follow observed changes and whether either model misses sharp changes or persistent trends.

### 3. Predicted-versus-actual scatter plots

![Predicted-versus-actual scatter plots]<img width="989" height="889" alt="scatter_pred_vs_actual" src="https://github.com/user-attachments/assets/b963ce9d-3559-4890-b584-ae7232fb5cc0" />


Each panel compares predicted energy with actual energy for one PV system and model. Points closer to the 1:1 reference line indicate closer agreement between predictions and observations. The displayed R² values belong to the plotting run that produced this figure.

### 4. Land-based model metrics

![Land-based model metrics]<img width="1390" height="396" alt="metrics_land_based" src="https://github.com/user-attachments/assets/1d161d2e-5de7-4c49-acb7-95c052d26a04" />


The grouped bars compare baseline and physics-informed performance across training, validation, and test partitions using R², RMSE, and MAE. RMSE and MAE in this plot use the notebook's normalized target scale.

### 5. Floating PV model metrics

![Floating PV model metrics]<img width="1390" height="396" alt="metrics_floating" src="https://github.com/user-attachments/assets/51672dd7-4600-4152-8227-94085352aafe" />


This figure provides the same metric comparison for the floating PV system. Comparing the train, validation, and test bars helps assess whether a model's performance remains consistent across data partitions.

### 6. Training and validation loss curves

![Training and validation loss curves]<img width="1089" height="790" alt="loss_curves" src="https://github.com/user-attachments/assets/e760639e-9d8c-475d-a87f-ce5b3175fc54" />


These curves show the training and validation losses over epochs for both models and both PV systems. They help assess convergence and identify visible divergence between training and validation behaviour.

**Important note on figures and metrics:** the notebook's plotting section retrains the models separately from the earlier printed final-comparison cell. Because of this, the figures may show different run-specific values from the results table above. For a publication-ready or final portfolio version, rerun the notebook from the beginning and export the table and all figures from the same experiment run.

## Repository contents

A practical repository layout is:

- `CNN_LSTM.ipynb` — notebook containing data preparation, model definition, training, evaluation, and plots.
- `README.md` — project documentation.
- `images/` — exported figures referenced in this README.
- `merged_hybrid_generation_weather_filled.csv` — required input data, only if you have permission to redistribute it.

The CSV is required to reproduce the experiment. If it cannot be shared publicly, explain how an authorized user can obtain it without uploading restricted or confidential data.

## Reproducing the experiment

1. Open the notebook in Jupyter or Google Colab.
2. Install the required packages if needed: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, and `matplotlib`.
3. Make `merged_hybrid_generation_weather_filled.csv` available in the runtime's working directory.
4. Run the notebook cells in order.
5. Review the validation-weight search, final model comparison, and generated plots.
6. If updating the reported results, export the metrics and figures from the same run so they remain consistent.

Exact values may vary between runs and software or hardware environments.

## Supervision and attribution

## Supervision

This project was conducted under the supervision of Mr. K.B. Owusu, focusing on the application of CNN–LSTM models and physics-informed learning to solar PV energy forecasting.

## Limitations
- Reproduction depends on access to the external CSV dataset.
- RMSE and MAE reported in the table are based on scaled targets(normalized) and are not in MWh.
  
  ## Citation

Antwi, N. K. (2026). *Physics-Informed CNN–LSTM for Solar PV Energy Forecasting*. GitHub repository.

This repository contains a supervised research implementation comparing conventional and physics-informed CNN–LSTM models for daily energy forecasting in land-based and floating solar PV systems.
