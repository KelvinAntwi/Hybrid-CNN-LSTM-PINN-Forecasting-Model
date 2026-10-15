# Physics-Informed CNN–LSTM for Solar PV Energy Forecasting

This project evaluates a conventional CNN–LSTM model and a physics-informed CNN–LSTM model for forecasting daily energy generation from land-based and floating solar photovoltaic (PV) systems.

The physics-informed approach combines data-driven prediction with an approximate solar-energy estimate based on irradiation and temperature. The aim is to examine whether adding this physical constraint improves forecasting performance on a chronological test set.

## Project Objectives

- Forecast daily energy generation for land-based and floating PV systems.
- Compare a conventional CNN–LSTM model with a physics-informed version (CNN–LSTM-PINN).
- Examine the effect of the physics-loss weight, λ, on validation performance.
- Evaluate both approaches using a chronological training, validation, and testing split.
- Visualize model predictions, training behaviour, and evaluation metrics.

## Data and Input Features

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

## Model Architecture

The TensorFlow/Keras model contains:

1. A one-dimensional convolutional layer with 8 filters, kernel size 1, ReLU activation, and causal padding.
2. An LSTM layer with 16 units.
3. A dense hidden layer with 8 ReLU units.
4. A single linear output for the energy prediction.

Input features and target values are scaled using `MinMaxScaler`. The scalers are fitted on the training partition only and then applied to validation and test data.

**Implementation detail:** Each row is reshaped into a sequence of length one. The model therefore does not receive a multi-timestep sequence directly; historical information is supplied mainly through the engineered lag and rolling features.

**Physics-Informed Learning**

The physics-informed CNN–LSTM combines the usual prediction error with a penalty that encourages predictions to remain consistent with a simplified physics-based energy estimate.

Physics-Based Energy Estimate

The notebook calculates the approximate energy estimate as:

E_physics = max(0, η_cal × G × [1 + α_p × (T − T_STC)])

where:

E_physics is the approximate physics-based energy estimate.

η_cal is a calibration factor calculated from the median energy-to-irradiation ratio in the training partition.

G is global irradiation.

T is the daily maximum air temperature.

T_STC is the standard test-condition temperature, set to 25 °C.

α_p is the temperature coefficient, set to −0.004.

max(0, …) prevents the calculated estimate from becoming negative.

The calibration factor is derived from training data rather than assumed to be a fixed PV-module efficiency. This allows the estimate to reflect the scale of the dataset used in the experiment.

The temperature correction is an approximation because the notebook uses daily maximum air temperature as a proxy for module temperature. The estimate is therefore not a complete PV-system simulation and does not directly account for all factors affecting actual solar generation.

Combined Training Loss

The total training loss is:

L_total = L_data + λ × L_physics

The two components are:

L_data = MSE(y_true, y_pred) — measures the difference between observed energy and the model's prediction.

L_physics = MSE(E_physics, y_pred) — penalizes differences between the physics-based estimate and the model's prediction.

λ — controls the contribution of the physics penalty to the total loss.

The baseline model uses λ = 0.0, removing the physics penalty from the loss. The main physics-informed comparison uses λ = 0.1, and the notebook also evaluates λ = 0.2.

A larger λ gives the physics-consistency term more influence during training. However, a stronger physics penalty does not guarantee better forecasting performance, particularly because the energy estimate is an approximation of actual PV-system behaviour.

## Training and Evaluation Procedure

- Chronological split: 60% training, 20% validation, and 20% testing.
- Optimizer: Adam.
- Initial learning rate: 0.002.
- Maximum training duration: 80 epochs.
- Batch size: 16.
- Early stopping: validation loss, patience of 12 epochs, with best weights restored.
- Learning-rate reduction: factor of 0.5 after 5 epochs without validation-loss improvement; minimum learning rate of 0.00001.
- Random seeds: 1, 2, 3, 42, and 100 for the main comparison.
- Model selection: the run with the highest validation R² is selected; the test set is not used to choose the seed.

## Results and Performance Evaluation

The conventional CNN–LSTM (λ = 0.0) and physics-informed CNN–LSTM (λ = 0.1) were evaluated using the coefficient of determination (R²), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

**Interpretation:** Higher R² indicates a better fit to the observed values, while lower RMSE and MAE indicate smaller prediction errors. RMSE and MAE in the tables below are calculated on the MinMax-scaled target (0–1), not in MWh.

### 1. Physics-Loss Weight Sensitivity Analysis

The grid search investigated how different physics-loss weights affected validation R². A weight of λ = 0.0 represents the conventional data-driven baseline, while λ = 0.1 and λ = 0.2 introduce physics-based penalties of increasing strength.

**Table 1. Validation R² across different physics-loss weights**

| PV System / Experiment | λ = 0.0 | λ = 0.1 | λ = 0.2 |
|---|---:|---:|---:|
| Land-based — Initial Grid Search | **0.9397** | 0.8356 | 0.7616 |
| Floating — Initial Grid Search | **0.8935** | 0.8748 | 0.8726 |
| Land-based — Full Pipeline | 0.9109 | **0.9122** | 0.8442 |
| Floating — Full Pipeline | 0.8792 | 0.8827 | **0.8897** |

The initial grid search favoured the baseline model for both PV systems. In the full-pipeline experiment, λ = 0.1 produced the highest validation R² for land-based PV, while λ = 0.2 produced the highest validation R² for floating PV.

These results show that increasing the physics-loss weight does not necessarily improve forecasting performance. A stronger constraint can place greater emphasis on an approximate physical relationship that may not fully represent the operating conditions of the PV system.

The main comparison below uses λ = 0.1 as a common physics-informed configuration. It should not be interpreted as the universally optimal weight for both systems, since the grid-search results varied between experiments.

### 2. Final Model Comparison

The notebook's final-comparison experiment evaluated five random seeds: 1, 2, 3, 42, and 100. Model selection was based on validation R² rather than test performance.

#### Training Performance

**Table 2. Training-set metrics**

| Model Configuration | R² | RMSE (Scaled) | MAE (Scaled) |
|---|---:|---:|---:|
| Land-based — Baseline (λ = 0.0) | **0.9612** | **0.0406** | **0.0305** |
| Land-based — Physics-Informed (λ = 0.1) | 0.9458 | 0.0480 | 0.0387 |
| Floating — Baseline (λ = 0.0) | 0.9471 | 0.0467 | 0.0357 |
| Floating — Physics-Informed (λ = 0.1) | **0.9488** | **0.0460** | **0.0341** |

The land-based baseline achieved better training metrics than its physics-informed counterpart. For floating PV, the physics-informed model achieved slightly better training metrics.

#### Validation Performance

**Table 3. Validation-set metrics**

| Model Configuration | R² | RMSE (Scaled) | MAE (Scaled) |
|---|---:|---:|---:|
| Land-based — Baseline (λ = 0.0) | 0.8472 | 0.0967 | 0.0768 |
| Land-based — Physics-Informed (λ = 0.1) | **0.8988** | **0.0787** | **0.0567** |
| Floating — Baseline (λ = 0.0) | 0.8720 | 0.0469 | 0.0350 |
| Floating — Physics-Informed (λ = 0.1) | **0.8827** | **0.0449** | **0.0341** |

The physics-informed model outperformed the baseline on the validation set for both PV systems, achieving higher R² and lower RMSE and MAE.

#### Testing Performance

**Table 4. Held-out test-set metrics**

| Model Configuration | R² | RMSE (Scaled) | MAE (Scaled) |
|---|---:|---:|---:|
| Land-based — Baseline (λ = 0.0) | 0.5335 | 0.1178 | 0.1045 |
| Land-based — Physics-Informed (λ = 0.1) | **0.8866** | **0.0581** | **0.0474** |
| Floating — Baseline (λ = 0.0) | 0.9331 | 0.0626 | 0.0456 |
| Floating — Physics-Informed (λ = 0.1) | **0.9344** | **0.0620** | **0.0434** |

The test set provides the final evaluation of how the models performed on observations withheld from training and validation.

### 3. Discussion of Results

For land-based PV, the physics-informed model increased test R² from 0.5335 to 0.8866. Its scaled RMSE decreased by approximately 50.7%, while its scaled MAE decreased by approximately 54.6%.

The land-based baseline performed better during training but substantially worse on the test set in this recorded experiment. The physics-informed model's stronger test performance suggests improved generalization in this particular run.

For floating PV, the baseline already achieved strong test performance. The physics-informed model increased R² from 0.9331 to 0.9344, while scaled RMSE decreased by approximately 1.0% and scaled MAE decreased by approximately 4.8%.

Overall, the recorded results suggest that adding an approximate physics constraint can improve forecasting performance, particularly for the land-based system. However, the magnitude of improvement differs between the two systems, and the grid-search results show that the effect depends on the selected physics-loss weight and experimental run.

**Reproducibility note:** These metrics describe the notebook's recorded final-comparison run. The plotting section retrains the models separately, so figures may display different run-specific values from Tables 2–4. For a reproducible benchmark, rerun the complete experiment and export all tables and figures from the same run.

## Figures and Visual Results

The following figures illustrate the physics-loss weight experiment, model predictions, agreement between predicted and observed values, evaluation metrics, and training behaviour.

### 1. Effect of Physics-Loss Weight on Validation R²

![Effect of physics-loss weight on validation R²](https://github.com/user-attachments/assets/72a0518f-bcb2-44ea-9aa6-4eeaf1b0480c)

This figure compares validation R² for land-based and floating PV across λ values of 0.0, 0.1, and 0.2. It illustrates how the physics penalty can affect the two systems differently and why λ should be selected using validation data rather than assumed to improve performance at every setting.

### 2. Actual Versus Predicted Daily Energy

![Actual versus predicted daily energy](https://github.com/user-attachments/assets/443d1f2d-889b-4ee9-8e37-98af33c46fd3)

The chronological test-set plot overlays measured energy with predictions from the baseline CNN–LSTM and the physics-informed model. It helps reveal how closely the forecasts follow observed changes and whether either model misses sharp changes or persistent trends.

### 3. Predicted-Versus-Actual Scatter Plots

![Predicted-versus-actual scatter plots](https://github.com/user-attachments/assets/b963ce9d-3559-4890-b584-ae7232fb5cc0)

Each panel compares predicted energy with actual energy for one PV system and model. Points closer to the 1:1 reference line indicate closer agreement between predictions and observations. The displayed R² values belong to the plotting run that produced this figure.

### 4. Land-Based Model Metrics

![Land-based model metrics](https://github.com/user-attachments/assets/1d161d2e-5de7-4c49-acb7-95c052d26a04)

The grouped bars compare baseline and physics-informed performance across training, validation, and test partitions using R², RMSE, and MAE. RMSE and MAE in this plot use the notebook's normalized target scale.

### 5. Floating PV Model Metrics

![Floating PV model metrics](https://github.com/user-attachments/assets/51672dd7-4600-4152-8227-94085352aafe)

This figure provides the same metric comparison for the floating PV system. Comparing the training, validation, and test bars helps assess whether a model's performance remains consistent across data partitions.

### 6. Training and Validation Loss Curves

![Training and validation loss curves](https://github.com/user-attachments/assets/e760639e-9d8c-475d-a87f-ce5b3175fc54)

These curves show the training and validation losses over epochs for both models and both PV systems. They help assess convergence and identify visible divergence between training and validation behaviour.

**Important note on figures and metrics:** The notebook's plotting section retrains the models separately from the earlier printed final-comparison cell. Consequently, the figures may show different run-specific values from the numerical results above.

## Repository Contents

A practical repository layout is:

- `CNN_LSTM.ipynb` — notebook containing data preparation, model definition, training, evaluation, and plots.
- `README.md` — project documentation.
- `images/` — optional directory for locally stored versions of the figures.
- `merged_hybrid_generation_weather_filled.csv` — required input data, only if you have permission to redistribute it.

The CSV is required to reproduce the experiment. If it cannot be shared publicly, explain how an authorized user can obtain it without uploading restricted or confidential data.

## Reproducing the Experiment

1. Open the notebook in Jupyter or Google Colab.
2. Install the required packages if needed: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, and `matplotlib`.
3. Make `merged_hybrid_generation_weather_filled.csv` available in the runtime's working directory.
4. Run the notebook cells in order.
5. Review the validation-weight search, final model comparison, and generated plots.
6. If updating the reported results, export the metrics and figures from the same run so they remain consistent.

Exact values may vary between runs and software or hardware environments.

## Supervision and Attribution

This project was conducted under the supervision of **Mr. K.B. Owusu**, focusing on the application of CNN–LSTM models and physics-informed learning to solar PV energy forecasting.

## Limitations

- Reproduction depends on access to the external CSV dataset.
- RMSE and MAE are reported on normalized targets and are not in MWh.

## Citation

Antwi, N. K. (2026). *Physics-Informed CNN–LSTM for Solar PV Energy Forecasting*. GitHub repository.

https://github.com/KelvinAntwi/Hybrid-CNN-LSTM-PINN-Forecasting-Model/tree/main

This repository contains a supervised research implementation comparing conventional and physics-informed CNN–LSTM models for daily energy forecasting in land-based and floating solar PV systems.
