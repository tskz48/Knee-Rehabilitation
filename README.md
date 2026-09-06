# Deep Learning-Based Gait Trajectory Forecasting

## Project Description

This project investigates deep learning approaches for multi-step forecasting of lower-limb gait signals. Using six time-series features representing angular velocity (`X1`, `X2`, `X3`) and linear acceleration (`Y1`, `Y2`, `Y3`) from the foot, shank, and thigh, the models use the previous 25 timesteps to predict the next 5 timesteps for all six signals simultaneously.

Four neural-network architectures are implemented and compared:

- CNN-LSTM-Attention
- ConvLSTM1D
- GRU
- LSTM-Attention

The models are evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Normalized Root Mean Squared Error (NRMSE), and Correlation Coefficient (CC). The notebook also visualizes true and predicted trajectories and compares model performance across each gait variable.

## Dataset

The notebook expects a CSV file called `GaitData.csv` containing 6,000 rows and the following columns:

| Feature | Body Segment | Signal |
| --- | --- | --- |
| `X1` | Foot | Angular velocity |
| `Y1` | Foot | Linear acceleration |
| `X2` | Shank | Angular velocity |
| `Y2` | Shank | Linear acceleration |
| `X3` | Thigh | Angular velocity |
| `Y3` | Thigh | Linear acceleration |

The dataset is loaded from Google Drive in the current notebook:

```python
DATA_PATH = "/content/drive/MyDrive/GaitData.csv"
```

Update this path if your dataset is stored elsewhere.

## Forecasting Setup

The time series is converted into supervised learning samples using sliding windows.

- Input sequence length: **25 timesteps**
- Forecast horizon: **5 timesteps**
- Number of input features: **6**
- Window stride: **5 timesteps**
- Output size: **30 values** (`5 timesteps × 6 features`)
- Test segment: **last 75 timesteps**
- Validation split: **20% of the remaining training/validation data**

For the dataset used in the notebook, this produces:

| Split | Timesteps | Sequences |
| --- | ---: | ---: |
| Training | 4,740 | 943 |
| Validation | 1,185 | 232 |
| Test | 75 | 10 |

## Models

### 1. CNN-LSTM-Attention

The CNN-LSTM-Attention model combines local feature extraction, temporal modelling, and attention-based sequence aggregation.

Architecture:

- 3 × `Conv1D` layers with 64 filters and kernel size 3
- Dropout of 0.3 after each convolutional layer
- 5 × stacked LSTM layers with 256 units
- Dropout of 0.3 after each LSTM layer
- Custom temporal attention layer
- Dense multi-step output layer

### 2. ConvLSTM1D

The ConvLSTM model combines convolutional operations with recurrent temporal modelling.

Architecture:

- `ConvLSTM1D` with 32 filters and `return_sequences=True`
- Dropout of 0.3
- `ConvLSTM1D` with 32 filters and `return_sequences=False`
- Dropout of 0.3
- Flatten layer
- Dense multi-step output layer

### 3. GRU

The GRU model provides a more compact recurrent architecture.

Architecture:

- GRU with 128 units and `return_sequences=True`
- Dropout of 0.3
- GRU with 128 units
- Dropout of 0.3
- Dense multi-step output layer

### 4. LSTM-Attention

The LSTM-Attention model uses stacked LSTMs followed by a custom temporal attention mechanism.

Architecture:

- 5 × stacked LSTM layers with 256 units
- Dropout of 0.3 after each LSTM layer
- Custom temporal attention layer
- Dense multi-step output layer

## Attention Mechanism

The custom attention layer learns one attention score for each LSTM hidden state. The scores are normalized across the temporal dimension using softmax, and the final context vector is calculated as the weighted sum of the hidden states.

This allows the model to place different levels of importance on different timesteps in the input sequence before generating the forecast.

## Training Configuration

All four models use the same general training configuration to support comparison:

- Optimizer: **SGD**
- Learning rate: **0.07**
- Momentum: **0.9**
- Gradient clipping: **clip norm = 1.0**
- Loss function: **Mean Squared Error**
- Maximum epochs: **100**
- Batch size: **64**
- Dropout: **0.3**
- Random seed: **42**

Callbacks:

- `EarlyStopping`
  - patience = 10
  - restores the best validation weights
- `ReduceLROnPlateau`
  - factor = 0.2
  - patience = 5
  - minimum learning rate = `1e-5`

## Evaluation Metrics

Performance is calculated separately for each of the six gait variables.

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and true values.

### Root Mean Squared Error (RMSE)

Penalizes larger prediction errors more strongly than MAE.

### Normalized RMSE (NRMSE)

RMSE is normalized by the range of the true values:

```text
NRMSE (%) = RMSE / (max(true) - min(true)) × 100
```

### Correlation Coefficient (CC)

Measures the linear agreement between the predicted and true trajectories. Values closer to 1 indicate stronger agreement.

## Results

The saved notebook run produced the following average per-variable metrics:

| Model | Average MAE | Average RMSE | Average NRMSE (%) | Average CC |
| --- | ---: | ---: | ---: | ---: |
| CNN-LSTM-Attention | 0.172 | 0.224 | 4.502 | 0.971 |
| ConvLSTM1D | **0.147** | **0.187** | **3.758** | **0.980** |
| GRU | 0.156 | 0.193 | 3.914 | 0.979 |
| LSTM-Attention | 0.185 | 0.242 | 4.938 | 0.967 |

In this run, ConvLSTM1D achieved the best average performance across the four reported metrics, with GRU performing very closely behind it.

## Visualizations

The notebook generates:

- True vs. predicted gait trajectories for all six variables
- RMSE and CC comparisons across the four models
- MAE and NRMSE comparisons across the four models

## Requirements

The project is designed to run in Google Colab.

Main Python libraries:

```text
numpy
pandas
scikit-learn
tensorflow
matplotlib
```

To install them locally:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

## Running the Project

1. Open `GaitAnalysis.ipynb` in Google Colab.
2. Upload `GaitData.csv` to Google Drive.
3. Update `DATA_PATH` if necessary.
4. Run the notebook cells.
5. The notebook will:
   - load and split the gait dataset,
   - create time-series sequences,
   - train all four deep learning models,
   - generate multi-step predictions,
   - calculate the evaluation metrics,
   - and display model-comparison plots.

## Project Structure

```text
Gait-Trajectory-Forecasting/
├── GaitAnalysis.ipynb
├── GaitData.csv
└── README.md
```

If the dataset is not intended to be distributed publicly, omit `GaitData.csv` from the repository and provide instructions for obtaining it instead.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab

## Purpose

The project demonstrates how recurrent, convolutional-recurrent, and attention-based deep learning architectures can be applied to multi-step gait-signal forecasting and systematically compared using both prediction-error and trajectory-correlation metrics.
