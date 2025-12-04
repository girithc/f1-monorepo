# F1 Race Prediction Monorepo

This monorepo contains the complete source code for the F1 Race Prediction project, combining a Next.js frontend with a Python-based Machine Learning backend.

## 🏗 Architecture

The project is divided into two main components:

-   **`frontend/`**: A modern web application built with **Next.js 15**, **React 18**, and **Tailwind CSS**. It provides an interactive interface for users to view race predictions, compare driver strategies, and analyze "what-if" scenarios.
-   **`backend/`**: A Python environment responsible for data processing, feature engineering, and training Machine Learning models (**XGBoost**, **LSTM**). It serves predictions and strategy analysis to the frontend.

## 🧠 Machine Learning & Data Science

The core of this project is its predictive engine, which uses historical F1 data to forecast race outcomes and optimize pit strategies.

### Models

We employ a hybrid approach using both Gradient Boosting and Deep Learning:

1.  **XGBoost Regressor (`backend/server/model.py`)**:
    -   **Goal**: Predict final race position.
    -   **Type**: Gradient Boosted Decision Trees (GBDT).
    -   **Key Features**: Grid position, Car Performance Index, Pit Stop aggregates (count, duration), Circuit Overtake Difficulty.
    -   **Constraints**: Monotonic constraints are applied (e.g., better grid position -> better finish).
    -   **Loss Function**: Mean Absolute Error (MAE).

2.  **LSTM Regressor (`backend/server/lstm.py`)**:
    -   **Goal**: Predict finish position based on lap-by-lap evolution.
    -   **Type**: Long Short-Term Memory (LSTM) Neural Network.
    -   **Input**: Sequences of lap times, position changes, and pit flags.
    -   **Architecture**: 2-layer LSTM followed by a linear regression head.

3.  **Strategy LSTM (`backend/server/strategy_lstm.py`)**:
    -   **Goal**: Evaluate the impact of *future* pit strategies on race outcome.
    -   **Type**: Bidirectional LSTM with auxiliary static and plan inputs.
    -   **Architecture**:
        -   **Sequence Encoder**: Encodes historical lap data (pace, tire age).
        -   **Static Encoder**: Encodes race context (Grid, Total Laps, CPI).
        -   **Plan Encoder**: Encodes the proposed future strategy (remaining stops, pit laps, durations).
        -   **Fusion**: Concatenates all embeddings to predict final position.

### ⚙️ Feature Engineering

We derive several advanced metrics to improve model accuracy:

-   **Car Performance Index (CPI)**: A 0-1 score representing a team's raw speed relative to the field for a given season, derived from qualifying pace.
-   **Overtake Index**: A circuit-specific metric quantifying how easy it is to overtake, calculated from historical position changes.
-   **Tire Strategy Aggregates**:
    -   `tire_aggr_index`: Proxy for strategy aggressiveness (stints / total pit duration).
    -   `first_stop_delta`: Timing of the first stop relative to the average field strategy.
-   **Sanitized Pit Durations**: For strategy planning, we use "standardized" pit times (median per team/circuit) to remove the noise of mechanic errors and focus on strategic intent.

### 📊 Datasets

The project relies on the standard **Ergast F1 Dataset**, including:
-   `results.csv`: Race results and finishing positions.
-   `lap_times.csv`: Lap-by-lap timing data.
-   `pit_stops.csv`: Pit stop timing and lap info.
-   `qualifying.csv`: Qualifying session times (used for CPI).
-   `races.csv`, `circuits.csv`, `constructors.csv`, `drivers.csv`: Metadata.

*Note: Data is filtered to the Hybrid Era (2014+) to ensure relevance.*

## 🚀 Getting Started

### Prerequisites
-   Node.js 18+
-   Python 3.9+
-   Git

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd f1-monorepo
    ```

2.  **Frontend Setup**:
    ```bash
    cd frontend
    npm install
    # Run development server
    npm run dev
    ```

3.  **Backend Setup**:
    ```bash
    cd backend
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    pip install -r requirements.txt
    # Run training or server scripts
    python server/main.py
    ```

## 📂 Directory Structure

```
f1-monorepo/
├── frontend/                 # Next.js Web App
│   ├── src/                  # Source code
│   ├── public/               # Static assets
│   └── ...
├── backend/                  # Python ML Backend
│   ├── data/                 # CSV Datasets
│   ├── server/               # Model code & API
│   │   ├── model.py          # XGBoost implementation
│   │   ├── lstm.py           # Basic LSTM
│   │   └── strategy_lstm.py  # Strategy LSTM
│   ├── artifacts/            # Trained models (.pkl, .pt)
│   └── ...
└── README.md                 # This file
```
