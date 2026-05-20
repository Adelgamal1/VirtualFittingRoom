"""Create random sine/cosine sequences and predict a tested wave with LSTM."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def create_random_wave_sequences(
    number_of_sequences: int = 1000,
    sequence_length: int = 80,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, y, and full generated waves.

    X contains the first sequence_length - 1 points of each wave.
    y contains the final point, so the model learns to predict the next value.
    """
    rng = np.random.default_rng(random_seed)
    time_steps = np.linspace(0, 2 * np.pi, sequence_length)
    waves = []

    for _ in range(number_of_sequences):
        wave_type = rng.choice(["sine", "cosine"])
        amplitude = rng.uniform(0.5, 2.0)
        frequency = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.normal(0, 0.03, size=sequence_length)

        if wave_type == "sine":
            wave = amplitude * np.sin(frequency * time_steps + phase)
        else:
            wave = amplitude * np.cos(frequency * time_steps + phase)

        waves.append(wave + noise)

    waves_array = np.array(waves, dtype=np.float32)
    x_values = waves_array[:, :-1].reshape(number_of_sequences, sequence_length - 1, 1)
    y_values = waves_array[:, -1].reshape(number_of_sequences, 1)

    return x_values, y_values, waves_array


def build_lstm_model(input_steps: int) -> Sequential:
    """Build an LSTM model with 100 hidden units."""
    model = Sequential(
        [
            LSTM(100, input_shape=(input_steps, 1)),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM sine/cosine wave predictor")
    parser.add_argument("--sequences", type=int, default=1000)
    parser.add_argument("--length", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", default="assignments/sl_assignment_2026/wave_prediction.png")
    args = parser.parse_args()

    x_values, y_values, waves = create_random_wave_sequences(
        number_of_sequences=args.sequences,
        sequence_length=args.length,
    )

    train_size = int(0.8 * args.sequences)
    x_train, x_test = x_values[:train_size], x_values[train_size:]
    y_train, y_test = y_values[:train_size], y_values[train_size:]

    model = build_lstm_model(input_steps=args.length - 1)
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=min(args.epochs, 10),
        batch_size=32,
        verbose=1,
    )

    test_mse = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0)

    print(f"Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Actual tested wave next value: {y_test[0, 0]:.6f}")
    print(f"Predicted tested wave next value: {predictions[0, 0]:.6f}")

    tested_wave_index = train_size
    known_points = waves[tested_wave_index, :-1]
    actual_next = y_test[0, 0]
    predicted_next = predictions[0, 0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(known_points)), known_points, label="Known tested wave")
    plt.scatter(len(known_points), actual_next, label="Actual next value", color="green", s=80)
    plt.scatter(len(known_points), predicted_next, label="Predicted next value", color="red", s=80)
    plt.title("LSTM Prediction For Tested Sine/Cosine Wave")
    plt.xlabel("Time step")
    plt.ylabel("Wave value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)

    print(f"Saved prediction plot to: {output_path}")


if __name__ == "__main__":
    main()

