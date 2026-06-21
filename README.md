# Speech Separation with Dynamic Temporal Gate 🎧

This repository contains the final project for the Speech Separation / Denoising course. The project explores the improvement of a Time-Domain Convolutional Network (TCN) by upgrading static cyclic weighting to a **Dynamic Temporal Gate** mechanism (Direction B).

## 🚀 Project Overview

In traditional speech separation models, fixed weighting mechanisms struggle to adapt to time-varying background noises (e.g., sudden dog barks, passing buses). 
This project conducts an A/B test between two architectures:
1. **Baseline Model:** Uses static cyclic weighting (`wList`) that remains fixed after training.
2. **Direction B (Temporal Gate):** Introduces a $1 \times 1$ Convolutional layer followed by a Sigmoid activation function to dynamically calculate a temporal gate $g^{(j)}$. This acts as a real-time "volume knob," selectively enhancing speech and suppressing noise at each time step.

## 📊 Experimental Setup
- **Hardware:** NVIDIA A100 GPU
- **Dataset:** 1/2 subset (~10GB) with fixed `random.seed(42)` for fair A/B testing.
- **Hyperparameters:** `batch_size = 2`, `epochs = 10`, `num_workers = 4`.
- **Optimization:** `cudnn.benchmark = True`, `pin_memory = True` for I/O acceleration.

## 🏆 Final Results

By visualizing the SI-SNR Loss over 10 epochs, the Temporal Gate model successfully broke through the Baseline's performance ceiling:
- **Baseline Best SI-SNR Loss:** ~ -9.5 dB
- **Temporal Gate Best SI-SNR Loss:** ~ -10.5 dB
- **Improvement:** **~ 1.0 dB** significant enhancement without severe overfitting.

*(See the `/results` folder for detailed Loss vs Epochs zoomed plots).*

## ⚙️ How to Run the Code

### 1. Run Baseline Model
Navigate to the `baseline` directory and execute the shell script:
```bash
cd baseline
sh train_blue.sh