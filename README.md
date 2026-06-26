# Speech Separation with Dynamic Temporal Gate 🎧

This repository contains the final project for the Speech Separation / Denoising course. The project explores the improvement of a Time-Domain Convolutional Network (TCN) by upgrading static cyclic weighting to a **Dynamic Temporal Gate** mechanism (Direction B).

## 🚀 Project Overview

In traditional speech separation models, fixed weighting mechanisms struggle to adapt to time-varying background noises (e.g., sudden dog barks, passing buses). 
This project conducts an A/B test between two architectures:
1. **Baseline Model:** Uses static cyclic weighting (`wList`) that remains fixed after training.
2. **Direction B (Temporal Gate):** Introduces a $1 \times 1$ Convolutional layer followed by a Sigmoid activation function to dynamically calculate a temporal gate $g^{(j)}$. This acts as a real-time "volume knob," selectively enhancing speech and suppressing noise at each time step.

## 📊 Experimental Setup

- **Hardware:** NVIDIA A100 GPU (Google Colab)
- **Dataset:** 1/2 subset of the original dataset (~10GB). Fixed `random.seed(42)` ensures identical subsets for fair A/B testing.
  - **Training Data (Train):** 8,640 utterances (approx. 9.6 hours of audio).
  - **Validation Data (Dev):** ~1,440 utterances.
- **Training Time:** ~23.36 minutes per epoch (Total ~3.9 hours for 10 epochs).
- **Hyperparameters:** `batch_size = 2`, `epochs = 10`.
- **Optimization:** `cudnn.benchmark = True`, `num_workers = 4`, and `pin_memory = True` for I/O acceleration.

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
