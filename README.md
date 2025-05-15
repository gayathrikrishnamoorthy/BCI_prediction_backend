# EEG Left/Right Classification Backend

This project provides a Flask-based backend service for EEG-based left/right hand movement classification using a trained PyTorch model.

---

## Project Overview

The backend loads a pre-trained EEG classification model and exposes a REST API to predict left or right hand movement based on EEG input data. The model expects multi-channel EEG signals as input and returns the predicted class.

---

## Features

- PyTorch CNN model for EEG classification
- Flask REST API endpoint `/predict` accepting EEG data as JSON
- Real-time prediction support for incoming EEG signals
- Easy to extend for different EEG datasets or models

---

## Requirements

- Python 3.7+
- PyTorch
- Flask
- NumPy

Install dependencies with:

```bash
pip install torch flask numpy
