# CoDiCast

Code for the paper "CoDiCast: Conditional Diffusion Model for Weather Prediction with Uncertainty Quantification".


### Directory tree
```bash
├── checkpoint
│   └── model_weights.txt
├── data
│   └── data.txt
├── evaluation
│   ├── global_forecast_rmse_acc_56.ipynb
│   └── global_forecast_visual.ipynb
├── layers
│   ├── denoiser.py
│   └── diffusion.py
├── saved_models
│   └── encoder_cnn_56deg_5var.h5
├── training
│   ├── ddpm_weather_56c2_56_5var_best.ipynb
│   ├── ddpm_weather_56c2_56_5var_best_1500.ipynb
│   ├── ddpm_weather_56c2_56_5var_best_2000.ipynb
│   ├── ddpm_weather_56c2_56_5var_best_250.ipynb
│   ├── ddpm_weather_56c2_56_5var_best_500.ipynb
│   ├── ddpm_weather_56c2_56_5var_best_750.ipynb
│   ├── ddpm_weather_56c2_56_5var_best_no_attention.ipynb
│   ├── ddpm_weather_56c2_56_5var_best_no_attention_encoder.ipynb
│   ├── ddpm_weather_56c2_56_5var_best_no_encoder.ipynb
│   ├── ddpm_weather_56c2_56_5var_best_quadratic.ipynb
│   └── encoder_cnn_56deg_5var.ipynb
├── utils
│   ├── metrics.py
│   ├── normalization.py
│   ├── preprocess.py
│   └── visuals.py
```

### Run
`
df
`
