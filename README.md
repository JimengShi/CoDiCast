# CoDiCast


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

### Environment
```
conda create -n `ENV_NAME` python=3.10
conda activate `ENV_NAME`
pip install tensorflow-gpu==2.15.0
pip install numpy==1.26.4
pip install pandas==1.5.3
pip install matplotlib==3.8.3
```

### Data
Variables are:
- geopotential at 500 hPa pressure level (Z500)
- atmospheric temperature at 850 hPa pressure level (T850)
- ground temperature (T2m)
- 10 meter U wind component (U10)
- 10 meter V wind component (V10)


### Run
- Run `ipynb` files in the `training` folder. Please run `pre-train` first for the encoder model, and train CoDiCast.
- Run `ipynb` files in the `evaluation` folder for quantitative and qualitative experimental results.
