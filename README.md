# forskningsmetodik_2025
This repository contains scripts and notebooks for processing our dataset, preparing it for model training, inference, and analyzing model output. The workflow includes inspecting raw data, processing masks, and running the main analysis script.

# 📂 Repository Structure

```check.ipynb``` – Inspect raw data provided by Carolina and creates the necessary folder data_12_04.

```mask_processing.ipynb``` – Preprocess masks (data_12_04) for model training using the clean_mask(mask) function.

```forskningsmetodik.ipynb``` – Main analysis script. Requires processed data and masks generated from previous steps.

# ⚡ Requirements

*This code runs well on Google Colab using a T4 GPU.*

Required dependencies:
```
!pip install delta-microscopy[jax-gpu]
```

Python packages used in the notebooks:
```
import delta
import logging
from IPython.display import HTML
from base64 import b64encode
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from delta.config import Config
import os
from PIL import Image
import shutil
import re
import glob
import pandas as pd
import random
```

# 🚀 Usage

- Inspect and organize data

  *Run check.ipynb to inspect the raw dataset and create the data_12_04 folder. Adapt the data loading cell if your dataset is stored in a different location (e.g., Google Drive).*

- Process masks
  *Run mask_processing.ipynb to generate processed masks for model training using the clean_mask(mask) function.*

- Run main analysis
  *Finally, run forskningsmetodik.ipynb. Make sure the data_12_04 folder and processed masks are available before running.*

# Notes

Datasets are currently stored in Google Drive, but you can adapt paths as needed for local use. Please do try to run the notebooks first in a colab environment, ensuring python/library compatability.

This is still an undergoing project, running this might be a bit messy.

# References

Delta Microscopy
 – Python library used for data handling and analysis.

# Results (So far!

Below are plots for Pos102, showing the area- and length-based growth rates per frame. The plots include:
- Mean
- Standard deviation (std)
- Standard error of the mean (SEM)
- Rolling-window smoothed values for easier visualization of trends

The last plot shows the area per frame, also including mean, std, SEM, and rolling-window smoothing.
<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/74e115c3-ccba-497c-8f92-d20932a60a08" /> <img width="989" height="389" alt="image" src="https://github.com/user-attachments/assets/c9d7fd8a-40ae-4b1f-9478-e6e038fc5e65" /> <img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/81e5f2dc-2680-486f-9f2a-1a0c32aa6516" />

 
From our analysis so far, both area- and length-based growth rates indicate that the cells are indeed growing, which is consistent with the observed increase in cell area over time. Although the measurements are somewhat noisy, they align with growth visible by eye. To enable early detection, a small neural network (e.g, LSTM or 1D CNN) could be trained to classify the sample. Using short smoothing windows allows the model to leverage data from the earliest frames, and combining the raw area with the noisy growth-rate features should provide sufficient information for accurate classification even at these early time points.

Next steps:
- *Implement and train pipeline for early detection*
- *Evaluate over available data*

 
 
