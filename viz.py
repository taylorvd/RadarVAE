#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from ray.tune.visual_utils import load_results_to_df, generate_plotly_dim_dict
plotly.offline.init_notebook_mode(connected=True)
plotly.io.renderers.default = "notebook" # This is required to render the plot in Sphinx

RESULTS_DIR = "/home/taylorlv/ray_results/lambda_2023-04-26_11-33-42"
df = load_results_to_df(RESULTS_DIR)