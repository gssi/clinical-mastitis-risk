# Useful ibraries 

import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path
from functools import reduce
import json
import multiprocessing
import os
import pickle
from joblib import Parallel, delayed
from scipy.stats import skew, kurtosis, wasserstein_distance
from sklearn.metrics import mean_absolute_error
from typing import List, Dict
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, average_precision_score, make_scorer
from scipy.stats import wasserstein_distance, ks_2samp, entropy, ttest_ind
from sklearn.feature_selection import RFE
import shap
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
                             classification_report, confusion_matrix, r2_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier
import lightgbm as lgb
import gc
from collections import Counter
from sklearn.utils import resample
import time
from sklearn.inspection import permutation_importance, partial_dependence
from scipy.stats import mannwhitneyu
from sklearn.linear_model import HuberRegressor
from sklearn.inspection import PartialDependenceDisplay
import math
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

# Logging 
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)