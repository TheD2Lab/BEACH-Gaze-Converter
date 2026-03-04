# BEACH-Gaze Converter
The Beach-Gaze Converter toolkit is desinged to convert raw gaze (x, y, and timestamps), such as those generated using [WebGazer](https://webgazer.cs.brown.edu), into a data format that is compatible with [Beach-Gaze](https://github.com/TheD2Lab/BEACH-Gaze).

## Setting up the environment
Depending on what ide you use, the environment setup may be different. We used many libraies to implement 
our scripts. In your environment please install:

1. pandas
2. numpy (should be included when installing pandas)
3. scikit-learn
4. kneed
5. natsort
6. matplotlib
7. st-dbscan

All of the above can be installed with pip install 'library name listed above'. If you are having trouble
installing the packages, please search up how to install it online.

<u>**ST-DBSCAN MANUAL INSTALLATION**</u>: If you encounter issues installing `st-dbscan` via pip (often due to Python version compatibility), you can install it manually by following these steps:
1. Go to the PyPI page for **st-dbscan**: [st-dbscan on PyPI](https://pypi.org/project/st-dbscan/#files)
2. Download version **`st_dbscan-0.2.3`**
3. Open the extracted zip folder and navigate to: *st_dbscan-0.2.3/src/*
4. Copy the **`st_dbscan`** folder into the root directory of your repository (same level as `stdbscan_tuning.py`)
5. Do **not** modify the following import statement in the improved_gaze_converter.py:
```python
from st_dbscan import ST_DBSCAN
```

<u>**SPECIAL CASE**</u>: After the installation, st-dbscan must be modified in order for the scripts to work. Currently,
the init.py has a library that will be deprecated soon. We can simply remove that portion so that the code will
work. That snippet only keeps the st-dbscan updated if any checks are called. Below is what you should remove from
the init.py that can be found in the .venv/lib/site-packages/st_dbscan/init.py (or st_dbscan/init.py if you installed the library manually).

```python
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

```

Once you've installed and modified the libraries, you are set to go.

Please use the files in this order to produce the right inputs for the gaze converter:
find_elbow > stdbscan_tuning > improved_gaze_converter. You can use the gaze converter
directly, but make sure a csv containing parameters for ST-DBSCAN is provided. Reference the
best_params folder for the csv format that is needed.

## improved_gaze_converter.py
A script that helps convert WebGazer data into a format acceptable for BeachGaze. The format is identical to the output
of GazePoint's data. Currently we use a clustering algorithm called ST-DBSCAN to help identify fixations. From there,
we can calculate any other field that is needed to match the format of GazePoint's data. The script:

- Inputs: csv file that follows WebGazer format (x,y,TIME,TIMETICK) and a csv file that contains the parameters used for ST-DBSCAN.
          Optional: To determine if gaze points are in a specific AOI, provide aoi_config.json.
- Outputs: csv file that follows GazePoint format inside created WG_all_gaze folder.

```python
# best parameters found at line 39 to chose the path for the parameter csv
# examples of the csv format found in the best_params folder
# Use best eps values found
PARAMS_CSV = "best_params/best_param_values_mod.csv"
_params_df = pd.read_csv(PARAMS_CSV, sep=None, engine="python")
PARAMS = _params_df.set_index("ID")[["Best_EPS_Spatial", "Best_EPS_Temporal", "Best_MIN_SAMPLES"]].to_dict("index")

# WebGazer data found at line 332 to chose the path 
# examples of the csv format found in webcam_data folder
raw_files = glob.glob("webcam_data/p*_webcam_gaze_data.csv")
```

## find_elbow.py
ST-DBSCAN requires parameters in order to output good clusters. These parameters are eps Spatial, eps Temporal, and min_samples.
The eps values will help determine the radius of the neighborhood. It helps decide which points are part of a cluster. We need an eps value for both space and time because the points do not appear all at one point in time. We must cluster based on how close a point is spatially and temporally.

The script searches for a good starting point for both eps values. The min_sample is kept as 6 (other combinations will be tested in the final script). The min_sample decides how many points are needed in a group to be considered a cluster.

- Script will prompt a menu and user must select an option (read 'things to keep in mind section for specifics')
- Inputs: 
    - Option 1: csv file that follows WebGazer format
    - Option 2: folder path with multiple csv files that follow WebGazer format
- Ouputs:
    - Option 1: results ouputted in the terminal and graphs that show the knee/elbow points
    - Option 2: txt and csv file that output details about the the input and the starting eps values for that data (examples can be found in the elbow_knee_values folder of this repo)

### Components and things to keep in mind when running this script
- At the top of the code, there is a config area. This place is made to edit your file or folder path. There is also
values to change for your specific screen size. You may also use a different min_sample starting point if you'd like.
- The files will always be ouputted with the same name, but if you'd like to change their name look at the run_all() 
function
- <u>**Please keep in mind**</u>: if you already have files of the same name as the outputted files, the files will be appended to or overwritten. **SO BE CAREFUL**.
- The script ouputs three options.
    1. looks at a single csv file and outputs in terminal with graphs
    2. looks at a folder with multiple files and outputs txt and csv file to show results
    3. exit

### Areas that need direct manipulation
This is at the beginning of the file:
```python
# --- Config ---
CSV_PATH = Path("./raw_WG_data/p30_webcam_gaze_data.csv") # change if needed
FOLDER_PATH = Path("./raw_WG_data") # change if needed

# many different ways to decide on min_sample
# 1) ln(x) where x is the size of the data set
# 2) 2 * the features in your dataset (which is being used here)
# 3) standard 4-5 found in papers
# in the end it depends on your dataset
MIN_SAMPLES = 6

SCREEN_W, SCREEN_H = 1920.0, 1080.0
```

## stdbscan_tuning.py
This script uses the starting points found in the previous script and performs a grid search to find the best parameters
to cluster your data. Examples of the outputs can be found in the best_params folder of this repo.

Scoring for the clusters is calculated as 

score = C - Penalty 

where:
- C = 1 / 1 + D
- Penalty = noise ratio + cluster count penalty
- D = avg spatial distance for a cluster + avg temporal distance for a cluster
- noise ratio = num_noise / num_total_points
- cluster count penalty = 1 / num_clusters + 1

We penalize for low cluster count and high noise ratio.

- Inputs: csv file with the starting points and folder path for WebGazer data
- Outputs: csv and txt file with the information of the best parameters, noise count, row count of input csv, and cluster count

### Components and things to keep in mind when running this script
- At the top of the code, there is a config area. This place is made to edit your folder and csv input paths. There is also
values to change for your specific screen size.
- <u>**Please keep in mind**</u>: if you already have files of the same name as the outputted files, the files will be appended to or overwritten. **SO BE CAREFUL**.

### Areas that need direct manipulation
This is at the beginning of the file:
```python
# --- Config ---
# input folder
FOLDER_PATH = Path("./raw_WG_data") # change if needed

#CSV path with all of the starting point eps values
EPS_CSV_PATH = Path("eps_values.csv") #change if needed

SCREEN_W, SCREEN_H = 1920.0, 1080.0
```
These can be found at lines 159 - 163 in the main function. These are the settings to change the ranges of the grid search. You can choose to go a certain percentage
plus or minus the intial eps parameters for each item. The min_sample_grid is an array that contains integer values of what min_samples to test out. Feel free to edit
the values.
```python
# create the grid space to get ready for the grid search
epsS_grid = np.linspace(1*initial_epsS, 3*initial_epsS, 10)
epsT_grid = np.linspace(0.01*initial_epsT, 1.5*initial_epsT, 10)
min_sample_grid = [3]
```

## CURRENT ISSUE 
The find_elbow.py does not give a very good starting point apparently. That script may be fixed or removed.
As for now the trend is that a higher spatial eps and lower temporal eps creates better clusters and has less problems when running
through the converter and BeachGaze. Further research is needed. We've edited the tuning code to search for 3 times higher than the starting
spatial point and as low of a temporal point you can go.
