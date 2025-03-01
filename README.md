# DISS384
This repository contains the research resources used for my dissertation (DISS384) at FLAME University. My thesis focuses on precipitation nowcasting.

## How to Use

Run the `setup_conda_env.sh` script to set up your environment. This script ensures that most of the log messages by Tensorflow aren't printed out.

> If there is a better way to do this, please let me know.

## CLI Usage

You can use the CLI tool for caching data to speed up your workflow:

```bash
# To cache data for processing
python -m nowcast cache -v HEM -d 2022-05-01:2022-05-31

python -m nowcast cache -v OLR -d 2021-09-20_05:30:2022-09-30_17:29

python -m nowcast cache -v OLR HEM -d 2022-08-01:2022-08-31
```

## Data Source

The data being used in this project has been pulled from MOSDAC, more details will be covered in the dissertation draft that will be posted to this repository.
