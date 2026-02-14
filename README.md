# aind-analysis-arch-result-access

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-98.5%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.9-blue?logo=python)


APIs to access analysis results in the AIND behavior pipeline.

## Installation

```bash
pip install aind-analysis-arch-result-access
```

## Usage

Try the demo: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14Hph9QuySbgSQBKl8PGi_nCQfoLcLUI-?usp=sharing)

See the [demo notebook](notebook/demo_result_access_API.ipynb) for comprehensive examples.

### Fetch dynamic foraging MLE model fitting results

**New:** Now supports both the **AIND Analysis Framework** and Han's original pipeline with full backward compatibility.

```python
from aind_analysis_arch_result_access import get_mle_model_fitting

# Get MLE fitting results for a specific session
df = get_mle_model_fitting(
    subject_id="730945",
    session_date="2024-10-24"
)
print(df.columns)
# Index(['_id', 'nwb_name', 'session_date', 'status', 'subject_id',
#        'analysis_time', 'agent_alias', 'n_trials', 'log_likelihood',
#        'prediction_accuracy', 'k_model', 'AIC', 'BIC', 'LPT', 'LPT_AIC',
#        'LPT_BIC', 'params', 'prediction_accuracy_test',
#        'prediction_accuracy_fit', 'prediction_accuracy_test_bias_only',
#        'pipeline_source', 'S3_location', 'latent_variables', 'qvalue_spread'],
#       dtype='object')

print(df[["agent_alias", "n_trials", "AIC", "pipeline_source"]])
#                      agent_alias  n_trials          AIC          pipeline_source
# 0  QLearning_L1F0_CKfull_softmax       394   241.922187  han's analysis pipeline
# 1  QLearning_L1F1_CKfull_softmax       394   238.848589  han's analysis pipeline
# 2       ForagingCompareThreshold       394   242.957376  han's analysis pipeline
# ...
```

#### Control which pipeline version to fetch

By default, only the most recent analysis version is returned (prefers AIND Framework if available):

```python
# Default: only fetch the most recent version
df = get_mle_model_fitting(
    subject_id="778869",
    session_date="2025-07-26",
    only_recent_version=True  # default
)
# Found 8 records in AIND Analysis Framework
# Found 12 records in Han's prototype analysis pipeline
# Total: 20 MLE fitting records!
# --- After filtering for successful fittings: 14 records (6 skipped) ---
# --- After filtering for most recent versions: 8 records  ---
#     AIND Analysis Framework: 8
#     Han's prototype analysis pipeline: 0
```

To get results from both pipelines:

```python
# Get all versions from both pipelines
df = get_mle_model_fitting(
    subject_id="778869",
    session_date="2025-07-26",
    only_recent_version=False
)
# Found 8 records in AIND Analysis Framework
# Found 12 records in Han's prototype analysis pipeline
# Total: 20 MLE fitting records!
# --- After filtering for successful fittings: 14 records (6 skipped) ---
# WARNING: Duplicated records for the same session and agent_alias!
#          You should check the nwb_name, n_trials, or pipeline_source
#          to select the ones you want.
```

#### Filter by agent/model

```python
# Get all sessions for a specific model
df = get_mle_model_fitting(
    subject_id="730945",
    agent_alias="QLearning_L2F1_CK1_softmax"
)
```

#### Download figures

```python
df = get_mle_model_fitting(
    subject_id="730945",
    session_date="2024-10-24",
    if_download_figures=True,
    download_path="./mle_figures"
)
```

Example figure:

<img width="1153" alt="image" src="https://github.com/user-attachments/assets/84ebd7d3-ac49-4b8f-a0a6-41cced555437" />

#### Advanced: Custom queries

```python
df = get_mle_model_fitting(
    from_custom_query={
        "analysis_results.fit_settings.agent_alias": "QLearning_L2F1_CK1_softmax",
        "analysis_results.n_trials": {"$gt": 600},
    },
    if_include_latent_variables=False
)
print(df.shape)
# (807, 22)
```

---

### Access Han's pipeline utilities
---

### Access Han's pipeline utilities

#### Fetch the session master table in [Streamlit](https://foraging-behavior-browser.allenneuraldynamics-test.org/)

```python
from aind_analysis_arch_result_access.han_pipeline import get_session_table

df_master = get_session_table(if_load_bpod=False)  # `if_load_bpod=True` will load additional 4000+ old sessions from bpod

# Get only recent sessions
df_recent = get_session_table(if_load_bpod=False, only_recent_n_month=6)  # Last 6 months
```
#### Fetch logistic regression results

Get logistic regression results from one session:

```python
from aind_analysis_arch_result_access.han_pipeline import get_logistic_regression
import pandas as pd

df_logistic = get_logistic_regression(
    df_sessions=pd.DataFrame({
        "subject_id": ["769253"],
        "session_date": ["2025-03-12"],
    }),
    model="Su2022",
)
```

Get logistic regression results in batch:

```python
df_logistic = get_logistic_regression(
    df_master.query("subject_id == '769253'"),  # All sessions from a single subject
    model="Su2022",
    if_download_figures=True,  # Also download fitting plots
    download_path="./tmp",
)
```

#### Fetch trial table (🚧 under development)
#### Fetch analysis figures (🚧 under development)

---

## Contributing

### Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e .[dev]
```

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```bash
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```bash
black .
```

- Use **isort** to automatically sort import statements:
```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Semantic Release

The table below, from [semantic release](https://github.com/semantic-release/semantic-release), shows which commit message gets you which release type when `semantic-release` runs (using the default configuration):

| Commit message                                                                                                                                                                                   | Release type                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `fix(pencil): stop graphite breaking when too much pressure applied`                                                                                                                             | ~~Patch~~ Fix Release, Default release                                                                          |
| `feat(pencil): add 'graphiteWidth' option`                                                                                                                                                       | ~~Minor~~ Feature Release                                                                                       |
| `perf(pencil): remove graphiteWidth option`<br><br>`BREAKING CHANGE: The graphiteWidth option has been removed.`<br>`The default graphite width of 10mm is always used for performance reasons.` | ~~Major~~ Breaking Release <br /> (Note that the `BREAKING CHANGE: ` token must be in the footer of the commit) |

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o docs/source/ src
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html docs/source/ docs/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).

### Read the Docs Deployment
Note: Private repositories require **Read the Docs for Business** account. The following instructions are for a public repo.

The following are required to import and build documentations on *Read the Docs*:
- A *Read the Docs* user account connected to Github. See [here](https://docs.readthedocs.com/platform/stable/guides/connecting-git-account.html) for more details.
- *Read the Docs* needs elevated permissions to perform certain operations that ensure that the workflow is as smooth as possible, like installing webhooks. If you are not the owner of the repo, you may have to request elevated permissions from the owner/admin. 
- A **.readthedocs.yaml** file in the root directory of the repo. Here is a basic template:
```yaml
# Read the Docs configuration file
# See https://docs.readthedocs.io/en/stable/config-file/v2.html for details

# Required
version: 2

# Set the OS, Python version, and other tools you might need
build:
  os: ubuntu-24.04
  tools:
    python: "3.13"

# Path to a Sphinx configuration file.
sphinx:
  configuration: docs/source/conf.py

# Declare the Python requirements required to build your documentation
python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - dev
```

Here are the steps for building docs in *Read the Docs*. See [here](https://docs.readthedocs.com/platform/stable/intro/add-project.html) for detailed instructions:
- From *Read the Docs* dashboard, click on **Add project**.
- For automatic configuration, select **Configure automatically** and type the name of the repo. A repo with public visibility should appear as you type. 
- Follow the subsequent steps.
- For manual configuration, select **Configure manually** and follow the subsequent steps

Once a project is created successfully, you will be able to configure/modify the project's settings; such as **Default version**, **Default branch** etc.
