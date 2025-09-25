# aind-analysis-arch-result-access

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-98.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.9-blue?logo=python)


APIs to access analysis results in the AIND behavior pipeline.

## Installation

```bash
pip install aind-analysis-arch-result-access
```

## Usage

Try the demo: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14Hph9QuySbgSQBKl8PGi_nCQfoLcLUI-?usp=sharing)

### Access pipeline v1.0 (Han's "temporary" pipeline)
#### Fetch the session master table in [Streamlit](https://foraging-behavior-browser.allenneuraldynamics-test.org/)
```python
from aind_analysis_arch_result_access.han_pipeline import get_session_table
df_master = get_session_table(if_load_bpod=False)  # `if_load_bpod=True` will load additional 4000+ old sessions from bpod
```
#### Fetch logistic regression results
- Get logistic regression results from one session
    ```python
    from aind_analysis_arch_result_access.han_pipeline import get_logistic_regression
    df_logistic = get_logistic_regression(
        df_sessions=pd.DataFrame(
            {
                "subject_id": ["769253"],
                "session_date": ["2025-03-12"],
            }
        ),
        model="Su2022",
    )
    ```
- Get logistic regression results in batch (from any dataframe with `subject_id` and `session_date` columns)
    ```python
    df_logistic = get_logistic_regression(
        df_master.query("subject_id == '769253'"),  # All sessions from a single subject (query from the `df_master` above)
        model="Su2022",
        if_download_figures=True,  # Also download fitting plots
        download_path="./tmp",
    )
    ```

#### Fetch trial table (🚧 under development)
#### Fetch analysis figures (🚧 under development)
### Access pipeline v2.0 (AIND analysis architecture)
#### Fetch dynamic foraging MLE model fitting results
- Get all MLE fitting results from one session

    ```python
    from aind_analysis_arch_result_access.han_pipeline import get_mle_model_fitting
    df = get_mle_model_fitting(subject_id="730945", session_date="2024-10-24")

    print(df.columns)
    print(df[["agent_alias", "AIC", "prediction_accuracy_10-CV_test"]])
    ```
    output
    ```
    Query: {'analysis_spec.analysis_name': 'MLE fitting', 'analysis_spec.analysis_ver': 'first version @ 0.10.0', 'subject_id': '730945', 'session_date': '2024-10-24'}
    Found 5 MLE fitting records!
    Found 5 successful MLE fitting!
    Get latent variables from s3: 100%|██████████| 5/5 [00:00<00:00, 58.01it/s]

    Index(['_id', 'nwb_name', 'status', 'agent_alias', 'log_likelihood', 'AIC',
          'BIC', 'LPT', 'LPT_AIC', 'LPT_BIC', 'k_model', 'n_trials',
          'prediction_accuracy', 'prediction_accuracy_test',
          'prediction_accuracy_fit', 'prediction_accuracy_test_bias_only',
          'params', 'prediction_accuracy_10-CV_test',
          'prediction_accuracy_10-CV_test_std', 'prediction_accuracy_10-CV_fit',
          'prediction_accuracy_10-CV_fit_std',
          'prediction_accuracy_10-CV_test_bias_only',
          'prediction_accuracy_10-CV_test_bias_only_std', 'latent_variables'],
          dtype='object')

                      agent_alias          AIC  prediction_accuracy_10-CV_test
    0  QLearning_L1F1_CK1_softmax   239.519051                        0.898151
    1         QLearning_L1F0_epsi   403.621460                        0.762075
    2  QLearning_L2F1_CK1_softmax   236.265381                        0.903280
    3                        WSLS  4051.958064                        0.636196
    4      QLearning_L2F1_softmax   236.512476                        0.888611
    ```
    Now the latent variables also contain the `rpe`.
    ```python
    df.latent_variables.iloc[0].keys()
    ```
    output
    ```
    dict_keys(['q_value', 'choice_kernel', 'choice_prob', 'rpe'])
    ```

-  Also download figures
    ```python
    df = get_mle_model_fitting(
        subject_id="730945",
        session_date="2024-10-24",
        if_download_figures=True,
        download_path="./mle_figures",
    )
    !ls ./mle_figures
    ```
    output
    ```
    Query: {'analysis_spec.analysis_name': 'MLE fitting', 'analysis_spec.analysis_ver': 'first version @ 0.10.0', 'subject_id': '730945', 'session_date': '2024-10-24'}
    Found 5 MLE fitting records!
    Found 5 successful MLE fitting!
    Get latent variables from s3: 100%|██████████| 5/5 [00:00<00:00, 85.87it/s]
    Download figures from s3: 100%|██████████| 5/5 [00:00<00:00, 86.45it/s]

    730945_2024-10-24_17-38-06_QLearning_L1F0_epsi_58cc5b6f6e.png
    730945_2024-10-24_17-38-06_QLearning_L1F1_CK1_softmax_3ffdf98012.png
    730945_2024-10-24_17-38-06_QLearning_L2F1_CK1_softmax_5ce7f1f816.png
    730945_2024-10-24_17-38-06_QLearning_L2F1_softmax_ec59be40c0.png
    730945_2024-10-24_17-38-06_WSLS_7c61d01e0f.png
    ```
    Example figure:

    <img width="1153" alt="image" src="https://github.com/user-attachments/assets/84ebd7d3-ac49-4b8f-a0a6-41cced555437" />


- Get fittings from all sessions of a mouse for a specific model
    ```python
    df = get_mle_model_fitting(
        subject_id="730945",
        agent_alias="QLearning_L2F1_CK1_softmax",
        if_download_figures=False,
    )
    print(df.iloc[:10][["nwb_name", "agent_alias"]])
    ```
    output
    ```
    Query: {'analysis_spec.analysis_name': 'MLE fitting', 'analysis_spec.analysis_ver': 'first version @ 0.10.0', 'subject_id': '730945', 'analysis_results.fit_settings.agent_alias': 'QLearning_L2F1_CK1_softmax'}
    Found 32 MLE fitting records!
    Found 32 successful MLE fitting!
    Get latent variables from s3: 100%|██████████| 32/32 [00:00<00:00, 80.81it/s]

                            nwb_name                 agent_alias
    0  730945_2024-08-27_16-07-16.nwb  QLearning_L2F1_CK1_softmax
    1  730945_2024-09-05_16-47-58.nwb  QLearning_L2F1_CK1_softmax
    2  730945_2024-10-23_15-33-07.nwb  QLearning_L2F1_CK1_softmax
    3  730945_2024-09-19_17-26-54.nwb  QLearning_L2F1_CK1_softmax
    4  730945_2024-09-04_16-04-38.nwb  QLearning_L2F1_CK1_softmax
    5  730945_2024-08-30_15-55-05.nwb  QLearning_L2F1_CK1_softmax
    6  730945_2024-08-29_15-50-57.nwb  QLearning_L2F1_CK1_softmax
    7  730945_2024-10-24_17-38-06.nwb  QLearning_L2F1_CK1_softmax
    8  730945_2024-09-12_17-21-58.nwb  QLearning_L2F1_CK1_softmax
    9  730945_2024-09-03_15-49-53.nwb  QLearning_L2F1_CK1_softmax
    ```

- (for advanced users) Use your own docDB query
    ```python
    df = get_mle_model_fitting(
        from_custom_query={
            "analysis_results.fit_settings.agent_alias": "QLearning_L2F1_CK1_softmax",
            "analysis_results.n_trials" : {"$gt": 600},
        },
        if_include_latent_variables=False,
        if_download_figures=False,
    )
    ```
    output
    ```
    Query: {'analysis_spec.analysis_name': 'MLE fitting', 'analysis_spec.analysis_ver': 'first version @ 0.10.0', 'analysis_results.fit_settings.agent_alias': 'QLearning_L2F1_CK1_softmax', 'analysis_results.n_trials': {'$gt': 600}}
    Found 807 MLE fitting records!
    Found 807 successful MLE fitting!
    ```


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
