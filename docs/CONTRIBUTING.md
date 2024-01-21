# Contributing to AgeModelling

---
This guide shows the guidelines to contribute to this project.

//TODO: Explain which is our commit, merge, push, etc. philosophy (Good-to-knows)

## Project Structure

The project structure is composed by files which are tracked or ignored by git. Usually only code-related files are
tracked, while big files, such as images or binaries, should be avoided.

```text
ageml
│
├── bin                             # Scripts, CLIs and GUIs that use the package. Not packaged. Helpful for debugging.
│   ├── scripts/
│   ├── main.py                     # Usually, main entry point for the main application.
│   └── all_that_should_not_be_in_the_build
│
│
│
├── data                            # NOT TRACKED BY GIT. Where test data should be included in the developers' file systems
│   └──.gitkeep
│
├── docs
│   ├──CONTRIBUTING.md              # Contribution Guidelines.
│   └──...
│
├── resources                       # Folder with figures and other supporting files 
│
│
├── src                             # Contains all the source code of the package
│    └── ageml                    
│         ├── ...   
│         ├── my_awesome_subpkg1
│         │   ├── __init__.py
│         │   └── awesome_fun.py                           
│         └── my_awesome_subpkg2
│             ├── __init__.py
│             └── awesome_fun.py
│
│
├── tests                           # Testing folder, follows the package structure of src/pyTemplate
│   ├── __init__.py
│   ├── test_my_awesome_subpkg1
│   └── test_my_awesome_subpkg2
│       ├── __init__.py
│       └── test_awesome_fun
│
│
├── .coverage                       # File to measure code coverage, percentage of tested code lines 
├── README.md                        
├── pyproject.toml                  # Requirements for environment settings, packaging and so on
├── poetry.lock                     # Dependency for building the system
├── noxfile.py                      # Defines the linting, coverage, pytest sessions
├── setup.cfg                       # Defines the linting rules
└── .gitignore                      # Files/directories that should not be tracked
```

## Prepare and set up the package

Poetry is our environment manager and build system. To install the required packages for creating the environment with poetry.

__NOTE__: If you are using Mac OS, make sure you are installing pip correctly, either by installing python3 via homebrew, or other tested methods.
`pip install poetry nox nox-poetry`

If you want the virtual environment of the project to be located in the project root, run this line (make sure you have poetry installed running `poetry --version`):
`poetry config virtualenvs.in-project true`

Poetry will provide to install in the virtual environment (in developer mode) when running_ `poetry install`.
A _pyproject.toml_ file is provided for creating the environment using poetry.

Refer to the [poetry documentation](https://python-poetry.org/docs/) for more information.

### Activate the environment

At this point, a virtual environment should have been created automatically with all the required dependencies.
If this is something that could be launched somehow, activate the poetry shell:

```(bash)
poetry shell
```

Or you can also run:

```(bash)
. <path_to_your_virtual_env>/bin/activate
```

### Running the tests

Before pushing anything with significant changes in code/functionality, tests should be (ideally) run locally.
With the environment activated, run:

```(bash)
nox -s test
```

For linting:

```(bash)
nox -s lint
```

For coverage:

```(bash)
nox -s coverage
```

### Commits

We try to follow the [**seven rules of a great Git commit message**](https://cbea.ms/git-commit/).

We like using the following standard prefixes for commit messages. Rigidness is not of our liking, so as long as the commit message is informative about its changes, you are good to go.

- `BUG:` *Fix for runtime crash or incorrect result*
- `DOC:` *Documentation change*
- `ENH:` *New functionality*
- `PERF:` *Performance improvement*
- `REF:` *Only refactoring -> moving classes, files, splitting functions*
- `TST:` *Adding/improving testing of existing functions*
- `STYLE:` *No logic impact (indentation, comments, variable names)*
- `WIP:` *Work In Progress not ready for merge*
- `GIT:` *Modify some repository settings (gitignore, gitmodules, others)*
- `DEL:` *Deleted files, functions, classes, resources and so on*
- `CI:` *Changes in the the CI/CD Pipelines*

### Coding Style Guide

We try following the [PEP-8 Standard](https://peps.python.org/pep-0008/), and we use [Flake-8](https://flake8.pycqa.org/en/latest/) for linting the code.
