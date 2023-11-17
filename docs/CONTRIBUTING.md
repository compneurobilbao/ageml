# Contributing to AgeModelling
---
This guide shows the guidelines to contribute to this project.

//TODO: Explain which is our commit, merge, push, etc. philosophy (Good-to-knows)


## Project Structure

The project structure is composed by files which are tracked or ignored by git. Usually only code-related files are
tracked, while big files, such as images or binaries, should be avoided.

```
ageml
│
├── bin                             # Scripts, CLIs and GUIs that use the package. Helpful for debugging.
│   ├── scripts/
│   ├── main.py                     # Usually, main entry point for the main application.
│   └── anything_that_should_not_be_exported_as_a_package

│
│
├── data                            # **Not tracked** - where test data should be included in the developers' file systems
│   └──.gitkeep
│
├── docs
│   ├──CONTRIBUTING.md          # Contribution Guidelines.
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

To install the required packages for creating the environment with poetry.
`pip install poetry, nox, nox-poetry`

If you want the virtual environment of the project to be located in the project root, run this line (make sure you have poetry installed running `poetry --version`):
`poetry config virtualenvs.in-project true` (makes sense)

Poetry will provide to install in the virtual environment (in developer mode) when running_ `poetry install`.
A _pyproject.toml_ file is provided for creating the environment using poetry.

### Activate the environment

At this point, a virtual environment should have been created automatically with all the required dependencies.
If this is something that could be launched somehow, activate the poetry shell:

```
poetry shell
```

Or you can also run:
```
. <path_to_your_virtual_env>/bin/activate
```

### Running the tests
Before pushing anything with significant changes in code/functionality, tests should be (ideally) run locally.
With the environment activated, run:
```
nox -s test
```
For linting:
```
nox -s lint
```
For coverage:
```
nox -s cover
```

### Commits

We try to follow the [**seven rules of a great Git commit message**](https://cbea.ms/git-commit/).

We like using the following standard prefixes for commit messages:

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