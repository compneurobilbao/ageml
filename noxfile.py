import nox
from nox_poetry import Session, session

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["lint", "test", "cover"]


@session(reuse_venv=True)
def test(s: Session) -> None:
    s.run("poetry", "install", external=True)
    s.run("python", "-m", "pytest", "tests")


# Code coverage
@session(reuse_venv=True)
def coverage(s: Session) -> None:
    # Coverage analysis
    s.run("poetry", "install", external=True)
    s.run("py.test", "--cov=src", "tests")
    s.run("coverage", "report", "--show-missing")  # "--fail-under=95")


@session(reuse_venv=True)
def lint(s: Session) -> None:
    # Run pyproject-flake8 entrypoint to support reading configuration from pyproject.toml.
    s.run("poetry", "install", external=True)
    s.run("flake8")
