import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["lint", "test", "coverage"]


@nox.session(reuse_venv=True)
def test(session: nox.Session) -> None:
    session.run("uv", "sync", "--group", "dev", external=True)
    session.run("uv", "run", "python", "-m", "pytest", "tests", external=True)


# Code coverage
@nox.session(reuse_venv=True)
def coverage(session: nox.Session) -> None:
    # Coverage analysis
    session.run("uv", "sync", "--group", "dev", external=True)
    session.run("uv", "run", "py.test", "--cov=src", "tests", external=True)
    session.run("uv", "run", "coverage", "report", "--show-missing", external=True)  # "--fail-under=95")


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    # Run the ruff linter, way faster than flake8
    session.run("uv", "sync", "--group", "dev", external=True)
    session.run("uv", "run", "ruff", "check", external=True)
