"""Errors shared by the ingest steps."""
import click


class MissingStepInput(click.ClickException):
    """A step was run before the step that produces its input.

    Subclasses ClickException so the CLI renders it as a plain message rather
    than a traceback — these are operator errors, not bugs.
    """

    def __init__(self, *, what: str, run_step: str):
        super().__init__(f"{what}. Run `mm-ingest {run_step}` first.")
