import shutil
from parser.error import CompileError, format_error
from parser.scanner import Scanner

import click


@click.command()
@click.argument("sourcefile")
@click.option(
    "--print-source",
    default=True,
    help="Whether or not to print the source before scanning.",
)
def scan(sourcefile: str, print_source: bool) -> None:
    with open(sourcefile, "r", errors="replace") as f:
        source = f.read()

        if print_source:
            print(source)
            click.echo("-" * shutil.get_terminal_size().columns + "\n")

        try:
            scanner = Scanner(source)
            for token in scanner:
                click.echo(token)
        except CompileError as e:
            print(format_error(e))


@click.group()
def cli(): ...


cli.add_command(scan)
