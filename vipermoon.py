import shutil
from parser.ast import print_ast
from parser.error import CompileError, format_error
from parser.parser import Parser
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


@click.command()
@click.argument("sourcefile")
@click.option(
    "--print-source",
    default=True,
    help="Whether or not to print the source.",
)
@click.option(
    "--print-tokens", default=True, help="Whether or not to print the tokens."
)
def parse(sourcefile: str, print_source: bool, print_tokens: bool) -> None:
    with open(sourcefile, "r", errors="replace") as f:
        source = f.read()

        if print_source:
            print(source)
            click.echo("-" * shutil.get_terminal_size().columns + "\n")

        if print_tokens:
            for token in Scanner(source):
                print(token)
            click.echo("-" * shutil.get_terminal_size().columns + "\n")

        try:
            print_ast(Parser(source).parse_chunk())
        except CompileError as e:
            print(format_error(e))


@click.group()
def cli(): ...


cli.add_command(scan)
cli.add_command(parse)
