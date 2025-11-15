import shutil
from parser.ast import print_ast
from parser.error import CompileError, format_error
from parser.parser import Parser
from parser.scanner import Scanner
from pprint import pprint

import click

from compiler.assembler import Assembler
from compiler.compile import compile
from compiler.disassembler import Disassembler
from interpreter.interpreter import LuaFrame, run_frame


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


@click.command()
@click.argument("sourcefile")
def disassemble(sourcefile: str) -> None:
    with open(sourcefile, "rb") as f:
        disasm = Disassembler(f.read())
        disasm.disassemble().dump()


@click.command()
@click.argument("sourcefile")
def run(sourcefile: str) -> None:
    with open(sourcefile, "rb") as f:
        disasm = Disassembler(f.read())
        proto = disasm.disassemble()
        frame = LuaFrame(proto)
        run_frame(frame)


@click.command("compile")
@click.argument("sourcefile")
def compile_command(sourcefile: str) -> None:
    with open(sourcefile, "r", errors="replace") as f:
        ast = Parser(f.read()).parse_chunk()
        print_ast(ast)
        click.echo("-" * shutil.get_terminal_size().columns + "\n")
        proto = compile(ast)

        asm = Assembler()
        asm.write_header()
        asm.write_proto(proto)
        asm.save()


@click.group()
def cli(): ...


cli.add_command(scan)
cli.add_command(parse)
cli.add_command(disassemble)
cli.add_command(run)
cli.add_command(compile_command)
