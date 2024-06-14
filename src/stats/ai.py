from os import listdir
from pathlib import Path
from typing import List

import click
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from pandas import DataFrame
from progress.bar import Bar
from pyfs import isDirectory, resolvePath

from src.stats import NATURE_SUBJECTS


@click.command()
@click.option(
    "-d",
    "--dir",
    "abstractDirectory",
    required=False,
    type=Path,
    help="Path to abstract directory to read files",
    default=Path("../../data/abstracts"),
    show_default=True,
)
@click.option(
    "-o",
    "--output-dir",
    "jsonDirectory",
    required=False,
    type=Path,
    help="Path to store JSON output",
    default=Path("../../data/json"),
    show_default=True,
)
def main(abstractDirectory: Path, jsonDirectory: Path) -> None:
    absAbstractDirectory: Path = resolvePath(path=abstractDirectory)
    absJSONDirectory: Path = resolvePath(path=jsonDirectory)

    assert isDirectory(path=absAbstractDirectory)
    assert isDirectory(path=absJSONDirectory)

    fileData: dict[str, List[str]] = {}
    data: dict[str, List[str]] = {}

    systemPrompt: str = f"Classify the following text as one of the following classes and return only the classification: {','.join(NATURE_SUBJECTS)}"

    output_parser = StrOutputParser()
    chatPrompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [("system", systemPrompt), ("user", "{input}")]
    )

    llm: Ollama = Ollama(model="gemma")

    chain: RunnableSequence = chatPrompt | llm | output_parser

    filepaths: List[Path] = [
        Path(absAbstractDirectory, fp) for fp in listdir(path=absAbstractDirectory)
    ]

    with Bar("Reading abstract files into a DataFrame...", max=len(filepaths)) as bar:
        filepath: Path
        for filepath in filepaths:
            ptm: str = filepath.stem

            with open(file=filepath, mode="r") as fp:
                fileData[ptm] = [
                    abstract.strip().lower() for abstract in fp.readlines()
                ]
                fp.close()

            bar.next()

    df: DataFrame = DataFrame(data=fileData)
    ptms: List[str] = df.columns.to_list()

    ptm: str
    for ptm in ptms:
        data[ptm] = []
        with Bar(f"Analyzing {ptm} abstracts...", max=df[ptm].shape[0]) as bar:
            for abstract in df[ptm]:
                data[ptm].append(chain.invoke({"input": abstract}))
                bar.next()

    DataFrame(data=data).T.to_json(
        path_or_buf=Path(absJSONDirectory, "ai_nature_classes.json"),
        indent=4,
    )


if __name__ == "__main__":
    main()
