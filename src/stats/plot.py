from pathlib import Path
from sqlite3 import Connection
from string import Template
from typing import List

import click
import matplotlib.pyplot as plt
import seaborn
from humanize import intcomma
from matplotlib.axes import Axes
from pandas import DataFrame

from src.stats.stats import *


def _humanizeInt(number: int | float) -> str:
    """
    _humanizeInt Humanize a number into an interger string

    :param number: A value to humanize
    :type number: int | float
    :return: A humanized version of the `number` parameter
    :rtype: str
    """
    return intcomma(value=number, ndigits=0)


def _renameURL(url: str) -> str:
    """
    _renameURL Given a URL, rename it to a specific string

    :param url: A URL to rename
    :type url: str
    :return: A renamed version of the URL
    :rtype: str
    """
    url = url.strip()

    if url == "":
        return "Unknown"

    hostname: str = url.split(".")[0]

    match hostname:
        case "arxiv":
            return "arXiv"
        case "aclanthology":
            return "ACL\nAnthology"
        case "github":
            return "GitHub"
        case "huggingface":
            return "Hugging Face"
        case _:
            return hostname


def plot_DatasetSizes(
    oaSize: int,
    pmSize: int,
    filepath: Path,
) -> None:
    """
    plot_DatasetSizes Plot the number of papers of OpenAlex and PeaTMOSS

    :param oaSize: Number of papers in the OpenAlex dataset
    :type oaSize: int
    :param pmSize: Number of papers in the PeaTMOSS dataset
    :type pmSize: int
    :param filepath: A path to save the output figure
    :type filepath: Path
    """
    data: dict[str, List[str | int]] = {
        "x": ["OpenAlex", "PeaTMOSS"],
        "y": [oaSize, pmSize],
    }

    df: DataFrame = DataFrame(data=data)

    graph: Axes = seaborn.barplot(data=df, x="x", y="y")
    graph.set_yscale(value="log")
    plt.title(label="Number of Papers per Dataset")
    plt.xlabel(xlabel="Dataset")
    plt.ylabel(ylabel="Number of Papers")
    graph.bar_label(
        container=graph.containers[0],
        fmt=_humanizeInt,
    )
    plt.savefig(filepath)
    plt.clf()


def plot_PMPublicationVenuePaperCount(
    venuePaperCounts: Series,
    filepath: Path,
) -> None:
    """
    plot_PMPublicationVenuePaperCount Plot the number of PeaTMOSS papers per publication venue

    :param venuePaperCounts: A count of the venues that papers are published in
    :type venuePaperCounts: Series
    :param filepath: Path to save figure to
    :type filepath: Path
    """
    data: Series = venuePaperCounts.iloc[0:4]

    url: str
    for url in data.index:
        data = data.rename({url: _renameURL(url=url)})

    other: int = venuePaperCounts[4:].sum()
    data["Other"] = other
    data.sort_values(inplace=True, ascending=False)

    df: DataFrame = DataFrame(data=data)
    graph: Axes = seaborn.barplot(
        data=df,
        x="url",
        y="count",
    )
    plt.title(label="Number of PeaTMOSS Papers per Venue")
    plt.xlabel(xlabel="Venue", labelpad=7)
    plt.ylabel(ylabel="Number of Papers")
    graph.bar_label(
        container=graph.containers[0],
        fmt=_humanizeInt,
    )

    plt.tight_layout()
    plt.savefig(filepath)
    plt.clf()


def plot_MostCitedArXivPMPapers(
    oaDB: Connection, paperCitationCounts: Series, filepath: Path
) -> None:
    """
    plot_MostCitedArXivPMPapers Plot the most cited PeaTMOSS models published to arXiv

    :param oaDB: A sqlite3.Connection object for an OpenAlex database
    :type oaDB: Connection
    :param paperCitationCounts: A pandas.Series of PeaTMOSS arXiv papers and their citations
    :type paperCitationCounts: Series
    :param filepath: A path to save the figure to
    :type filepath: Path
    """
    # Top 6 choosen because the 4th entry is a dataset and not a DNN
    data: Series = paperCitationCounts[0:6]
    data.drop(labels=data.index[3], inplace=True)

    # Commented out because it resolves OpenAlex IDs to DOI URLs
    # queryTemplate: Template = Template(
    #     template="SELECT doi FROM works WHERE oa_id = '${oaID}'"
    # )
    # oaID: str
    # for oaID in data.index:
    #     query: str = queryTemplate.substitute(oaID=oaID)
    #     result: str = runOneValueSQLQuery(db=oaDB, query=query)[0]

    #     print(f"https://doi.org/{result}")
    labels: List[str] = ["ResNeXt", "Transformer-XL", "HRNet", "MAE", "RegNet"]
    data.index = labels

    df: DataFrame = DataFrame(data=data)
    df.reset_index(drop=False, inplace=True)
    graph: Axes = seaborn.barplot(
        data=df,
        x="index",
        y="count",
    )
    plt.title(label="Number of Citations per PeaTMOSS Model")
    plt.xlabel(xlabel="PeaTMOSS Model")
    plt.ylabel(ylabel="Number of Citations")
    graph.bar_label(
        container=graph.containers[0],
        fmt=_humanizeInt,
    )

    plt.tight_layout()
    plt.savefig(filepath)
    plt.clf()


@click.command()
@click.option(
    "-p",
    "--peatmoss",
    "pmPath",
    type=Path,
    help="Path to PeaTMOSS database",
    required=True,
)
@click.option(
    "-o",
    "--openalex",
    "oaPath",
    type=Path,
    help="Path to OpenAlex database",
    required=True,
)
@click.option(
    "-a",
    "--peatmoss-arxiv-citation-count",
    "pmArxivCitationCount",
    type=Path,
    help="Path to pickled PeaTMOSS arXiv Citation Count",
    required=False,
    default=Path("pmArXivCitations.pickle"),
    show_default=True,
)
def main(pmPath: Path, oaPath: Path, pmArxivCitationCount: Path) -> None:
    seaborn.set_style(style="darkgrid")

    absPMPath: Path = resolvePath(path=pmPath)
    absOAPath: Path = resolvePath(path=oaPath)

    assert isFile(path=absPMPath)
    assert isFile(path=absOAPath)

    pmDB: Connection = connectToDB(dbPath=absPMPath)
    oaDB: Connection = connectToDB(dbPath=absOAPath)

    oaPaperCounts: int = oa_CountPapersByDOI(oaDB=oaDB)
    pmPaperCounts: int = pm_CountPapersByID(pmDB=pmDB)

    pmVenueCounts: Series = pm_CountPapersPerJournal(
        pmDB=pmDB,
    )

    pmPaperCitationCounts: Series
    try:
        pmPaperCitationCounts = pandas.read_pickle(
            filepath_or_buffer=pmArxivCitationCount,
        )
    except FileNotFoundError:
        pmPaperCitationCounts = oapm_CountCitationsOfArXivPMPapers(
            pmDB=pmDB,
            oaDB=oaDB,
        )
        pmPaperCitationCounts.to_pickle(path=pmArxivCitationCount)

    plot_PMPublicationVenuePaperCount(
        venuePaperCounts=pmVenueCounts,
        filepath="numberofPeaTMOSSPapersPerVenue.png",
    )

    plot_DatasetSizes(
        oaSize=oaPaperCounts,
        pmSize=pmPaperCounts,
        filepath=Path("comparisonOfDatasetPaperCounts.png"),
    )

    plot_MostCitedArXivPMPapers(
        oaDB=oaDB,
        paperCitationCounts=pmPaperCitationCounts,
        filepath="numberOfCitationsPerPMModel.png",
    )


if __name__ == "__main__":
    main()
