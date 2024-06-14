from sqlite3 import Connection, Cursor
from typing import Any, Iterable, List

OA_DOI_COUNT: int = 7885681
OA_OAID_COUNT: int = 13435534
OA_CITATION_COUNT: int = 113563323
OAPM_ARXIV_PM_PAPERS_IN_OA: int = 14

NATURE_SUBJECTS: List[str] = [
    "Physics",
    "Astronomy and planetary science",
    "Chemistry",
    "Materials science",
    "Mathematics and computing",
    "Engineering",
    "Nanoscience and technology",
    "Optics and photonics",
    "Energy science and technology",
    "Earth and environmental sciences",
    "Climate sciences",
    "Ecology",
    "Environmental sciences",
    "Solid Earth sciences",
    "Planetary science",
    "Environmental social sciences",
    "Biogeochemistry",
    "Ocean sciences",
    "Hydrology",
    "Natural hazards",
    "Limnology",
    "Space physics",
    "Biological sciences",
    "Genetics",
    "Microbiology",
    "Neuroscience",
    "Ecology",
    "Immunology",
    "Evolution",
    "Cancer",
    "Cell biology",
    "Biochemistry",
    "Molecular biology",
    "Zoology",
    "Developmental biology",
    "Biological techniques",
    "Structural biology",
    "Physiology",
    "Biotechnology",
    "Drug discovery",
    "Stem cells",
    "Plant sciences",
    "Computational biology and bioinformatics",
    "Psychology",
    "Biophysics",
    "Chemical biology",
    "Systems biology",
    "Health sciences",
    "Diseases",
    "Health care",
    "Medical research",
    "Anatomy",
    "Pathogenesis",
    "Risk factors",
    "Biomarkers",
    "Neurology",
    "Signs and symptoms",
    "Endocrinology",
    "Health occupations",
    "Scientific community and society",
    "Scientific community",
    "Social sciences",
    "Business and industry",
    "Developing world",
    "Agriculture",
    "Water resources",
    "Geography",
    "Energy and society",
    "Forestry",
]


def runOneValueSQLQuery(db: Connection, query: str) -> Iterable[Any]:
    """
    _runOneValueSQLQuery Execute an SQL query that returns one value

    :param db: An sqlite3.Connection object
    :type db: Connection
    :param query: A SQLite3 compatible query
    :type query: str
    :return: An iterator containing any value
    :rtype: Iterator[Any]
    """
    cursor: Cursor = db.execute(query)
    return cursor.fetchone()
