from rdflib.namespace import Namespace
from typing import List
import rdflib


class SPARQLQuerySolver:
    #For now we assume the query is given with the prefixes
    #WD = Namespace('http://www.wikidata.org/entity/')
    #WDT = Namespace('http://www.wikidata.org/prop/direct/')
    #SCHEMA = Namespace('http://schema.org/')
    #DDIS = Namespace('http://ddis.ch/atai/')

    def __init__(self, data_path: str = 'dataset/14_graph.nt', format: str = 'turtle'):
        self.graph = rdflib.Graph()
        self.graph.parse(data_path, format=format)

    def solveQuery(self, query: str) -> List[str]:
        try:
            results = self.graph.query(query)
            # Extract only the literal
            return [str(result[0]) for result in results]
        except Exception as e:
            print(f"An error occurred during SPARQL query execution: {e}")
            return []