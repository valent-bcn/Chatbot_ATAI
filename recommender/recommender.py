import pandas as pd
import numpy as np
import os



class RecommendationSolver:
    '''
    This class is used to recommend movies based on the similarity matrix
    '''
    def __init__(self):
        # We start loading the data from .parquet files
        self.similarity_df = pd.read_parquet('dataset/similarity_matrix/similarity.parquet', engine='pyarrow')
        #films
        self.films_df = pd.read_csv('dataset/films_clean.csv')

    def process_recommendation_direct(self, entities: dict):
        # Ottieni gli ID degli entity
        ids = [key.split('/')[-1] for key in entities.keys()]

        # Intersezione degli ID comuni
        common_ids = self.similarity_df.index.intersection(ids)

        # Calcola il punteggio totale sommando le righe corrispondenti agli IDs comuni
        if not common_ids.empty:
            total_scores = self.similarity_df.loc[common_ids].sum(axis=0).to_numpy()
        else:
            total_scores = np.zeros(self.similarity_df.shape[1])

        # Trova i primi 5 indici con i punteggi più alti usando argsort
        top_indices = np.argsort(-total_scores)

        # Mappa gli indici ai QIDs
        base_url = 'http://www.wikidata.org/entity/'
        top_qids = [base_url + self.similarity_df.columns[i] for i in top_indices]

        # Rimuovi le entità già presenti in result['entities']
        excluded_ids = set(entities.keys())  # Entità da escludere
        filtered_qids = [qid for qid in top_qids if qid not in excluded_ids]

        # Seleziona i primi 5 risultati dopo l'esclusione
        filtered_qids = filtered_qids[:5]

        # Trova le Label corrispondenti
        top_films = self.films_df[self.films_df['ID'].isin(filtered_qids)]

        # Crea un dizionario con ID e Label
        top_dict = dict(zip(top_films['ID'], top_films['Label']))

        return top_dict

    # Esempio di uso della funzione
    #top_films = process_recommendation_direct(result['entities'], matrices, films_df)
    #print(top_films)

