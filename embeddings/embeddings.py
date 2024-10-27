import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class EmbeddingResolver:
    def __init__(self):
        # Definisce i percorsi ai file nella cartella `dataset`
        entity_embed_path = 'dataset/ddis-graph-embeddings/entity_embeds.npy'
        relation_embed_path = 'dataset/ddis-graph-embeddings/relation_embeds.npy'
        entity_del_path = 'dataset/ddis-graph-embeddings/entity_ids.del'
        relation_del_path = 'dataset/ddis-graph-embeddings/relation_ids.del'
        entities_clean_path = 'dataset/entities_clean.csv'
        
        # Carica gli embeddings
        self.entity_embeddings = np.load(entity_embed_path)
        self.relation_embeddings = np.load(relation_embed_path)

        # Carica gli identificatori dal file .del per ottenere gli ID
        self.entity_ids = self._load_del_file(entity_del_path)
        self.relation_ids = self._load_del_file(relation_del_path)

        # Carica il dataset con i label delle entità
        self.entities_df = pd.read_csv(entities_clean_path)

    def _load_del_file(self, del_file_path):
        # Carica il file .del e crea una lista di identificatori
        with open(del_file_path, 'r', encoding='utf-8') as file:
            return [line.strip().split('\t')[1] for line in file]

    def find_most_plausible_responses(self, decomposed_output, top_n=3):
        # Estrai l'ID di entità e relazione dal risultato decomposizione
        entity_id = list(decomposed_output.get('entities', {}).keys())[0]
        relation_id = list(decomposed_output.get('pos_tags', {}).keys())[0]

        # Verifica che gli ID esistano nei rispettivi dataset
        if entity_id not in self.entity_ids or relation_id not in self.relation_ids:
            return None

        # Trova gli indici degli embedding per entità e relazione
        entity_index = self.entity_ids.index(entity_id)
        relation_index = self.relation_ids.index(relation_id)

        # Recupera gli embeddings di entità e relazione
        entity_embed = self.entity_embeddings[entity_index].reshape(1, -1)
        relation_embed = self.relation_embeddings[relation_index].reshape(1, -1)

        # Combina gli embeddings di entità e relazione
        combined_embed = entity_embed + relation_embed

        # Calcola la similarità tra il vettore combinato e tutti gli embeddings delle entità
        similarities = cosine_similarity(combined_embed, self.entity_embeddings).flatten()

        # Trova gli indici delle entità con la similarità più alta
        top_indices = np.argsort(-similarities)[:top_n]

        # Ottieni l'ID e il label delle entità più plausibili
        plausible_responses = []
        for idx in top_indices:
            if similarities[idx] >= (similarities[top_indices[0]] - 0.02):  # Mantieni solo le risposte con similarità vicina
                entity_id = self.entity_ids[idx]
                label_row = self.entities_df.loc[self.entities_df['ID'] == entity_id]
                if not label_row.empty:
                    label = label_row.iloc[0]['Label']
                    plausible_responses.append(label)

        return plausible_responses
