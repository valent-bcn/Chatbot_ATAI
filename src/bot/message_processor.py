from flair.data import Sentence
from flair.models import SequenceTagger
import re
from rapidfuzz import process, fuzz
import Levenshtein
import pandas as pd

class AttributeRecognizer:
    def __init__(self, relations_path):
        # Load the data from CSV
        data = pd.read_csv(relations_path) # relations expanded
        # the last '/' separates the identifier from the label
        # dict
        self.relations_dict = {row['Label']: row['ID'].split('/')[-1] for index, row in data.iterrows()}

    def recognize(self, recognized_labels: str) -> tuple:
        # Use the dictionary's keys to find the closest match to the recognized labels
        if recognized_labels == 'directed':
            return ('P57', 'director')
        best_match_label, score, best_match_index = process.extractOne(recognized_labels, list(self.relations_dict.keys()), scorer=fuzz.WRatio)
        # Return the identifier corresponding to the best match label
        return self.relations_dict[best_match_label], best_match_label


class MessageCleaner:
    def __init__(self):
        # Compile a regex that matches the beginning of the interrogative part of the question
        self.wh_pattern = re.compile(r'\b(who|what|when|where)\b', re.IGNORECASE)

    def clean(self, message: str) -> str:
        # Find the position where the first "wh" word occurs
        match = self.wh_pattern.search(message)
        if match:
            # Extract the substring from the "wh" word to the end, excluding the final "?"
            start_index = match.start()
            cleaned_message = message[start_index:].rstrip('?').strip()
            return cleaned_message
        return message  # Return the original message if no "wh" word is found
class MessageDecomposer:
    def __init__(self):
        self.cleaner = MessageCleaner()  # Assuming MessageCleaner is defined elsewhere
        self.ner_tagger = SequenceTagger.load('ner')
        self.pos_tagger = SequenceTagger.load('pos-fast')  # Load POS tagging model
        self.special_chars = bool

    def decompose(self, message: str) -> tuple:
        cleaned_message = self.cleaner.clean(message)
        sentence = Sentence(cleaned_message)

        # Predict NER to identify and segregate entities
        self.ner_tagger.predict(sentence)
        ner_entities = sentence.get_spans('ner')

        # Collect all entity texts
        entities_text = ' '.join(ent.text for ent in ner_entities)

        # Create a set of character spans to exclude (these cover the entity spans)
        exclude_indices = set()
        for ent in ner_entities:
            for token in ent.tokens:
                #print(token)
                exclude_indices.add(token.idx - 1)  # token.idx is 1-based

        # Create a new Sentence object from the tokens not covered by any entity span
        filtered_tokens = [token for i, token in enumerate(sentence.tokens) if i not in exclude_indices]

        # Create a new Sentence object from the filtered tokens for POS tagging
        filtered_sentence = Sentence(' '.join(token.text for token in filtered_tokens))

        # Predict POS tags for the filtered sentence
        self.pos_tagger.predict(filtered_sentence)

        pos_tags_text = ' '.join(token.text for token in filtered_sentence if token.labels[0].value in
                                 ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
                                 and token.text not in ['is', 'was', 'were', 'are', 'be', 'been', 'being', 'has', 'have', 'had'])

        return pos_tags_text, entities_text


# Example usage
decomposer = MessageDecomposer()
pos_tags, entities = decomposer.decompose("Who is the director of The Angry Birds Movie 2?") # hard core of directed and maybe broadcaster
#hard code date when "when"
print(f"POS Tags: {pos_tags}")
print(f"Entities: {entities}")

# Identifier
recognizer = AttributeRecognizer('relations_expanded.csv')
print(recognizer.recognize(pos_tags))