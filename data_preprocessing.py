import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

patterns = []
responses = []
tags = []
for intent in data:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'][0])  # Assuming you will take the first response as the desired response
        tags.append(intent['tag'])

# Tokenize the patterns
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(patterns)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(patterns)
padded_sequences = pad_sequences(sequences, padding='post')

# Convert tags to numerical values
unique_tags = list(set(tags))
tag_index = {tag: i for i, tag in enumerate(unique_tags)}
numerical_tags = [tag_index[tag] for tag in tags]
