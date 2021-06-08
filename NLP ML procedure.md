# NLP ML procedure

1. text processing
    - short text, repetitive -> categorical, not NLP -> LabelEncoder
    - unique text with complex meaning -> NLP
        - short text unique -> fasttext / ngram char level
        - longer text -> ngram on word level
        - need to transform the result to numerical representation
            - CountVectorizer
            - use frequency from vocabulary directly -> GlobalAveragePooling1D

2. NN structure
    - each feature is on the Input layer
    - NLP sparse features goes through NN -> concatenate with other features
        - CountVectorizer + Dense (200,200,100,1) + PReLU
        - pad_sequences + Embedding + GlobalAveragePooling1D + BatchNormalization + Dense + Activation + Dense
        - CountVectorizer + Dense (100, 100, 100, 50, 1) + PReLU

0. Addtional
    - CSR: tocsr()
    - ridge model: Ridge()