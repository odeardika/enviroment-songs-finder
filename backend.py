

import system


def preprocess(lyrics):
    lyrics = system.loweeCaseAndNoNumber(lyrics)
    lyrics = system.emojiRemover(lyrics)
    lyrics = system.punctuationRemover(lyrics)
    lyrics = system.normalization(lyrics)
    lyrics = system.no_abbreviation(lyrics)
    lyrics = system.tokenization(lyrics)
    lyrics = system.stopWord(lyrics)
    lyrics_stem = system.stemming(lyrics)
    lyrics_stem = system.remove_empty_token(lyrics_stem)
    lyrics_word = system.wordSort(lyrics_stem)
    TFIDF = system.TFIDF(lyrics_stem,lyrics_word)
    return TFIDF

def check_emotion(df, model, input):
    input = input.reindex(columns=df.columns, fill_value=0)
    prediction = model.predict(input)
    label_to_emotion = {
        0: "lingkungan",
        1: "nonlingkungan",
    }

    return label_to_emotion[prediction[0]]