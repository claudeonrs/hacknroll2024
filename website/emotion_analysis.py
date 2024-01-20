from transformers import pipeline

PIPELINE = 'sentiment-analysis'
MODEL = 'arpanghoshal/EmoRoBERTa'
emotion = pipeline(PIPELINE, model=MODEL)

def emoroberta_classification(text, debug=False):
    '''
    Outputs emotion
    
    Just need to input text

    debug = True if you want to print the original text
    '''
    emotion_labels = emotion(text)
    # emotion_label = emotion_labels[0]
    if debug:
        print(text)
    emotion_label = emotion_labels[0]['label']
    return emotion_label