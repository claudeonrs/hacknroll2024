# !pip3 install transformers
# !pip3 install torch torchvision torchaudio
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

ROBERTA_MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL)

def roberta_classification(text, debug=False):
    '''
    Outputs probability if the text is negative/neutral/positive
    
    Just need to input text

    debug = True if you want to print the original text
    '''
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    if debug:
        print(text)
    return scores_dict