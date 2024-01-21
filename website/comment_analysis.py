from transformers import pipeline

PIPELINE = "summarization"
MODEL = f"facebook/bart-large-cnn" # Use Bart Model
summarizer = pipeline(PIPELINE,model=MODEL)
STAR_MAX_COUNT = 5
comment_agg = {}

def bart_classification(comment_list,subgroup_size):
    '''
    Summarise every subgroup_size number of comments iteratively till 1 final output comment
    '''
    curr_comment_list,next_comment_list,i=comment_list,[],0
    while len(curr_comment_list)>1 and i<len(curr_comment_list):
        substring_to_summarize=','.join(comment_list[i:i+subgroup_size]).translate(str.maketrans("\n\t","  "))
        next_comment_list.append(summarizer(substring_to_summarize,max_length=100,do_sample=False))
        i+=subgroup_size
    curr_comment_list=next_comment_list
    return curr_comment_list[0]