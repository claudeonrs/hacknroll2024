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
        substring_to_summarize=','.join(comment_list[0:subgroup_size]).translate(str.maketrans("\n\t","  "))
        
        curr_comment_list.append(summarizer(substring_to_summarize,max_length=100,do_sample=False)[0]['summary_text'])
        # i+=subgroup_size
        # curr_comment_list = summarizer(next_comment_list)
        for i in range(subgroup_size):
            if len(curr_comment_list) == 1:
                break
            else:
                curr_comment_list.pop(0)
    return curr_comment_list[0]