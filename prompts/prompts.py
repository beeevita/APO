# Task
# task = """Please perform Sentiment Classification task. Given the sentence, assign a sentiment label from ['negative', 'positive']. Return label only without any other text."""
task = {
    "sst2": {
        "seed5": "You are a sentiment classifier. To do this, you must first understand the meaning of the sentence and any relevant context. And then you should classify it as positive or negative",
        "seed10": "Rate each sentence as positive or negative.",
        "seed15": "In this task, you are given sentences from movie reviews. The task is to classify a sentence as  positive or as negative",
    },
    "cr": {
        "seed5":"In this task, you are given sentences from product reviews. The task is to classify a sentence as  positive or as negative.",
        "seed10": "You are a sentiment classifier. To do this, you must first understand the meaning of the sentence and any relevant context. And then you should classify it as positive or negative.",
        "seed15": "Please perform Sentiment Classification task. Given the sentence, assign a sentiment label from ['negative', 'positive']. Return label only without any other text"
    },
    "mr": {
        "seed5": "classify each input as either positive or negative.",
        "seed10": "Given a tweet, classify it as having a positive or negative sentiment.",
        "seed15": "Determine the sentiment of a product review based on the text provided, positive or negative."
    },
    "subj": {
        "seed5": "produce input-output pairs that illustrate the subjectivity of reviews and opinions.",
        "seed10": "produce input-output pairs that illustrate the subjectivity of reviews and opinions." ,
        "seed15": "produce input-output pairs that illustrate the subjectivity of reviews and opinions."
    }
}
