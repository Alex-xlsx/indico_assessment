import json
import pandas as pd
from indico.queries import JobStatus, ModelGroupPredict
from indico import IndicoClient, IndicoConfig

# CSV Cleaner ----------------------------------------

def csv_cleaner(csv, column):
    # Establish base variables
    df = pd.read_csv(csv)
    text_list = []
    target_list = []

    # Clean out incomplete entries and put clean data into lists
    for line in df[column]:
        try:
            text_hold = (json.loads(line))['text']
            target_hold = (json.loads(line))['target']
            if text_hold != "" and target_hold != "":
                text_list.append(text_hold)
                target_list.append(target_hold)
        except KeyError:
            pass

    # Convert lists to dictionary
    clean_data = {
        "text": text_list,
        "target": target_list
    }

    # Convert dict into dataframe
    clean_df = pd.DataFrame.from_dict(clean_data)

    # Count rows for every unique Target
    unique_targets = (clean_df['target'].unique())
    for item in unique_targets:
        rows = 0
        for line in clean_df["target"]:
            if line == item:
                rows += 1
        print(f'{item} has {rows} rows')

    # Export a CSV sorted by target then by text
    (clean_df.sort_values(by=["target", "text"])).to_csv('clean_dataset.csv', index=False)


csv_cleaner('sentence_dataset.csv', 'sentences')
# Model Group Predict ----------------------------------------

# Indico API auth
my_config = IndicoConfig(
    host='app.indico.io',
    api_token_path='indico_api_token.txt'
)

class PredictModel:
    def __init__(self, csv, config, model_id):
        self.data = pd.read_csv(csv)
        self.client = IndicoClient(config=config)
        self.job = self.client.call(ModelGroupPredict(
            model_id= model_id,
            data=self.data['text'].tolist()
        ))
        self.predictions_list = []

    # Gather list of predictions
    def find_predictions(self):
        self.predictions_list = self.client.call(JobStatus(id=self.job.id, wait=True)).result

    # Refine and compare predictions
    def compare_predictions(self):
        index = 0
        true = 0
        for line in self.predictions_list:
            higher_prediction = max(self.predictions_list[index], key=self.predictions_list[index].get)
            if higher_prediction == self.data['target'][index]:
                true += 1
            index += 1
        print(f'{(round(true/index,4))*100}% were correctly guessed.')


new_model = PredictModel('clean_dataset.csv', my_config, 33077)
new_model.find_predictions()
new_model.compare_predictions()

# Bigram Probability Calculator ----------------------------------------
# Isolate each unique instance of every word in a dict
def bigram_function(csv):
    my_df = pd.read_csv(csv)
    df_text = my_df['text']
    word_list = []
    for line in df_text:
        temp_list = line.split()
        for word in temp_list:
            temp_word = word.lower()
            word_list.append(temp_word)

    unique_words = [i for n, i in enumerate(word_list) if i not in word_list[n + 1:]]
    word_dict = {each: {} for each in unique_words}

    def next_word(target, source):
        index = 0
        for single in source:
            try:
                if single == target:
                    if source[index + 1] in word_dict[target]:
                        word_dict[target][source[index + 1]] += 1
                    else:
                        word_dict[target].update({source[index + 1]: 1})
            except IndexError:
                pass
            index += 1


    for word in word_dict:
        next_word(word, word_list)
        values = word_dict[word].values()
        total = sum(values)
        word_dict[word].update({"_total": total})

    def word_probability(word):
        found_match = False
        for single in word_dict:
            if word == single:
                found_match = True
                for each in word_dict[word]:
                    if each != "_total":
                        print(f'{each} : {round((word_dict[word][each] / word_dict[word]["_total"])*100,3)}%')
        if found_match is False:
            print("Sorry, no matches were found")


    done = False
    while done is False:
        print("Search for bigrams. Type '_done' when you are finished")
        given_word = input("Please give a word: ")
        if given_word == "_done":
            done = True
            print("Thank you!")
        else:
            word_probability(given_word)

bigram_function('clean_dataset.csv')
