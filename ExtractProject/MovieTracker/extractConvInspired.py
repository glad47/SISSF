
## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14
## Description: procee TGRedial Dataset used by SISSF(Social Information Sensitive Conversational Recommendation Using Multi-View Contrastive Learning)
import csv
import re
import pickle
import json
import random
from scipy.sparse import dok_matrix
from scipy.sparse import save_npz
# import nltk
# from nltk.tokenize import word_tokenize
import ast
# from transformers import pipeline

# Download the necessary resources
# nltk.download('punkt')
random.seed(42)
# Load the JSON file containing the conversations

# Load a sentiment analysis pipeline
# sentiment_pipeline = pipeline("sentiment-analysis")

seekers = set()
recommenders = set()
users = set()
con= []
seekerList = []
recList = []
convIds = []

conversations = []
test_conversations = []
valid_conversations = []
movies = []
movieDict= {}
userDict= {}
dialougeDict = {}
dialogue = []
dialToUserRole= {}
dialogueMessageDict = {}
dialogueMovies = {}


with open('input/inspired/dialog_data/train.tsv', encoding='utf-8') as file:
    tsv_reader = csv.DictReader(file, delimiter='\t')
    for row in tsv_reader:
        conversations.append(row)  

with open('input/inspired/dialog_data/test.tsv', encoding='utf-8') as file:
    tsv_reader = csv.DictReader(file, delimiter='\t')
    for row in tsv_reader:
        test_conversations.append(row)         

with open('input/inspired/dialog_data/dev.tsv', encoding='utf-8') as file:
    tsv_reader = csv.DictReader(file, delimiter='\t')
    for row in tsv_reader:
        valid_conversations.append(row)     

with open('input/inspired/movie_database.tsv', encoding='utf-8') as file:
    tsv_reader = csv.DictReader(file, delimiter='\t')
    for row in tsv_reader:
        movies.append(row)  

with open('input/inspired/survey_data/list_of_dialog_ids_with_movie_id_all.tsv', encoding='utf-8') as file:
    tsv_reader = csv.DictReader(file, delimiter='\t')
    for row in tsv_reader:
        con.append(row)

with open('input/inspired/survey_data/seeker_demographic.tsv', encoding='utf-8') as file:
    tsv_reader = csv.DictReader(file, delimiter='\t')
    for row in tsv_reader:
        seekerList.append(row) 
with open('input/inspired/survey_data/rec_demographic.tsv', encoding='utf-8') as file:
    tsv_reader = csv.DictReader(file, delimiter='\t')
    for row in tsv_reader:
        recList.append(row) 


# map movie title into (id -> title (year))
def mapMovieToIdandTitle(movieDict):
    id = 1
    for movie in movies:
        name = movie['title'].strip()
        year = movie['year'].strip()
        title = name + " (" + year + ")"
        if title not in movieDict: 
            movieDict[title] = id
            id += 1

def mapUserToUserIdsAndDialogueToIdsAndUserToDialogue(con,seekerList,recList, userDict, dialougeDict,dialToUserRole): 
    id= 1 
    diagId= 1    
    for conversation in con:
        # print(conversation)
        # Extract the initiator and respondent worker IDs
        rec_tem_id = conversation['rec_survey_id']
        seeker_tem_id = conversation['seek_survey_id']
        
        index = [i for i, sublist in enumerate(seekerList) if sublist['seeker_survey_id'] == seeker_tem_id]
        indexRec = [i for i, sublist in enumerate(recList) if sublist['rec_survey_id'] == rec_tem_id]
        
        if len(index) == 0 or  len(indexRec) == 0:
            continue
        seeker_id=seekerList[index[0]]['seeker_id']
        rec_id=recList[indexRec[0]]['recommender_id']
        if seeker_id not in userDict: 
            userDict[seeker_id] = id
            id += 1
        if rec_id not in userDict: 
            userDict[rec_id] = id
            id += 1    
        if conversation['dialog_id'] not in dialougeDict:
            dialougeDict[conversation['dialog_id']] = diagId
            dialToUserRole[conversation['dialog_id']] = {"conversationId": diagId,"initiatorWorkerId": userDict[seeker_id], "respondentWorkerId": userDict[rec_id]}
            diagId += 1


def replacePlaceHolderWithCorrectLabel(conversations,valid_conversations,test_conversations,dialToUserRole,dialogueMessageDict, movieDict, dialogueMovies):
     # Regular expression to find the patterns
    pattern = r'\[(\w+)_([A-Z]+)(?:_(\d+))?\]'
    prevDialogId = 0
    prevSpeaker = None
    initiatorQuestions = []
    respondentQuestions = []
    for conversation in conversations + valid_conversations + test_conversations:
        #inilized all users set
        dialog_id = conversation['dialog_id']
        if dialog_id != prevDialogId:
            prevDialogId = dialog_id
            prevSpeaker = None
            initiatorQuestions = []
            respondentQuestions = []

        if dialog_id not in dialToUserRole:
            continue
        sentence = conversation['text_with_placeholder']
        utt_id = conversation['utt_id']
        speaker = conversation['speaker']
        if speaker == 'RECOMMENDER':
            workerId= dialToUserRole[dialog_id]["respondentWorkerId"]
        else:
            workerId= dialToUserRole[dialog_id]["initiatorWorkerId"]
        mentionedItems= []        
        words = sentence.split()
        # Loop through each word and replace if pattern matches
        for i, word in enumerate(words):
            match = re.match(pattern, word)
            if match:
                full_match, type_part, order_part = match.groups()
                if type_part == 'TITLE':
                    movies = ast.literal_eval(conversation['movie_dict'])
                    title = next((movie for movie, order in movies.items() if int(order) == int(order_part)), None)
                    if title == None:
                        title = conversation['movies'].split(';')[0]
                    if title not in movieDict:   
                        movieDict[title] = len(movieDict)      
                    words[i] = '@' + str(movieDict[title]) 
                    if dialog_id in dialogueMovies:
                        dialogueMovies[dialog_id].add(words[i])
                    else :
                        dialogueMovies[dialog_id] = {words[i]}

                if type_part == 'GENRE':
                    genres = ast.literal_eval(conversation['genre_dict'])
                    genre = next((genre for genre, order in genres.items() if int(order) == int(order_part)), None)
                    if genre == None:
                        genre = conversation['genres'].split(';')[0]
                    words[i] = genre
                if type_part == 'DIRECTOR':
                    people = ast.literal_eval(conversation['director_dict'])
                    person = next((person for person, order in people.items() if int(order) == int(order_part)), None)
                    if person == None:
                        person = conversation['people_names'].split(';')[0]
                    words[i] = person 
                if type_part == 'ACTOR':
                    people = ast.literal_eval(conversation['actor_dict'])
                    person = next((person for person, order in people.items() if int(order) == int(order_part)), None)
                    if person == None:
                        person = conversation['people_names'].split(';')[0]
                    words[i] = person  
                if type_part == 'PLOT':
                    words[i] = conversation['text']            


        # Join the words back into a sentence
        new_text = ' '.join(words) 
        # perfrom semantic anaylsis for all mentioned items
        # if len(mentionedItems) > 0:
        #     if len(mentionedItems) > 1:
        #         for item in mentionedItems:
        #             start_index = new_text.find(item)
        #             end_index = start_index + len(item)
        #             context = new_text[max(0, start_index-30):min(len(new_text), end_index+30)]
        #             result = sentiment_pipeline(context)
        #             print(f"item: {item}, Context: {context}, Sentiment: {result}")
        #     else:
        #         result = sentiment_pipeline(new_text)
        #         print(f" Sentiment: {result}")


        # tonized_text = word_tokenize(new_text)  
        message = { "role": speaker, "messageId": utt_id, "text" : new_text, "senderWorkerId":workerId}
        if dialog_id in dialogueMessageDict:
            if prevSpeaker == speaker:
                dialogueMessageDict[dialog_id][-1]["text"] = dialogueMessageDict[dialog_id][-1]["text"] + ' ' +new_text
            else:
                dialogueMessageDict[dialog_id].append(message)

        else:
            dialogueMessageDict[dialog_id] = [message]

        prevSpeaker = speaker    
   

    

    

    
# def tokenizeMessagesInDialogue(conversations,valid_conversations,test_conversations,dialogueMessageDict):
#     for conversation in conversations + valid_conversations + test_conversations:
#         #inilized all users set
#         dialog_id = conversation['dialog_id']
#         sentence = conversation['text_with_placeholder']
#         words = word_tokenize(sentence)
#         if dialog_id in dialogueMessageDict:
#             dialogueMessageDict[dialog_id].append(words)
#         else:
#             dialogueMessageDict[dialog_id] = [words]    
     
# def createDialogueData(conversations,valid_conversations,test_conversations,movieDict, dialogue):


mapMovieToIdandTitle(movieDict)
mapUserToUserIdsAndDialogueToIdsAndUserToDialogue(con,seekerList,recList, userDict, dialougeDict,dialToUserRole)
replacePlaceHolderWithCorrectLabel(conversations,valid_conversations,test_conversations,dialToUserRole,dialogueMessageDict, movieDict, dialogueMovies)
# tokenizeMessagesInDialogue(conversations,valid_conversations,test_conversations,dialogueMessageDict)
dialogueMoviesNew = {}
for key, value in dialogueMovies.items():
    dialogueMoviesNew[key] = list(value)

dialogueMessageDict = [{key : value} for key, value in dialogueMessageDict.items()]


with open('MovieTracker/conversations.json', 'w') as json_file:
    json.dump(dialogueMessageDict, json_file, indent=4)

with open('MovieTracker/mentionedMovies.json', 'w') as json_file:
    json.dump(dialogueMoviesNew, json_file, indent=4)    


         

# # get all users and movies
# user_social = {}
# unique_movies = set()
# unique_users = set()

# for conversation in conversations + valid_conversations + test_conversations:
#     #inilized all users set
#     conv_id = conversation[0]
#     unique_users.add(convMapToUserIdsAndRoles[conv_id]["recommender"])
#     unique_users.add(convMapToUserIdsAndRoles[conv_id]["seeker"])
#     #ininlized all movies set 
#     if  isinstance(conversation[6], dict):
#         unique_movies.update(movie[0] for cov_id, movie in conversation[6].items())


# # Create a mapping for movies and users to indices
# movie_to_index = {movie_id: index for index, movie_id in enumerate(unique_movies)}
# user_to_index = {user_id: index for index, user_id in enumerate(unique_users)}


# # Initialize a sparse matrix with dimensions (number of users, number of movies)
# interaction_matrix = dok_matrix((len(unique_users), len(unique_movies)), dtype=int)

# # Initialize a sparse matrix with dimensions (number of users, number of movies)
# interaction_matrix_test = dok_matrix((len(unique_users), len(unique_movies)), dtype=int)


# # Loop through each conversation in the list
# for conversation in total[0]:
#     # print(conversation)
#     # Extract the user  IDs
#     user_id = conversation['user_id']
    
#     # Create sets for the initiator and respondent liked movies
#     movie_liked = set()
    
#     # Initialize a dictionary to track the first interaction for each user
#     first_interaction = {}
    
#     if isinstance(conversation['mentionMovies'], dict):
#          for mesge_id, details in conversation['mentionMovies'].items():
#             movie_liked.add(movie_to_index[details[0]])  # Add to the set of liked movies
#             if user_id not in first_interaction:
#                 # If it is, add to the interaction_matrix
#                 interaction_matrix[user_to_index[user_id], movie_to_index[details[0]]] = 1
#                 # Mark as first interaction processed
#                 first_interaction[user_id] = True
#             else:
#                 # If not the first interaction, decide randomly  
#                 if random.random() < 0.9:    
#                     interaction_matrix[user_to_index[user_id], movie_to_index[details[0]]] = 1
#                 else:
#                     interaction_matrix_test[user_to_index[user_id], movie_to_index[details[0]]] = 1

   

                   

    







# # Create a list of tuples containing (original ID, remapped ID) for users
# user_id_mapping = [(org_id, remap_id) for org_id, remap_id in user_to_index.items()]

# # Create a list of tuples containing (original ID, remapped ID) for movies
# movie_id_mapping = [(org_id, remap_id) for org_id, remap_id in movie_to_index.items()]

# Write the user ID mapping to a text file
# with open('outputNGCFGT/user_list.txt', 'w') as user_file:
#     user_file.write("org_id remap_id\n")
#     for org_id, remap_id in user_id_mapping:
#         user_file.write(f"{org_id} {remap_id}\n")

# # Write the movie ID mapping to a text file
# with open('outputNGCFGT/item_list.txt', 'w') as movie_file:
#     movie_file.write("org_id remap_id\n")
#     for org_id, remap_id in movie_id_mapping:
#         movie_file.write(f"{org_id} {remap_id}\n")

# print("User and movie ID mappings have been saved to user_id_mapping.txt and movie_id_mapping.txt respectively.")


# # Create a dictionary to hold positive interactions for each user
# positive_interactions = {}

# # Iterate over each user
# for user_id, user_idx in user_to_index.items():
#     # Get the indices of the positive interactions for this user
#     positive_movie_indices = interaction_matrix[user_idx].nonzero()[1]
#     # Get the original movie IDs for the positive interactions
#     positive_movie_indices = [movie_idx for movie_idx in positive_movie_indices if interaction_matrix[user_idx, movie_idx] == 1]
#     # Store the positive interactions for this user only if they have non-empty positive_movie_indices
#     if positive_movie_indices:
#         positive_interactions[user_id] = positive_movie_indices

# # Write the positive interactions to the text file
# with open('outputNGCFGT/train.txt', 'w') as file:
#     for user_id, positive_movie_ids in positive_interactions.items():
#         file.write(f"{user_id} {' '.join(map(str, positive_movie_ids))}\n")


# positive_interactions_test = {}

# # Iterate over each user
# for user_id, user_idx in user_to_index.items():
#     # Get the indices of the positive interactions for this user
#     positive_movie_indices = interaction_matrix_test[user_idx].nonzero()[1]
#     # Get the original movie IDs for the positive interactions
#     positive_movie_indices = [movie_idx for movie_idx in positive_movie_indices if interaction_matrix_test[user_idx, movie_idx] == 1]
#     # Store the positive interactions for this user only if they have non-empty positive_movie_indices
#     if positive_movie_indices:
#         positive_interactions_test[user_id] = positive_movie_indices



# # Write the positive interactions to the text file
# with open('outputNGCFGT/test.txt', 'w') as file:
#     for user_id, positive_movie_ids in positive_interactions_test.items():
#         file.write(f"{user_id} {' '.join(map(str, positive_movie_ids))}\n")        

# print("Positive interactions test have been saved to positive_interactions.txt.")



# negative_interactions = {}

# # Iterate over each user
# for user_id, user_idx in user_to_index.items():
#     # Get the indices of the negative interactions for this user in the training matrix
#     negative_movie_indices_train = [movie_idx for movie_idx in range(interaction_matrix.shape[1])
#                                     if interaction_matrix[user_idx, movie_idx] == 0]
    
#     # Get the indices of the negative interactions for this user in the test matrix
#     negative_movie_indices_test = [movie_idx for movie_idx in range(interaction_matrix_test.shape[1])
#                                    if interaction_matrix_test[user_idx, movie_idx] == 0]
    
#     # Combine the negative interactions from both matrices
#     combined_negative_indices = list(set(negative_movie_indices_train + negative_movie_indices_test))
    
#     # Store the combined negative interactions for this user
#     if combined_negative_indices:
#         negative_interactions[user_id] = combined_negative_indices

# # Write the combined negative interactions to the text file
# with open('outputNGCFGT/train_negative.txt', 'w') as file:
#     for user_id, negative_movie_ids in negative_interactions.items():
#         file.write(f"{user_id} {' '.join(map(str, negative_movie_ids))}\n")

# print("Combined negative interactions have been saved to negative_interactions.txt.")