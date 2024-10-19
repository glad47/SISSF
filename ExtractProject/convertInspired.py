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
import nltk
from nltk.tokenize import word_tokenize
import ast
# from transformers import pipeline

# Download the necessary resources
nltk.download('punkt')
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
movieTrucking = {}
dataset = []

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

with open('input/inspired/track.json', 'r', encoding='utf-8') as file:
    truckingArray = json.load(file)
    for item in  truckingArray:
        movieTrucking[item['dialog_id']] = {"respondentQuestions" : item['respondentQuestions'], "initiatorQuestions" : item['initiatorQuestions']}   


# map movie title into (id -> title (year))
def mapMovieToIdandTitle(movieDict):
    print("mapMovieToIdandTitle")
    id = 1
    for movie in movies:
        name = movie['title'].strip()
        year = movie['year'].strip()
        title = name + " (" + year + ")"
        if title not in movieDict: 
            movieDict[title] = id
            id += 1

def mapUserToUserIdsAndDialogueToIdsAndUserToDialogue(con,seekerList,recList, userDict, dialougeDict,dialToUserRole): 
    print("mapUserToUserIdsAndDialogueToIdsAndUserToDialogue")
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

def connect_at_with_number(words):
    result = []
    i = 0
    while i < len(words):
        if words[i] == "@" and i + 1 < len(words) and words[i + 1].isdigit():
            result.append("@" + words[i + 1])
            i += 2  # Skip the next word since it's already connected
        else:
            result.append(words[i])
            i += 1
    return result

def tokenizeMessages(conversations,valid_conversations,test_conversations,dialToUserRole,dialogueMessageDict, movieDict, dialogueMovies):
    print("tokenizeMessages")
    # Regular expression to find the patterns
    pattern = r'\[(\w+)_([A-Z]+)(?:_(\d+))?\]'
    prevDialogId = 0
    for conversation in conversations + valid_conversations + test_conversations:
        #inilized all users set
        dialog_id = conversation['dialog_id']
        if dialog_id != prevDialogId:
            prevDialogId = dialog_id

        if dialog_id not in dialToUserRole:
            continue
        sentence = conversation['text_with_placeholder']
        utt_id = conversation['utt_id']
        speaker = conversation['speaker']
        if speaker == 'RECOMMENDER':
            workerId= dialToUserRole[dialog_id]["respondentWorkerId"]
        else:
            workerId= dialToUserRole[dialog_id]["initiatorWorkerId"]    
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
        


        tonized_text = word_tokenize(new_text) 
        tonized_text = connect_at_with_number(tonized_text)  
        message = { "role": speaker, "messageId": utt_id, "text" : tonized_text, "senderWorkerId":workerId}
        if dialog_id in dialogueMessageDict:
            dialogueMessageDict[dialog_id].append(message)
        else:
            dialogueMessageDict[dialog_id] = [message]

     
   

    

    


     
def createDialogueData(conversations,valid_conversations,test_conversations,dialToUserRole,dialogueMessageDict, movieDict, dataset):
    print("createDialogueData")
    reversedMovieDict = {v: k for k, v in movieDict.items()}
    for dialog_id in dialogueMessageDict:
        #inilized all users set
        # dialog_id = conversation['dialog_id']
    

        if dialog_id not in dialToUserRole or dialog_id not in movieTrucking:
            continue
        conversationId = dialToUserRole[dialog_id]["conversationId"]
        respondentWorkerId= dialToUserRole[dialog_id]["respondentWorkerId"]
        
        initiatorWorkerId= dialToUserRole[dialog_id]["initiatorWorkerId"]    
        messages = dialogueMessageDict[dialog_id]
        initiatorQuestions = movieTrucking[dialog_id]['initiatorQuestions']
        respondentQuestions = movieTrucking[dialog_id]['respondentQuestions']
        movieMentions = {}
        for  item in initiatorQuestions:
            movieMentions[item] = reversedMovieDict[int(item)] 
        for  item in respondentQuestions:
            movieMentions[item] = reversedMovieDict[int(item)] 
        item = {"conversationId": conversationId, 'initiatorWorkerId': initiatorWorkerId, \
                         "respondentWorkerId": respondentWorkerId, 'messages' : messages, \
                         "movieMentions" : movieMentions, "initiatorQuestions": initiatorQuestions, \
                         "respondentQuestions" :respondentQuestions }
        dataset.append(item)    






        

mapMovieToIdandTitle(movieDict)
mapUserToUserIdsAndDialogueToIdsAndUserToDialogue(con,seekerList,recList, userDict, dialougeDict,dialToUserRole)
tokenizeMessages(conversations,valid_conversations,test_conversations,dialToUserRole,dialogueMessageDict, movieDict, dialogueMovies)
createDialogueData(conversations,valid_conversations,test_conversations,dialToUserRole,dialogueMessageDict, movieDict, dataset)

with open('input/inspired/conversations.json', 'w') as json_file:
    json.dump(dataset, json_file, indent=4)



         