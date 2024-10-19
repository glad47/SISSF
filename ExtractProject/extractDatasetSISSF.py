
## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14
## Description: procee Dataset used by SISSF(Social Information Sensitive Conversational Recommendation Using Multi-View Contrastive Learning)



import json
from scipy.sparse import dok_matrix
from scipy.sparse import save_npz
import pandas as pd
import numpy as np
import random
import pickle
import contractions
import re
import sys
import io
import math
from sklearn.model_selection import train_test_split

contraction_mapping = {
    "I'm": "I am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "I've": "I have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "I'll": "I will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "we'll": "we will",
    "they'll": "they will",
    "I'd": "I would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "it'd": "it would",
    "we'd": "we would",
    "they'd": "they would",
    "that's": "that is",
    "who's": "who is",
    "what's": "what is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",
    "let's": "let us",
    "ma'am": "madam",
    "o'clock": "of the clock",
    "ain't": "is not",
    "won't": "will not",
    "can't": "cannot",
    "shan't": "shall not",
    "don't": "do not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mightn't": "might not",
    "mustn't": "must not",
    "would've": "would have",
    "should've": "should have",
    "could've": "could have",
    "might've": "might have",
    "must've": "must have"
}

def expand_contractions(text, contraction_mapping):
    # Use a regular expression to replace the contractions
    for contraction, expansion in contraction_mapping.items():
        # Escape the contraction to safely use it in a regular expression
        pattern = re.escape(contraction)
        # Use a lambda function to perform the replacement with the IGNORECASE flag
        text = re.sub(pattern, lambda m: expansion, text, flags=re.IGNORECASE)
    return text



# Set the default encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# Load the JSON file containing the conversations
conversations = []
test_conversations = []
new_conversations = []
movie_entiy_mapping = {}
modifies_conversation = []
val_conv = []
test_conv = []
train_conv = []
all_conv = {}
token_mapping = {}

entity_mapping = {}

with open('input/redial/train_data.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        # Check if the line is not empty
        if line.strip():
            conversations.append(json.loads(line))
with open('input/redial/test_data.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        # Check if the line is not empty
        if line.strip():
            test_conversations.append(json.loads(line))   



with open('input/redial/token2id.json', 'r', encoding='utf-8') as file:
    token_mapping = json.load(file)

with open('input/redial/train_data_c2crs.json', 'r', encoding='utf-8') as file:
    train_conv = json.load(file)

with open('input/redial/test_data_c2crs.json', 'r', encoding='utf-8') as file:
    test_conv = json.load(file)

with open('input/redial/valid_data_c2crs.json', 'r', encoding='utf-8') as file:
    val_conv = json.load(file)  

with open('input/redial/entity2id.json', 'r', encoding='utf-8') as file:
    entity_mapping = json.load(file)      

with open('input/redial/redial_context_movie_id2crslab_entityId.json', 'r', encoding='utf-8') as file:
    movie_entiy_mapping = json.load(file)   


ratings = pd.read_excel('input/redial/ratings.xlsx')

# Initialize a dictionary to hold user interactions





# print("**************************************") 
# print(len(token_mapping))   



# get all users and movies
user_social = {}
unique_movies = set()
unique_users = set()
allConvsIda= [con['conversationId'] for con in conversations] + [con['conversationId'] for con in test_conversations]
allConv = conversations  + test_conversations
movieLikedDict = {}
newTokensUserIds= set()
userRelations = {}
userRelationsWeights = {}
userTie ={}
users_training= set()
users_test= set()
users_val= set()
users_training_count= {}
users_test_count= {}
users_val_count= {}
newConversations = []
unique_movies = set()
unique_users = set()
token_freq = {}
user_interactions = {}
user_ratings = {}
item_interactions = {}
item_ratings = {}  
all_conv_c2cs = {} 
all_liked_items_byUser = {}
recommenderSeekerTracking = {}
user_to_index = {}
movie_to_index = {}




def createUniqueMapping(unique_movies, unique_users):
    print("createUniqueMapping")
    with open('input/redial/user_list.txt') as f:
            check = False
            for l in f.readlines():
                if check :
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        uid = int(l[0])
                        user_id = int(l[1])
                        user_to_index[uid] = user_id 
                        unique_users.add(user_id)
                else : 
                    check = True         

    with open('input/redial/item_list.txt') as f:
        check = False
        for l in f.readlines():
            if check :
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    iid = int(l[0])
                    item_id = int(l[1])
                    movie_to_index[iid] = item_id
                    unique_movies.add(item_id)
                    
            else : 
                check = True
    return movie_to_index, user_to_index
 

def init_interaction_matrix(unique_users, unique_movies):
    print("init_interaction_matrix")
    # Initialize a sparse matrix with dimensions (number of users, number of movies)
    return dok_matrix((len(unique_users), len(unique_movies)), dtype=int)


def process_original_dataset(user_to_index,movie_to_index):
    print("process_original_dataset")
    for conv in train_conv + test_conv + val_conv : 
        all_conv_c2cs[conv['conv_id']] = conv
    for conversation in test_conversations + conversations:
        # print(conversation)
        # Extract the initiator and respondent worker IDs
        initiator_mentioned_movies = []
        # initiator_rec_movies = []
        respondent_mentioned_movies = []
        respondent_rec_movies = []
        initiator_mentioned_entities = []
        respondent_mentioned_entities = []
        initiator_id = conversation['initiatorWorkerId']
        respondent_id = conversation['respondentWorkerId']
        conversation['initiatorWorkerId'] = user_to_index[initiator_id]
        conversation['respondentWorkerId'] = user_to_index[respondent_id]
        if isinstance(conversation['initiatorQuestions'], dict):
            for movie_id, details in conversation['initiatorQuestions'].items():
                initiator_mentioned_movies.append(movie_to_index[int(movie_id)])
                initiator_mentioned_entities.append(movie_entiy_mapping[movie_id])  
                # if details['suggested'] == 0 :
                #     initiator_rec_movies.append(movie_to_index[movie_id])

        if isinstance(conversation['respondentQuestions'], dict):
            for movie_id, details in conversation['respondentQuestions'].items():
                respondent_mentioned_movies.append(movie_to_index[int(movie_id)])  # Add to the set of liked movies
                respondent_mentioned_entities.append(movie_entiy_mapping[movie_id])
                if details['suggested'] == 1 :
                    respondent_rec_movies.append(movie_to_index[int(movie_id)])
        conversation['initiator_mentioned_items'] = initiator_mentioned_movies
        conversation['initiator_mentioned_entities'] = initiator_mentioned_entities  
        conversation['respondent_mentioned_items'] = respondent_mentioned_movies
        conversation['respondent_mentioned_entities'] = respondent_mentioned_entities
        # conversation['initiator_rec_movies'] = initiator_rec_movies
        conversation['respondent_rec_movies'] = respondent_rec_movies            
        for index, msg in enumerate(conversation['messages']):
            try:
                dialog_ref = all_conv_c2cs[conversation['conversationId']]['dialog']
                msg['workerId'] =  user_to_index[msg['senderWorkerId']]
                msg['text'] = dialog_ref[index]['text']    
                if len(dialog_ref[index]['entity']) > 1:
                    msg['entities'] = []
                    for entity in dialog_ref[index]['entity']:
                        try:
                            msg['entities'].append(entity_mapping[entity])
                            if msg['workerId'] == conversation['initiatorWorkerId'] :
                                if entity_mapping[entity] not in initiator_mentioned_entities :
                                    initiator_mentioned_entities.append(entity_mapping[entity])
                            else :    
                                if entity_mapping[entity] not in respondent_mentioned_entities :    
                                    respondent_mentioned_entities.append(entity_mapping[entity])
                        except KeyError:
                            # Handle the missing key, e.g., by using a default value or logging an error
                            print(f"Key not found: {entity}")
                            # For example, you can append a default value or skip the entity
                            # msg['entities'].append(default_value)  # If you have a default value
                            # or simply pass if you want to ignore missing keys
                            pass
                else:
                    msg['entities'] = []

                msg['items'] = []  
                msg['rec_items'] = [] 
                for word in msg['text']: 
                    if '@' in word and word[1:].isdigit():
                        movie_id = int(word[1:])
                        msg['items'].append(movie_to_index[movie_id]) 
                        msg['entities'].append(movie_entiy_mapping[word[1:]])
                        if msg['workerId'] == conversation['respondentWorkerId'] :
                            if movie_to_index[movie_id] in respondent_rec_movies:
                                msg['rec_items'].append(movie_to_index[movie_id])   
                        
                                  
            except KeyError:          
                print(f"Conversation Key not found: {conversation['conversationId']}")
                msg['workerId'] =  user_to_index[msg['senderWorkerId']]
                msg['text'] = expand_contractions(msg['text'], contraction_mapping)
                msg['text'] = re.findall(r'\w+|@\d+|[^\w\s]', msg['text'])
                
                msg['entities'] = []
                msg['items'] = []   
                msg['rec_items'] = []
                for word in msg['text']: 
                    if '@' in word and word[1:].isdigit():
                        movie_id = int(word[1:])
                        msg['items'].append(movie_to_index[movie_id]) 
                        msg['entities'].append(movie_entiy_mapping[word[1:]])
                        if msg['workerId'] == conversation['respondentWorkerId'] :
                            if movie_to_index[movie_id] in respondent_rec_movies:
                                msg['rec_items'].append(movie_to_index[movie_id])   
            except  IndexError:
                print(f"Conversation Message not found: {index}")
                msg['workerId'] =  user_to_index[msg['senderWorkerId']]
                msg['text'] = expand_contractions(msg['text'], contraction_mapping)
                msg['text'] = re.findall(r'\w+|@\d+|[^\w\s]', msg['text'])
                
                msg['entities'] = []
                msg['items'] = []   
                msg['rec_items'] = []
                for word in msg['text']: 
                    if '@' in word and word[1:].isdigit():
                        movie_id = int(word[1:])
                        msg['items'].append(movie_to_index[movie_id]) 
                        msg['entities'].append(movie_entiy_mapping[word[1:]])
                        if msg['workerId'] == conversation['respondentWorkerId'] :
                            if movie_to_index[movie_id] in respondent_rec_movies:
                                msg['rec_items'].append(movie_to_index[movie_id])   
              


def statics_data(allConvsIda, allConv,train_conv, users_set, user_count_dict):
    print("statics_data")
    for conversation in train_conv:
        if conversation['conv_id'] in allConvsIda: 
            index = allConvsIda.index(conversation['conv_id'])
            originalConv = allConv[index] 
            initiator_id = originalConv['initiatorWorkerId']
            respondent_id = originalConv['respondentWorkerId']
            users_set.add(initiator_id)
            users_set.add(respondent_id)
            if initiator_id in user_count_dict:
                user_count_dict[initiator_id] += 1
            else:
                user_count_dict[initiator_id] = 1

            if respondent_id in user_count_dict:
                user_count_dict[respondent_id] += 1
            else:
                user_count_dict[respondent_id] = 1    


def create_new_data(allConvsIda, allConv,train_conv, user_count_dict, newConversations):
    print("create_new_data")
    for conversation in train_conv:
        if conversation['conv_id'] in allConvsIda: 
            index = allConvsIda.index(conversation['conv_id'])
            originalConv = allConv[index] 
            initiator_id = originalConv['initiatorWorkerId']
            respondent_id = originalConv['respondentWorkerId']
            if user_count_dict[initiator_id] >= 2 and user_count_dict[respondent_id] >= 2:
                newConversations.append(originalConv)


def split_convs_randomly(lst, percentage):
    print("split_convs_randomly")
    num_items_to_pick = int(len(lst) * percentage / 100)
    picked_items = []
    remaining_items = lst.copy()
    user_count = {}
    count =0
    random.seed(42)
    while len(picked_items) < num_items_to_pick:
        
        item = random.choice(remaining_items)
        initiator_id = item['initiatorWorkerId']
        respondent_id = item['respondentWorkerId']
        if (users_training_count[initiator_id] - 1 ) > user_count.get(initiator_id, 0) and (users_training_count[respondent_id] - 1 ) > user_count.get(respondent_id, 0):
            picked_items.append(item)
            remaining_items.remove(item)
            if initiator_id in user_count:
                user_count[initiator_id] += 1
            else:
                user_count[initiator_id] = 1

            if respondent_id in user_count:
                user_count[respondent_id] += 1
            else:
                user_count[respondent_id] = 1
            count += 1
            print(count)    
    
    return picked_items, remaining_items   


def split_convs_proportionally(lst, percentage):
    print("split_convs_proportionally")
    num_items_to_pick = int(len(lst) * percentage / 100)
    picked_items = []
    remaining_items = lst.copy()
    user_count = {}

    
    
    
    random.seed(42)
    while len(picked_items) < num_items_to_pick:
        item = random.choice(remaining_items)
        initiator_id = item['initiatorWorkerId']
        respondent_id = item['respondentWorkerId']
        if (users_training_count[initiator_id] * percentage / 100)  > user_count.get(initiator_id, 0) and \
           (users_training_count[respondent_id] * percentage / 100)  > user_count.get(respondent_id, 0):
            picked_items.append(item)
            remaining_items.remove(item)
            if initiator_id in user_count:
                user_count[initiator_id] += 1
            else:
                user_count[initiator_id] = 1

            if respondent_id in user_count:
                user_count[respondent_id] += 1
            else:
                user_count[respondent_id] = 1
    return picked_items, remaining_items               
            


def split_convs_randomly_just_split(lst, percentage):
    print("split_convs_randomly_just_split")
    num_items_to_pick = int(len(lst) * percentage / 100)
    picked_items = []
    remaining_items = lst.copy()
    random.seed(42)
    while len(picked_items) < num_items_to_pick:
        
        item = random.choice(remaining_items)
        picked_items.append(item)
        remaining_items.remove(item)
    
    return picked_items, remaining_items      
            

           

def process_data(train_conv,movieLikedDict, userRelations, token_freq, recommenderSeekerTracking):
    print("process_data")
    for conversation in train_conv:
        initiator_id = conversation['initiatorWorkerId']
        respondent_id = conversation['respondentWorkerId']
        # newTokensUserIds.add('%' + str(initiator_id))
        # newTokensUserIds.add('%'+ str(respondent_id))
        # conversation['initiatorWorkerId'] =initiator_id
        # conversation['respondentWorkerId'] =respondent_id
        
        start = 1    
        for index, msg in enumerate(conversation['messages']): 
            for token in msg['text']:
                    try:
                        token_id =token_mapping[token]
                        if token_id in token_freq:
                            # Token exists, increment its frequency
                            token_freq[token_id] += 1
                        else:
                            # Token doesn't exist, add it with a frequency of 1
                            token_freq[token_id] = 1
                    except KeyError:
                        print(f"Token not found: {token}")
                        if '@' in token and token[1:].isdigit():
                            ew_id = len(token_mapping)
                            token_mapping[token] = new_id
                            token_freq[new_id] = 1
                            continue
                        # Generate a random number between 0 and 1
                        
                        if random.random() < 0.5:  # 50% chance
                            new_id = len(token_mapping)
                            token_mapping[token] = new_id
                            token_freq[new_id] = 1
                        else:
                            print("Token not added due to 50% probability function.")
        if isinstance(conversation['initiatorQuestions'], dict):
            for movie_id, details in conversation['initiatorQuestions'].items():
                if details['liked'] == 1 :
                    if initiator_id in movieLikedDict:
                        movieLikedDict[initiator_id].add(movie_id)
                    else:
                        movieLikedDict[initiator_id] = {movie_id}

        if isinstance(conversation['respondentQuestions'], dict):
            for movie_id, details in conversation['respondentQuestions'].items():
                if details['liked'] == 1 :
                    if respondent_id in movieLikedDict:
                        movieLikedDict[respondent_id].add(movie_id)
                    else:
                        movieLikedDict[respondent_id] = {movie_id} 
        if (respondent_id in movieLikedDict) and (initiator_id in movieLikedDict):  
            if len(movieLikedDict[respondent_id].intersection(movieLikedDict[initiator_id])) > 0 :
                if initiator_id in userRelations: 
                    userRelations[initiator_id].add(respondent_id)
                else:
                    userRelations[initiator_id] = {respondent_id} 
                if respondent_id in recommenderSeekerTracking: 
                    recommenderSeekerTracking[respondent_id].add(initiator_id)
                else:
                    recommenderSeekerTracking[respondent_id] = {initiator_id}     
                if respondent_id in userRelations: 
                    userRelations[respondent_id].add(initiator_id)
                else:
                    userRelations[respondent_id] = {initiator_id} 
    userRelations = {key: list(value) for key, value in userRelations.items()}   

def processFriends(movieLikedDict, userRelations, recommenderSeekerTracking):
    print("processFriends")
    for user in userRelations:
        if recommenderSeekerTracking.get(user):
            for seeker1 in recommenderSeekerTracking.get(user):
                for seeker2 in recommenderSeekerTracking.get(user):
                    if seeker1 != seeker2:
                        if len(movieLikedDict[seeker1].intersection(movieLikedDict[seeker2])) > 0 :
                            if seeker1 in userRelations: 
                                userRelations[seeker1].add(seeker2)
                            else:
                                userRelations[seeker1] = {seeker2} 
                            if seeker2 in userRelations: 
                                userRelations[seeker2].add(seeker1)
                               
                            else:
                                userRelations[seeker2] = {seeker1} 
                
def processUserRelationsWeight(movieLikedDict, userRelations):
    print("processUserRelationsWeight")
    for user1 in userRelations:
        if userRelations.get(user1):
            for user2 in userRelations.get(user1):
                likedItemUser1 = movieLikedDict.get(user1, set())
                likedItemUser2 = movieLikedDict.get(user2, set())
                if len(likedItemUser1.intersection(likedItemUser2)) > 0 :
                    # we needind the average ratings  for all common items 
                    commonItems = likedItemUser1.intersection(likedItemUser2)
                    user1Ratings = 0
                    user2Ratings = 0
                    for item in commonItems:
                        user1Ratings += ratings.loc[user1, movie_to_index[int(item)]]
                        user2Ratings += ratings.loc[user2,  movie_to_index[int(item)]]
                    averageRatings = (user1Ratings / len(commonItems)  +  user2Ratings / len(commonItems) ) /2
                    if math.floor(averageRatings) > 5:
                        averageRatings = 5
                    else:
                        averageRatings = math.floor(averageRatings)    
                    if user1 in userRelationsWeights: 
                        userRelationsWeights[user1].append(averageRatings)
                    else:
                        userRelationsWeights[user1] = [averageRatings]
                else :
                    if user1 in userRelationsWeights: 
                        userRelationsWeights[user1].append(0)
                    else:
                        userRelationsWeights[user1] = [0]

              
                     



                     
def process_test_data(test_conv ):
    print("process_test_data")
    for conversation in test_conv:
        initiator_id = conversation['initiatorWorkerId']
        respondent_id = conversation['respondentWorkerId']
       
 
    

def create_interaction_matrix(convs, interaction_matrix, movie_to_index):
    print("create_interaction_matrix")
    for conversation in convs:
        # print(conversation)
        # Extract the initiator and respondent worker IDs
        initiator_id = conversation['initiatorWorkerId']
        respondent_id = conversation['respondentWorkerId']
        
        # Create sets for the initiator and respondent liked movies
        initiator_liked = set()
        respondent_liked = set()
        if isinstance(conversation['initiatorQuestions'], dict):
            for movie_id, details in conversation['initiatorQuestions'].items():
                if details['liked']:  # Check if the movie is liked
                    initiator_liked.add(movie_to_index[int(movie_id)])  # Add to the set of liked movies
                interaction_matrix[initiator_id, movie_to_index[int(movie_id)]] = 1 if details.get('liked') else -1
        if isinstance(conversation['respondentQuestions'], dict):
            for movie_id, details in conversation['respondentQuestions'].items():
                if details['liked']:  # Check if the movie is liked
                    respondent_liked.add(movie_to_index[int(movie_id)])  # Add to the set of liked movies
                interaction_matrix[respondent_id, movie_to_index[int(movie_id)]] = 1 if details.get('liked') else -1   


def create_user_item_ratings(interaction_matrix,user_interactions,user_ratings,item_interactions, item_ratings ):
    print("create_user_item_ratings")
    # Iterate over each user in the interaction matrix
    for user_index in range(interaction_matrix.shape[0]):
        interacted_indices= []
        # Get the indices of items that have been interacted with by the user
        interacted_indices = np.where(interaction_matrix[user_index].toarray() != 0)[1]    
        
        # Add the user and their interacted items to the dictionary
        user_interactions[user_index] = interacted_indices
        inter= []
        for item in interacted_indices:
            inter.append(round(ratings.loc[user_index, item]))


        # Add the user and their interacted items to the dictionary
        user_ratings[user_index] = inter


    interaction_matrix_transposed = interaction_matrix.transpose()
    # Iterate over each user in the interaction matrix
    for item_index in range(interaction_matrix_transposed.shape[0]):
        interacted_indices= []
        # Get the indices of items that have been interacted with by the user
        interacted_indices = np.where(interaction_matrix_transposed[item_index].toarray() != 0)[1]    
        
        updated_indices = [index + len(unique_movies) for index in interacted_indices]
        # Add the user and their interacted items to the dictionary
        item_interactions[item_index] = updated_indices
        inter= []
        for user in interacted_indices:
            inter.append(round(ratings.loc[user, item_index]))
    

        # Add the user and their interacted items to the dictionary
        item_ratings[item_index] = inter




def create_interaction_dataset(interaction_matrix, user_interactions,user_ratings,item_interactions, item_ratings):
    print("create_interaction_dataset")
    user_dataset = []
    item_dataset = []
    ratings_dataset = []

    # Iterate over each user in the interaction matrix
    for user_index in range(interaction_matrix.shape[0]):
        interacted_indices= []
        # Get the indices of items that have been interacted with by the user
        interacted_indices = np.where(interaction_matrix[user_index].toarray() != 0)[1]    
        
        # Add the user and their interacted items to the dictionary
        user_interactions[user_index] = interacted_indices
        inter= []
        for item in interacted_indices:
            user_dataset.append(user_index)
            item_dataset.append(item)
            ratings_dataset.append(round(ratings.loc[user_index, item]))


    random.seed(42)

    # Assuming 'user_dataset', 'item_dataset', and 'ratings_dataset' are your lists

    # Combine the three lists into a list of tuples
    combined_dataset = list(zip(user_dataset, item_dataset, ratings_dataset))

    # Shuffle the combined dataset to randomize the order
    random.shuffle(combined_dataset)

    # Define the size of the test set (e.g., 20% of the total data)
    test_size = int(0.2 * len(combined_dataset))

    # Split the combined dataset into train and test sets
    train_dataset = combined_dataset[test_size:]
    test_dataset = combined_dataset[:test_size]

    # Unzip the train and test datasets back into separate lists
    user_train, item_train, ratings_train = zip(*train_dataset)
    user_test, item_test, ratings_test = zip(*test_dataset)

    # Convert the tuples back to lists
    train_u = list(user_train)
    train_v = list(item_train)
    train_r = list(ratings_train)

    test_u = list(user_test)
    test_v = list(item_test)
    test_r = list(ratings_test)
    ratings_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

    datasets = [
        user_interactions, # history_u_lists
        user_ratings, # history_ur_lists          
        item_interactions, # history_v_lists  
        item_ratings, # history_vr_lists
        train_u, # train_u
        train_v, # train_v
        train_r, # train_r
        test_u, # test_u
        test_v, # test_v
        test_r, # test_r
        userRelations, # social_adj_lists
        userRelationsWeights, # social_adj_ratings
        ratings_map # ratings_list
    ]
    # Save the combined datasets to a .pickle file
    with open('outputSISSF/redial_interactions.pickle', 'wb') as f:
        pickle.dump(datasets, f)
       

movie_to_index, user_to_index = createUniqueMapping(unique_movies, unique_users)
interaction_matrix = init_interaction_matrix(unique_users, unique_movies)
process_original_dataset(user_to_index,movie_to_index)
statics_data(allConvsIda, allConv,train_conv + test_conv + val_conv,users_training, users_training_count) 
create_new_data(allConvsIda, allConv,train_conv + test_conv + val_conv, users_training_count, newConversations) 

newSmallTest, newTrain = split_convs_proportionally(newConversations, 20)
newVal, newTest = split_convs_randomly_just_split(newSmallTest, 50 )
process_data(newTrain,movieLikedDict, userRelations, token_freq, recommenderSeekerTracking)
processFriends(movieLikedDict, userRelations, recommenderSeekerTracking) 
processUserRelationsWeight(movieLikedDict, userRelations) 
process_test_data(newTest)
process_test_data(newVal)
create_interaction_matrix(newTrain, interaction_matrix, movie_to_index)
create_user_item_ratings(interaction_matrix,user_interactions,user_ratings,item_interactions, item_ratings )
create_interaction_dataset(interaction_matrix, user_interactions,user_ratings,item_interactions, item_ratings)









users_training = list(users_training)






with open('outputSISSF/train_data.json', 'w') as json_file:
    json.dump(newTrain, json_file, indent=4)

with open('outputSISSF/valid_data.json', 'w') as json_file:
    json.dump(newTest, json_file, indent=4)    

with open('outputSISSF/test_data.json', 'w') as json_file:
    json.dump(newVal, json_file, indent=4)


with open('outputSISSF/token2id.json', 'w') as json_file:
    json.dump(token_mapping, json_file, indent=4)

with open('outputSISSF/token_freq.json', 'w') as json_file:
    json.dump(token_freq, json_file, indent=4)    





print(len(newTest))
print(len(newTrain))
print(len(newVal))


