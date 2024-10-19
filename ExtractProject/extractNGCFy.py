## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14
## Description: procee Dataset used by SISSF(Social Information Sensitive Conversational Recommendation Using Multi-View Contrastive Learning)


import json
import random
from scipy.sparse import dok_matrix
from scipy.sparse import save_npz
random.seed(42)
# Load the JSON file containing the conversations
conversations = []
test_conversations = []
with open('input/redial/train_data.jsonl', 'r') as file:
    for line in file:
        # Check if the line is not empty
        if line.strip():
            conversations.append(json.loads(line))
with open('input/redial/test_data.jsonl', 'r') as file:
    for line in file:
        # Check if the line is not empty
        if line.strip():
            test_conversations.append(json.loads(line))            

# get all users and movies
user_social = {}
unique_movies = set()
unique_users = set()
for conversation in conversations + test_conversations:
    #inilized all users set
    initiator_id = conversation['initiatorWorkerId']
    respondent_id = conversation['respondentWorkerId']
    unique_users.add(initiator_id)
    unique_users.add(respondent_id)
    # if isinstance(conversation.get('initiatorQuestions', {}), dict):
    #     for movie_id, details in conversation['initiatorQuestions'].items():
    #         if details.get('liked'):
                
    #             break
    # if isinstance(conversation.get('respondentQuestions', {}), dict):
    #     for movie_id, details in conversation['respondentQuestions'].items():
    #         if details.get('liked'):
                
    #             break
    #ininlized all movies set 
    if isinstance(conversation.get('movieMentions', {}), dict):
        unique_movies.update(movie_id for movie_id, name in conversation['movieMentions'].items())


# Create a mapping for movies and users to indices
movie_to_index = {movie_id: index for index, movie_id in enumerate(unique_movies)}
user_to_index = {user_id: index for index, user_id in enumerate(unique_users)}


# Initialize a sparse matrix with dimensions (number of users, number of movies)
interaction_matrix = dok_matrix((len(unique_users), len(unique_movies)), dtype=int)

# Initialize a sparse matrix with dimensions (number of users, number of movies)
interaction_matrix_test = dok_matrix((len(unique_users), len(unique_movies)), dtype=int)


# Loop through each conversation in the list
for conversation in conversations + test_conversations:
    # print(conversation)
    # Extract the initiator and respondent worker IDs
    initiator_id = conversation['initiatorWorkerId']
    respondent_id = conversation['respondentWorkerId']
    
    # Create sets for the initiator and respondent liked movies
    initiator_liked = set()
    respondent_liked = set()
    initiator_disliked = set()
    respondent_disliked = set()
    # Initialize a dictionary to track the first interaction for each user
    first_interaction = {}
    
    if isinstance(conversation['initiatorQuestions'], dict):
         for movie_id, details in conversation['initiatorQuestions'].items():
            if details['liked']:  # Check if the movie is liked
                initiator_liked.add(movie_to_index[movie_id])  # Add to the set of liked movies
            if initiator_id not in first_interaction:
                # If it is, add to the interaction_matrix
                interaction_matrix[user_to_index[initiator_id], movie_to_index[movie_id]] = 1 if details.get('liked') else -1
                # Mark as first interaction processed
                first_interaction[initiator_id] = True
            else:
                # If not the first interaction, decide randomly  
                if random.random() < 0.9:    
                    interaction_matrix[user_to_index[initiator_id], movie_to_index[movie_id]] = 1 if details.get('liked') else -1
                else:
                    interaction_matrix_test[user_to_index[initiator_id], movie_to_index[movie_id]] = 1 if details.get('liked') else -1
    if isinstance(conversation['respondentQuestions'], dict):
        for movie_id, details in conversation['respondentQuestions'].items():
            if details['liked']:  # Check if the movie is liked
                respondent_liked.add(movie_to_index[movie_id])  # Add to the set of liked movies
            if respondent_id not in first_interaction:
                # If it is, add to the interaction_matrix
                interaction_matrix[user_to_index[respondent_id], movie_to_index[movie_id]] = 1 if details.get('liked') else -1  
                # Mark as first interaction processed
                first_interaction[respondent_id] = True
            else:    
                if random.random() < 0.9:     
                    interaction_matrix[user_to_index[respondent_id], movie_to_index[movie_id]] = 1 if details.get('liked') else -1  
                else:
                    interaction_matrix_test[user_to_index[respondent_id], movie_to_index[movie_id]] = 1 if details.get('liked') else -1      

                   

    







# Create a list of tuples containing (original ID, remapped ID) for users
user_id_mapping = [(org_id, remap_id) for org_id, remap_id in user_to_index.items()]

# Create a list of tuples containing (original ID, remapped ID) for movies
movie_id_mapping = [(org_id, remap_id) for org_id, remap_id in movie_to_index.items()]

# Write the user ID mapping to a text file
with open('outputNGCF/user_list.txt', 'w') as user_file:
    user_file.write("org_id remap_id\n")
    for org_id, remap_id in user_id_mapping:
        user_file.write(f"{org_id} {remap_id}\n")

# Write the movie ID mapping to a text file
with open('outputNGCF/item_list.txt', 'w') as movie_file:
    movie_file.write("org_id remap_id\n")
    for org_id, remap_id in movie_id_mapping:
        movie_file.write(f"{org_id} {remap_id}\n")

print("User and movie ID mappings have been saved to user_id_mapping.txt and movie_id_mapping.txt respectively.")


# Create a dictionary to hold positive interactions for each user
positive_interactions = {}

# Iterate over each user
for user_id, user_idx in user_to_index.items():
    # Get the indices of the positive interactions for this user
    positive_movie_indices = interaction_matrix[user_idx].nonzero()[1]
    # Get the original movie IDs for the positive interactions
    positive_movie_indices = [movie_idx for movie_idx in positive_movie_indices if interaction_matrix[user_idx, movie_idx] == 1]
    # Store the positive interactions for this user only if they have non-empty positive_movie_indices
    if positive_movie_indices:
        positive_interactions[user_id] = positive_movie_indices

# Write the positive interactions to the text file
with open('outputNGCF/train.txt', 'w') as file:
    for user_id, positive_movie_ids in positive_interactions.items():
        file.write(f"{user_id} {' '.join(map(str, positive_movie_ids))}\n")


positive_interactions_test = {}

# Iterate over each user
for user_id, user_idx in user_to_index.items():
    # Get the indices of the positive interactions for this user
    positive_movie_indices = interaction_matrix_test[user_idx].nonzero()[1]
    # Get the original movie IDs for the positive interactions
    positive_movie_indices = [movie_idx for movie_idx in positive_movie_indices if interaction_matrix_test[user_idx, movie_idx] == 1]
    # Store the positive interactions for this user only if they have non-empty positive_movie_indices
    if positive_movie_indices:
        positive_interactions_test[user_id] = positive_movie_indices



# Write the positive interactions to the text file
with open('outputNGCF/test.txt', 'w') as file:
    for user_id, positive_movie_ids in positive_interactions_test.items():
        file.write(f"{user_id} {' '.join(map(str, positive_movie_ids))}\n")        

print("Positive interactions test have been saved to positive_interactions.txt.")



negative_interactions = {}

# Iterate over each user
for user_id, user_idx in user_to_index.items():
    # Get the indices of the negative interactions for this user in the training matrix
    negative_movie_indices_train = [movie_idx for movie_idx in range(interaction_matrix.shape[1])
                                    if interaction_matrix[user_idx, movie_idx] == -1]
    
    # Get the indices of the negative interactions for this user in the test matrix
    negative_movie_indices_test = [movie_idx for movie_idx in range(interaction_matrix_test.shape[1])
                                   if interaction_matrix_test[user_idx, movie_idx] == -1]
    
    # Combine the negative interactions from both matrices
    combined_negative_indices = list(set(negative_movie_indices_train + negative_movie_indices_test))
    
    # Store the combined negative interactions for this user
    if combined_negative_indices:
        negative_interactions[user_id] = combined_negative_indices

# Write the combined negative interactions to the text file
with open('outputNGCF/train_negative.txt', 'w') as file:
    for user_id, negative_movie_ids in negative_interactions.items():
        file.write(f"{user_id} {' '.join(map(str, negative_movie_ids))}\n")

print("Combined negative interactions have been saved to negative_interactions.txt.")