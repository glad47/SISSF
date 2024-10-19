

from flask import Flask, request, jsonify
import json 


app = Flask(__name__)
with open('MovieTracker/conversations.json', 'r', encoding='utf-8') as file:
    conversations = json.load(file)

with open('MovieTracker/mentionedMovies.json', 'r', encoding='utf-8') as file:
    movies = json.load(file)

with open('MovieTracker/track.json', 'r', encoding='utf-8') as file:
    track = json.load(file)  
idx = len(track)






@app.route('/get_new_conv', methods=['GET'])
def get_conversations():
    global idx
    if idx == len(conversations) :
       toSend = {"conv_id":"Complete" ,"conv": "Complete", "mov" :  "", "current" : "Complete", "total":"Complete"}
    else: 
        conv = conversations[idx]
        key = list(conv.keys())[0]
        mov = movies.get(key, dict())
        toSend = {"conv_id":key ,"conv": conv[key], "mov" :  mov, "current" : idx, "total": len(conversations)}
    return jsonify(toSend)

@app.route('/submit', methods=['POST'])
def submit_form():
    global idx
    data = request.json
    
    # Append data to the appropriate arrays
    track.append(data)
    
    # Write the updated data back to the JSON file
    with open('MovieTracker/track.json', 'w') as file:
        json.dump(track, file, indent=4)

    idx += 1    
    
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, port=5002)