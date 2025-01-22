/*
 * @Author: your name
 * @Date: 2024-09-14 13:07:25
 * @LastEditTime: 2024-09-19 17:05:14
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \movie-tracker\src\App.js
 */
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ConversationList from './ConversationList';
import ParentForm from './ParentForm';

function App() {
  const [conversation, setConversation] = useState(null);
  const [reset, setReset] = useState(false);

  useEffect(() => {
      axios.get('http://127.0.0.1:5002/get_new_conv')
          .then(response => setConversation(response.data))
          .catch(error => console.error('Error fetching data:', error));
  }, []);

  const handleSubmit = (formData) => {
      axios.post('http://127.0.0.1:5002/submit', formData)
          .then(response => {
            if( response.data.status == "success"){
              axios.get('http://127.0.0.1:5002/get_new_conv')
              .then(response => {
                if (response.data["conv_id"] == "Complete"){
                setConversation(true)
                }else{
                  setConversation(response.data)
                  setReset(true)
                }
                
              })
            
              .catch(error => console.error('Error fetching data:', error));
            }
          })
          .catch(error => console.error('Error submitting form:', error));
  };

  const resetReset = () => {
    setReset(false)
  }

  return (
      <div className="App">
          {conversation == true ? (<div className="mainCont">"Complete"</div>) : (<div className="App">
            {conversation != null ? (

<div style={{display: 'flex' }}>
    <ConversationList conversation={conversation} />
    <ParentForm reset={reset} onReset={resetReset} onSubmit={handleSubmit} conversation={conversation} />
</div>
) : (
<div className="mainCont">"Waiting"</div>
)}
          </div>) }
          
      </div>
  );
}
export default App;