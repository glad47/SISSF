import React from 'react';
import './ConversationList.css';

const ConversationList = ({ conversation }) => {
    const highlightText = (text) => {
        const parts = text.split(/(@\w+)/g);
        return parts.map((part, index) =>
            part.startsWith('@') ? <span key={index} className="highlight">{part}</span> : part
        );
    };

    return (
        <div className="conversation-container">
            <h1>Conversations : {conversation['current'] != null ? `${conversation['current']} / ${conversation['total']}`   : null }</h1>
            <h2>conv_id: {conversation['conv_id'] != null ? conversation['conv_id'] : null }</h2>
            <div className="conversation">
                {conversation["conv"].map((message, index) => (
                    <p key={index}>
                        <strong>{message.role}:</strong> {highlightText(message.text)}
                    </p>
                ))}
            </div>
        </div>
    );
};

export default ConversationList;
