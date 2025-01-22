import React, { useState, useEffect } from 'react';
import './MovieForm.css';

const MovieForm = ({ onRemove, onDataChange, conversation, index, role }) => {
    const [formData, setFormData] = useState({
        movieId: '',
        suggested: '',
        seen: '',
        liked: '',
    });

    useEffect(() => {
        onDataChange(formData);
    }, [formData]);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({
            ...formData,
            [name.split('-')[0]]: value,
        });
    };

    const handleRemove = () => {
        onRemove();
    }; 

    return (
        <div className="movie-form">
            <div className="close-button" onClick={handleRemove}>X</div>
            <h2>{role} Form</h2>
            <label>
                Select Movie ID:
                <select name="movieId" value={formData.movieId} onChange={handleChange}>
                    <option value="">Select a movie</option>
                    {conversation['mov'].length > 0 ? conversation['mov'].map((mov, idx) => (
                        <option key={idx} value={mov.replace('@', '')}>{mov}</option>
                    )) : null}
                </select>
            </label>
            <div className="radio-group">
                <label>Suggested:</label>
                <label>
                    <input
                        type="radio"
                        name={`suggested-${index}`}
                        value="0"
                        checked={formData.suggested === '0'}
                        onChange={handleChange}
                    />
                    Mentioned by Seeker
                </label>
                <label>
                    <input
                        type="radio"
                        name={`suggested-${index}`}
                        value="1"
                        checked={formData.suggested === '1'}
                        onChange={handleChange}
                    />
                    Suggested by Recommender
                </label>
            </div>
            <div className="radio-group">
                <label>Seen:</label>
                <label>
                    <input
                        type="radio"
                        name={`seen-${index}`}
                        value="0"
                        checked={formData.seen === '0'}
                        onChange={handleChange}
                    />
                    Not Seen
                </label>
                <label>
                    <input
                        type="radio"
                        name={`seen-${index}`}
                        value="1"
                        checked={formData.seen === '1'}
                        onChange={handleChange}
                    />
                    Seen
                </label>
                <label>
                    <input
                        type="radio"
                        name={`seen-${index}`}
                        value="2"
                        checked={formData.seen === '2'}
                        onChange={handleChange}
                    />
                    Did Not Say
                </label>
            </div>
            <div className="radio-group">
                <label>Liked:</label>
                <label>
                    <input
                        type="radio"
                        name={`liked-${index}`}
                        value="0"
                        checked={formData.liked === '0'}
                        onChange={handleChange}
                    />
                    Did Not Like
                </label>
                <label>
                    <input
                        type="radio"
                        name={`liked-${index}`}
                        value="1"
                        checked={formData.liked === '1'}
                        onChange={handleChange}
                    />
                    Liked
                </label>
                <label>
                    <input
                        type="radio"
                        name={`liked-${index}`}
                        value="2"
                        checked={formData.liked === '2'}
                        onChange={handleChange}
                    />
                    Did Not Say
                </label>
            </div>
        </div>
    );
};

export default MovieForm;