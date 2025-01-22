import React, { useState, useEffect } from 'react';
import MovieForm from './MovieForm';
import './ParentForm.css';

const ParentForm = ({ conversation, onSubmit, reset, onReset }) => {
    const [recommenderForms, setRecommenderForms] = useState([]);
    const [seekerForms, setSeekerForms] = useState([]);
    const [formData, setFormData] = useState([]);
    const [formData2, setFormData2] = useState([]);
    const [invalidForms, setInvalidForms] = useState([]);

    useEffect(() => {
        if(reset == true){
            setRecommenderForms([])
            setSeekerForms([])
            setFormData([])
            setFormData2([])
            setInvalidForms([])
            onReset()
        }


    }, [reset])

    const handleDataChange = (index, data, role) => {
        if (role === 'RECOMMENDER') {
            const newFormData = [...formData];
            newFormData[index] = { ...data, role };
            setFormData(newFormData);
        } else if (role === 'SEEKER') {
            const newFormData = [...formData2];
        newFormData[index] = { ...data, role };
        setFormData2(newFormData);
        }
        
    };
    const handleDataRemove = (index, role) => {
        if (role === 'RECOMMENDER') {
            const newFormData = [...formData];
            const newRecommenderForms = [...recommenderForms];
            newFormData.splice(index, 1); 
            newRecommenderForms.splice(index, 1);
            setFormData(newFormData);
            setRecommenderForms(newRecommenderForms);
        } else if (role === 'SEEKER') {
            const newFormData = [...formData2];
            const newSeekerForms = [...seekerForms];
            newFormData.splice(index, 1); 
            newSeekerForms.splice(index, 1);
            setFormData2(newFormData);
            setSeekerForms(newSeekerForms);
            
        }
        // const newInvalidForms = [...invalidForms];
        // newInvalidForms.splice(index, 1);
        // setInvalidForms(newInvalidForms)
    };

    const handleAddForm = (role) => {
        if (role === 'RECOMMENDER') {
            setRecommenderForms([...recommenderForms, `formrecommender${recommenderForms.length}`]);
        } else if (role === 'SEEKER') {
            setSeekerForms([...seekerForms, `formseeker${seekerForms.length}`]);
        }
    };

    const validateForms = () => {
        const invalidIndices = formData.map((data, index) => {
            return !data.movieId || !data.suggested || !data.seen || !data.liked ? index : null;
        }).filter(index => index !== null);
        
        const invalidIndices2 = formData2.map((data, index) => {
            return !data.movieId || !data.suggested || !data.seen || !data.liked ? index : null;
        }).filter(index => index !== null);
        setInvalidForms(invalidIndices + invalidIndices2);
        return invalidIndices.length === 0;
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (validateForms()) {
            console.log('Collected Data Recommender:', formData);
            console.log('Collected Data Seeker:', formData2);
            var initiatorQuestions = {}
            var respondentQuestions= {}

            formData.map((data, index) => {
                respondentQuestions[data.movieId] = {"suggested": parseInt(data.suggested, 10), "seen": parseInt(data.seen, 10),  "liked": parseInt(data.liked, 10)}
            })

            formData2.map((data, index) => {
                initiatorQuestions[data.movieId] = {"suggested": parseInt(data.suggested, 10), "seen": parseInt(data.seen, 10),  "liked": parseInt(data.liked, 10)}
            })
            var toSend = {"dialog_id": conversation['conv_id'],"respondentQuestions" : respondentQuestions,"initiatorQuestions":initiatorQuestions }
            onSubmit(toSend)
            // Handle the collected data submission here
        } else {
            console.log('Validation failed');
        }
    };

    return (
        <div className="cont">
            <h1>Entry Form</h1>
            <form onSubmit={handleSubmit} className="parent-form">
            
            <div className="form-section">
                <h2>Recommender</h2>
                {recommenderForms.map((form, index) => (
                    <div key={form} className={invalidForms.includes(index) ? 'invalid-form' : ''}>
                        <MovieForm
                            index={form}
                            role="RECOMMENDER"
                            conversation={conversation}
                            onDataChange={(data) => handleDataChange(index, data, 'RECOMMENDER')}
                            onRemove= {() => {handleDataRemove(index, 'RECOMMENDER')}}
                        />
                    </div>
                ))}
                <button type="button" onClick={() => handleAddForm('RECOMMENDER')} className="add-button">+ Add Recommender Form</button>
            </div>
            <div className="form-section">
                <h2>Seeker</h2>
                {seekerForms.map((form, index) => (
                    <div key={form} className={invalidForms.includes(index) ? 'invalid-form' : ''}>
                        <MovieForm
                            index={form}
                            role="SEEKER"
                            conversation={conversation}
                            onDataChange={(data) => handleDataChange(index, data, 'SEEKER')}
                            onRemove= {() => {handleDataRemove(index, 'SEEKER')}}
                        />
                    </div>
                ))}
                <button type="button" onClick={() => handleAddForm('SEEKER')} className="add-button">+ Add Seeker Form</button>
                
            </div>

            <div className="submit-section">
                <button type="submit">Submit</button>
            </div>
           
            
        </form>
        </div>
        
    );
};

export default ParentForm;