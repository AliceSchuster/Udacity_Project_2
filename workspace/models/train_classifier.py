import sys
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib

def load_data(database_filepath):
    """
    Function: load data from database into dataframe
    Args:
        database_filepath: path of the database
    Return:
        X: Disaster messages (features)
        Y: Disaster categories (target)  
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    X = df['message']
    Y = df[df.columns[4:]]
    return X, Y


def tokenize(text):
    """ 
    Function: normalize, tokenize, remove stop words, and lemmatize text string
    Args:
        text: the message
    Return:
        lemmed: cleaned string list
    """
    # Normalize text: remove URLs
    text = re.sub(r"(https?:\/\/|www\.)\S*", "urlplaceholder", text, re.MULTILINE)    
    
    # Normalize text: remove punctuation characters and capitalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())       
    
    # Tokenize words by splitting the message into a sequence of words
    words = word_tokenize(text)
    
    # Remove stop words, such as "a", "and", "the", "of"
    words = [w for w in words if w not in stopwords.words("english")]    
    
    # Lemmatization: Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    # Lemmatize verbs by specifying pos (e.g., studying --> study)
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    return lemmed


def build_model():
    """
    Function: build a machine learning model for classifing the disaster messages
    Args:
        None
    Return:
        cv: Grid Search object
    """
    # Pipeline: Random Forest Classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])   
    parameters =  {
    'clf__estimator__n_estimators': [50, 100], # number of trees
    'clf__estimator__min_samples_split': [5, 10, 20] # minimum number of samples required to split a node
    } 
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
    

def evaluate_model(model, X_test, Y_test):
    """
    Function: calculate and print the precision, recall, and f1-score for all output category    
    Args:
        model: machine learning model
        X_test: disaster messages
        Y_test: disaster categories
    Return:
        None
    """        
    Y_pred = model.predict(X_test)    
    for col_no, column in enumerate(Y_test):
        print(column)
        print(classification_report(Y_test[column], Y_pred[:, col_no]))
   

def save_model(model, model_filepath):
    """
    Function: save model as a pickle file
    Args:
        model: machine learning model
        model_filepath: filepath of machine learning model
    Return:
        None
    """
    #pickle.dump(model, open(model_filepath, 'wb'))
    
    joblib.dump(model, model_filepath + '.gz', compress='gzip')
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()