import csv
from flask import Flask, render_template, request
import difflib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors



def Knn_recommendation(title, cosine_sim,df2):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    model = NearestNeighbors(metric='cosine',algorithm='brute')
    # Knnearest Neighbors
    Title = df2['title'].iloc[movie_indices]
    Date = df2['release_date'].iloc[movie_indices]
    voteAverage = df2['vote_average'].iloc[movie_indices]
    moviesOverview=df2['overview'].iloc[movie_indices]
    moviesKeyowords=df2['keywords'].iloc[movie_indices]
    moviesId=df2['id'].iloc[movie_indices]
    
    Results = pd.DataFrame(columns=['Title','Year'])
    Results['Title'] = Title
    Results['Year'] = Date
    Results['Ratings'] = voteAverage
    Results['Overview']=moviesOverview
    Results['Types']=moviesKeyowords
    Results['ID']=moviesId
    return Results



data = pd.read_csv('./Dataset/tmdb.csv')
countVector = CountVectorizer(stop_words='english')
countMatrix = countVector.fit_transform(data['soup'])

cosineSimilarity= cosine_similarity(countMatrix, countMatrix)

data = data.reset_index()
indices = pd.Series(data.index, index=data['title'])
Titles = [data['title'][i] for i in range(len(data['title']))]

def Recommendation(title):
    cosineSimilarity = cosine_similarity(countMatrix, countMatrix)
    idx = indices[title]
    similarityScores = list(enumerate(cosineSimilarity[idx]))
    similarityScores = sorted(similarityScores, key=lambda x: x[1], reverse=True)
    similarityScores = similarityScores[1:11]
    movieIndices = [i[0] for i in similarityScores]
    Title = data['title'].iloc[movieIndices]
    Date = data['release_date'].iloc[movieIndices]
    voteAverage = data['vote_average'].iloc[movieIndices]
    moviesOverview=data['overview'].iloc[movieIndices]
    moviesKeyowords=data['keywords'].iloc[movieIndices]
    moviesId=data['id'].iloc[movieIndices]


    Results = pd.DataFrame(columns=['Title','Year'])
    Results['Title'] = Title
    Results['Year'] = Date
    Results['Ratings'] = voteAverage
    Results['Overview']=moviesOverview
    Results['Types']=moviesKeyowords
    Results['ID']=moviesId
    return Results

def Suggestion():
    data = pd.read_csv('./Dataset/tmdb.csv')
    return list(data['title'].str.capitalize())

app = Flask(__name__)
@app.route("/")
@app.route("/home")
def home():
    NewMovies=[]
    with open('recommendedMovie.csv','r') as csvfile:
        readCSV = csv.reader(csvfile)
        NewMovies.append(random.choice(list(readCSV)))
    m_name = NewMovies[0][0]
    m_name = m_name.title()
    
    with open('recommendedMovie.csv', 'a',newline='') as csv_file:
        fieldnames = ['Movie']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({'Movie': m_name})
        result_final = Recommendation(m_name)
        names = []
        dates = []
        ratings = []
        overview=[]
        types=[]
        mid=[]
        for i in range(len(result_final)):
            names.append(result_final.iloc[i][0])
            dates.append(result_final.iloc[i][1])
            ratings.append(result_final.iloc[i][2])
            overview.append(result_final.iloc[i][3])
            types.append(result_final.iloc[i][4])
            mid.append(result_final.iloc[i][5])
    suggestions = Suggestion()
    return render_template("home.html",suggestions=suggestions,movie_type=types[5:],movieid=mid,movie_overview=overview,movie_names=names,movie_date=dates,movie_ratings=ratings,search_name=m_name)

@app.route("/KnearestNeighbors")
def Knn():
    NewMovies=[]
    with open('recommendedMovie.csv','r') as csvfile:
        readCSV = csv.reader(csvfile)
        NewMovies.append(random.choice(list(readCSV)))
    m_name = NewMovies[0][0]
    m_name = m_name.title()
    
    with open('recommendedMovie.csv', 'a',newline='') as csv_file:
        fieldnames = ['Movie']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({'Movie': m_name})
        result_final = Recommendation(m_name)
        names = []
        dates = []
        ratings = []
        overview=[]
        types=[]
        mid=[]
        for i in range(len(result_final)):
            names.append(result_final.iloc[i][0])
            dates.append(result_final.iloc[i][1])
            ratings.append(result_final.iloc[i][2])
            overview.append(result_final.iloc[i][3])
            types.append(result_final.iloc[i][4])
            mid.append(result_final.iloc[i][5])
    suggestions = Suggestion()
    return render_template("knninputs.html",suggestions=suggestions,movie_type=types[5:],movieid=mid,movie_overview=overview,movie_names=names,movie_date=dates,movie_ratings=ratings,search_name=m_name)



# Set up the main route
@app.route('/KnnRecommendation', methods=['GET', 'POST'])

def KnnRecommendation():
    if request.method == 'GET':
        return(render_template('knnInputs.html'))

    if request.method == 'POST':
        m_name = request.form['movie_name']
        m_name = m_name.title()
        if m_name not in Titles:
            return(render_template('negative.html',name=m_name))
        else:
            with open('recommendedMovie.csv', 'a',newline='') as csv_file:
                fieldnames = ['Movie']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'Movie': m_name})
            result_final = Recommendation(m_name)
            names = []
            dates = []
            ratings = []
            overview=[]
            types=[]
            mid=[]
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                dates.append(result_final.iloc[i][1])
                ratings.append(result_final.iloc[i][2])
                overview.append(result_final.iloc[i][3])
                types.append(result_final.iloc[i][4])
                mid.append(result_final.iloc[i][5])

    return render_template('knnresults.html',movie_type=types[5:],movieid=mid,movie_overview=overview,movie_names=names,movie_date=dates,movie_ratings=ratings,search_name=m_name)

@app.route('/positive', methods=['GET', 'POST'])

def recommendationPage():
    if request.method == 'GET':
        return(render_template('home.html'))

    if request.method == 'POST':
        m_name = request.form['movie_name']
        m_name = m_name.title()
        if m_name not in Titles:
            return(render_template('negative.html',name=m_name))
        else:
            with open('recommendedMovie.csv', 'a',newline='') as csv_file:
                fieldnames = ['Movie']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'Movie': m_name})
            result_final = Recommendation(m_name)
            names = []
            dates = []
            ratings = []
            overview=[]
            types=[]
            mid=[]
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                dates.append(result_final.iloc[i][1])
                ratings.append(result_final.iloc[i][2])
                overview.append(result_final.iloc[i][3])
                types.append(result_final.iloc[i][4])
                mid.append(result_final.iloc[i][5])

    return render_template('positive.html',movie_type=types[5:],movieid=mid,movie_overview=overview,movie_names=names,movie_date=dates,movie_ratings=ratings,search_name=m_name)


if __name__ == '__main__':
    app.run()
