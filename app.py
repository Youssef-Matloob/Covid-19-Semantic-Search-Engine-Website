from runServer import app
from flask import render_template, request
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
import numpy as np 
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import scispacy
import spacy
import en_core_sci_lg
from scipy.spatial.distance import cosine
import joblib
from IPython.display import HTML, display
from ipywidgets import interact, Layout, HBox, VBox, Box
import ipywidgets as widgets
from IPython.display import clear_output
from tqdm import tqdm
from os.path import isfile
import seaborn as sb
import matplotlib.pyplot as plt
from joblib import dump , load

nlp = en_core_sci_lg.load(disable=["tagger", "parser", "ner"])
nlp.max_length = 2000000



def spacy_tokenizer(sentence):
    return [word.lemma_ for word in nlp(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1)]

def print_top_words(model, vectorizer, n_top_words):
    feature_names = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        message = "\nTopic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        #print(message)
    #print()

def get_k_nearest_docs(doc_dist, k=5, lower=1950, upper=2020, only_covid19=False, get_dist=False):
    '''
    doc_dist: topic distribution (sums to 1) of one article
    
    Returns the index of the k nearest articles (as by Jensen–Shannon divergence in topic space). 
    '''
    
    relevant_time = df_covid.publish_year.between(lower, upper)
    #relevant_time = '2019'
    if only_covid19:
        temp = doc_topic_dist[relevant_time & is_covid19_article]
        
    else:
        temp = doc_topic_dist[relevant_time]
         
    distances = temp.apply(lambda x: cosine(x, doc_dist), axis=1)
    k_nearest = distances[distances != 0].nsmallest(n=k).index
    
    if get_dist:
        k_distances = distances[distances != 0].nsmallest(n=k)
        return k_nearest, k_distances
    else:
        return k_nearest

def plot_article_dna(paper_id, width=20):
    t = df_covid[df_covid.paper_id == paper_id].title.values[0]
    doc_topic_dist[df_covid.paper_id == paper_id].T.plot(kind='bar', legend=None, title=t, figsize=(width, 4))
    plt.xlabel('Topic')

def compare_dnas(paper_id, recommendation_id, width=20):
    t = df_covid[df_covid.paper_id == recommendation_id].title.values[0]
    temp = doc_topic_dist[df_covid.paper_id == paper_id]
    ymax = temp.max(axis=1).values[0]*1.25
    temp = pd.concat([temp, doc_topic_dist[df_covid.paper_id == recommendation_id]])
    temp.T.plot(kind='bar', title=t, figsize=(width, 4), ylim= [0, ymax])
    plt.xlabel('Topic')
    plt.legend(['Selection', 'Recommendation'])

# compare_dnas('90b5ecf991032f3918ad43b252e17d1171b4ea63', 'a137eb51461b4a4ed3980aa5b9cb2f2c1cf0292a')

def dna_tabs(paper_ids):
    k = len(paper_ids)
    outs = [widgets.Output() for i in range(k)]

    tab = widgets.Tab(children = outs)
    tab_titles = ['Paper ' + str(i+1) for i in range(k)]
    for i, t in enumerate(tab_titles):
        tab.set_title(i, t)
    display(tab)

    for i, t in enumerate(tab_titles):
        with outs[i]:
            ax = plot_article_dna(paper_ids[i])
            plt.show(ax)

def compare_tabs(paper_id, recommendation_ids):
    k = len(recommendation_ids)
    outs = [widgets.Output() for i in range(k)]

    tab = widgets.Tab(children = outs)
    tab_titles = ['Paper ' + str(i+1) for i in range(k)]
    for i, t in enumerate(tab_titles):
        tab.set_title(i, t)
    display(tab)

    for i, t in enumerate(tab_titles):
        with outs[i]:
            ax = compare_dnas(paper_id, recommendation_ids[i])
            plt.show(ax)

def recommendation(paper_id, k=5, lower=2000, upper=2020, only_covid19=False, plot_dna=False):
    '''
    Returns the title of the k papers that are closest (topic-wise) to the paper given by paper_id.
    '''
    
    print(df_covid.title[df_covid.paper_id == paper_id].values[0])

    recommended, dist = get_k_nearest_docs(doc_topic_dist[df_covid.paper_id == paper_id].iloc[0], k, lower, upper, only_covid19, get_dist=True)
    recommended = df_covid.iloc[recommended].copy()
    recommended['similarity'] = 1 - dist 
    
    
    #h = '/n'.join([ n + '/n' +' (Similarity: ' + "{:.2f}".format(s) + ')' for  n, s in recommended[['title', 'similarity']].values])
    #display(HTML(h))
    print(recommended[['title', 'similarity']].values)
    # for  n, s in recommended[['title', 'similarity']].values:
    #   print(n)
    #   print(s)

    if plot_dna:
        compare_tabs(paper_id, recommended.paper_id.values)


def relevant_articles(tasks, k=3, lower=2000, upper=2020, only_covid19=False):
    tasks = [tasks] if type(tasks) is str else tasks 
    
    tasks_vectorized = vectorizer.transform(tasks)
    tasks_topic_dist = pd.DataFrame(lda.transform(tasks_vectorized))

    for index, bullet in enumerate(tasks):
        print(bullet)
        recommended ,dist= get_k_nearest_docs(tasks_topic_dist.iloc[index], k, lower, upper, only_covid19, get_dist=True)
        recommended = df_covid.iloc[recommended]

        #recommended = df_covid.iloc[recommended].copy()
        recommended['similarity'] = 1 - dist 
        
        print(recommended)
        #h = '<br/>'.join(['<a href="' + l + '" target="_blank">'+ n + '</a>' for l, n in recommended[['url','title']].values])
        #display(HTML(h))
        return recommended



df_covid = load('df_covid.csv')
vectorizer = load('vectorizer.csv')
data_vectorized = load('data_vectorized.csv')
lda = load('lda.csv')
doc_topic_dist = pd.read_csv('doc_topic_dist.csv')
is_covid19_article = df_covid.body_text.str.contains('COVID-19|SARS-CoV-2|2019-nCov|SARS Coronavirus 2|2019 Novel Coronavirus')



@app.route('/')
def hello_world(name = None):
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
        searchText = [str(x) for x in request.form.values()][0]
        print(searchText)
        #
        # Model code
        #
        #

        results = relevant_articles(searchText)
        title = []
        sim = []
        content = []
        url = []
        for i,row in results.iterrows():
            abstract = str(row['abstract'])
            abstract = abstract.replace("<br>", " ")
            content.append(abstract)
            sim.append(row['similarity'])
            title_ = str(row['title'])
            #print(title_)
            title_ = title_.replace("<br>", " ")
            #print(title_)
            title.append(title_)
            url.append(row['url'])
        #title = ["Pikachu", "Charizard", "Squirtle"]
        #content = ["Pikachu", "Charizard", "Squirtle"]
        #sim = [10, 40 , 50]
        #url = ["Pikachu", "Charizard", "Squirtle"]
        return render_template("predict.html", len = len(title), title = title, content = content, sim = sim, url = url, searchText = searchText)

if __name__ == "__main__":
    app.run(host= '127.0.0.1', debug=True)