#!/usr/bin/env python
# coding: utf-8

# In[199]:


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import glob
import nltk

import itertools
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import chart_studio.plotly as py
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

import gensim.models.word2vec as w2v
from sklearn.manifold import TSNE
import plotly.express as px

from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from fastcluster import linkage
from matplotlib.colors import rgb2hex, colorConverter


# In[12]:


all_files=glob.glob("/Users/rishabhsrivastava/Downloads/CSV'S/DIKSHA APP REVIEWS AND RATINGS/"+"/*.csv")
dflist = []


# In[13]:


for filename in all_files:
    # Dataframe of one file
    df_sm = pd.read_csv(filename, index_col=None, header=0)
    dflist.append(df_sm)
    
df = pd.concat(dflist, axis=0, ignore_index=True)


# In[14]:


df.dropna(subset=["Review Text"],inplace=True)


# In[19]:


df


# In[30]:


eng_data = df.loc[df['Reviewer Language']=='en']
eng_df = pd.DataFrame(eng_data)
# eng_df


# In[31]:


eng_df.reset_index(inplace = True) 


# In[10]:


# cluster1 = eng_df.loc[eng_df['Star Rating'] <= 3]
# cluster1


# In[11]:


# cluster2 = eng_df.loc[eng_df['Star Rating'] > 3]
# cluster2


# In[32]:


eng_df 


# In[33]:


def bar_graph(df):
    count = eng_df['Star Rating'].value_counts()
    print("rate count")
    print(  count)
    count.plot.bar()


# In[34]:


bar_graph(eng_df)


# In[35]:


from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import re

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
       # The yield statement suspends function’s execution and sends a value back to the caller.
        yield subtree.leaves()

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
                    and word.lower() not in stopwords)
    return accepted

def get_terms(tree):

    for leaf in leaves(tree):
        term = [ w for w,t in leaf if acceptable_word(w) ]
        yield term


# In[36]:


grammar =r"""
  NP: {<DT|JJ|NN.*>+}          
  PP: {<IN><NP>}            
  VP: {<VB.*><NP|PP|CLAUSE>+$}
  CLAUSE: {<NP><VP>}          
  """
      
    


# In[37]:


def phrase_extraction(text, grammar):
    text = text.lower()
    sentence_re = r'''(?x)          
      (?:[A-Z]\.)+        
    | \w+(?:-\w+)*        
    | \$?\d+(?:\.\d+)?%?  
    | \.\.\.              
    | [][.,;"'?():_`-]    
    '''
    
    ls = [] 
    word_token_ls = text.split(" ")

    toks = nltk.regexp_tokenize(text, sentence_re)
    postoks = nltk.tag.pos_tag(toks)
    
    chunker = nltk.RegexpParser(grammar)
    
    tree = chunker.parse(postoks)
    terms = get_terms(tree)
    for term in terms:
        ls.append(" ".join(term)) 
    return list(set(ls))


# In[38]:


ls= list(eng_df["Review Text"])
print(ls)


# In[41]:


#Creating a new Dtaframe which is required for final csv
new_df = pd.DataFrame(columns=['Review Text', 'Keywords'])

for i in ls:
    if(i == ''):
        new_df = new_df.append({'Review Text': i, 'Keywords': '[]'}, ignore_index=True)
    else:
        x = phrase_extraction(i, grammar)
        new_df = new_df.append({'Review Text': i, 'Keywords': x}, ignore_index=True)

new_df
    
    


# In[42]:


new_df.insert(2,'Star Rating',eng_df['Star Rating'])


# In[99]:


def unlisting_list_of_list(list_of_list):
    merged = list(itertools.chain(*list_of_list))
    kywrd_ls= list(set(merged))
    return(kywrd_ls)


# In[100]:


kywrd_df_ls = list(new_df["Keywords"])
kywrd_df_ls


# In[102]:


# merged = list(itertools.chain(*kywrd_df_ls))
# kywrd_ls= list(set(merged))
# kywrd_ls
kywrd_ls=unlisting_list_of_list(kywrd_df_ls)
print(kywrd_ls)


# In[46]:


review_df_ls = list(new_df["Review Text"])
review_df_ls


# In[63]:


def sentiment_score_count(review_df_ls):
    sentiment_class = []
    pos_score = []
    neu_score = []
    neg_score = []
    for i in review_df_ls:
        sid_obj = SentimentIntensityAnalyzer()  
        sentiment_dict = sid_obj.polarity_scores(i)  
        #print("Overall sentiment dictionary is : ", sentiment_dict) 
        neg_score.append(sentiment_dict['neg']) 
        neu_score.append(sentiment_dict['neu']) 
        pos_score.append(sentiment_dict['pos'])
        if sentiment_dict['compound'] >= 0.05 : 
            sentiment_class.append("Positive") 

        elif sentiment_dict['compound'] <= - 0.05 : 
            sentiment_class.append("Negative") 

        else : 
            sentiment_class.append("Neutral")
            
    sentiment_df = pd.DataFrame(list(zip(neg_score, neu_score, pos_score, sentiment_class )), 
               columns =['neg_score','neu_score', 'pos_score', 'sentiment_class']) 
    return sentiment_df 


# In[68]:


sentiment_df = sentiment_score_count(review_df_ls)
sentiment_df


# In[69]:


# joining two dataframes
joined_df = new_df.join(sentiment_df) 
joined_df


# In[70]:


selected_col = eng_df[['Review Title','Review Submit Date and Time', 'Review Submit Millis Since Epoch', 'Review Last Update Date and Time', 'Review Last Update Millis Since Epoch']].copy()
selected_col


# In[71]:


final_df = joined_df.join(selected_col) 
final_df


# In[72]:


def polarity_and_subjectivity(review_df_ls):
    # List of sentiments of each review text
    sentiments = []

    for i in review_df_ls:
        blob = TextBlob(i)
        s = blob.sentiment
        sentiments.append(s)
    
    # Creating a dataframe of sentiments
    sentiment = pd.DataFrame(sentiments)
    return sentiment


# In[73]:


sentiment = polarity_and_subjectivity(review_df_ls)
sentiment


# In[74]:


final_df2 = final_df.join(sentiment) 
final_df2


# In[83]:


reviewdf = final_df["Keywords"]
reviewdf


# In[84]:


def wordcloud_generator(reviewdf):
    comment_words = ' '
    stopwords = set(STOPWORDS) 
    for val in reviewdf: 
        val = str(val) 

        # split the value 
        tokens = val.split() 

        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 

        for words in tokens: 
            comment_words = comment_words + words + ' '

    wordcloud = WordCloud(width = 800, height = 800, 
    background_color = 'white', 
    stopwords = stopwords, 
    min_font_size = 10).generate(comment_words) 

    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.show()
    


# In[85]:


wordcloud_generator(reviewdf)


# In[ ]:





# In[89]:


def sentiment_graph_for_each_rating(df):
    g = sns.FacetGrid(df, col = "sentiment_class")
    g.map(plt.hist, "Star Rating")
    plt.show()


# In[90]:


sentiment_graph_for_each_rating(final_df2)


# In[154]:


def vectorization_of_list(keyword_list):
    #word embedding(vectorization)
    embed = hub.Module("/Users/rishabhsrivastava/Downloads/3 2/")
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(keyword_list))
#         print(message_embeddings)
        lst = []
        for i in message_embeddings:
            df = pd.DataFrame([i])
            lst.append(df)
    frame = pd.concat(lst)
    return frame


# In[155]:


frame = vectorization_of_list(kywrd_ls)
frame


# In[156]:


keywordsdf = pd.DataFrame(kywrd_ls,columns=['keywords'])
keywordsdf


# In[172]:


frame.set_index(keywordsdf["keywords"], inplace = True) 
frame


# In[173]:


def TSNE_3D(df):
    get_ipython().run_line_magic('pylab', 'inline')

    #Reduce Dimensinality
    X_embedded = TSNE(n_components=3).fit_transform(df)
    vec_df = pd.DataFrame(X_embedded, columns=["ft1","ft2","ft3"])
    #vec_df
    #plot 3-D graph
    fig = px.scatter_3d(vec_df,x="ft1",y="ft2",z="ft3")
    fig.show()


# In[174]:


TSNE_3D(frame)


# In[178]:


frame.insert(512,"keywords",frame.index)
frame.reset_index(drop=True, inplace=True)


# In[179]:


frame["keywords"]


# In[188]:


def principal_component_analysis_3D(df):
    sns.set_style("white")
    my_dpi=96
    plt.figure(figsize=(480/my_dpi, 480/my_dpi), dpi=my_dpi)
    df["keywords"]=pd.Categorical(df["keywords"])
    my_color=df["keywords"].cat.codes
    df = df.drop("keywords", 1)
    
    # Run The PCA
    pca = PCA(n_components=3)
    pca.fit(df)
    
    result = pd.DataFrame(pca.transform(df), columns=['PCA%i' % i for i in range(3)], index=df.index)
#     return result
    
    # Plot initialisation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], c=my_color, cmap="Set2_r", s=60)
 

    


# In[189]:


principal_component_analysis_3D(frame)


# In[195]:


def dendrogram_genetator(df):
    plt.figure(figsize=(10, 7))  
    plt.title("Dendrograms")  
    dend = shc.dendrogram(shc.linkage(df, method='ward'))
    


# In[196]:


dendrogram_genetator(vec_df)


# In[197]:


def dendrogram_genetator_with_thresold(df,thresold):
    plt.figure(figsize=(10, 7))
#     y=800
    plt.title("Dendrograms")  
    dend = shc.dendrogram(shc.linkage(df, method='ward'))
    plt.axhline(y, color='r', linestyle='--')
    


# In[198]:


dendrogram_genetator_with_thresold(vec_df,800)


# In[200]:


def hierarchial_clustering(df):
    cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(df)
    
    plt.figure(figsize=(10, 7))  
    plt.scatter(df['ft1'], df['ft2'], c=cluster.labels_) 
    


# In[201]:


hierarchial_clustering(vec_df)


# In[202]:


#result.reset_index(drop=True, inplace=True)
vec_df.set_index(keywordsdf["keywords"], inplace = True) 
vec_df


# In[228]:


# sns.set_palette('Set1', 10, 0.65)
# palette = (sns.color_palette())
# #set_link_color_palette(map(rgb2hex, palette))
# sns.set_style('white')


# In[229]:


# np.random.seed(25)


# In[230]:


# link = linkage(vec_df, metric='correlation', method='ward')
# link


# In[231]:


# figsize(8, 3)
# den = dendrogram(link, labels=vec_df.index)
# plt.xticks(rotation=90)
# no_spine = {'left': True, 'bottom': True, 'right': True, 'top': True}
# sns.despine(**no_spine);

# plt.tight_layout()
# plt.savefig('feb.png');


# In[232]:


# den


# In[233]:


# cluster_idxs = defaultdict(list)
# for c, pi in zip(den['color_list'], den['icoord']):
#     for leg in pi[1:3]:
#         i = (leg - 5.0) / 10.0
#         if abs(i - int(i)) < 1e-5:
#             cluster_idxs[c].append(int(i))


# In[234]:


# cluster_idxs


# In[235]:


# class Clusters(dict):
#     def _repr_html_(self):
#         html = '<table style="border: 0;">'
#         for c in self:
#             hx = rgb2hex(colorConverter.to_rgb(c))
#             html += '<tr style="border: 0;">' \
#             '<td style="background-color: {0}; ' \
#                        'border: 0;">' \
#             '<code style="background-color: {0};">'.format(hx)
#             html += c + '</code></td>'
#             html += '<td style="border: 0"><code>' 
#             html += repr(self[c]) + '</code>'
#             html += '</td></tr>'
        
#         html += '</table>'
        
#         return html


# In[236]:


# cluster_classes = Clusters()
# for c, l in cluster_idxs.items():
#     i_l = [den['ivl'][i] for i in l]
#     cluster_classes[c] = i_l


# In[237]:


# cluster_classes


# In[226]:


def cluster_element_extraction(vec_df):
    sns.set_palette('Set1', 10, 0.65)
    palette = (sns.color_palette())
    #set_link_color_palette(map(rgb2hex, palette))
    sns.set_style('white')
    
    np.random.seed(25)
    
    link = linkage(vec_df, metric='correlation', method='ward')

    figsize(8, 3)
    den = dendrogram(link, labels=vec_df.index)
    plt.xticks(rotation=90)
    no_spine = {'left': True, 'bottom': True, 'right': True, 'top': True}
    sns.despine(**no_spine);

    plt.tight_layout()
    plt.savefig('feb2.png');
    
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
                
    class Clusters(dict):
        def _repr_html_(self):
            html = '<table style="border: 0;">'
            for c in self:
                hx = rgb2hex(colorConverter.to_rgb(c))
                html += '<tr style="border: 0;">'                 '<td style="background-color: {0}; '                            'border: 0;">'                 '<code style="background-color: {0};">'.format(hx)
                html += c + '</code></td>'
                html += '<td style="border: 0"><code>' 
                html += repr(self[c]) + '</code>'
                html += '</td></tr>'

            html += '</table>'

            return html
    
    cluster_classes = Clusters()
    for c, l in cluster_idxs.items():
        i_l = [den['ivl'][i] for i in l]
        cluster_classes[c] = i_l
        
    return cluster_classes
    


# In[227]:


cluster_element_extraction(vec_df)


# In[213]:


vec_df


# In[215]:


new_df


# In[238]:



listoflist = new_df['Keywords'].tolist()


# In[239]:


def list_of_lists_to_list(list_of_list):
    keyword_list = []
    kywrd_freq_list = []
    for i in list_of_list:
        kywrd_freq_list.append(len(i))
        for j in i:
            keyword_list.append(j)

    return keyword_list, kywrd_freq_list


# In[240]:


keyword_list, kywrd_freq_list = list_of_lists_to_list(listoflist)
print(keyword_list, kywrd_freq_list)


# In[241]:


list_df = pd.DataFrame(keyword_list,columns =['Keywords']) 
list_df


# In[242]:


rating_list = new_df['Star Rating'].tolist()
print(len(rating_list))


# In[243]:


kywrd_rating_list =[]
for i in range(len(rating_list)):
    for j in range(len(kywrd_freq_list)):
        if(i==j):
            kywrd_rating_list.extend(repeat(rating_list[i],kywrd_freq_list[j]))
#print(kywrd_rating_list)
        


# In[244]:


rate_df = pd.DataFrame(kywrd_rating_list,columns =['Star Rating']) 
rate_df


# In[245]:


key_rate_df = list_df.join(rate_df)
key_rate_df


# In[266]:


def clustering_rating_wise(key_rate_df, rating):
    key_data = key_rate_df.loc[key_rate_df['Star Rating'] == rating]
    key_rate_df1 = pd.DataFrame(key_data)
#     return key_rate_df1
    ky_ls =key_rate_df1['Keywords'].tolist()
    
    frame = vectorization_of_list(ky_ls)
    frame.set_index(key_rate_df1["Keywords"], inplace = True) 
    return cluster_element_extraction(frame)


# In[267]:


clustering_rating_wise(key_rate_df, 5)


# In[223]:


key_rate_df["Keywords"].value_counts()[:50]


# In[255]:


wordcloud_generator(key_rate_df1["Keywords"])


# In[ ]:




