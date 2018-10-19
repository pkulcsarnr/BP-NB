import webhoseio
import json
import numpy as np
import re
import time

#PROPERTIES
reLoadData = False
articles = list()
types = list()
words = list()
#NLTK´S stopwords + mine
stopWords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours","ers", "yourself", "yourselves", "he","isnt","cant" "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "wasnt", "were", "be", "been", "being", "have", "havent", "has", "had", "having", "do", "does", "doesnt", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

class Regs: 
    specialChars = '' 
    digits = '' 
    singleChars = ''
    multipleWhiteSpaces = ''
    stopWords = list()
regexes = Regs()



#FUNCTION
#MAIN
##################################################################################################
def main():
    print("Starting Naive Bayes by PKS")

    print("Starting downloading data")
    downloadData()

    print("Starting processing the data")
    processData()
##################################################################################################



#FUNCTION
#DOWNLOAD DATA FROM API
###################################################################################################
def downloadData():
    #decide if the data is needed to download again
    #do not load on every run, becouse of request limit
    if reLoadData == True :
        #Config the API connection with the token
        webhoseio.config(token="d0d26a65-a56f-413d-acc8-e54bafbb7377") 
        #Create queryparams for sports data
        query_params_sport = {
            "q": "site_category:sports language:english site_type:news domain_rank:<100",
            "sort": "relevancy"
            }
        #query articles for sport
        output = webhoseio.query("filterWebContent", query_params_sport)
        #save data to file
        with open('sportsdata.json', 'w') as outfile:
            json.dump(output, outfile)
        print("Downloaded data for sports category, and created sportsdata.json file")

        #Create queryparams for business data
        query_params_business = {
            "q": "site_category:business language:english site_type:news domain_rank:<100",
            "sort": "relevancy"
            }
        #query articles for sport
        output = webhoseio.query("filterWebContent", query_params_business)
        #save data to file
        with open('businessdata.json', 'w') as outfile:
            json.dump(output, outfile)
        print("Downloaded data for sports category, and created businessdata.json file")
    else :
        print("There was no need to download data again")
####################################################################################################



#FUNCTION
#LOAD DATA FROM FILES
####################################################################################################
def loadData(category):
    if category == "sports":
        with open('sportsdata.json') as json_data:
            sports = json.load(json_data)
        return(sports)
    elif category == "business":
        with open('politicsdata.json') as json_data:
            business = json.load(json_data)
        return(business)
####################################################################################################


#FUNCTION
#PROCESS DATA
#1. load data from files
#2. lower case the data
#3. remove punctuations
#4. remove stopwords
#5. create vectors of words
#6. remove numbers
####################################################################################################
def processData():
    preCompileRegexes()
    #load data for sports
    sportsData = loadData("sports")
    if len(sportsData["posts"]) > 0 :
        print("Preprocessing of " + str(len(sportsData["posts"])) + " sports articles starting")
        start = time.time()
        for index, article in enumerate(sportsData["posts"], start=0):
            if index < 99 :
                text = article["text"]
                text = preprocessText(text)
                articles.append(text.split(' '))     
                types.append("1")       
        end = time.time()
        print("Preprocessing finished in ",end - start)
    #load data for business
    businessData = loadData("business")
    if len(businessData["posts"]) > 0 :
        print("Preprocessing of " + str(len(businessData["posts"])) + " business articles starting")
        start = time.time()
        for index, article in enumerate(businessData["posts"], start=0):
            text = article["text"]
            text = preprocessText(text)
            articles.append(text.split(' '))     
            types.append("0")         
        end = time.time()
        print("Preprocessing finished in ",end - start)
    #create one array fromm all articles
    words = [item for sublist in articles for item in sublist]
    #remove duplicate values from words list
    words = list(set(words))
    words.remove("")
    words = sorted(words)
    print("There are " + str(len(words)) + " words")

    print("Creating matrix of containing words")
    start = time.time()
    articleWords = np.zeros((len(articles), len(words) + 1))
    for index, article in enumerate(articles, start=0):
        articleWords[index, 0 ] = types[index] 
        for j, word in enumerate(article, start=0):
            if word != '':
                a = words.index(word) + 1
                articleWords[index, words.index(word) + 1] = 1        

    end = time.time()
    print("Creating matrix finished in ",end - start)
    np.savetxt("foo.csv", articleWords, fmt='%1.0f', delimiter=" ")

    X = articleWords
    #calculate the phi values
    Y = np.zeros((2, X.shape[1]))

    #number of rows (number of samples in training set)
    m = X.shape[0]

    #number of words
    n = X.shape[1] - 1

    #number of rows clasified to first / second group
    sumY1 = float(np.sum(X[:,0]))
    sumY0 = float(m - np.sum(X[:,0]))

    #calcualting Phi_{y=0} and Phi_{y=1\
    Y[0,0] = (sumY0 + 1) / float(m + 2)
    Y[1,0] = (sumY1 + 1) / float(m + 2)

    for j in range(1, X.shape[1]):
        #calcualting Phi_{j|y=0} and Phi_{j|y=1\
        Y[0,j] = (np.sum(X[X[:,0]<1,j]) + 1) / (sumY0 + 2)
        Y[1,j] = (np.sum(X[X[:,0]>0,j]) + 1) / (sumY1 + 2)

    #test it on sport [99]
    testtext = sportsData["posts"][99]["text"]
    testtext = preprocessText(testtext)
    testarticle = testtext.split(' ')
    testArticleWords = np.zeros((1, len(words)))
    for k, word in enumerate(testarticle, start=0):
        if word != '' :
            if word in words:
                testArticleWords[0, words.index(word)] = 1

    a = Y[1,0]
    b = Y[0,0]

    #get probability that this vector is classified to first gorup y=0
    for j in range(1, X.shape[1]):
        if(testArticleWords[0,j - 1] == 0):
            a = a * (1 - Y[1,j])
            b = b * (1 - Y[0,j])
        else:
            a = a * Y[1,j]
            b = b * Y[0,j]

    phi0z = b / ( a + b)
    phi1z = a / ( a + b)

    print(phi0z)
    print(phi1z)

def preCompileRegexes():
    regexes.specialChars = re.compile('[^\w\s]')
    regexes.digits = re.compile('\d')
    regexes.singleChars = re.compile('\s.\s')
    regexes.multipleWhiteSpaces = re.compile('[ ]{2,}')
    for sw in stopWords:
        exp = '\\b' + sw + '+\W'
        regexes.stopWords.append(re.compile(exp))


def preprocessText(text):
    text = text.lower()
    #new lines
    text = text.replace('\n', ' ')
    #special characters
    text = re.sub(regexes.specialChars, '', text)
    #digits
    text = re.sub(regexes.digits, '', text)
    #stopwords
    for sw in regexes.stopWords:
        text = re.sub(sw , '', text)
    #single characters (ex donald j trump => donald trump)
    text = re.sub(regexes.singleChars, ' ', text)
    #multiple white spaces
    text = re.sub(regexes.multipleWhiteSpaces, ' ', text)
    return(text)


main()




