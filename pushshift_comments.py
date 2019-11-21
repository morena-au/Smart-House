import csv
import datetime
import json
import re
import sys

import pandas as pd
import requests


def relevantComments(query_term, size=sys.argv[1]):
    '''
    Input:
     > query term
     > size

    Output:
     > csv file: author, body, created_utc, id, link_id, parent_id, permalink, score, subreddit
    '''

    # API link: https://github.com/pushshift/api
    url_comment = 'https://api.pushshift.io/reddit/search/comment/?q=' + \
        str(query_term)+'&subreddit=homeautomation,smarthome&size=' + \
        str(size)

    try:
        r = requests.get(url_comment)
        data = json.loads(r.text)

        # List of information we are interested to collect
        author = []
        body = []
        created_utc = []
        id = []
        link_id = []
        parent_id = []
        permalink = []
        score = []
        subreddit = []

        # loop through all the comments and collect the info
        for i in data['data']:
            author.append(i['author'])
            body.append(i['body'])
            created_utc.append(i['created_utc'])
            id.append(i['id'])
            link_id.append(i['link_id'])
            parent_id.append(i['parent_id'])
            permalink.append(i['permalink'])
            score.append(i['score'])
            subreddit.append(i['subreddit'])

        data = {'author': author,
                'body': body,
                'created_utc': created_utc,
                'id': id,
                'link_id': link_id,
                'parent_id': parent_id,
                'permalink': permalink,
                'score': score,
                'subreddit': subreddit}

        print('Call Succeed')

        return pd.DataFrame(data)

    except:
        print('Call Failed')


# when run directly by python create df
if __name__ == '__main__':

    query_term = ['privacy', 'security', 'trust']
    data = []

    for i in query_term:
        data.append(relevantComments(i, sys.argv[1]))

    # create unique dataframe
    df = pd.concat(data, ignore_index=True)
    print('Tot. rows (input * 3): ', df.shape[0])

    # Eliminate duplicate
    df = df.drop_duplicates()
    print('Tot. rows without duplicates: ', df.shape[0])

    # Write down csv file
    df.to_csv('./comments.csv', index=False)
    print('DONE!')
