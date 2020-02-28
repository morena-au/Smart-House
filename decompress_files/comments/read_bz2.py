import bz2
import json
import time
import pandas as pd
import re
import concurrent.futures
import os

# Get files
path = os.getcwd()
files = [i for i in os.listdir(path) if i.startswith('RC')]

t1 = time.perf_counter()

def read_bz2(name):

    # Initiate list
    id = []
    link_id = []
    parent_id = []
    created_utc = []
    body = []
    author = []
    permalink = []
    score = []
    subreddit = []

    name_csv = re.search(r'(\d{4}-\d{2})', str(name)).group(1)

    print('Start reading bz2 file..', name_csv)
    with bz2.open(str(name), 'rb') as file:
        for line in file:
            comment = json.loads(line)

            try:
                if comment['subreddit'] in ['smarthome', 'homeautomation']:
                    id.append(comment['id'])
                    link_id.append(comment['link_id'])
                    parent_id.append(comment['parent_id'])
                    created_utc.append(comment['created_utc'])
                    body.append(comment['body'])
                    author.append(comment['author'])
                    score.append(comment['score'])
                    subreddit.append(comment['subreddit'])
                    permalink.append(comment['permalink'])
            except KeyError as e:
                if repr(e) == "KeyError('permalink')":
                    # append an empty string
                    permalink.append('')
                else:
                    pass
    
    print('\n\nDONE!')
    print('lenght: ', len(id))

    comments = {'id': id,
                'link_id': link_id,
                'parent_id': parent_id,
                'created_utc': created_utc,
                'body': body,
                'author': author,
                'permalink': permalink,
                'score': score,
                'subreddit': subreddit}

    comments = pd.DataFrame(comments)
    
    # Write down csv file
    print('Creating CSV file ', name_csv)
    comments.to_csv('./comments{}.csv'.format(name_csv), index=False)
    print('-'*40)

#for i in files:
#    read_bz2(i)

# using context manager run multiprocess/multithreads
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(read_bz2, files)
    
t2 = time.perf_counter()
print(f'Running time in secs: {t2-t1}')