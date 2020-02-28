# Read submission with extension xz, bz2 and zst
import os
import re
import bz2
import json
import pandas as pd
import concurrent.futures
import time
import lzma
import zstandard as zstd

t1 = time.perf_counter()

def read_xz(name):
    # Initiate list
    id = []
    created_utc = []
    title = []
    selftext = []
    author = []
    permalink = []
    score = []
    subreddit = []
    num_comments = []

    with lzma.open(str(name), mode = 'rt', encoding='utf-8') as file:
        for line in file:
            submission = json.loads(line)

            try:
                if submission['subreddit'] in ['smarthome', 'homeautomation']:
                    id.append(submission['id'])
                    created_utc.append(submission['created_utc'])
                    title.append(submission['title'])
                    selftext.append(submission['selftext'])
                    author.append(submission['author'])
                    permalink.append(submission['permalink'])
                    score.append(submission['score'])
                    subreddit.append(submission['subreddit'])
                    num_comments.append(submission['num_comments'])
            except KeyError:
                pass

    print('\n\nDONE!')
    print('lenght: ', len(id))

    submissions = {'id': id,
                'created_utc': created_utc,
                'title': title,
                'selftext': selftext,
                'author': author,
                'permalink': permalink,
                'score': score,
                'subreddit': subreddit,
                'num_comments': num_comments}

    submissions = pd.DataFrame(submissions)

    name_csv = re.search(r'(\d{4}-\d{2})', str(name)).group(1)
        
        # Write down csv file
    print('Creating CSV file ', name_csv)
    submissions.to_csv('./submissions{}.csv'.format(name_csv), index=False)
    print('-'*40)

def read_bz2(name):

    # Initiate list
    id = []
    created_utc = []
    title = []
    selftext = []
    author = []
    permalink = []
    score = []
    subreddit = []
    num_comments = []

    name_csv = re.search(r'(\d{4}-\d{2})', str(name)).group(1)

    print('Start reading bz2 file..', name_csv)
    with bz2.open(str(name), 'rb') as file:
        for line in file:
            submission = json.loads(line)

            try:
                if submission['subreddit'] in ['smarthome', 'homeautomation']:
                    id.append(submission['id'])
                    created_utc.append(submission['created_utc'])
                    title.append(submission['title'])
                    selftext.append(submission['selftext'])
                    author.append(submission['author'])
                    score.append(submission['score'])
                    subreddit.append(submission['subreddit'])
                    num_comments.append(submission['num_comments'])
                    permalink.append(submission['permalink'])
            except KeyError as e:
                if repr(e) == "KeyError('permalink')":
                    # append an empty string
                    permalink.append('')
                else:
                    pass

    print('\n\nDONE!')
    print('lenght: ', len(id))

    submissions = {'id': id,
                'created_utc': created_utc,
                'title': title,
                'selftext': selftext,
                'author': author,
                'permalink': permalink,
                'score': score,
                'subreddit': subreddit,
                'num_comments': num_comments}

    submissions = pd.DataFrame(submissions)
    
    # Write down csv file
    print('Creating CSV file ', name_csv)
    submissions.to_csv('./submissions{}.csv'.format(name_csv), index=False)
    print('-'*40)

def read_zst(name):
    
    # Initiate list
    id = []
    created_utc = []
    title = []
    selftext = []
    author = []
    permalink = []
    score = []
    subreddit = []
    num_comments = []

    with open(str(name), 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            previous_line = ""
            while True:
                # chunk size
                chunk = reader.read(16384)
                if not chunk:
                    break
            
                string_data = chunk.decode('utf-8')
                lines = string_data.split("\n")

                for i, line in enumerate(lines[:-1]):
                    if i == 0:
                        line = previous_line + line
                    
                    try:
                        submission = json.loads(line)
                    except json.decoder.JSONDecodeError:
                        pass

                    # do something with the submission here
                    try:
                        if submission['subreddit'] in ['smarthome', 'homeautomation']:
                            id.append(submission['id'])
                            created_utc.append(submission['created_utc'])
                            title.append(submission['title'])
                            selftext.append(submission['selftext'])
                            author.append(submission['author'])
                            permalink.append(submission['permalink'])
                            score.append(submission['score'])
                            subreddit.append(submission['subreddit'])
                            num_comments.append(submission['num_comments'])
                    except KeyError:
                        pass
                        
                    previous_line = lines[-1]

    print('\nDONE!')
    print('lenght: ', len(id))

    submissions = {'id': id,
                'created_utc': created_utc,
                'title': title,
                'selftext': selftext,
                'author': author,
                'permalink': permalink,
                'score': score,
                'subreddit': subreddit,
                'num_comments': num_comments}

    submissions = pd.DataFrame(submissions)

    name_csv = re.search(r'(\d{4}-\d{2})', str(name)).group(1)

    # Write down csv file
    print('Creating CSV file', name_csv)
    submissions.to_csv('./submissions{}.csv'.format(name_csv), index=False)
    print('-'*40)

# get the downloaded files
path = os.getcwd()
files = [i for i in os.listdir(path) if i.startswith('RS')]

xz_list = []
bz2_list = []
zst_list = []

for file in files:
    if file.endswith('xz'):
        xz_list.append(file)
    elif file.endswith('bz2'):
        bz2_list.append(file)
    elif file.endswith('zst'):
        zst_list.append(file)
    else:
        print('New files extension:', file, '\n')

# using context manager run multiprocess/multithreads
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(read_xz, xz_list)
    executor.map(read_bz2, bz2_list)
    executor.map(read_zst, zst_list)
    
t2 = time.perf_counter()
print(f'Running time in secs: {t2-t1}')