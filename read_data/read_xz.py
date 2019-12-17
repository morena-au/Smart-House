
def read_xz(name):
    import lzma
    import json
    import time
    import pandas as pd
    import re

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

    start_time = time.time()

    with lzma.open(str(name), mode = 'rt', encoding='utf-8') as file:
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
                    permalink.append(comment['permalink'])
                    score.append(comment['score'])
                    subreddit.append(comment['subreddit'])
            except KeyError:
                pass

    print('\n\nDONE!')
    print('lenght: ', len(id))
    print('Running time in secs: ', time.time() - start_time)

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

    name_csv = re.search(r'(\d{4}-\d{2})', str(name)).group(1)
        
        # Write down csv file
    print('Creating CSV file ', name_csv)
    comments.to_csv('./comments{}.csv'.format(name_csv), index=False)


files = ['RC_2018-08.xz', 'RC_2018-09.xz', 'RC_2018-05.xz', 'RC_2018-04.xz',
         'RC_2018-03.xz', 'RC_2018-02.xz', 'RC_2018-01.xz', 'RC_2017-12.xz']

for i in files:
    read_xz(i)