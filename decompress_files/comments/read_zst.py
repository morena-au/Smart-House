def read_zst(name):
    import zstandard as zstd
    import json
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
                        object = json.loads(line)
                    except json.decoder.JSONDecodeError:
                        pass

                    # do something with the object here
                    try:
                        if object['subreddit'] in ['smarthome', 'homeautomation']:
                            id.append(object['id'])
                            link_id.append(object['link_id'])
                            parent_id.append(object['parent_id'])
                            created_utc.append(object['created_utc'])
                            body.append(object['body'])
                            author.append(object['author'])
                            permalink.append(object['permalink'])
                            score.append(object['score'])
                            subreddit.append(object['subreddit'])
                    except KeyError:
                        pass
                        
                    previous_line = lines[-1]
    
    print('\nDONE!')
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

    name_csv = re.search(r'(\d{4}-\d{2})', str(name)).group(1)
    
    # Write down csv file
    print('Creating CSV file')
    comments.to_csv('./comments{}.csv'.format(name_csv), index=False)


files = ['RC_2019-03.zst', 'RC_2019-02.zst', 'RC_2019-01.zst', 'RC_2018-12.zst', 'RC_2018-11.zst', 'RC_2018-10.zst']

for i in files:
    read_zst(i)
