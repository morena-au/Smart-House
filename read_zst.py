import zstandard as zstd
import json
import pandas as pd

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

with open("RC_2019-06.zst", 'rb') as fh:
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

                previous_line = lines[-1]

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
comments.to_csv('./comments19_06.csv', index=False)