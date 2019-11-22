import csv
import datetime
import json
import re
import sys

import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt


def PS_aggs(query_term, subreddit, after, frequency):
    '''
    Input: 
        > query_term: String / Quoted String for phrases
        > subreddit: String
        > after: Epoch value or Integer + "s,m,h,d" (i.e. 30d for 30 days)

    Output:
        > panda daframe: doc_count, key, subreddit
    '''
    # API link: https://github.com/pushshift/api
    url_comment = 'https://api.pushshift.io/reddit/search/comment/?q=' + \
        str(query_term)+'&subreddit='+str(subreddit) + \
        '&after='+str(after)+'&aggs=created_utc&frequency=' + \
        str(frequency)+'&size=0'

    try:
        r = requests.get(url_comment)
        data = json.loads(r.text)

        # List of information we are interested to collect
        freq = []
        time = []
        group = []
        query = []

        # loop through all the comments and collect the info
        for i in data['aggs']['created_utc']:
            freq.append(i['doc_count'])
            time.append(i['key'])
            group.append(subreddit)
            query.append(query_term)

        data = {'time': time,
                'freq': freq,
                'query': query,
                'subreddit': subreddit}

        print('Call Succeed')

        return pd.DataFrame(data)

    except:
        print('Call Failed')


# populate the datasets
query_term = ['privacy', 'security', 'trust']
subreddit = ['homeautomation', 'smarthome']
df = []

for s in subreddit:
    for q in query_term:
        df.append(PS_aggs(q, s, after='3y', frequency='month'))

        data = pd.concat(df, ignore_index=True)

# subset datasets
privacy = data.loc[data['query'] == 'privacy']
security = data.loc[data['query'] == 'security']
trust = data.loc[data['query'] == 'trust']

# Create seaborn trend plot
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(16, 8), dpi=160, sharex=True)
# privacy
sns.lineplot(x="time", y="freq", hue="subreddit", data=privacy, ax=ax1)
ax1.set_title(
    'Number of comments mentioning PRIVACY each month over a period of three years')
ax1.set_xticks(privacy['time'].unique())
xticks = ax1.get_xticks()
ax1.set_xticklabels(
    [pd.to_datetime(tm, unit='s', origin='unix').strftime('%b %Y') for tm in xticks], rotation=90)
# security
sns.lineplot(x="time", y="freq", hue="subreddit", data=security, ax=ax2)
ax2.set_title(
    'Number of comments mentioning SECURITY each month over a period of three years')
ax2.set_xticks(privacy['time'].unique())
xticks = ax2.get_xticks()
ax2.set_xticklabels(
    [pd.to_datetime(tm, unit='s', origin='unix').strftime('%b %Y') for tm in xticks], rotation=90)
# trust
sns.lineplot(x="time", y="freq", hue="subreddit", data=trust, ax=ax3)
ax3.set_title(
    'Number of comments mentioning TRUST each month over a period of three years')
ax3.set_xticks(privacy['time'].unique())
xticks = ax3.get_xticks()
ax3.set_xticklabels(
    [pd.to_datetime(tm, unit='s', origin='unix').strftime('%b %Y') for tm in xticks], rotation=90)
plt.savefig('output/Trends')
plt.show()
