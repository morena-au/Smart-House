import pandas as pd
import requests
import json
import csv
import datetime
import re


def getPushShiftData(query_term, subreddit, after=1546300800, before=1561939200, sort='asc'):
    '''
    Input:
        - query_term
        - subreddit
        - timestamp in unix epoch defined in After and Before
        - sort: default asc (Oldest to Newest)
        e.g. term privacy restricted to the subreddit smarthome 6 months after 01.01.2019 12 AM

    Output:
        DICT STRUCTURE:

        {submission_id: [title + selftext, full_link, score, num_comments, subreddit, category,
                            {comment_id : [body, score, parent_id],
                             comment_id : [body, score, parent_id],
                             . . .}],

         submission_id: [title + selftext, full_link, score, num_comments, subreddit, category,
                             {comment_id : [body, score, parent_id],
                              comment_id : [body, score, parent_id],
                              . . .}],

         . . .

        }

    API link: https://github.com/pushshift/api
    '''

    # from relevant comments extract link_ids which refer to the submission id
    url_comment = 'https://api.pushshift.io/reddit/search/comment/?q=' + \
        str(query_term)+'&subreddit='+str(subreddit) + \
        '&after='+str(after)+'&before='+str(before) + \
        '&sort='+str(sort)+'&sort_type=score'

    r = requests.get(url_comment)
    data = json.loads(r.text)
    link_ids = []
    for comment in data['data']:
        link_ids.append(comment['link_id'])

    link_ids = pd.unique(link_ids)

    print('Count unique submissions from comment:')
    print(len(link_ids))
    print('-'*20)

    # Initiate final dict
    data_dict = {}

    for link in link_ids:
        # obtain submission's info
        url_submission = 'https://api.pushshift.io/reddit/search/submission/?subreddit=' + \
            str(subreddit)+'&ids='+str(link)

        r_sub = requests.get(url_submission)
        data_sub = json.loads(r_sub.text)

        # print out id if submission return an empty list
        # all comments from that subm disregarded
        if not data_sub['data']:
            print('\nSubmission returned empty:')
            print('https://api.pushshift.io/reddit/search/submission/?subreddit=' +
                  str(subreddit)+'&ids='+str(link))
            print('https://api.pushshift.io/reddit/search/comment/?subreddit=' +
                  str(subreddit)+'&link_id='+str(link) +
                  '&sort_type=score')
            print('https://reddit.com/{}'.format(re.sub('.*_', '', link)))
            print('\n\n')

        # Add comments {comment_id: [body, score, parent_id]}
        # sort them by score == BEST
        url_comment = 'https://api.pushshift.io/reddit/search/comment/?subreddit=' + \
            str(subreddit)+'&link_id='+str(link) + \
            '&sort_type=score'

        r_com = requests.get(url_comment)
        data_com = json.loads(r_com.text)

        for elm in data_sub['data']:
            # Add submission
            data_dict[elm['id']] = [elm['title'] + '\n' + elm['selftext'],
                                    elm['full_link'], elm['score'], elm['num_comments'],
                                    str(subreddit), str(query_term), {}]

            for com in data_com['data']:
                data_dict[elm['id']][6][com['id']] = [
                    com['body'], com['score'], com['parent_id']]

    print('Dictionary\'s length for submission from comments:')
    print(len(data_dict))
    print('-'*20)

    # check if there are other submission with the relevant search params
    url_sub = 'https://api.pushshift.io/reddit/search/submission/?q=' + \
        str(query_term)+'&subreddit='+str(subreddit) + \
        '&after='+str(after)+'&before='+str(before) + \
        '&sort='+str(sort)

    r_s = requests.get(url_sub)
    data_s = json.loads(r_s.text)

    sub_id = []
    for i in data_s['data']:
        sub_id.append(i['id'])

    # find the submission not included yet
    old = [re.sub('.*_', '', link) for link in link_ids]
    new = sub_id

    new = set(new) - set(old)

    print('Count new unique submissions from submission:')
    print(len(new))
    print('-'*20)

    for elm in data_s['data']:
        if elm['id'] in new:
            # Add comments {comment_id: [body, score, parent_id]}
            # sort them by score == BEST
            url_comment = 'https://api.pushshift.io/reddit/search/comment/?subreddit=' + \
                str(subreddit)+'&link_id='+str(elm['id']) + \
                '&sort_type=score'

            r_com = requests.get(url_comment)
            data_com = json.loads(r_com.text)

            # Add new submission to the dictionary
            data_dict[elm['id']] = [elm['title'] + '\n' + elm['selftext'],
                                    elm['full_link'], elm['score'], elm['num_comments'],
                                    str(subreddit), str(query_term), {}]

            for com in data_com['data']:
                data_dict[elm['id']][6][com['id']] = [
                    com['body'], com['score'], com['parent_id']]

    print('Dictionary\'s length with all submissions:')
    print(len(data_dict))
    print('='*10, 'End', '='*10)

    return data_dict


subreddit = ['smarthome', 'homeautomation']
query_term = ['privacy', 'security', 'trust']

# initiate data empty list of dicts
data = []

for num, reddit in enumerate(subreddit):
    for term in query_term:
        print('\nSUBREDDIT:', reddit, '\tQUERY TERM:', term)
        data.append(getPushShiftData(query_term=term,
                                     subreddit=reddit))
    print('\nDIAGNOSTICS: \n')
    for i in range(len(data)):
        print('Length with duplicates within subreddit')
        print('Dictionary {}\t{}'.format(i, len(data[i].keys())))

    # remove subreddit_ids that are duplicate within the subreddit
    # init as first dict of the subreddit
    unique_key = list(data[num*len(query_term)].keys())

    for i in range(num*len(query_term), len(data)-1):
        # Update dict only with unique keys
        data[i+1] = {k: v for k, v in data[i+1].items() if k not in unique_key}
        # add unique keys from all previous dicts
        unique_key = unique_key + list(data[i+1].keys())

    print('\nDIAGNOSTICS: \n')
    for i in range(len(data)):
        print('Length with NO duplicates within subreddit')
        print('Dictionary {}\t{}'.format(i, len(data[i].keys())))


# write the data in a json_file
with open('reddit_data.json', 'w') as outfile:
    # json.dump each dict individually and write the commas and new lines manually
    outfile.write(
        '[' +
        ',\n'.join(json.dumps(i) for i in data) +
        ']')
