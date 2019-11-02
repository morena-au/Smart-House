# Import library
import csv
import json
import re
import builtwith

import requests
from bs4 import BeautifulSoup
from user_agent import generate_user_agent


# generate a user agent
headers = {
    'User-Agent': generate_user_agent(device_type="desktop", os=('mac', 'linux'))}

# dictionary to collect all websites' content
data = {}


def scrape01(link='https://www.tomsguide.com/us/alexa-vs-siri-vs-google,review-4772.html'):
    '''
    Scrape content from link 1.
    '''

    # Get content from url
    source = requests.get(link).text

    # Content formatting
    soup = BeautifulSoup(source, 'lxml')

    article = soup.find('article')

    # Put title, summary and body all together
    title = article.h1.text
    summary = article.find('p', class_='strapline').text
    content = [title, summary]
    article_body = article.find('div', class_='text-copy bodyCopy auto')
    for i in article_body.find_all('p'):
        # append all paragraphs within the article body
        content.append(i.text)

    return content


def scrape02(link='https://thewirecutter.com/reviews/amazon-echo-vs-google-home/',
             comment_link='https://disqus.com/embed/comments/?base=default&f=thewirecutter&t_i=1000%20https%3A%2F%2Fthewirecutter.com%2F%3Fpost_type%3Dreview%26p%3D1000&t_u=https%3A%2F%2Fthewirecutter.com%2Freviews%2Famazon-echo-vs-google-home%2F&t_e=Amazon%20Echo%20vs.%20Google%20Home%3A%20Which%20Voice%20Controlled%20Speaker%20Is%20Best%20for%20You%3F&t_d=Amazon%20Echo%20vs.%20Google%20Home%3A%20Which%20Voice%20Controlled%20Speaker%20Is%20Best%20for%20You%3F&t_t=Amazon%20Echo%20vs.%20Google%20Home%3A%20Which%20Voice%20Controlled%20Speaker%20Is%20Best%20for%20You%3F&s_o=default#version=049b9d4c8356e0d7fe6487ff57f30ea3'):
    '''
    Scrape content from link 2.
    '''

    # Get content from url
    source = requests.get(link).text

    # Content formatting
    soup = BeautifulSoup(source, 'lxml')
    # print(soup.prettify())

    article = soup.find('article')

    # Put title, summary and body all together
    title = article.h1.text
    summary = article.find('section', class_='_1f90b2b1').p.text
    content = [title, summary]

    section1 = article.find('section', class_='db3dab13')
    for i in section1.find_all('p'):
        content.append(i.text)

    section2 = article.find('div', class_='c670ff69')
    for sec in section2.find_all('section'):
        for par in sec.find_all('p'):
            content.append(par.text)

    # Comments from external iframe
    comment_source = requests.get(comment_link).text
    comment_soup = BeautifulSoup(comment_source, 'lxml')

    post_body = comment_soup.find('script', id='disqus-threadData')

    # Make post body a valid json > extracting the dict-like content inside
    # <script id="disqus-threadData" type="text/json"> ... </script>
    # and load it using json library
    json_dict = json.loads(str(post_body).split(
        'json">')[1].split('</script')[0])

    # Access the list inside two nested dict
    posts_list = json_dict['response']['posts']

    for i in posts_list:
        # Turn message to BeautifulSoup object
        message = BeautifulSoup(i['message'], 'lxml')

        # Put paragraphs together within the same list's element
        comment = ''

        # Extract paragraphs within a post
        for par in message.find_all('p'):
            comment = comment + ' ' + par.text

        # Append the single comment
        content.append(comment)

    return content


# Populate the dictionary
data['Link_1'] = scrape01()
data['Link_2'] = scrape02()

# write csv
with open('data.csv', 'w', newline='') as csvfile:
    fieldnames = ['Link_1', 'Link_2']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow(data)

# webscraping privacy
privacy_train = {}


def privacy_01(link='https://www.cisecurity.org/newsletter/security-and-privacy-in-the-connected-home/'):

    source = requests.get(link, headers=headers).text

    soup = BeautifulSoup(source, 'lxml')

    try:
        body = soup.find('div', class_='div_img_responsive')

        body_par = []
        for i in body.find_all(['p', 'li']):
            body_par.append(i.text)

        # Delete the first row 'From the desk of Thomas F. Duffy, MS-ISAC Chair'
        body_par = body_par[1::]

        print('Link 1 Privacy - Success!')

        return {'link': link, 'category': 'privacy', 'body_par': body_par, 'comment_par': []}

    except:
        print('Link 1 Privacy - Error Occurred..')


def privacy_02(link='https://venturebeat.com/2019/05/15/privacy-remains-a-big-issue-in-todays-smart-home/'):

    source = requests.get(link).text
    soup = BeautifulSoup(source, 'lxml')

    try:
        body = soup.find('article', id='post-2493846')
        body_par = []
        for i in body.find_all('p'):
            body_par.append(i.text)

        # remove disclosure
        body_par = body_par[: -1]

        # remove image description Above
        body_par = [elm for elm in body_par if not elm.startswith('Above')]

        print('Link 2 Privacy - Success!')

        return {'link': link, 'category': 'privacy', 'body_par': body_par, 'comment_par': []}

    except:
        print('Link 2 Privacy - Error Occurred..')


def privacy_03(link='https://venturebeat.com/2018/12/20/alexa-glitch-let-a-user-eavesdrop-on-another-home/'):

    source = requests.get(link).text
    soup = BeautifulSoup(source, 'lxml')

    try:
        body = soup.find('article', id='post-2448604')
        body_par = []

        for i in body.find_all('p'):
            body_par.append(i.text)

        # remove writer
        body_par = body_par[: -1]

        # remove source from first element
        body_par[0] = re.sub(r'^(.*)(?=A )', '', body_par[0])

        print('Link 3 Privacy - Success!')

        return {'link': link, 'category': 'privacy', 'body_par': body_par, 'comment_par': []}

    except:
        print('Link 3 Privacy - Error Occurred..')


def privacy_04(link='https://www.csoonline.com/article/3273929/voice-squatting-attacks-hacks-turn-amazon-alexa-google-home-into-secret-eavesdroppers.html'):

    source = requests.get(link).text
    soup = BeautifulSoup(source, 'lxml')

    try:
        body = soup.find('div', id='drr-container')
        body_par = []
        for i in body.find_all('p'):
            body_par.append(i.text)

        # remove external reading and small paragraph 'The researchers explained'
        del body_par[3:5]

        print('Link 4 Privacy - Success!')

        return {'link': link, 'category': 'privacy', 'body_par': body_par, 'comment_par': []}

    except:
        print('Link 4 Privacy - Error Occurred..')


def privacy_05(link='https://venturebeat.com/2019/04/16/how-to-prevent-alexa-cortana-siri-google-assistant-and-bixby-from-recording-you/'):

    source = requests.get(link).text
    soup = BeautifulSoup(source, 'lxml')

    try:
        body = soup.find('article', id='post-2482858')
        body_par = []
        for i in body.find_all(['p', 'li']):
            body_par.append(i.text)

        # remove image descriptions: Above
        body_par = [elm for elm in body_par if not elm.startswith('Above')]

        print('Link 5 Privacy - Success!')

        return {'link': link, 'category': 'privacy', 'body_par': body_par, 'comment_par': []}

    except:
        print('Link 5 Privacy - Error Occurred..')


def privacy_06(link='https://www.washingtonpost.com/technology/2019/04/23/how-nest-designed-keep-intruders-out-peoples-homes-effectively-allowed-hackers-get/'):

    source = requests.get(link).text
    soup = BeautifulSoup(source, 'lxml')

    try:
        body = soup.find('div', class_='article-body')
        body_par = []
        for i in body.find_all('p'):
            body_par.append(i.text)

        print('Link 6 Privacy - Success!')

        # Load comments from downloaded iframe
        soup = BeautifulSoup(open('./Comments/Talk.html'), 'html.parser')
        comm = soup.find_all(
            'div', class_="talk-plugin-rich-text-text CommentContent__content___ZGv1q")
        comm_par = []

        for i in comm:
            comm_par.append(i.text)

        for num, i in enumerate(comm_par):
            # remove all "\n"
            comm_par[num] = re.sub('[\\n]', '', i)

        return {'link': link, 'category': 'privacy', 'body_par': body_par, 'comment_par': comm_par}

    except:
        print('Link 6 Privacy - Error Occurred..')


def privacy_07(link='https://www.iotworldtoday.com/2019/09/03/smart-home-companies-are-walking-a-privacy-tightrope/'):

    source = requests.get(link).text
    soup = BeautifulSoup(source, 'lxml')

    try:
        body = soup.find(
            'div', class_='columns small-12 single-post-content_text-container')
        body_par = []
        for i in body.find_all('p'):
            body_par.append(i.text)

        # remove element inside square brackets which are ads
        body_par = [brk for brk in body_par if not brk.startswith('[')]

        # remve all \xa0
        for num, i in enumerate(body_par):
            body_par[num] = re.sub('[\\xa0]', '', i)

        print('Link 7 Privacy - Success!')

        return {'link': link, 'category': 'privacy', 'body_par': body_par, 'comment_par': []}

    except:
        print('Link 7 Privacy - Error Occurred..')


def privacy_08(link='https://www.apple.com/newsroom/2019/08/improving-siris-privacy-protections/'):

    source = requests.get(link).text
    soup = BeautifulSoup(source, 'lxml')

    try:
        body_par = []
        for i in soup.find_all('div', class_='pagebody-copy'):
            body_par.append(i.text)

        # remove all "\n"
        for num, i in enumerate(body_par):
            body_par[num] = re.sub('[\\n]', '', i)

        print('Link 8 Privacy - Success!')

        return {'link': link, 'category': 'privacy', 'body_par': body_par, 'comment_par': []}

    except:
        print('Link 8 Privacy - Error Occurred..')


def privacy_09(link='https://www.theguardian.com/technology/2019/jul/26/apple-contractors-regularly-hear-confidential-details-on-siri-recordings'):

    source = requests.get(link).text
    soup = BeautifulSoup(source, 'lxml')

    try:
        body = soup.find(
            'div', class_='content__article-body from-content-api js-article__body')
        body_par = []
        for i in body.find_all('p'):
            body_par.append(i.text)

        print('Link 9 Privacy - Success!')

        return {'link': link, 'category': 'privacy', 'body_par': body_par, 'comment_par': []}

    except:
        print('Link 9 Privacy - Error Occurred..')


def privacy_10(link='http://www.rmmagazine.com/2019/10/01/smart-home-devices-and-privacy-risk/'):

    source = requests.get(link, headers=headers).text
    soup = BeautifulSoup(source, 'lxml')

    try:
        body = soup.find('div', class_='post-21656 post type-post status-publish format-standard has-post-thumbnail hentry category-columns category-compliance category-cyber-risk category-emerging-risks category-featured-1 category-forefront category-international category-legal-risk category-reputation-risk category-security category-technology tag-technology')
        body_par = []
        for i in body.find_all('p'):
            body_par.append(i.text)

        # remove autothor and date first and last 4 elements
        body_par = body_par[1:19]

        print('Link 10 Privacy - Success!')

        return {'link': link, 'category': 'privacy', 'body_par': body_par, 'comment_par': []}

    except:
        print('Link 10 Privacy - Error Occurred..')


def privacy_11(link='https://buildyoursmarthome.co/home-automation/the-privacy-of-data-how-secure-are-we-in-our-smart-homes/'):

    source = requests.get(link).text
    soup = BeautifulSoup(source, 'lxml')

    try:
        body = soup.find('div', class_='post-content description')
        body_par = []
        for i in body.find_all('p'):
            body_par.append(i.text)

        # remove empty and "\xa0"
        for num, i in enumerate(body_par):
            body_par[num] = re.sub('[\\xa0]', '', i)

        body_par = list(filter(None, body_par))

        print('Link 11 Privacy - Success!')

        return {'link': link, 'category': 'privacy', 'body_par': body_par, 'comment_par': []}

    except:
        print('Link 11 Privacy - Error Occurred..')


def privacy_12(link='https://www.welivesecurity.com/2019/10/08/consumer-perceptions-security-privacy-connected-home/'):

    source = requests.get(link).text
    soup = BeautifulSoup(source, 'lxml')

    try:
        body = soup.find(
            'div', class_='col-md-10 col-sm-10 col-xs-12 formatted')
        body_par = []

        for i in body.find_all('p'):
            body_par.append(i.text)

        print('Link 12 Privacy - Success!')

        return {'link': link, 'category': 'privacy', 'body_par': body_par, 'comment_par': []}

    except:
        print('Link 12 Privacy - Error Occurred..')


def privacy_13(link='https://en.wikipedia.org/wiki/Privacy#cite_note-67'):

    source = requests.get(link).text
    soup = BeautifulSoup(source, 'lxml')

    try:
        body = soup.find('div', class_='mw-parser-output')
        body_par = []

        for i in body.find_all('p'):
            body_par.append(i.text)

        # remove "\n" and [d+]
        for num, i in enumerate(body_par):
            body_par[num] = re.sub('\[\d+\]|\\n', '', i)

        print('Link 13 Privacy - Success!')

        return {'link': link, 'category': 'privacy', 'body_par': body_par, 'comment_par': []}

    except:
        print('Link 13 Privacy - Error Occurred..')


def privacy_14(link='https://en.wikipedia.org/wiki/Security'):

    source = requests.get(link).text
    soup = BeautifulSoup(source, 'lxml')

    try:
        body = soup.find('div', class_='mw-parser-output')
        body_par = []

        for i in body.find_all(['p', 'li']):
            body_par.append(i.text)

        # remove references and contents
        body_par = [elm for elm in body_par if not (
            elm.startswith('^') or elm[0].isdigit())]

        # eliminate external link
        body_par = body_par[: len(body_par)-2]

        # remove "\n" and [d+]
        for num, i in enumerate(body_par):
            body_par[num] = re.sub('\[\d+\]|\\n', '', i)

        # remove the empty element in the list
        body_par = list(filter(None, body_par))

        print('Link 14 Privacy - Success!')

        return {'link': link, 'category': 'privacy', 'body_par': body_par, 'comment_par': []}

    except:
        print('Link 14 Privacy - Error Occurred..')


train_data = privacy_14()
