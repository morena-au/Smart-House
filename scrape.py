# Import library
import requests
from bs4 import BeautifulSoup
import json

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
