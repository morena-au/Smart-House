# spelling test based on word frequency of all the comments in the database
from spellchecker import SpellChecker
import MySQL_data as data
import codecs

# # find word frequency
# comments = data.comments
# free_text = ' '.join(list(comments["body"].astype("str")))
# with codecs.open(".\\DataSource_backup\\free_text.txt", "w", "utf-8") as file:
#     file.write(free_text)

def spelling_check(comment_token_list):
    """
    Input list of comment's tokens
    Output misspelled word === correct word === possible candidates
    Press <Enter> to analyse the following comment
    """
    spell = SpellChecker()
    # load contextual words frequency
    spell.word_frequency.load_text_file('.\\DataSource_backup\\free_text.txt')
    # check for missplellings
    misspelled = spell.unknown(comment_token_list)
    for word in misspelled:
        print(word)
        print("="*20)
        print(spell.correction(word))
        print("="*20)
        print(spell.candidates(word))
    print("END COMMENT")
    input()



