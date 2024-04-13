from langdetect import detect
import warnings

# Filter out UserWarnings and LangChainDeprecationWarnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")


def lawchat(query, chain):
    # Get the user's question
    user_input = query

    bot_output = chain(user_input)
    bot_output = bot_output['result']

    return bot_output
