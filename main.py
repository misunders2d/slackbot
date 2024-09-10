import os
from dotenv import load_dotenv
load_dotenv()
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import vector_search as vs

BOT_TOKEN = os.getenv('BOT_TOKEN')
APP_TOKEN = os.getenv('APP_TOKEN')

# Initializes your app with your bot token and socket mode handler
app = App(token=BOT_TOKEN)

@app.event("message_deleted")
def handle_message_deleted(event):
    pass

@app.message()
def handle_message(message, say):
    bot_user_id = app.client.auth_test()["user_id"]  # Get the bot's user ID
    bot_name = f"<@{bot_user_id}>"

    # Check if the message contains the bot's name
    if bot_name in message['text']:
        query = message['text']
        vector_results = vs.vector_search(query)
        search_results = '\n\n'.join([f"Problem: {result['problem']}, solution: {result['solution']}" for key, result in vector_results.items() if result['prob'] > 0.35])
        response = vs.get_response(query, search_results)
        say(response.choices[0].message.content, thread_ts=message['ts'])

# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, APP_TOKEN).start()