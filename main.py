import os
from dotenv import load_dotenv
load_dotenv()

from io import BytesIO
import pandas as pd

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import vector_search as vs

BOT_TOKEN = os.getenv('BOT_TOKEN')
APP_TOKEN = os.getenv('APP_TOKEN')

# Initializes your app with your bot token and socket mode handler
app = App(token=BOT_TOKEN)

# Function to list all available problems in the db
def list_problems():
    all_texts = vs.fetch_data_from_db()
    problems = [x['problem'] for x in all_texts]
    return problems

# Function to create excel file from problems list
def create_excel(data):
    df = pd.DataFrame([(i+1,problem) for i, problem in enumerate(data)], columns = ['#','Problem'])
    output = BytesIO()
    df.to_excel(output, index = False, engine = 'openpyxl')
    output.seek(0)
    return output

# Handle the slash command
@app.command("/problem_list")
def handle_problem_list(ack, command):
    ack()  # Acknowledge the command
    problems = list_problems()  # Call your function
    file_stream = create_excel(problems)  # Create the Excel file in memory
    try:
        response = app.client.files_upload_v2(
            channel=command['channel_id'],  # Respond in the same channel
            file=file_stream,
            filename="problems.xlsx",
            title="Problems List"
        )
        # Optionally send a message after uploading the file
        app.client.chat_postMessage(
            channel=command['channel_id'],
            text=f"Here's the list of {len(problems)} problems that are currently available in the database"
        )
    except Exception as e:
        print(f"Error sending file: {e}")
        app.client.chat_postMessage(
            channel=command['channel_id'],
            text="Failed to send the file."
        )



    # response_str = f"We currently have {len(response)} problems listed in the database:\n- " + '\n- '.join(response)
    # respond(response_str)  # Send the response back to Slack

@app.message()
def handle_message(message, say):
    bot_user_id = app.client.auth_test()["user_id"]  # Get the bot's user ID
    bot_name = f"<@{bot_user_id}>"

    # Check if the message contains the bot's name
    if bot_name in message['text']:
        query = message['text'].replace('bot_name','')
        vector_results = vs.vector_search(query)
        search_results = '\n\n'.join(
            [
                f"""Problem: {result['problem']}
                \nsolution: {result['solution']}
                \nDate created: {result['date_created']}
                \nDate modified: {result['date_modified']}"""\
                for key, result in vector_results.items() if result['prob'] > 0.35
            ]
        )
        response = vs.get_response(query, search_results)
        say(response.choices[0].message.content, thread_ts=message['ts'])

@app.event("message")
def handle_message(event):
    pass


# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, APP_TOKEN).start()