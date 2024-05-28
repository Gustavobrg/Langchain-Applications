import os

os.environ['ELEVEN_API_KEY']=''
os.environ['OPENAI_API_KEY'] = ""
os.environ['ACTIVELOOP_TOKEN'] = ""

import os
import openai
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from elevenlabs import generate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from streamlit_chat import message
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Constants
TEMP_AUDIO_PATH = "temp_audio.wav"
AUDIO_FORMAT = "audio/wav"

# Load environment variables from .env file and return the keys
openai.api_key = os.environ.get('OPENAI_API_KEY')
eleven_api_key = os.environ.get('ELEVEN_API_KEY')
my_activeloop_org_id = "gustavobarbosa060"
my_activeloop_dataset_name = "langchain_jarvis_assistant"
dataset_path= f'hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}'
active_loop_data_set_path = dataset_path


# Load embeddings and DeepLake database
def load_embeddings_and_database(active_loop_data_set_path):
    embeddings = OpenAIEmbeddings()
    db = DeepLake(
        dataset_path=active_loop_data_set_path,
        read_only=True,
        embedding_function=embeddings
    )
    return db

# Transcribe audio using OpenAI Whisper API
def transcribe_audio(audio_file_path):

    import whisper

    try:
        # load the model
        model = whisper.load_model("base")

        result = model.transcribe(audio_file_path)
        return result['text']
    except Exception as e:
        print(f"Error calling Whisper API: {str(e)}")
        return None

# Record audio using audio_recorder and transcribe using transcribe_audio
def record_and_transcribe_audio():
    audio_bytes = audio_recorder()
    transcription = None
    if audio_bytes:
        st.audio(audio_bytes, format=AUDIO_FORMAT)

        with open(TEMP_AUDIO_PATH, "wb") as f:
            f.write(audio_bytes)

        if st.button("Transcribe"):
            transcription = transcribe_audio(TEMP_AUDIO_PATH)
            os.remove(TEMP_AUDIO_PATH)
            display_transcription(transcription)

    return transcription

# Display the transcription of the audio on the app
def display_transcription(transcription):
    if transcription:
        st.write(f"Transcription: {transcription}")
        with open("audio_transcription.txt", "w+") as f:
            f.write(transcription)
    else:
        st.write("Error transcribing audio.")

# Get user input from Streamlit text input field
def get_user_input(transcription):
    return st.text_input("", value=transcription if transcription else "", key="input")

# Search the database for a response based on the user's query
# Search the database for a response based on the user's query
def search_db(user_input, db):
    print(user_input)
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['k'] = 4
    model = ChatOpenAI(model_name='gpt-3.5-turbo')
    qa = RetrievalQA.from_llm(model, retriever=retriever, return_source_documents=True)
    return qa({'query': user_input})



# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))
        #Voice using Eleven API
        voice= "Rachel"
        text= history["generated"][i]
        audio = generate(text=text, voice=voice,api_key=eleven_api_key)
        st.audio(audio, format='audio/mp3')

# Main function to run the app
def main():
    # Initialize Streamlit app with a title
    st.write("# JarvisBase 🧙")
   
    # Load embeddings and the DeepLake database
    db = load_embeddings_and_database(active_loop_data_set_path)

    # Record and transcribe audio
    transcription = record_and_transcribe_audio()

    # Get user input from text input or audio transcription
    user_input = get_user_input(transcription)

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]
        
    # Search the database for a response based on user input and update session state
    if user_input:
        output = search_db(user_input, db)
        print(output['source_documents'])
        st.session_state.past.append(user_input)
        response = str(output["result"])
        st.session_state.generated.append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)

# Run the main function when the script is executed
if __name__ == "__main__":
    main()