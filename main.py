from groq import Groq
import streamlit as st
import streamlit_feedback


# Initialize Groq client
try:
    client = Groq(api_key=st.secrets['api']['Groq_API_KEY'])
except KeyError:
    st.error("API key for Groq is missing in secrets.")
    st.stop()


# sidebar controls
model_name = st.sidebar.selectbox(label="Choose the model", options=["llama3-70b-8192"], index=0)
temperature = st.sidebar.slider(label="Set Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
MAX_CHAT_HISTORY_LENGTH = int(st.sidebar.number_input(label="Max history length", min_value=1, max_value=10, value=4))

def clear_chat():
    st.session_state.chat_history = []
st.sidebar.button(label="Clear chat", on_click=clear_chat)


# get the response from Llama LLM
def get_chat_response(message, chat_history):
    chat_history.append({"role": "user", "content": message})
    # Keep the chat history within the specified max length
    if len(chat_history) > MAX_CHAT_HISTORY_LENGTH:
        chat_history = chat_history[-MAX_CHAT_HISTORY_LENGTH:]

    try:
        stream = client.chat.completions.create(
            messages=chat_history,
            model=model_name,
            temperature=temperature,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        st.error(f"Error while generating response: {e}")
        return


def main():
    st.title("Ask the AI 🤖")
    
    # Session state to store chat history and the last response
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    else:
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg['content'])

    user_input = st.chat_input("Enter your message...", key="user_input")
    if user_input:    # st.button(label="Send")
        if user_input.strip():
            st.chat_message("user").write(user_input)
            with st.spinner("AI is typing..."):
                response_placeholder = st.empty()    # For dynamic updates
                ai_response = ""
                for chunk in get_chat_response(user_input, st.session_state.chat_history):
                    ai_response += chunk
                    response_placeholder.chat_message("assistant").write(ai_response)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                feedback = streamlit_feedback.streamlit_feedback(feedback_type='thumbs',
                                                                optional_text_label='[Optional] please provide an explanation',
                                                                key=f'feedback_{len(st.session_state.chat_history)}')
        else:
            # pass
            st.warning("Please enter a message!")


if __name__ == "__main__":
    main()
