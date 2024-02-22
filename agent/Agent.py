import os
import streamlit as st

import langchain.agents as agents
from langchain import hub
from langchain.agents import AgentExecutor, Tool
from langchain.chains import LLMMathChain
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from utils.html_templates import USER_AVATAR_URL, BOT_AVATAR_URL


AGENT_OPTIONS = {
    'structured': ('create_structured_chat_agent', "hwchase17/structured-chat-agent"),
    'json': ('create_json_chat_agent', "hwchase17/react-chat-json"),
    'react': ('create_react_agent', "hwchase17/react"),
    'openai': ('create_openai_functions_agent', "hwchase17/openai-functions-agent"),
}


def main():
    st.set_page_config(
        page_title="MRKL Agent", page_icon="🦜", layout="wide",
    )
    st.header("🕵️ MRKL Agent")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = StreamlitChatMessageHistory()

    # Setup credentials and other application settings
    with st.sidebar:
        user_api_key = st.text_input(
            "Google API Key", type="password", help="Set this to run the agent."
        )
        if user_api_key:
            api_key = user_api_key
        else:
            from dotenv import load_dotenv, find_dotenv
            load_dotenv(find_dotenv())
            api_key = os.environ['GOOGLE_API_KEY']
        col1, col2 = st.columns([1, 1])
        agent_selected = col1.selectbox('Agent types', options=AGENT_OPTIONS.keys())
        temperature = col2.slider('Temperature', min_value=0.0, max_value=1.0, step=0.1, value=0.0)
        verbose = st.checkbox("Verbose", value=False)

    # Initialize agent
    agent = init_agent(api_key, agent_selected, temperature, verbose)

    if st.sidebar.button("Clear message history"):
        st.session_state.chat_history.clear()

    st.chat_message('model', avatar=BOT_AVATAR_URL).write("How can I assist you?", unsafe_allow_html=True)
    for msg in st.session_state.chat_history.messages:
        if msg.type == 'human':
            st.chat_message('user', avatar=USER_AVATAR_URL).write(msg.content)
        else:
            st.chat_message('model', avatar=BOT_AVATAR_URL).write(msg.content, unsafe_allow_html=True)

    if user_query := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user", avatar=USER_AVATAR_URL).write(user_query)
        answer_container = st.chat_message("model", avatar=BOT_AVATAR_URL)
        st_callback = StreamlitCallbackHandler(answer_container)
        answer = agent.invoke({"input": user_query}, 
                              {"configurable": {"session_id": "<new_sesh>"}, 'callbacks': [st_callback]})
        answer_container.write(answer["output"], unsafe_allow_html=True)


@st.cache_resource
def init_agent(api_key, agent_selected='structured', temperature=0, verbose=False):
    # Tools setup
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature, google_api_key=api_key, 
                                 convert_system_message_to_human=True, streaming=True)
    llm_math_chain = LLMMathChain.from_llm(llm)
    tools = [
        DuckDuckGoSearchResults(api_wrapper=DuckDuckGoSearchAPIWrapper(region="wt-wt", time="d", max_results=2), 
                                source="text"),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
        ),
    ]
    # Initialize agent executor
    agent_creator, prompt_source = AGENT_OPTIONS[agent_selected]
    agent = getattr(agents, agent_creator)(llm, tools, hub.pull(prompt_source))
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=verbose, handle_parsing_errors=True,
        max_iterations=5,
    )
    # Create executor wrapper with message history
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: st.session_state.chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history


if __name__ == "__main__":
    main()