# --- This script combines the logic from all cells in your notebook ---

# --- 1. Imports (from Cell 1) ---
import os
import json
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import BraveSearch
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import LLMMathChain
from langchain.tools import Tool
from langchain.tools.render import render_text_description
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# --- 2. Persistent Chat History Class (from Cell 1) ---
class FileChatMessageHistory(BaseChatMessageHistory):
    # ... (The full FileChatMessageHistory class code is here) ...
    def __init__(self, session_id: str, file_path: str = "chat_history.json"):
        self.session_id = session_id
        self.file_path = file_path
        self.history = self._load_history()
    def _load_history(self):
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                session_data = data.get(self.session_id, [])
                messages = []
                for msg_data in session_data:
                    if msg_data['type'] == 'human': messages.append(HumanMessage(content=msg_data['content']))
                    elif msg_data['type'] == 'ai': messages.append(AIMessage(content=msg_data['content']))
                return messages
        except (FileNotFoundError, json.JSONDecodeError): return []
    def _save_history(self):
        try:
            with open(self.file_path, 'r') as f: data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError): data = {}
        session_data = [{'type': msg.type, 'content': msg.content} for msg in self.history]
        data[self.session_id] = session_data
        with open(self.file_path, 'w') as f: json.dump(data, f, indent=4)
    @property
    def messages(self): return self.history
    @messages.setter
    def messages(self, value): self.history = value; self._save_history()
    def add_message(self, message): self.history.append(message); self._save_history()
    def clear(self): self.history = []; self._save_history()

# --- 3. Main App UI (from Cell 1) ---
st.set_page_config(page_title="AI History Tutor", page_icon="🧑‍🏫")
st.title("🧑‍🏫 AI History Tutor")
st.write("Your conversations are saved and will be remembered across sessions.")
persona_options = ["Helpful Assistant", "Harappan Trader", "Mauryan Official", "Gupta Period Scholar"]
selected_persona = st.selectbox("Choose my persona:", persona_options)

# --- 4. Caching and Resource Loading (from Cell 2) ---
@st.cache_resource
def load_resources():
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDZdvehzgwS3j90fiIF5iWXlPoq4QhrRsQ"
    os.environ["BRAVE_API_KEY"] = "BSATa0Fz6QpHgE4OqNkKoXboJdMVkUa"
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    return llm, retriever

# --- 5. Tool Definitions (from Cell 3) ---
# --- 3. Building the Tools 🛠️ ---
@st.cache_resource
def get_tools(_llm, _retriever):
    # --- Base Tools ---
    ncert_search_tool = Tool(
        name="NCERTSearch", 
        func=_retriever.invoke, 
        description="Use for specific questions about Indian History from the NCERT textbook knowledge base."
    )
    
    brave_search = BraveSearch.from_api_key(api_key=os.environ["BRAVE_API_KEY"], search_kwargs={"count": 3})
    web_search_tool = Tool(
        name="WebSearch", 
        func=brave_search.run, 
        description="Use to search the internet for current events or topics not in the textbooks."
    )
    
    wikipedia_tool = Tool(
        name="WikipediaSearch", 
        func=WikipediaAPIWrapper().run, 
        description="Use to search Wikipedia for broad, general knowledge."
    )
    
    calculator_tool = Tool(
        name="Calculator", 
        func=LLMMathChain.from_llm(_llm).run, 
        description="Use for any math questions or calculations."
    )

    # --- Advanced "Teacher" Tools ---
    concept_explainer_chain = (PromptTemplate.from_template("Explain the historical concept of '{topic}' in simple terms for a 10th-grade student, using a helpful analogy.") | _llm | StrOutputParser())
    concept_explainer_tool = Tool(
        name="ConceptExplainer", 
        func=concept_explainer_chain.invoke, 
        description="Use this to explain complex historical concepts, terms, or ideas in simple terms."
    )

    timeline_chain = ({"context": _retriever} | PromptTemplate.from_template("Based on the following text, extract key events and dates and format them as a chronological markdown timeline.\n\nText: {context}") | _llm | StrOutputParser())
    timeline_generator_tool = Tool(
        name="TimelineGenerator", 
        func=timeline_chain.invoke, 
        description="Use this to generate a timeline of events for a specific historical period or dynasty."
    )

    biographer_chain = ({"context": _retriever} | PromptTemplate.from_template("Synthesize the information below to create a concise biography of the historical figure mentioned.\n\nInformation: {context}") | _llm | StrOutputParser())
    key_figure_biographer_tool = Tool(
        name="KeyFigureBiographer", 
        func=biographer_chain.invoke, 
        description="Use this to get a biography of a specific historical person."
    )

    quiz_chain = ({"context": _retriever} | PromptTemplate.from_template("Based on the following context about {topic}, generate a 5-question multiple-choice quiz. Provide a separate answer key at the end.\n\nContext: {context}") | _llm | StrOutputParser())
    quiz_generator_tool = Tool(
        name="QuizGenerator", 
        func=quiz_chain.invoke, 
        description="Use this to create a quiz on a specific historical topic."
    )

    map_chain = (PromptTemplate.from_template("Provide the modern-day location of the ancient place '{place}' and a Google Maps link for it.") | _llm | StrOutputParser())
    map_tool = Tool(
        name="GeographyTool", 
        func=map_chain.invoke, 
        description="Use this to find the modern geographical location of an ancient city or place and provide a map link."
    )
    
    compare_chain = (RunnableParallel(context1=_retriever, context2=_retriever) | PromptTemplate.from_template("Compare and contrast {topic1} and {topic2}. Provide a structured answer with similarities and differences.\n\nTopic 1 Info: {context1}\n\nTopic 2 Info: {context2}") | _llm | StrOutputParser())
    compare_tool = Tool(
        name="CompareAndContrastTool", 
        func=lambda query: compare_chain.invoke(dict(zip(["topic1", "topic2"], query.split(',')))), 
        description="Use this to compare and contrast two historical topics, figures, or dynasties. Input should be a comma-separated list of two items."
    )

    historical_context_chain = (PromptTemplate.from_template("For the Indian historical event '{event}', use a web search to find out what was happening in China and the Roman Empire at roughly the same time.") | _llm | StrOutputParser())
    historical_context_tool = Tool(
        name="HistoricalContextTool", 
        func=historical_context_chain.invoke, 
        description="Use this to get global context for a specific event in Indian history."
    )

    debate_chain = (PromptTemplate.from_template("Present a balanced debate on the historical topic: '{topic}'. Argue for two opposing viewpoints clearly and concisely.") | _llm | StrOutputParser())
    debate_tool = Tool(
        name="DebateTool", 
        func=debate_chain.invoke, 
        description="Use this to get a balanced debate on two sides of a controversial historical topic."
    )

    source_analysis_chain = (PromptTemplate.from_template("You are a history teacher. A student has provided the following historical text. Ask the student three critical thinking questions to help them analyze the source for its origin, purpose, and potential bias. Do not provide answers, only ask questions.\n\nText: '{text}'") | _llm | StrOutputParser())
    source_analysis_tool = Tool(
        name="SourceAnalysisTool", 
        func=source_analysis_chain.invoke, 
        description="Use this when a user provides a short historical text and asks for help analyzing it."
    )

    visual_aid_tool = Tool(
        name="VisualAidFinder", 
        func=lambda topic: brave_search.run(f"image of {topic}"), 
        description="Use this to find a link to a relevant historical image, map, or artifact."
    )

    youtube_tool = Tool(
        name="YouTubeExplainerTool", 
        func=lambda topic: brave_search.run(f"educational YouTube video about {topic}"), 
        description="Use this to find a link to an educational YouTube video on a historical topic."
    )

    return [
        ncert_search_tool, web_search_tool, wikipedia_tool, calculator_tool, 
        concept_explainer_tool, timeline_generator_tool, key_figure_biographer_tool, 
        quiz_generator_tool, map_tool, compare_tool, historical_context_tool, 
        debate_tool, source_analysis_tool, visual_aid_tool, youtube_tool
    ]

# --- 6. Main Application Logic (from Cell 4) ---
llm, retriever = load_resources()
tools = get_tools(llm, retriever)

system_prompts = {
    "Helpful Assistant": "You are a helpful AI assistant and history tutor.",
    "Harappan Trader": "You are a trader from the ancient Harappan city of Lothal...",
    "Mauryan Official": "You are a high-ranking official in the court of Emperor Ashoka...",
    "Gupta Period Scholar": "You are a scholar and poet during the Gupta Empire..."
}

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompts[selected_persona]),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: FileChatMessageHistory(session_id),
    input_messages_key="input",
    history_messages_key="chat_history",
)

# --- 7. Streamlit Chat UI (from Cell 4) ---
session_id = "user_main_session"

if "messages" not in st.session_state:
    history = FileChatMessageHistory(session_id)
    st.session_state.messages = [{"role": ("user" if isinstance(msg, HumanMessage) else "assistant"), "content": msg.content} for msg in history.messages]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your history question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_with_chat_history.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": session_id}}
            )
            st.markdown(response['output'])
    st.session_state.messages.append({"role": "assistant", "content": response['output']})