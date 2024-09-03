import bs4, os, getpass
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

os.environ['USER_AGENT'] = "TEST_AGENT"
os.environ['OPENAI_API_KEY'] = getpass.getpass()
# Load, chunk and index the contents of the blog.

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))

retriever = vectorstore.as_retriever()

user_query = "agent memory"

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="""
    Answer has relevant information about {user_query}. 
    And output is json format.
    evaluate retrieval quality between user query and retrieved chunk. 
    only answer is relevance.
    
    Example:
        If relevant, output relevance is yes.
        else output relevance is no.
    """,
    input_variables=["user_query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

result = chain.invoke({"user_query": user_query})
print(result)