import validators
import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader , YoutubeAudioLoader , UnstructuredURLLoader


st.title("Youtube and URL to Summary with : Langchain")
st.subheader("Summarize any URL or Youtube video with a Link!")
st.subheader("(Small error : Please put the groq api key to proceed.)")

with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

llm  = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    max_retries=2,
    temperature=1,
    streaming=True
)

prompt_temp = """
    You are a Helpful Summary Bot that summarizes any Youtube video or Website url.
    The summary should be of 500 words max.
    Use number points and Highlight important keywords and information.
    
    Content : {text}
"""

prompt = PromptTemplate(template=prompt_temp, input_variables=['text'])


url = st.text_input(label="Enter your URL " , label_visibility="collapsed")

if st.button("Summarize"):
    if not groq_api_key.strip() or not url.strip():
        st.error("Please enter a URL")
    elif not validators.url(url):
        st.error("Invalid URL")
        
    else :
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url,add_video_info=False)
                else:
                    loader = UnstructuredURLLoader(urls=[url],ssl_verify=False,headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                
                docs = loader.load()
                
                chain = load_summarize_chain(llm=llm , chain_type="stuff" , prompt=prompt , verbose=True)
                output_summary = chain.run(docs)
                
                st.success(output_summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")
