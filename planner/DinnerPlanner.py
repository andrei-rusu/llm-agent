import os
import re
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from translators import translate_text

from langchain.prompts import PromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.output_parsers import NumberedListOutputParser
from langchain_core.runnables import RunnableSerializable


class ExtendedNumberedListOutputParser(NumberedListOutputParser):

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call."""
        pattern = r"\d+\.\s([\s\S]*?)(?=\n\d+\.|\Z)"

        # Extract the text of each item
        matches = re.findall(pattern, text)
        return matches


def translate_to_user_lang(to_translate: str, user_input: str = 'english', translator: str = 'google') -> str:
    try:
        lang = translate_text(user_input, translator='google', is_detail_result=True)['data'][0][2]
        return translate_text(to_translate, translator=translator, to_language=lang)
    except Exception:
        return to_translate


@st.cache_resource
def load_chain() -> RunnableSerializable[dict, list[str]]:
    output_parser = ExtendedNumberedListOutputParser()
    prompt = PromptTemplate(
        template="Give me example of {number} meals with ingredients: {ingredients}. Specify dish name. Format instructions as bullet points. \
                  Make sure your answer is using the input language. {format_instructions}",
        input_variables=['ingredients', 'number'],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )
    llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True, temperature=0)

    chain = prompt | llm | output_parser
    return chain


def main():
    load_dotenv(find_dotenv())

    st.title("What's for dinner?")

    with st.form(key="my_form"):
        user_prompt = st.text_input("Enter list of ingredients:")
        user_number = int(st.selectbox("How many ideas:", ["1", "2", "3"]))
        _, col2, _ = st.columns([1,0.45,1])
        with col2:
            submit_button = st.form_submit_button(label=" Generate ")

    if submit_button and user_prompt:
        chain = load_chain()
        display_text = f'{translate_to_user_lang("Let us open the book of recipes", user_prompt)}...'
        print(display_text)
        with st.spinner(display_text):
            output = chain.invoke(dict(ingredients=user_prompt, number=user_number))
        cols = st.columns(user_number)
        for i, col in enumerate(cols):
            with col:
                st.write(output[i])


if __name__ == "__main__":
    main()