from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import load_prompt

# prompt = load_prompt("./prompt.json")
chat = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])



examples = [
    {
        "question": "불고기",
        "answer": 
        """Pronounced: bool-goh-gee|Explain:|Bulgogi is a popular Korean dish made with thinly sliced beef marinated in a mixture of soy sauce| sugar| sesame oil| garlic| and pepper. It is known for its sweet and savory flavor profile. Bulgogi is typically grilled or pan-fried and often served with rice| lettuce leaves| and ssamjang (a spicy dipping sauce).|Allergy:|Please note that bulgogi may contain soy and sesame| which are common allergens for some individuals."""
    },
    {
        "question": "순대국",
        "answer": """Pronounced: soon-dae-guk|Explain:|Soondae guk is a traditional Korean soup made with blood sausage (soondae)| vegetables| and sometimes noodles. The soup has a rich and savory flavor| with the blood sausage providing a unique texture. It is often enjoyed as a comforting and hearty meal| especially during colder months.|Allergy:|Please note that soondae guk contains blood sausage| which may be an allergen for some individuals. """
    },
    {
        "question": "감자탕",
        "answer": """Pronounced: gam-ja-tang|Explain:|Gamjatang is a traditional Korean soup made with pork spine| potatoes| and a variety of vegetables. The soup has a rich and hearty flavor| with a mildly spicy broth seasoned with red chili paste| garlic| and fermented soybean paste. It is often enjoyed as a comforting meal and is especially popular during colder seasons or as a late-night dish.|Allergy:|Please note that gamjatang contains pork and soybean-based seasonings| which may be allergens for some individuals."""
    }

]




example_template = """
    Human: {question}
    AI: {answer}
"""

example_prompt = ChatPromptTemplate.from_messages([
    ("human: {question}"),
    ("ai", "{answer}")
])

example_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 외국인에게 음식에 대해 알려주는 음식 설명 전문가야. 짧게 두 문장으로 설명해줘. 항상 영어로 말해줘. 여러 개의 음식이 들어오면 둘 다 알려줘"),
    example_prompt,
    ("human", "{food}")
])


chain = final_prompt | chat

# chain.invoke(
#     {
#         "food": "제육볶음"
#     }
# )