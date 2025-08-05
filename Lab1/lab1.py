from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv
import os

# Load Environment Variables
load_dotenv()
gak = os.getenv("GOOGLE_API_KEY")

# Initializing LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=gak
)

# Creating Prompts
order_prompt = PromptTemplate.from_template(
    "You are Aryan Patel, a customer support specialist for Amazon. Respond directly and professionally to the following order-related customer query. Do not include any introduction or explanation:\n\n{input}"
)

refund_prompt = PromptTemplate.from_template(
    "You are Krish Patel, a customer support specialist for Amazon. Respond clearly, empathetically, and professionally to the following refund-related customer query. Do not include any meta text or extra introduction:\n\n{input}"
)

product_prompt = PromptTemplate.from_template(
    "You are Varshil Patel, a product expert for Amazon. Answer the following product inquiry accurately and professionally. Avoid any introductory phrases or explanation:\n\n{input}"
)

general_prompt = PromptTemplate.from_template(
    "You are Harsh Mistry, a customer service representative at Amazon. Politely acknowledge the customer's feedback below. Thank them sincerely, and explain how Amazon will use their feedback to improve. Do not include extra introductory sentences:\n\n{input}"
)

router_prompt = PromptTemplate.from_template(
    """Classify the following customer query into one of the following categories:
1) order_issue
2) refund_request
3) product_inquiry
4) general_feedback

Return only the category name (e.g., order_issue). Do not include any explanation or greeting.

Customer query: {input}"""
)

classifier_chain = router_prompt | llm | StrOutputParser()
order_chain = order_prompt | llm | StrOutputParser()
refund_chain = refund_prompt | llm | StrOutputParser()
product_chain = product_prompt | llm | StrOutputParser()
general_chain = general_prompt | llm | StrOutputParser()
fallback_chain = lambda x: "We could not classify your query. Apologies for the inconvenience."

branch = RunnableBranch(
    (lambda x: "order_issue" in x["category"], order_chain),
    (lambda x: "refund_request" in x["category"], refund_chain),
    (lambda x: "product_inquiry" in x["category"], product_chain),
    (lambda x: "general_feedback" in x["category"], general_chain),
    fallback_chain,
)

full_chain = RunnablePassthrough.assign(
    category=classifier_chain
).assign(
    response=branch
)

if __name__ == "__main__":
    while True:
        user_input = input("Customer: ").strip()
        if not user_input:
            print("Exiting...")
            break
        
        result = full_chain.invoke({"input": user_input})
        print(f"Category: {result['category']}")
        print(f"Response: {result['response']}")