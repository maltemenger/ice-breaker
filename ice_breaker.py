from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

from third_parties.linkedin import scrape_linkedin_profile


#information = """
#Elon Reeve Musk [ˈiːlɒn ˈɹiːv ˈmʌsk] (* 28. Juni 1971 in Pretoria, Südafrika) ist ein vornehmlich in den Vereinigten Staaten, jedoch auch global wirkender Unternehmer und Milliardär. Er besitzt durch Geburt sowohl die südafrikanische als auch die kanadische Staatsbürgerschaft; 2002 erhielt er zusätzlich die US-amerikanische. Er wurde als Mitinhaber, technischer Leiter und Mitgründer des PayPal-Vorgängers X.com und des Raumfahrtunternehmens SpaceX sowie als Leiter und Mitinhaber des Elektroautoherstellers Tesla bekannt. Darüber hinaus ist er an weiteren Unternehmen beteiligt, seit Oktober 2022 auch an dem Mikrobloggingdienst Twitter. Mit einem geschätzten Vermögen von 242 Milliarden US-Dollar (September 2023) ist er der reichste Mensch der Welt.
#"""

if __name__ == "__main__":
    load_dotenv()


    summery_template = """
    given the linkedin information {information} about a person from I want you to create:
    1. a short summary
    2. two interesting facts about them
    """

    summery_prompt_template = PromptTemplate(input_variables=["information"], template= summery_template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summery_prompt_template)

    data = scrape_linkedin_profile("not_used")
    result = chain.invoke(input={"information": data})

    print(result['text'])




