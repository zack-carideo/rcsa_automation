from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from ..prompts.registry import PromptRegistry
load_dotenv() 


registry = PromptRegistry()
prompt, meta = registry.get("control_description_qc")

llm = ChatAnthropic(
    model=meta["model"],
    temperature=meta["temperature"],
)

chain = prompt | llm


if __name__ == "__main__":
    result = chain.invoke(
        {'json_input':{
            "risk_description": "test",
            "control_description": "Structuring - multiple cash deposits below $10K",
        }
         }
    )

    print(result.content)