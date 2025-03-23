from langchain.schema import SystemMessage, HumanMessage, AIMessage
from huggingface_hub import InferenceClient

from llm.utils import extract_json_from_string, compare_string_fuzzy
from llm.types import request_types, request_types_str


def classify_email(text: str, model: InferenceClient):
    ai_role_prompt = 'Remember that you are a banker working on email classification'
    
    goal_prompt = f'''
    You are given an email. Extract the main intent of the email.

    Email:"""{text}"""

    Imagine you are a banker and working on these emails the bank has received. Your answers must be with respect to the bank. Identify the request type, sub request type, and intent of the email. The list of request types are given in the below json file - the subrequest is inside the request.
    Ensure that the request and sub request you are returning is found in the list.
    """{request_types_str}"""
    '''
    
    llm = LLMChat(model)
    llm.add_system_message(ai_role_prompt)
    
    llm.prompt(goal_prompt)
    
    follow_up_question = '''
        give the final output as a json file with the fields "requestType", "subRequestType", "intent".
        For example: 
        {
            "requestType":  "Au",
            "subRequestType": "None",
            "intent": "To change loan from branch Maine to Branch Oakland" 
        }
        Return only the json file, nothing else
    '''
    
    second_response = llm.prompt(follow_up_question)
    
    formatted_response = extract_json_from_string(second_response)
    
    request_type = formatted_response['requestType']
    # print(f'identified response type to be {request_type}')

    request_type_fields = [rt['fields'] for rt in request_types['requestType'] if compare_string_fuzzy(rt['name'], request_type)][0]
    
    specific_fields_prompt = f'''
        extract the given fields from the email: {request_type_fields}, also include any other fields you think might be relevant (do not create duplicate fields with similar values), return a single json list (if there are multiple lists, combine them and remove the duplicates) in the following format: [
            {{
                "field1Name": "field1Value",
                "field2Name": "field2Value",
            }}
        ]
    '''
    
    third_response = llm.prompt(specific_fields_prompt)
    
    request_field_values = extract_json_from_string(third_response)
    
    formatted_response['attributes'] = request_field_values
    
    return formatted_response

class LLMChat:
    def __init__(self, llm: InferenceClient):
        self.llm = llm
        self.chat_history = []
        self.user_message_count = 0
    
    def add_system_message(self, message: str):
        self.chat_history.append({"role": "system", "content": message})
    
    def __add_user_message(self, message: str):
        self.chat_history.append({"role": "user", "content": message})
        
    def __add_assistant_message(self, message: str):
        self.chat_history.append({"role": "assistant", "content": message})
        
    def prompt(self, prompt: str) -> str:
        self.user_message_count += 1
        self.__add_user_message(prompt)
        print(f'prompt type: {type(prompt)}')
        print(f'-   PROMPT #{self.user_message_count}: {self.chat_history[-1]['content'][1:100]}...')
        chat_completion_output = self.llm.chat_completion(
            messages=self.chat_history,
            max_tokens=1024,
            temperature=0.001
        )
        response = chat_completion_output.choices[0].message.content
        print(f'- RESPONSE #{self.user_message_count}: {response}')
        
        self.__add_assistant_message(response)
        
        return response