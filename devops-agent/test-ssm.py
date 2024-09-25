import boto3
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

model_kwargs =  { 
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}
from langchain_aws import ChatBedrock
model = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

# Invoke Example
messages = [
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
]

prompt = ChatPromptTemplate.from_messages(messages)

chain = prompt | model | StrOutputParser()

# Chain Invoke
# response = chain.invoke({"question": "tell me a joke"})
# print(response)

from langchain import hub
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_xml_agent
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

@tool("run_shell_command")
def get_instance_info(input):
    """The function is to run a shell command and returns the result of the shell command as output in plain text. the parameters instanceIds is an array of instance IDs, and the parameter command is the one that will be executed in the EC2 instance via SSM. If instanceIds is missing from customer provided information, do not fake it, stop calling tools and ask for more information"""
    print("input: ", input)
    inputJson = json.loads(input)
    # if inputJson[0] == "fake_instanceId":
    #     return "Please provide instance ID"
    ssm_client = boto3.Session().client('ssm')
    command_parameters = {
        'InstanceIds': inputJson[0],  # List of instance IDs
        'DocumentName': 'AWS-RunShellScript',  # SSM Document for running shell scripts
        'Parameters': {
            'commands': [inputJson[1]]  # Shell script command(s) to run
        }
    }
    # Send the command
    response = ssm_client.send_command(**command_parameters)
    print(response)
    # Get the command ID
    command_id = response['Command']['CommandId']
    print(f'Command ID: {command_id}')

    # Wait for the command to complete
    waiter = ssm_client.get_waiter('command_executed')
    waiter.wait(
        CommandId=command_id,
        InstanceId=command_parameters['InstanceIds'][0],
    )

    # Get the command output
    output = ssm_client.get_command_invocation(
        CommandId=command_id,
        InstanceId=command_parameters['InstanceIds'][0],  # Get output for the first instance
    )

    # Print the output
    print(output['StandardOutputContent'])
    return output['StandardOutputContent']
tools = [search, get_instance_info]

# prompt from LangChain Hub: Get the prompt to use - you can modify this!
xml_agent_convo_prompt = hub.pull("hwchase17/xml-agent-convo")

# XML Agent Prompt
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)

template = '''You are DevOps assistant. The infrastructure is running on AWS. Help the user get information or run some commands. You are not allowed to add additional information to the user input.

You have access to the following tools:

{tools}

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>

You are responsible for tool input validation and error handling. If the information is required by the tool and is not provided from user input, you should respond with an error message and ask for more information. For example, tool 'run_shell_command' requires instance ID from the user input. If you do not find the instance ID from user input, you should not call the tool, ask for more information and respond with '<final_answer>cannot process the request, please include a valid instance ID in the input</final_answer>'

For example, if you have a tool called 'run_shell_command' that could run the script on a EC2 instance, you need to make sure the command you run fullfil the requirement from user. Ask the user for more information. If you are asked to list top 10 resource usage in the EC2 instance, you would consider that output with 10 lines would only list 4-5 resources, which means you have to take at least 16 lines from the shell output, you would respond:

<tool>run_shell_command</tool><tool_input>[[\"instanceId\"], \"top -b -n 1 | head -n 16\"]</tool_input>
<observation> op - 06:57:09 up 43 days, 46 min,  0 users,  load average: 0.01, 0.01, 0.00
Tasks: 131 total,   1 running, 130 sleeping,   0 stopped,   0 zombie
%Cpu(s):  0.0 us,  0.0 sy,  0.0 ni,100.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
MiB Mem :   7814.2 total,   6157.6 free,    226.0 used,   1430.6 buff/cache
MiB Swap:      0.0 total,      0.0 free,      0.0 used.   7415.5 avail Mem 

    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
      1 root      20   0  174252  15552   9548 S   0.0   0.2   4:11.65 systemd
      2 root      20   0       0      0      0 S   0.0   0.0   0:01.56 kthreadd
</observation>

the <observation> section in the response wrap the response data from the tool.

When you think you are done, respond with a final answer between <final_answer></final_answer>, <final_answer> must include all information from the <observation> section.

For example:

<final_answer>the output of the command is:  op - 06:57:09 up 43 days, 46 min,  0 users,  load average: 0.01, 0.01, 0.00
Tasks: 131 total,   1 running, 130 sleeping,   0 stopped,   0 zombie
%Cpu(s):  0.0 us,  0.0 sy,  0.0 ni,100.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
MiB Mem :   7814.2 total,   6157.6 free,    226.0 used,   1430.6 buff/cache
MiB Swap:      0.0 total,      0.0 free,      0.0 used.   7415.5 avail Mem 

    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
      1 root      20   0  174252  15552   9548 S   0.0   0.2   4:11.65 systemd
      2 root      20   0       0      0      0 S   0.0   0.0   0:01.56 kthreadd
</final_answer>

Begin!

Previous Conversation:
{chat_history}

Question: {input}
{agent_scratchpad}'''

prompt_template = PromptTemplate(
    input_variables=["agent_scratchpad", "chat_history", "input", "tools"],
    template=template
)

human_message_prompt = HumanMessagePromptTemplate(
    prompt=prompt_template
)

prompt = ChatPromptTemplate(
    input_variables=["agent_scratchpad", "input", "tools"],
    partial_variables={"chat_history": ""},
    messages=[human_message_prompt]
)

# Construct the XML Agent
agent = create_xml_agent(model, tools, prompt)


# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True, verbose=True)

# Invoke XML Agent
# response = agent_executor.invoke({"input": "What is Amazon Bedrock?"})
# print(response)
# print(response['intermediate_steps'][0][0])
# print(response['intermediate_steps'][0][1])
chat_history = f''
# Using with chat history
from langchain_core.messages import AIMessage, HumanMessage

response_with_history = agent_executor.invoke(
    {
        "input": "run a shell command to get top 10 processes that consume resources on EC2 with instanceID i-0e6de72898bb2e9bf",
        # "input": "get me the kernel info of EC2 instanceID i-0e6de72898bb2e9bf",
        # "input": "what is the kernel version",
        # Notice that chat_history is a string
        # since this prompt is aimed at LLMs, not chat models
        "chat_history": f"",
    }
)
print(response_with_history)
print(response_with_history['output'])


