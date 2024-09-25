import subprocess
import streamlit as st
import boto3
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_aws import ChatBedrock
from langchain.agents import AgentExecutor, create_xml_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate

# Initialize Bedrock client
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

model_kwargs = {
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

model = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

# Define tools
search = DuckDuckGoSearchRun()

@tool("list_ec2_instances")
def list_ec2_instances(input):
    """List all EC2 instances in the current region. the tool returns all EC2 instances information in the region input is random number as string"""
    ec2_client = boto3.client('ec2')
    response = ec2_client.describe_instances()
    instances = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instance_id = instance['InstanceId']
            instance_state = instance['State']['Name']
            instance_type = instance['InstanceType']
            instance_name = next((tag['Value'] for tag in instance.get('Tags', []) if tag['Key'] == 'Name'), 'N/A')
            instances.append(f"ID: {instance_id}, Name: {instance_name}, Type: {instance_type}, State: {instance_state}")
    return "\n".join(instances)

@tool("validate_ec2_instanceId")
def validate_ec2_instanceId(input):
    """List all EC2 instances in the current region. the input is instance ID, the tool returns EC2 instances information and a boolean value to tell if the provided instance ID is in the region """
    ec2_client = boto3.client('ec2')
    in_region = False
    response = ec2_client.describe_instances()
    instances = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instance_id = instance['InstanceId']
            if instance_id == input:
                in_region = True
            instance_state = instance['State']['Name']
            instance_type = instance['InstanceType']
            instance_name = next((tag['Value'] for tag in instance.get('Tags', []) if tag['Key'] == 'Name'), 'N/A')
            instances.append(f"ID: {instance_id}, Name: {instance_name}, Type: {instance_type}, State: {instance_state}")
    return "\n".join(instances), in_region

@tool("run_shell_command")
def get_instance_info(input):
    """Run a shell command on an EC2 instance via SSM. Input should be a list of elements. The first element is a list of instance IDs and the second element is to be executed command"""
    print("input: ", input)
    inputJson = json.loads(input)
    ssm_client = boto3.Session().client('ssm')
    command_parameters = {
        'InstanceIds': inputJson[0],
        'DocumentName': 'AWS-RunShellScript',
        'Parameters': {
            'commands': [inputJson[1]]
        }
    }
    response = ssm_client.send_command(**command_parameters)
    command_id = response['Command']['CommandId']
    
    waiter = ssm_client.get_waiter('command_executed')
    waiter.wait(
        CommandId=command_id,
        InstanceId=command_parameters['InstanceIds'][0],
    )
    
    output = ssm_client.get_command_invocation(
        CommandId=command_id,
        InstanceId=command_parameters['InstanceIds'][0],
    )
    
    return output['StandardOutputContent']

@tool
def execute_aws_cli(command: str) -> str:
    """Execute an AWS CLI command locally and return the output."""
    # Validate input
    if not command.startswith("aws "):
        return "Error: Invalid AWS CLI command. The command must start with 'aws'."

    try:
        # Execute the AWS CLI command locally using subprocess
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: Command execution failed. Exit code: {e.returncode}, Error: {e.stderr.strip()}"
    except Exception as e:
        return f"Unexpected error executing AWS CLI command: {str(e)}"

tools = [get_instance_info, validate_ec2_instanceId, list_ec2_instances, execute_aws_cli]

# Define prompt template
template = '''You are DevOps assistant. The infrastructure is running on AWS. Help the user get information or run some commands. You are not allowed to add additional information to the user input.

You have access to the following tools:

{tools}

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>

For the execute_aws_cli tool, make sure to provide the full AWS CLI command as a string. For example: "aws ec2 describe-instances"

You are responsible for tool input validation and error handling. If the information is required by the tool which is not provided from user input or no a valid value, you should respond with an error message and ask for more information. 

For example, tool 'run_shell_command' requires instance ID from the user input. If you do not find the instance ID from user input, you should not call the tool, provide a list of available instances to the user, ask for more information and respond with '<final_answer>cannot process the request, please include a valid instance ID in the input. The instances available in this region are ... </final_answer>'.

For example, tool 'run_shell_command' requires instance ID from the user input. You should validate the instance ID by using tool 'validate_ec2_instanceId'. the tool 'validate_ec2_instanceId' will response with all existing instances in the region and a boolean value if instance ID find among the available instances.

For example, tool 'run_shell_command' requires instance ID from the user input. If you do not find the instance ID from user input, you should not call the tool, ask for more information and respond with '<final_answer>cannot process the request, please include a valid instance ID in the input</final_answer>'

You are also responsible for tool input security validation. It is  If user input asks for deleting all files in the system, or delete files from system path, you should not continue executing the command. Instead, you should stop all actions and response with '<final_answer>The command is to ... as it is a harmful command and I can not fulfill the requirement.</final_answer>'.

For example, if you have a tool called 'run_shell_command' that could run the script on a EC2 instance, you need to make sure the command you run fullfil the requirement from user. Ask the user for more information. If you are asked to list top 10 resource usage in the EC2 instance, you would consider that output with 10 lines would only list 4-5 resources, which means you have to take at least 16 lines from the shell output, you would respond:

<tool>run_shell_command</tool><tool_input>[[\"the instanceId\"], \"top -b -n 1 | head -n 16\"]</tool_input>
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

For example:
<final_answer>
he output of the command 'aws s3 ls' lists all the S3 buckets in the current AWS account and region:
2022-05-19 11:59:45 yagr-demo-sg
2024-08-01 21:36:32 yagr-demo-us-east-1
2022-09-09 15:29:35 yagr-glue-demos
2023-06-19 18:40:45 yagr-jars
2022-09-07 08:38:13 yagr-m2-cache
2023-10-30 11:47:39 yagr-media-input
2023-10-30 15:02:48 yagr-media-origin
2023-10-30 11:47:48 yagr-media-output
2023-07-19 19:11:58 yagr-sharing
2023-04-19 12:54:31 yagr-tfstate-log-us
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

# Create agent and executor
agent = create_xml_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True, verbose=True)

# Streamlit UI
st.title("DevOps Assistant - Langchain")

question = st.text_input("Enter your question:")
instance_id = st.text_input("Enter EC2 Instance ID (if applicable):")

if st.button("Submit"):
    if question:
        with st.spinner("Processing..."):
            input_text = f"{question} {f'for EC2 instance {instance_id}' if instance_id else ''}"
            response = agent_executor.invoke(
                {
                    "input": input_text,
                    "chat_history": "",
                }
            )
            st.write("Response:")
            output = response['output'].split("\n")
            for line in output:
                st.write(line)
    else:
        st.warning("Please enter a question.")
