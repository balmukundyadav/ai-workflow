Please write streamlite python code that uses multiple ai agents call, I would likento call backend server based on the agent used to answer questions, and backend servers are running on different hosts and port
ChatGPT said:
ChatGPT
Here’s a Streamlit app code that supports calling multiple AI agents running on different backend servers. Each backend server corresponds to an agent, and the user can select the agent to handle the query. The app sends the question to the selected backend and displays the response.

python
Copy
Edit
import streamlit as st
import requests

# Dictionary to map agents to their respective backend servers
AGENTS = {
    "Agent 1": {"host": "http://localhost", "port": 8001},
    "Agent 2": {"host": "http://localhost", "port": 8002},
    "Agent 3": {"host": "http://192.168.1.10", "port": 8003},
}

def call_backend(agent, question):
    """
    Sends the question to the selected agent's backend server.

    Args:
        agent (str): The selected agent name.
        question (str): The user query.

    Returns:
        str: The response from the backend server.
    """
    server_info = AGENTS.get(agent)
    if not server_info:
        return "Invalid agent selected."
    
    url = f"{server_info['host']}:{server_info['port']}/answer"
    try:
        response = requests.post(url, json={"question": question}, timeout=10)
        response.raise_for_status()
        return response.json().get("answer", "No response found.")
    except requests.exceptions.RequestException as e:
        return f"Error connecting to {agent}: {e}"

# Streamlit UI
st.title("Multi-Agent AI Query App")

# Sidebar: Agent selection
selected_agent = st.sidebar.selectbox("Select AI Agent", list(AGENTS.keys()))

# Input: User question
question = st.text_input("Ask a question:")

# Button: Submit question
if st.button("Submit"):
    if question.strip():
        with st.spinner(f"Querying {selected_agent}..."):
            response = call_backend(selected_agent, question)
        st.success("Response:")
        st.write(response)
    else:
        st.error("Please enter a question.")

# Footer
st.sidebar.markdown("### Backend Agents")
for agent, info in AGENTS.items():
    st.sidebar.write(f"- {agent}: {info['host']}:{info['port']}")
How It Works
Agent Selection:

A selectbox in the sidebar lets the user select which AI agent (backend) to query.
Input Question:

The user enters their question in the text input field.
Backend Call:

The app sends the question to the backend server of the selected agent using a POST request.
Each backend server is defined by a host and port.
Display Response:

The response from the backend server is displayed to the user.
Assumptions
Each backend server exposes an API endpoint (/answer) to handle questions and returns a JSON response like:
json
Copy
Edit
{
  "answer": "This is the response."
}
The backend servers are running on the specified host and port.
Run the App
Save the code in a file named app.py.
Install Streamlit and Requests:
bash
Copy
Edit
pip install streamlit requests
Run the app:
bash
Copy
Edit
streamlit run app.py
Would you like help setting up the backend servers or testing this setup?







You said:
Cool, if any agent's server is down then. It should show status in ui that server is not not reachable 
ChatGPT said:
ChatGPT
To handle the scenario where a backend server is down and show the server status in the UI, we can include a health check for each server. This can be done by periodically checking the status of each backend and displaying it in the sidebar. Here's an updated version of the code:

python
Copy
Edit
import streamlit as st
import requests

# Dictionary to map agents to their respective backend servers
AGENTS = {
    "Agent 1": {"host": "http://localhost", "port": 8001},
    "Agent 2": {"host": "http://localhost", "port": 8002},
    "Agent 3": {"host": "http://192.168.1.10", "port": 8003},
}

def check_server_status():
    """
    Check the status of all servers.

    Returns:
        dict: A dictionary with agent names as keys and their status ("reachable" or "unreachable").
    """
    statuses = {}
    for agent, server in AGENTS.items():
        url = f"{server['host']}:{server['port']}/health"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                statuses[agent] = "reachable"
            else:
                statuses[agent] = "unreachable"
        except requests.exceptions.RequestException:
            statuses[agent] = "unreachable"
    return statuses

def call_backend(agent, question):
    """
    Sends the question to the selected agent's backend server.

    Args:
        agent (str): The selected agent name.
        question (str): The user query.

    Returns:
        str: The response from the backend server.
    """
    server_info = AGENTS.get(agent)
    if not server_info:
        return "Invalid agent selected."
    
    url = f"{server_info['host']}:{server_info['port']}/answer"
    try:
        response = requests.post(url, json={"question": question}, timeout=10)
        response.raise_for_status()
        return response.json().get("answer", "No response found.")
    except requests.exceptions.RequestException as e:
        return f"Error connecting to {agent}: {e}"

# Streamlit UI
st.title("Multi-Agent AI Query App")

# Check the status of backend servers
server_statuses = check_server_status()

# Sidebar: Display server statuses
st.sidebar.markdown("### Backend Agents")
for agent, status in server_statuses.items():
    if status == "reachable":
        st.sidebar.success(f"{agent}: {status}")
    else:
        st.sidebar.error(f"{agent}: {status}")

# Input: User selects agent and asks question
selected_agent = st.sidebar.selectbox(
    "Select AI Agent", 
    [agent for agent in AGENTS.keys() if server_statuses[agent] == "reachable"],
    help="Only reachable agents are listed here."
)

question = st.text_input("Ask a question:")

# Submit button
if st.button("Submit"):
    if question.strip():
        with st.spinner(f"Querying {selected_agent}..."):
            response = call_backend(selected_agent, question)
        st.success("Response:")
        st.write(response)
    else:
        st.error("Please enter a question.")
How This Update Works
Server Health Check:

The check_server_status() function attempts a GET request to /health endpoints of each backend server.
Servers that respond with status code 200 are marked as reachable; otherwise, they are marked as unreachable.
Display Status in Sidebar:

Each agent's status is shown in the sidebar as either:
✅ Green (reachable)
❌ Red (unreachable)
Disable Unreachable Agents:

Only reachable agents are listed in the dropdown for querying.
Error Handling:

If the server is unreachable, the status is updated in real-time, and the user is prevented from selecting that agent.
Expected API Endpoints
/health: A simple endpoint on each backend server that returns a 200 OK response to indicate the server is up:

python
Copy
Edit
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200
/answer: Handles user questions and returns an AI-generated response.

Run the App
Run backend servers with /health and /answer endpoints.
Start the Streamlit app:
bash
Copy
Edit
streamlit run app.py
This approach ensures that users are only allowed to interact with live servers, and the status of each server is displayed in the UI. Let me know if you need help setting up backend servers!