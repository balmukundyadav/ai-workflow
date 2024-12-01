
To create a Python program that fetches source code from a GitLab project repository organized in different directories and subdirectories, and sets up a Retrieval-Augmented Generation (RAG) pipeline using LangChain, we will follow a few key steps:

Steps for the program:
Connect to GitLab API: Use GitLab's REST API to fetch all files from a project repository, including those in nested directories and subdirectories.
Store the source code: Save the fetched files locally or in a structure that LangChain can process.
Set up LangChain: Configure a LangChain pipeline to interact with the retrieved code, where you can query specific files or sections of code for further processing.
Retrieval-Augmented Generation (RAG): Use a vector store for retrieval and integrate a generative model like GPT to help generate answers or code suggestions based on the retrieved documents.
Prerequisites:
GitLab API Token: You will need a GitLab Personal Access Token (PAT) to authenticate and fetch the repository.
LangChain: Install LangChain for setting up the RAG pipeline.
OpenAI: You will also need OpenAI’s API (if you're using models like GPT-3/4 or Codex).
Requests: To interact with GitLab API.
FAISS/Chroma: For setting up the vector store for retrieval.
Python libraries: requests, langchain, openai, faiss (for vector search), os.
Installation
First, install the necessary libraries:

bash
Copy code
pip install requests langchain openai faiss-cpu
Python Code
Here is a Python script that fetches GitLab project repository source code, stores it, and sets up the RAG pipeline with LangChain.

python
Copy code
import os
import requests
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# GitLab API authentication details
GITLAB_API_URL = "https://gitlab.com/api/v4"
REPO_URL = "https://gitlab.com/your-org/your-repo"  # Replace with your GitLab repo URL
PRIVATE_TOKEN = "your-private-token"  # Replace with your GitLab Personal Access Token
PROJECT_ID = "your-project-id"  # Replace with your GitLab project ID

# Function to get all files from GitLab repo (including nested files)
def get_files_from_gitlab(project_id, token):
    headers = {'Private-Token': token}
    api_url = f"{GITLAB_API_URL}/projects/{project_id}/repository/tree?recursive=true"
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching repository: {response.content}")
    
    files = response.json()
    return files

# Function to download file content from GitLab
def get_file_content_from_gitlab(project_id, file_path, token):
    headers = {'Private-Token': token}
    api_url = f"{GITLAB_API_URL}/projects/{project_id}/repository/files/{file_path}/raw?ref=main"
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching file {file_path}: {response.content}")
    
    return response.text

# Fetch all files and store them locally in a directory
def fetch_and_store_files(files, project_id, token, download_dir='repo_files'):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    for file in files:
        file_path = file['path']
        if file['type'] == 'blob':  # Only download files, not directories
            print(f"Fetching file: {file_path}")
            content = get_file_content_from_gitlab(project_id, file_path, token)
            local_path = os.path.join(download_dir, file_path)

            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)

# Fetch all files from the GitLab repository
files = get_files_from_gitlab(PROJECT_ID, PRIVATE_TOKEN)
fetch_and_store_files(files, PROJECT_ID, PRIVATE_TOKEN)

# Setting up the LangChain pipeline (RAG)
def setup_rag_pipeline(repo_dir):
    # Load all files from the local directory into LangChain
    documents = []
    for root, dirs, files in os.walk(repo_dir):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({"content": content, "metadata": {"source": file_path}})

    # Set up the OpenAI embedding model
    embeddings = OpenAIEmbeddings()

    # Store the documents in a vector store (Chroma/FAISS)
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Set up the retrieval-based QA pipeline
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",  # You can use other chain types depending on your use case
        retriever=vectorstore.as_retriever()
    )

    return qa_chain

# Run the RAG pipeline
rag_chain = setup_rag_pipeline('repo_files')

# Now you can ask questions based on the code
def ask_question_to_rag_pipeline(question):
    result = rag_chain.run(question)
    return result

# Example usage: ask the pipeline a question
question = "What does the function 'parse_error_logs' do in the code?"
answer = ask_question_to_rag_pipeline(question)
print(f"Answer: {answer}")
Explanation:
GitLab Integration:

get_files_from_gitlab(): Fetches a list of all files (and directories) in the GitLab repository using the recursive=true query parameter to get all files and subdirectories.
get_file_content_from_gitlab(): Downloads the raw content of a given file using GitLab's API.
fetch_and_store_files(): Downloads all files and stores them locally in a structured directory (repo_files).
LangChain Setup (RAG Pipeline):

setup_rag_pipeline():
Reads all stored files from the repo_files directory.
Creates a list of documents, where each document represents a file in the repo.
Uses OpenAI's embeddings to convert the code into a vector format for retrieval.
Stores the vectors in Chroma (or FAISS) as a vector store.
Sets up a RetrievalQA chain, which is the RAG setup, where LangChain will use the retriever to search relevant documents (code) and generate answers using an LLM (e.g., GPT).
Question Answering:

ask_question_to_rag_pipeline(): Takes a question related to the code (e.g., function explanation, bug fix suggestions) and returns an answer based on the repository content using the RAG pipeline.
How It Works:
This script fetches source code files from a GitLab repository.
It then stores the source code locally in a folder structure similar to the GitLab repository.
The LangChain RAG pipeline is set up by loading the source code files, embedding them, storing the embeddings in a vector store (Chroma/FAISS), and using OpenAI’s LLM for answering queries related to the source code.
Example Query:
After running the pipeline, you can ask questions like:

“What does the function parse_error_logs do?”
“How can I fix the FileNotFoundError in this repository?”
The pipeline will search the codebase, find the relevant sections, and return an answer using the LLM.

Notes:
Ensure that you replace placeholders like PRIVATE_TOKEN and PROJECT_ID with actual values for your GitLab repository.
The retrieval-based QA (RAG) pipeline can be customized with different models or embeddings.
This code is using Chroma as the vector store, but you can also use FAISS if you prefer.


import os
import subprocess
import requests

def clone_gitlab_repo(gitlab_url, personal_access_token, project_id, destination_path):
    """
    Clones a GitLab project repository using the API to fetch the repository URL.

    :param gitlab_url: Base URL of the GitLab instance (e.g., 'https://gitlab.com').
    :param personal_access_token: Your GitLab personal access token.
    :param project_id: ID or path of the GitLab project.
    :param destination_path: Path to clone the repository into.
    """
    headers = {
        "Authorization": f"Bearer {personal_access_token}"
    }
    api_url = f"{gitlab_url}/api/v4/projects/{project_id}"

    try:
        # Get the project details
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        project_data = response.json()

        # Get the HTTPS clone URL
        clone_url = project_data.get("http_url_to_repo")
        if not clone_url:
            raise ValueError("Unable to retrieve the repository URL.")

        # Clone the repository using `git`
        print(f"Cloning repository from {clone_url} into {destination_path}...")
        subprocess.run(["git", "clone", clone_url, destination_path], check=True)
        print("Repository cloned successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error accessing GitLab API: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error while cloning repository: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
if __name__ == "__main__":
    # Replace these with your GitLab details
    GITLAB_URL = "https://gitlab.com"
    PERSONAL_ACCESS_TOKEN = "your_personal_access_token"
    PROJECT_ID = "your_project_id_or_path"  # e.g., "username/project-name" or project numeric ID
    DESTINATION_PATH = "path_to_clone_repo"

    clone_gitlab_repo(GITLAB_URL, PERSONAL_ACCESS_TOKEN, PROJECT_ID, DESTINATION_PATH)



import os
import requests

def download_gitlab_repo_archive(gitlab_url, personal_access_token, project_id, destination_path):
    """
    Downloads a GitLab project repository archive using the API.

    :param gitlab_url: Base URL of the GitLab instance (e.g., 'https://gitlab.com').
    :param personal_access_token: Your GitLab personal access token.
    :param project_id: ID or path of the GitLab project.
    :param destination_path: Path to save the downloaded archive.
    """
    headers = {
        "Authorization": f"Bearer {personal_access_token}"
    }
    api_url = f"{gitlab_url}/api/v4/projects/{project_id}/repository/archive"

    try:
        # Send the GET request to download the archive
        print(f"Downloading repository archive from {api_url}...")
        response = requests.get(api_url, headers=headers, stream=True)
        response.raise_for_status()

        # Save the archive to the specified path
        with open(destination_path, "wb") as archive_file:
            for chunk in response.iter_content(chunk_size=8192):
                archive_file.write(chunk)

        print(f"Repository archive saved to {destination_path}.")

    except requests.exceptions.RequestException as e:
        print(f"Error accessing GitLab API: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
if __name__ == "__main__":
    # Replace these with your GitLab details
    GITLAB_URL = "https://gitlab.com"
    PERSONAL_ACCESS_TOKEN = "your_personal_access_token"
    PROJECT_ID = "your_project_id_or_path"  # e.g., "username/project-name" or project numeric ID
    DESTINATION_PATH = "path_to_save_archive.zip"

    download_gitlab_repo_archive(GITLAB_URL, PERSONAL_ACCESS_TOKEN, PROJECT_ID, DESTINATION_PATH)
