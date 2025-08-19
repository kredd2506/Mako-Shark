# llm_utils.py
import os
from openai import OpenAI

# Initialize the NRP LLM client with the temporary API key
client = OpenAI(
    api_key="NRP-API-key-here",  # Using the provided temporary key directly
    base_url="https://llm.nrp-nautilus.io/"
)

def get_llm_response(messages, model="gemma3"):
    """Get response from NRP LLM"""
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message.content

def get_embedding(text, model="embed-mistral"):
    """Get embedding from NRP embedding model"""
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

def summarize_content(content, max_length=500):
    """Summarize content using LLM"""
    prompt = f"""
    Summarize the following documentation content in under {max_length} characters.
    Focus on key concepts, main points, and technical details:
    
    {content[:4000]}  # Limit to avoid token overflow
    """
    messages = [
        {"role": "developer", "content": "You are a technical documentation expert."},
        {"role": "user", "content": prompt}
    ]
    return get_llm_response(messages)

def extract_key_concepts(content):
    """Extract key concepts from content using LLM"""
    prompt = """
    Extract the key technical concepts, terms, and technologies from the following documentation.
    Return as a comma-separated list:
    
    """ + content[:3000]
    
    messages = [
        {"role": "developer", "content": "You are a technical documentation expert."},
        {"role": "user", "content": prompt}
    ]
    concepts = get_llm_response(messages)
    return [c.strip() for c in concepts.split(",") if c.strip()]

def find_cross_reference(nrp_content, k8s_content):
    """Find relationships between NRP and Kubernetes content"""
    prompt = f"""
    Given the following documentation snippets:
    
    NRP Documentation:
    {nrp_content[:2000]}
    
    Kubernetes Documentation:
    {k8s_content[:2000]}
    
    Explain how these concepts relate to each other and what Kubernetes concepts would help a user better understand the NRP content.
    """
    messages = [
        {"role": "developer", "content": "You are an expert in cloud-native platforms and Kubernetes."},
        {"role": "user", "content": prompt}
    ]
    return get_llm_response(messages)