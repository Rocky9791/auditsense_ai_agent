
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment — never hardcode it

def claude_llm(prompt: str, model: str = "claude-sonnet-5") -> str:
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text