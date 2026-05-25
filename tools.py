import wikipedia
from langchain_core.tools import tool


@tool
def search_wikipedia(query: str, sentences: int = 3) -> str:
    """
    Search Wikipedia for information about a topic.
    
    Args:
        query: Search query for Wikipedia
        sentences: Number of sentences to include in summary (default: 3)
        
    Returns:
        Formatted string with Wikipedia summary
    """
    try:
        # Search for the page
        page = wikipedia.page(query, auto_suggest=False)
        
        # Get summary
        summary = wikipedia.summary(query, sentences=sentences, auto_suggest=False)
        
        result = f"""
Title: {page.title}
URL: {page.url}
Summary: {summary}
"""
        return result
    
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Query '{query}' is ambiguous. Possible options: {', '.join(e.options[:5])}"
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


# List of available tools
tools = [search_wikipedia]
