import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
import wikipedia

@tool
def search_wikipedia(query: str) -> str:
    """Searches Wikipedia for a given query and returns a summary of the most relevant page.
    Use this when you need a concise, high-level summary of a historical event, scientific concept, or public figure.
    """
    try:
        # Search for matching pages
        search_results = wikipedia.search(query)
        if not search_results:
            return f"No Wikipedia pages found matching query: '{query}'"
        
        # Fetch the summary of the first match
        page_title = search_results[0]
        summary = wikipedia.summary(page_title, sentences=5)
        return f"Page Title: {page_title}\n\nSummary:\n{summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        # If there are multiple meanings, return options
        options = ", ".join(e.options[:5])
        return f"Query '{query}' is ambiguous. Did you mean one of these? {options}"
    except wikipedia.exceptions.PageError:
        return f"Wikipedia page not found for: '{query}'"
    except Exception as e:
        return f"An error occurred searching Wikipedia: {str(e)}"

@tool
def scrape_webpage(url: str) -> str:
    """Fetches a webpage from a URL and extracts its text content.
    Use this to read articles, documentation, or news pages when you have the direct link.
    """
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"
    
    try:
        # Make request with a standard User-Agent header
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
        response.raise_for_status()
        
        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style", "header", "footer", "nav"]):
            element.decompose()
        
        # Get cleaned text
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = "\n".join(lines[:100]) # Cap at first ~100 lines to avoid token overflow
        
        if len(cleaned_text) > 4000:
            cleaned_text = cleaned_text[:4000] + "\n... [Content Truncated]"
            
        return f"Webpage Title: {soup.title.string if soup.title else 'No Title'}\n\nContent:\n{cleaned_text}"
    except Exception as e:
        return f"An error occurred scraping the webpage: {str(e)}"

# Registry of tools for easy importing
all_tools = [search_wikipedia, scrape_webpage]

if __name__ == "__main__":
    print("Testing tools directly...")
    
    print("\n--- Wikipedia Tool ---")
    wiki_result = search_wikipedia.invoke({"query": "Artificial Intelligence"})
    print(wiki_result)
    
    print("\n--- Scraper Tool ---")
    scrape_result = scrape_webpage.invoke({"url": "https://example.com"})
    print(scrape_result)
