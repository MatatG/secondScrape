from flask import Flask, render_template, request, jsonify, Response
import anthropic
import requests
from bs4 import BeautifulSoup
import json
import time
import csv
from io import StringIO
import tiktoken
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from datetime import datetime

app = Flask(__name__)

def get_webpage_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 429:
            print("Rate limited. Waiting 60 seconds...")
            time.sleep(60)
            response = requests.get(url)
        response.raise_for_status()
        return clean_html_content(response.text)
    except Exception as e:
        print(f"Error fetching webpage: {e}")
        return None

def clean_html_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove unnecessary elements
    for element in soup(["script", "style", "head", "iframe"]):
        element.decompose()
    
    # Get main content and navigation elements
    main_content = soup.find('main') or soup.find('div', {'role': 'main'}) or soup
    nav_elements = soup.find_all(['nav', 'a'])
    
    # Extract navigation links and their text
    navigation_info = []
    for link in nav_elements:
        if link.name == 'a' and link.get('href'):
            href = link.get('href')
            text = link.get_text(strip=True)
            if text and href and not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                navigation_info.append(f"{text}: {href}")
    
    # Get main content text
    content_text = main_content.get_text(separator=' ', strip=True)
    
    # Combine navigation and content with clear separation
    final_text = "NAVIGATION OPTIONS:\n" + "\n".join(navigation_info[:20]) + "\n\nPAGE CONTENT:\n" + content_text
    
    return final_text[:15000]  # Still maintain token limit

# Initialize tokenizer for Claude
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")  # Use Claude's encoding
    num_tokens = len(encoding.encode(string))
    return num_tokens

class TokenTracker:
    def __init__(self):
        # Current rates per 1K tokens (as of 2024)
        self.rates = {
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},  # $3.00 / $15.00 per MTok
            "claude-3-5-haiku-20241022": {"input": 0.001, "output": 0.005},   # $1.00 / $5.00 per MTok
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},      # $15.00 / $75.00 per MTok
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},    # $3.00 / $15.00 per MTok
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125}  # $0.25 / $1.25 per MTok
        }
        self.reset()
    
    def reset(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.requests = 0
        self.model_used = None
    
    def add_request(self, prompt: str, response: str, model: str):
        input_tokens = num_tokens_from_string(prompt)
        output_tokens = num_tokens_from_string(response)
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.requests += 1
        self.model_used = model
    
    def calculate_cost(self):
        if not self.model_used or self.model_used not in self.rates:
            return 0.0
        
        rate = self.rates[self.model_used]
        input_cost = (self.total_input_tokens / 1000) * rate["input"]
        output_cost = (self.total_output_tokens / 1000) * rate["output"]
        
        return input_cost + output_cost
    
    def get_stats(self):
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "requests": self.requests,
            "cost": round(self.calculate_cost(), 4),
            "model": self.model_used
        }

# Create a global token tracker
token_tracker = TokenTracker()

def analyze_with_claude(html_content, url, context=""):
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    model = "claude-3-5-haiku-20241022"  # Specify the model explicitly
    
    system_prompt = """You are a focused web scraping assistant. You work for a company making a list of all accommodations world wide. Your goal is to find hotel room names and details.
    If you see room information on the current page, extract it. If not, suggest where to look next on the site.
    Keep responses brief and direct. Make sure to extract ALL accommodation types (rooms, suites, apartments, etc.)
    Return hotel name, room name."""

    prompt = f"""Looking at {url}

    Should I extract room info from this page or navigate elsewhere? If navigation needed, provide the specific URL to check.
    
    Page content:
    {html_content}"""

    message = client.messages.create(
        model=model,
        max_tokens=4000,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    # Track tokens for this request
    token_tracker.add_request(prompt, message.content[0].text, model)
    
    return {
        "analysis": message.content[0].text,
        "thinking": f"Analyzing page at {url}\nLooking for room information or navigation links..."
    }

def structure_room_data(analysis_text):
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    model = "claude-3-5-haiku-20241022"
    
    system_prompt = """You are a data structuring assistant. Convert the hotel room information into a CSV format with the following fields:
    hotel_name,room_name.
    
    If any field is not available, use 'N/A'. Keep the data clean and consistent.
    Only respond with the CSV data, no additional text."""
    
    prompt = f"""Please convert this room information into structured CSV data:
    
    {analysis_text}"""

    message = client.messages.create(
        model=model,
        max_tokens=4000,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    # Track tokens for this request
    token_tracker.add_request(prompt, message.content[0].text, model)
    
    return {
        "data": message.content[0].text,
        "thinking": "Converting unstructured room data to CSV format..."
    }

def scrape_rooms(initial_url):
    visited_urls = set()
    rooms_data = []
    urls_to_visit = [initial_url]
    logs = []
    thinking_steps = []
    structured_results = []
    
    while urls_to_visit and len(visited_urls) < 10:
        current_url = urls_to_visit.pop(0)
        
        if current_url in visited_urls:
            continue
            
        logs.append(f"Checking: {current_url}")
        visited_urls.add(current_url)
        
        html_content = get_webpage_content(current_url)
        if not html_content:
            logs.append("Failed to fetch page content")
            continue
        
        analysis_result = analyze_with_claude(html_content, current_url)
        thinking_steps.append(analysis_result["thinking"])
        logs.append(f"Analysis: {analysis_result['analysis']}")
        
        if "http" not in analysis_result["analysis"].lower():
            structure_result = structure_room_data(analysis_result["analysis"])
            thinking_steps.append(structure_result["thinking"])
            logs.append(f"Structured Data:\n{structure_result['data']}")
            structured_results.append(structure_result["data"])
        
        if "http" in analysis_result["analysis"].lower():
            new_urls = [url for url in analysis_result["analysis"].split() if url.startswith("http")]
            urls_to_visit.extend(new_urls)
        
        time.sleep(1)
    
    return {
        "logs": logs,
        "thinking": thinking_steps,
        "results": structured_results
    }

def process_hotel_batch(urls):
    # Reset token tracker at the start of batch processing
    token_tracker.reset()
    
    all_results = []
    all_logs = []
    all_thinking = []
    
    for url in urls:
        url = url.strip()
        if not url:
            continue
            
        all_logs.append(f"\n=== Processing hotel: {url} ===")
        
        try:
            results = scrape_rooms(url)
            all_results.extend(results["results"])
            all_logs.extend(results["logs"])
            all_thinking.extend(results["thinking"])
        except Exception as e:
            all_logs.append(f"Error processing {url}: {str(e)}")
        
        # Add delay between hotels to avoid rate limiting
        time.sleep(2)
    
    return {
        "logs": all_logs,
        "thinking": all_thinking,
        "results": all_results,
        "token_stats": token_tracker.get_stats()  # Add token statistics to batch results
    }



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    # Check content type to determine how to handle the request
    content_type = request.content_type or ''
    
    if 'multipart/form-data' in content_type:
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename.endswith('.csv'):
                stream = StringIO(file.stream.read().decode("UTF8"), newline=None)
                urls = []
                csv_reader = csv.reader(stream)
                for row in csv_reader:
                    if row and row[0].strip():
                        urls.append(row[0].strip())
                
                if not urls:
                    return jsonify({"error": "No valid URLs found in CSV"}), 400
                    
                results = process_hotel_batch(urls)
                return jsonify(results)  # Now includes token_stats
            else:
                return jsonify({"error": "Please upload a valid CSV file"}), 400
    else:
        # Handle single URL submission
        url = None
        if request.is_json:
            url = request.json.get('url')
        else:
            url = request.form.get('url')
            
        if not url:
            return jsonify({"error": "URL is required"}), 400
        
        # Reset token tracker for single URL processing
        token_tracker.reset()
        results = scrape_rooms(url)
        results["token_stats"] = token_tracker.get_stats()
        
        return jsonify(results)

@app.route('/download-csv')
def download_csv():
    results = request.args.get('data', '')
    
    si = StringIO()
    cw = csv.writer(si)
    
    # Write header row if not present
    first_row = next(csv.reader(StringIO(results)), None)
    if first_row and 'hotel_name' not in first_row[0].lower():
        cw.writerow(['hotel_name', 'room_name'])
    
    # Write data rows
    for row in csv.reader(StringIO(results)):
        if row:  # Skip empty rows
            cw.writerow(row)
    
    output = si.getvalue()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=hotel_rooms_{timestamp}.csv"}
    )

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
