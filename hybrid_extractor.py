import re
import sys
import os
import spacy
from typing import List, Set

class HybridExtractor:
    def __init__(self):
        # Load spaCy model - free and built-in
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # If model not downloaded, guide user to install it
            print("Please install the spaCy model first:")
            print("python -m spacy download en_core_web_sm")
            sys.exit(1)
            
        # Company and organization names to exclude
        self.company_names = [
            "palo alto networks", "google cloud", "fortinet", "google", "openai", 
            "microsoft", "amazon", "adaptive security", "cyber magazine", 
            "forward networks", "digital twins"
        ]
        
        # Month names for date filtering
        self.months = [
            "january", "february", "march", "april", "may", "june", "july", 
            "august", "september", "october", "november", "december",
            "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
        ]
        
    def extract(self, text: str) -> List[str]:
        """Extract authors using a hybrid approach combining NLP and pattern matching"""
        # Clean the text
        text = self._clean_text(text)
        
        # Determine document type
        doc_type = self._detect_document_type(text)
        
        # Use different strategies based on document type
        if doc_type == "academic":
            return self._extract_academic_authors(text)
        else:
            return self._extract_news_authors(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean the text for better processing"""
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Normalize line endings
        text = re.sub(r'\r\n', '\n', text)
        return text
    
    def _detect_document_type(self, text: str) -> str:
        """Detect if the document is academic or news"""
        lower_text = text.lower()
        academic_indicators = ["abstract", "keywords", "author links", "doi:", "creative commons"]
        
        if sum(1 for indicator in academic_indicators if indicator in lower_text) >= 2:
            return "academic"
        else:
            return "news"
    
    def _extract_academic_authors(self, text: str) -> List[str]:
        """Extract authors from academic papers"""
        authors = []
        
        # Look for author section in academic papers
        author_match = re.search(r"Author(?:s)?\s+links.*?\n(.*?)(?:Show more|Abstract)", text, re.DOTALL | re.IGNORECASE)
        
        if author_match:
            author_section = author_match.group(1).strip()
            
            # Process comma-separated authors with affiliations
            if ',' in author_section:
                parts = [p.strip() for p in author_section.split(',')]
                for part in parts:
                    # Remove affiliation markers
                    author_name = re.sub(r'\s+[a-z](\s|$)', ' ', part).strip()
                    # Match properly formatted names
                    name_match = re.match(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', author_name)
                    if name_match and name_match.group(1) not in authors:
                        authors.append(name_match.group(1))
            else:
                # Use NER for single author or complex format
                doc = self.nlp(author_section)
                for ent in doc.ents:
                    if ent.label_ == "PERSON" and ent.text not in authors:
                        authors.append(ent.text)
        
        # If no authors found by pattern matching, try NER on the first part of the document
        if not authors:
            # Process first 1000 chars with NER
            doc = self.nlp(text[:1000])
            for ent in doc.ents:
                if ent.label_ == "PERSON" and self._is_valid_name(ent.text) and ent.text not in authors:
                    authors.append(ent.text)
        
        return self._post_process_authors(authors)
    
    def _extract_news_authors(self, text: str) -> List[str]:
        """Extract authors from news articles"""
        authors = []
        
        # Strategy 1: Look for byline in first few lines
        lines = text.split('\n')
        clean_lines = [line.strip() for line in lines if line.strip()]
        
        for i, line in enumerate(clean_lines[:10]):
            # Check for "By Author Name" pattern
            if line.startswith("By ") and len(line) < 50:
                byline = line[3:].strip()
                # Remove date information
                for month in self.months:
                    byline = re.sub(f"\\b{month}\\b.*$", "", byline, flags=re.IGNORECASE)
                
                if self._is_valid_name(byline) and byline not in authors:
                    authors.append(byline)
                    break
            
            # Check for standalone author name
            elif 2 <= len(line.split()) <= 3 and i > 0:
                if all(word[0].isupper() for word in line.split() if len(word) > 1):
                    if self._is_valid_name(line) and line not in authors:
                        authors.append(line)
                        break
        
        # Strategy 2: Use NER to find people mentioned in key positions
        if not authors:
            first_1000 = text[:1000]
            doc = self.nlp(first_1000)
            
            candidates = []
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    candidates.append((ent.text, ent.start_char))
            
            # Sort by position (names that appear earlier are more likely to be authors)
            candidates.sort(key=lambda x: x[1])
            
            # Take the first person mentioned as the likely author
            for candidate, _ in candidates[:3]:
                if self._is_valid_name(candidate) and candidate not in authors:
                    authors.append(candidate)
        
        # Strategy 3: Look for specific patterns for names with prefixes
        if not authors:
            special_patterns = [
                r"([A-Z][a-z]+\s+Mc[A-Z][a-z]+)",  # Dave McGrail
                r"([A-Z][a-z]+\s+Mac[A-Z][a-z]+)"  # John MacKenzie
            ]
            
            for pattern in special_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if self._is_valid_name(match) and match not in authors:
                        authors.append(match)
                        if len(authors) >= 3:
                            break
        
        return self._post_process_authors(authors)
    
    def _is_valid_name(self, name: str) -> bool:
        """Check if a string is likely a valid person name"""
        if not name:
            return False
        
        # Remove trailing punctuation
        name = name.rstrip('.,;:')
        
        # Remove month names that might be dates
        for month in self.months:
            name = re.sub(f"\\s+{month}\\b.*$", "", name, flags=re.IGNORECASE)
        
        # Remove affiliation markers
        name = re.sub(r'\s+[a-z]$', '', name)
        
        # Basic validation
        words = name.split()
        
        # Name must have at least 2 words, but not too many
        if len(words) < 2 or len(words) > 4:
            return False
        
        # All words should be properly capitalized
        if not all(word[0].isupper() for word in words if len(word) > 1):
            return False
        
        # Words should not be too short
        if not all(len(word) > 1 for word in words):
            return False
        
        # Check against company name list
        if any(company in name.lower() for company in self.company_names):
            return False
        
        return True
    
    def _post_process_authors(self, authors: List[str]) -> List[str]:
        """Clean up and finalize author list"""
        seen = set()
        clean_authors = []
        
        for author in authors:
            # Final cleanup
            clean_author = author.strip()
            
            # Remove month names
            for month in self.months:
                clean_author = re.sub(f"\\s+{month}\\b.*$", "", clean_author, flags=re.IGNORECASE)
            
            # Remove affiliation letters
            clean_author = re.sub(r'\s+[a-z]$', '', clean_author)
            
            # Deduplicate
            if clean_author.lower() not in seen and self._is_valid_name(clean_author):
                seen.add(clean_author.lower())
                clean_authors.append(clean_author)
        
        return clean_authors[:5]  # Limit to 5 authors

def main():
    if len(sys.argv) < 2:
        print("Usage: python hybrid_extractor.py <path_to_file.txt>")
        return
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    extractor = HybridExtractor()
    authors = extractor.extract(text)
    
    print("Extracted authors:")
    if authors:
        for i, author in enumerate(authors, 1):
            print(f"{i}. {author}")
    else:
        print("No authors found")

if __name__ == "__main__":
    main()