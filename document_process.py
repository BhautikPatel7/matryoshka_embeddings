import pandas as pd
import re
import json
from pathlib import Path

class DocumentProcessor:
    """
    Transforms raw StackOverflow Q&A into clean, searchable documents.
    Each document = Question Title + Body + Best Answer
    """

    def __init__(self, qa_df, tag_df):
        self.qa_df = qa_df
        self.tag_df = tag_df
        self.document = []


    def clean_html(self, text:str):
        """Remove HTML tags from StackOverflow posts"""
        if pd.isna(text):
             return ""
         
        text = re.sub(r'<[^>]+>', '', str(text))

        text = text.replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&amp;', '&').replace('&quot;', '"')
        text = text.replace('&#39;', "'")

        text = re.sub(r'\s+', ' ', text).strip()

        return text 
    
    def get_tags_for_question(self, queastion_id):
        """Get all tags associated with a question"""

        queastio_tag = self.tag_df[self.qa_df["Id"] == queastion_id]
        return list(queastio_tag["Tag"].values) if len(queastio_tag) > 0 else []
    
    def create_document(self, row):
        """"
        Create a single searchable document from Q&A pair.
        
        Returns dict with:
        - id: unique identifier
        - text: full searchable text (title + question + answer)
        - metadata: scores, tags, dates
        
        """

        title = self.clean_html(text=row["Title"])
        queastion_body = self.clean_html(text=row["Body_question"])
        answer_body = self.clean_html(text=row["Body_answer"])

        full_text = f"{title} \n\n {queastion_body} \n\n {answer_body}"

        tags = self.get_tags_for_question(queastion_id=row['Id_question'])

        doc = {
            'id': f"qa_{row['Id_question']}",
            'question_id': int(row['Id_question']),
            'title': title,
            'text': full_text,
            'question_score': int(row['Score_question']),
            'answer_score': int(row['Score_answer']),
            'tags': tags,
            'created_date': str(row['CreationDate_question']),
            # Track text lengths for analysis
            'text_length': len(full_text),
            'has_answer': True
        }

        return doc

    def process_all(self, max_docs = None):
        """
        Process all Q&A pairs into documents.
        
        Args:
            max_docs: Limit number of documents (for testing). None = all docs.
        """


        print(f"Processing {len(self.qa_df)} Q&A pairs...")
        df_to_process = self.qa_df.head(max_docs) if max_docs else self.qa_df


        for idx, row in df_to_process.iterrows():
            try:
                doc = self.create_document(row)
                self.document.append(doc)
                
                # Progress indicator
                if (idx + 1) % 10000 == 0:
                    print(f"  Processed {idx + 1} documents...")
                    
            except Exception as e:
                print(f"  Error processing row {idx}: {e}")
                continue
        
        print(f"✅ Created {len(self.document)} documents")
        return self.document

    def save_documents(self, output_path="data/processed_documents.jsonl"):
        """Save documents in JSONL format (one JSON object per line)"""
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in self.document:
                f.write(json.dumps(doc) + '\n')
        
        print(f"💾 Saved {len(self.document)} documents to {output_path}")  

    
    def get_statistics(self):
        """Print useful statistics about the processed dataset"""
        if not self.document:
            print("No documents processed yet!")
            return
        
        text_lengths = [doc['text_length'] for doc in self.document]
        question_scores = [doc['question_score'] for doc in self.document]
        
        print("Dataset Statistics:")
        print(f"  Total documents: {len(self.document):,}")
        print(f"  Avg text length: {sum(text_lengths)/len(text_lengths):.0f} chars")
        print(f"  Min text length: {min(text_lengths):,} chars")
        print(f"  Max text length: {max(text_lengths):,} chars")
        print(f"  Avg question score: {sum(question_scores)/len(question_scores):.1f}")
        
        # Tag statistics
        all_tags = [tag for doc in self.document for tag in doc['tags']]
        unique_tags = set(all_tags)
        print(f"  Unique tags: {len(unique_tags)}")
        print(f"  Total tag associations: {len(all_tags)}")