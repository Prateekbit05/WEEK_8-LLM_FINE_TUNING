import pandas as pd
import json
import jsonlines
import re
import random
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import os

class InstructionDatasetBuilder:
    """
    Build instruction tuning dataset from customers and people CSV files
    Includes: QA, Reasoning, Extraction tasks
    """
    
    def __init__(self, customers_path: str, people_path: str):
        print("📂 Loading datasets...")
        
        # Check if files exist
        if not os.path.exists(customers_path):
            raise FileNotFoundError(f"❌ File not found: {customers_path}")
        if not os.path.exists(people_path):
            raise FileNotFoundError(f"❌ File not found: {people_path}")
        
        self.customers = pd.read_csv(customers_path)
        self.people = pd.read_csv(people_path)
        
        # Clean data
        self.customers = self.customers.dropna()
        self.people = self.people.dropna()
        
        print(f"✅ Loaded {len(self.customers)} customers")
        print(f"✅ Loaded {len(self.people)} people")
        
        # Show column names
        print(f"\n📋 Customer columns: {list(self.customers.columns)}")
        print(f"📋 People columns: {list(self.people.columns)}")
        
        self.instructions = []
    
    # ========== QA TEMPLATES (ENHANCED WITH DIVERSITY) ==========
    def generate_qa_samples(self, n_samples: int = 500):
        """Generate Question-Answer pairs with diverse starters"""
        print("\n🔹 Generating QA samples...")
        
        samples = []
        
        # ✅ DIVERSE QUESTION TEMPLATES
        question_starters = [
            ("What is", "is"),
            ("Can you tell me", "is"),
            ("I need to know", "is"),
            ("Please provide", "is"),
            ("Could you share", "is"),
            ("Who is", "is"),
            ("Where is", "is located in"),
            ("How can I contact", "can be contacted at"),
            ("Which company does", "works for"),
            ("Do you have", "Yes, it is"),
        ]
        
        for _ in tqdm(range(n_samples)):
            # Randomly choose from customers or people
            if random.random() > 0.5:
                row = self.customers.sample(1).iloc[0]
            else:
                row = self.people.sample(1).iloc[0]
            
            cols = row.index.tolist()
            name_col = [col for col in cols if 'name' in col.lower()][0] if any('name' in col.lower() for col in cols) else cols[0]
            
            # Choose random question type
            qa_type = random.choice(['email', 'phone', 'location', 'company', 'age', 'job'])
            
            # Select diverse starter
            starter, verb = random.choice(question_starters)
            
            if qa_type == 'email' and any('email' in col.lower() for col in cols):
                email_col = [col for col in cols if 'email' in col.lower()][0]
                
                q = f"{starter} the email for {row[name_col]}?"
                a = f"The email {verb} {row[email_col]}."
            
            elif qa_type == 'phone' and any('phone' in col.lower() for col in cols):
                phone_col = [col for col in cols if 'phone' in col.lower()][0]
                
                q = f"{starter} {row[name_col]}'s phone number?"
                a = f"The phone number {verb} {row[phone_col]}."
            
            elif qa_type == 'location':
                city_col = [col for col in cols if 'city' in col.lower()][0] if any('city' in col.lower() for col in cols) else None
                country_col = [col for col in cols if 'country' in col.lower()][0] if any('country' in col.lower() for col in cols) else None
                
                if city_col and country_col:
                    q = f"{starter} where {row[name_col]} is based?"
                    a = f"{row[name_col]} {verb} {row[city_col]}, {row[country_col]}."
                else:
                    continue
            
            elif qa_type == 'company' and any('company' in col.lower() for col in cols):
                company_col = [col for col in cols if 'company' in col.lower()][0]
                
                q = f"{starter} which organization {row[name_col]} represents?"
                a = f"{row[name_col]} represents {row[company_col]}."
            
            elif qa_type == 'age' and any('age' in col.lower() for col in cols):
                age_col = [col for col in cols if 'age' in col.lower()][0]
                
                q = f"{starter} the age of {row[name_col]}?"
                a = f"{row[name_col]} {verb} {row[age_col]} years old."
            
            elif qa_type == 'job' and any(col for col in cols if 'job' in col.lower() or 'title' in col.lower()):
                job_col = [col for col in cols if 'job' in col.lower() or 'title' in col.lower()][0]
                
                q = f"{starter} {row[name_col]}'s role?"
                a = f"Their role {verb} {row[job_col]}."
            
            else:
                continue
            
            samples.append({
                "instruction": q,
                "input": "",
                "output": a
            })
        
        self.instructions.extend(samples)
        print(f"✅ Generated {len(samples)} QA samples")
    
    # ========== REASONING TEMPLATES ==========
    def generate_reasoning_samples(self, n_samples: int = 500):
        """Generate reasoning tasks"""
        print("\n🔹 Generating Reasoning samples...")
        
        samples = []
        
        for _ in tqdm(range(n_samples)):
            row = self.customers.sample(1).iloc[0] if random.random() > 0.5 else self.people.sample(1).iloc[0]
            cols = row.index.tolist()
            
            reasoning_type = random.choice([
                "age_comparison",
                "profile_analysis",
                "data_inference",
                "demographic_insight",
                "business_context",
                "location_analysis"
            ])
            
            name_col = [col for col in cols if 'name' in col.lower()][0] if any('name' in col.lower() for col in cols) else cols[0]
            
            if reasoning_type == "age_comparison" and any('age' in col.lower() for col in cols):
                row2 = self.people.sample(1).iloc[0] if any('age' in col.lower() for col in self.people.columns) else row
                age_col = [col for col in cols if 'age' in col.lower()][0]
                
                age1 = int(row[age_col]) if pd.notna(row[age_col]) else 30
                age2 = int(row2.get(age_col, 30)) if pd.notna(row2.get(age_col, 30)) else 30
                
                instruction = f"Compare the ages of {row[name_col]} and {row2.get(name_col, 'another person')}. Who is older and by how much?"
                
                if age1 > age2:
                    output = f"{row[name_col]} is older. {row[name_col]} is {age1} years old while {row2.get(name_col, 'the other person')} is {age2} years old, making {row[name_col]} {age1-age2} years older."
                elif age2 > age1:
                    output = f"{row2.get(name_col, 'The other person')} is older. They are {age2} years old while {row[name_col]} is {age1} years old, making them {age2-age1} years older."
                else:
                    output = f"Both {row[name_col]} and {row2.get(name_col, 'the other person')} are the same age: {age1} years old."
            
            elif reasoning_type == "profile_analysis":
                instruction = f"Analyze the profile of {row[name_col]} and provide business insights."
                
                insights = []
                if any('age' in col.lower() for col in cols):
                    age_col = [col for col in cols if 'age' in col.lower()][0]
                    age = int(row[age_col]) if pd.notna(row[age_col]) else 30
                    if age < 30:
                        insights.append(f"At {age} years old, they are likely in the early career stage, potentially tech-savvy and open to innovative solutions")
                    elif age < 50:
                        insights.append(f"At {age} years old, they are likely in mid-career with decision-making authority")
                    else:
                        insights.append(f"At {age} years old, they are likely a senior professional with extensive experience")
                
                if any('city' in col.lower() for col in cols):
                    city_col = [col for col in cols if 'city' in col.lower()][0]
                    insights.append(f"Located in {row[city_col]}, which affects timezone considerations and regional market dynamics")
                
                if any('company' in col.lower() for col in cols):
                    company_col = [col for col in cols if 'company' in col.lower()][0]
                    insights.append(f"Association with {row[company_col]} suggests B2B engagement opportunities")
                
                output = f"Profile analysis for {row[name_col]}: " + ". ".join(insights) + "."
            
            elif reasoning_type == "data_inference":
                instruction = f"What can you infer about {row[name_col]}'s professional background from the available data?"
                output = f"Based on the data, {row[name_col]} appears to be a professional contact. The information suggests they are suitable for B2B engagement, professional networking opportunities, and targeted business development initiatives. Their profile indicates potential for strategic partnerships."
            
            elif reasoning_type == "demographic_insight":
                instruction = f"Provide demographic insights and marketing recommendations for {row[name_col]}."
                
                if any('age' in col.lower() for col in cols):
                    age_col = [col for col in cols if 'age' in col.lower()][0]
                    age = int(row[age_col]) if pd.notna(row[age_col]) else 30
                    
                    if age < 25:
                        segment = "Gen Z"
                        traits = "digitally native, values authenticity, prefers mobile-first experiences, socially conscious"
                        marketing = "Use social media, influencer partnerships, and authentic storytelling"
                    elif age < 40:
                        segment = "Millennial"
                        traits = "tech-savvy, experience-focused, values work-life balance, environmentally aware"
                        marketing = "Focus on experiential marketing, sustainability messaging, and digital channels"
                    elif age < 60:
                        segment = "Gen X"
                        traits = "pragmatic, brand loyal, values quality and reliability"
                        marketing = "Emphasize quality, ROI, and long-term value propositions"
                    else:
                        segment = "Baby Boomer"
                        traits = "values quality and service, prefers established brands, relationship-focused"
                        marketing = "Prioritize personalized service, proven track records, and relationship building"
                    
                    output = f"{row[name_col]} falls into the {segment} demographic ({age} years old). This group is typically {traits}. Marketing approach: {marketing}."
                else:
                    output = f"{row[name_col]} is a professional contact in our database suitable for targeted B2B engagement."
            
            elif reasoning_type == "business_context":
                instruction = f"What is the recommended business engagement strategy for {row[name_col]}?"
                
                strategies = []
                if any('company' in col.lower() for col in cols):
                    company_col = [col for col in cols if 'company' in col.lower()][0]
                    strategies.append(f"Account-based approach targeting {row[company_col]}")
                
                strategies.append("Professional B2B engagement")
                strategies.append("Personalized communication based on profile")
                strategies.append("Value-driven messaging focused on ROI and efficiency")
                
                output = f"For {row[name_col]}, recommended strategy: " + ", ".join(strategies) + "."
            
            else:  # location_analysis
                instruction = f"Analyze the geographical and market implications of {row[name_col]}'s location."
                
                if any('city' in col.lower() for col in cols) and any('country' in col.lower() for col in cols):
                    city_col = [col for col in cols if 'city' in col.lower()][0]
                    country_col = [col for col in cols if 'country' in col.lower()][0]
                    
                    output = f"{row[name_col]} is located in {row[city_col]}, {row[country_col]}. This location implies: timezone considerations for optimal communication timing, potential shipping and logistics requirements, regional market preferences and cultural nuances, and localized marketing strategies. Consider these factors when planning engagement."
                else:
                    output = f"{row[name_col]}'s location should be considered for timezone-appropriate communication and regional market strategies."
            
            samples.append({
                "instruction": instruction,
                "input": "",
                "output": output
            })
        
        self.instructions.extend(samples)
        print(f"✅ Generated {len(samples)} Reasoning samples")
    
    # ========== EXTRACTION TEMPLATES ==========
    def generate_extraction_samples(self, n_samples: int = 500):
        """Generate extraction tasks"""
        print("\n🔹 Generating Extraction samples...")
        
        samples = []
        
        for _ in tqdm(range(n_samples)):
            row = self.customers.sample(1).iloc[0] if random.random() > 0.5 else self.people.sample(1).iloc[0]
            cols = row.index.tolist()
            
            choice = random.choice(['profile_extraction', 'contact_info', 'structured_conversion', 'entity_list', 'field_extraction'])
            
            name_col = [col for col in cols if 'name' in col.lower()][0] if any('name' in col.lower() for col in cols) else cols[0]
            
            if choice == 'profile_extraction':
                # Build text from available fields
                text_parts = [f"{row[name_col]}"]
                
                if any('company' in col.lower() for col in cols):
                    company_col = [col for col in cols if 'company' in col.lower()][0]
                    text_parts.append(f"works at {row[company_col]}")
                
                if any('email' in col.lower() for col in cols):
                    email_col = [col for col in cols if 'email' in col.lower()][0]
                    text_parts.append(f"email: {row[email_col]}")
                
                if any('city' in col.lower() for col in cols):
                    city_col = [col for col in cols if 'city' in col.lower()][0]
                    text_parts.append(f"based in {row[city_col]}")
                
                text = ", ".join(text_parts) + "."
                
                instruction = "Extract the complete profile information from this text in JSON format."
                
                extracted = {"name": row[name_col]}
                if any('email' in col.lower() for col in cols):
                    extracted["email"] = str(row[[col for col in cols if 'email' in col.lower()][0]])
                if any('company' in col.lower() for col in cols):
                    extracted["company"] = str(row[[col for col in cols if 'company' in col.lower()][0]])
                if any('city' in col.lower() for col in cols):
                    extracted["city"] = str(row[[col for col in cols if 'city' in col.lower()][0]])
                
                output = json.dumps(extracted, indent=2)
            
            elif choice == 'contact_info':
                text_parts = [f"Contact {row[name_col]}"]
                
                if any('email' in col.lower() for col in cols):
                    email_col = [col for col in cols if 'email' in col.lower()][0]
                    text_parts.append(f"at {row[email_col]}")
                
                if any('phone' in col.lower() for col in cols):
                    phone_col = [col for col in cols if 'phone' in col.lower()][0]
                    text_parts.append(f"or call {row[phone_col]}")
                
                text = " ".join(text_parts) + "."
                
                instruction = "Extract all contact information from this text."
                
                extracted = {"name": row[name_col]}
                if any('email' in col.lower() for col in cols):
                    extracted["email"] = str(row[[col for col in cols if 'email' in col.lower()][0]])
                if any('phone' in col.lower() for col in cols):
                    extracted["phone"] = str(row[[col for col in cols if 'phone' in col.lower()][0]])
                
                output = json.dumps(extracted, indent=2)
            
            elif choice == 'structured_conversion':
                text_parts = [f"Record: {row[name_col]}"]
                
                if any('age' in col.lower() for col in cols):
                    age_col = [col for col in cols if 'age' in col.lower()][0]
                    text_parts.append(f"Age: {row[age_col]}")
                
                if any('email' in col.lower() for col in cols):
                    email_col = [col for col in cols if 'email' in col.lower()][0]
                    text_parts.append(f"Email: {row[email_col]}")
                
                if any('company' in col.lower() for col in cols):
                    company_col = [col for col in cols if 'company' in col.lower()][0]
                    text_parts.append(f"Company: {row[company_col]}")
                
                text = ", ".join(text_parts)
                
                instruction = "Convert this record into structured JSON format."
                
                extracted = {}
                for col in cols[:5]:  # First 5 columns
                    extracted[col] = str(row[col])
                
                output = json.dumps(extracted, indent=2)
            
            elif choice == 'entity_list':
                text = f"{row[name_col]} is a professional"
                
                if any('age' in col.lower() for col in cols):
                    age_col = [col for col in cols if 'age' in col.lower()][0]
                    text += f", {row[age_col]} years old"
                
                if any('city' in col.lower() for col in cols):
                    city_col = [col for col in cols if 'city' in col.lower()][0]
                    text += f", located in {row[city_col]}"
                
                text += "."
                
                instruction = "Extract all named entities (person, location, age) from this text."
                
                entities = {"PERSON": row[name_col]}
                
                if any('age' in col.lower() for col in cols):
                    age_col = [col for col in cols if 'age' in col.lower()][0]
                    entities["AGE"] = str(row[age_col])
                
                if any('city' in col.lower() for col in cols):
                    city_col = [col for col in cols if 'city' in col.lower()][0]
                    entities["LOCATION"] = str(row[city_col])
                
                output = json.dumps(entities, indent=2)
            
            else:  # field_extraction
                fields_to_extract = []
                text = f"Customer information: "
                
                if any('email' in col.lower() for col in cols):
                    email_col = [col for col in cols if 'email' in col.lower()][0]
                    fields_to_extract.append(f"email is {row[email_col]}")
                
                if any('phone' in col.lower() for col in cols):
                    phone_col = [col for col in cols if 'phone' in col.lower()][0]
                    fields_to_extract.append(f"phone is {row[phone_col]}")
                
                text += ", ".join(fields_to_extract)
                
                instruction = "Extract the email and phone number from this text."
                
                extracted = {}
                if any('email' in col.lower() for col in cols):
                    extracted["email"] = str(row[[col for col in cols if 'email' in col.lower()][0]])
                if any('phone' in col.lower() for col in cols):
                    extracted["phone"] = str(row[[col for col in cols if 'phone' in col.lower()][0]])
                
                output = json.dumps(extracted, indent=2)
            
            samples.append({
                "instruction": instruction,
                "input": text,
                "output": output
            })
        
        self.instructions.extend(samples)
        print(f"✅ Generated {len(samples)} Extraction samples")
    
    def clean_and_validate(self):
        """Remove duplicates and invalid samples"""
        print("\n🧹 Cleaning dataset...")
        
        initial_count = len(self.instructions)
        
        # Remove empty outputs
        self.instructions = [s for s in self.instructions if s['output'].strip()]
        
        # Remove duplicates
        seen = set()
        cleaned = []
        for sample in self.instructions:
            key = (sample['instruction'], sample['output'])
            if key not in seen:
                seen.add(key)
                cleaned.append(sample)
        
        self.instructions = cleaned
        
        print(f"✅ Cleaned: {initial_count} → {len(self.instructions)} samples")
    
    def save_datasets(self, train_ratio=0.8):
        """Save train and validation splits"""
        print("\n💾 Saving datasets...")
        
        random.shuffle(self.instructions)
        
        split_idx = int(len(self.instructions) * train_ratio)
        train_data = self.instructions[:split_idx]
        val_data = self.instructions[split_idx:]
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save train
        with jsonlines.open('data/train.jsonl', 'w') as f:
            f.write_all(train_data)
        
        # Save val
        with jsonlines.open('data/val.jsonl', 'w') as f:
            f.write_all(val_data)
        
        print(f"✅ Train: {len(train_data)} samples → data/train.jsonl")
        print(f"✅ Val: {len(val_data)} samples → data/val.jsonl")
        
        return train_data, val_data


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Initialize builder
    builder = InstructionDatasetBuilder(
        customers_path='data/customers-100000.csv',
        people_path='data/people-100000.csv'
    )
    
    # ✅ BALANCED SAMPLE GENERATION (500 each = 1500 total)
    builder.generate_qa_samples(n_samples=500)
    builder.generate_reasoning_samples(n_samples=500)
    builder.generate_extraction_samples(n_samples=500)
    
    # Clean
    builder.clean_and_validate()
    
    # Save
    train_data, val_data = builder.save_datasets(train_ratio=0.8)
    
    print("\n" + "="*50)
    print("✅ DATASET CREATION COMPLETE")
    print("="*50)
    print(f"📊 Train: {len(train_data)} samples")
    print(f"📊 Val: {len(val_data)} samples")
    print(f"📊 Total: {len(train_data) + len(val_data)} samples")