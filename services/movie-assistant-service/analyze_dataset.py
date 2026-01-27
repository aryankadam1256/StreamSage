"""
Analyze the collected movies_6000.json file
"""
import json
from collections import Counter

# Load data
print("Loading movies_6000.json...")
with open('d:/PROJECTS/StreamSage/movies_6000.json', 'r', encoding='utf-8') as f:
    movies_data = json.load(f)

print(f"\n{'='*70}")
print(f"DATASET ANALYSIS - {len(movies_data)} Movies")
print(f"{'='*70}")

# Check data completeness
complete_movies = 0
has_details = 0
has_credits = 0
has_keywords = 0
has_recommendations = 0

genres_counter = Counter()
years_counter = Counter()
languages_counter = Counter()

for movie_id, movie in movies_data.items():
    if movie.get('details'):
        has_details += 1
        details = movie['details']
        
        # Count genres
        for genre in details.get('genres', []):
            genres_counter[genre['name']] += 1
        
        # Count years
        release_date = details.get('release_date', '')
        if release_date:
            year = release_date[:4]
            years_counter[year] += 1
        
        # Count languages
        lang = details.get('original_language', 'unknown')
        languages_counter[lang] += 1
    
    if movie.get('credits'):
        has_credits += 1
    if movie.get('keywords'):
        has_keywords += 1
    if movie.get('recommendations'):
        has_recommendations += 1
    
    # Complete if has all 4 components
    if all([movie.get('details'), movie.get('credits'), 
            movie.get('keywords'), movie.get('recommendations')]):
        complete_movies += 1

print(f"\n📊 Data Completeness:")
print(f"  Complete movies (all 4 components): {complete_movies} ({complete_movies/len(movies_data)*100:.1f}%)")
print(f"  Has details: {has_details} ({has_details/len(movies_data)*100:.1f}%)")
print(f"  Has credits: {has_credits} ({has_credits/len(movies_data)*100:.1f}%)")
print(f"  Has keywords: {has_keywords} ({has_keywords/len(movies_data)*100:.1f}%)")
print(f"  Has recommendations: {has_recommendations} ({has_recommendations/len(movies_data)*100:.1f}%)")

print(f"\n🎬 Top 10 Genres:")
for genre, count in genres_counter.most_common(10):
    print(f"  {genre}: {count}")

print(f"\n📅 Year Distribution:")
year_ranges = {
    '1970s': sum(count for year, count in years_counter.items() if '1970' <= year < '1980'),
    '1980s': sum(count for year, count in years_counter.items() if '1980' <= year < '1990'),
    '1990s': sum(count for year, count in years_counter.items() if '1990' <= year < '2000'),
    '2000s': sum(count for year, count in years_counter.items() if '2000' <= year < '2010'),
    '2010s': sum(count for year, count in years_counter.items() if '2010' <= year < '2020'),
    '2020s': sum(count for year, count in years_counter.items() if '2020' <= year < '2030'),
}
for decade, count in year_ranges.items():
    if count > 0:
        print(f"  {decade}: {count}")

print(f"\n🌍 Languages:")
for lang, count in languages_counter.most_common(5):
    print(f"  {lang}: {count}")

# Sample movie
sample_id = list(movies_data.keys())[0]
sample = movies_data[sample_id]
if sample.get('details'):
    d = sample['details']
    print(f"\n📽️ Sample Movie:")
    print(f"  ID: {sample_id}")
    print(f"  Title: {d.get('title')}")
    print(f"  Year: {d.get('release_date', '')[:4]}")
    print(f"  Runtime: {d.get('runtime')} minutes")
    print(f"  Genres: {', '.join([g['name'] for g in d.get('genres', [])])}")
    print(f"  Rating: {d.get('vote_average')}/10 ({d.get('vote_count')} votes)")
    print(f"  Overview: {d.get('overview', '')[:150]}...")

print(f"\n{'='*70}")
print(f"✅ Dataset is ready for processing!")
print(f"{'='*70}\n")
