import os
import time
from openai import OpenAI, RateLimitError

# Define your pool of keys here. We'll pull from environment variables, 
# but you can temporarily hardcode them here for testing if you prefer.
API_KEYS = [
    os.getenv("GROQ_API_KEY_1", ""),
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", "")
]

# Filter out empty keys
API_KEYS = [k for k in API_KEYS if k.strip()]

def run_cycler_test():
    if not API_KEYS:
        print("❌ No API keys found.")
        print("Please set GROQ_API_KEY_1, GROQ_API_KEY_2, etc. in your .env file")
        print("or hardcode them directly in the API_KEYS list in this script.")
        return

    print(f"Loaded {len(API_KEYS)} API keys. Starting cycler test...")
    print("We will fire 15 queries sequentially to test rate limit fallbacks.")
    
    current_key_idx = 0

    for i in range(1, 16):
        success = False
        attempts = 0
        
        while not success and attempts < len(API_KEYS):
            current_key = API_KEYS[current_key_idx]
            
            # Initialize client. Groq is OpenAI-compatible, so we can use the OpenAI SDK 
            # by just changing the base_url.
            client = OpenAI(
                api_key=current_key,
                base_url="https://api.groq.com/openai/v1"
            )
            
            try:
                print(f"\n[Query {i}] Attempting with Key #{current_key_idx + 1}...")
                
                start_time = time.time()
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant", # Using supported model
                    messages=[
                        {"role": "user", "content": f"Tell me a 1-sentence random fun fact. Query ID: {i}"}
                    ],
                    max_tokens=40
                )
                elapsed = time.time() - start_time
                
                print(f"✅ Success! (Key #{current_key_idx + 1}) | Time: {elapsed:.2f}s")
                print(f"   Response: {response.choices[0].message.content.strip()}")
                success = True
                
            except RateLimitError as e:
                print(f"⚠️  Rate Limit Hit on Key #{current_key_idx + 1}!")
                # Rotate to the next key in the array
                current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                attempts += 1
                print(f"🔄 Rotating to Key #{current_key_idx + 1}...")
                
            except Exception as e:
                print(f"❌ Unexpected Error on Key #{current_key_idx + 1}: {type(e).__name__} - {e}")
                # Rotate on other errors as well
                current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                attempts += 1
        
        if not success:
            print(f"\n🚨 [Query {i}] Failed after trying all {len(API_KEYS)} keys (All exhausted).")
            print("⏳ Waiting 10 seconds for cooldown before next query...")
            time.sleep(10)
            
        time.sleep(0.5) # Slight delay so we can read the console

if __name__ == "__main__":
    run_cycler_test()
