import asyncio
import json
from typing import Dict, List, Optional, Tuple

import ollama
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """You are an expert at creating high-quality datasets for training an AI model to write LinkedIn posts. 

### **Your Task**
For each given LinkedIn post, generate a structured **instruction-answer pair** that follows these rules:

1️⃣ **Instruction (GUIDE FOR GENERATION)**
   - Clearly **define the post type** (e.g., personal story, industry insight, technical deep dive, engagement post).
   - Provide **specific constraints** (e.g., "make it engaging and thought-provoking", "use an analogy", "keep it under 250 words").
   - Include relevant **keywords** to guide topic focus.

2️⃣ **Answer (FULL LINKEDIN POST TEXT)**
   - **MUST include the full LinkedIn post word-for-word** as the response.
   - **No paraphrasing, no summaries**—preserve the original content.
   - **Format it as a natural LinkedIn post**, following best practices in readability and engagement.

---

### **Response Format (JSON)**
Return a JSON array, where each object contains:
- **instruction**: A detailed writing prompt for generating a similar LinkedIn post.
- **answer**: The full LinkedIn post text, word-for-word.

#### **Example**
```json
[
  {{
    "instruction": "Write an engaging LinkedIn post sharing a personal lesson from working on AI observability. The post should highlight a real challenge, a surprising insight, and a practical takeaway. Keep it concise, direct, and end with an open question for engagement.",
    "answer": "[FULL LINKEDIN POST TEXT]"
  }},
  {{
    "instruction": "Write a thought-provoking industry insight post on why AI monitoring is often overlooked but critical. The post should start with a bold statement, use a relatable analogy, and offer a concrete action plan for AI teams. End with a call to action.",
    "answer": "🚦 [FULL LINKEDIN POST TEXT]"
  }}
]
```

DO NOT summarize or paraphrase the LinkedIn post. Always provide the full text.
Each instruction must be distinct, covering different angles (e.g., storytelling, technical insights, engagement hooks).
Ensure the dataset is varied, representing different writing styles and tones.

Your goal is to create high-quality training data to help an AI model generate authentic and engaging LinkedIn posts.

"""


class LinkedInQAPair(BaseModel):
    """
    A model representing an instruction-answer pair for LinkedIn post generation.

    Attributes:
        instruction (str): Detailed instructions for writing a LinkedIn post
        answer (str): The full LinkedIn post text
        post_id (Optional[str]): Optional identifier for the source post
    """

    instruction: str = Field(
        ..., description="Detailed writing prompt for generating a LinkedIn post"
    )
    answer: str = Field(..., description="The full LinkedIn post text, word-for-word")
    post_id: Optional[str] = Field(
        None, description="Identifier for the source LinkedIn post"
    )


async def generate_qa_pairs_async(
    post_content: str, post_id: str, semaphore, num_pairs: int = 3
) -> Tuple[str, List[Dict[str, str]]]:
    """Generate question-answer pairs for a LinkedIn post using LLM."""
    async with semaphore:
        print(f"Processing post {post_id}...")

        try:
            # Create async client
            client = ollama.AsyncClient()
            # Format the system prompt with the requested number of pairs
            formatted_prompt = SYSTEM_PROMPT.format(num_pairs=num_pairs)

            response = await client.chat(
                model="llama3.2:3b",  # Using the model that's available in your system
                messages=[
                    {"role": "system", "content": formatted_prompt},
                    {"role": "user", "content": post_content},
                ],
                options={
                    "temperature": 0.7,  # Adjust this value as needed (0.0 to 1.0)
                    "top_k": 0.3,  # Adjust this value as needed
                },
                format=LinkedInQAPair.model_json_schema(),
            )
            print(f"Response: {response}")

            try:
                # Extract the content from the response
                content = response.message.content

                # Parse the JSON content (it's a single QA pair, not an array)
                qa_pair = json.loads(content)

                # Add post_id to the QA pair
                qa_pair["post_id"] = post_id
                print(f"QA pair: {qa_pair}")
                # Return as a list with a single item
                return post_id, [qa_pair]
            except Exception as e:
                print(f"Error parsing response for post {post_id}: {e}")
                print(f"Response content: {response}")
                return post_id, []

        except Exception as e:
            print(f"Error processing post {post_id}: {e}")
            return post_id, []


async def process_linkedin_posts_async(
    json_path: str,
    output_file: str = None,
    max_concurrent: int = 4,
    num_pairs: int = 3,
    max_posts: int = None,
):
    """Process LinkedIn posts from a JSON file and generate QA pairs."""
    print(f"Processing LinkedIn posts from: {json_path}")
    print(f"Generating {num_pairs} QA pairs per post")

    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Extract posts from the data structure
    posts = json_data.get("data", [])

    # Limit the number of posts if specified
    if max_posts is not None and max_posts > 0:
        posts = posts[:max_posts]
        print(
            f"Limited to {len(posts)} posts (out of {len(json_data.get('data', []))})"
        )
    else:
        print(f"Loaded {len(posts)} posts")

    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for each post
    tasks = []
    for post in posts:
        post_id = post.get("urn", "unknown")
        post_content = post.get("text", "")

        if post_content:
            task = generate_qa_pairs_async(post_content, post_id, semaphore, num_pairs)
            tasks.append(task)
        else:
            print(f"Skipping post {post_id} - no content")

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)

    # Combine all QA pairs into a dataset
    dataset = []
    for post_id, qa_pairs in results:
        for qa_pair in qa_pairs:
            # Add post_id as metadata
            qa_pair["post_id"] = post_id
            dataset.append(qa_pair)

    print(f"Generated {len(dataset)} QA pairs from {len(results)} posts")

    # Output results
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"QA dataset saved to {output_file}")
    else:
        print("\nGenerated QA Dataset (sample):")
        for i, qa in enumerate(dataset[:3]):  # Show first 3 examples
            print(f"Example {i + 1}:")
            print(f"Q: {qa['instruction']}")
            print(f"A: {qa['answer']}")
            print()

    return dataset


async def main(
    json_path: str,
    output_file: str = None,
    max_concurrent: int = 4,
    num_pairs: int = 3,
    max_posts: int = None,
):
    """
    Main function to process LinkedIn posts and generate QA pairs.

    Args:
        json_path (str): Path to the JSON file containing LinkedIn posts
        output_file (str, optional): Path to save the QA dataset. If None, prints sample to console.
        max_concurrent (int, optional): Maximum number of concurrent LLM operations. Defaults to 4.
        num_pairs (int, optional): Number of QA pairs to generate per post. Defaults to 3.
        max_posts (int, optional): Maximum number of posts to process. If None, processes all posts.
    """
    print(f"Using max {max_concurrent} concurrent requests")
    return await process_linkedin_posts_async(
        json_path, output_file, max_concurrent, num_pairs, max_posts
    )


if __name__ == "__main__":
    # Example usage
    json_input = "data/posts_test_data_decoded.json"  # Your LinkedIn posts JSON file
    dataset_output = "data/qa_dataset.json"  # Output file for QA pairs
    concurrent_jobs = 4  # Adjust based on your system capabilities
    qa_pairs_per_post = 3  # Number of QA pairs to generate per post
    posts_limit = 10  # Set to a number to limit posts, or None to process all

    asyncio.run(
        main(
            json_input, dataset_output, concurrent_jobs, qa_pairs_per_post, posts_limit
        )
    )
