import asyncio
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from openenv import GenericEnvClient

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_decision_from_llm(observation):
    prompt = f"""
You are an Automated Procurement Auditor.

Analyze the request and decide:
- Approve → if within policy
- Reject → if violates policy

Input:
{observation}

Output STRICTLY in JSON:
{{
  "decision": "approve or reject",
  "reasoning": "short clear explanation"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content

    # Try to safely parse JSON
    try:
        parsed = json.loads(content)
        decision = parsed.get("decision", "reject").lower()
        reasoning = parsed.get("reasoning", content)
    except:
        decision = "approve" if "approve" in content.lower() else "reject"
        reasoning = content

    return decision, reasoning


async def run_agent():
    env = AutoEnv.from_env("MyEnvV4")

    observation = await env.reset()

    print("\n========== START ==========")
    print("Initial Input:\n", observation)

    total_reward = 0

    for step in range(5):
        print(f"\n----- Step {step + 1} -----")

        decision, reasoning = get_decision_from_llm(observation)

        print("Decision:", decision)
        print("Reasoning:", reasoning)

        action = {
            "decision": decision,
            "reasoning": reasoning
        }

        result = await env.step(action)

        reward = result.get("reward", 0)
        total_reward += reward

        print("Reward:", reward)
        print("Done:", result.get("done"))

        observation = result.get("observation", {})

        if result.get("done"):
            break

    print("\n========== END ==========")
    print("Total Score:", total_reward)


if __name__ == "__main__":
    asyncio.run(run_agent())