import json

def build_duplicate_control_prompt(
    control_description: str,
    control_inventory: list[dict],
) -> str:

    return f"""You are a controls rationalization expert at a financial institution.
Your task is to identify potential duplicate controls that should be consolidated.

<control_under_review>
{control_description}
</control_under_review>

<control_inventory>
{json.dumps(control_inventory, indent=2, default=str)}
</control_inventory>

Review the control under review against every control in the inventory.
Identify any controls that are highly likely duplicates — meaning they share the same
control objective, mitigate the same risk, or operate via the same mechanism,
even if the wording differs.

Return your response in this exact JSON format:
{{
  "duplicate_candidates": [
    {{
      "control_id": "<id from inventory>",
      "control_name": "<name from inventory>",
      "similarity_rationale": "<why these controls are likely duplicates>",
      "confidence": "high | medium | low"
    }}
  ],
  "recommendation": "<brief consolidation recommendation or 'No duplicates identified'>"
}}

Only include controls with medium or high confidence. Return empty list if none found.
Return JSON only, no preamble."""


if __name__ == "__main__":
    control_description = """
    Control ID: CTRL-0042
    Name: Quarterly User Access Review
    Description: IT Security performs a quarterly review of all user access rights
    to core banking systems. Access that is no longer appropriate is revoked within
    5 business days of review completion.
    Risk: Unauthorized access / segregation of duties violation
    """

    control_inventory = df.to_dict(orient="records")  # your controls DataFrame

    prompt   = build_duplicate_control_prompt(control_description, control_inventory)
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    import json
    result = json.loads(response.content[0].text)
    
    

import tiktoken
def estimate_batch_size(
    records: list[dict],
    token_budget: int = 6_000,   # headroom for prompt + response
    sample_n: int = 20,
) -> int:
    """
    CHOOSE BATCH SIZE BASED ON TOKEN LIMITATIONS
    """
    enc     = tiktoken.get_encoding("cl100k_base")
    sample  = records[:sample_n]
    avg_tok = len(enc.encode(json.dumps(sample, default=str))) / sample_n
    return max(1, int(token_budget // avg_tok))

batch_size = estimate_batch_size(lookup_records)
print(f"Recommended batch size: {batch_size}")

def batch_records(records: list[dict], batch_size: int):
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]