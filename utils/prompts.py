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


import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright, Page, TimeoutError

# ── Config ────────────────────────────────────────────────────────────────────
COPILOT_URL    = "https://copilot.microsoft.com/"
RESPONSE_TIMEOUT_MS = 120_000   # 2 min max wait per response
STABLE_POLLS        = 4         # consecutive unchanged polls = done
POLL_INTERVAL_S     = 1.5

# ── Selectors (isolate here — Copilot's DOM changes frequently) ───────────────
SEL = {
    "textarea"    : "textarea",
    "file_input"  : "input[type='file']",
    "response"    : "[data-testid='response-message']",  # update as needed
    "stop_button" : "[aria-label='Stop generating']",
}


# ── Core helpers ──────────────────────────────────────────────────────────────

async def wait_for_response_complete(page: Page) -> str:
    """
    Wait until Copilot stops generating.
    Strategy: poll the response container; declare done when content
    is stable for STABLE_POLLS consecutive checks AND stop button is gone.
    Returns the final response text.
    """
    last_text    = ""
    stable_count = 0
    elapsed      = 0
    max_elapsed  = RESPONSE_TIMEOUT_MS / 1000

    while elapsed < max_elapsed:
        await asyncio.sleep(POLL_INTERVAL_S)
        elapsed += POLL_INTERVAL_S

        # Prefer a targeted selector over full body innerHTML
        try:
            current_text = await page.inner_text(SEL["response"])
        except Exception:
            current_text = await page.inner_text("body")

        stop_visible = await page.is_visible(SEL["stop_button"])

        if current_text == last_text and not stop_visible:
            stable_count += 1
        else:
            stable_count = 0
            last_text    = current_text

        if stable_count >= STABLE_POLLS:
            return current_text

    raise TimeoutError(f"Response did not stabilize within {max_elapsed}s")


async def new_chat(page: Page) -> None:
    """Navigate to a fresh Copilot session."""
    await page.goto(COPILOT_URL)
    await page.wait_for_selector(SEL["textarea"], timeout=15_000)


async def run_task(
    page: Page,
    task_text: str,
    file_path: str | None = None,
    retries: int = 2,
) -> str:
    """
    Send a single prompt (with optional file attachment) and return the response.
    Retries on failure.
    """
    for attempt in range(1, retries + 1):
        try:
            await new_chat(page)

            if file_path:
                file_input = page.locator(SEL["file_input"])
                await file_input.set_input_files(file_path)
                await asyncio.sleep(1)   # let upload register

            await page.fill(SEL["textarea"], task_text)
            await page.keyboard.press("Enter")

            response = await wait_for_response_complete(page)
            return response

        except Exception as e:
            print(f"  [attempt {attempt}/{retries}] failed: {e}")
            if attempt == retries:
                raise
            await asyncio.sleep(3)


# ── Batch runner ──────────────────────────────────────────────────────────────

async def run_batch(tasks: list[dict], output_path: str = "results.json") -> list[dict]:
    """
    tasks: list of {"id": str, "prompt": str, "file": str | None}
    Writes results incrementally so a crash doesn't lose completed work.
    """
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()

        # Single persistent page — avoids tab proliferation
        page = await context.new_page()

        for i, task in enumerate(tasks, start=1):
            print(f"[{i}/{len(tasks)}] Running task: {task['id']}")
            try:
                response = await run_task(
                    page,
                    task_text = task["prompt"],
                    file_path = task.get("file"),
                )
                result = {"id": task["id"], "status": "ok", "response": response}
            except Exception as e:
                result = {"id": task["id"], "status": "error", "error": str(e)}

            results.append(result)

            # Incremental write — survive mid-run crashes
            Path(output_path).write_text(json.dumps(results, indent=2))
            print(f"  → saved ({result['status']})")

        await browser.close()

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tasks = [
        {"id": "task_1", "prompt": "Execute Task A...", "file": r"C:\path\file1.txt"},
        {"id": "task_2", "prompt": "Execute Task B...", "file": r"C:\path\file2.txt"},
        {"id": "task_3", "prompt": "Execute Task C...", "file": None},
    ]

    asyncio.run(run_batch(tasks, output_path="results.json"))