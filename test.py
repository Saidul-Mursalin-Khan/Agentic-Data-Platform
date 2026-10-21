# """
# Agentic Data Platform — Interview Demo
# Run from the project root:
#     python demo_prompt_tests.py
# """

# import csv
# import time

# from langchain_core.messages import HumanMessage
# from agents.data_agent import data_agent

# DEVTO = "https://dev.to/api/articles"
# SRC = "data/extract/extracted_data.csv"
# DST = "data/transform"

# TESTS = [
#     # ==================== SQL: 10 successful examples ====================
#     ("SQL01", "SQL - Distinct values",
#      "What are the different types of payment methods in our database?"),

#     ("SQL02", "SQL - Count",
#      "How many users do we have?"),

#     ("SQL03", "SQL - Conditional count",
#      "How many active users are there?"),

#     ("SQL04", "SQL - Group by",
#      "How many drivers and riders do we have?"),

#     ("SQL05", "SQL - Sorting and limit",
#      "Show me exactly 25 rides."),

#     ("SQL06", "SQL - Aggregation",
#      "What is the average fare of completed rides?"),

#     ("SQL07", "SQL - Date filtering",
#      "How many rides were requested in 2025?"),

#     ("SQL08", "SQL - Join and aggregation",
#      "Which payment methods are used the most, and "
#      "what is the total amount paid with each?"),

#     ("SQL09", "SQL - Text filtering",
#      "Show users whose first name starts with J."),

#     ("SQL10", "SQL - Data quality",
#      "Are there completed rides with a fare of zero?"),

#     # ==================== ETL: 10 successful examples ====================
#     ("ETL01", "ETL - Extract API to CSV",
#      f"Extract '{DEVTO}' with per_page=30 and "
#      "save as CSV in data/extract."),

#     ("ETL02", "ETL - Filter rows",
#      f"Transform '{SRC}' and save to '{DST}' as CSV. "
#      "Keep only posts whose tags contain python."),

#     ("ETL03", "ETL - Sort and top N",
#      f"Transform '{SRC}' and save to '{DST}' as CSV. "
#      "Sort by public_reactions_count descending and keep the top 5."),

#     ("ETL04", "ETL - Group and count",
#      f"Transform '{SRC}' and save to '{DST}' as CSV. "
#      "Count articles per author (user.name)."),

#     ("ETL05", "ETL - Add calculated column",
#      f"Transform '{SRC}' and save to '{DST}' as CSV. "
#      "Add title_length with the number of characters in each title."),

#     ("ETL06", "ETL - Select and rename columns",
#      f"Transform '{SRC}' and save to '{DST}' as CSV. "
#      "Keep only id, title and url, and rename title to headline."),

#     ("ETL07", "ETL - Remove duplicates",
#      f"Transform '{SRC}' and save to '{DST}' as CSV. "
#      "Remove duplicate rows based on title."),

#     ("ETL08", "ETL - Date filtering",
#      f"Transform '{SRC}' and save to '{DST}' as CSV. "
#      "Keep articles published in the last 7 days."),

#     ("ETL09", "ETL - Format conversion",
#      f"Transform '{SRC}' and save to '{DST}' as JSON. "
#      "Keep only the first 10 rows."),

#     ("ETL10", "ETL - Chained workflow",
#      f"Extract '{DEVTO}' as CSV into data/extract, "
#      "then keep only posts with the python tag and save "
#      "the result to data/transform as CSV."),
# ]


# def get_answer(response):
#     """Extract the final readable answer from the agent response."""
#     last = response["messages"][-1]

#     if isinstance(last, dict):
#         if last.get("final_answer"):
#             return str(last["final_answer"])

#         messages = last.get("messages", [])
#         if messages:
#             return str(getattr(messages[-1], "content", messages[-1]))

#     return str(getattr(last, "content", last))


# def main():
#     results = []

#     for test_id, category, prompt in TESTS:
#         print(f"\n{'=' * 65}")
#         print(f"[{test_id}] {category}")
#         print(f"Prompt: {prompt}")

#         start = time.time()

#         try:
#             response = data_agent.invoke({
#                 "messages": [HumanMessage(content=prompt)],
#                 "route_response": "",
#             })
#             answer = get_answer(response)
#             status = "OK" if answer.strip() else "EMPTY"

#         except Exception as exc:
#             answer = f"{type(exc).__name__}: {exc}"
#             status = "ERROR"

#         seconds = round(time.time() - start, 1)

#         print(f"Status: {status} | Time: {seconds}s")
#         print(f"Answer: {answer}")

#         results.append([
#             test_id, category, prompt, status, seconds, answer
#         ])

#     with open("demo_results.csv", "w", newline="", encoding="utf-8") as file:
#         writer = csv.writer(file)
#         writer.writerow([
#             "id", "category", "prompt", "status", "seconds", "answer"
#         ])
#         writer.writerows(results)

#     successful = sum(row[3] == "OK" for row in results)

#     print(f"\n{'=' * 65}")
#     print(f"Demo completed: {successful}/{len(results)} prompts returned answers.")
#     print("Results saved to demo_results.csv")
#     print("Note: OK means the agent returned an answer, not that its data was verified.")


# if __name__ == "__main__":
#     main()


"""
Agentic Data Platform — Interview Demo
Run from the project root:
    python demo_prompt_tests.py
"""

import csv
import json
import time
from pathlib import Path

from langchain_core.messages import HumanMessage
from agents.data_agent import data_agent

DEVTO = "https://dev.to/api/articles"
SRC = "data/extract"          # folder: the newest file inside is used (ETL01's fresh export)
DST = "data/transform"

TESTS = [
    # ==================== SQL: 10 successful examples ====================
    ("SQL01", "SQL - Distinct values",
     "What are the different types of payment methods in our database?"),

    ("SQL02", "SQL - Count",
     "How many users do we have?"),

    ("SQL03", "SQL - Conditional count",
     "How many active users are there?"),

    ("SQL04", "SQL - Group by",
     "How many drivers and riders do we have?"),

    ("SQL05", "SQL - Sorting and limit",
     "Show me exactly 25 rides."),

    ("SQL06", "SQL - Aggregation",
     "What is the average fare of completed rides?"),

    ("SQL07", "SQL - Date filtering",
     "How many rides were requested in 2025?"),

    ("SQL08", "SQL - Join and aggregation",
     "Which payment methods are used the most, and "
     "what is the total amount paid with each?"),

    ("SQL09", "SQL - Text filtering",
     "Show users whose first name starts with J."),

    ("SQL10", "SQL - Data quality",
     "Are there completed rides with a fare of zero?"),

    # ==================== ETL: 10 successful examples ====================
    # ETL01 runs first and creates the newest file in data/extract.
    # ETL02-ETL09 then transform that file (SRC is the folder, the newest file is used).
    ("ETL01", "ETL - Extract API to CSV",
     f"Extract '{DEVTO}' with per_page=30 and "
     "save as CSV in data/extract."),

    ("ETL02", "ETL - Filter rows",
     f"Transform '{SRC}' and save to '{DST}' as CSV. "
     "Keep only posts whose tags contain python."),

    ("ETL03", "ETL - Sort and top N",
     f"Transform '{SRC}' and save to '{DST}' as CSV. "
     "Sort by public_reactions_count descending and keep the top 5."),

    ("ETL04", "ETL - Group and count",
     f"Transform '{SRC}' and save to '{DST}' as CSV. "
     "Count articles per author (user.name)."),

    ("ETL05", "ETL - Add calculated column",
     f"Transform '{SRC}' and save to '{DST}' as CSV. "
     "Add title_length with the number of characters in each title."),

    ("ETL06", "ETL - Select and rename columns",
     f"Transform '{SRC}' and save to '{DST}' as CSV. "
     "Keep only id, title and url, and rename title to headline."),

    ("ETL07", "ETL - Remove duplicates",
     f"Transform '{SRC}' and save to '{DST}' as CSV. "
     "Remove duplicate rows based on title."),

    ("ETL08", "ETL - Date filtering",
     f"Transform '{SRC}' and save to '{DST}' as CSV. "
     "Keep articles published in the last 7 days."),

    ("ETL09", "ETL - Format conversion",
     f"Transform '{SRC}' and save to '{DST}' as JSON. "
     "Keep only the first 10 rows."),

    ("ETL10", "ETL - Chained workflow",
     f"Extract '{DEVTO}' as CSV into data/extract, "
     "then keep only posts with the python tag and save "
     "the result to data/transform as CSV."),

    # ==================== GENERAL: answered directly by the LLM ====================
    ("GEN01", "General - Greeting",
     "hello"),

    ("GEN02", "General - Concept question",
     "What is a data lake?"),
]


def get_answer(response):
    """Extract the final readable answer from the agent response."""
    last = response["messages"][-1]

    if isinstance(last, dict):
        if last.get("final_answer"):
            return str(last["final_answer"])

        messages = last.get("messages", [])
        if messages:
            return str(getattr(messages[-1], "content", messages[-1]))

    return str(getattr(last, "content", last))


def show_generated_files():
    """Lists every kept file (nothing is overwritten) and checks that JSON files are valid."""
    print(f"\n{'=' * 65}")
    print("Generated files (oldest first, nothing is overwritten):")

    for folder in ("data/extract", "data/transform"):
        files = sorted(Path(folder).glob("*.*"), key=lambda p: p.stat().st_mtime)
        for f in files:
            note = ""
            if f.suffix == ".json":
                try:
                    json.loads(f.read_text(encoding="utf-8"))
                    note = "(valid JSON)"
                except Exception:
                    note = "(NOT valid JSON, probably an old JSON-lines file)"
            print(f"  {f} {note}")


def main():
    results = []

    for test_id, category, prompt in TESTS:
        print(f"\n{'=' * 65}")
        print(f"[{test_id}] {category}")
        print(f"Prompt: {prompt}")

        start = time.time()

        try:
            response = data_agent.invoke({
                "messages": [HumanMessage(content=prompt)],
                "route_response": "",
            })
            answer = get_answer(response)
            status = "OK" if answer.strip() else "EMPTY"

        except Exception as exc:
            answer = f"{type(exc).__name__}: {exc}"
            status = "ERROR"

        seconds = round(time.time() - start, 1)

        print(f"Status: {status} | Time: {seconds}s")
        print(f"Answer: {answer}")

        results.append([
            test_id, category, prompt, status, seconds, answer
        ])

    with open("demo_results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "id", "category", "prompt", "status", "seconds", "answer"
        ])
        writer.writerows(results)

    successful = sum(row[3] == "OK" for row in results)

    show_generated_files()

    print(f"\n{'=' * 65}")
    print(f"Demo completed: {successful}/{len(results)} prompts returned answers.")
    print("Results saved to demo_results.csv")
    print("Note: OK means the agent returned an answer, not that its data was verified.")


if __name__ == "__main__":
    main()