"""Small sample golden dataset for demo/testing. Production datasets live in data/evals/golden.jsonl."""
SAMPLE_GOLDEN = [
    {
        "query": "What are the pillars of the AWS Well-Architected Framework?",
        "expected_answer_contains": ["operational excellence", "security", "reliability", "performance", "cost optimization", "sustainability"],
        "relevant_chunk_ids": [],  # populated at ingestion time by scripts/ingest.py
    },
    {
        "query": "How should I design a system for fault tolerance on AWS?",
        "expected_answer_contains": ["availability zones", "redundancy", "failover"],
        "relevant_chunk_ids": [],
    },
    {
        "query": "What is the principle of least privilege in IAM?",
        "expected_answer_contains": ["minimum permissions", "IAM", "access"],
        "relevant_chunk_ids": [],
    },
]
