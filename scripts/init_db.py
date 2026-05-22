"""Initialize the database schema + pgvector extension."""
from sqlalchemy import text

from src.storage.db import engine
from src.storage.models import Base


def main() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    print("Database initialized.")


if __name__ == "__main__":
    main()
