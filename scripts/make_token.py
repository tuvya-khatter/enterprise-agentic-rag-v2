"""CLI to issue a JWT for testing."""
import argparse
from uuid import uuid4

from src.auth.jwt import create_access_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="demo-user")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--role", default="user")
    args = parser.parse_args()

    token = create_access_token(
        user_id=str(uuid4()),
        tenant_id=args.tenant,
        role=args.role,
    )
    print(token)


if __name__ == "__main__":
    main()
