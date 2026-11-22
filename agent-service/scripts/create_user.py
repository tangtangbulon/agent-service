"""创建用户，或重置已有用户的密码。

用法：
    uv run python -m scripts.create_user <username> --email <email> [--display-name <名称>]
密码通过交互输入，不会出现在命令行历史里。
"""

import argparse
import asyncio
import getpass

from app.core.security import hash_password
from app.infra.database import AsyncSessionFactory, close_database
from app.modules.auth.models import User
from app.modules.auth.repository import UserRepository


def read_password() -> str:
    password = getpass.getpass("密码：")
    if len(password) < 8:
        raise SystemExit("密码至少需要 8 位")
    if password != getpass.getpass("再次输入密码："):
        raise SystemExit("两次输入的密码不一致")
    return password


async def upsert_user(username: str, email: str | None, display_name: str | None, password: str) -> None:
    async with AsyncSessionFactory() as session, session.begin():
        user = await UserRepository(session).find_by_username(username)
        if user is None:
            if not email:
                raise SystemExit("新建用户必须提供 --email")
            session.add(
                User(
                    username=username,
                    email=email,
                    display_name=display_name,
                    password_hash=hash_password(password),
                )
            )
            print(f"已创建用户 {username}")
            return

        user.password_hash = hash_password(password)
        if email:
            user.email = email
        if display_name:
            user.display_name = display_name
        print(f"已重置用户 {username} 的密码")


async def main() -> None:
    parser = argparse.ArgumentParser(description="创建用户或重置密码")
    parser.add_argument("username")
    parser.add_argument("--email")
    parser.add_argument("--display-name")
    args = parser.parse_args()

    password = read_password()
    try:
        await upsert_user(args.username, args.email, args.display_name, password)
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
