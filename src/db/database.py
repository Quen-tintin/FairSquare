"""
Async SQLAlchemy engine + session factory.
All DB interactions go through get_db() dependency in FastAPI,
or async with AsyncSessionLocal() in standalone scripts.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=(settings.environment == "development"),
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """All ORM models inherit from this."""
    pass


async def get_db():
    """FastAPI dependency — yields a session and closes it after the request."""
    async with AsyncSessionLocal() as session:
        yield session
