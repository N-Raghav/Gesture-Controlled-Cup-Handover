import asyncio

from gesture_service import GestureDetectorService  # noqa: F401
from viam.module.module import Module


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())